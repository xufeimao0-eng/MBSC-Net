#!/bin/bash
# PHEME significance test: 2 models × 5 seeds = 10 runs on 4 GPUs
# full vs frozen_bert, Hadamard bilinear, paired t-test
PYTHON=python
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR=$PROJ/results/logs/pheme
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/sigtest_*.log

COMMON="--dataset pheme  --patience 15"
RESULT_BASE="$PROJ/results/checkpoints/pheme_sigtest"

echo "============================================"
echo "PHEME SigTest — 2 models × 5 seeds = 10 runs"
echo "GPUs: 0,1,2,3  |  Start: $(date)"
echo "============================================"

run_one() {
    local gpu=$1 model=$2 seed=$3
    local log="$LOG_DIR/sigtest_${model}_seed${seed}.log"
    echo "[$(date +%H:%M)] GPU${gpu}: ${model} seed=${seed}"
    CUDA_VISIBLE_DEVICES=${gpu} $PYTHON $PROJ/scripts/run_significance.py \
        --model "${model}" --seed "${seed}" --gpu 0 ${COMMON} > "$log" 2>&1
    if [ $? -eq 0 ]; then
        F1=$(python3 -c "import json;d=json.load(open('${RESULT_BASE}/${model}_seed${seed}/seed_42/results.json'));print(d['f1'])" 2>/dev/null || echo "N/A")
        echo "[$(date +%H:%M)] GPU${gpu}: ${model} seed=${seed} DONE F1=${F1}"
    else
        echo "[$(date +%H:%M)] GPU${gpu}: ${model} seed=${seed} FAILED"
    fi
}

# GPU 0: full/42, full/123, frozen_bert/789
(
    run_one 0 full 42
    run_one 0 full 123
    run_one 0 frozen_bert 789
    echo "[GPU0] All done"
) > "$LOG_DIR/sigtest_gpu0.log" 2>&1 &

# GPU 1: frozen_bert/42, frozen_bert/123, full/456
(
    run_one 1 frozen_bert 42
    run_one 1 frozen_bert 123
    run_one 1 full 456
    echo "[GPU1] All done"
) > "$LOG_DIR/sigtest_gpu1.log" 2>&1 &

# GPU 2: full/789, frozen_bert/456
(
    run_one 2 full 789
    run_one 2 frozen_bert 456
    echo "[GPU2] All done"
) > "$LOG_DIR/sigtest_gpu2.log" 2>&1 &

# GPU 3: full/1024, frozen_bert/1024
(
    run_one 3 full 1024
    run_one 3 frozen_bert 1024
    echo "[GPU3] All done"
) > "$LOG_DIR/sigtest_gpu3.log" 2>&1 &

echo "Waiting for all GPUs..."
wait

echo ""
echo "=== ALL DONE $(date) ==="

# Collect & analyze
python3 << 'PYEOF'
import json, os, numpy as np
from scipy.stats import ttest_rel

base = os.path.join(Config.CHECKPOINT_DIR, 'pheme_sigtest')
seeds = [42, 123, 456, 789, 1024]

print("\nPer-seed results:")
print(f"{'Seed':<8} {'Full F1':>10} {'Frozen F1':>10} {'Δ':>10}")
print("-" * 42)

full_f1s, froz_f1s = [], []
for s in seeds:
    f_full = os.path.join(base, f'full_seed{s}', 'seed_42', 'results.json')
    f_froz = os.path.join(base, f'frozen_bert_seed{s}', 'seed_42', 'results.json')
    full = json.load(open(f_full)) if os.path.exists(f_full) else None
    froz = json.load(open(f_froz)) if os.path.exists(f_froz) else None
    f1f = full['f1'] if full else float('nan')
    f1z = froz['f1'] if froz else float('nan')
    full_f1s.append(f1f)
    froz_f1s.append(f1z)
    print(f"{s:<8} {f1f:>10.4f} {f1z:>10.4f} {f1f-f1z:>+10.4f}")

full_f1s = np.array(full_f1s)
froz_f1s = np.array(froz_f1s)

print(f"\n{'='*50}")
print(f"Full mean ± std:    {full_f1s.mean():.4f} ± {full_f1s.std():.4f}")
print(f"Frozen mean ± std:  {froz_f1s.mean():.4f} ± {froz_f1s.std():.4f}")
print(f"Mean Δ:             {(full_f1s - froz_f1s).mean():.4f}")

t_stat, p_val = ttest_rel(full_f1s, froz_f1s)
print(f"\nPaired t-test: t={t_stat:.4f}, p={p_val:.6f}")
if p_val < 0.001:
    print("*** SIGNIFICANT (p < 0.001) ***")
elif p_val < 0.01:
    print("** SIGNIFICANT (p < 0.01) **")
elif p_val < 0.05:
    print("* SIGNIFICANT (p < 0.05) *")
else:
    print("NOT significant (p >= 0.05)")
print(f"{'='*50}")
PYEOF