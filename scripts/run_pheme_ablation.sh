#!/bin/bash
# PHEME Ablation Study — 3 GPUs (1/2/3), sequential per GPU
# 5 variants: simple_concat, concat_semantic, ce_only, no_contrastive, frozen_bert
# All use Hadamard bilinear, patience=15
set -e
PYTHON=python
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR=$PROJ/results/logs/pheme
RESULT_DIR=$PROJ/results/logs/pheme
mkdir -p "$LOG_DIR" "$RESULT_DIR"

COMMON="--dataset pheme  --patience 15"

echo "============================================"
echo "PHEME Ablation — GPU 1/2/3 sequential — $(date)"
echo "  5 variants: simple_concat, concat_semantic, ce_only, no_contrastive, frozen_bert"
echo "  Bilinear: hadamard | Patience: 15"
echo "============================================"

# GPU 1: simple_concat then frozen_bert (sequential)
(
    echo "[GPU1 $(date +%H:%M)] simple_concat"
    CUDA_VISIBLE_DEVICES=1 $PYTHON $PROJ/scripts/run_ablation.py --gpu 0 --variant simple_concat ${COMMON} > "$LOG_DIR/ablation_simple_concat.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/ablation_simple_concat.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU1 $(date +%H:%M)] simple_concat F1=$F1"

    echo "[GPU1 $(date +%H:%M)] frozen_bert"
    CUDA_VISIBLE_DEVICES=1 $PYTHON $PROJ/scripts/run_ablation.py --gpu 0 --variant frozen_bert ${COMMON} > "$LOG_DIR/ablation_frozen_bert.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/ablation_frozen_bert.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU1 $(date +%H:%M)] frozen_bert F1=$F1"
    echo "[GPU1 $(date +%H:%M)] Done."
) > "$LOG_DIR/gpu1_ablation.log" 2>&1 &
PID1=$!

# GPU 2: concat_semantic then no_contrastive (sequential)
(
    echo "[GPU2 $(date +%H:%M)] concat_semantic"
    CUDA_VISIBLE_DEVICES=2 $PYTHON $PROJ/scripts/run_ablation.py --gpu 0 --variant concat_semantic ${COMMON} > "$LOG_DIR/ablation_concat_semantic.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/ablation_concat_semantic.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU2 $(date +%H:%M)] concat_semantic F1=$F1"

    echo "[GPU2 $(date +%H:%M)] no_contrastive"
    CUDA_VISIBLE_DEVICES=2 $PYTHON $PROJ/scripts/run_ablation.py --gpu 0 --variant no_contrastive ${COMMON} > "$LOG_DIR/ablation_no_contrastive.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/ablation_no_contrastive.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU2 $(date +%H:%M)] no_contrastive F1=$F1"
    echo "[GPU2 $(date +%H:%M)] Done."
) > "$LOG_DIR/gpu2_ablation.log" 2>&1 &
PID2=$!

# GPU 3: ce_only (only 1)
(
    echo "[GPU3 $(date +%H:%M)] ce_only"
    CUDA_VISIBLE_DEVICES=3 $PYTHON $PROJ/scripts/run_ablation.py --gpu 0 --variant ce_only ${COMMON} > "$LOG_DIR/ablation_ce_only.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/ablation_ce_only.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU3 $(date +%H:%M)] ce_only F1=$F1"
    echo "[GPU3 $(date +%H:%M)] Done."
) > "$LOG_DIR/gpu3_ablation.log" 2>&1 &
PID3=$!

echo "Waiting PID=$PID1 $PID2 $PID3 ..."
wait $PID1 $PID2 $PID3

echo ""
echo "=== ALL DONE $(date) ==="
for variant in simple_concat concat_semantic ce_only no_contrastive frozen_bert; do
    f="$RESULT_DIR/ablation_${variant}.json"
    if [ -f "$f" ]; then
        python3 -c "import json;d=json.load(open('$f'));print(f\"  {d['variant']:20s} F1={d['f1']:.4f} Acc={d['accuracy']:.4f} Prec={d['precision']:.4f} Rec={d['recall']:.4f} AUC={d['auc']:.4f}\")"
    else
        echo "  $variant  (no result)"
    fi
done
echo "Logs: $LOG_DIR/"
