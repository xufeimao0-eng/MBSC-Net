#!/bin/bash
# PHEME 对比实验 — 3 GPU (1/2/3) 并行，每卡2个模型
# 8 models: spotfake, eann, safe, hmcan, mdfend, pivot, bmr, c3n
# GPU1: spotfake, eann (+ bmr 等前两个中任一完成)
# GPU2: safe, hmcan (+ c3n 等前两个中任一完成)
# GPU3: mdfend, pivot
set -e
PYTHON=python
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR=$PROJ/results/logs/pheme
RESULT_DIR=$PROJ/results/logs/pheme
mkdir -p "$LOG_DIR" "$RESULT_DIR"
rm -f "$LOG_DIR"/baseline_pheme_*.log "$LOG_DIR"/gpu*_baselines.log

echo "============================================"
echo "PHEME Baseline — GPU 1/2/3 x2 — $(date)"
echo "  Text: CLIP (English)"
echo "============================================"

# GPU 1: spotfake & eann (parallel), then bmr (sequential)
(
    echo "[GPU1 $(date +%H:%M)] spotfake & eann"
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 1 --model spotfake --dataset pheme > "$LOG_DIR/baseline_pheme_spotfake.log" 2>&1 &
    P1=$!
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 1 --model eann --dataset pheme > "$LOG_DIR/baseline_pheme_eann.log" 2>&1 &
    P2=$!
    wait $P1; F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_spotfake.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU1 $(date +%H:%M)] spotfake F1=$F1"
    wait $P2; F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_eann.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU1 $(date +%H:%M)] eann F1=$F1"
    # bmr after spotfake & eann both done
    echo "[GPU1 $(date +%H:%M)] bmr"
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 1 --model bmr --dataset pheme > "$LOG_DIR/baseline_pheme_bmr.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_bmr.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU1 $(date +%H:%M)] bmr F1=$F1"
    echo "[GPU1 $(date +%H:%M)] Done."
) > "$LOG_DIR/gpu1_baselines.log" 2>&1 &
PID1=$!

# GPU 2: safe & hmcan (parallel), then c3n (sequential)
(
    echo "[GPU2 $(date +%H:%M)] safe & hmcan"
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 2 --model safe --dataset pheme > "$LOG_DIR/baseline_pheme_safe.log" 2>&1 &
    P1=$!
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 2 --model hmcan --dataset pheme > "$LOG_DIR/baseline_pheme_hmcan.log" 2>&1 &
    P2=$!
    wait $P1; F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_safe.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU2 $(date +%H:%M)] safe F1=$F1"
    wait $P2; F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_hmcan.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU2 $(date +%H:%M)] hmcan F1=$F1"
    # c3n after safe & hmcan both done
    echo "[GPU2 $(date +%H:%M)] c3n"
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 2 --model c3n --dataset pheme > "$LOG_DIR/baseline_pheme_c3n.log" 2>&1
    F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_c3n.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU2 $(date +%H:%M)] c3n F1=$F1"
    echo "[GPU2 $(date +%H:%M)] Done."
) > "$LOG_DIR/gpu2_baselines.log" 2>&1 &
PID2=$!

# GPU 3: mdfend & pivot (parallel, only 2 models)
(
    echo "[GPU3 $(date +%H:%M)] mdfend & pivot"
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 3 --model mdfend --dataset pheme > "$LOG_DIR/baseline_pheme_mdfend.log" 2>&1 &
    P1=$!
    $PYTHON $PROJ/scripts/run_baselines.py --gpu 3 --model pivot --dataset pheme > "$LOG_DIR/baseline_pheme_pivot.log" 2>&1 &
    P2=$!
    wait $P1; F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_mdfend.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU3 $(date +%H:%M)] mdfend F1=$F1"
    wait $P2; F1=$(python3 -c "import json;d=json.load(open('$RESULT_DIR/baseline_pivot.json'));print(d['f1'])" 2>/dev/null||echo N/A)
    echo "[GPU3 $(date +%H:%M)] pivot F1=$F1"
    echo "[GPU3 $(date +%H:%M)] Done."
) > "$LOG_DIR/gpu3_baselines.log" 2>&1 &
PID3=$!

echo "Waiting PID=$PID1 $PID2 $PID3 ..."
wait $PID1 $PID2 $PID3

echo ""
echo "=== ALL DONE $(date) ==="
for model in spotfake eann safe hmcan mdfend pivot bmr c3n; do
    f="$RESULT_DIR/baseline_${model}.json"
    if [ -f "$f" ]; then
        F1=$(python3 -c "import json;d=json.load(open('$f'));print(f\"{d['f1']:.4f}\")")
        ACC=$(python3 -c "import json;d=json.load(open('$f'));print(f\"{d['accuracy']:.4f}\")")
        echo "  $model  F1=$F1  Acc=$ACC"
    else
        echo "  $model  (no result)"
    fi
done
echo "Logs: $LOG_DIR/"
