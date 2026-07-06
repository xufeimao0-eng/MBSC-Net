#!/bin/bash
# Multi-seed Chinese-CLIP training on weibo21 (event-level split)
# Seeds: 42, 123, 456
# 200 epochs, patience=50, GPU 3

PYTHON=python
LOG_DIR=$PROJ/results/logs

echo "============================================"
echo "Multi-Seed Chinese-CLIP Training"
echo "Dataset: weibo21 (event-level split)"
echo "Epochs: 200, Patience: 50"
echo "Seeds: 42 123 456"
echo "GPU: 3"
echo "Start: $(date)"
echo "============================================"

for seed in 42 123 456; do
    LOG="$LOG_DIR/weibo21_chinese_clip_seed${seed}.log"
    echo ""
    echo "[$(date)] Starting seed=$seed ..."
    CUDA_VISIBLE_DEVICES=3 $PYTHON train.py \
        --dataset weibo21 \
         \
        --seed $seed > "$LOG" 2>&1

    RET=$?
    echo "[$(date)] Seed=$seed done (exit=$RET)"

    # Extract best Val F1
    if [ -f "$LOG" ]; then
        BEST=$(grep "Best (F1=" "$LOG" | tail -1 | grep -oP 'F1=\K[0-9.]+')
        echo "  Best Val F1: ${BEST:-N/A}"
    fi
done

echo ""
echo "[$(date)] All seeds complete!"
