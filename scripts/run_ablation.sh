#!/bin/bash
# Ablation study for MBSC-Net model on weibo21
# 6 experiments: A(simple_concat), B(concat_semantic), C(full MBSC-Net),
#                D(ce_only), E(CE+Contrastive), F(frozen_bert)
# Single seed (42), 200 epochs, patience=50

PYTHON=python
PROJ="$(cd "$(dirname "$0")/.." && pwd)"

run_ablation() {
    local NAME=$1
    local FUSION=$2
    local LOSS=$3
    local FREEZE=$4
    local GPU=$5

    LOG="$PROJ/results/logs/ablation_${NAME}.log"
    echo "[$(date +%H:%M)] Starting $NAME (GPU=$GPU) fusion=$FUSION loss=$LOSS freeze=$FREEZE"

    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -c "
import sys; sys.path.insert(0, '$PROJ')
from config import Config
Config.ABLATION_FUSION = '$FUSION'
Config.ABLATION_LOSS = '$LOSS'
Config.ABLATION_FREEZE_BERT = $FREEZE
from train import train
train('weibo', 'bert_chinese', 'clip', 42)
" > $LOG 2>&1 &

    echo "  PID: $!"
}

echo "============================================"
echo "MBSC-Net Ablation Study — 6 Experiments"
echo "Dataset: weibo21  |  Seed: 42  |  Epochs: 200  |  Patience: 50"
echo "Start: $(date)"
echo "============================================"

# Fusion ablations
run_ablation "A_simple_concat"  "simple_concat"    "full"  "False"  1
run_ablation "B_concat_semantic" "concat_semantic" "full"  "False"  2
# C (full MBSC-Net) already done — use checkpoint at results/checkpoints/weibo/default/seed_42/

# Loss ablations
run_ablation "D_ce_only"        "full"  "ce_only"       "False"  3
# E (CE+Align) = no_contrastive
run_ablation "E_ce_align"       "full"  "no_contrastive" "False"  1  # after A finishes on GPU1

# Frozen BERT
run_ablation "F_frozen_bert"    "full"  "full"  "True"   2  # after B finishes on GPU2

echo ""
echo "[$(date +%H:%M)] All launched!"
echo "============================================"
echo "Monitor:"
echo "  tail -5 results/logs/ablation_*.log"
echo ""
echo "After all complete, run:"
echo "  bash scripts/eval_ablation.sh"
echo "============================================"
