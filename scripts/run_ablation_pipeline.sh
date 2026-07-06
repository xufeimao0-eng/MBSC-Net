#!/bin/bash
# Ablation pipeline: train → evaluate → delete checkpoint → next
# Usage: bash scripts/run_ablation_pipeline.sh <GPU> <EXP_LABEL>
#   where EXP_LABEL is A,B,D,E,F or a comma-separated list like "A,B,E"

PYTHON=python
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS=/results/logs/weibo/ablation_results.jsonl

GPU=$1
EXPS=$2

if [ -z "$GPU" ] || [ -z "$EXPS" ]; then
    echo "Usage: bash run_ablation_pipeline.sh <GPU> <EXP_LIST>"
    echo "  EXP_LIST: A,B,D,E,F (comma-separated, e.g. 'A,B')"
    exit 1
fi

# Experiment definitions
declare -A FUSION LOSS FREEZE LABEL
FUSION[A]="simple_concat";   LOSS[A]="full";             FREEZE[A]="False"; LABEL[A]="A_simple_concat"
FUSION[B]="concat_semantic"; LOSS[B]="full";             FREEZE[B]="False"; LABEL[B]="B_concat_semantic"
FUSION[C]="full";            LOSS[C]="full";             FREEZE[C]="False"; LABEL[C]="C_full"
FUSION[D]="full";            LOSS[D]="ce_only";          FREEZE[D]="False"; LABEL[D]="D_ce_only"
FUSION[E]="full";            LOSS[E]="no_contrastive";   FREEZE[E]="False"; LABEL[E]="E_ce_align"
FUSION[F]="full";            LOSS[F]="full";             FREEZE[F]="True";  LABEL[F]="F_frozen_bert"

echo "============================================"
echo "Ablation Pipeline — GPU $GPU"
echo "Experiments: $EXPS"
echo "Start: $(date)"
echo "============================================"

IFS=',' read -ra EXP_LIST <<< "$EXPS"

for exp in "${EXP_LIST[@]}"; do
    fusion="${FUSION[$exp]}"
    loss="${LOSS[$exp]}"
    freeze="${FREEZE[$exp]}"
    label="${LABEL[$exp]}"

    if [ -z "$fusion" ]; then
        echo "Unknown experiment: $exp (valid: A,B,D,E,F)"
        continue
    fi

    echo ""
    echo "============================================"
    echo "[$(date +%H:%M)] Starting $label"
    echo "  fusion=$fusion  loss=$loss  freeze_bert=$freeze"
    echo "============================================"

    # Step 0: Clean old checkpoint if exists
    CkPT_DIR="$PROJ/results/checkpoints/weibo/ablation/${fusion}/seed_42"
    if [ "$fusion" = "full" ] && [ "$loss" = "full" ] && [ "$freeze" = "False" ]; then
        CkPT_DIR="$PROJ/results/checkpoints/weibo/default/seed_42"
    elif [ "$fusion" = "full" ] && [ "$loss" != "full" ]; then
        CkPT_DIR="$PROJ/results/checkpoints/weibo/ablation/${loss}/seed_42"
    elif [ "$freeze" = "True" ]; then
        CkPT_DIR="$PROJ/results/checkpoints/weibo/ablation/frozen_bert/seed_42"
    fi
    rm -f "$CkPT_DIR/best_model.pt"

    # Step 1: Train
    echo "[$(date +%H:%M)] Training..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -c "
import sys; sys.path.insert(0, '$PROJ')
from config import Config
Config.ABLATION_FUSION = '$fusion'
Config.ABLATION_LOSS = '$loss'
Config.ABLATION_FREEZE_BERT = $freeze
from train import train
train('weibo', 'bert_chinese', 'clip', 42)
" > $PROJ/results/logs/ablation_${label}_run2.log 2>&1

    TRAIN_RET=$?
    echo "[$(date +%H:%M)] Training exit=$TRAIN_RET"

    if [ $TRAIN_RET -ne 0 ]; then
        echo "  ERROR: Training failed, skipping..."
        continue
    fi

    # Step 2: Evaluate
    CkPT="$CkPT_DIR/best_model.pt"
    if [ ! -f "$CkPT" ]; then
        echo "  ERROR: No checkpoint at $CkPT"
        continue
    fi

    echo "[$(date +%H:%M)] Evaluating..."
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON $PROJ/scripts/eval_ablation_single.py \
        "$CkPT" "$fusion" "$loss" "$freeze" "$RESULTS" 2>&1

    EVAL_RET=$?
    if [ $EVAL_RET -ne 0 ]; then
        echo "  ERROR: Evaluation failed"
    fi

    # Step 3: Delete checkpoint
    echo "[$(date +%H:%M)] Deleting checkpoint..."
    rm -f "$CkPT"
    rm -f "$CkPT_DIR/history.json"
    echo "  Deleted: $CkPT"

    echo "[$(date +%H:%M)] $label — COMPLETE"
done

echo ""
echo "============================================"
echo "Pipeline done! $(date)"
echo "Results: $RESULTS"
echo "============================================"
cat "$RESULTS" 2>/dev/null | while read line; do
    echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin.readline()); print(f'{d[\"fusion\"]}/{d[\"loss\"]}: Val={d[\"val_f1\"]:.4f} Test F1={d[\"test_f1\"]:.4f} AUC={d[\"test_auc\"]:.4f}')" 2>/dev/null
done
