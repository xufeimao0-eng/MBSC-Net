#!/bin/bash
# pheme: 3 GPUs x 1 worker
cd "$(dirname "$0")/.."
COMMON="--dataset pheme   --patience 15"
LOG_DIR="$PROJ/results/logs/sensitivity_grid_pheme"
mkdir -p "$LOG_DIR"

COMBOS=(
"0.0,1.0" "0.1,0.9" "0.2,0.8" "0.3,0.7" "0.4,0.6"
"0.5,0.5" "0.6,0.4" "0.7,0.3" "0.8,0.2" "0.9,0.1" "1.0,0.0"
)

run_gpu() {
  local gpu=$1 log="$LOG_DIR/gpu${gpu}.log"
  local count=0
  for i in "${!COMBOS[@]}"; do
    [ $((i % 3)) -eq $((gpu - 1)) ] && count=$((count+1))
  done
  echo "[$(date +%H:%M:%S)] GPU${gpu} started ${count} combos" | tee "$log"
  for i in "${!COMBOS[@]}"; do
    [ $((i % 3)) -ne $((gpu - 1)) ] && continue
    IFS=',' read ce cont <<< "${COMBOS[$i]}"
    echo "" | tee -a "$log"
    echo "===== [$(date +%H:%M:%S)] GPU${gpu}: CE=${ce} CONT=${cont} =====" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/run_sensitivity.py \
      --ce "${ce}" --cont "${cont}" --gpu 0 \
      ${COMMON} 2>&1 | tee -a "$log"
    echo "===== [$(date +%H:%M:%S)] GPU${gpu}: Done CE=${ce} =====" | tee -a "$log"
  done
  echo "[$(date +%H:%M:%S)] GPU${gpu} FINISHED" | tee -a "$log"
}

for gpu in 1 2 3; do
  run_gpu $gpu &
done
wait
echo "ALL DONE — $(date)"
