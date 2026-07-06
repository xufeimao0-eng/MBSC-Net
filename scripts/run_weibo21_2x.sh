#!/bin/bash
# weibo21: 3 GPUs x 2 workers = 6 slots
cd "$(dirname "$0")/.."
COMMON="--dataset weibo21   --patience 15"
LOG_DIR="$PROJ/results/logs/sensitivity_grid_weibo21"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/gpu*_w*.log

COMBOS=(
"0.0,1.0" "0.1,0.9" "0.2,0.8" "0.3,0.7" "0.4,0.6"
"0.5,0.5" "0.6,0.4" "0.7,0.3" "0.8,0.2" "0.9,0.1" "1.0,0.0"
)

SLOTS=6
declare -a SD
for i in $(seq 0 $((SLOTS-1))); do SD[$i]=""; done

for i in "${!COMBOS[@]}"; do
  s=$((i % SLOTS))
  [ -z "${SD[$s]}" ] && SD[$s]="${COMBOS[$i]}" || SD[$s]="${SD[$s]}|${COMBOS[$i]}"
done

WORKERS_PER_GPU=2
echo "weibo21: 3 GPUs x ${WORKERS_PER_GPU} workers = ${SLOTS} slots, $(date)"
for s in $(seq 0 $((SLOTS-1))); do
  gpu=$((1 + s/WORKERS_PER_GPU))
  wid=$((s % WORKERS_PER_GPU))
  cnt=$(echo "${SD[$s]}" | tr '|' '\n' | wc -l)
  echo "  GPU${gpu}/W${wid}: ${cnt} combos"
done

run_worker() {
  local gpu=$1 wid=$2 data="$3"
  local log="$LOG_DIR/gpu${gpu}_w${wid}.log"
  IFS='|' read -ra combos <<< "$data"
  echo "[$(date +%H:%M:%S)] GPU${gpu}/W${wid} started ${#combos[@]} combos" | tee "$log"
  for combo in "${combos[@]}"; do
    IFS=',' read ce cont <<< "$combo"
    echo "" | tee -a "$log"
    echo "===== [$(date +%H:%M:%S)] GPU${gpu}/W${wid}: CE=${ce} CONT=${cont} =====" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/run_sensitivity.py \
      --ce "${ce}" --cont "${cont}" --gpu 0 \
      ${COMMON} 2>&1 | tee -a "$log"
    echo "===== [$(date +%H:%M:%S)] GPU${gpu}/W${wid}: Done CE=${ce} =====" | tee -a "$log"
  done
  echo "[$(date +%H:%M:%S)] GPU${gpu}/W${wid} FINISHED" | tee -a "$log"
}

for s in $(seq 0 $((SLOTS-1))); do
  gpu=$((1 + s/WORKERS_PER_GPU))
  wid=$((s % WORKERS_PER_GPU))
  run_worker $gpu $wid "${SD[$s]}" &
done
wait
echo "ALL DONE — $(date)"
