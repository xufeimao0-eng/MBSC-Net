#!/bin/bash
cd "$(dirname "$0")/.."
COMMON="--dataset pheme   --patience 15"
LOG_DIR="$PROJ/results/logs/sensitivity_grid_pheme"
rm -f "$LOG_DIR"/gpu*_w*.log

COMBOS=(
"0.0,1.0" "0.1,0.9" "0.2,0.8" "0.3,0.7" "0.4,0.6"
"0.5,0.5" "0.6,0.4" "0.7,0.3" "0.8,0.2" "0.9,0.1" "1.0,0.0"
)
SLOTS=6
WORKERS_PER_GPU=2
declare -a SD
for i in $(seq 0 $((SLOTS-1))); do SD[$i]=""; done
for i in "${!COMBOS[@]}"; do
  s=$((i % SLOTS))
  if [ -z "${SD[$s]}" ]; then SD[$s]="${COMBOS[$i]}"; else SD[$s]="${SD[$s]}|${COMBOS[$i]}"; fi
done

echo "pheme: 3 GPUs x ${WORKERS_PER_GPU} workers = ${SLOTS} slots, $(date)"
for s in $(seq 0 $((SLOTS-1))); do
  gpu=$((1 + s/WORKERS_PER_GPU))
  wid=$((s % WORKERS_PER_GPU))
  cnt=$(echo "${SD[$s]}" | tr '|' '\n' | wc -l)
  echo "  GPU${gpu}/W${wid}: ${cnt} combos"
done

run_worker() {
  local gpu=$1 wid=$2 log="$LOG_DIR/gpu${gpu}_w${wid}.log"
  IFS='|' read -ra combos <<< "$3"
  echo "[$(date +%H:%M:%S)] GPU${gpu}/W${wid} started ${#combos[@]} combos" | tee "$log"
  for combo in "${combos[@]}"; do
    IFS=',' read ce ct <<< "$combo"
    echo "" | tee -a "$log"
    echo "===== [$(date +%H:%M:%S)] GPU${gpu}/W${wid}: CE=${ce} CONT=${ct} =====" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=${gpu} python3 scripts/run_sensitivity.py \
      --ce "${ce}" --cont "${ct}" --gpu 0 ${COMMON} 2>&1 | tee -a "$log"
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
