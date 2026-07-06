#!/bin/bash
# ===========================================================================
# weibo21: Loss Weight Sensitivity Grid Search -- MBSC-Net (Hadamard)
# 11 combos, 3 GPUs (1,2,3) x 2 workers = 6 parallel slots
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$0")/.."

COMBOS=(
  "0.0 1.0"  "0.1 0.9"  "0.2 0.8"  "0.3 0.7"  "0.4 0.6"
  "0.5 0.5"  "0.6 0.4"  "0.7 0.3"  "0.8 0.2"  "0.9 0.1"  "1.0 0.0"
)

NUM_COMBOS=${#COMBOS[@]}
GPU_ID_BASE=1
NUM_GPUS=3
WORKERS_PER_GPU=2
TOTAL_SLOTS=$((NUM_GPUS * WORKERS_PER_GPU))

echo "============================================"
echo " weibo21: Loss Weight Sensitivity Grid"
echo " MBSC-Net (Hadamard)"
echo " Total: ${NUM_COMBOS} combos (ce+cont=1.0, step=0.1)"
echo " GPUs: ${NUM_GPUS} x ${WORKERS_PER_GPU} workers = ${TOTAL_SLOTS} slots"
echo " Started: $(date)"
echo "============================================"

declare -a SLOT_COMBOS SLOT_COUNTS
for i in $(seq 0 $((TOTAL_SLOTS - 1))); do
  SLOT_COMBOS[$i]=""
  SLOT_COUNTS[$i]=0
done

for i in "${!COMBOS[@]}"; do
  slot=$((i % TOTAL_SLOTS))
  if [ -z "${SLOT_COMBOS[$slot]}" ]; then
    SLOT_COMBOS[$slot]="${COMBOS[$i]}"
  else
    SLOT_COMBOS[$slot]="${SLOT_COMBOS[$slot]}|${COMBOS[$i]}"
  fi
  SLOT_COUNTS[$slot]=$((SLOT_COUNTS[$slot] + 1))
done

for slot in $(seq 0 $((TOTAL_SLOTS - 1))); do
  gpu=$((GPU_ID_BASE + slot / WORKERS_PER_GPU))
  worker=$((slot % WORKERS_PER_GPU))
  echo "Slot ${slot} (GPU${gpu}/W${worker}): ${SLOT_COUNTS[$slot]} combos"
done
echo "============================================"

COMMON_ARGS="--dataset weibo21   --patience 15"
LOG_DIR="$PROJ/results/logs/sensitivity_grid_weibo21"
mkdir -p "${LOG_DIR}"

run_worker() {
  local gpu_id=$1 worker_id=$2 slot=$3 combos_str=$4
  local log_file="${LOG_DIR}/gpu${gpu_id}_w${worker_id}.log"

  echo "[$(date +%H:%M:%S)] GPU${gpu_id}/W${worker_id} started with ${SLOT_COUNTS[$slot]} combos" | tee -a "${log_file}"
  IFS='|' read -ra combos <<< "${combos_str}"

  for combo in "${combos[@]}"; do
    read ce cont <<< "${combo}"
    echo "" | tee -a "${log_file}"
    echo "===== [$(date +%H:%M:%S)] GPU${gpu_id}/W${worker_id}: CE=${ce} CONT=${cont} =====" | tee -a "${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 scripts/run_sensitivity.py \
      --ce "${ce}" --cont "${cont}" --gpu 0 \
      ${COMMON_ARGS} 2>&1 | tee -a "${log_file}"

    echo "===== [$(date +%H:%M:%S)] GPU${gpu_id}/W${worker_id}: Done CE=${ce} =====" | tee -a "${log_file}"
  done

  echo "" | tee -a "${log_file}"
  echo "[$(date +%H:%M:%S)] GPU${gpu_id}/W${worker_id} FINISHED all ${SLOT_COUNTS[$slot]} combos" | tee -a "${log_file}"
}

PIDS=()
for slot in $(seq 0 $((TOTAL_SLOTS - 1))); do
  gpu=$((GPU_ID_BASE + slot / WORKERS_PER_GPU))
  worker=$((slot % WORKERS_PER_GPU))
  run_worker ${gpu} ${worker} ${slot} "${SLOT_COMBOS[$slot]}" &
  PIDS+=($!)
  echo "Launched slot ${slot}: GPU${gpu}/W${worker} PID=${PIDS[-1]}"
done

echo ""
echo "Logs: ${LOG_DIR}/gpu{N}_w{M}.log"
echo "Waiting for all ${TOTAL_SLOTS} workers to finish..."
wait "${PIDS[@]}"

echo ""
echo "============================================"
echo " ALL DONE at $(date)"
echo " Results: $PROJ/results/checkpoints/weibo21_sensitivity_hadamard/"
