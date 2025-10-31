#!/usr/bin/env bash
# Sequential runner for: python3 run_training --cfg configs/material/<PATH>.yaml
# Usage: bash run_all.sh

set -euo pipefail

# --- [ENV / MODULES] ----------------------------------------------------------
# source ~/.bashrc            # uncomment if you need conda
# conda activate myenv        # or: module load cuda/12.1, etc.

# Optional: choose a GPU
# export CUDA_VISIBLE_DEVICES=0

# Optional: wait until GPU is idle before each run
wait_for_gpu() {
  # comment this function out if you don't need it
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] Waiting for free GPU memory..."
    while true; do
      # change the threshold (MiB) as you like
      if nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk '{sum+=$1} END{exit !(sum<100)}'; then
        break
      fi
      sleep 30
    done
  fi
}

# --- [CONFIGS TO RUN] ---------------------------------------------------------
# List ONLY the suffix after "configs/material/"
CONFIGS=(
  "syn/horse.yaml"
  "nerf/luyu.yaml"
  "nerf/potion.yaml"
)

# You can also auto-discover with a glob, e.g.:
# mapfile -t CONFIGS < <(cd configs/material && ls nerf_synthetic/*.yaml)

# --- [LOGGING] ----------------------------------------------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/run_${STAMP}"
mkdir -p "$LOG_ROOT"

# --- [RUN] --------------------------------------------------------------------
PREFIX="python3 run_training.py --cfg configs/material"

echo "[INFO] Starting sequential runs: ${#CONFIGS[@]} configs"
echo "[INFO] Logs: $LOG_ROOT"

for cfg in "${CONFIGS[@]}"; do
  safe_name="${cfg//\//_}"                 # replace / with _
  log_file="${LOG_ROOT}/${safe_name%.yaml}.log"

  echo "------------------------------------------------------------"
  echo "[START] $(date '+%F %T')  ${cfg}"
  wait_for_gpu

  # Run and stream logs to both console and file
  set +e
  ( time ${PREFIX}/${cfg} ) 2>&1 | tee -a "$log_file"
  status=${PIPESTATUS[0]}
  set -e

  if [[ $status -ne 0 ]]; then
    echo "[FAIL ] $(date '+%F %T')  ${cfg} (exit $status)"
    echo "${cfg} $status" >> "${LOG_ROOT}/failures.txt"
    # To stop on first failure, uncomment the next line:
    # exit $status
  else
    echo "[DONE ] $(date '+%F %T')  ${cfg}"
    echo "${cfg}" >> "${LOG_ROOT}/completed.txt"
  fi
done

echo "------------------------------------------------------------"
echo "[ALL DONE] $(date '+%F %T')  Logs in: $LOG_ROOT"
[[ -f "${LOG_ROOT}/failures.txt" ]] && echo "[WARN] Some runs failed. See failures.txt"
