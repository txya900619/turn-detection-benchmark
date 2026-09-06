#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/mnt/md0/user_wayne/.hf_cache
EASY_TURN_GPU="${EASY_TURN_GPU:-2}"
export CUDA_VISIBLE_DEVICES="$EASY_TURN_GPU"

uv run --frozen turn-eval run-fastturn \
  --suite full \
  --model easy-turn \
  --device cuda:0 \
  --batch-size "${EASY_TURN_BATCH_SIZE:-1}" \
  --progress-interval-s 30 \
  --local-files-only \
  "$@"
