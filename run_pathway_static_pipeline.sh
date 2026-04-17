#!/usr/bin/env bash
set -euo pipefail

PYTHON="/root/miniconda3/envs/scgpt/bin/python"
PROJECT_DIR="/data/bgi/data/projects/multimodal/zyh/scData"
SCRIPT_PATH="$PROJECT_DIR/pathway_static_pipeline.py"
OUTPUT_DIR="$PROJECT_DIR/outputs/pathway_static_pipeline"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_shard_topk() {
  local shard_name="$1"
  local sample_ratio="${2:-1.0}"
  local log_file="$LOG_DIR/pathway_static_${shard_name}.log"
  echo "[$(timestamp)] shard-topk start ${shard_name} sample_ratio=${sample_ratio}"
  "$PYTHON" "$SCRIPT_PATH" shard-topk \
    --shard-name "$shard_name" \
    --sample-ratio "$sample_ratio" \
    --output-dir "$OUTPUT_DIR" \
    > "$log_file" 2>&1
  echo "[$(timestamp)] shard-topk finished ${shard_name}"
}

run_merge() {
  local log_file="$LOG_DIR/pathway_static_merge.log"
  echo "[$(timestamp)] merge-topk start"
  "$PYTHON" "$SCRIPT_PATH" merge-topk --output-dir "$OUTPUT_DIR" > "$log_file" 2>&1
  echo "[$(timestamp)] merge-topk finished"
}

run_build() {
  local log_file="$LOG_DIR/pathway_static_build.log"
  echo "[$(timestamp)] build-prototypes start"
  "$PYTHON" "$SCRIPT_PATH" build-prototypes --output-dir "$OUTPUT_DIR" > "$log_file" 2>&1
  echo "[$(timestamp)] build-prototypes finished"
}

case "${1:-}" in
  shard-topk)
    run_shard_topk "$2" "${3:-1.0}"
    ;;
  merge-topk)
    run_merge
    ;;
  build-prototypes)
    run_build
    ;;
  *)
    echo "Usage:"
    echo "  $0 shard-topk <split_taskX_writerY.db> [sample_ratio]"
    echo "  $0 merge-topk"
    echo "  $0 build-prototypes"
    exit 1
    ;;
esac
