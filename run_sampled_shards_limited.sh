#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/bgi/data/projects/multimodal/zyh/scData"
LMDB_ROOT="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText"
RUNNER="$PROJECT_DIR/run_pathway_static_pipeline.sh"
OUTPUT_DIR="$PROJECT_DIR/outputs/pathway_static_pipeline/per_shard_topk"
LOG_DIR="$PROJECT_DIR/logs"
MASTER_LOG="$LOG_DIR/run_sampled_shards_limited.log"
SAMPLE_RATIO="${1:-0.01}"
MAX_PARALLEL="${2:-4}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

count_running() {
  pgrep -f "$PROJECT_DIR/pathway_static_pipeline.py shard-topk" | wc -l || true
}

wait_for_slot() {
  while true; do
    local running
    running="$(count_running)"
    if [ "$running" -lt "$MAX_PARALLEL" ]; then
      break
    fi
    sleep 10
  done
}

echo "[$(timestamp)] limited batch launch start sample_ratio=${SAMPLE_RATIO} max_parallel=${MAX_PARALLEL}" | tee -a "$MASTER_LOG"

mapfile -t SHARDS < <(find "$LMDB_ROOT" -maxdepth 1 -type d -name '*.db' | sort)

launched=0
skipped=0

for shard_path in "${SHARDS[@]}"; do
  shard_name="$(basename "$shard_path")"
  result_path="$OUTPUT_DIR/${shard_name}.topk.json"
  if [ -f "$result_path" ]; then
    echo "[$(timestamp)] skip completed ${shard_name}" | tee -a "$MASTER_LOG"
    skipped=$((skipped + 1))
    continue
  fi

  wait_for_slot

  echo "[$(timestamp)] launch ${shard_name}" | tee -a "$MASTER_LOG"
  nohup bash "$RUNNER" shard-topk "$shard_name" "$SAMPLE_RATIO" \
    > "$LOG_DIR/nohup_${shard_name}.log" 2>&1 < /dev/null &
  echo "[$(timestamp)] pid=$! shard=${shard_name}" | tee -a "$MASTER_LOG"
  launched=$((launched + 1))
  sleep 1
done

echo "[$(timestamp)] limited batch dispatch done launched=${launched} skipped=${skipped}" | tee -a "$MASTER_LOG"
