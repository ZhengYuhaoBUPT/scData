#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/bgi/data/projects/multimodal/zyh/scData"
LMDB_ROOT="/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText"
RUNNER="$PROJECT_DIR/run_pathway_static_pipeline.sh"
LOG_DIR="$PROJECT_DIR/logs"
MASTER_LOG="$LOG_DIR/run_all_sampled_shards.log"
SAMPLE_RATIO="${1:-0.01}"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] batch launch start sample_ratio=${SAMPLE_RATIO}" | tee -a "$MASTER_LOG"

mapfile -t SHARDS < <(find "$LMDB_ROOT" -maxdepth 1 -type d -name '*.db' | sort)

for shard_path in "${SHARDS[@]}"; do
  shard_name="$(basename "$shard_path")"
  echo "[$(timestamp)] launch ${shard_name}" | tee -a "$MASTER_LOG"
  nohup bash "$RUNNER" shard-topk "$shard_name" "$SAMPLE_RATIO" \
    > "$LOG_DIR/nohup_${shard_name}.log" 2>&1 < /dev/null &
  echo "[$(timestamp)] pid=$! shard=${shard_name}" | tee -a "$MASTER_LOG"
done

echo "[$(timestamp)] batch launch done count=${#SHARDS[@]}" | tee -a "$MASTER_LOG"
