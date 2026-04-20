#!/usr/bin/env bash
set -euo pipefail

PYTHON="/root/miniconda3/envs/scgpt/bin/python"
PROJECT_DIR="/data/bgi/data/projects/multimodal/zyh/scData"
SCRIPT_PATH="$PROJECT_DIR/scripts/build_topk_from_scgpt_lmdb.py"
LMDB_ROOT="/home/qijinyin/wanghaoran/zxy/features/per_gene_feat/whitelist_lmdb_8shards_compact"
OUTPUT_DIR="$PROJECT_DIR/outputs/scgpt_8shards_topk/per_shard_topk"
LOG_DIR="$PROJECT_DIR/logs/scgpt_8shards_topk"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

mapfile -t SHARDS < <(find "$LMDB_ROOT" -maxdepth 1 -type d -name 'shard_*.db' | sort)

for idx in "${!SHARDS[@]}"; do
  shard_path="${SHARDS[$idx]}"
  shard_name="$(basename "$shard_path")"
  gpu_id="$idx"
  nohup env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" "$SCRIPT_PATH" \
    --lmdb-path "$shard_path" \
    --output-dir "$OUTPUT_DIR" \
    > "$LOG_DIR/${shard_name}.log" 2>&1 < /dev/null &
  echo "launched shard=${shard_name} gpu=${gpu_id} pid=$!"
done
