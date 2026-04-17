#!/usr/bin/env bash
set -euo pipefail

PYTHON="/root/miniconda3/envs/scgpt/bin/python"
PROJECT_DIR="/data/bgi/data/projects/multimodal/zyh/scData"
SCRIPT_PATH="$PROJECT_DIR/scripts/train_scgene_qformer.py"
OUTPUT_DIR="$PROJECT_DIR/outputs/training_runs/scgene_qformer_merged_formal_v1"

mkdir -p "$OUTPUT_DIR"

exec "$PYTHON" "$SCRIPT_PATH" \
  --topk-json "$PROJECT_DIR/outputs/pathway_static_pipeline/merged_topk.json" \
  --output-dir "$OUTPUT_DIR" \
  --max-topk-cells 4000 \
  --max-train-cells 512 \
  --encoder-batch-size 16 \
  --train-batch-size 16 \
  --num-epochs 10 \
  --learning-rate 1e-4
