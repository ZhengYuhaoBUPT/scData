#!/usr/bin/env bash
set -euo pipefail

PYTHON="/root/miniconda3/envs/scgpt/bin/python"
PROJECT_DIR="/data/bgi/data/projects/multimodal/zyh/scData"
SCRIPT_PATH="$PROJECT_DIR/scripts/train_scgene_qformer.py"
OUTPUT_DIR="$PROJECT_DIR/outputs/training_runs/scgene_qformer_formal_v1"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/scgene_qformer_formal_v1.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

exec "$PYTHON" "$SCRIPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-topk-cells 2000 \
  --max-train-cells 256 \
  --encoder-batch-size 16 \
  --train-batch-size 16 \
  --num-epochs 10 \
  --learning-rate 1e-4
