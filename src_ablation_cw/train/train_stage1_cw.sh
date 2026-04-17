#!/bin/bash
# ============================================================================
# 🚀 SC-Show-O CW Ablation Stage 1 (Cell-only) - 一键启动脚本
# ============================================================================
# 使用方法：
#   bash src_ablation_cw/train/train_stage1_cw.sh
#   bash src_ablation_cw/train/train_stage1_cw.sh fg
# ============================================================================

export TZ="Asia/Shanghai"
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "🕐 当前北京时间：$CURRENT_TIME"


# 设置网络代理
export http_proxy=http://10.8.36.50:3143
export https_proxy=http://10.8.36.50:3143
export HTTP_PROXY=http://10.8.36.50:3143
export HTTPS_PROXY=http://10.8.36.50:3143

JOB_NAME="same_as_cw_ablation_stage1"
PROJECT_ROOT="/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo"
LOG_DIR="$PROJECT_ROOT/run/logs"
LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
mkdir -p "$LOG_DIR"

echo "🚀 启动训练任务：$JOB_NAME"
echo "📝 日志路径：$LOG_FILE"

export SWANLAB_TIMEZONE="Asia/Shanghai"
set -e

CONFIG_PATH="$PROJECT_ROOT/src_ablation_cw/config/config_cw_ablation_cell_only.json"
ACCEL_CONFIG="$PROJECT_ROOT/src_ablation_cw/config/accelerate_cw_ablation_deepspeed.yaml"
SCRIPT_PATH="$PROJECT_ROOT/src_ablation_cw/train/train_stage1_cw_cell_only.py"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           CW Ablation Stage 1 (Cell-only, Pretrain Dialog)        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda")
ENV_PATH="/mnt/c20250607/user/wanghaoran/envs/envs/sc_showo"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"                                                                                                                                                                    
    echo "✅ Conda 环境已激活: $CONDA_DEFAULT_ENV"
    echo "   Python: $(which python)"                                                                                                                                                             
else            
    echo "❌ 未找到 conda.sh，尝试直接使用绝对路径..."
    export PATH="$ENV_PATH/bin:$PATH"
    echo "   Python: $(which python)"
fi

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_deepspeed
export MAX_JOBS=32
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

echo "✅ CUDA_HOME: $CUDA_HOME"
echo "📂 项目路径：$PROJECT_ROOT"
echo "📝 日志文件：$LOG_FILE"
echo "⚙️  配置文件：$CONFIG_PATH"
echo "⚙️  accelerate：$ACCEL_CONFIG"
echo ""

echo "🎮 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
echo ""

MODE="${1:-background}"

if [ "$MODE" == "foreground" ] || [ "$MODE" == "fg" ] || [ "$MODE" == "1" ]; then
  echo "🚀 以前台模式启动（实时日志）..."
  cd "$PROJECT_ROOT"
  accelerate launch     --config_file "$ACCEL_CONFIG"     --main_process_port=19991     "$SCRIPT_PATH" --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
else
  echo "🚀 以后台模式启动..."
  cd "$PROJECT_ROOT"
  nohup accelerate launch     --config_file "$ACCEL_CONFIG"     --main_process_port=19991     "$SCRIPT_PATH" --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &

  PID=$!
  echo "✅ 训练已在后台启动（PID: $PID）"
  sleep 3
  if ps -p $PID > /dev/null; then
    echo "✅ 进程运行正常"
    echo "📊 最新日志（最后 20 行）："
    tail -n 20 "$LOG_FILE" || true
    echo "💡 tail -f $LOG_FILE"
    echo "💡 kill $PID"
  else
    echo "❌ 进程启动失败，请检查日志：cat $LOG_FILE"
    exit 1
  fi
fi
