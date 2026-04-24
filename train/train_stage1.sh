#!/bin/bash
# ============================================================================
# 🚀 SC-Show-O Stage 1 Training - 一键启动脚本
# ============================================================================
# 使用方法：
#   # 后台运行（默认，推荐）
#   bash src/train/train_stage1.sh
#   
#   # 前台运行（实时查看日志）
#   bash src/train/train_stage1.sh fg
#   
#   # 查看日志
#   tail -f run/logs/stage1.log
# ============================================================================

# ========== 时间配置 ==========
# 🌏 设置北京时间为系统时区
export TZ="Asia/Shanghai"

# 📅 获取北京时间（格式：YYYY-MM-DD_HH-MM-SS）
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "🕐 当前北京时间：$CURRENT_TIME"

# ========== 任务配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SC_SHOWO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONFIG_PATH="${SC_SHOWO_CONFIG:-$PROJECT_ROOT/config/config.json}"
ACCEL_CONFIG="${SC_SHOWO_ACCEL_CONFIG:-$PROJECT_ROOT/accelerate_configs/sc_showo_multi_gpu_deepspeed.yaml}"
TRAIN_SCRIPT="${SC_SHOWO_STAGE1_SCRIPT:-$PROJECT_ROOT/train/train_stage1_bidirectional_rank_sin_rankloss.py}"
JOB_NAME="stage1_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SC_SHOWO_LOG_DIR:-$PROJECT_ROOT/run/logs}"
mkdir -p "$LOG_DIR"

echo "🚀 启动训练任务：$JOB_NAME"
echo "📝 日志路径：$LOG_DIR/${JOB_NAME}.log"

# ========== SwanLab 配置 ==========
# 🇨🇳 设置 SwanLab 使用北京时间
export SWANLAB_TIMEZONE="Asia/Shanghai"

set -e

LOG_FILE="$LOG_DIR/${JOB_NAME}.log"
CONDA_PATH="${SC_SHOWO_CONDA_PATH:-}"
CONDA_ENV="${SC_SHOWO_CONDA_ENV:-sc_showo}"
MAIN_PROCESS_PORT="${SC_SHOWO_MAIN_PROCESS_PORT:-19999}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           SC-Show-O Stage 1 Training                              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# ⚙️  手动激活 Conda 环境（不需要 conda init）
# ============================================================================

echo "🐍 检查 Conda 环境..."

# 可选：设置 SC_SHOWO_CONDA_PATH 后自动激活；否则使用当前 shell 环境。
if [ -n "$CONDA_PATH" ] && [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "✅ Conda 环境已激活：$CONDA_DEFAULT_ENV"
else
    echo "ℹ️  未设置 SC_SHOWO_CONDA_PATH，使用当前 Python 环境：$(command -v python || true)"
fi

# 设置 CUDA 环境变量；优先使用当前环境，不强依赖 Conda。
if [ -n "${CONDA_PREFIX:-}" ]; then
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
fi
if [ -n "${CUDA_HOME:-}" ]; then
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
fi

# ✅ 强制 DeepSpeed 每次重新编译 CPU Adam（不使用缓存）
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_deepspeed
export MAX_JOBS=32  # 限制编译使用的 CPU 核心数
# 延长心跳检查时间，防止误杀
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
# 禁用这个过于敏感的监控（如果确定代码没死循环）
export TORCH_NCCL_ENABLE_MONITORING=0
# 强制使用 Socket 通信作为辅助
export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"

# ✅ 开启 NCCL 调试日志（如果再卡住，它会告诉你哪张卡挂了）
export NCCL_DEBUG=INFO

echo "✅ CUDA_HOME: ${CUDA_HOME:-<not set>}"
if command -v nvcc >/dev/null 2>&1; then
    echo "✅ 验证 NVCC 版本: $(nvcc -V | grep release)"
else
    echo "ℹ️  nvcc 不在 PATH 中，跳过 NVCC 检查"
fi
echo ""

# ============================================================================
# 🏃 启动训练
# ============================================================================

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"

echo "📂 项目路径：$PROJECT_ROOT"
echo "📄 配置文件：$CONFIG_PATH"
echo "🚀 训练脚本：$TRAIN_SCRIPT"
echo "📝 日志文件：$LOG_FILE"
echo ""

if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ 配置文件不存在：$CONFIG_PATH"
    echo "💡 可设置 SC_SHOWO_CONFIG=/path/to/config.json"
    exit 1
fi
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ 训练脚本不存在：$TRAIN_SCRIPT"
    echo "💡 可设置 SC_SHOWO_STAGE1_SCRIPT=/path/to/train_script.py"
    exit 1
fi

ACCEL_ARGS=(--main_process_port="$MAIN_PROCESS_PORT")
if [ -f "$ACCEL_CONFIG" ]; then
    ACCEL_ARGS=(--config_file "$ACCEL_CONFIG" "${ACCEL_ARGS[@]}")
else
    echo "ℹ️  未找到 accelerate config：$ACCEL_CONFIG，将使用 accelerate 默认配置"
fi

# ✅ 检查 GPU
echo "🎮 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo ""

# ✅ DeepSpeed 兼容性检查
echo "⚙️  检查 DeepSpeed 配置..."
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')" 2>/dev/null || true
python -c "import deepspeed; print('✅ DeepSpeed 导入成功')" 2>/dev/null || echo "⚠️  DeepSpeed 未安装"
echo ""

# 获取运行模式（默认为 background）
MODE="${1:-background}"

if [ "$MODE" == "foreground" ] || [ "$MODE" == "fg" ] || [ "$MODE" == "1" ]; then
    echo "🚀 以前台模式启动（实时日志）..."
    echo ""
    
    cd "$PROJECT_ROOT"
    accelerate launch "${ACCEL_ARGS[@]}" "$TRAIN_SCRIPT" --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
else
    echo "🚀 以后台模式启动..."
    echo ""
    
    cd "$PROJECT_ROOT"
    nohup accelerate launch "${ACCEL_ARGS[@]}" "$TRAIN_SCRIPT" --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "✅ 训练已在后台启动（PID: $PID）"
    echo ""
    
    # 等待 3 秒检查是否成功启动
    sleep 3
    if ps -p $PID > /dev/null; then
        echo "✅ 进程运行正常"
        echo ""
        echo "📊 最新日志（最后 20 行）："
        echo "────────────────────────────────────────────────────────────"
        tail -n 20 "$LOG_FILE"
        echo "────────────────────────────────────────────────────────────"
        echo ""
        echo "💡 常用命令："
        echo "   # 实时查看日志"
        echo "   tail -f $LOG_FILE"
        echo ""
        echo "   # 查看 GPU 使用"
        echo "   watch -n 1 nvidia-smi"
        echo ""
        echo "   # 停止训练"
        echo "   kill $PID"
        echo ""
    else
        echo "❌ 进程启动失败，请检查日志："
        echo "   cat $LOG_FILE"
        exit 1
    fi
fi