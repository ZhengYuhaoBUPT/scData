#!/bin/bash
# ============================================================================
# 🚀 SC-Show-O Stage 2 Training - 一键启动脚本
# ============================================================================
# 使用方法：
#   # 后台运行（默认，推荐）
#   bash src/train/train_stage2.sh
#   
#   # 前台运行（实时查看日志）
#   bash src/train/train_stage2.sh fg
#   
#   # 查看日志
#   tail -f run/logs/stage2.log
# ============================================================================

# ========== 时间配置 ==========
# 🌏 设置北京时间为系统时区
export TZ="Asia/Shanghai"

# 📅 获取北京时间（格式：YYYY-MM-DD_HH-MM-SS）
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "🕐 当前北京时间：$CURRENT_TIME"

# ========== 任务配置 ==========
JOB_NAME="stage2_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/root/wanghaoran/zxy/project/sc_showo/run/logs"
mkdir -p "$LOG_DIR"

echo "🚀 启动训练任务：$JOB_NAME"
echo "📝 日志路径：$LOG_DIR/${JOB_NAME}.log"

# ========== SwanLab 配置 ==========
# 🇨🇳 设置 SwanLab 使用北京时间
export SWANLAB_TIMEZONE="Asia/Shanghai"

set -e

PROJECT_ROOT="/root/wanghaoran/zxy/project/sc_showo"
LOG_FILE="$PROJECT_ROOT/run/logs/stage2.log"
CONDA_PATH="/home/qijinyin/miniconda3"  # 根据你的实际路径调整

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           SC-Show-O Stage 2 Training (SFT + Gene Diffusion)       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# ⚙️  手动激活 Conda 环境（不需要 conda init）
# ============================================================================

echo "🐍 准备激活 Conda 环境..."

# 方法：直接 source conda.sh 并激活环境
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate sc_showo
    echo "✅ Conda 环境已激活：$CONDA_DEFAULT_ENV"
else
    echo "⚠️  未找到 Conda 安装路径：$CONDA_PATH"
    echo "💡 请修改脚本中的 CONDA_PATH 变量为你的实际路径"
    exit 1
fi

# 设置 CUDA 环境变量
# 自动获取当前 Conda 环境路径
export CUDA_HOME=$CONDA_PREFIX

# 确保路径优先级
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# ✅ 强制 DeepSpeed 每次重新编译 CPU Adam（不使用缓存）
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_deepspeed
export MAX_JOBS=32  # 限制编译使用的 CPU 核心数
# 延长心跳检查时间，防止误杀
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
# 禁用这个过于敏感的监控（如果确定代码没死循环）
export TORCH_NCCL_ENABLE_MONITORING=0
# 强制使用 Socket 通信作为辅助
export NCCL_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ✅ 开启 NCCL 调试日志（如果再卡住，它会告诉你哪张卡挂了）
export NCCL_DEBUG=INFO

echo "✅ 使用环境内 CUDA: $CUDA_HOME"
echo "✅ 验证 NVCC 版本：$(nvcc -V | grep release)"

echo "✅ CUDA_HOME: $CUDA_HOME"
echo ""

# ============================================================================
# 🏃 启动训练
# ============================================================================

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"

echo "📂 项目路径：$PROJECT_ROOT"
echo "📝 日志文件：$LOG_FILE"
echo ""

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
    accelerate launch \
        --config_file "$PROJECT_ROOT/accelerate_configs/sc_showo_multi_gpu_deepspeed.yaml" \
        --main_process_port=19999 \
        "$PROJECT_ROOT/src/train/train_stage2_curriculum_lora.py" 2>&1 | tee -a "$LOG_FILE"
else
    echo "🚀 以后台模式启动..."
    echo ""
    
    cd "$PROJECT_ROOT"
    nohup accelerate launch \
        --config_file "$PROJECT_ROOT/accelerate_configs/sc_showo_multi_gpu_deepspeed.yaml" \
        --main_process_port=19999 \
        "$PROJECT_ROOT/src/train/train_stage2_curriculum_lora.py" > "$LOG_FILE" 2>&1 &
    
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