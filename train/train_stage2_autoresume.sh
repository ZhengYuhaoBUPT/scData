#!/bin/bash
# ============================================================================
# 🚀 SC-Show-O Stage 2 Training - 自动断点续训脚本
# ============================================================================
# 使用方法：
#   # 后台运行（默认）
#   bash src/train/train_stage2_autoresume.sh
#
#   # 前台运行
#   bash src/train/train_stage2_autoresume.sh fg
#
#   # 实时日志
#   tail -f /mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo/run/logs/stage2_debug.log
# ============================================================================

# ========== 时间配置 ==========
export TZ="Asia/Shanghai"
CURRENT_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "🕐 当前北京时间：$CURRENT_TIME"

# ========== 任务配置 ==========
JOB_NAME="stage2_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/root/wanghaoran/zxy/project/sc_showo/run/logs"
mkdir -p "$LOG_DIR"
echo "🚀 启动训练任务：$JOB_NAME"
echo "📝 日志路径：$LOG_DIR/${JOB_NAME}.log"

# ========== SwanLab 配置 ==========
export SWANLAB_TIMEZONE="Asia/Shanghai"

set -e

PROJECT_ROOT="/mnt/c20250607/user/wanghaoran/zxy/zxy/zxy/project/sc_showo"
LOG_FILE="$PROJECT_ROOT/run/logs/stage2_debug.log"
ENV_PATH="/mnt/c20250607/user/wanghaoran/envs/envs/sc_showo"
ACCEL_CONFIG="$PROJECT_ROOT/accelerate_configs/sc_showo_multi_gpu_deepspeed.yaml"
TRAIN_SCRIPT="$PROJECT_ROOT/src/train/train_stage2_curriculum_v2_1d.py"
CONFIG_PATH="$PROJECT_ROOT/config/config.json"
RUNTIME_CONFIG="/tmp/sc_showo_stage2_autoresume_config.json"
RESTART_SLEEP_SEC=15
MAX_RESTARTS=1000000

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     SC-Show-O Stage 2 Training (Auto Resume + Crash Recovery)     ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/opt/conda")
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

export http_proxy=http://10.8.36.50:3143
export https_proxy=http://10.8.36.50:3143
export HTTP_PROXY=http://10.8.36.50:3143
export HTTPS_PROXY=http://10.8.36.50:3143

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_deepspeed
export MAX_JOBS=32
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

echo "✅ 使用环境内 CUDA: $CUDA_HOME"
echo "✅ 验证 NVCC 版本：$(nvcc -V | grep release)"
echo "✅ CUDA_HOME: $CUDA_HOME"
echo ""

mkdir -p "$(dirname "$LOG_FILE")"
echo "📂 项目路径：$PROJECT_ROOT"
echo "📝 日志文件：$LOG_FILE"
echo ""
echo "🎮 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo ""
echo "⚙️  检查 DeepSpeed 配置..."
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')" 2>/dev/null || true
python -c "import deepspeed; print('✅ DeepSpeed 导入成功')" 2>/dev/null || echo "⚠️  DeepSpeed 未安装"
echo ""

get_save_dir() {
    python - "$CONFIG_PATH" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
print(cfg["checkpoint"]["save_dir"])
PY
}

get_latest_ckpt_multiple_50() {
    local save_dir="$1"
    python - "$save_dir" <<'PY'
import os,re,sys
save_dir=sys.argv[1]
if not os.path.isdir(save_dir):
    print("")
    raise SystemExit(0)
best=(-1,"")
pat=re.compile(r"^checkpoint-step-(\d+)$")
for n in os.listdir(save_dir):
    m=pat.match(n)
    if not m:
        continue
    step=int(m.group(1))
    if step % 50 != 0:
        continue
    p=os.path.join(save_dir,n)
    if not os.path.isdir(p):
        continue
    if step>best[0]:
        best=(step,p)
print(best[1] if best[0]>=0 else "")
PY
}

write_runtime_config() {
    local resume_path="$1"
    python - "$CONFIG_PATH" "$RUNTIME_CONFIG" "$resume_path" <<'PY'
import json,sys
src,dst,resume=sys.argv[1],sys.argv[2],sys.argv[3]
cfg=json.load(open(src))
cfg.setdefault("checkpoint", {})
cfg["checkpoint"]["resume_from"] = resume if resume else None
with open(dst,"w") as f:
    json.dump(cfg,f,indent=2,ensure_ascii=False)
print(f"[AUTO-RESUME] runtime config written: {dst}")
print(f"[AUTO-RESUME] checkpoint.resume_from = {cfg['checkpoint']['resume_from']}")
PY
}

kill_residual_py310() {
    echo "[AUTO-RESUME] 清理残留训练进程..."
    pkill -9 -f "train_stage2_curriculum_v2_1d.py" || true
    pkill -9 -f "accelerate.*sc_showo_multi_gpu_deepspeed.yaml" || true
    pkill -9 -f "python3.10" || true
    sleep 2
}

run_supervisor_foreground() {
    cd "$PROJECT_ROOT"
    local restarts=0
    local last_resume=""

    while true; do
        local save_dir
        save_dir="$(get_save_dir)"
        local latest_ckpt
        latest_ckpt="$(get_latest_ckpt_multiple_50 "$save_dir")"

        if [ -n "$latest_ckpt" ]; then
            echo "[AUTO-RESUME] 检测到最新50倍数checkpoint: $latest_ckpt"
        else
            echo "[AUTO-RESUME] 未检测到可用checkpoint，将从头启动"
        fi

        write_runtime_config "$latest_ckpt"

        set +e
        accelerate launch \
            --config_file "$ACCEL_CONFIG" \
            --main_process_port=19999 \
            "$TRAIN_SCRIPT" \
            --config "$RUNTIME_CONFIG" 2>&1 | tee -a "$LOG_FILE"
        exit_code=${PIPESTATUS[0]}
        set -e

        if [ "$exit_code" -eq 0 ]; then
            echo "[AUTO-RESUME] 训练正常结束，退出监督循环"
            break
        fi

        restarts=$((restarts+1))
        echo "[AUTO-RESUME] 检测到异常退出，exit_code=$exit_code, restart=$restarts"

        if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
            echo "[AUTO-RESUME] 达到最大重启次数 $MAX_RESTARTS，停止"
            return 1
        fi

        kill_residual_py310

        if [ -n "$latest_ckpt" ] && [ "$last_resume" = "$latest_ckpt" ]; then
            echo "[AUTO-RESUME] 连续两次从同一checkpoint恢复失败: $latest_ckpt"
        fi
        last_resume="$latest_ckpt"

        echo "[AUTO-RESUME] ${RESTART_SLEEP_SEC}s 后自动重启..."
        sleep "$RESTART_SLEEP_SEC"
    done
}

MODE="${1:-background}"

if [ "$MODE" == "__supervise" ]; then
    run_supervisor_foreground
    exit $?
fi

if [ "$MODE" == "foreground" ] || [ "$MODE" == "fg" ] || [ "$MODE" == "1" ]; then
    echo "🚀 以前台模式启动（自动断点续训）..."
    echo ""
    run_supervisor_foreground
else
    echo "🚀 以后台模式启动（自动断点续训）..."
    echo ""
    nohup bash "$0" __supervise > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✅ 监督进程已后台启动（PID: $PID）"
    sleep 3
    if ps -p "$PID" > /dev/null; then
        echo "✅ 监督进程运行正常"
        echo "📊 最新日志（最后 30 行）："
        echo "────────────────────────────────────────────────────────────"
        tail -n 30 "$LOG_FILE"
        echo "────────────────────────────────────────────────────────────"
        echo "💡 常用命令："
        echo "   tail -f $LOG_FILE"
        echo "   watch -n 1 nvidia-smi"
        echo "   kill $PID"
    else
        echo "❌ 监督进程启动失败，请检查日志：$LOG_FILE"
        exit 1
    fi
fi
