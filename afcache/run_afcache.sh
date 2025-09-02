#!/usr/bin/env bash
# ------------------------------------------------------------
#  FLUX-AFCache 多卡推理脚本（重构版）
# ------------------------------------------------------------
set -euo pipefail   # 遇到错误即退出；变量未定义即报错；管道出错即失败
[[ "${TRACE:-0}" == "1" ]] && set -x   # TRACE=1 ./run.sh 可调试

# ---------- 0. 基础环境 ----------
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ---------- 1. 用户可改参数 ----------
MODEL_TYPE="FLUX-AFCache"   # 以后想换模型，只需改这一行
N_GPUS=4                    # 总 GPU 数
HEIGHT=1024
WIDTH=1024
INFERENCE_STEP=50
PROMPT="A cat standing on the grass."
RESULT_ROOT="./results"

# ---------- 2. 模型→配置映射 ----------
# 返回格式: SCRIPT MODEL_ID INFERENCE_STEP
get_model_config() {
    local mt=$1
    case "$mt" in
        FLUX-AFCache)
            echo "./afcache_flux_pipeline.py /home/lyb/FLUX.1-dev 50"
            ;;
        *)
            echo "Unsupported MODEL_TYPE: $mt" >&2
            return 1
            ;;
    esac
}

read -r SCRIPT MODEL_ID DEFAULT_STEP <<<"$(get_model_config "$MODEL_TYPE")"
INFERENCE_STEP=${INFERENCE_STEP:-$DEFAULT_STEP}   # 若外部没传，用默认值

# ---------- 3. 并行策略 ----------
# 对于基于 cache 的方法，pipefusion 并行度设为 1（与注释保持一致）
PIPEFUSION_DEG=1
ULYSSES_DEG=2
RING_DEG=2
(( PIPEFUSION_DEG * ULYSSES_DEG * RING_DEG == N_GPUS )) || {
    echo "ERROR: 并行度乘积 != N_GPUS" >&2
    exit 1
}

# ---------- 4. 运行时目录 ----------
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="${RESULT_ROOT}/${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# ---------- 5. 启动命令 ----------
echo "========== 启动信息 =========="
echo "MODEL_TYPE      : $MODEL_TYPE"
echo "SCRIPT          : $SCRIPT"
echo "MODEL_ID        : $MODEL_ID"
echo "INFERENCE_STEP  : $INFERENCE_STEP"
echo "N_GPUS          : $N_GPUS"
echo "并行策略        : pipefusion=$PIPEFUSION_DEG ulysses=$ULYSSES_DEG ring=$RING_DEG"
echo "结果目录        : $RUN_DIR"
echo "=============================="

torchrun \
--nproc_per_node="$N_GPUS" \
"$SCRIPT" \
--model "$MODEL_ID" \
--pipefusion_parallel_degree "$PIPEFUSION_DEG" \
--ulysses_degree "$ULYSSES_DEG" \
--ring_degree "$RING_DEG" \
--height "$HEIGHT" \
--width "$WIDTH" \
--no_use_resolution_binning \
--num_inference_steps "$INFERENCE_STEP" \
--warmup_steps 1 \
--prompt "$PROMPT" \