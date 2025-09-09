set -x

export PYTHONPATH=$PWD:$PYTHONPATH

export MODEL_TYPE="FLUX-TaylorSeer"

declare -A MODEL_CONFIGS=(
    ["FLUX-TaylorSeer"]="./xfusers_taylorseer_flux.py /home/lyb/FLUX.1-dev"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results

TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

N_GPUS=4
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 2 --ring_degree 2"

# 你想要的最大步数
MAX_STEPS=28

for (( STEP=1; STEP<=MAX_STEPS; STEP++ ))
do
    echo "========== Running with $STEP inference steps =========="

    torchrun --nproc_per_node=$N_GPUS $SCRIPT \
        --model $MODEL_ID \
        $PARALLEL_ARGS \
        $TASK_ARGS \
        --num_inference_steps $STEP \
        --warmup_steps 1 \
        --prompt "A golden retriever with fluffy fur standing on vibrant green grass in a sunny meadow, with wildflowers scattered around, soft morning light, highly detailed, photorealistic, 4k quality" \
        --output_type pil

done