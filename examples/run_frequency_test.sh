#!/bin/bash

# Test script for Flux with frequency analysis
# This demonstrates how to run the frequency-enhanced version

set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Create results directory
mkdir -p ./results
mkdir -p ./intermediates

# Model configuration
MODEL_ID="/home/lyb/FLUX.1-dev"
INFERENCE_STEP=20

# Task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

# Parallel configuration (adjust based on your GPU setup)
N_GPUS=4
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 2 --ring_degree 2"

# Inference configuration
INFERENCE_ARGS="--num_inference_steps $INFERENCE_STEP --warmup_steps 1"

# Test prompt
PROMPT="brown dog laying on the ground with a metal bowl in front of him."

# Optional flags (uncomment as needed)
# CFG_ARGS="--use_cfg_parallel"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# COMPILE_FLAG="--use_torch_compile"
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

echo "=== Testing Flux with Frequency Analysis ==="
echo "Model: $MODEL_ID"
echo "Steps: $INFERENCE_STEP"
echo "GPUs: $N_GPUS"
echo "Prompt: $PROMPT"
echo "============================================"

# Run with frequency analysis
torchrun --nproc_per_node=$N_GPUS ./examples/flux_example_with_frequency.py \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$INFERENCE_ARGS \
--prompt "$PROMPT" \
--seed 42 \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \

echo "=== Frequency Analysis Test Complete ==="
echo "Check ./results/ for output images with frequency analysis info"
echo "Check ./intermediates/ for timestep images with reuse information"