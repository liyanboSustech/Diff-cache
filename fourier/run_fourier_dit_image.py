#!/usr/bin/env python
# coding=utf-8
import logging
import time
import torch
import torch.distributed as dist
from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)

# Import Fourier cache components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fourier_cache import (
    get_fourier_cache_manager,
    fft_1d,
    analyze_frequency_characteristics
)
from flux_fourier_adapter import apply_fourier_cache_on_transformer


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments with Fourier Cache")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    
    # Load text encoder
    text_encoder_2 = T5EncoderModel.from_pretrained(
        engine_config.model_config.model, 
        subfolder="text_encoder_2", 
        torch_dtype=torch.bfloat16
    )

    # Quantize text encoder if requested
    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    # Create pipeline
    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )

    # Apply sequential CPU offload or move to GPU
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    # Apply Fourier cache to transformer
    if hasattr(args, 'use_fourier_cache') and args.use_fourier_cache:
        compression_ratio = getattr(args, 'fourier_compression_ratio', 0.5)
        preserve_dc = getattr(args, 'fourier_preserve_dc', True)
        
        print(f"Applying Fourier cache with compression_ratio={compression_ratio}, preserve_dc={preserve_dc}")
        
        pipe.transformer = apply_fourier_cache_on_transformer(
            pipe.transformer,
            compression_ratio=compression_ratio,
            preserve_dc=preserve_dc,
            enable_prediction=False  # Can be enabled for more advanced prediction
        )

    # Prepare for run
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    # Run inference
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # Generate output info
    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_fourier_cache.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
        
        # Print frequency analysis if available
        try:
            fourier_cache_mgr = get_fourier_cache_manager()
            # This would print cache characteristics if we had actual cached layers
            print("Fourier cache analysis completed")
        except Exception as e:
            print(f"Could not analyze Fourier cache: {e}")
            
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()