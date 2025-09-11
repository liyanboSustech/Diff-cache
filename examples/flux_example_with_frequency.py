import logging
import time
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser.model_executor.pipelines import xFuserFluxPipeline
from xfuser.config import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
)
from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer
from fourier.latent_frequency_cache import get_latent_frequency_cache


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments with Frequency Analysis")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    
    # Initialize frequency analysis cache
    frequency_cache = get_latent_frequency_cache(
        similarity_threshold=0.85,  # Adjust based on your needs
        compression_ratio=0.3,       # Frequency compression ratio
        cache_lookback=5,           # Look back 5 steps for similar features
        enable_feature_reuse=True   # Enable feature reuse
    )
    
    text_encoder_2 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    cache_args = {
            "use_teacache": engine_args.use_teacache,
            "use_fbcache": engine_args.use_fbcache,
            "rel_l1_thresh": 0.12,
            "return_hidden_states_first": False,
            "num_steps": input_config.num_inference_steps,
        }

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    # Add frequency analysis callback
    def frequency_analysis_callback(pipe, step, t, callback_kwargs):
        """Callback for frequency analysis during generation"""
        latents = callback_kwargs.get('latents')
        if latents is not None and get_world_group().rank == 0:  # Only on rank 0 to avoid duplicate work
            # Process with frequency analysis
            processed_latents, analysis_info = frequency_cache.process_timestep_latents(
                latents, step
            )
            
            # Update latents with processed version
            callback_kwargs['latents'] = processed_latents
            
            # Print analysis info every few steps
            if step % 5 == 0:
                reuse_status = "✓ REUSED" if analysis_info.get('feature_reused') else "Computed"
                efficiency = analysis_info.get('reuse_efficiency', 0)
                print(f"[Frequency Analysis] Step {step:2d}: {reuse_status} | Efficiency: {efficiency:.1%}")
        
        return callback_kwargs

    # Run pipeline with frequency analysis callback
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=input_config.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        callback_on_step_end=frequency_analysis_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    # Get performance statistics
    perf_stats = frequency_cache.get_performance_stats()
    
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
                # Add frequency analysis info to filename
                freq_suffix = f"freq_reuse{perf_stats['reuse_count']}" if perf_stats['reuse_count'] > 0 else "freq_normal"
                image_name = f"flux_result_{parallel_info}_{image_rank}_{freq_suffix}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
        print(f"[Frequency Analysis Performance]")
        print(f"  Total Steps: {perf_stats['total_steps']}")
        print(f"  Features Reused: {perf_stats['reuse_count']}")
        print(f"  Reuse Efficiency: {perf_stats['reuse_efficiency']:.1%}")
        print(f"  Computation Saved: {perf_stats['computation_saved_ratio']:.1%}")
        print(f"  Cache Size: {perf_stats['cache_size']}")
    
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()