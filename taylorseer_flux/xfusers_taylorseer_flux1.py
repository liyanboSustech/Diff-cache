import logging
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
)

from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers

from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

from forwards import taylorseer_flux_single_block_forward, taylorseer_flux_double_block_forward, taylorseer_flux_forward, taylorseer_xfuser_flux_forward

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    text_encoder_2 = T5EncoderModel.from_pretrained(engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
    
    # Create results directories
    timestep_results_dir = "./results/timesteps"
    os.makedirs(timestep_results_dir, exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )
    
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper

    # if is not distributed environment or tensor_parallel_degree is 1, ensure to use wrapper
    if not dist.is_initialized() or get_tensor_model_parallel_world_size() == 1:
        # check if transformer is already wrapped
        if not isinstance(pipe.transformer, xFuserFluxTransformer2DWrapper):
            from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
            
            # save original transformer
            original_transformer = pipe.transformer
            
            # apply wrapper
            pipe.transformer = xFuserFluxTransformer2DWrapper(original_transformer)
            
    pipe.transformer.__class__.num_steps = input_config.num_inference_steps
    pipe.transformer.__class__.forward = taylorseer_xfuser_flux_forward

    for double_transformer_block in pipe.transformer.transformer_blocks:
        double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward

    for single_transformer_block in pipe.transformer.single_transformer_blocks:
        single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward
    
    joint_attention_kwargs = {}

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=1)

    torch.cuda.reset_peak_memory_stats()
    
    # Fixed callback function
    def save_timestep_callback(pipeline, step, timestep, callback_kwargs):
        """
        Fixed callback that handles xFuser's distributed latents format
        """
        # Only save on the last pipeline stage and last data parallel group
        if not (is_pipeline_last_stage() and is_dp_last_group()):
            return callback_kwargs
            
        try:
            latents = callback_kwargs["latents"]
            
            # Handle sequence parallel gathering if needed
            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
                latents_list = []
                for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            :,
                            get_runtime_state().pp_patches_token_start_idx_local[pp_patch_idx]:
                            get_runtime_state().pp_patches_token_start_idx_local[pp_patch_idx + 1],
                            :,
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)
            
            # Process latents for VAE decode - bypass problematic _unpack_latents
            batch_size = latents.shape[0]
            h, w = input_config.height // 8, input_config.width // 8
            
            # Handle different latent formats from xFuser
            if len(latents.shape) == 3:  # [batch, seq_len, channels]
                seq_len, channels = latents.shape[1], latents.shape[2]
                
                # Case 1: Standard format [batch, h*w, 16]
                if seq_len == h * w and channels == 16:
                    processed_latents = latents.view(batch_size, h, w, 16).permute(0, 3, 1, 2)
                
                # Case 2: xFuser compressed format [batch, 4096, 64] -> need to reshape to [batch, 16384, 16]
                elif seq_len * channels == h * w * 16:
                    print(f"Step {step}: Reshaping xFuser format {latents.shape} to standard format")
                    # Reshape to [batch, h*w, 16] then to spatial format
                    latents_reshaped = latents.view(batch_size, h * w, 16)
                    processed_latents = latents_reshaped.view(batch_size, h, w, 16).permute(0, 3, 1, 2)
                
                # Case 3: Other formats - try to infer the correct reshaping
                elif seq_len * channels * batch_size == h * w * 16:
                    print(f"Step {step}: Attempting to reshape format {latents.shape}")
                    total_elements = latents.numel()
                    target_shape = (batch_size, 16, h, w)
                    if total_elements == batch_size * 16 * h * w:
                        processed_latents = latents.view(target_shape)
                    else:
                        print(f"Step {step}: Cannot reshape {latents.shape} with {total_elements} elements to target {target_shape}")
                        return callback_kwargs
                else:
                    print(f"Step {step}: Unexpected latent shape {latents.shape}, expected something compatible with [{batch_size}, {h*w}, 16]")
                    print(f"  seq_len * channels = {seq_len * channels}, expected h*w*16 = {h * w * 16}")
                    return callback_kwargs
                    
            elif len(latents.shape) == 4:  # Already in [batch, channels, height, width]
                if latents.shape[1] == 16 and latents.shape[2] == h and latents.shape[3] == w:
                    processed_latents = latents
                else:
                    print(f"Step {step}: Unexpected 4D latent shape {latents.shape}, expected [{batch_size}, 16, {h}, {w}]")
                    return callback_kwargs
                    
            elif len(latents.shape) == 2:  # Flattened format [batch, total_elements]
                total_elements = latents.shape[1]
                if total_elements == h * w * 16:
                    print(f"Step {step}: Reshaping flattened format {latents.shape}")
                    processed_latents = latents.view(batch_size, h, w, 16).permute(0, 3, 1, 2)
                else:
                    print(f"Step {step}: Unexpected flattened shape {latents.shape}")
                    return callback_kwargs
                    
            else:
                print(f"Step {step}: Unsupported latent shape {latents.shape}")
                return callback_kwargs
            
            # Apply VAE scaling
            processed_latents = (processed_latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            
            # Decode to image
            with torch.no_grad():
                image = pipeline.vae.decode(processed_latents, return_dict=False)[0]
                image = pipeline.image_processor.postprocess(image, output_type="pil")
                
                # Handle batch of images
                for i, img in enumerate(image):
                    timestep_filename = f"timestep_{step:03d}_t_{timestep:.0f}_batch_{i}.png"
                    img.save(os.path.join(timestep_results_dir, timestep_filename))
                    print(f"Saved timestep {step} image: {timestep_filename}")
        
        except Exception as e:
            print(f"Error saving timestep {step}: {e}")
            print(f"Latents shape: {latents.shape if 'latents' in locals() else 'undefined'}")
            import traceback
            traceback.print_exc()
        
        return callback_kwargs
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    
    output = pipe(
        height=input_config.height,
        width=input_config.width,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=256,
        guidance_scale=0.0,
        joint_attention_kwargs=joint_attention_kwargs,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        callback_on_step_end=save_timestep_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) * 1e-3
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    
    if input_config.output_type == "pil" and output is not None:
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_timestep_{input_config.num_inference_steps}_{parallel_info}_image_rank{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()