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

from typing import Any, Dict, Optional, Tuple, Union
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
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
            # 
            from xfuser.model_executor.models.transformers.transformer_flux import xFuserFluxTransformer2DWrapper
            
            # save original transformer
            original_transformer = pipe.transformer
            
            # apply wrapper
            pipe.transformer = xFuserFluxTransformer2DWrapper(original_transformer)
            
            # # check if wrapper has correct attributes
            # if not hasattr(pipe.transformer, 'stage_info'):
            #     class StageInfo:
            #         def __init__(self):
            #             self.after_flags = {"single_transformer_blocks": True, "transformer_blocks": True}
                
            #     pipe.transformer.stage_info = StageInfo()
            
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
    if input_config.output_type == "pil":
        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
        if pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = dp_group_index * dp_batch_size + i
                image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {i} saved to ./results/{image_name}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()
