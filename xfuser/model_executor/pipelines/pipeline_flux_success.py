# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import torch
import torch.distributed
from diffusers import FluxPipeline
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift

from xfuser.config import EngineConfig, InputConfig
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_runtime_state,
    get_pp_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    is_dp_last_group,
    get_world_group,
    get_vae_parallel_group,
    get_dit_world_size,
)
from xfuser.core.distributed.group_coordinator import GroupCoordinator
from .base_pipeline import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


@xFuserPipelineWrapperRegister.register(FluxPipeline)
class xFuserFluxPipeline(xFuserPipelineBaseWrapper):

    # ========================= 新增：每步保存图像 =========================
    def _save_timestep_image(self, latents: torch.Tensor, step: int,
                         output_dir: str = "./intermediates"):
        from xfuser.core.distributed import (
            get_world_group, get_sp_group, get_sequence_parallel_world_size,
            is_pipeline_last_stage,
        )
        import torch, os

        # 1. 所有 SP rank 都必须到场！
        print(f"[DEBUG] entry step={step}, world_rank={get_world_group().rank}, "
            f"sp_rank={get_sp_group().rank_in_group}", flush=True)

        # 2. 收集完整 latent
        if get_sequence_parallel_world_size() > 1:
            latents = get_sp_group().all_gather(latents, dim=-2)

        # 仅 rank0 保存
        if get_world_group().rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        
            latents = latents.contiguous()
            B, L, C = latents.shape               # [1, 4096, 64]
            h = w = int(np.sqrt(L)) * 2           # 128
            print(f"[DEBUG] step={step}, latents.shape={latents.shape}, h=w={h}, C={C}")
            # 正确 reshape
            latents = latents.view(B, h//2, w//2, C//4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C//4, h, w)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            with torch.no_grad():
                image = self.vae.decode(latents.to(self.vae.device), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type="pil")[0]
            path = os.path.join(output_dir, f"step_{step:03d}.png")
            image.save(path)
            print(f"[DEBUG] step_{step:03d}.png saved", flush=True)

        # 同步
        torch.distributed.barrier(group=get_world_group().device_group)
    # ========================= 原有 from_pretrained =========================
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict = {},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = FluxPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)

    # ========================= 原有 prepare_run =========================
    def prepare_run(
        self,
        input_config: InputConfig,
        steps: int = 3,
        sync_steps: int = 1,
    ):
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        warmup_steps = get_runtime_state().runtime_config.warmup_steps
        get_runtime_state().runtime_config.warmup_steps = sync_steps
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=steps,
            max_sequence_length=input_config.max_sequence_length,
            generator=torch.Generator(device="cuda").manual_seed(42),
            output_type=input_config.output_type,
        )
        get_runtime_state().runtime_config.warmup_steps = warmup_steps

    # ========================= 原有属性 =========================
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # ========================= 原有 __call__ =========================
    @torch.no_grad()
    @xFuserPipelineBaseWrapper.check_model_parallel_state(cfg_parallel_available=False)
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > num_pipeline_warmup_steps
            ):
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    timesteps=timesteps[:num_pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                latents = self._async_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    timesteps=timesteps[num_pipeline_warmup_steps:],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_image_ids,
                    guidance=guidance,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    sync_only=True,
                )

        image = None

        def process_latents(latents):
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            return latents

        if not output_type == "latent":
            if get_runtime_state().runtime_config.use_parallel_vae and get_runtime_state().parallel_config.vae_parallel_size > 0:
                latents = self.gather_latents_for_vae(latents)
                if latents is not None:
                    latents = process_latents(latents)
                self.send_to_vae_decode(latents)
            else:
                if get_runtime_state().runtime_config.use_parallel_vae:
                    latents = self.gather_broadcast_latents(latents)
                    latents = process_latents(latents)
                    image = self.vae.decode(latents, return_dict=False)[0]
                else:
                    if is_dp_last_group():
                        latents = process_latents(latents)
                        image = self.vae.decode(latents, return_dict=False)[0]

        if self.is_dp_last_group():
            if output_type == "latent":
                image = latents
            elif image is not None:
                image = self.image_processor.postprocess(image, output_type=output_type)
            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return FluxPipelineOutput(images=image)
        else:
            return None

    # ========================= 原有 _init_sync_pipeline =========================
    def _init_sync_pipeline(
        self, latents: torch.Tensor, latent_image_ids: torch.Tensor,
        prompt_embeds: torch.Tensor, text_ids: torch.Tensor
    ):
        get_runtime_state().set_patched_mode(patch_mode=False)

        latents_list = [
            latents[:, start_idx:end_idx, :]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        latents = torch.cat(latents_list, dim=-2)
        latent_image_ids_list = [
            latent_image_ids[start_idx:end_idx]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        ]
        latent_image_ids = torch.cat(latent_image_ids_list, dim=-2)

        if get_runtime_state().split_text_embed_in_sp:
            if prompt_embeds.shape[-2] % get_sequence_parallel_world_size() == 0:
                prompt_embeds = torch.chunk(prompt_embeds, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            else:
                get_runtime_state().split_text_embed_in_sp = False

        if get_runtime_state().split_text_embed_in_sp:
            if text_ids.shape[-2] % get_sequence_parallel_world_size() == 0:
                text_ids = torch.chunk(text_ids, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            else:
                get_runtime_state().split_text_embed_in_sp = False

        return latents, latent_image_ids, prompt_embeds, text_ids

    # ========================= 原有 _sync_pipeline =========================
    def _sync_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance,
        timesteps: List[int],
        num_warmup_steps: int,
        progress_bar,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        sync_only: bool = False,
    ):
        latents, latent_image_ids, prompt_embeds, text_ids = self._init_sync_pipeline(latents, latent_image_ids, prompt_embeds, text_ids)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            if is_pipeline_last_stage():
                last_timestep_latents = latents

            if get_pipeline_parallel_world_size() == 1:
                pass
            elif is_pipeline_first_stage() and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()
                if not is_pipeline_first_stage():
                    encoder_hidden_state = get_pp_group().pipeline_recv(
                        0, "encoder_hidden_state"
                    )

            latents, encoder_hidden_state = self._backbone_forward(
                latents=latents,
                encoder_hidden_states=(
                    prompt_embeds if is_pipeline_first_stage() else encoder_hidden_state
                ),
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                latent_image_ids=latent_image_ids,
                guidance=guidance,
                t=t,
            )

            if is_pipeline_last_stage():
                latents_dtype = latents.dtype
                latents = self._scheduler_step(latents, last_timestep_latents, t)

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # >>>>>>>>>> 新增：每步保存图像 <<<<<<<<<<
                self._save_timestep_image(latents, i)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

            if sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1:
                pass
            elif get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)
                if not is_pipeline_last_stage():
                    get_pp_group().pipeline_send(
                        encoder_hidden_state, name="encoder_hidden_state"
                    )

        if (
            sync_only
            and get_sequence_parallel_world_size() > 1
            and is_pipeline_last_stage()
        ):
            sp_degree = get_sequence_parallel_world_size()
            sp_latents_list = get_sp_group().all_gather(latents, separate_tensors=True)
            latents_list = []
            for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                latents_list += [
                    sp_latents_list[sp_patch_idx][
                        :,
                        get_runtime_state()
                        .pp_patches_token_start_idx_local[pp_patch_idx] : get_runtime_state()
                        .pp_patches_token_start_idx_local[pp_patch_idx + 1],
                        :,
                    ]
                    for sp_patch_idx in range(sp_degree)
                ]
            latents = torch.cat(latents_list, dim=-2)

        return latents

    # ========================= 原有 _async_pipeline =========================
    def _async_pipeline(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance,
        timesteps: List[int],
        num_warmup_steps: int,
        progress_bar,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = get_runtime_state().num_pipeline_patch
        num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        patch_latents, patch_latent_image_ids = self._init_async_pipeline(
            num_timesteps=len(timesteps),
            latents=latents,
            num_pipeline_warmup_steps=num_pipeline_warmup_steps,
            latent_image_ids=latent_image_ids,
        )
        last_patch_latents = (
            [None for _ in range(num_pipeline_patch)]
            if (is_pipeline_last_stage())
            else None
        )

        first_async_recv = True
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            for patch_idx in range(num_pipeline_patch):
                if is_pipeline_last_stage():
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if first_async_recv:
                        if not is_pipeline_first_stage() and patch_idx == 0:
                            get_pp_group().recv_next()
                        get_pp_group().recv_next()
                        first_async_recv = False

                    if not is_pipeline_first_stage() and patch_idx == 0:
                        last_encoder_hidden_states = (
                            get_pp_group().get_pipeline_recv_data(
                                idx=patch_idx, name="encoder_hidden_states"
                            )
                        )
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(
                        idx=patch_idx
                    )

                patch_latents[patch_idx], next_encoder_hidden_states = (
                    self._backbone_forward(
                        latents=patch_latents[patch_idx],
                        encoder_hidden_states=(
                            prompt_embeds
                            if is_pipeline_first_stage()
                            else last_encoder_hidden_states
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=patch_latent_image_ids[patch_idx],
                        guidance=guidance,
                        t=t,
                    )
                )
                if is_pipeline_last_stage():
                    latents_dtype = patch_latents[patch_idx].dtype
                    patch_latents[patch_idx] = self._scheduler_step(
                        patch_latents[patch_idx],
                        last_patch_latents[patch_idx],
                        t,
                    )

                    if patch_latents[patch_idx].dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            patch_latents[patch_idx] = patch_latents[patch_idx].to(latents_dtype)

                    # >>>>>>>>>> 新增：每步保存图像（仅 rank0） <<<<<<<<<<
                    # 先合并 patch 再保存
                    full_latents = torch.cat(patch_latents, dim=-2)
                    self._save_timestep_image(full_latents, i)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop(
                            "prompt_embeds", prompt_embeds
                        )

                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(
                            patch_latents[patch_idx], segment_idx=patch_idx
                        )
                else:
                    if patch_idx == 0:
                        get_pp_group().pipeline_isend(
                            next_encoder_hidden_states, name="encoder_hidden_states"
                        )
                    get_pp_group().pipeline_isend(
                        patch_latents[patch_idx], segment_idx=patch_idx
                    )

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    elif is_pipeline_first_stage():
                        get_pp_group().recv_next()
                    else:
                        if patch_idx == num_pipeline_patch - 1:
                            get_pp_group().recv_next()

                get_runtime_state().next_patch()

            if i == len(timesteps) - 1 or (
                (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
                and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        latents = None
        if is_pipeline_last_stage():
            latents = torch.cat(patch_latents, dim=-2)
            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = get_sp_group().all_gather(
                    latents, separate_tensors=True
                )
                latents_list = []
                for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
                    latents_list += [
                        sp_latents_list[sp_patch_idx][
                            ...,
                            get_runtime_state()
                            .pp_patches_token_start_idx_local[
                                pp_patch_idx
                            ] : get_runtime_state()
                            .pp_patches_token_start_idx_local[pp_patch_idx + 1],
                            :,
                        ]
                        for sp_patch_idx in range(sp_degree)
                    ]
                latents = torch.cat(latents_list, dim=-2)
        return latents

    # ========================= 原有 _init_async_pipeline =========================
    def _init_async_pipeline(
        self,
        num_timesteps: int,
        latents: torch.Tensor,
        num_pipeline_warmup_steps: int,
        latent_image_ids: torch.Tensor,
    ):
        get_runtime_state().set_patched_mode(patch_mode=True)

        if is_pipeline_first_stage():
            latents = (
                get_pp_group().pipeline_recv()
                if num_pipeline_warmup_steps > 0
                else latents
            )
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
        elif is_pipeline_last_stage():
            patch_latents = list(
                latents.split(get_runtime_state().pp_patches_token_num, dim=-2)
            )
        else:
            patch_latents = [
                None for _ in range(get_runtime_state().num_pipeline_patch)
            ]

        patch_latent_image_ids = list(
            latent_image_ids[start_idx:end_idx]
            for start_idx, end_idx in get_runtime_state().pp_patches_token_start_end_idx_global
        )

        recv_timesteps = (
            num_timesteps - 1 if is_pipeline_first_stage() else num_timesteps
        )

        if is_pipeline_first_stage():
            for _ in range(recv_timesteps):
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_task(patch_idx)
        else:
            for _ in range(recv_timesteps):
                get_pp_group().add_pipeline_recv_task(0, "encoder_hidden_states")
                for patch_idx in range(get_runtime_state().num_pipeline_patch):
                    get_pp_group().add_pipeline_recv_task(patch_idx)

        return patch_latents, patch_latent_image_ids

    # ========================= 原有 _backbone_forward =========================
    def _backbone_forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids,
        latent_image_ids,
        guidance,
        t: Union[float, torch.Tensor],
    ):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        ret = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        if self.engine_config.parallel_config.dit_parallel_size > 1:
            noise_pred, encoder_hidden_states = ret
        else:
            noise_pred, encoder_hidden_states = ret, None
        return noise_pred, encoder_hidden_states

    # ========================= 原有 _scheduler_step =========================
    def _scheduler_step(
        self,
        noise_pred: torch.Tensor,
        latents: torch.Tensor,
        t: Union[float, torch.Tensor],
    ):
        return self.scheduler.step(
            noise_pred,
            t,
            latents,
            return_dict=False,
        )[0]