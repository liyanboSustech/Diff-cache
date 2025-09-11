import functools
import unittest
from torch import nn
from diffusers import FluxTransformer2DModel
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY
from .fourier_cache_impl import FourierCachedTransformerBlocks


def create_fourier_cached_transformer_blocks(
    use_cache, 
    transformer, 
    rel_l1_thresh, 
    return_hidden_states_first, 
    num_steps,
    compression_ratio=0.5
):
    if use_cache != "Fourier":
        raise ValueError(f"Unsupported use_cache value: {use_cache}")
        
    return FourierCachedTransformerBlocks(
        transformer.transformer_blocks,
        transformer.single_transformer_blocks,
        transformer=transformer,
        rel_l1_thresh=rel_l1_thresh,
        return_hidden_states_first=return_hidden_states_first,
        num_steps=num_steps,
        name=TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer)),
        compression_ratio=compression_ratio,
    )


def apply_fourier_cache_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    rel_l1_thresh=0.12,
    return_hidden_states_first=False,
    num_steps=8,
    compression_ratio=0.5,
):
    cached_transformer_blocks = nn.ModuleList([
        create_fourier_cached_transformer_blocks(
            "Fourier", 
            transformer, 
            rel_l1_thresh, 
            return_hidden_states_first, 
            num_steps,
            compression_ratio
        )
    ])

    dummy_single_transformer_blocks = nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
        ), unittest.mock.patch.object(
            self,
            "single_transformer_blocks",
            dummy_single_transformer_blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    return transformer