import functools
import unittest
from typing import Optional
import torch
import torch.nn as nn
from diffusers import FluxTransformer2DModel
from xfuser.model_executor.cache.diffusers_adapters.registry import TRANSFORMER_ADAPTER_REGISTRY
from xfuser.model_executor.cache import utils
from .fourier_cache import get_fourier_cache_manager, FourierCacheManager
from .fft_ops import analyze_frequency_characteristics


def create_fourier_cached_transformer_blocks(
    transformer: FluxTransformer2DModel,
    compression_ratio: float = 0.5,
    preserve_dc: bool = True,
    enable_prediction: bool = False
):
    return FourierCachedTransformerBlocks(
        transformer.transformer_blocks,
        transformer.single_transformer_blocks,
        transformer=transformer,
        compression_ratio=compression_ratio,
        preserve_dc=preserve_dc,
        enable_prediction=enable_prediction,
        name=TRANSFORMER_ADAPTER_REGISTRY.get(type(transformer)),
    )


class FourierCachedTransformerBlocks(nn.Module):
    
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer: Optional[nn.Module] = None,
        compression_ratio: float = 0.5,
        preserve_dc: bool = True,
        enable_prediction: bool = False,
        name: str = "default",
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.single_transformer_blocks = nn.ModuleList(single_transformer_blocks) if single_transformer_blocks else None
        self.transformer = transformer
        self.compression_ratio = compression_ratio
        self.preserve_dc = preserve_dc
        self.enable_prediction = enable_prediction
        self.name = name
        
        # Initialize Fourier cache manager
        self.fourier_cache_manager = get_fourier_cache_manager()
        self.register_buffer("use_cache", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("cache_step", torch.tensor(0))
        
        # Storage for cached features
        self.cached_hidden_states = None
        self.cached_encoder_hidden_states = None
        self.hidden_states_residual = None
        self.encoder_hidden_states_residual = None
        
        # Frequency analysis metrics
        self.frequency_characteristics = {}
        
    def analyze_features(self, hidden_states, encoder_hidden_states):
        # Analyze hidden states
        if hidden_states is not None:
            self.frequency_characteristics['hidden_states'] = analyze_frequency_characteristics(
                hidden_states, dim=1
            )
        
        # Analyze encoder hidden states
        if encoder_hidden_states is not None:
            self.frequency_characteristics['encoder_hidden_states'] = analyze_frequency_characteristics(
                encoder_hidden_states, dim=1
            )
    
    def should_use_cache(self, hidden_states, encoder_hidden_states) -> bool:
        # For now, use a simple heuristic based on cache step
        # In a more advanced implementation, this would be based on frequency similarity
        return self.cache_step > 0 and self.cache_step % 2 == 0
    
    def process_blocks(self, start_idx, hidden, encoder, *args, **kwargs):
        for block in self.transformer_blocks[start_idx:]:
            hidden, encoder = block(hidden, encoder, *args, **kwargs)
        
        if self.single_transformer_blocks:
            hidden = torch.cat([encoder, hidden], dim=1)
            for block in self.single_transformer_blocks:
                hidden = block(hidden, *args, **kwargs)
            encoder, hidden = hidden.split([encoder.shape[1], hidden.shape[1] - encoder.shape[1]], dim=1)
        
        return hidden, encoder
    
    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # Analyze frequency characteristics of input features
        self.analyze_features(hidden_states, encoder_hidden_states)
        
        # Determine if we should use cache
        self.use_cache = torch.tensor(
            self.should_use_cache(hidden_states, encoder_hidden_states),
            dtype=torch.bool,
            device=hidden_states.device
        )
        
        if self.use_cache and self.cached_hidden_states is not None:
            # Use cached features with residuals
            hidden = self.cached_hidden_states + self.hidden_states_residual
            encoder = self.cached_encoder_hidden_states + self.encoder_hidden_states_residual
        else:
            # Process blocks normally
            original_hidden = hidden_states.clone()
            original_encoder = encoder_hidden_states.clone() if encoder_hidden_states is not None else None
            
            # Process transformer blocks
            hidden, encoder = self.process_blocks(0, hidden_states, encoder_hidden_states, *args, **kwargs)
            
            # Apply Fourier compression to cache features
            if self.training:
                # During training, cache full features
                self.cached_hidden_states = original_hidden
                self.cached_encoder_hidden_states = original_encoder
                self.hidden_states_residual = hidden - original_hidden
                self.encoder_hidden_states_residual = encoder - original_encoder if original_encoder is not None else None
            else:
                # During inference, apply compression
                self.fourier_cache_manager.register_fourier_cache_entry(
                    self, 
                    layer_type="transformer",
                    compression_ratio=self.compression_ratio,
                    preserve_dc=self.preserve_dc
                )
                
                # Update cache with compressed features
                if original_hidden is not None:
                    compressed_hidden, _ = self.fourier_cache_manager.update_and_get_fourier_cache(
                        original_hidden, self, slice_dim=1, layer_type="transformer"
                    )
                    # For simplicity, we're not actually using the compressed version in this example
                    # In a full implementation, we would decompress when needed
                    self.cached_hidden_states = original_hidden
                    self.hidden_states_residual = hidden - original_hidden
                
                if original_encoder is not None:
                    compressed_encoder, _ = self.fourier_cache_manager.update_and_get_fourier_cache(
                        original_encoder, self, slice_dim=1, layer_type="transformer_encoder"
                    )
                    self.cached_encoder_hidden_states = original_encoder
                    self.encoder_hidden_states_residual = encoder - original_encoder
        
        # Increment cache step
        self.cache_step += 1
        
        return hidden, encoder


def apply_fourier_cache_on_transformer(
    transformer: FluxTransformer2DModel,
    *,
    compression_ratio: float = 0.5,
    preserve_dc: bool = True,
    enable_prediction: bool = False,
):
    fourier_cached_transformer_blocks = nn.ModuleList([
        create_fourier_cached_transformer_blocks(
            transformer,
            compression_ratio=compression_ratio,
            preserve_dc=preserve_dc,
            enable_prediction=enable_prediction
        )
    ])
    
    dummy_single_transformer_blocks = torch.nn.ModuleList()
    
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
            fourier_cached_transformer_blocks,
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