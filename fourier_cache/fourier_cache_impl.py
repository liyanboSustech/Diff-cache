import torch
from torch import nn
from xfuser.model_executor.cache import utils
from .simple_fft import fft_compress, fft_decompress, compute_frequency_energy


class FourierCachedTransformerBlocks(utils.CachedTransformerBlocks):
    
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        rel_l1_thresh=0.6,
        return_hidden_states_first=True,
        num_steps=-1,
        name="default",
        compression_ratio=0.5,
        callbacks=None,
    ):
        super().__init__(
            transformer_blocks,
            single_transformer_blocks=single_transformer_blocks,
            transformer=transformer,
            rel_l1_thresh=rel_l1_thresh,
            return_hidden_states_first=return_hidden_states_first,
            num_steps=num_steps,
            name=name,
            callbacks=callbacks,
        )
        self.compression_ratio = compression_ratio
        # Store compressed versions of residuals
        self.compressed_hidden_states_residual = None
        self.compressed_encoder_hidden_states_residual = None
        self.compression_mask = None
        
    def get_start_idx(self) -> int:
        return 0
        
    def are_two_tensor_similar(self, t1: torch.Tensor, t2: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.l1_distance(t1, t2) < threshold
        
    def get_modulated_inputs(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # For Fourier cache, we use the inputs directly
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        
        # Check if we have previous compressed residuals
        prev_compressed_hidden = self.compressed_hidden_states_residual
        prev_compression_mask = self.compression_mask
        
        # Store current inputs for next iteration
        self.cache_context.original_hidden_states = original_hidden_states
        self.cache_context.original_encoder_hidden_states = original_encoder_hidden_states
        
        return (
            original_hidden_states,  # current inputs
            prev_compressed_hidden,   # previous compressed data
            original_hidden_states,   # original hidden states
            original_encoder_hidden_states  # original encoder hidden states
        )
        
    def process_blocks(self, start_idx: int, hidden: torch.Tensor, encoder: torch.Tensor, *args, **kwargs):
        original_hidden = hidden.clone()
        original_encoder = encoder.clone() if encoder is not None else None
        
        # Process all transformer blocks
        for block in self.transformer_blocks[start_idx:]:
            hidden, encoder = block(hidden, encoder, *args, **kwargs)
            hidden, encoder = (hidden, encoder) if self.return_hidden_states_first else (encoder, hidden)

        # Process single transformer blocks if they exist
        if self.single_transformer_blocks:
            hidden = torch.cat([encoder, hidden], dim=1)
            for block in self.single_transformer_blocks:
                hidden = block(hidden, *args, **kwargs)
            encoder, hidden = hidden.split([encoder.shape[1], hidden.shape[1] - encoder.shape[1]], dim=1)

        # Compute residuals
        hidden_residual = hidden - original_hidden
        encoder_residual = encoder - original_encoder if original_encoder is not None else None
        
        # Apply Fourier compression to residuals
        if hidden_residual is not None:
            compressed_hidden, mask, _ = fft_compress(hidden_residual, self.compression_ratio)
            self.compressed_hidden_states_residual = compressed_hidden
            self.compression_mask = mask
            
        if encoder_residual is not None and encoder_residual.numel() > 0:
            compressed_encoder, _, _ = fft_compress(encoder_residual, self.compression_ratio)
            self.compressed_encoder_hidden_states_residual = compressed_encoder
            
        # Store uncompressed residuals for immediate use
        self.cache_context.hidden_states_residual = hidden_residual
        self.cache_context.encoder_hidden_states_residual = encoder_residual
        
        return hidden, encoder
        
    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        self.callback_handler.trigger_event("on_forward_begin", self)

        # Get modulated inputs
        current_inputs, prev_compressed_data, orig_hidden, orig_encoder = \
            self.get_modulated_inputs(hidden_states, encoder_hidden_states, *args, **kwargs)

        # Decide whether to use cache based on similarity
        # For simplicity, we'll use a step-based approach
        use_cache = self.cnt > 0 and (self.cnt % 2) == 0  # Use cache every 2 steps
        self.use_cache = torch.tensor(use_cache, dtype=torch.bool, device=hidden_states.device)

        self.callback_handler.trigger_event("on_forward_remaining_begin", self)
        
        if self.use_cache and self.compressed_hidden_states_residual is not None:
            # Decompress cached residuals
            hidden_residual = fft_decompress(
                self.compressed_hidden_states_residual, 
                self.compression_mask, 
                self.cache_context.original_hidden_states.shape
            )
            hidden = orig_hidden + hidden_residual
            
            if self.compressed_encoder_hidden_states_residual is not None:
                # Note: This is simplified - in practice we'd need to store the encoder shape
                encoder_residual = fft_decompress(
                    self.compressed_encoder_hidden_states_residual,
                    self.compression_mask,  # Reuse mask for simplicity
                    self.cache_context.original_encoder_hidden_states.shape
                )
                encoder = orig_encoder + encoder_residual
            else:
                encoder = orig_encoder
        else:
            # Process blocks normally
            hidden, encoder = self.process_blocks(self.get_start_idx(), orig_hidden, orig_encoder, *args, **kwargs)

        self.callback_handler.trigger_event("on_forward_end", self)
        self.cnt += 1
        
        return ((hidden, encoder) if self.return_hidden_states_first else (encoder, hidden))