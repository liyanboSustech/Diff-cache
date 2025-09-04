import torch
from typing import Optional, Tuple, Dict, Any
from xfuser.core.cache_manager import CacheEntry, get_cache_manager
from .fft_ops import fft_1d, ifft_1d, compress_frequency, decompress_frequency, analyze_frequency_characteristics


class FourierCacheEntry(CacheEntry):
    def __init__(self, 
                 cache_type: str = "fourier_cache",
                 num_cache_tensors: int = 1,
                 compression_ratio: float = 0.5,
                 preserve_dc: bool = True):
        super().__init__(cache_type, num_cache_tensors)
        self.compression_ratio = compression_ratio
        self.preserve_dc = preserve_dc
        self.frequency_masks = [None] * num_cache_tensors
        self.original_shapes = [None] * num_cache_tensors


class FourierCacheManager:
    supported_cache_type = ["fourier_cache", "fourier_predictive_cache"]
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        
    def register_fourier_cache_entry(self, 
                                     layer: Any, 
                                     layer_type: str = "attn",
                                     cache_type: str = "fourier_cache",
                                     compression_ratio: float = 0.5,
                                     preserve_dc: bool = True):
        if cache_type not in self.supported_cache_type:
            raise ValueError(
                f"Cache type: {cache_type} is not supported. Supported cache type: {self.supported_cache_type}"
            )
            
        # Register with the main cache manager
        self.cache_manager.register_cache_entry(layer, layer_type, cache_type)
        
        # Update to Fourier-specific cache entry
        cache_entry = self.cache_manager.cache[(layer_type, layer)]
        fourier_entry = FourierCacheEntry(
            cache_type=cache_type,
            num_cache_tensors=len(cache_entry.tensors),
            compression_ratio=compression_ratio,
            preserve_dc=preserve_dc
        )
        self.cache_manager.cache[(layer_type, layer)] = fourier_entry

    def update_and_get_fourier_cache(self,
                                     new_kv: torch.Tensor,
                                     layer: Any,
                                     slice_dim: int = 1,
                                     layer_type: str = "attn",
                                     compression_ratio: Optional[float] = None):
        cache_entry = self.cache_manager.cache[(layer_type, layer)]
        
        # Use provided compression ratio or default
        if compression_ratio is None:
            compression_ratio = cache_entry.compression_ratio
            
        # Apply Fourier compression
        compressed_kv, mask = compress_frequency(
            new_kv, 
            compression_ratio=compression_ratio, 
            dim=slice_dim,
            preserve_dc=cache_entry.preserve_dc
        )
        
        # Store compressed tensor and mask
        cache_entry.tensors[0] = compressed_kv
        cache_entry.frequency_masks[0] = mask
        cache_entry.original_shapes[0] = new_kv.shape
        
        return compressed_kv, mask

    def get_decompressed_cache(self,
                               layer: Any,
                               slice_dim: int = 1,
                               layer_type: str = "attn") -> torch.Tensor:
        cache_entry = self.cache_manager.cache[(layer_type, layer)]
        
        if cache_entry.tensors[0] is None:
            return None
            
        # Decompress using stored mask
        decompressed = decompress_frequency(
            cache_entry.tensors[0],
            cache_entry.frequency_masks[0],
            dim=slice_dim
        )
        
        return decompressed

    def analyze_cache_characteristics(self,
                                      layer: Any,
                                      layer_type: str = "attn") -> Dict[str, Any]:
        cache_entry = self.cache_manager.cache[(layer_type, layer)]
        
        if cache_entry.tensors[0] is None:
            return {}
            
        # Decompress to analyze original characteristics
        decompressed = self.get_decompressed_cache(layer, layer_type=layer_type)
        if decompressed is not None:
            return analyze_frequency_characteristics(decompressed, dim=1)
        return {}


# Global instance
_FOURIER_CACHE_MGR = FourierCacheManager()


def get_fourier_cache_manager():
    global _FOURIER_CACHE_MGR
    assert _FOURIER_CACHE_MGR is not None, "Fourier cache manager has not been initialized."
    return _FOURIER_CACHE_MGR