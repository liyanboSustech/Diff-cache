import torch
from typing import Optional, Dict, Any, Tuple
from .latent_frequency_analyzer import LatentFrequencyAnalyzer
from .fourier_cache import get_fourier_cache_manager


class LatentFrequencyCache:
    """
    Integrates frequency-domain analysis with the existing fourier cache system
    for efficient feature reuse in diffusion models.
    """
    
    def __init__(self, 
                 compression_ratio: float = 0.3,
                 similarity_threshold: float = 0.85,
                 cache_lookback: int = 5,
                 enable_feature_reuse: bool = True):
        
        self.frequency_analyzer = LatentFrequencyAnalyzer(
            compression_ratio=compression_ratio,
            similarity_threshold=similarity_threshold
        )
        self.fourier_cache_manager = get_fourier_cache_manager()
        self.cache_lookback = cache_lookback
        self.enable_feature_reuse = enable_feature_reuse
        
        # Statistics tracking
        self.reuse_count = 0
        self.total_steps = 0
        self.computation_saved = 0
        
    def process_timestep_latents(self, 
                                latents: torch.Tensor, 
                                step: int,
                                force_compute: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process latents at a given timestep with frequency-domain feature reuse.
        
        Args:
            latents: Current latent representation [B, L, C]
            step: Current timestep
            force_compute: Force computation without reuse
            
        Returns:
            Tuple of (processed_latents, analysis_info)
        """
        self.total_steps += 1
        analysis_info = {}
        
        if not self.enable_feature_reuse or force_compute:
            # Direct computation without reuse
            analysis_result = self.frequency_analyzer.analyze_latent_frequency(latents, step)
            analysis_info.update(analysis_result)
            analysis_info['feature_reused'] = False
            return latents, analysis_info
        
        # Check for reusable features
        reusable_features = self.frequency_analyzer.get_reusable_features(
            latents, self.similarity_threshold
        )
        
        if reusable_features is not None:
            # Feature reuse: blend current latents with cached features
            blended_latents = self._blend_with_cached_features(latents, reusable_features)
            
            self.reuse_count += 1
            # Estimate computation saved (rough approximation)
            self.computation_saved += 0.7  # Assume 70% computation saved per reuse
            
            analysis_info = {
                'step': step,
                'feature_reused': True,
                'reuse_efficiency': self.reuse_count / self.total_steps,
                'computation_saved_ratio': self.computation_saved / self.total_steps
            }
            
            return blended_latents, analysis_info
        
        # No reusable features found, analyze and cache current
        analysis_result = self.frequency_analyzer.analyze_latent_frequency(latents, step)
        analysis_info.update(analysis_result)
        analysis_info['feature_reused'] = False
        
        return latents, analysis_info
    
    def _blend_with_cached_features(self, 
                                  current_latents: torch.Tensor,
                                  cached_latents: torch.Tensor,
                                  blend_ratio: float = 0.3) -> torch.Tensor:
        """
        Blend current latents with cached features for smooth feature reuse.
        """
        # Adaptive blend ratio based on timestep (earlier steps = more reuse)
        # This is a simple heuristic - could be made more sophisticated
        adaptive_blend = blend_ratio * (1.0 - min(self.total_steps / 20, 0.5))
        
        blended = (1 - adaptive_blend) * current_latents + adaptive_blend * cached_latents
        
        return blended
    
    def integrate_with_fourier_cache(self, 
                                    layer: Any,
                                    layer_type: str = "attn") -> Optional[torch.Tensor]:
        """
        Integrate with the existing fourier cache system for additional optimization.
        """
        try:
            # Try to get decompressed cache from fourier cache
            cached_tensor = self.fourier_cache_manager.get_decompressed_cache(
                layer, layer_type=layer_type
            )
            
            if cached_tensor is not None:
                # Convert to latent format if needed
                if cached_tensor.dim() == 3:  # Already in latent format [B, L, C]
                    return cached_tensor
                elif cached_tensor.dim() == 4:  # Spatial format [B, C, H, W]
                    B, C, H, W = cached_tensor.shape
                    L = H * W
                    return cached_tensor.reshape(B, C, L).permute(0, 2, 1)
                    
        except Exception:
            # Silently handle integration failures
            pass
            
        return None
    
    def get_similarity_analysis(self, latents1: torch.Tensor, 
                               latents2: torch.Tensor) -> Dict[str, float]:
        """
        Get detailed similarity analysis between two latent representations.
        """
        similarity = self.frequency_analyzer.compute_frequency_similarity(latents1, latents2)
        
        # Get frequency signatures for both
        spatial1 = self.frequency_analyzer._reshape_latent_to_spatial(latents1)
        spatial2 = self.frequency_analyzer._reshape_latent_to_spatial(latents2)
        
        fft1 = self.frequency_analyzer._compute_2d_fft(spatial1)
        fft2 = self.frequency_analyzer._compute_2d_fft(spatial2)
        
        sig1 = self.frequency_analyzer._compute_frequency_signature(fft1)
        sig2 = self.frequency_analyzer._compute_frequency_signature(fft2)
        
        return {
            'overall_similarity': similarity,
            'low_freq_diff': abs(float(sig1['low_freq_ratio'] - sig2['low_freq_ratio'])),
            'med_freq_diff': abs(float(sig1['med_freq_ratio'] - sig2['med_freq_ratio'])),
            'high_freq_diff': abs(float(sig1['high_freq_ratio'] - sig2['high_freq_ratio'])),
            'magnitude_diff': abs(float(sig1['magnitude_mean'] - sig2['magnitude_mean'])),
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the frequency cache."""
        efficiency = self.reuse_count / max(self.total_steps, 1)
        saved_ratio = self.computation_saved / max(self.total_steps, 1)
        
        return {
            'total_steps': self.total_steps,
            'reuse_count': self.reuse_count,
            'reuse_efficiency': efficiency,
            'computation_saved_ratio': saved_ratio,
            'cache_size': len(self.frequency_analyzer.cached_features),
            'similarity_threshold': self.similarity_threshold,
            'compression_ratio': self.compression_ratio,
        }
    
    def clear_cache(self):
        """Clear both frequency and fourier caches."""
        self.frequency_analyzer.clear_cache()
        self.reuse_count = 0
        self.total_steps = 0
        self.computation_saved = 0
    
    @property
    def similarity_threshold(self) -> float:
        return self.frequency_analyzer.similarity_threshold
    
    @property
    def compression_ratio(self) -> float:
        return self.frequency_analyzer.compression_ratio
    
    def update_parameters(self, 
                         similarity_threshold: Optional[float] = None,
                         compression_ratio: Optional[float] = None):
        """Update cache parameters."""
        if similarity_threshold is not None:
            self.frequency_analyzer.similarity_threshold = similarity_threshold
        if compression_ratio is not None:
            self.frequency_analyzer.compression_ratio = compression_ratio


# Global instance
_latent_frequency_cache = None


def get_latent_frequency_cache(**kwargs) -> LatentFrequencyCache:
    """Get or create the global latent frequency cache instance."""
    global _latent_frequency_cache
    
    if _latent_frequency_cache is None:
        _latent_frequency_cache = LatentFrequencyCache(**kwargs)
    
    return _latent_frequency_cache