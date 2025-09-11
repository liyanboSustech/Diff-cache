"""
Example integration of latent frequency analysis with Flux pipeline.
This shows how to modify the pipeline to use frequency-domain feature reuse.
"""

import torch
from typing import Optional, Dict, Any
from fourier.latent_frequency_cache import get_latent_frequency_cache


class FluxPipelineWithFrequencyCache:
    """
    Wrapper for Flux pipeline that adds frequency-domain feature reuse.
    """
    
    def __init__(self, original_pipeline, **frequency_cache_kwargs):
        self.pipeline = original_pipeline
        self.frequency_cache = get_latent_frequency_cache(**frequency_cache_kwargs)
        self.enabled = True
        
    def enable_frequency_cache(self):
        """Enable frequency-domain feature reuse."""
        self.enabled = True
        
    def disable_frequency_cache(self):
        """Disable frequency-domain feature reuse."""
        self.enabled = False
    
    def process_timestep_with_frequency_analysis(self, 
                                               latents: torch.Tensor, 
                                               step: int,
                                               force_compute: bool = False) -> tuple:
        """
        Process latents at a timestep with frequency analysis.
        
        Returns:
            Tuple of (processed_latents, analysis_info)
        """
        if not self.enabled:
            return latents, {'feature_reused': False, 'step': step}
        
        processed_latents, analysis_info = self.frequency_cache.process_timestep_latents(
            latents, step, force_compute
        )
        
        return processed_latents, analysis_info
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.frequency_cache.get_performance_stats()
    
    def clear_frequency_cache(self):
        """Clear the frequency cache."""
        self.frequency_cache.clear_cache()


def integrate_frequency_cache_to_pipeline(pipeline, **cache_kwargs):
    """
    Integrate frequency cache functionality into an existing Flux pipeline.
    
    Usage:
        # Create wrapper
        flux_with_cache = integrate_frequency_cache_to_pipeline(
            flux_pipeline, 
            similarity_threshold=0.85,
            compression_ratio=0.3
        )
        
        # Use like normal pipeline
        result = flux_with_cache.pipeline(prompt="your prompt")
        
        # Get performance stats
        stats = flux_with_cache.get_performance_stats()
    """
    return FluxPipelineWithFrequencyCache(pipeline, **cache_kwargs)


# Example of how to modify the _save_timestep_image method in pipeline_flux.py
def enhanced_timestep_processing(self, latents: torch.Tensor, step: int):
    """
    Enhanced version of _save_timestep_image that includes frequency analysis.
    This would replace or augment the existing method in pipeline_flux.py.
    """
    from fourier_cache.latent_frequency_cache import get_latent_frequency_cache
    
    # Get frequency cache
    freq_cache = get_latent_frequency_cache()
    
    # Process with frequency analysis
    processed_latents, analysis_info = freq_cache.process_timestep_latents(
        latents, step
    )
    
    # Print analysis info for debugging
    if step % 5 == 0:  # Print every 5 steps to avoid spam
        print(f"[Frequency Analysis] Step {step}: "
              f"Reused={analysis_info.get('feature_reused', False)}, "
              f"Efficiency={analysis_info.get('reuse_efficiency', 0):.2%}")
    
    # Continue with existing image saving logic
    # (This would be the existing _save_timestep_image code)
    # For example:
    if hasattr(self, '_save_timestep_image_original'):
        self._save_timestep_image_original(processed_latents, step)
    
    return processed_latents


# Example usage in the main pipeline
def example_pipeline_modification():
    """
    Example showing how to modify the pipeline __call__ method.
    """
    # This would be added to the pipeline_flux.py file
    
    def __call___with_frequency_cache(self, *args, **kwargs):
        # Initialize frequency cache if not already done
        if not hasattr(self, 'frequency_cache'):
            from fourier_cache.latent_frequency_cache import get_latent_frequency_cache
            self.frequency_cache = get_latent_frequency_cache()
        
        # Add frequency analysis callback
        original_callback = kwargs.get('callback_on_step_end')
        
        def frequency_callback(pipe, i, t, callback_kwargs):
            # Process frequency analysis
            latents = callback_kwargs.get('latents')
            if latents is not None:
                processed_latents, analysis_info = self.frequency_cache.process_timestep_latents(
                    latents, i
                )
                callback_kwargs['latents'] = processed_latents
                
                # Update callback kwargs for next step
                if analysis_info.get('feature_reused'):
                    print(f"Step {i}: Features reused from cache")
            
            # Call original callback if exists
            if original_callback:
                return original_callback(pipe, i, t, callback_kwargs)
            return callback_kwargs
        
        kwargs['callback_on_step_end'] = frequency_callback
        
        # Call original __call__ method
        return super().__call__(*args, **kwargs)
    
    return __call__with_frequency_cache


if __name__ == "__main__":
    # Simple test/demo
    print("Flux Frequency Cache Integration Example")
    print("This module provides frequency-domain feature reuse for Flux models.")
    print("See the integration examples above for usage details.")