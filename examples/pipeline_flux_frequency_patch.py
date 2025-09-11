"""
Patch file to add frequency analysis support to pipeline_flux.py
This shows the minimal changes needed to integrate frequency analysis.
"""

# Add these imports at the top of pipeline_flux.py (around line 16-25)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'fourier_cache'))
from latent_frequency_cache import get_latent_frequency_cache
"""

# Add this method to the xFuserFluxPipeline class (around line 150, before __call__)
"""
    def initialize_frequency_analysis(self, **kwargs):
        '''Initialize frequency analysis cache for the pipeline'''
        if not hasattr(self, 'frequency_cache'):
            self.frequency_cache = get_latent_frequency_cache(**kwargs)
        return self.frequency_cache
    
    def get_frequency_analysis_callback(self):
        '''Get frequency analysis callback for pipeline execution'''
        if not hasattr(self, 'frequency_cache'):
            self.initialize_frequency_analysis()
        
        def frequency_callback(pipe, step, t, callback_kwargs):
            latents = callback_kwargs.get('latents')
            if latents is not None:
                processed_latents, analysis_info = self.frequency_cache.process_timestep_latents(
                    latents, step
                )
                callback_kwargs['latents'] = processed_latents
                
                # Optional: log analysis info
                if step % 5 == 0 and hasattr(pipe, '_log_frequency_analysis'):
                    reuse_status = "REUSED" if analysis_info.get('feature_reused') else "Computed"
                    efficiency = analysis_info.get('reuse_efficiency', 0)
                    print(f"[FreqAnalysis] Step {step}: {reuse_status} | Efficiency: {efficiency:.1%}")
            
            return callback_kwargs
        
        return frequency_callback
"""

# Replace the existing _save_timestep_image method (around line 446) with this enhanced version:
"""
    def _save_timestep_image_with_frequency(self, latents: torch.Tensor, step: int,
                                         output_dir: str = "./intermediates",
                                         enable_frequency_analysis: bool = True):
        from xfuser.core.distributed import (
            get_world_group, get_sp_group, get_sequence_parallel_world_size,
            is_pipeline_last_stage,
        )
        import torch, os

        # 1. All SP rank must attend!
        print(f"[DEBUG] entry step={step}, world_rank={get_world_group().rank}, "
            f"sp_rank={get_sp_group().rank_in_group}", flush=True)

        # 2. Collect complete latent
        if get_sequence_parallel_world_size() > 1:
            latents = get_sp_group().all_gather(latents, dim=-2)

        # Apply frequency analysis if enabled
        if enable_frequency_analysis and hasattr(self, 'frequency_cache'):
            processed_latents, analysis_info = self.frequency_cache.process_timestep_latents(
                latents, step
            )
            
            # Log analysis info
            if step % 5 == 0:
                reuse_status = "✓ REUSED" if analysis_info.get('feature_reused') else "Computed"
                efficiency = analysis_info.get('reuse_efficiency', 0)
                print(f"[Frequency Analysis] Step {step:2d}: {reuse_status} | Efficiency: {efficiency:.1%}")
            
            latents = processed_latents

        # Only rank 0 saves
        if get_world_group().rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        
            latents = latents.contiguous()
            B, L, C = latents.shape               # [1, 4096, 64]
            h = w = int(np.sqrt(L)) * 2           # 128
            print(f"[DEBUG] step={step}, latents.shape={latents.shape}, h=w={h}, C={C}")
            # Correct reshape
            latents = latents.view(B, h//2, w//2, C//4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, C//4, h, w)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            with torch.no_grad():
                image = self.vae.decode(latents.to(self.vae.device), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type="pil")[0]
            
            # Add frequency analysis info to filename
            if hasattr(self, 'frequency_cache'):
                perf_stats = self.frequency_cache.get_performance_stats()
                freq_suffix = f"_freq_reuse{perf_stats['reuse_count']}" if perf_stats['reuse_count'] > 0 else "_freq_normal"
            else:
                freq_suffix = ""
                
            path = os.path.join(output_dir, f"step_{step:03d}{freq_suffix}.png")
            image.save(path)
            print(f"[DEBUG] step_{step:03d}{freq_suffix}.png saved", flush=True)

        # Sync
        torch.distributed.barrier(group=get_world_group().device_group)
"""

# In the _sync_pipeline method (around line 446), replace the existing call with:
"""
    # Enhanced: Save timestep image with frequency analysis
    self._save_timestep_image_with_frequency(latents, i)
"""

# In the _async_pipeline method (around line 585), replace the existing call with:
"""
    # Enhanced: Save timestep image with frequency analysis (rank 0 only)
    # First merge patches then save
    full_latents = torch.cat(patch_latents, dim=-2)
    self._save_timestep_image_with_frequency(full_latents, i)
"""

# Add this method to get frequency analysis performance (add at end of class):
"""
    def get_frequency_analysis_stats(self):
        '''Get frequency analysis performance statistics'''
        if hasattr(self, 'frequency_cache'):
            return self.frequency_cache.get_performance_stats()
        return {}
    
    def clear_frequency_cache(self):
        '''Clear frequency analysis cache'''
        if hasattr(self, 'frequency_cache'):
            self.frequency_cache.clear_cache()
"""