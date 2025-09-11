import torch
import torch.fft
from typing import Dict, List, Optional, Tuple
import numpy as np


class LatentFrequencyAnalyzer:
    """
    Analyze frequency-domain similarity directly from latents for efficient feature reuse.
    """
    
    def __init__(self, compression_ratio: float = 0.3, similarity_threshold: float = 0.85):
        self.compression_ratio = compression_ratio
        self.similarity_threshold = similarity_threshold
        self.cached_features = {}
        
    def _reshape_latent_to_spatial(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Reshape latents from [B, L, C] to [B, C, H, W] for 2D FFT analysis.
        For Flux: L = 4096, C = 64 -> H = W = 64, C = 64
        """
        B, L, C = latents.shape
        # For Flux, L is typically 4096 which is 64*64
        H = W = int(np.sqrt(L))
        if H * W != L:
            # Handle non-square case
            H = int(np.sqrt(L * 16/9))  # Assume 16:9 aspect ratio as fallback
            W = L // H
            
        # Reshape to spatial dimensions [B, C, H, W]
        spatial_latents = latents.permute(0, 2, 1).reshape(B, C, H, W)
        return spatial_latents
    
    def _compute_2d_fft(self, spatial_latents: torch.Tensor) -> torch.Tensor:
        """Compute 2D FFT of spatial latents."""
        return torch.fft.fft2(spatial_latents, dim=(-2, -1))
    
    def _compute_frequency_signature(self, fft_latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute frequency signature for similarity comparison."""
        # Compute magnitude spectrum
        magnitude = torch.abs(fft_latents)
        
        # Compute energy distribution across frequency bands
        B, C, H, W = magnitude.shape
        
        # Divide into frequency bands
        center_h, center_w = H // 2, W // 2
        
        # Low frequencies (center 25%)
        low_h, low_w = H // 4, W // 4
        low_freq_mask = torch.zeros_like(magnitude, dtype=torch.bool)
        low_freq_mask[:, :, 
                     center_h-low_h:center_h+low_h, 
                     center_w-low_w:center_w+low_w] = True
        
        # Medium frequencies (25%-50% from center)
        med_h, med_w = H // 4, W // 4
        med_freq_mask = torch.zeros_like(magnitude, dtype=torch.bool)
        med_freq_mask[:, :, 
                     center_h-2*low_h:center_h+2*low_h, 
                     center_w-2*low_w:center_w+2*low_w] = True
        med_freq_mask = med_freq_mask & ~low_freq_mask
        
        # High frequencies (remaining)
        high_freq_mask = ~(low_freq_mask | med_freq_mask)
        
        # Compute energy in each band
        low_energy = torch.sum(magnitude[low_freq_mask] ** 2)
        med_energy = torch.sum(magnitude[med_freq_mask] ** 2)
        high_energy = torch.sum(magnitude[high_freq_mask] ** 2)
        total_energy = low_energy + med_energy + high_energy
        
        # Normalize
        signature = {
            'low_freq_ratio': low_energy / total_energy,
            'med_freq_ratio': med_energy / total_energy,
            'high_freq_ratio': high_energy / total_energy,
            'magnitude_mean': torch.mean(magnitude),
            'magnitude_std': torch.std(magnitude),
            'phase_mean': torch.mean(torch.angle(fft_latents)),
        }
        
        return signature
    
    def _compress_frequency_representation(self, fft_latents: torch.Tensor) -> torch.Tensor:
        """Compress frequency representation for efficient storage."""
        # Keep only low frequency components
        B, C, H, W = fft_latents.shape
        keep_count = int(H * W * self.compression_ratio)
        
        # Create low-pass filter
        center_h, center_w = H // 2, W // 2
        keep_radius = int(np.sqrt(keep_count / np.pi))
        
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y = y.to(fft_latents.device)
        x = x.to(fft_latents.device)
        
        distance = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        mask = distance <= keep_radius
        
        # Apply mask
        compressed = fft_latents * mask.float().unsqueeze(0).unsqueeze(0)
        
        return compressed
    
    def compute_frequency_similarity(self, latents1: torch.Tensor, latents2: torch.Tensor) -> float:
        """
        Compute frequency-domain similarity between two latent representations.
        Returns similarity score between 0 and 1.
        """
        # Reshape to spatial
        spatial1 = self._reshape_latent_to_spatial(latents1)
        spatial2 = self._reshape_latent_to_spatial(latents2)
        
        # Compute FFT
        fft1 = self._compute_2d_fft(spatial1)
        fft2 = self._compute_2d_fft(spatial2)
        
        # Compute frequency signatures
        sig1 = self._compute_frequency_signature(fft1)
        sig2 = self._compute_frequency_signature(fft2)
        
        # Compute similarity based on frequency energy distribution
        energy_diff = (
            torch.abs(sig1['low_freq_ratio'] - sig2['low_freq_ratio']) +
            torch.abs(sig1['med_freq_ratio'] - sig2['med_freq_ratio']) +
            torch.abs(sig1['high_freq_ratio'] - sig2['high_freq_ratio'])
        )
        
        # Compute magnitude similarity
        mag_sim = 1 - torch.abs(sig1['magnitude_mean'] - sig2['magnitude_mean']) / (
            sig1['magnitude_mean'] + sig2['magnitude_mean'] + 1e-8
        )
        
        # Combined similarity score
        similarity = mag_sim * (1 - energy_diff)
        
        return float(similarity)
    
    def analyze_latent_frequency(self, latents: torch.Tensor, step: int) -> Dict[str, any]:
        """
        Analyze frequency characteristics of latents at a given timestep.
        """
        # Reshape to spatial
        spatial_latents = self._reshape_latent_to_spatial(latents)
        
        # Compute FFT
        fft_latents = self._compute_2d_fft(spatial_latents)
        
        # Compute frequency signature
        signature = self._compute_frequency_signature(fft_latents)
        
        # Compress for caching
        compressed_fft = self._compress_frequency_representation(fft_latents)
        
        # Store in cache
        self.cached_features[step] = {
            'compressed_fft': compressed_fft.detach(),
            'signature': {k: v.detach() for k, v in signature.items()},
            'original_shape': latents.shape
        }
        
        return {
            'step': step,
            'frequency_signature': {k: float(v) for k, v in signature.items()},
            'compression_ratio': self.compression_ratio,
            'cached': True
        }
    
    def find_similar_timesteps(self, current_latents: torch.Tensor, 
                             lookback_steps: int = 5) -> List[Tuple[int, float]]:
        """
        Find similar timesteps in recent history for feature reuse.
        Returns list of (step, similarity_score) tuples.
        """
        if not self.cached_features:
            return []
        
        similarities = []
        current_step = max(self.cached_features.keys()) + 1 if self.cached_features else 0
        
        # Check recent steps
        for step in range(max(0, current_step - lookback_steps), current_step):
            if step in self.cached_features:
                # Reconstruct from compressed representation
                cached_data = self.cached_features[step]
                
                # Compute similarity
                similarity = self.compute_frequency_similarity(
                    current_latents, 
                    self._decompress_to_latent(cached_data)
                )
                
                if similarity > self.similarity_threshold:
                    similarities.append((step, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _decompress_to_latent(self, cached_data: Dict) -> torch.Tensor:
        """Decompress cached representation back to latent format."""
        compressed_fft = cached_data['compressed_fft']
        original_shape = cached_data['original_shape']
        
        # Inverse FFT
        spatial_reconstructed = torch.fft.ifft2(compressed_fft, dim=(-2, -1)).real
        
        # Reshape back to original latent format
        B, C, H, W = spatial_reconstructed.shape
        L = H * W
        
        latent_reconstructed = spatial_reconstructed.reshape(B, C, L).permute(0, 2, 1)
        
        return latent_reconstructed
    
    def get_reusable_features(self, current_latents: torch.Tensor, 
                            similarity_threshold: Optional[float] = None) -> Optional[torch.Tensor]:
        """
        Get reusable features from cache based on frequency similarity.
        Returns the most similar cached latent if similarity exceeds threshold.
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
            
        similar_steps = self.find_similar_timesteps(current_latents)
        
        if similar_steps:
            best_step, best_similarity = similar_steps[0]
            if best_similarity > similarity_threshold:
                cached_data = self.cached_features[best_step]
                return self._decompress_to_latent(cached_data)
        
        return None
    
    def clear_cache(self):
        """Clear the frequency cache."""
        self.cached_features.clear()