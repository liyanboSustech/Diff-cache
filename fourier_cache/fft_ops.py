
import torch
import torch.fft
from typing import Optional, Tuple


def fft_1d(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    
    return torch.fft.fft(tensor, dim=dim)


def ifft_1d(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    
    return torch.fft.ifft(tensor, dim=dim).real


def fft_2d(tensor: torch.Tensor, dims: Tuple[int, int] = (-2, -1)) -> torch.Tensor:
    
    return torch.fft.fft2(tensor, dim=dims)


def ifft_2d(tensor: torch.Tensor, dims: Tuple[int, int] = (-2, -1)) -> torch.Tensor:
    
    return torch.fft.ifft2(tensor, dim=dims).real


def fft_3d(tensor: torch.Tensor, dims: Tuple[int, int, int] = (-3, -2, -1)) -> torch.Tensor:
    
    return torch.fft.fftn(tensor, dim=dims)


def ifft_3d(tensor: torch.Tensor, dims: Tuple[int, int, int] = (-3, -2, -1)) -> torch.Tensor:
    
    return torch.fft.ifftn(tensor, dim=dims).real


def compress_frequency(tensor: torch.Tensor, 
                      compression_ratio: float = 0.5, 
                      dim: int = -1,
                      preserve_dc: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # Apply FFT if tensor is real
    is_real = not torch.is_complex(tensor)
    if is_real:
        fft_tensor = fft_1d(tensor, dim=dim)
    else:
        fft_tensor = tensor
    
    # Create frequency mask
    n = fft_tensor.shape[dim]
    keep_count = max(1, int(n * compression_ratio))
    
    # For real signals, we need to preserve conjugate symmetry
    if is_real and not preserve_dc:
        # For real signals, keep symmetric frequencies around DC
        half_n = n // 2
        keep_left = min(keep_count // 2, half_n)
        keep_right = min(keep_count - keep_left, n - half_n)
        
        mask = torch.zeros(n, dtype=torch.bool, device=tensor.device)
        mask[half_n-keep_left:half_n] = True  # Negative frequencies
        mask[half_n:half_n+keep_right] = True  # Positive frequencies
    else:
        # Keep low frequencies around DC (index 0)
        mask = torch.zeros(n, dtype=torch.bool, device=tensor.device)
        if preserve_dc:
            # Always keep DC component
            center = 0
            mask[center] = True
            keep_count -= 1
        
        # Distribute remaining frequencies around DC
        half_keep = keep_count // 2
        if half_keep > 0:
            # Wrap around for circular indexing
            for i in range(half_keep):
                idx_pos = (center + i + 1) % n
                idx_neg = (center - i - 1) % n
                mask[idx_pos] = True
                if i < keep_count - half_keep:  # Handle odd keep_count
                    mask[idx_neg] = True
    
    # Apply mask to compress
    compressed = fft_tensor.clone()
    expand_dims = [1] * fft_tensor.dim()
    expand_dims[dim] = n
    mask_reshaped = mask.view(*expand_dims)
    compressed = compressed * mask_reshaped
    
    return compressed, mask


def decompress_frequency(compressed_tensor: torch.Tensor, 
                        mask: torch.Tensor, 
                        dim: int = -1) -> torch.Tensor:
    
    # Inverse FFT to get back to time domain
    reconstructed = ifft_1d(compressed_tensor, dim=dim)
    return reconstructed


def analyze_frequency_characteristics(tensor: torch.Tensor, 
                                   dim: int = -1) -> dict:
    
    fft_tensor = fft_1d(tensor, dim=dim)
    
    # Compute magnitude spectrum
    magnitude = torch.abs(fft_tensor)
    
    # Compute energy distribution
    total_energy = torch.sum(magnitude ** 2)
    normalized_energy = magnitude ** 2 / total_energy
    
    # Find dominant frequencies
    n = fft_tensor.shape[dim]
    freq_indices = torch.arange(n, device=tensor.device)
    
    # DC component (index 0)
    dc_energy = normalized_energy.select(dim, 0)
    
    # Low frequency energy (first 10% of frequencies)
    low_freq_count = max(1, n // 10)
    low_freq_energy = torch.sum(normalized_energy.narrow(dim, 0, low_freq_count), dim=dim)
    
    # High frequency energy (last 10% of frequencies)
    high_freq_start = n - low_freq_count
    high_freq_energy = torch.sum(normalized_energy.narrow(dim, high_freq_start, low_freq_count), dim=dim)
    
    return {
        'dc_energy': dc_energy,
        'low_freq_energy': low_freq_energy,
        'high_freq_energy': high_freq_energy,
        'total_energy': total_energy,
        'dominant_frequency_indices': torch.topk(magnitude, k=min(5, n), dim=dim).indices
    }