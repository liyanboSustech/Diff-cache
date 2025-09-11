
import torch
import torch.fft


def fft_compress(tensor: torch.Tensor, compression_ratio: float = 0.5) -> tuple:
    # For simplicity, we'll work with the last dimension
    dim = -1
    original_shape = tensor.shape
    
    # Apply FFT
    fft_tensor = torch.fft.fft(tensor, dim=dim)
    
    # Create mask to keep only low frequencies
    n = fft_tensor.shape[dim]
    keep_count = max(1, int(n * compression_ratio))
    
    # Keep low frequencies around DC (index 0)
    mask = torch.zeros(n, dtype=torch.bool, device=tensor.device)
    
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
    
    # Apply mask
    compressed = fft_tensor.clone()
    expand_dims = [1] * fft_tensor.dim()
    expand_dims[dim] = n
    mask_reshaped = mask.view(*expand_dims)
    compressed = compressed * mask_reshaped
    
    return compressed, mask, original_shape


def fft_decompress(compressed_tensor: torch.Tensor, mask: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    # Apply inverse FFT
    decompressed = torch.fft.ifft(compressed_tensor, dim=-1).real
    return decompressed


def compute_frequency_energy(tensor: torch.Tensor) -> dict:
    fft_tensor = torch.fft.fft(tensor, dim=-1)
    # Compute magnitude spectrum
    magnitude = torch.abs(fft_tensor)
    # Compute energy distribution
    total_energy = torch.sum(magnitude ** 2)
    normalized_energy = magnitude ** 2 / total_energy
    # Find dominant frequencies
    n = fft_tensor.shape[-1]
    # DC component (index 0)
    dc_energy = normalized_energy[..., 0]
    # Low frequency energy (first 10% of frequencies)
    low_freq_count = max(1, n // 10)
    low_freq_energy = torch.sum(normalized_energy[..., :low_freq_count], dim=-1)
    
    # High frequency energy (last 10% of frequencies)
    high_freq_start = n - low_freq_count
    high_freq_energy = torch.sum(normalized_energy[..., high_freq_start:], dim=-1)
    
    return {
        'dc_energy': dc_energy,
        'low_freq_energy': low_freq_energy,
        'high_freq_energy': high_freq_energy,
        'total_energy': total_energy
    }