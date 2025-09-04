#!/usr/bin/env python
"""
Simple test script to verify Fourier cache functionality.
"""
import torch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fourier_cache.simple_fft import fft_compress, fft_decompress, compute_frequency_energy


def test_fft_compression():
    """Test FFT compression and decompression."""
    print("Testing FFT compression...")
    
    # Create a simple test tensor
    tensor = torch.randn(2, 3, 16, 32)  # batch_size=2, channels=3, height=16, width=32
    print(f"Original tensor shape: {tensor.shape}")
    
    # Compress the tensor
    compressed, mask, original_shape = fft_compress(tensor, compression_ratio=0.5)
    print(f"Compressed tensor shape: {compressed.shape}")
    print(f"Compression mask shape: {mask.shape}")
    print(f"Compression ratio: {mask.sum().item() / mask.numel():.2f}")
    
    # Decompress the tensor
    decompressed = fft_decompress(compressed, mask, original_shape)
    print(f"Decompressed tensor shape: {decompressed.shape}")
    
    # Check if shapes match
    assert decompressed.shape == tensor.shape, f"Shape mismatch: {decompressed.shape} vs {tensor.shape}"
    
    # Check reconstruction quality
    error = torch.mean(torch.abs(tensor - decompressed))
    print(f"Reconstruction error (MAE): {error.item():.6f}")
    
    # For lossy compression, we expect some error
    assert error < 0.1, f"Reconstruction error too high: {error.item()}"
    
    print("FFT compression test passed!")


def test_frequency_analysis():
    """Test frequency energy analysis."""
    print("\nTesting frequency analysis...")
    
    # Create test tensors with different characteristics
    # 1. Low frequency signal (smooth)
    x = torch.linspace(0, 4*torch.pi, 32)
    low_freq_signal = torch.sin(x).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add batch/channel dims
    low_freq_signal = low_freq_signal.expand(2, 3, 16, -1)  # batch=2, channels=3, height=16
    
    # 2. High frequency signal (oscillating)
    high_freq_signal = torch.sin(8*x).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    high_freq_signal = high_freq_signal.expand(2, 3, 16, -1)
    
    # Analyze frequency characteristics
    low_freq_metrics = compute_frequency_energy(low_freq_signal)
    high_freq_metrics = compute_frequency_energy(high_freq_signal)
    
    print(f"Low frequency signal - DC energy: {low_freq_metrics['dc_energy'].mean().item():.4f}")
    print(f"Low frequency signal - Low freq energy: {low_freq_metrics['low_freq_energy'].mean().item():.4f}")
    print(f"Low frequency signal - High freq energy: {low_freq_metrics['high_freq_energy'].mean().item():.4f}")
    
    print(f"High frequency signal - DC energy: {high_freq_metrics['dc_energy'].mean().item():.4f}")
    print(f"High frequency signal - Low freq energy: {high_freq_metrics['low_freq_energy'].mean().item():.4f}")
    print(f"High frequency signal - High freq energy: {high_freq_metrics['high_freq_energy'].mean().item():.4f}")
    
    # High frequency signal should have more high-frequency energy
    assert high_freq_metrics['high_freq_energy'].mean() > low_freq_metrics['high_freq_energy'].mean()
    
    print("Frequency analysis test passed!")


if __name__ == "__main__":
    test_fft_compression()
    test_frequency_analysis()
    print("\nAll tests passed!")
