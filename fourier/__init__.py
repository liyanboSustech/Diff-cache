"""
Fourier Cache - Frequency domain acceleration for DiT models.
"""

__version__ = "0.1.0"

from .simple_fft import (
    fft_compress,
    fft_decompress,
    compute_frequency_energy
)

from .flux_adapter import (
    apply_fourier_cache_on_transformer
)

__all__ = [
    "fft_compress",
    "fft_decompress",
    "compute_frequency_energy",
    "apply_fourier_cache_on_transformer"
]