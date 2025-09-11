#!/usr/bin/env python3
"""
Fourier Spectrum Analyzer for Image Comparison
Analyzes high, medium, and low frequency components of images in the intermediates folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2


def load_image(image_path):
    """Load image and convert to grayscale numpy array."""
    img = Image.open(image_path).convert('L')
    return np.array(img)


def compute_fourier_transform(image):
    """Compute 2D Fourier Transform of image."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    return magnitude_spectrum, f_shift


def create_frequency_bands(shape, low_cutoff=0.1, high_cutoff=0.4):
    """Create masks for low, medium, and high frequency bands."""
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create coordinate grids
    y = np.arange(rows) - center_row
    x = np.arange(cols) - center_col
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    # Calculate distance from center (normalized)
    D = np.sqrt(X**2 + Y**2) / min(center_row, center_col)
    
    # Create frequency band masks
    low_mask = D <= low_cutoff
    medium_mask = (D > low_cutoff) & (D <= high_cutoff)
    high_mask = D > high_cutoff
    
    return low_mask, medium_mask, high_mask


def analyze_frequency_bands(magnitude_spectrum, low_mask, medium_mask, high_mask):
    """Analyze energy in different frequency bands."""
    low_energy = np.sum(magnitude_spectrum[low_mask])
    medium_energy = np.sum(magnitude_spectrum[medium_mask])
    high_energy = np.sum(magnitude_spectrum[high_mask])
    total_energy = np.sum(magnitude_spectrum)
    
    return {
        'low': low_energy / total_energy if total_energy > 0 else 0,
        'medium': medium_energy / total_energy if total_energy > 0 else 0,
        'high': high_energy / total_energy if total_energy > 0 else 0,
        'low_abs': low_energy,
        'medium_abs': medium_energy,
        'high_abs': high_energy,
        'total': total_energy
    }


def process_images(intermediates_dir):
    """Process all images in intermediates directory."""
    intermediates_path = Path(intermediates_dir)
    image_files = sorted([f for f in intermediates_path.glob("*.png")])
    
    results = []
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        # Load image
        image = load_image(img_path)
        
        # Compute Fourier transform
        magnitude_spectrum, _ = compute_fourier_transform(image)
        
        # Create frequency bands
        low_mask, medium_mask, high_mask = create_frequency_bands(image.shape)
        
        # Analyze frequency bands
        analysis = analyze_frequency_bands(magnitude_spectrum, low_mask, medium_mask, high_mask)
        
        results.append({
            'filename': img_path.name,
            'step': int(img_path.stem.split('_')[1]),
            'analysis': analysis,
            'magnitude_spectrum': magnitude_spectrum,
            'shape': image.shape
        })
    
    return results


def plot_spectrum_comparison(results):
    """Create comparison plots of frequency spectra."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Frequency band evolution
    steps = [r['step'] for r in results]
    low_freqs = [r['analysis']['low'] for r in results]
    medium_freqs = [r['analysis']['medium'] for r in results]
    high_freqs = [r['analysis']['high'] for r in results]
    
    axes[0, 0].plot(steps, low_freqs, 'b-', label='Low Frequency', linewidth=2)
    axes[0, 0].plot(steps, medium_freqs, 'g-', label='Medium Frequency', linewidth=2)
    axes[0, 0].plot(steps, high_freqs, 'r-', label='High Frequency', linewidth=2)
    axes[0, 0].set_xlabel('Processing Step')
    axes[0, 0].set_ylabel('Relative Energy')
    axes[0, 0].set_title('Frequency Evolution Across Processing Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Energy ratio heatmap
    energy_matrix = np.array([[r['analysis']['low'], r['analysis']['medium'], r['analysis']['high']] 
                             for r in results])
    im = axes[0, 1].imshow(energy_matrix.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_xlabel('Processing Step')
    axes[0, 1].set_ylabel('Frequency Band')
    axes[0, 1].set_title('Energy Distribution Heatmap')
    axes[0, 1].set_yticks([0, 1, 2])
    axes[0, 1].set_yticklabels(['Low', 'Medium', 'High'])
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot 3: Total energy evolution
    total_energies = [r['analysis']['total'] for r in results]
    axes[1, 0].semilogy(steps, total_energies, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Processing Step')
    axes[1, 0].set_ylabel('Total Energy (log scale)')
    axes[1, 0].set_title('Total Spectral Energy Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: High/Low frequency ratio
    high_low_ratios = [r['analysis']['high'] / (r['analysis']['low'] + 1e-10) for r in results]
    axes[1, 1].plot(steps, high_low_ratios, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Processing Step')
    axes[1, 1].set_ylabel('High/Low Frequency Ratio')
    axes[1, 1].set_title('High vs Low Frequency Balance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_individual_spectra(results, output_dir):
    """Save individual spectrum visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original spectrum
        spectrum = np.log(result['magnitude_spectrum'] + 1)
        axes[0].imshow(spectrum, cmap='viridis')
        axes[0].set_title(f'Log Magnitude Spectrum\n{result["filename"]}')
        axes[0].axis('off')
        
        # Frequency band visualization
        low_mask, medium_mask, high_mask = create_frequency_bands(result['shape'])
        combined_mask = low_mask.astype(int) + 2 * medium_mask.astype(int) + 3 * high_mask.astype(int)
        axes[1].imshow(combined_mask, cmap='Set3')
        axes[1].set_title('Frequency Bands\n(Blue=Low, Green=Medium, Red=High)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spectrum_{result["filename"]}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("FOURIER SPECTRUM ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total images processed: {len(results)}")
    print(f"Processing steps: {min(r['step'] for r in results)} to {max(r['step'] for r in results)}")
    
    print("\nFrequency Distribution (averaged across all steps):")
    avg_low = np.mean([r['analysis']['low'] for r in results])
    avg_medium = np.mean([r['analysis']['medium'] for r in results])
    avg_high = np.mean([r['analysis']['high'] for r in results])
    
    print(f"  Low frequency:    {avg_low:.3f} ({avg_low*100:.1f}%)")
    print(f"  Medium frequency: {avg_medium:.3f} ({avg_medium*100:.1f}%)")
    print(f"  High frequency:   {avg_high:.3f} ({avg_high*100:.1f}%)")
    
    print("\nStep-by-step analysis:")
    for result in results:
        print(f"  Step {result['step']:3d}: Low={result['analysis']['low']:.3f}, "
              f"Med={result['analysis']['medium']:.3f}, "
              f"High={result['analysis']['high']:.3f}")


def main():
    """Main function."""
    intermediates_dir = "./intermediates"
    output_dir = "./spectra"
    
    print("Starting Fourier Spectrum Analysis...")
    print(f"Input directory: {intermediates_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process all images
    results = process_images(intermediates_dir)
    
    # Generate plots
    print("Generating comparison plots...")
    fig = plot_spectrum_comparison(results)
    fig.savefig(os.path.join(output_dir, "frequency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual spectra
    print("Saving individual spectra...")
    save_individual_spectra(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()