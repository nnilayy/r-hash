#!/usr/bin/env python3
"""
Script to analyze BDE value ranges and distributions from preprocessed datasets
to create realistic augmentations.
"""

import pickle
import numpy as np
import os
from utils.pickle_patch import patch_pickle_loading

def analyze_bde_dataset(dataset_path):
    """
    Analyze BDE value ranges and distributions from a preprocessed dataset.
    
    Args:
        dataset_path: Path to the .pkl dataset file
    
    Returns:
        Dictionary with statistics about BDE values
    """
    print(f"Analyzing dataset: {dataset_path}")
    
    # Load the dataset
    patch_pickle_loading()
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Collect all BDE values
    all_samples = []
    for i in range(len(dataset)):
        sample, _ = dataset[i]  # (eeg_data, label)
        # Remove batch dimension if present: (1, electrodes, bands) -> (electrodes, bands)
        if sample.dim() == 3:
            sample = sample.squeeze(0)
        all_samples.append(sample.numpy())
    
    all_samples = np.array(all_samples)  # Shape: (num_samples, electrodes, bands)
    print(f"BDE tensor shape: {all_samples.shape}")
    
    # Calculate statistics
    stats = {
        'shape': all_samples.shape,
        'global_min': np.min(all_samples),
        'global_max': np.max(all_samples),
        'global_mean': np.mean(all_samples),
        'global_std': np.std(all_samples),
        'global_median': np.median(all_samples),
        'percentiles': {
            'p1': np.percentile(all_samples, 1),
            'p5': np.percentile(all_samples, 5),
            'p25': np.percentile(all_samples, 25),
            'p75': np.percentile(all_samples, 75),
            'p95': np.percentile(all_samples, 95),
            'p99': np.percentile(all_samples, 99),
        }
    }
    
    # Per-band statistics (theta, alpha, beta, gamma)
    band_names = ['theta', 'alpha', 'beta', 'gamma']
    stats['per_band'] = {}
    
    for band_idx in range(all_samples.shape[2]):  # 4 bands
        band_data = all_samples[:, :, band_idx]  # All samples, all electrodes, this band
        stats['per_band'][band_names[band_idx]] = {
            'min': np.min(band_data),
            'max': np.max(band_data), 
            'mean': np.mean(band_data),
            'std': np.std(band_data),
            'median': np.median(band_data),
            'p5': np.percentile(band_data, 5),
            'p95': np.percentile(band_data, 95),
        }
    
    # Per-electrode statistics
    stats['per_electrode'] = {}
    for electrode_idx in range(all_samples.shape[1]):  # num_electrodes
        electrode_data = all_samples[:, electrode_idx, :]  # All samples, this electrode, all bands
        stats['per_electrode'][f'electrode_{electrode_idx}'] = {
            'min': np.min(electrode_data),
            'max': np.max(electrode_data),
            'mean': np.mean(electrode_data),
            'std': np.std(electrode_data),
            'median': np.median(electrode_data),
            'p5': np.percentile(electrode_data, 5),
            'p95': np.percentile(electrode_data, 95),
        }
    
    # Value distribution histogram - skip if matplotlib not available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        
        # Overall distribution
        plt.subplot(2, 3, 1)
        plt.hist(all_samples.flatten(), bins=50, alpha=0.7, density=True)
        plt.title('Overall BDE Value Distribution')
        plt.xlabel('BDE Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Per-band distributions
        colors = ['red', 'blue', 'green', 'orange']
        for i, band_name in enumerate(band_names):
            plt.subplot(2, 3, i+2)
            band_data = all_samples[:, :, i].flatten()
            plt.hist(band_data, bins=30, alpha=0.7, color=colors[i], density=True)
            plt.title(f'{band_name.capitalize()} Band Distribution')
            plt.xlabel('BDE Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        
        # Band comparison
        plt.subplot(2, 3, 6)
        for i, band_name in enumerate(band_names):
            band_data = all_samples[:, :, i].flatten()
            plt.hist(band_data, bins=30, alpha=0.5, color=colors[i], 
                    density=True, label=band_name.capitalize())
        plt.title('All Bands Comparison')
        plt.xlabel('BDE Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        dataset_name = os.path.basename(dataset_path).replace('.pkl', '')
        plt.savefig(f'bde_analysis_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_saved = True
        
    except ImportError:
        print("Matplotlib not available - skipping plots")
        plot_saved = False
    
    return stats

def print_statistics(stats):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print("BDE VALUE RANGE ANALYSIS")
    print("="*80)
    
    print(f"\n📊 GLOBAL STATISTICS:")
    print(f"   Shape: {stats['shape']}")
    print(f"   Min: {stats['global_min']:.6f}")
    print(f"   Max: {stats['global_max']:.6f}")  
    print(f"   Mean: {stats['global_mean']:.6f}")
    print(f"   Std: {stats['global_std']:.6f}")
    print(f"   Median: {stats['global_median']:.6f}")
    
    print(f"\n📈 PERCENTILES:")
    for k, v in stats['percentiles'].items():
        print(f"   {k.upper()}: {v:.6f}")
    
    print(f"\n🎵 PER-BAND STATISTICS:")
    for band_name, band_stats in stats['per_band'].items():
        print(f"\n   {band_name.upper()} ({band_name} band):")
        for stat_name, stat_value in band_stats.items():
            print(f"     {stat_name}: {stat_value:.6f}")
    
    print(f"\n🧠 SAMPLE ELECTRODE STATISTICS (first 3):")
    electrode_keys = list(stats['per_electrode'].keys())[:3]
    for electrode_key in electrode_keys:
        electrode_stats = stats['per_electrode'][electrode_key]
        print(f"\n   {electrode_key.upper()}:")
        for stat_name, stat_value in electrode_stats.items():
            print(f"     {stat_name}: {stat_value:.6f}")

def create_realistic_augmentations(stats):
    """
    Create augmentation parameters based on actual BDE value ranges.
    
    Args:
        stats: Statistics dictionary from analyze_bde_dataset
    
    Returns:
        Dictionary with realistic augmentation parameters
    """
    global_std = stats['global_std']
    global_mean = stats['global_mean']
    value_range = stats['global_max'] - stats['global_min']
    
    # Calculate realistic augmentation parameters
    augmentation_params = {
        'realistic_noise_std': global_std * 0.1,  # 10% of natural variation
        'realistic_amplitude_range': (
            1.0 - 2 * global_std / abs(global_mean) if global_mean != 0 else 0.95,
            1.0 + 2 * global_std / abs(global_mean) if global_mean != 0 else 1.05
        ),
        'band_specific_noise': {},
        'band_scaling_ranges': {},
        'electrode_noise_levels': {},
    }
    
    # Per-band parameters
    for band_name, band_stats in stats['per_band'].items():
        band_std = band_stats['std']
        band_mean = band_stats['mean']
        
        augmentation_params['band_specific_noise'][band_name] = band_std * 0.15
        augmentation_params['band_scaling_ranges'][band_name] = (
            1.0 - band_std / abs(band_mean) if band_mean != 0 else 0.9,
            1.0 + band_std / abs(band_mean) if band_mean != 0 else 1.1
        )
    
    # Per-electrode noise levels (sample first 5)
    electrode_keys = list(stats['per_electrode'].keys())[:5]
    for electrode_key in electrode_keys:
        electrode_stats = stats['per_electrode'][electrode_key]
        electrode_std = electrode_stats['std']
        augmentation_params['electrode_noise_levels'][electrode_key] = electrode_std * 0.12
    
    return augmentation_params

def print_augmentation_recommendations(params):
    """Print realistic augmentation parameter recommendations."""
    print("\n" + "="*80)
    print("REALISTIC AUGMENTATION PARAMETERS")
    print("="*80)
    
    print(f"\n🎯 GLOBAL AUGMENTATION PARAMETERS:")
    print(f"   Realistic Noise Std: {params['realistic_noise_std']:.6f}")
    print(f"   Realistic Amplitude Range: {params['realistic_amplitude_range'][0]:.4f} - {params['realistic_amplitude_range'][1]:.4f}")
    
    print(f"\n🎵 BAND-SPECIFIC PARAMETERS:")
    for band_name, noise_level in params['band_specific_noise'].items():
        scale_range = params['band_scaling_ranges'][band_name]
        print(f"   {band_name.upper()}:")
        print(f"     Noise Level: {noise_level:.6f}")
        print(f"     Scale Range: {scale_range[0]:.4f} - {scale_range[1]:.4f}")
    
    print(f"\n🧠 ELECTRODE-SPECIFIC NOISE LEVELS (sample):")
    for electrode_key, noise_level in params['electrode_noise_levels'].items():
        print(f"   {electrode_key}: {noise_level:.6f}")

if __name__ == "__main__":
    # Analyze DREAMER binary arousal dataset
    dataset_path = "preprocessed_datasets/dreamer_binary_arousal_dataset.pkl"
    
    if os.path.exists(dataset_path):
        stats = analyze_bde_dataset(dataset_path)
        print_statistics(stats)
        
        # Create realistic augmentation parameters
        augmentation_params = create_realistic_augmentations(stats)
        print_augmentation_recommendations(augmentation_params)
        
        print(f"\n✅ Analysis complete!")
        print(f"\n💡 Use these parameters to create realistic BDE augmentations!")
        
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Make sure you've run the preprocessing scripts first.")
