#!/usr/bin/env python3
"""
Example script showing how to test augmentations before running full LOSO training.
Run this to see augmentation effects on a small sample.
"""

import numpy as np
import torch
from augmentations import EEGAugmentationSuite, AUGMENTATION_CONFIGS
from dataset_augmentations import AugmentedDatasetReshape

def test_augmentations():
    """Test augmentations on sample data"""
    
    # Create sample BDE data (similar to your real data)
    # Shape: (num_samples, num_electrodes * bde_dim)
    num_samples = 10
    num_electrodes = 14  # DREAMER dataset
    bde_dim = 4
    
    # Generate synthetic BDE-like features
    np.random.seed(42)
    X_sample = np.random.normal(0, 1, (num_samples, num_electrodes * bde_dim))
    y_sample = np.random.randint(0, 3, num_samples)  # 3 classes for SEED-like
    
    print("🧪 TESTING INDIVIDUAL AUGMENTATIONS")
    print("="*50)
    
    # Initialize augmentation suite
    augmenter = EEGAugmentationSuite(seed=42)
    
    # Test single sample (reshaped to electrode x bde format)
    sample_reshaped = X_sample[0].reshape(num_electrodes, bde_dim)
    
    print(f"Original sample shape: {sample_reshaped.shape}")
    print(f"Original sample stats: mean={np.mean(sample_reshaped):.4f}, std={np.std(sample_reshaped):.4f}")
    print()
    
    # Test individual augmentations
    augmentations_to_test = [
        ('gaussian_noise', lambda x: augmenter.gaussian_noise(x, noise_std=0.02)),
        ('amplitude_scaling', lambda x: augmenter.amplitude_scaling(x, scale_range=(0.8, 1.2))),
        ('channel_dropout', lambda x: augmenter.channel_dropout(x, dropout_prob=0.2)),
        ('frequency_band_dropout', lambda x: augmenter.frequency_band_dropout(x, band_dropout_prob=0.25)),
        ('spatial_smoothing', lambda x: augmenter.spatial_smoothing(x, smooth_prob=0.3)),
        ('frequency_band_scaling', lambda x: augmenter.frequency_band_scaling(x, scale_range=(0.7, 1.3))),
    ]
    
    for name, aug_func in augmentations_to_test:
        aug_sample = aug_func(sample_reshaped.copy())
        diff = aug_sample - sample_reshaped
        
        print(f"📊 {name.upper()}")
        print(f"  Augmented stats: mean={np.mean(aug_sample):.4f}, std={np.std(aug_sample):.4f}")
        print(f"  Change stats: mean={np.mean(diff):.4f}, std={np.std(diff):.4f}")
        print(f"  Max change: {np.max(np.abs(diff)):.4f}")
        print()
    
    print("🏷️  TESTING PRESET CONFIGURATIONS")
    print("="*50)
    
    # Test preset configurations
    for config_name, config in AUGMENTATION_CONFIGS.items():
        aug_sample = augmenter.custom_augmentation(sample_reshaped.copy(), config)
        diff = aug_sample - sample_reshaped
        
        print(f"📦 {config_name.upper()} CONFIG")
        print(f"  Applied methods: {list(config.keys())}")
        print(f"  Augmented stats: mean={np.mean(aug_sample):.4f}, std={np.std(aug_sample):.4f}")
        print(f"  Max change: {np.max(np.abs(diff)):.4f}")
        print()
    
    print("🔄 TESTING DATASET WRAPPER")
    print("="*50)
    
    # Test dataset wrapper
    aug_dataset = AugmentedDatasetReshape(
        X_sample, y_sample, 
        num_electrodes=num_electrodes,
        apply_augmentations=True,
        augmentation_config='moderate',
        seed=42
    )
    
    print(f"Original dataset shape: {X_sample.shape}")
    print(f"Augmented dataset length: {len(aug_dataset)}")
    
    # Get a few samples to see variation
    sample_variations = []
    for i in range(3):  # Same index, different augmentations due to randomness
        x_aug, y = aug_dataset[0]
        sample_variations.append(x_aug.numpy())
    
    print(f"Sample variations (same input, different augmentations):")
    for i, var in enumerate(sample_variations):
        print(f"  Variation {i+1}: mean={np.mean(var):.4f}, std={np.std(var):.4f}")
    
    print("\n✅ AUGMENTATION TESTING COMPLETE!")
    print("You can now run the full LOSO training with augmentations enabled.")

if __name__ == "__main__":
    test_augmentations()
