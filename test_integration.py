#!/usr/bin/env python3
"""
Quick test to verify augmentation integration works before full LOSO run.
"""

import numpy as np
import torch
import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from augmentations import EEGAugmentationSuite, AUGMENTATION_CONFIGS
    from dataset_augmentations import AugmentedDatasetReshape, MixupAugmentedDataset
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_integration():
    """Test the augmentation integration with realistic data shapes"""
    
    print("🧪 Testing Augmentation Integration")
    print("="*50)
    
    # Simulate realistic data dimensions (similar to your DREAMER dataset)
    num_samples = 1000
    num_electrodes = 14  
    bde_dim = 4
    num_subjects = 5
    
    # Create synthetic flattened BDE data (as it comes from SMOTE)
    np.random.seed(42)
    X_flat = np.random.normal(0, 1, (num_samples, num_electrodes * bde_dim))
    y = np.random.randint(0, 2, num_samples)  # Binary classification
    subject_ids = np.random.randint(0, num_subjects, num_samples)
    
    print(f"Input data shapes:")
    print(f"  X_flat: {X_flat.shape}")
    print(f"  y: {y.shape}")
    print(f"  subject_ids: {subject_ids.shape}")
    
    # Test 1: AugmentedDatasetReshape
    print(f"\n1️⃣ Testing AugmentedDatasetReshape...")
    try:
        dataset1 = AugmentedDatasetReshape(
            X_flat, y, 
            num_electrodes=num_electrodes,
            apply_augmentations=True,
            augmentation_config='conservative',
            seed=42
        )
        
        # Test data loading
        x_sample, y_sample = dataset1[0]
        print(f"  Sample shape: {x_sample.shape}, Label: {y_sample}")
        print(f"  Dataset length: {len(dataset1)}")
        print("  ✅ AugmentedDatasetReshape works!")
        
    except Exception as e:
        print(f"  ❌ AugmentedDatasetReshape failed: {e}")
        return False
    
    # Test 2: MixupAugmentedDataset
    print(f"\n2️⃣ Testing MixupAugmentedDataset...")
    try:
        dataset2 = MixupAugmentedDataset(
            X_flat, y, subject_ids,
            num_electrodes=num_electrodes,
            apply_augmentations=True,
            augmentation_config='conservative',
            enable_mixup=False,  # Start without mixup
            seed=42
        )
        
        # Test data loading
        x_sample, y_sample = dataset2[0]
        print(f"  Sample shape: {x_sample.shape}, Label: {y_sample}")
        print(f"  Dataset length: {len(dataset2)}")
        print("  ✅ MixupAugmentedDataset works!")
        
    except Exception as e:
        print(f"  ❌ MixupAugmentedDataset failed: {e}")
        return False
    
    # Test 3: DataLoader compatibility
    print(f"\n3️⃣ Testing DataLoader compatibility...")
    try:
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(dataset1, batch_size=32, shuffle=True, num_workers=0)
        batch_x, batch_y = next(iter(dataloader))
        print(f"  Batch X shape: {batch_x.shape}")
        print(f"  Batch y shape: {batch_y.shape}")
        print("  ✅ DataLoader works!")
        
    except Exception as e:
        print(f"  ❌ DataLoader failed: {e}")
        return False
    
    # Test 4: SMOTE-like data expansion
    print(f"\n4️⃣ Testing SMOTE-like expansion...")
    try:
        # Simulate SMOTE expansion (more samples than original)
        expansion_factor = 1.5
        new_size = int(num_samples * expansion_factor)
        
        X_expanded = np.repeat(X_flat, 2, axis=0)[:new_size]  # Simple expansion
        y_expanded = np.repeat(y, 2, axis=0)[:new_size]
        
        # Create subject IDs for expanded data (cycling through originals)
        subject_ids_expanded = []
        for i in range(new_size):
            subject_ids_expanded.append(subject_ids[i % len(subject_ids)])
        subject_ids_expanded = np.array(subject_ids_expanded)
        
        print(f"  Expanded shapes: X={X_expanded.shape}, y={y_expanded.shape}, subjects={len(subject_ids_expanded)}")
        
        dataset_expanded = MixupAugmentedDataset(
            X_expanded, y_expanded, subject_ids_expanded,
            num_electrodes=num_electrodes,
            apply_augmentations=True,
            augmentation_config='conservative',
            enable_mixup=False,
            seed=42
        )
        
        print(f"  Expanded dataset length: {len(dataset_expanded)}")
        print("  ✅ SMOTE-like expansion works!")
        
    except Exception as e:
        print(f"  ❌ SMOTE-like expansion failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed! Integration should work in LOSO training.")
    return True

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🚀 Ready to run LOSO training with augmentations!")
        print("Set ENABLE_AUGMENTATIONS=True and ENABLE_MIXUP=False in train_loso.py")
    else:
        print("\n🚫 Fix the errors above before running LOSO training")
        sys.exit(1)
