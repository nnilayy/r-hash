import torch
import numpy as np
import random
from torch.utils.data import Dataset
from augmentations import EEGAugmentationSuite


class AugmentedDatasetReshape(Dataset):
    """
    Enhanced DatasetReshape with subject-aware EEG augmentations.
    Applies augmentations during training while keeping validation data clean.
    """
    
    def __init__(self, X, y, num_electrodes=14, 
                 apply_augmentations=True, 
                 augmentation_config='moderate',
                 subject_ids=None,
                 seed=None):
        """
        Args:
            X: Flattened BDE features
            y: Labels 
            num_electrodes: Number of electrodes per sample
            apply_augmentations: Whether to apply augmentations
            augmentation_config: 'conservative', 'moderate', 'aggressive', or custom dict
            subject_ids: Array of subject IDs for group-wise augmentation
            seed: Random seed for augmentations
        """
        # Reshape data same as original DatasetReshape
        # Handle both squeezed and unsqueezed cases
        X_reshaped = X.reshape(-1, num_electrodes, 4)
        if len(X_reshaped.shape) == 4:  # If there's an extra dimension
            X_reshaped = X_reshaped.squeeze(1)
            
        self.X = torch.tensor(X_reshaped, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        # Store subject IDs for group-wise augmentation
        self.subject_ids = subject_ids
        
        # Debug info
        print(f"AugmentedDatasetReshape: X shape {X.shape} -> {self.X.shape}, y shape {y.shape} -> {self.y.shape}")
        
        # Augmentation setup
        self.apply_augmentations = apply_augmentations
        self.augmenter = EEGAugmentationSuite(seed=seed)
        
        # Load augmentation configuration
        if isinstance(augmentation_config, str):
            from augmentations import AUGMENTATION_CONFIGS
            self.aug_config = AUGMENTATION_CONFIGS.get(augmentation_config, 
                                                      AUGMENTATION_CONFIGS['moderate'])
        else:
            self.aug_config = augmentation_config
        
        # Create subject-wise augmentation mask
        self.augmentation_mask = self._create_subject_augmentation_mask()
            
        print(f"Dataset initialized with augmentations: {apply_augmentations}")
        if apply_augmentations:
            print(f"Augmentation config: {list(self.aug_config.keys())}")
            if self.subject_ids is not None:
                aug_count = sum(self.augmentation_mask)
                print(f"Subject-aware augmentation: {aug_count}/{len(self)} samples ({aug_count/len(self)*100:.1f}%)")
    
    def _create_subject_augmentation_mask(self):
        """Create boolean mask for which samples should be augmented (subject-aware)."""
        if self.subject_ids is None:
            # Fallback to random augmentation
            aug_prob = self.aug_config.get('augmentation_probability', 0.3)
            return [random.random() < aug_prob for _ in range(len(self.X))]
        
        augmentation_mask = [False] * len(self.X)
        aug_prob = self.aug_config.get('augmentation_probability', 0.3)
        
        # Group samples by subject
        unique_subjects = np.unique(self.subject_ids)
        print(f"Creating subject-wise augmentation for {len(unique_subjects)} subjects")
        
        for subject in unique_subjects:
            # Find all samples for this subject
            subject_indices = np.where(self.subject_ids == subject)[0]
            n_subject_samples = len(subject_indices)
            
            # Calculate how many to augment for this subject
            n_to_augment = int(n_subject_samples * aug_prob)
            
            # Randomly select which samples to augment for this subject
            augment_indices = np.random.choice(subject_indices, n_to_augment, replace=False)
            
            # Set augmentation mask
            for idx in augment_indices:
                augmentation_mask[idx] = True
                
            print(f"  Subject {subject}: {n_to_augment}/{n_subject_samples} samples augmented")
        
        return augmentation_mask
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # Shape: (num_electrodes, 4)
        y = self.y[idx]
        
        # Apply augmentations only during training
        if self.apply_augmentations:
            # Convert to numpy for augmentation
            x_np = x.numpy()
            
            # Use subject-aware augmentation mask
            should_augment = self.augmentation_mask[idx] if hasattr(self, 'augmentation_mask') else None
            
            # Apply augmentations based on config and mask
            x_aug_np = self.augmenter.custom_augmentation(x_np, self.aug_config, should_augment=should_augment)
            
            # Convert back to tensor
            x = torch.from_numpy(x_aug_np).float()
        
        return x, y
    
    def disable_augmentations(self):
        """Disable augmentations (useful for validation)"""
        self.apply_augmentations = False
    
    def enable_augmentations(self):
        """Enable augmentations (useful for training)"""
        self.apply_augmentations = True


class MixupAugmentedDataset(Dataset):
    """
    Advanced dataset with cross-subject mixup capabilities.
    Requires access to multiple subjects' data for mixup.
    """
    
    def __init__(self, X, y, subject_ids, num_electrodes=14,
                 apply_augmentations=True,
                 augmentation_config='moderate',
                 enable_mixup=True,
                 mixup_alpha=0.2,
                 seed=None):
        # Base setup with proper tensor reshaping
        X_reshaped = X.reshape(-1, num_electrodes, 4)
        if len(X_reshaped.shape) == 4:  # If there's an extra dimension
            X_reshaped = X_reshaped.squeeze(1)
            
        self.X = torch.tensor(X_reshaped, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = np.array(subject_ids)  # Ensure it's numpy array
        
        # Debug info
        print(f"MixupAugmentedDataset: X shape {X.shape} -> {self.X.shape}")
        print(f"Y shape: {y.shape} -> {self.y.shape}")
        print(f"Subject IDs shape: {len(subject_ids)} -> {self.subject_ids.shape}")
        
        # Verify lengths match
        assert len(self.X) == len(self.y) == len(self.subject_ids), \
            f"Length mismatch: X={len(self.X)}, y={len(self.y)}, subjects={len(self.subject_ids)}"
        
        # Augmentation setup
        self.apply_augmentations = apply_augmentations
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
        self.augmenter = EEGAugmentationSuite(seed=seed)
        
        # Configuration
        if isinstance(augmentation_config, str):
            from augmentations import AUGMENTATION_CONFIGS
            self.aug_config = AUGMENTATION_CONFIGS.get(augmentation_config, 
                                                      AUGMENTATION_CONFIGS['moderate'])
        else:
            self.aug_config = augmentation_config
        
        # Create subject-wise sample cache for mixup
        self.subject_samples = {}
        unique_subjects = np.unique(self.subject_ids)
        print(f"Creating mixup cache for {len(unique_subjects)} subjects")
        
        for subj in unique_subjects:
            subj_indices = np.where(self.subject_ids == subj)[0]
            self.subject_samples[subj] = subj_indices
            print(f"  Subject {subj}: {len(subj_indices)} samples")
        
        # Create subject-wise augmentation mask (same logic as AugmentedDatasetReshape)
        self.augmentation_mask = self._create_subject_augmentation_mask()
        if apply_augmentations:
            aug_count = sum(self.augmentation_mask)
            print(f"Subject-aware augmentation: {aug_count}/{len(self)} samples ({aug_count/len(self)*100:.1f}%)")
    
    def _create_subject_augmentation_mask(self):
        """Create boolean mask for which samples should be augmented (subject-aware)."""
        augmentation_mask = [False] * len(self.X)
        aug_prob = self.aug_config.get('augmentation_probability', 0.3)
        
        # Group samples by subject
        unique_subjects = np.unique(self.subject_ids)
        
        for subject in unique_subjects:
            # Find all samples for this subject
            subject_indices = np.where(self.subject_ids == subject)[0]
            n_subject_samples = len(subject_indices)
            
            # Calculate how many to augment for this subject
            n_to_augment = int(n_subject_samples * aug_prob)
            
            # Randomly select which samples to augment for this subject
            if n_to_augment > 0:
                augment_indices = np.random.choice(subject_indices, n_to_augment, replace=False)
                
                # Set augmentation mask
                for idx in augment_indices:
                    augmentation_mask[idx] = True
        
        return augmentation_mask
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        current_subject = self.subject_ids[idx]
        
        if self.apply_augmentations:
            x_np = x.numpy()
            
            # Use subject-aware augmentation mask
            should_augment = self.augmentation_mask[idx]
            
            # Apply standard augmentations if this sample should be augmented
            if should_augment:
                x_aug_np = self.augmenter.custom_augmentation(x_np, self.aug_config, should_augment=True)
                
                # Apply mixup with different subject if enabled
                if self.enable_mixup and len(self.subject_samples) > 1:
                    # Choose different subject
                    other_subjects = [s for s in self.subject_samples.keys() if s != current_subject]
                    if other_subjects:
                        other_subject = np.random.choice(other_subjects)
                        other_idx = np.random.choice(self.subject_samples[other_subject])
                        other_x = self.X[other_idx].numpy()
                        
                        # Apply mixup
                        x_mixed, lam = self.augmenter.mixup(x_aug_np, other_x, self.mixup_alpha)
                        x_aug_np = x_mixed
                        
                        # Note: For simplicity, keeping original label
                        # Advanced: could implement soft labels for mixup
                
                x = torch.from_numpy(x_aug_np).float()
            else:
                # No augmentation for this sample
                x = torch.from_numpy(x_np).float()
        
        return x, y
    
    def disable_augmentations(self):
        self.apply_augmentations = False
        self.enable_mixup = False
    
    def enable_augmentations(self):
        self.apply_augmentations = True
        self.enable_mixup = True
