import torch
import numpy as np
from torch.utils.data import Dataset
from augmentations import EEGAugmentationSuite


class AugmentedDatasetReshape(Dataset):
    """
    Enhanced DatasetReshape with on-the-fly EEG augmentations.
    Applies augmentations during training while keeping validation data clean.
    """
    
    def __init__(self, X, y, num_electrodes=14, 
                 apply_augmentations=True, 
                 augmentation_config='moderate',
                 seed=None):
        """
        Args:
            X: Flattened BDE features
            y: Labels 
            num_electrodes: Number of electrodes per sample
            apply_augmentations: Whether to apply augmentations
            augmentation_config: 'conservative', 'moderate', 'aggressive', or custom dict
            seed: Random seed for augmentations
        """
        # Reshape data same as original DatasetReshape
        self.X = torch.tensor(
            X.reshape(-1, num_electrodes, 4), dtype=torch.float32
        ).squeeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        
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
            
        print(f"Dataset initialized with augmentations: {apply_augmentations}")
        if apply_augmentations:
            print(f"Augmentation config: {list(self.aug_config.keys())}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]  # Shape: (num_electrodes, 4)
        y = self.y[idx]
        
        # Apply augmentations only during training
        if self.apply_augmentations:
            # Convert to numpy for augmentation
            x_np = x.numpy()
            
            # Apply augmentations based on config
            x_aug_np = self.augmenter.custom_augmentation(x_np, self.aug_config)
            
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
        # Base setup
        self.X = torch.tensor(X.reshape(-1, num_electrodes, 4), dtype=torch.float32).squeeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subject_ids = subject_ids
        
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
        unique_subjects = np.unique(subject_ids)
        for subj in unique_subjects:
            subj_indices = np.where(subject_ids == subj)[0]
            self.subject_samples[subj] = subj_indices
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        current_subject = self.subject_ids[idx]
        
        if self.apply_augmentations:
            x_np = x.numpy()
            
            # Apply standard augmentations
            x_aug_np = self.augmenter.custom_augmentation(x_np, self.aug_config)
            
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
        
        return x, y
    
    def disable_augmentations(self):
        self.apply_augmentations = False
        self.enable_mixup = False
    
    def enable_augmentations(self):
        self.apply_augmentations = True
        self.enable_mixup = True
