import torch
import numpy as np
import random
from typing import Optional, Tuple, Dict, List


class EEGAugmentations:
    """
    EEG-specific augmentations for BDE features to improve LOSO generalization.
    Works with Band Differential Entropy (BDE) features from transformer input.
    
    Input shape: (num_electrodes, bde_dim) where bde_dim=4 (theta, alpha, beta, gamma)
    """
    
    def __init__(
        self,
        noise_std: float = 0.02,
        amplitude_range: Tuple[float, float] = (0.9, 1.1),
        channel_dropout_prob: float = 0.15,
        temporal_mask_prob: float = 0.1,
        band_dropout_prob: float = 0.1,
        mixup_alpha: float = 0.2,
        apply_prob: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Args:
            noise_std: Standard deviation for Gaussian noise injection
            amplitude_range: (min, max) scaling factors for amplitude variation
            channel_dropout_prob: Probability of zeroing each channel
            temporal_mask_prob: Probability of masking temporal segments
            band_dropout_prob: Probability of zeroing each frequency band
            mixup_alpha: Alpha parameter for beta distribution in mixup
            apply_prob: Probability of applying augmentations
            seed: Random seed for reproducibility
        """
        self.noise_std = noise_std
        self.amplitude_range = amplitude_range
        self.channel_dropout_prob = channel_dropout_prob
        self.temporal_mask_prob = temporal_mask_prob
        self.band_dropout_prob = band_dropout_prob
        self.mixup_alpha = mixup_alpha
        self.apply_prob = apply_prob
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __call__(self, x: torch.Tensor, label: Optional[torch.Tensor] = None, 
                 mixup_batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations to EEG BDE features.
        
        Args:
            x: Input tensor (num_electrodes, bde_dim)
            label: Optional label for mixup
            mixup_batch: Optional batch for cross-subject mixup
            
        Returns:
            Augmented tensor and potentially mixed label
        """
        if random.random() > self.apply_prob:
            return x, label
            
        # Convert to numpy for easier manipulation
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
            is_tensor = True
        else:
            x_np = x
            is_tensor = False
            
        # Apply augmentations
        x_aug = self._apply_augmentations(x_np)
        
        # Apply cross-subject mixup if available
        if mixup_batch is not None and label is not None:
            x_aug, label = self._apply_mixup(x_aug, label, mixup_batch)
        
        # Convert back to tensor if needed
        if is_tensor:
            x_aug = torch.from_numpy(x_aug).float()
            
        return x_aug, label
    
    def _apply_augmentations(self, x: np.ndarray) -> np.ndarray:
        """Apply core EEG augmentations to BDE features."""
        x_aug = x.copy()
        
        # 1. Gaussian noise injection (simulate electrode noise)
        if random.random() < 0.7:
            noise = np.random.normal(0, self.noise_std, x_aug.shape)
            x_aug = x_aug + noise
        
        # 2. Amplitude scaling per channel (electrode contact variation)
        if random.random() < 0.6:
            scales = np.random.uniform(
                self.amplitude_range[0], 
                self.amplitude_range[1], 
                (x_aug.shape[0], 1)
            )
            x_aug = x_aug * scales
        
        # 3. Channel dropout (simulate bad electrodes)
        if random.random() < 0.5:
            num_channels = x_aug.shape[0]
            dropout_mask = np.random.random(num_channels) > self.channel_dropout_prob
            x_aug = x_aug * dropout_mask.reshape(-1, 1)
        
        # 4. Frequency band dropout (simulate band-specific artifacts)
        if random.random() < 0.4:
            num_bands = x_aug.shape[1]
            for band in range(num_bands):
                if random.random() < self.band_dropout_prob:
                    x_aug[:, band] = 0
        
        # 5. Per-band amplitude scaling (frequency-specific variations)
        if random.random() < 0.5:
            band_scales = np.random.uniform(0.8, 1.2, x_aug.shape[1])
            x_aug = x_aug * band_scales
        
        # 6. Spatial smoothing (neighboring electrode averaging)
        if random.random() < 0.3:
            x_aug = self._apply_spatial_smoothing(x_aug)
            
        return x_aug
    
    def _apply_spatial_smoothing(self, x: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing between neighboring channels."""
        # Simple smoothing: each channel gets 10% contribution from random neighbor
        x_smooth = x.copy()
        num_channels = x.shape[0]
        
        for i in range(num_channels):
            if random.random() < 0.3:  # Only smooth some channels
                neighbor = random.randint(0, num_channels - 1)
                x_smooth[i] = 0.9 * x[i] + 0.1 * x[neighbor]
        
        return x_smooth
    
    def _apply_mixup(self, x: np.ndarray, label: torch.Tensor, 
                     mixup_batch: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Apply cross-subject mixup for domain generalization."""
        if mixup_batch.size(0) == 0:
            return x, label
            
        # Sample random example from batch
        batch_size = mixup_batch.size(0)
        mix_idx = random.randint(0, batch_size - 1)
        mix_sample = mixup_batch[mix_idx].numpy()
        
        # Sample mixing coefficient from beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Mix features
        x_mixed = lam * x + (1 - lam) * mix_sample
        
        # Create mixed label (for cross-entropy, we'd need soft labels)
        # For now, keep original label but could extend to soft mixing
        mixed_label = label
        
        return x_mixed, mixed_label
    
    def get_augmentation_config(self) -> Dict:
        """Return current augmentation configuration."""
        return {
            'noise_std': self.noise_std,
            'amplitude_range': self.amplitude_range,
            'channel_dropout_prob': self.channel_dropout_prob,
            'temporal_mask_prob': self.temporal_mask_prob,
            'band_dropout_prob': self.band_dropout_prob,
            'mixup_alpha': self.mixup_alpha,
            'apply_prob': self.apply_prob,
        }


class AugmentedDatasetWrapper:
    """
    Wrapper to apply augmentations to existing dataset during training.
    """
    
    def __init__(self, dataset, augmentations: EEGAugmentations, enable_mixup: bool = True):
        self.dataset = dataset
        self.augmentations = augmentations
        self.enable_mixup = enable_mixup
        self._mixup_cache = []
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        # Prepare mixup batch if enabled
        mixup_batch = None
        if self.enable_mixup and len(self._mixup_cache) > 10:
            # Use cached samples from different subjects
            mixup_batch = torch.stack(random.sample(self._mixup_cache, 
                                                   min(5, len(self._mixup_cache))))
        
        # Apply augmentations
        x_aug, y_aug = self.augmentations(x, y, mixup_batch)
        
        # Update mixup cache (keep recent samples)
        if len(self._mixup_cache) > 100:
            self._mixup_cache = self._mixup_cache[-50:]
        self._mixup_cache.append(x.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x.copy()))
        
        return x_aug, y_aug if y_aug is not None else y


# Usage configuration presets
CONSERVATIVE_CONFIG = {
    'noise_std': 0.01,
    'amplitude_range': (0.95, 1.05),
    'channel_dropout_prob': 0.1,
    'band_dropout_prob': 0.05,
    'apply_prob': 0.6,
}

MODERATE_CONFIG = {
    'noise_std': 0.02,
    'amplitude_range': (0.9, 1.1), 
    'channel_dropout_prob': 0.15,
    'band_dropout_prob': 0.1,
    'apply_prob': 0.8,
}

AGGRESSIVE_CONFIG = {
    'noise_std': 0.05,
    'amplitude_range': (0.8, 1.2),
    'channel_dropout_prob': 0.25,
    'band_dropout_prob': 0.2,
    'apply_prob': 0.9,
}
