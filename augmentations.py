import torch
import numpy as np
import random
from typing import Optional, Tuple, Dict, List, Union
from scipy import signal
from scipy.ndimage import gaussian_filter1d


class EEGAugmentationSuite:
    """
    Comprehensive EEG augmentation suite for BDE features.
    Each augmentation method can be used independently or combined.
    
    Input: BDE features with shape (num_electrodes, bde_dim=4) 
    where bde_dim represents [theta, alpha, beta, gamma] bands
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    # ========================================================================
    # AMPLITUDE DOMAIN AUGMENTATIONS
    # ========================================================================
    
    def gaussian_noise(self, x: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
        """
        Add Gaussian noise to simulate electrode noise and measurement uncertainty.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            noise_std: Standard deviation of Gaussian noise
        Returns:
            Noisy BDE features
        """
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    def amplitude_scaling(self, x: np.ndarray, scale_range: Tuple[float, float] = (0.85, 1.15)) -> np.ndarray:
        """
        Scale amplitude per electrode to simulate contact impedance variations.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            scale_range: (min_scale, max_scale) for random scaling
        Returns:
            Amplitude-scaled features
        """
        num_electrodes = x.shape[0]
        scales = np.random.uniform(scale_range[0], scale_range[1], (num_electrodes, 1))
        return x * scales
    
    def per_channel_amplitude_jitter(self, x: np.ndarray, jitter_std: float = 0.1) -> np.ndarray:
        """
        Add per-channel amplitude jitter with different scales per electrode.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            jitter_std: Standard deviation for jitter scaling
        Returns:
            Jittered features
        """
        num_electrodes = x.shape[0]
        jitter = 1 + np.random.normal(0, jitter_std, (num_electrodes, 1))
        jitter = np.clip(jitter, 0.5, 2.0)  # Prevent extreme values
        return x * jitter
    
    def power_normalization_shift(self, x: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """
        Shift power normalization baseline to simulate different recording conditions.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            shift_factor: Factor for baseline shift
        Returns:
            Baseline-shifted features
        """
        baseline_shift = np.random.uniform(-shift_factor, shift_factor)
        return x + baseline_shift
    
    # ========================================================================
    # SPATIAL DOMAIN AUGMENTATIONS
    # ========================================================================
    
    def channel_dropout(self, x: np.ndarray, dropout_prob: float = 0.15) -> np.ndarray:
        """
        Randomly zero out electrodes to simulate bad electrode contacts.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            dropout_prob: Probability of dropping each electrode
        Returns:
            Features with dropped channels
        """
        num_electrodes = x.shape[0]
        dropout_mask = np.random.random(num_electrodes) > dropout_prob
        return x * dropout_mask.reshape(-1, 1)
    
    def channel_permutation(self, x: np.ndarray, swap_prob: float = 0.1) -> np.ndarray:
        """
        Randomly swap electrode pairs to simulate slight electrode displacement.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            swap_prob: Probability of swapping electrode pairs
        Returns:
            Features with permuted channels
        """
        x_perm = x.copy()
        num_electrodes = x.shape[0]
        
        for i in range(0, num_electrodes - 1, 2):
            if random.random() < swap_prob:
                # Swap adjacent electrodes
                x_perm[i], x_perm[i + 1] = x_perm[i + 1].copy(), x_perm[i].copy()
        
        return x_perm
    
    def spatial_smoothing(self, x: np.ndarray, smooth_prob: float = 0.3, 
                         smooth_weight: float = 0.15) -> np.ndarray:
        """
        Apply spatial smoothing by averaging with neighboring electrodes.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            smooth_prob: Probability of smoothing each electrode
            smooth_weight: Weight for neighbor contribution
        Returns:
            Spatially smoothed features
        """
        x_smooth = x.copy()
        num_electrodes = x.shape[0]
        
        for i in range(num_electrodes):
            if random.random() < smooth_prob:
                # Choose random neighbor(s)
                neighbors = []
                if i > 0:
                    neighbors.append(i - 1)
                if i < num_electrodes - 1:
                    neighbors.append(i + 1)
                
                if neighbors:
                    neighbor_idx = random.choice(neighbors)
                    x_smooth[i] = (1 - smooth_weight) * x[i] + smooth_weight * x[neighbor_idx]
        
        return x_smooth
    
    def electrode_noise_correlation(self, x: np.ndarray, corr_strength: float = 0.1) -> np.ndarray:
        """
        Add spatially correlated noise to simulate common-mode interference.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            corr_strength: Strength of spatial correlation
        Returns:
            Features with correlated noise
        """
        num_electrodes, bde_dim = x.shape
        
        # Generate common noise component
        common_noise = np.random.normal(0, corr_strength, bde_dim)
        
        # Add correlated noise to all channels
        correlated_noise = np.tile(common_noise, (num_electrodes, 1))
        
        # Add some independent noise
        independent_noise = np.random.normal(0, corr_strength * 0.5, x.shape)
        
        return x + correlated_noise + independent_noise
    
    # ========================================================================
    # FREQUENCY DOMAIN AUGMENTATIONS
    # ========================================================================
    
    def frequency_band_dropout(self, x: np.ndarray, band_dropout_prob: float = 0.1) -> np.ndarray:
        """
        Randomly zero out entire frequency bands.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            band_dropout_prob: Probability of dropping each band
        Returns:
            Features with dropped frequency bands
        """
        x_dropped = x.copy()
        bde_dim = x.shape[1]
        
        for band in range(bde_dim):
            if random.random() < band_dropout_prob:
                x_dropped[:, band] = 0
        
        return x_dropped
    
    def frequency_band_scaling(self, x: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """
        Apply different scaling to each frequency band.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            scale_range: (min_scale, max_scale) for each band
        Returns:
            Band-scaled features
        """
        bde_dim = x.shape[1]
        band_scales = np.random.uniform(scale_range[0], scale_range[1], bde_dim)
        return x * band_scales
    
    def frequency_band_noise(self, x: np.ndarray, band_noise_std: float = 0.02) -> np.ndarray:
        """
        Add different noise levels to each frequency band.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            band_noise_std: Standard deviation for band-specific noise
        Returns:
            Features with band-specific noise
        """
        bde_dim = x.shape[1]
        band_noise_levels = np.random.uniform(0.5, 1.5, bde_dim) * band_noise_std
        
        noise = np.random.normal(0, 1, x.shape)
        noise = noise * band_noise_levels.reshape(1, -1)
        
        return x + noise
    
    def frequency_band_shift(self, x: np.ndarray, shift_std: float = 0.05) -> np.ndarray:
        """
        Simulate frequency band boundary shifts by adding systematic offsets.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            shift_std: Standard deviation of systematic shift
        Returns:
            Band-shifted features
        """
        bde_dim = x.shape[1]
        band_shifts = np.random.normal(0, shift_std, bde_dim)
        return x + band_shifts.reshape(1, -1)
    
    # ========================================================================
    # TEMPORAL DOMAIN AUGMENTATIONS
    # ========================================================================
    
    def temporal_jitter(self, x: np.ndarray, jitter_std: float = 0.01) -> np.ndarray:
        """
        Add temporal jitter by slightly shifting feature values.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            jitter_std: Standard deviation of temporal jitter
        Returns:
            Temporally jittered features
        """
        jitter = np.random.normal(0, jitter_std, x.shape)
        return x + jitter
    
    def temporal_masking(self, x: np.ndarray, mask_prob: float = 0.1, 
                        mask_value: float = 0.0) -> np.ndarray:
        """
        Randomly mask some feature values to simulate temporal dropouts.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            mask_prob: Probability of masking each feature
            mask_value: Value to use for masking
        Returns:
            Temporally masked features
        """
        mask = np.random.random(x.shape) > mask_prob
        return x * mask + mask_value * (1 - mask)
    
    # ========================================================================
    # CROSS-SUBJECT DOMAIN AUGMENTATIONS
    # ========================================================================
    
    def mixup(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        MixUp augmentation between two samples (potentially from different subjects).
        
        Args:
            x1: First sample (num_electrodes, bde_dim)
            x2: Second sample (num_electrodes, bde_dim)
            alpha: Beta distribution parameter
        Returns:
            Mixed sample and mixing coefficient lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        mixed = lam * x1 + (1 - lam) * x2
        return mixed, lam
    
    def subject_style_transfer(self, x: np.ndarray, target_mean: np.ndarray, 
                              target_std: np.ndarray, transfer_strength: float = 0.3) -> np.ndarray:
        """
        Apply statistical normalization from another subject.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            target_mean: Target subject's mean statistics
            target_std: Target subject's std statistics
            transfer_strength: Strength of style transfer (0-1)
        Returns:
            Style-transferred features
        """
        # Current statistics
        current_mean = np.mean(x, axis=0, keepdims=True)
        current_std = np.std(x, axis=0, keepdims=True) + 1e-8
        
        # Normalize to unit Gaussian
        normalized = (x - current_mean) / current_std
        
        # Apply target statistics
        target_features = normalized * target_std + target_mean
        
        # Blend with original
        return (1 - transfer_strength) * x + transfer_strength * target_features
    
    def cross_subject_noise(self, x: np.ndarray, other_samples: List[np.ndarray], 
                           noise_strength: float = 0.05) -> np.ndarray:
        """
        Add patterns from other subjects as structured noise.
        
        Args:
            x: Input BDE features (num_electrodes, bde_dim)
            other_samples: List of samples from other subjects
            noise_strength: Strength of cross-subject noise
        Returns:
            Features with cross-subject noise
        """
        if not other_samples:
            return x
        
        # Select random sample from other subjects
        noise_source = random.choice(other_samples)
        
        # Add as structured noise
        return x + noise_strength * noise_source
    
    # ========================================================================
    # COMPOSITE AUGMENTATION METHODS
    # ========================================================================
    
    def light_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Conservative augmentation preset."""
        x = self.gaussian_noise(x, noise_std=0.01)
        x = self.amplitude_scaling(x, scale_range=(0.95, 1.05))
        x = self.channel_dropout(x, dropout_prob=0.05)
        return x
    
    def moderate_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Moderate augmentation preset."""
        x = self.gaussian_noise(x, noise_std=0.02)
        x = self.amplitude_scaling(x, scale_range=(0.9, 1.1))
        x = self.channel_dropout(x, dropout_prob=0.15)
        x = self.frequency_band_scaling(x, scale_range=(0.8, 1.2))
        x = self.spatial_smoothing(x, smooth_prob=0.2)
        return x
    
    def heavy_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Aggressive augmentation preset."""
        x = self.gaussian_noise(x, noise_std=0.05)
        x = self.amplitude_scaling(x, scale_range=(0.8, 1.2))
        x = self.channel_dropout(x, dropout_prob=0.25)
        x = self.frequency_band_dropout(x, band_dropout_prob=0.2)
        x = self.frequency_band_scaling(x, scale_range=(0.7, 1.3))
        x = self.spatial_smoothing(x, smooth_prob=0.3)
        x = self.electrode_noise_correlation(x, corr_strength=0.1)
        return x
    
    def custom_augmentation(self, x: np.ndarray, config: Dict[str, any]) -> np.ndarray:
        """
        Apply custom augmentation based on configuration dictionary.
        
        Args:
            x: Input BDE features
            config: Dictionary specifying which augmentations to apply and their parameters
        Returns:
            Augmented features
        """
        if config.get('gaussian_noise', False):
            x = self.gaussian_noise(x, **config.get('gaussian_noise_params', {}))
        
        if config.get('amplitude_scaling', False):
            x = self.amplitude_scaling(x, **config.get('amplitude_scaling_params', {}))
        
        if config.get('channel_dropout', False):
            x = self.channel_dropout(x, **config.get('channel_dropout_params', {}))
        
        if config.get('frequency_band_scaling', False):
            x = self.frequency_band_scaling(x, **config.get('frequency_band_scaling_params', {}))
        
        if config.get('spatial_smoothing', False):
            x = self.spatial_smoothing(x, **config.get('spatial_smoothing_params', {}))
        
        # Add more augmentations as specified in config
        
        return x
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_available_augmentations(self) -> List[str]:
        """Return list of all available augmentation methods."""
        return [
            # Amplitude domain
            'gaussian_noise', 'amplitude_scaling', 'per_channel_amplitude_jitter', 
            'power_normalization_shift',
            # Spatial domain
            'channel_dropout', 'channel_permutation', 'spatial_smoothing', 
            'electrode_noise_correlation',
            # Frequency domain
            'frequency_band_dropout', 'frequency_band_scaling', 'frequency_band_noise', 
            'frequency_band_shift',
            # Temporal domain
            'temporal_jitter', 'temporal_masking',
            # Cross-subject domain
            'mixup', 'subject_style_transfer', 'cross_subject_noise',
            # Composite
            'light_augmentation', 'moderate_augmentation', 'heavy_augmentation'
        ]
    
    def visualize_augmentation(self, original: np.ndarray, augmented: np.ndarray, 
                              title: str = "Augmentation Comparison") -> None:
        """
        Visualize the effect of augmentation (requires matplotlib).
        
        Args:
            original: Original BDE features
            augmented: Augmented BDE features
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(title)
            
            # Plot original
            im1 = axes[0, 0].imshow(original.T, aspect='auto', cmap='viridis')
            axes[0, 0].set_title('Original')
            axes[0, 0].set_ylabel('Frequency Bands')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Plot augmented
            im2 = axes[0, 1].imshow(augmented.T, aspect='auto', cmap='viridis')
            axes[0, 1].set_title('Augmented')
            plt.colorbar(im2, ax=axes[0, 1])
            
            # Plot difference
            diff = augmented - original
            im3 = axes[1, 0].imshow(diff.T, aspect='auto', cmap='RdBu_r', 
                                   vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
            axes[1, 0].set_title('Difference')
            axes[1, 0].set_xlabel('Electrodes')
            axes[1, 0].set_ylabel('Frequency Bands')
            plt.colorbar(im3, ax=axes[1, 0])
            
            # Plot statistics comparison
            orig_mean = np.mean(original, axis=0)
            aug_mean = np.mean(augmented, axis=0)
            
            axes[1, 1].bar(range(len(orig_mean)), orig_mean, alpha=0.7, label='Original')
            axes[1, 1].bar(range(len(aug_mean)), aug_mean, alpha=0.7, label='Augmented')
            axes[1, 1].set_title('Mean per Band')
            axes[1, 1].set_xlabel('Frequency Bands')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


# Example usage configurations
AUGMENTATION_CONFIGS = {
    'conservative': {
        'gaussian_noise': True,
        'gaussian_noise_params': {'noise_std': 0.01},
        'channel_dropout': True,
        'channel_dropout_params': {'dropout_prob': 0.05},
        'amplitude_scaling': True,
        'amplitude_scaling_params': {'scale_range': (0.95, 1.05)}
    },
    
    'moderate': {
        'gaussian_noise': True,
        'gaussian_noise_params': {'noise_std': 0.02},
        'channel_dropout': True,
        'channel_dropout_params': {'dropout_prob': 0.15},
        'amplitude_scaling': True,
        'amplitude_scaling_params': {'scale_range': (0.9, 1.1)},
        'frequency_band_scaling': True,
        'frequency_band_scaling_params': {'scale_range': (0.8, 1.2)},
        'spatial_smoothing': True,
        'spatial_smoothing_params': {'smooth_prob': 0.2}
    },
    
    'aggressive': {
        'gaussian_noise': True,
        'gaussian_noise_params': {'noise_std': 0.05},
        'channel_dropout': True,
        'channel_dropout_params': {'dropout_prob': 0.25},
        'amplitude_scaling': True,
        'amplitude_scaling_params': {'scale_range': (0.8, 1.2)},
        'frequency_band_dropout': True,
        'frequency_band_dropout_params': {'band_dropout_prob': 0.2},
        'frequency_band_scaling': True,
        'frequency_band_scaling_params': {'scale_range': (0.7, 1.3)},
        'spatial_smoothing': True,
        'spatial_smoothing_params': {'smooth_prob': 0.3},
        'electrode_noise_correlation': True,
        'electrode_noise_correlation_params': {'corr_strength': 0.1}
    }
}
