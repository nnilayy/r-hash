import numpy as np
import random
from typing import Optional, Tuple, Dict


class EEGAugmentationSuite:
    """
    Clean BDE augmentation suite - only functions that work for (num_electrodes, 4) tensors.
    Removed all temporal, spatial, and other useless functions.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    # Core BDE augmentations only
    def gaussian_noise(self, x: np.ndarray, noise_std: float = 0.051502) -> np.ndarray:
        """Add realistic Gaussian noise based on universal BDE analysis."""
        noise = np.random.normal(0, noise_std, x.shape)
        return x + noise
    
    def amplitude_scaling(self, x: np.ndarray, scale_range: Tuple[float, float] = (0.92, 1.08)) -> np.ndarray:
        """Scale amplitude uniformly across all features."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return x * scale_factor
    
    def frequency_band_scaling(self, x: np.ndarray, scale_range: Tuple[float, float] = (0.88, 1.12)) -> np.ndarray:
        """Apply different scaling to each frequency band (theta, alpha, beta, gamma)."""
        band_scales = np.random.uniform(scale_range[0], scale_range[1], 4)  # 4 bands
        return x * band_scales.reshape(1, -1)
    
    def frequency_band_noise(self, x: np.ndarray, band_noise_std: float = 0.06) -> np.ndarray:
        """Add different noise levels to each frequency band."""
        band_noise_levels = np.random.uniform(0.5, 1.5, 4) * band_noise_std
        noise = np.random.normal(0, 1, x.shape) * band_noise_levels.reshape(1, -1)
        return x + noise
    
    def frequency_band_dropout(self, x: np.ndarray, band_dropout_prob: float = 0.05) -> np.ndarray:
        """Zero out frequency bands occasionally."""
        x_dropped = x.copy()
        for band in range(4):  # 4 bands
            if random.random() < band_dropout_prob:
                x_dropped[:, band] = 0
        return x_dropped
    
    def electrode_noise_correlation(self, x: np.ndarray, corr_strength: float = 0.08) -> np.ndarray:
        """Add correlated noise between adjacent electrodes."""
        num_electrodes = x.shape[0]
        # Simple adjacent electrode correlation
        for i in range(num_electrodes - 1):
            if random.random() < 0.3:  # 30% chance of correlation
                correlation_noise = np.random.normal(0, corr_strength, 4)
                x[i] += correlation_noise
                x[i + 1] += correlation_noise * 0.5
        return x
    
    def mixup(self, x1: np.ndarray, x2: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        MixUp augmentation between two samples.
        
        Args:
            x1: First sample (num_electrodes, 4)
            x2: Second sample (num_electrodes, 4)
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
    
    def custom_augmentation(self, x: np.ndarray, config: Dict[str, any]) -> np.ndarray:
        """
        Apply augmentations based on configuration dictionary.
        
        Args:
            x: Input BDE features (num_electrodes, 4)
            config: Dictionary specifying which augmentations to apply
        Returns:
            Augmented features
        """
        # Check augmentation probability - only augment some samples
        aug_prob = config.get('augmentation_probability', 1.0)  # Default: augment all
        if random.random() > aug_prob:
            return x  # Skip augmentation for this sample
        
        if config.get('gaussian_noise', False):
            x = self.gaussian_noise(x, **config.get('gaussian_noise_params', {}))
        
        if config.get('amplitude_scaling', False):
            x = self.amplitude_scaling(x, **config.get('amplitude_scaling_params', {}))
        
        if config.get('frequency_band_scaling', False):
            x = self.frequency_band_scaling(x, **config.get('frequency_band_scaling_params', {}))
        
        if config.get('frequency_band_noise', False):
            x = self.frequency_band_noise(x, **config.get('frequency_band_noise_params', {}))
        
        if config.get('frequency_band_dropout', False):
            x = self.frequency_band_dropout(x, **config.get('frequency_band_dropout_params', {}))
        
        if config.get('electrode_noise_correlation', False):
            x = self.electrode_noise_correlation(x, **config.get('electrode_noise_correlation_params', {}))
        
        return x


# Clean configurations based on comprehensive BDE analysis (1.4M samples)
# Universal std: ~1.0, Absolute range: (-11.0, 7.8), Safe range: (-8.8, 6.3)
AUGMENTATION_CONFIGS = {
    'conservative': {
        # Conservative: 20% of natural variation for stable training
        'augmentation_probability': 0.3,  # Apply to only 30% of samples
        'gaussian_noise': True,
        'gaussian_noise_params': {'noise_std': 0.20},  # 20% of natural std
        'amplitude_scaling': True,
        'amplitude_scaling_params': {'scale_range': (0.80, 1.20)}  # ±20% scaling
    },
    
    'moderate': {
        # Moderate: 50% of natural variation for cross-dataset generalization
        'augmentation_probability': 0.5,  # Apply to 50% of samples (balanced)
        'gaussian_noise': True,
        'gaussian_noise_params': {'noise_std': 0.50},  # 50% of natural std
        'frequency_band_noise': True,
        'frequency_band_noise_params': {'band_noise_std': 0.60},  # Band-specific noise
        'amplitude_scaling': True,
        'amplitude_scaling_params': {'scale_range': (0.65, 1.35)},  # ±35% scaling
        'frequency_band_scaling': True,
        'frequency_band_scaling_params': {'scale_range': (0.70, 1.30)}  # ±30% per band
    },
    
    'aggressive': {
        # Aggressive: Full natural variation - bridges all dataset gaps
        # Can transform DEAP (-4.2,4.9) ↔ SEED (-10.9,4.8) ↔ DREAMER (-7.1,7.8)
        'augmentation_probability': 0.4,  # Apply to only 40% - keep 60% original
        'gaussian_noise': True,
        'gaussian_noise_params': {'noise_std': 1.00},  # Full natural std variation
        'frequency_band_noise': True,
        'frequency_band_noise_params': {'band_noise_std': 0.80},  # Strong band noise
        'amplitude_scaling': True,
        'amplitude_scaling_params': {'scale_range': (0.50, 1.50)},  # ±50% scaling
        'frequency_band_scaling': True,
        'frequency_band_scaling_params': {'scale_range': (0.60, 1.40)},  # ±40% per band
        'frequency_band_dropout': True,
        'frequency_band_dropout_params': {'band_dropout_prob': 0.15},  # 15% dropout
        'electrode_noise_correlation': True,
        'electrode_noise_correlation_params': {'corr_strength': 0.30}  # 30% correlation
    }
}
