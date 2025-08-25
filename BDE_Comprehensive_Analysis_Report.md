# Comprehensive BDE Value Range Analysis
**Analysis Date:** 2025-08-25 08:46:23

This report analyzes BDE (Band Differential Entropy) value ranges across all 13 preprocessed datasets for RBTransformer.

## Dataset Overview

### 1. seed_multi_emotion_dataset.pkl

**Samples:** 258,345 | **Electrodes:** 62 | **Bands:** 4

**Global BDE Statistics:**
- Range: -10.947251 to 4.805040
- Mean: -1.482341 | Std: 1.569226
- Median: -1.258150
- Percentiles: P5=-4.5112, P95=0.4380

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -10.9473 | 4.8050 | -1.2123 | 1.4427 | -4.3062 | 0.2880 |
| Alpha | -10.7544 | 4.7084 | -1.5509 | 1.4534 | -4.5500 | 0.0563 |
| Beta | -10.2327 | 4.7018 | -1.3495 | 1.5675 | -4.2951 | 0.7511 |
| Gamma | -10.2936 | 4.3023 | -1.8167 | 1.7297 | -4.7689 | 0.7469 |

### 2. deap_multi_valence_dataset.pkl

**Samples:** 79,360 | **Electrodes:** 32 | **Bands:** 4

**Global BDE Statistics:**
- Range: -4.198148 to 4.912032
- Mean: 0.346027 | Std: 0.782404
- Median: 0.293900
- Percentiles: P5=-0.8647, P95=1.7165

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -3.7644 | 4.4134 | 0.3121 | 0.7595 | -0.9306 | 1.6012 |
| Alpha | -4.1981 | 4.4134 | 0.3223 | 0.6962 | -0.7648 | 1.5111 |
| Beta | -3.5828 | 4.3758 | 0.3404 | 0.7617 | -0.8045 | 1.6980 |
| Gamma | -3.9121 | 4.9120 | 0.4093 | 0.8950 | -0.9607 | 1.9938 |

### 3. deap_multi_arousal_dataset.pkl

**Samples:** 79,360 | **Electrodes:** 32 | **Bands:** 4

**Global BDE Statistics:**
- Range: -4.198148 to 4.912032
- Mean: 0.346027 | Std: 0.782404
- Median: 0.293900
- Percentiles: P5=-0.8647, P95=1.7165

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -3.7644 | 4.4134 | 0.3121 | 0.7595 | -0.9306 | 1.6012 |
| Alpha | -4.1981 | 4.4134 | 0.3223 | 0.6962 | -0.7648 | 1.5111 |
| Beta | -3.5828 | 4.3758 | 0.3404 | 0.7617 | -0.8045 | 1.6980 |
| Gamma | -3.9121 | 4.9120 | 0.4093 | 0.8950 | -0.9607 | 1.9938 |

### 4. deap_multi_dominance_dataset.pkl

**Samples:** 79,360 | **Electrodes:** 32 | **Bands:** 4

**Global BDE Statistics:**
- Range: -4.198148 to 4.912032
- Mean: 0.346027 | Std: 0.782404
- Median: 0.293900
- Percentiles: P5=-0.8647, P95=1.7165

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -3.7644 | 4.4134 | 0.3121 | 0.7595 | -0.9306 | 1.6012 |
| Alpha | -4.1981 | 4.4134 | 0.3223 | 0.6962 | -0.7648 | 1.5111 |
| Beta | -3.5828 | 4.3758 | 0.3404 | 0.7617 | -0.8045 | 1.6980 |
| Gamma | -3.9121 | 4.9120 | 0.4093 | 0.8950 | -0.9607 | 1.9938 |

### 5. deap_binary_valence_dataset.pkl

**Samples:** 79,360 | **Electrodes:** 32 | **Bands:** 4

**Global BDE Statistics:**
- Range: -4.198148 to 4.912032
- Mean: 0.346027 | Std: 0.782404
- Median: 0.293900
- Percentiles: P5=-0.8647, P95=1.7165

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -3.7644 | 4.4134 | 0.3121 | 0.7595 | -0.9306 | 1.6012 |
| Alpha | -4.1981 | 4.4134 | 0.3223 | 0.6962 | -0.7648 | 1.5111 |
| Beta | -3.5828 | 4.3758 | 0.3404 | 0.7617 | -0.8045 | 1.6980 |
| Gamma | -3.9121 | 4.9120 | 0.4093 | 0.8950 | -0.9607 | 1.9938 |

### 6. deap_binary_arousal_dataset.pkl

**Samples:** 79,360 | **Electrodes:** 32 | **Bands:** 4

**Global BDE Statistics:**
- Range: -4.198148 to 4.912032
- Mean: 0.346027 | Std: 0.782404
- Median: 0.293900
- Percentiles: P5=-0.8647, P95=1.7165

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -3.7644 | 4.4134 | 0.3121 | 0.7595 | -0.9306 | 1.6012 |
| Alpha | -4.1981 | 4.4134 | 0.3223 | 0.6962 | -0.7648 | 1.5111 |
| Beta | -3.5828 | 4.3758 | 0.3404 | 0.7617 | -0.8045 | 1.6980 |
| Gamma | -3.9121 | 4.9120 | 0.4093 | 0.8950 | -0.9607 | 1.9938 |

### 7. deap_binary_dominance_dataset.pkl

**Samples:** 79,360 | **Electrodes:** 32 | **Bands:** 4

**Global BDE Statistics:**
- Range: -4.198148 to 4.912032
- Mean: 0.346027 | Std: 0.782404
- Median: 0.293900
- Percentiles: P5=-0.8647, P95=1.7165

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -3.7644 | 4.4134 | 0.3121 | 0.7595 | -0.9306 | 1.6012 |
| Alpha | -4.1981 | 4.4134 | 0.3223 | 0.6962 | -0.7648 | 1.5111 |
| Beta | -3.5828 | 4.3758 | 0.3404 | 0.7617 | -0.8045 | 1.6980 |
| Gamma | -3.9121 | 4.9120 | 0.4093 | 0.8950 | -0.9607 | 1.9938 |

### 8. dreamer_multi_valence_dataset.pkl

**Samples:** 92,184 | **Electrodes:** 14 | **Bands:** 4

**Global BDE Statistics:**
- Range: -7.107270 to 7.846405
- Mean: -0.046077 | Std: 1.187829
- Median: -0.476108
- Percentiles: P5=-1.2111, P95=2.3731

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -7.1073 | 7.8464 | -0.3070 | 1.0793 | -1.3574 | 1.8743 |
| Alpha | -6.5167 | 7.5995 | -0.1714 | 1.0701 | -1.2796 | 2.0208 |
| Beta | -6.0853 | 7.1156 | 0.1775 | 1.2986 | -1.0775 | 2.8057 |
| Gamma | -6.0588 | 7.0851 | 0.1165 | 1.2195 | -1.0726 | 2.5686 |

### 9. dreamer_multi_arousal_dataset.pkl

**Samples:** 92,184 | **Electrodes:** 14 | **Bands:** 4

**Global BDE Statistics:**
- Range: -7.107270 to 7.846405
- Mean: -0.046077 | Std: 1.187829
- Median: -0.476108
- Percentiles: P5=-1.2111, P95=2.3731

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -7.1073 | 7.8464 | -0.3070 | 1.0793 | -1.3574 | 1.8743 |
| Alpha | -6.5167 | 7.5995 | -0.1714 | 1.0701 | -1.2796 | 2.0208 |
| Beta | -6.0853 | 7.1156 | 0.1775 | 1.2986 | -1.0775 | 2.8057 |
| Gamma | -6.0588 | 7.0851 | 0.1165 | 1.2195 | -1.0726 | 2.5686 |

### 10. dreamer_multi_dominance_dataset.pkl

**Samples:** 92,184 | **Electrodes:** 14 | **Bands:** 4

**Global BDE Statistics:**
- Range: -7.107270 to 7.846405
- Mean: -0.046077 | Std: 1.187829
- Median: -0.476108
- Percentiles: P5=-1.2111, P95=2.3731

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -7.1073 | 7.8464 | -0.3070 | 1.0793 | -1.3574 | 1.8743 |
| Alpha | -6.5167 | 7.5995 | -0.1714 | 1.0701 | -1.2796 | 2.0208 |
| Beta | -6.0853 | 7.1156 | 0.1775 | 1.2986 | -1.0775 | 2.8057 |
| Gamma | -6.0588 | 7.0851 | 0.1165 | 1.2195 | -1.0726 | 2.5686 |

### 11. dreamer_binary_valence_dataset.pkl

**Samples:** 92,184 | **Electrodes:** 14 | **Bands:** 4

**Global BDE Statistics:**
- Range: -7.107270 to 7.846405
- Mean: -0.046077 | Std: 1.187829
- Median: -0.476108
- Percentiles: P5=-1.2111, P95=2.3731

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -7.1073 | 7.8464 | -0.3070 | 1.0793 | -1.3574 | 1.8743 |
| Alpha | -6.5167 | 7.5995 | -0.1714 | 1.0701 | -1.2796 | 2.0208 |
| Beta | -6.0853 | 7.1156 | 0.1775 | 1.2986 | -1.0775 | 2.8057 |
| Gamma | -6.0588 | 7.0851 | 0.1165 | 1.2195 | -1.0726 | 2.5686 |

### 12. dreamer_binary_arousal_dataset.pkl

**Samples:** 92,184 | **Electrodes:** 14 | **Bands:** 4

**Global BDE Statistics:**
- Range: -7.107270 to 7.846405
- Mean: -0.046077 | Std: 1.187829
- Median: -0.476108
- Percentiles: P5=-1.2111, P95=2.3731

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -7.1073 | 7.8464 | -0.3070 | 1.0793 | -1.3574 | 1.8743 |
| Alpha | -6.5167 | 7.5995 | -0.1714 | 1.0701 | -1.2796 | 2.0208 |
| Beta | -6.0853 | 7.1156 | 0.1775 | 1.2986 | -1.0775 | 2.8057 |
| Gamma | -6.0588 | 7.0851 | 0.1165 | 1.2195 | -1.0726 | 2.5686 |

### 13. dreamer_binary_dominance_dataset.pkl

**Samples:** 92,184 | **Electrodes:** 14 | **Bands:** 4

**Global BDE Statistics:**
- Range: -7.107270 to 7.846405
- Mean: -0.046077 | Std: 1.187829
- Median: -0.476108
- Percentiles: P5=-1.2111, P95=2.3731

**Per-Band Statistics:**
| Band | Min | Max | Mean | Std | P5 | P95 |
|------|-----|-----|------|-----|----|----|
| Theta | -7.1073 | 7.8464 | -0.3070 | 1.0793 | -1.3574 | 1.8743 |
| Alpha | -6.5167 | 7.5995 | -0.1714 | 1.0701 | -1.2796 | 2.0208 |
| Beta | -6.0853 | 7.1156 | 0.1775 | 1.2986 | -1.0775 | 2.8057 |
| Gamma | -6.0588 | 7.0851 | 0.1165 | 1.2195 | -1.0726 | 2.5686 |

## Cross-Dataset Analysis

Analysis across **13 datasets** with **1,287,609 total samples**.

### Global Cross-Dataset Statistics

**Absolute BDE Range:** -10.947251 to 7.846405
**Mean of Dataset Means:** 0.024413 ± 0.473996
**Average Dataset Std:** 1.030048 ± 0.249314

### Cross-Dataset Band Comparison

| Band | Abs Min | Abs Max | Mean of Means | Avg Std |
|------|---------|---------|---------------|---------|
| Theta | -10.9473 | 7.8464 | -0.0909 | 0.9596 |
| Alpha | -10.7544 | 7.5995 | -0.0496 | 0.9271 |
| Beta | -10.2327 | 7.1156 | 0.1352 | 1.0715 |
| Gamma | -10.2936 | 7.0851 | 0.1029 | 1.1090 |

### Dataset Comparison Summary

| Dataset | Samples | Electrodes | BDE Range | Global Mean | Global Std |
|---------|---------|------------|-----------|-------------|------------|
| seed_multi_emotion_dataset.pkl | 258,345 | 62 | -10.947 to 4.805 | -1.4823 | 1.5692 |
| deap_multi_valence_dataset.pkl | 79,360 | 32 | -4.198 to 4.912 | 0.3460 | 0.7824 |
| deap_multi_arousal_dataset.pkl | 79,360 | 32 | -4.198 to 4.912 | 0.3460 | 0.7824 |
| deap_multi_dominance_dataset.pkl | 79,360 | 32 | -4.198 to 4.912 | 0.3460 | 0.7824 |
| deap_binary_valence_dataset.pkl | 79,360 | 32 | -4.198 to 4.912 | 0.3460 | 0.7824 |
| deap_binary_arousal_dataset.pkl | 79,360 | 32 | -4.198 to 4.912 | 0.3460 | 0.7824 |
| deap_binary_dominance_dataset.pkl | 79,360 | 32 | -4.198 to 4.912 | 0.3460 | 0.7824 |
| dreamer_multi_valence_dataset.pkl | 92,184 | 14 | -7.107 to 7.846 | -0.0461 | 1.1878 |
| dreamer_multi_arousal_dataset.pkl | 92,184 | 14 | -7.107 to 7.846 | -0.0461 | 1.1878 |
| dreamer_multi_dominance_dataset.pkl | 92,184 | 14 | -7.107 to 7.846 | -0.0461 | 1.1878 |
| dreamer_binary_valence_dataset.pkl | 92,184 | 14 | -7.107 to 7.846 | -0.0461 | 1.1878 |
| dreamer_binary_arousal_dataset.pkl | 92,184 | 14 | -7.107 to 7.846 | -0.0461 | 1.1878 |
| dreamer_binary_dominance_dataset.pkl | 92,184 | 14 | -7.107 to 7.846 | -0.0461 | 1.1878 |

## Universal Augmentation Parameters

Based on cross-dataset analysis, here are recommended universal augmentation parameters:

### Global Parameters

- **Universal Noise Std:** 0.051502
- **Universal Amplitude Range:** -18.4161 to 20.4161
- **Absolute BDE Range:** -10.9473 to 7.8464
- **Safe Augmentation Range:** -8.7578 to 6.2771

### Per-Band Universal Parameters

| Band | Noise Std | Scaling Range | Value Range |
|------|-----------|---------------|-------------|
| Theta | 0.057578 | -3.8366 to 5.8366 | -10.9473 to 7.8464 |
| Alpha | 0.055623 | -8.9557 to 10.9557 | -10.7544 to 7.5995 |
| Beta | 0.064289 | -2.2220 to 4.2220 | -10.2327 to 7.1156 |
| Gamma | 0.066540 | -4.5539 to 6.5539 | -10.2936 to 7.0851 |

### Recommended Augmentation Strategies

**Conservative (Light):**
- Noise: 3% of universal std = 0.030901
- Amplitude scaling: ±3% = 0.97 to 1.03

**Moderate:**
- Noise: 5% of universal std = 0.051502
- Amplitude scaling: ±8% = 0.92 to 1.08
- Band-specific perturbations: ±5%

**Aggressive:**
- Noise: 8% of universal std = 0.082404
- Amplitude scaling: ±12% = 0.88 to 1.12
- Band-specific perturbations: ±10%
- Cross-band correlations: enabled
