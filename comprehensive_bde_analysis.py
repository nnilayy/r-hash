#!/usr/bin/env python3
"""
Comprehensive BDE Range Analysis for All 13 Preprocessed Datasets
Analyzes all SEED, DEAP, and DREAMER datasets to create universal augmentation parameters.
"""

import pickle
import numpy as np
import os
from utils.pickle_patch import patch_pickle_loading
from datetime import datetime
import glob


def analyze_all_datasets():
    """
    Analyze all 13 preprocessed datasets and generate comprehensive statistics.
    """
    # Define all 13 dataset files
    datasets = [
        # SEED
        "seed_multi_emotion_dataset.pkl",
        
        # DEAP Multi-class
        "deap_multi_valence_dataset.pkl",
        "deap_multi_arousal_dataset.pkl", 
        "deap_multi_dominance_dataset.pkl",
        
        # DEAP Binary
        "deap_binary_valence_dataset.pkl",
        "deap_binary_arousal_dataset.pkl",
        "deap_binary_dominance_dataset.pkl",
        
        # DREAMER Multi-class
        "dreamer_multi_valence_dataset.pkl",
        "dreamer_multi_arousal_dataset.pkl",
        "dreamer_multi_dominance_dataset.pkl",
        
        # DREAMER Binary
        "dreamer_binary_valence_dataset.pkl",
        "dreamer_binary_arousal_dataset.pkl", 
        "dreamer_binary_dominance_dataset.pkl"
    ]
    
    # Initialize report
    report_lines = []
    report_lines.append("# Comprehensive BDE Value Range Analysis")
    report_lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("This report analyzes BDE (Band Differential Entropy) value ranges across all 13 preprocessed datasets for RBTransformer.")
    report_lines.append("")
    report_lines.append("## Dataset Overview")
    report_lines.append("")
    
    # Storage for global statistics across all datasets
    all_global_stats = []
    all_band_stats = {
        'theta': [], 'alpha': [], 'beta': [], 'gamma': []
    }
    dataset_info = []
    
    # Analyze each dataset
    for i, dataset_file in enumerate(datasets, 1):
        dataset_path = os.path.join("preprocessed_datasets", dataset_file)
        
        if not os.path.exists(dataset_path):
            print(f"⚠️  Dataset not found: {dataset_file}")
            report_lines.append(f"### {i}. {dataset_file}")
            report_lines.append("**Status:** ❌ File not found")
            report_lines.append("")
            continue
        
        print(f"Analyzing {i}/13: {dataset_file}")
        
        try:
            # Load and analyze dataset
            patch_pickle_loading()
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            
            # Collect all samples
            all_samples = []
            for j in range(len(dataset)):
                sample, _ = dataset[j]
                if sample.dim() == 3:
                    sample = sample.squeeze(0)  # Remove batch dimension
                all_samples.append(sample.numpy())
            
            all_samples = np.array(all_samples)
            num_samples, num_electrodes, num_bands = all_samples.shape
            
            # Calculate statistics
            stats = calculate_dataset_statistics(all_samples)
            
            # Store for global analysis
            all_global_stats.append(stats['global'])
            for band_name in ['theta', 'alpha', 'beta', 'gamma']:
                all_band_stats[band_name].append(stats['per_band'][band_name])
            
            dataset_info.append({
                'name': dataset_file,
                'samples': num_samples,
                'electrodes': num_electrodes,
                'bands': num_bands,
                'stats': stats
            })
            
            # Add to report
            report_lines.extend(format_dataset_report(i, dataset_file, num_samples, 
                                                    num_electrodes, num_bands, stats))
            
        except Exception as e:
            print(f"❌ Error analyzing {dataset_file}: {e}")
            report_lines.append(f"### {i}. {dataset_file}")
            report_lines.append(f"**Status:** ❌ Error: {e}")
            report_lines.append("")
    
    # Calculate cross-dataset statistics
    if all_global_stats:
        cross_dataset_stats = calculate_cross_dataset_statistics(all_global_stats, all_band_stats)
        report_lines.extend(format_cross_dataset_report(cross_dataset_stats, dataset_info))
        
        # Generate universal augmentation parameters
        augmentation_params = generate_universal_augmentation_parameters(cross_dataset_stats, dataset_info)
        report_lines.extend(format_augmentation_parameters(augmentation_params))
        
        # Save comprehensive report
        with open("BDE_Comprehensive_Analysis_Report.md", "w") as f:
            f.write("\n".join(report_lines))
        
        print(f"\n✅ Analysis complete! Report saved as 'BDE_Comprehensive_Analysis_Report.md'")
        print(f"📊 Analyzed {len(dataset_info)} datasets successfully")
        
        return augmentation_params, cross_dataset_stats, dataset_info
    
    else:
        print("❌ No datasets could be analyzed")
        return None, None, None


def calculate_dataset_statistics(all_samples):
    """Calculate comprehensive statistics for a single dataset."""
    stats = {
        'shape': all_samples.shape,
        'global': {
            'min': float(np.min(all_samples)),
            'max': float(np.max(all_samples)),
            'mean': float(np.mean(all_samples)),
            'std': float(np.std(all_samples)),
            'median': float(np.median(all_samples)),
            'percentiles': {
                'p1': float(np.percentile(all_samples, 1)),
                'p5': float(np.percentile(all_samples, 5)),
                'p25': float(np.percentile(all_samples, 25)),
                'p75': float(np.percentile(all_samples, 75)),
                'p95': float(np.percentile(all_samples, 95)),
                'p99': float(np.percentile(all_samples, 99)),
            }
        },
        'per_band': {}
    }
    
    # Per-band statistics
    band_names = ['theta', 'alpha', 'beta', 'gamma']
    for band_idx, band_name in enumerate(band_names):
        band_data = all_samples[:, :, band_idx]
        stats['per_band'][band_name] = {
            'min': float(np.min(band_data)),
            'max': float(np.max(band_data)),
            'mean': float(np.mean(band_data)),
            'std': float(np.std(band_data)),
            'median': float(np.median(band_data)),
            'p5': float(np.percentile(band_data, 5)),
            'p95': float(np.percentile(band_data, 95)),
        }
    
    return stats


def format_dataset_report(index, filename, num_samples, num_electrodes, num_bands, stats):
    """Format individual dataset report section."""
    lines = []
    lines.append(f"### {index}. {filename}")
    lines.append("")
    lines.append(f"**Samples:** {num_samples:,} | **Electrodes:** {num_electrodes} | **Bands:** {num_bands}")
    lines.append("")
    
    # Global statistics
    global_stats = stats['global']
    lines.append("**Global BDE Statistics:**")
    lines.append(f"- Range: {global_stats['min']:.6f} to {global_stats['max']:.6f}")
    lines.append(f"- Mean: {global_stats['mean']:.6f} | Std: {global_stats['std']:.6f}")
    lines.append(f"- Median: {global_stats['median']:.6f}")
    lines.append(f"- Percentiles: P5={global_stats['percentiles']['p5']:.4f}, P95={global_stats['percentiles']['p95']:.4f}")
    lines.append("")
    
    # Per-band statistics
    lines.append("**Per-Band Statistics:**")
    lines.append("| Band | Min | Max | Mean | Std | P5 | P95 |")
    lines.append("|------|-----|-----|------|-----|----|----|")
    
    for band_name in ['theta', 'alpha', 'beta', 'gamma']:
        band_stats = stats['per_band'][band_name]
        lines.append(f"| {band_name.capitalize()} | {band_stats['min']:.4f} | {band_stats['max']:.4f} | {band_stats['mean']:.4f} | {band_stats['std']:.4f} | {band_stats['p5']:.4f} | {band_stats['p95']:.4f} |")
    
    lines.append("")
    return lines


def calculate_cross_dataset_statistics(all_global_stats, all_band_stats):
    """Calculate statistics across all datasets."""
    cross_stats = {
        'global': {},
        'per_band': {}
    }
    
    # Global cross-dataset statistics
    all_mins = [stat['min'] for stat in all_global_stats]
    all_maxs = [stat['max'] for stat in all_global_stats]
    all_means = [stat['mean'] for stat in all_global_stats]
    all_stds = [stat['std'] for stat in all_global_stats]
    
    cross_stats['global'] = {
        'absolute_min': float(np.min(all_mins)),
        'absolute_max': float(np.max(all_maxs)),
        'mean_of_means': float(np.mean(all_means)),
        'std_of_means': float(np.std(all_means)),
        'mean_of_stds': float(np.mean(all_stds)),
        'std_of_stds': float(np.std(all_stds)),
        'dataset_count': len(all_global_stats)
    }
    
    # Per-band cross-dataset statistics
    for band_name in ['theta', 'alpha', 'beta', 'gamma']:
        band_stats_list = all_band_stats[band_name]
        if band_stats_list:
            band_mins = [stat['min'] for stat in band_stats_list]
            band_maxs = [stat['max'] for stat in band_stats_list]
            band_means = [stat['mean'] for stat in band_stats_list]
            band_stds = [stat['std'] for stat in band_stats_list]
            
            cross_stats['per_band'][band_name] = {
                'absolute_min': float(np.min(band_mins)),
                'absolute_max': float(np.max(band_maxs)),
                'mean_of_means': float(np.mean(band_means)),
                'std_of_means': float(np.std(band_means)),
                'mean_of_stds': float(np.mean(band_stds)),
                'std_of_stds': float(np.std(band_stds))
            }
    
    return cross_stats


def format_cross_dataset_report(cross_stats, dataset_info):
    """Format cross-dataset analysis section."""
    lines = []
    lines.append("## Cross-Dataset Analysis")
    lines.append("")
    lines.append(f"Analysis across **{cross_stats['global']['dataset_count']} datasets** with **{sum(info['samples'] for info in dataset_info):,} total samples**.")
    lines.append("")
    
    # Global cross-dataset statistics
    global_cross = cross_stats['global']
    lines.append("### Global Cross-Dataset Statistics")
    lines.append("")
    lines.append(f"**Absolute BDE Range:** {global_cross['absolute_min']:.6f} to {global_cross['absolute_max']:.6f}")
    lines.append(f"**Mean of Dataset Means:** {global_cross['mean_of_means']:.6f} ± {global_cross['std_of_means']:.6f}")
    lines.append(f"**Average Dataset Std:** {global_cross['mean_of_stds']:.6f} ± {global_cross['std_of_stds']:.6f}")
    lines.append("")
    
    # Cross-dataset band comparison
    lines.append("### Cross-Dataset Band Comparison")
    lines.append("")
    lines.append("| Band | Abs Min | Abs Max | Mean of Means | Avg Std |")
    lines.append("|------|---------|---------|---------------|---------|")
    
    for band_name in ['theta', 'alpha', 'beta', 'gamma']:
        if band_name in cross_stats['per_band']:
            band_cross = cross_stats['per_band'][band_name]
            lines.append(f"| {band_name.capitalize()} | {band_cross['absolute_min']:.4f} | {band_cross['absolute_max']:.4f} | {band_cross['mean_of_means']:.4f} | {band_cross['mean_of_stds']:.4f} |")
    
    lines.append("")
    
    # Dataset comparison table
    lines.append("### Dataset Comparison Summary")
    lines.append("")
    lines.append("| Dataset | Samples | Electrodes | BDE Range | Global Mean | Global Std |")
    lines.append("|---------|---------|------------|-----------|-------------|------------|")
    
    for info in dataset_info:
        global_stats = info['stats']['global']
        lines.append(f"| {info['name']} | {info['samples']:,} | {info['electrodes']} | {global_stats['min']:.3f} to {global_stats['max']:.3f} | {global_stats['mean']:.4f} | {global_stats['std']:.4f} |")
    
    lines.append("")
    return lines


def generate_universal_augmentation_parameters(cross_stats, dataset_info):
    """Generate universal augmentation parameters based on cross-dataset analysis."""
    global_cross = cross_stats['global']
    
    # Base parameters on cross-dataset statistics
    universal_params = {
        'universal_noise_std': global_cross['mean_of_stds'] * 0.05,  # 5% of average natural std
        'universal_amplitude_range': (
            1.0 - global_cross['std_of_means'] / abs(global_cross['mean_of_means']) if global_cross['mean_of_means'] != 0 else 0.92,
            1.0 + global_cross['std_of_means'] / abs(global_cross['mean_of_means']) if global_cross['mean_of_means'] != 0 else 1.08
        ),
        'absolute_bde_range': (global_cross['absolute_min'], global_cross['absolute_max']),
        'safe_augmentation_range': (
            global_cross['absolute_min'] * 0.8,  # 80% of absolute range for safety
            global_cross['absolute_max'] * 0.8
        ),
        'per_band_universal': {}
    }
    
    # Per-band universal parameters
    for band_name in ['theta', 'alpha', 'beta', 'gamma']:
        if band_name in cross_stats['per_band']:
            band_cross = cross_stats['per_band'][band_name]
            universal_params['per_band_universal'][band_name] = {
                'noise_std': band_cross['mean_of_stds'] * 0.06,
                'scaling_range': (
                    1.0 - band_cross['std_of_means'] / abs(band_cross['mean_of_means']) if band_cross['mean_of_means'] != 0 else 0.9,
                    1.0 + band_cross['std_of_means'] / abs(band_cross['mean_of_means']) if band_cross['mean_of_means'] != 0 else 1.1
                ),
                'value_range': (band_cross['absolute_min'], band_cross['absolute_max'])
            }
    
    return universal_params


def format_augmentation_parameters(params):
    """Format augmentation parameters section."""
    lines = []
    lines.append("## Universal Augmentation Parameters")
    lines.append("")
    lines.append("Based on cross-dataset analysis, here are recommended universal augmentation parameters:")
    lines.append("")
    
    lines.append("### Global Parameters")
    lines.append("")
    lines.append(f"- **Universal Noise Std:** {params['universal_noise_std']:.6f}")
    lines.append(f"- **Universal Amplitude Range:** {params['universal_amplitude_range'][0]:.4f} to {params['universal_amplitude_range'][1]:.4f}")
    lines.append(f"- **Absolute BDE Range:** {params['absolute_bde_range'][0]:.4f} to {params['absolute_bde_range'][1]:.4f}")
    lines.append(f"- **Safe Augmentation Range:** {params['safe_augmentation_range'][0]:.4f} to {params['safe_augmentation_range'][1]:.4f}")
    lines.append("")
    
    lines.append("### Per-Band Universal Parameters")
    lines.append("")
    lines.append("| Band | Noise Std | Scaling Range | Value Range |")
    lines.append("|------|-----------|---------------|-------------|")
    
    for band_name in ['theta', 'alpha', 'beta', 'gamma']:
        if band_name in params['per_band_universal']:
            band_params = params['per_band_universal'][band_name]
            scale_range = band_params['scaling_range']
            value_range = band_params['value_range']
            lines.append(f"| {band_name.capitalize()} | {band_params['noise_std']:.6f} | {scale_range[0]:.4f} to {scale_range[1]:.4f} | {value_range[0]:.4f} to {value_range[1]:.4f} |")
    
    lines.append("")
    
    lines.append("### Recommended Augmentation Strategies")
    lines.append("")
    lines.append("**Conservative (Light):**")
    lines.append(f"- Noise: 3% of universal std = {params['universal_noise_std'] * 0.6:.6f}")
    lines.append(f"- Amplitude scaling: ±3% = {1.0 - 0.03:.2f} to {1.0 + 0.03:.2f}")
    lines.append("")
    lines.append("**Moderate:**")
    lines.append(f"- Noise: 5% of universal std = {params['universal_noise_std']:.6f}")
    lines.append(f"- Amplitude scaling: ±8% = {1.0 - 0.08:.2f} to {1.0 + 0.08:.2f}")
    lines.append("- Band-specific perturbations: ±5%")
    lines.append("")
    lines.append("**Aggressive:**") 
    lines.append(f"- Noise: 8% of universal std = {params['universal_noise_std'] * 1.6:.6f}")
    lines.append(f"- Amplitude scaling: ±12% = {1.0 - 0.12:.2f} to {1.0 + 0.12:.2f}")
    lines.append("- Band-specific perturbations: ±10%")
    lines.append("- Cross-band correlations: enabled")
    lines.append("")
    
    return lines


if __name__ == "__main__":
    print("🔍 Starting comprehensive BDE analysis across all 13 datasets...")
    augmentation_params, cross_stats, dataset_info = analyze_all_datasets()
    
    if augmentation_params:
        print("\n🎯 Universal augmentation parameters generated!")
        print("📝 Check 'BDE_Comprehensive_Analysis_Report.md' for detailed results")
    else:
        print("\n❌ Analysis failed - no datasets could be processed")
