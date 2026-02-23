#!/usr/bin/env python3
"""
Analyze spatial patterns in weight change heatmaps.
Generates rankings and insights based on:
- Row-wise and column-wise variance
- Concentration metrics (max/mean ratio, top 10% mass, sparsity)
- Pattern comparisons across matrix types
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import re
from safetensors import safe_open
from collections import defaultdict


def load_weights_from_numpy(checkpoint_dir: Path, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """Load K, Q, V weights from numpy files."""
    weights = {}
    numpy_dir = checkpoint_dir / 'numpy_weights'
    
    if not numpy_dir.exists():
        return weights
    
    for matrix_type in ['k', 'q', 'v']:
        if is_lora:
            for suffix in ['A', 'B']:
                pattern = f'layer*_{matrix_type}_{suffix}.npy'
                for npy_file in sorted(numpy_dir.glob(pattern)):
                    layer_match = re.search(r'layer(\d+)', npy_file.name)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        key = f'layer{layer_idx}_{matrix_type}_{suffix}'
                        weights[key] = np.load(npy_file)
        else:
            pattern = f'layer*_{matrix_type}.npy'
            for npy_file in sorted(numpy_dir.glob(pattern)):
                layer_match = re.search(r'layer(\d+)', npy_file.name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    key = f'layer{layer_idx}_{matrix_type}'
                    weights[key] = np.load(npy_file)
    
    return weights


def load_weights_from_safetensors(checkpoint_dir: Path, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """Load O and MLP weights from safetensors."""
    weights = {}
    safetensors_files = sorted(checkpoint_dir.glob('*.safetensors'))
    
    if not safetensors_files:
        return weights
    
    if is_lora:
        adapter_file = checkpoint_dir / 'adapter_model.safetensors'
        if not adapter_file.exists():
            return weights
        safetensors_files = [adapter_file]
    
    for st_file in safetensors_files:
        with safe_open(str(st_file), framework='pt', device='cpu') as f:
            keys = f.keys()
            
            for key in keys:
                layer_match = re.search(r'layers\.(\d+)', key)
                if not layer_match:
                    continue
                layer_idx = int(layer_match.group(1))
                
                tensor = f.get_tensor(key)
                
                if not is_lora:
                    if 'self_attn.o_proj.weight' in key or 'attention.wo.weight' in key:
                        if f'layer{layer_idx}_o' not in weights:
                            weights[f'layer{layer_idx}_o'] = tensor.float().cpu().numpy()
                    elif 'mlp.gate_proj.weight' in key or 'feed_forward.w1.weight' in key:
                        if f'layer{layer_idx}_mlp_gate' not in weights:
                            weights[f'layer{layer_idx}_mlp_gate'] = tensor.float().cpu().numpy()
                    elif 'mlp.up_proj.weight' in key or 'feed_forward.w3.weight' in key:
                        if f'layer{layer_idx}_mlp_up' not in weights:
                            weights[f'layer{layer_idx}_mlp_up'] = tensor.float().cpu().numpy()
                    elif 'mlp.down_proj.weight' in key or 'feed_forward.w2.weight' in key:
                        if f'layer{layer_idx}_mlp_down' not in weights:
                            weights[f'layer{layer_idx}_mlp_down'] = tensor.float().cpu().numpy()
    
    return weights


def compute_weight_changes(weights_epoch0: Dict, weights_epoch6: Dict, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """Compute weight changes."""
    changes = {}
    
    if is_lora:
        # LoRA: compute (B_6 @ A_6) - (B_0 @ A_0)
        layer_indices = set()
        for key in weights_epoch0.keys():
            if '_A' in key:
                layer_idx = int(re.search(r'layer(\d+)', key).group(1))
                layer_indices.add(layer_idx)
        
        for layer_idx in sorted(layer_indices):
            for matrix_type in ['k', 'q', 'v']:
                key_A = f'layer{layer_idx}_{matrix_type}_A'
                key_B = f'layer{layer_idx}_{matrix_type}_B'
                
                if key_A in weights_epoch0 and key_B in weights_epoch0 and \
                   key_A in weights_epoch6 and key_B in weights_epoch6:
                    W0 = weights_epoch0[key_B] @ weights_epoch0[key_A]
                    W6 = weights_epoch6[key_B] @ weights_epoch6[key_A]
                    changes[f'layer{layer_idx}_{matrix_type}'] = W6 - W0
    else:
        # Full finetuning: W_6 - W_0
        for key in weights_epoch0.keys():
            if key in weights_epoch6:
                changes[key] = weights_epoch6[key] - weights_epoch0[key]
    
    return changes


def analyze_spatial_patterns(delta: np.ndarray) -> Dict[str, float]:
    """
    Analyze spatial patterns in weight changes.
    
    Returns:
        Dictionary with pattern metrics:
        - mean_abs: Mean absolute change
        - max_abs: Maximum absolute change
        - concentration_ratio: max/mean ratio (higher = more localized)
        - top10_mass: Fraction of total change in top 10% of weights
        - sparsity: Fraction of weights with |Δw| < 1e-5
        - row_variance: Variance of row-wise means
        - col_variance: Variance of column-wise means
        - row_max_idx: Index of row with highest mean change
        - col_max_idx: Index of column with highest mean change
    """
    abs_delta = np.abs(delta)
    
    # Basic statistics
    mean_abs = np.mean(abs_delta)
    max_abs = np.max(abs_delta)
    
    # Concentration ratio
    concentration_ratio = max_abs / mean_abs if mean_abs > 0 else 0
    
    # Top 10% mass
    flat = abs_delta.flatten()
    sorted_flat = np.sort(flat)[::-1]
    top10_idx = max(1, int(0.1 * len(sorted_flat)))
    top10_mass = np.sum(sorted_flat[:top10_idx]) / np.sum(sorted_flat) if np.sum(sorted_flat) > 0 else 0
    
    # Sparsity (fraction of near-zero weights)
    sparsity = np.mean(abs_delta < 1e-5)
    
    # Row-wise and column-wise analysis
    row_means = np.mean(abs_delta, axis=1)
    col_means = np.mean(abs_delta, axis=0)
    
    row_variance = np.var(row_means)
    col_variance = np.var(col_means)
    
    row_max_idx = int(np.argmax(row_means))
    col_max_idx = int(np.argmax(col_means))
    
    return {
        'mean_abs': mean_abs,
        'max_abs': max_abs,
        'concentration_ratio': concentration_ratio,
        'top10_mass': top10_mass,
        'sparsity': sparsity,
        'row_variance': row_variance,
        'col_variance': col_variance,
        'row_max_idx': row_max_idx,
        'col_max_idx': col_max_idx,
    }


def analyze_experiment(base_dir: Path, dataset: str, model: str, exp_type: str) -> Dict[str, Dict[int, Dict]]:
    """
    Analyze all weight changes for one experiment.
    
    Returns:
        Nested dict: {matrix_type: {layer_idx: pattern_metrics}}
    """
    exp_path = base_dir / dataset / model / exp_type / 'epoch_weights'
    checkpoint_epoch0 = exp_path / 'checkpoint-epoch-0'
    checkpoint_epoch6 = exp_path / 'checkpoint-epoch-6'
    
    if not checkpoint_epoch0.exists() or not checkpoint_epoch6.exists():
        print(f"  ⚠️  Checkpoints not found for {dataset}/{model}/{exp_type}")
        return {}
    
    is_lora = (exp_type == 'lora')
    
    # Load weights
    print(f"  Loading {dataset}/{model}/{exp_type}...")
    weights_0_numpy = load_weights_from_numpy(checkpoint_epoch0, is_lora)
    weights_6_numpy = load_weights_from_numpy(checkpoint_epoch6, is_lora)
    
    weights_0_st = load_weights_from_safetensors(checkpoint_epoch0, is_lora)
    weights_6_st = load_weights_from_safetensors(checkpoint_epoch6, is_lora)
    
    weights_0 = {**weights_0_numpy, **weights_0_st}
    weights_6 = {**weights_6_numpy, **weights_6_st}
    
    # Compute changes
    changes = compute_weight_changes(weights_0, weights_6, is_lora)
    
    # Organize by matrix type
    results = defaultdict(dict)
    
    for key, delta in changes.items():
        # Parse key: layer{idx}_{matrix_type}
        match = re.match(r'layer(\d+)_(.+)', key)
        if not match:
            continue
        
        layer_idx = int(match.group(1))
        matrix_type = match.group(2)
        
        # Analyze patterns
        patterns = analyze_spatial_patterns(delta)
        patterns['shape'] = delta.shape
        
        results[matrix_type][layer_idx] = patterns
    
    return dict(results)


def generate_rankings(all_results: Dict, output_dir: Path):
    """
    Generate ranking files based on pattern analysis.
    
    all_results structure:
    {
        (dataset, model, exp_type): {
            matrix_type: {
                layer_idx: pattern_metrics
            }
        }
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall ranking by mean absolute change
    print("\n" + "="*80)
    print("GENERATING RANKINGS")
    print("="*80)
    
    all_entries = []
    for (dataset, model, exp_type), matrix_results in all_results.items():
        for matrix_type, layer_results in matrix_results.items():
            for layer_idx, metrics in layer_results.items():
                all_entries.append({
                    'dataset': dataset,
                    'model': model,
                    'type': exp_type,
                    'matrix': matrix_type,
                    'layer': layer_idx,
                    **metrics
                })
    
    # Sort by mean absolute change
    all_entries.sort(key=lambda x: x['mean_abs'], reverse=True)
    
    # Write overall ranking
    ranking_file = output_dir / 'layer_ranking_by_mean_change.txt'
    with open(ranking_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LAYER RANKING BY MEAN ABSOLUTE CHANGE\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Rank':<6} {'Dataset':<10} {'Model':<15} {'Type':<6} {'Matrix':<12} {'Layer':<6} {'Mean|Δw|':<12} {'Max|Δw|':<12} {'Concentration':<14}\n")
        f.write("-"*100 + "\n")
        
        for rank, entry in enumerate(all_entries, 1):
            f.write(f"{rank:<6} {entry['dataset']:<10} {entry['model']:<15} {entry['type']:<6} "
                   f"{entry['matrix']:<12} {entry['layer']:<6} {entry['mean_abs']:<12.6e} "
                   f"{entry['max_abs']:<12.6e} {entry['concentration_ratio']:<14.2f}\n")
    
    print(f"✅ Saved: {ranking_file}")
    
    # 2. Ranking by concentration (localized changes)
    all_entries.sort(key=lambda x: x['concentration_ratio'], reverse=True)
    
    concentration_file = output_dir / 'layer_ranking_by_concentration.txt'
    with open(concentration_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LAYER RANKING BY CONCENTRATION (Max/Mean Ratio)\n")
        f.write("Higher ratio = more localized changes\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Rank':<6} {'Dataset':<10} {'Model':<15} {'Type':<6} {'Matrix':<12} {'Layer':<6} {'Concentration':<14} {'Top10%':<10} {'Sparsity':<10}\n")
        f.write("-"*100 + "\n")
        
        for rank, entry in enumerate(all_entries, 1):
            f.write(f"{rank:<6} {entry['dataset']:<10} {entry['model']:<15} {entry['type']:<6} "
                   f"{entry['matrix']:<12} {entry['layer']:<6} {entry['concentration_ratio']:<14.2f} "
                   f"{entry['top10_mass']:<10.4f} {entry['sparsity']:<10.4f}\n")
    
    print(f"✅ Saved: {concentration_file}")
    
    # 3. Ranking by sparsity
    all_entries.sort(key=lambda x: x['sparsity'], reverse=True)
    
    sparsity_file = output_dir / 'layer_ranking_by_sparsity.txt'
    with open(sparsity_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("LAYER RANKING BY SPARSITY\n")
        f.write("Higher sparsity = more weights unchanged\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Rank':<6} {'Dataset':<10} {'Model':<15} {'Type':<6} {'Matrix':<12} {'Layer':<6} {'Sparsity':<10} {'Mean|Δw|':<12}\n")
        f.write("-"*100 + "\n")
        
        for rank, entry in enumerate(all_entries, 1):
            f.write(f"{rank:<6} {entry['dataset']:<10} {entry['model']:<15} {entry['type']:<6} "
                   f"{entry['matrix']:<12} {entry['layer']:<6} {entry['sparsity']:<10.4f} "
                   f"{entry['mean_abs']:<12.6e}\n")
    
    print(f"✅ Saved: {sparsity_file}")
    
    # 4. Row/Column variance analysis
    variance_file = output_dir / 'spatial_variance_analysis.txt'
    with open(variance_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("SPATIAL VARIANCE ANALYSIS\n")
        f.write("Row variance = variance of output dimension changes\n")
        f.write("Col variance = variance of input dimension changes\n")
        f.write("="*100 + "\n\n")
        
        # Group by experiment
        for (dataset, model, exp_type), matrix_results in sorted(all_results.items()):
            f.write(f"\n{dataset.upper()} | {model} | {exp_type.upper()}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Matrix':<12} {'Layer':<6} {'Row Var':<12} {'Col Var':<12} {'Row>Col':<8} {'Max Row':<8} {'Max Col':<8}\n")
            f.write("-"*100 + "\n")
            
            for matrix_type in sorted(matrix_results.keys()):
                layer_results = matrix_results[matrix_type]
                for layer_idx in sorted(layer_results.keys()):
                    metrics = layer_results[layer_idx]
                    row_dominant = "Yes" if metrics['row_variance'] > metrics['col_variance'] else "No"
                    f.write(f"{matrix_type:<12} {layer_idx:<6} {metrics['row_variance']:<12.6e} "
                           f"{metrics['col_variance']:<12.6e} {row_dominant:<8} "
                           f"{metrics['row_max_idx']:<8} {metrics['col_max_idx']:<8}\n")
    
    print(f"✅ Saved: {variance_file}")
    
    # 5. Matrix type summary
    summary_file = output_dir / 'matrix_type_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("MATRIX TYPE PATTERN SUMMARY\n")
        f.write("="*100 + "\n\n")
        
        # Aggregate by matrix type across all experiments
        matrix_stats = defaultdict(lambda: {
            'mean_abs': [],
            'concentration': [],
            'sparsity': [],
            'top10_mass': []
        })
        
        for (dataset, model, exp_type), matrix_results in all_results.items():
            for matrix_type, layer_results in matrix_results.items():
                for metrics in layer_results.values():
                    matrix_stats[matrix_type]['mean_abs'].append(metrics['mean_abs'])
                    matrix_stats[matrix_type]['concentration'].append(metrics['concentration_ratio'])
                    matrix_stats[matrix_type]['sparsity'].append(metrics['sparsity'])
                    matrix_stats[matrix_type]['top10_mass'].append(metrics['top10_mass'])
        
        f.write(f"{'Matrix':<12} {'Avg Mean|Δw|':<15} {'Avg Concentration':<18} {'Avg Sparsity':<15} {'Avg Top10%':<12}\n")
        f.write("-"*100 + "\n")
        
        for matrix_type in sorted(matrix_stats.keys()):
            stats = matrix_stats[matrix_type]
            f.write(f"{matrix_type:<12} {np.mean(stats['mean_abs']):<15.6e} "
                   f"{np.mean(stats['concentration']):<18.2f} "
                   f"{np.mean(stats['sparsity']):<15.4f} "
                   f"{np.mean(stats['top10_mass']):<12.4f}\n")
    
    print(f"✅ Saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze spatial patterns in weight change heatmaps")
    parser.add_argument("--datasets", nargs="+", default=["imdb", "sst2", "mmlu"], help="Datasets to analyze")
    parser.add_argument("--models", nargs="+", default=["llama32_3b", "llama31_8b", "qwen_8b_base"], help="Models to analyze")
    parser.add_argument("--types", nargs="+", default=["full", "lora"], help="Experiment types")
    parser.add_argument("--base-dir", default="/home/kadir/topo/numpy_weights", help="Base directory")
    parser.add_argument("--output-dir", default="/home/kadir/topo/plots/weight_heatmaps", help="Output directory")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("ANALYZING SPATIAL PATTERNS IN WEIGHT CHANGES")
    print("="*80)
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Types: {args.types}")
    print("="*80)
    
    # Analyze all experiments
    all_results = {}
    
    for dataset in args.datasets:
        for model in args.models:
            for exp_type in args.types:
                results = analyze_experiment(base_dir, dataset, model, exp_type)
                if results:
                    all_results[(dataset, model, exp_type)] = results
    
    # Generate rankings
    if all_results:
        generate_rankings(all_results, output_dir)
        print("\n" + "="*80)
        print("DONE! Pattern analysis complete.")
        print("="*80)
    else:
        print("\n⚠️  No results to analyze!")


if __name__ == "__main__":
    main()
