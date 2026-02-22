#!/usr/bin/env python3
"""
Plot histograms of weight changes at epoch 6 for K, Q, V, O, and MLP matrices.
Loads weights directly from safetensors (no extraction needed).

Usage:
    python codes/utils/plot_weight_changes_histogram.py \
        --datasets mmlu imdb sst2 \
        --models llama32_3b llama31_8b qwen_8b_base \
        --output-dir plots/weight_histograms
"""

import os
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from safetensors import safe_open
from pathlib import Path
import re
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial


def load_weights_from_safetensors(checkpoint_dir: Path, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """
    Load K, Q, V, O, and MLP weights from safetensors checkpoint.
    
    Returns dict with keys like: 'layer0_k', 'layer0_o', 'layer0_mlp_down', etc.
    """
    weights = {}
    
    if is_lora:
        # Load LoRA adapters from adapter_model.safetensors
        adapter_file = checkpoint_dir / "adapter_model.safetensors"
        if not adapter_file.exists():
            print(f"  ⚠️  No adapter file found: {adapter_file}")
            return weights
        
        with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
            for key in keys:
                # Match: base_model.model.model.layers.{N}.self_attn.{q|k|v|o}_proj.lora_{A|B}.default.weight
                attn_match = re.match(
                    r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.lora_(A|B)\.default\.weight",
                    key
                )
                if attn_match:
                    layer_idx = int(attn_match.group(1))
                    proj_type = attn_match.group(2)
                    lora_type = attn_match.group(3)
                    tensor = f.get_tensor(key)
                    # Convert to float32 first, then to numpy
                    tensor_np = tensor.float().cpu().numpy()
                    weights[f"layer{layer_idx}_{proj_type}_{lora_type}"] = tensor_np
                
                # Match: base_model.model.model.layers.{N}.mlp.{down|gate|up}_proj.lora_{A|B}.default.weight
                mlp_match = re.match(
                    r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(down|gate|up)_proj\.lora_(A|B)\.default\.weight",
                    key
                )
                if mlp_match:
                    layer_idx = int(mlp_match.group(1))
                    proj_type = mlp_match.group(2)
                    lora_type = mlp_match.group(3)
                    tensor = f.get_tensor(key)
                    # Convert to float32 first, then to numpy
                    tensor_np = tensor.float().cpu().numpy()
                    weights[f"layer{layer_idx}_mlp_{proj_type}_{lora_type}"] = tensor_np
    
    else:
        # Load full weights from model safetensors
        safetensor_files = sorted(checkpoint_dir.glob("model-*.safetensors"))
        if not safetensor_files:
            safetensor_files = [checkpoint_dir / "model.safetensors"]
        
        for safetensor_file in safetensor_files:
            if not safetensor_file.exists():
                continue
            
            with safe_open(str(safetensor_file), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                
                for key in keys:
                    # Match: model.layers.{N}.self_attn.{q|k|v|o}_proj.weight
                    attn_match = re.match(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", key)
                    if attn_match:
                        layer_idx = int(attn_match.group(1))
                        proj_type = attn_match.group(2)
                        tensor = f.get_tensor(key)
                        # Convert to float32 first, then to numpy
                        tensor_np = tensor.float().cpu().numpy()
                        weights[f"layer{layer_idx}_{proj_type}"] = tensor_np
                    
                    # Match: model.layers.{N}.mlp.{down|gate|up}_proj.weight
                    mlp_match = re.match(r"model\.layers\.(\d+)\.mlp\.(down|gate|up)_proj\.weight", key)
                    if mlp_match:
                        layer_idx = int(mlp_match.group(1))
                        proj_type = mlp_match.group(2)
                        tensor = f.get_tensor(key)
                        # Convert to float32 first, then to numpy
                        tensor_np = tensor.float().cpu().numpy()
                        weights[f"layer{layer_idx}_mlp_{proj_type}"] = tensor_np
    
    return weights


def load_weights_from_numpy(checkpoint_dir: Path, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """
    Load K, Q, V weights from existing numpy_weights directory.
    """
    numpy_dir = checkpoint_dir / "numpy_weights"
    if not numpy_dir.exists():
        return {}
    
    weights = {}
    
    for npy_file in numpy_dir.glob("*.npy"):
        # Parse filename: layer{N}_{type}.npy or layer{N}_{type}_{A|B}.npy
        name = npy_file.stem
        weights[name] = np.load(npy_file).astype(np.float32)
    
    return weights


def compute_weight_changes(weights_epoch0: Dict, weights_epoch6: Dict, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute weight changes: epoch6 - epoch0
    
    For LoRA: compute BA product first, then difference
    For Full: direct difference
    """
    changes = {}
    
    if is_lora:
        # For LoRA, compute W = B @ A for each epoch, then difference
        # Get all unique layer/projection combinations
        layer_projs = set()
        for key in weights_epoch0.keys():
            if key.endswith('_A'):
                layer_proj = key[:-2]  # Remove '_A'
                layer_projs.add(layer_proj)
        
        for layer_proj in layer_projs:
            key_a = f"{layer_proj}_A"
            key_b = f"{layer_proj}_B"
            
            if key_a in weights_epoch0 and key_b in weights_epoch0 and \
               key_a in weights_epoch6 and key_b in weights_epoch6:
                # Compute W = B @ A for each epoch
                w0 = weights_epoch0[key_b] @ weights_epoch0[key_a]
                w6 = weights_epoch6[key_b] @ weights_epoch6[key_a]
                
                # Compute change
                changes[layer_proj] = w6 - w0
    
    else:
        # For full finetuning, direct difference
        for key in weights_epoch0.keys():
            if key in weights_epoch6:
                changes[key] = weights_epoch6[key] - weights_epoch0[key]
    
    return changes


def plot_histogram_for_experiment(dataset: str, model: str, exp_type: str, 
                                   base_dir: Path, output_dir: Path, matrix_type: str = 'k'):
    """
    Create histogram plot for one experiment showing weight changes for K, Q, V, O, MLP.
    """
    is_lora = (exp_type == "lora")
    
    # Load weights from epoch 0 and epoch 6
    epoch0_dir = base_dir / dataset / model / exp_type / "epoch_weights" / "checkpoint-epoch-0"
    epoch6_dir = base_dir / dataset / model / exp_type / "epoch_weights" / "checkpoint-epoch-6"
    
    if not epoch0_dir.exists() or not epoch6_dir.exists():
        print(f"  ⚠️  Missing checkpoints for {dataset}/{model}/{exp_type}")
        return
    
    print(f"  Processing {dataset}/{model}/{exp_type}...")
    
    # Load K, Q, V from numpy (already saved)
    weights_epoch0_numpy = load_weights_from_numpy(epoch0_dir, is_lora)
    weights_epoch6_numpy = load_weights_from_numpy(epoch6_dir, is_lora)
    
    # Load O and MLP from safetensors
    weights_epoch0_safetensors = load_weights_from_safetensors(epoch0_dir, is_lora)
    weights_epoch6_safetensors = load_weights_from_safetensors(epoch6_dir, is_lora)
    
    # Combine
    weights_epoch0 = {**weights_epoch0_numpy, **weights_epoch0_safetensors}
    weights_epoch6 = {**weights_epoch6_numpy, **weights_epoch6_safetensors}
    
    if not weights_epoch0 or not weights_epoch6:
        print(f"  ⚠️  Failed to load weights for {dataset}/{model}/{exp_type}")
        return
    
    # Compute changes
    changes = compute_weight_changes(weights_epoch0, weights_epoch6, is_lora)
    
    if not changes:
        print(f"  ⚠️  No weight changes computed for {dataset}/{model}/{exp_type}")
        return
    
    # Organize changes by matrix type and layer (keep layers separate)
    # For MLP, keep down, gate, and up separate but in same file
    if matrix_type == 'mlp':
        matrix_types_to_process = ['mlp_down', 'mlp_gate', 'mlp_up']
    else:
        matrix_types_to_process = [matrix_type]
    
    # Extract layer-wise changes: {(matrix_type, layer_idx): change_array}
    layer_changes = {}
    
    for key, change in changes.items():
        # Parse layer index from key (e.g., "layer5_k" -> layer=5, type=k)
        import re
        match = re.match(r"layer(\d+)_(k|q|v|o|mlp_\w+)", key)
        if match:
            layer_idx = int(match.group(1))
            key_matrix_type = match.group(2)
            
            # Check if this matches what we're looking for
            if key_matrix_type in matrix_types_to_process:
                layer_changes[(key_matrix_type, layer_idx)] = change.flatten()
    
    # Get number of layers
    if layer_changes:
        num_layers = max(layer_idx for (_, layer_idx) in layer_changes.keys()) + 1
    else:
        print(f"  ⚠️  No layer changes found!")
        return
    
    # Create subplot grid for all layers - ONE PLOT PER ROW (linear + log scale)
    # For MLP, we have 3 matrix types, so multiply rows by 3
    num_matrix_types = len(matrix_types_to_process)
    cols = 1  # Only 1 column
    rows = num_layers * 2 * num_matrix_types  # Two rows per layer per matrix type (linear + log)
    
    subplot_titles = []
    row_heights = []
    for i in range(num_layers):
        for mt in matrix_types_to_process:
            mt_name = mt.upper().replace('_', ' ')
            subplot_titles.append(f"Layer {i} - {mt_name} - Linear Scale")
            subplot_titles.append(f"Layer {i} - {mt_name} - Log Scale")
            row_heights.append(2)  # Linear plot gets 2 units
            row_heights.append(1)  # Log plot gets 1 unit
    
    # Normalize row heights
    total_height = sum(row_heights)
    row_heights_normalized = [h / total_height for h in row_heights]
    
    # Calculate appropriate vertical spacing (must be < 1/(rows-1))
    max_spacing = 1.0 / (rows - 1) if rows > 1 else 0.01
    vertical_spacing = min(0.005, max_spacing * 0.9)  # Use 90% of max to be safe
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=vertical_spacing,
        row_heights=row_heights_normalized
    )
    
    colors = {
        'k': '#1f77b4',
        'q': '#ff7f0e', 
        'v': '#2ca02c',
        'o': '#d62728',
        'mlp_down': '#9467bd',
        'mlp_gate': '#8c564b',
        'mlp_up': '#e377c2',
        'mlp': '#9467bd'
    }
    
    # Plot each layer separately with statistics
    plot_idx = 0
    for layer_idx in range(num_layers):
        for mt in matrix_types_to_process:
            if (mt, layer_idx) in layer_changes:
                changes_data = layer_changes[(mt, layer_idx)]
                
                # Calculate statistics
                mean_val = np.mean(changes_data)
                std_val = np.std(changes_data)
                median_val = np.median(changes_data)
                min_val = np.min(changes_data)
                max_val = np.max(changes_data)
                q25 = np.percentile(changes_data, 25)
                q75 = np.percentile(changes_data, 75)
                p95 = np.percentile(changes_data, 95)
                p99 = np.percentile(changes_data, 99)
                total_params = len(changes_data)
                near_zero_pct = 100 * np.sum(np.abs(changes_data) < 0.0001) / total_params
                non_zero = np.sum(np.abs(changes_data) > 1e-6)
                l2_norm = np.linalg.norm(changes_data)
                sparsity_pct = 100 * non_zero / total_params
                
                # Pre-compute histogram to reduce file size
                # Use 100 bins with explicit range to ensure 0 is at bin center
                hist, bin_edges = np.histogram(changes_data, bins=100)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]
                
                # Linear scale plot (row = plot_idx * 2 + 1)
                row_linear = plot_idx * 2 + 1
                col = 1
                
                # Add histogram bar plot - LINEAR SCALE (no colors, just blue)
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=hist,
                        name=f'Layer {layer_idx} Linear',
                        marker=dict(
                            color='rgba(31, 119, 180, 0.7)',  # Single blue color
                            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                        ),
                        showlegend=False,
                        width=bin_width,
                        hovertemplate='Δw: %{x:.6f}<br>Count: %{y:,}<extra></extra>'
                    ),
                    row=row_linear, col=col
                )
                
                # Add vertical line at x=0 for reference (LINEAR)
                fig.add_vline(
                    x=0, 
                    line_dash="dash", 
                    line_color="black", 
                    line_width=1.5,
                    opacity=0.5,
                    row=row_linear, col=col
                )
                
                # Log scale plot (row = plot_idx * 2 + 2)
                row_log = plot_idx * 2 + 2
                
                # Add histogram bar plot - LOG SCALE (no colors, just blue)
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=hist,
                        name=f'Layer {layer_idx} Log',
                        marker=dict(
                            color='rgba(31, 119, 180, 0.7)',  # Single blue color
                            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                        ),
                        showlegend=False,
                        width=bin_width,
                        hovertemplate='Δw: %{x:.6f}<br>Count: %{y:,}<extra></extra>'
                    ),
                    row=row_log, col=col
                )
                
                # Add vertical line at x=0 for reference (LOG)
                fig.add_vline(
                    x=0, 
                    line_dash="dash", 
                    line_color="black", 
                    line_width=1.5,
                    opacity=0.5,
                    row=row_log, col=col
                )
                
                # Add statistics as text annotation with better formatting
                stats_text = (
                    f"<b style='font-size:14px'>📊 Statistics</b><br>"
                    f"<br>"
                    f"<b>Central Tendency:</b><br>"
                    f"  Mean: <b>{mean_val:.6f}</b><br>"
                    f"  Median: {median_val:.6f}<br>"
                    f"  Std Dev: {std_val:.6f}<br>"
                    f"<br>"
                    f"<b>Range:</b><br>"
                    f"  Min: <span style='color:red'>{min_val:.6f}</span><br>"
                    f"  Max: <span style='color:blue'>{max_val:.6f}</span><br>"
                    f"<br>"
                    f"<b>Percentiles:</b><br>"
                    f"  25th: {q25:.6f}<br>"
                    f"  75th: {q75:.6f}<br>"
                    f"  95th: {p95:.6f}<br>"
                    f"  99th: {p99:.6f}<br>"
                    f"<br>"
                    f"<b>Distribution:</b><br>"
                    f"  Near Zero: <b>{near_zero_pct:.1f}%</b><br>"
                    f"  Changed: {sparsity_pct:.1f}%<br>"
                    f"  L2 Norm: {l2_norm:.4f}<br>"
                    f"<br>"
                    f"<b>Count:</b><br>"
                    f"  Total: {total_params:,}<br>"
                    f"  Non-zero: {non_zero:,}"
                )
                
                # Add statistics annotation (only on linear plot)
                subplot_num_linear = layer_idx * 2 + 1
                if subplot_num_linear == 1:
                    xref_str = "x domain"
                    yref_str = "y domain"
                else:
                    xref_str = f"x{subplot_num_linear} domain"
                    yref_str = f"y{subplot_num_linear} domain"
                
                fig.add_annotation(
                    text=stats_text,
                    xref=xref_str,
                    yref=yref_str,
                    x=1.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=12, family="monospace"),
                    align="left",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=8
                )
                
                plot_idx += 1
    
    # Update layout
    matrix_name = matrix_type.upper().replace('_', ' ')
    title_text = f"{matrix_name} Matrix Weight Changes at Epoch 6 (Per Layer)<br>{dataset.upper()} | {model} | {exp_type.upper()}"
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        title_font=dict(size=24),
        title_pad=dict(t=100),  # Bigger space after title
        width=1600,  # Wider to accommodate stats on the right
        height=800 * num_layers,  # 800px per layer (same height regardless of matrix type count)
        showlegend=False,
        font=dict(size=16),  # Global font size
        margin=dict(t=150, b=50, l=100, r=350)  # More right margin for statistics
    )
    
    # Update all x and y axes labels with larger fonts and grid
    plot_idx = 0
    for i in range(num_layers):
        for mt in matrix_types_to_process:
            # Linear scale plot
            row_linear = plot_idx * 2 + 1
        col = 1
        fig.update_xaxes(
            title_text="Weight Change (Δw)", 
            row=row_linear, col=col, 
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0, 0, 0, 0.3)'
        )
        fig.update_yaxes(
            title_text="Count", 
            row=row_linear, col=col, 
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        
        # Log scale plot
        row_log = plot_idx * 2 + 2
        fig.update_xaxes(
            title_text="Weight Change (Δw)", 
            row=row_log, col=col, 
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0, 0, 0, 0.3)'
        )
        fig.update_yaxes(
            title_text="Count (log)", 
            row=row_log, col=col, 
            title_font=dict(size=18),
            tickfont=dict(size=16),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            type="log"  # LOG SCALE
        )
        
        plot_idx += 1
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}_{model}_{exp_type}_{matrix_type}_weight_changes_epoch6.html"
    fig.write_html(str(output_file))
    print(f"  ✅ Saved: {output_file}")
    
    # Compute and return statistics for ranking
    layer_stats = {}
    for layer_idx in range(num_layers):
        for mt in matrix_types_to_process:
            if (mt, layer_idx) in layer_changes:
                changes_data = layer_changes[(mt, layer_idx)]
                
                # Compute ranking metrics
                mean_abs_change = np.mean(np.abs(changes_data))
                l2_norm = np.linalg.norm(changes_data)
                std_dev = np.std(changes_data)
                pct_changed = 100 * np.sum(np.abs(changes_data) > 1e-6) / len(changes_data)
                
                layer_stats[(mt, layer_idx)] = {
                    'mean_abs': mean_abs_change,
                    'l2_norm': l2_norm,
                    'std_dev': std_dev,
                    'pct_changed': pct_changed,
                    'num_params': len(changes_data)
                }
    
    return layer_stats


def generate_ranking_summary(all_stats: Dict, output_dir: Path):
    """Generate ranking summary text files: 1 consolidated + 1 per model."""
    
    # Collect all layer rankings across all experiments
    all_rankings = []
    
    for (dataset, model, exp_type, matrix_type), stats in all_stats.items():
        if not stats:
            continue
        
        # Group by layer
        layer_data = {}
        for (mt, layer_idx), layer_stats in stats.items():
            if layer_idx not in layer_data:
                layer_data[layer_idx] = []
            layer_data[layer_idx].append((mt, layer_stats))
        
        # Create ranking entries
        for layer_idx, mt_stats_list in layer_data.items():
            # Aggregate stats for this layer
            mean_abs_avg = np.mean([s['mean_abs'] for _, s in mt_stats_list])
            l2_norm_avg = np.mean([s['l2_norm'] for _, s in mt_stats_list])
            std_dev_avg = np.mean([s['std_dev'] for _, s in mt_stats_list])
            pct_changed_avg = np.mean([s['pct_changed'] for _, s in mt_stats_list])
            
            all_rankings.append({
                'dataset': dataset.upper(),
                'model': model,
                'type': exp_type.upper(),
                'matrix': matrix_type.upper(),
                'layer': layer_idx,
                'mean_abs': mean_abs_avg,
                'l2_norm': l2_norm_avg,
                'std_dev': std_dev_avg,
                'pct_changed': pct_changed_avg
            })
    
    # Sort by mean absolute change
    all_rankings.sort(key=lambda x: x['mean_abs'], reverse=True)
    
    # ========================================================================
    # 1. CONSOLIDATED RANKING (ALL EXPERIMENTS)
    # ========================================================================
    summary_file_consolidated = output_dir / "layer_rankings_consolidated.txt"
    
    with open(summary_file_consolidated, 'w') as f:
        f.write("=" * 160 + "\n")
        f.write("CONSOLIDATED LAYER RANKING BY WEIGHT CHANGES AT EPOCH 6\n")
        f.write("=" * 160 + "\n\n")
        f.write("All layers from all datasets, models, types, and matrices ranked by Mean|Δw|\n\n")
        f.write("=" * 160 + "\n\n")
        
        # Write consolidated table
        f.write(f"{'Rank':<6} {'Dataset':<8} {'Model':<15} {'Type':<6} {'Matrix':<8} {'Layer':<7} "
               f"{'Mean|Δw|':<14} {'L2 Norm':<12} {'Std Dev':<14} {'%Changed':<10}\n")
        f.write(f"{'-'*6} {'-'*8} {'-'*15} {'-'*6} {'-'*8} {'-'*7} {'-'*14} {'-'*12} {'-'*14} {'-'*10}\n")
        
        for rank, item in enumerate(all_rankings, 1):
            f.write(f"{rank:<6} {item['dataset']:<8} {item['model']:<15} {item['type']:<6} {item['matrix']:<8} "
                   f"{item['layer']:<7} {item['mean_abs']:<14.6e} {item['l2_norm']:<12.2f} "
                   f"{item['std_dev']:<14.6e} {item['pct_changed']:<10.2f}%\n")
        
        # Add summary
        f.write(f"\n{'=' * 160}\n")
        f.write(f"SUMMARY\n")
        f.write(f"{'=' * 160}\n\n")
        f.write(f"Total entries: {len(all_rankings)}\n\n")
        
        f.write(f"TOP 10 MOST CHANGING LAYERS:\n")
        f.write(f"{'-' * 80}\n")
        for i, item in enumerate(all_rankings[:10], 1):
            f.write(f"{i:2}. {item['dataset']:<8} | {item['model']:<15} | {item['type']:<6} | "
                   f"{item['matrix']:<8} | Layer {item['layer']:<2} | Mean|Δw|={item['mean_abs']:.6e}\n")
        
        f.write(f"\nBOTTOM 10 LEAST CHANGING LAYERS:\n")
        f.write(f"{'-' * 80}\n")
        for i, item in enumerate(all_rankings[-10:], len(all_rankings)-9):
            f.write(f"{i:2}. {item['dataset']:<8} | {item['model']:<15} | {item['type']:<6} | "
                   f"{item['matrix']:<8} | Layer {item['layer']:<2} | Mean|Δw|={item['mean_abs']:.6e}\n")
    
    print(f"✅ Consolidated ranking saved: {summary_file_consolidated}")
    
    # ========================================================================
    # 2. PER-MODEL RANKINGS
    # ========================================================================
    # Get unique models
    unique_models = sorted(set(item['model'] for item in all_rankings))
    
    for model_name in unique_models:
        # Filter rankings for this model
        model_rankings = [item for item in all_rankings if item['model'] == model_name]
        model_rankings.sort(key=lambda x: x['mean_abs'], reverse=True)
        
        summary_file_model = output_dir / f"layer_rankings_{model_name}.txt"
        
        with open(summary_file_model, 'w') as f:
            f.write("=" * 140 + "\n")
            f.write(f"LAYER RANKING FOR {model_name.upper()}\n")
            f.write("=" * 140 + "\n\n")
            f.write(f"All layers from all datasets, types, and matrices for {model_name}\n")
            f.write(f"Ranked by Mean|Δw|\n\n")
            f.write("=" * 140 + "\n\n")
            
            # Write table
            f.write(f"{'Rank':<6} {'Dataset':<8} {'Type':<6} {'Matrix':<8} {'Layer':<7} "
                   f"{'Mean|Δw|':<14} {'L2 Norm':<12} {'Std Dev':<14} {'%Changed':<10}\n")
            f.write(f"{'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*7} {'-'*14} {'-'*12} {'-'*14} {'-'*10}\n")
            
            for rank, item in enumerate(model_rankings, 1):
                f.write(f"{rank:<6} {item['dataset']:<8} {item['type']:<6} {item['matrix']:<8} "
                       f"{item['layer']:<7} {item['mean_abs']:<14.6e} {item['l2_norm']:<12.2f} "
                       f"{item['std_dev']:<14.6e} {item['pct_changed']:<10.2f}%\n")
            
            # Add summary
            f.write(f"\n{'=' * 140}\n")
            f.write(f"SUMMARY FOR {model_name.upper()}\n")
            f.write(f"{'=' * 140}\n\n")
            f.write(f"Total entries: {len(model_rankings)}\n\n")
            
            f.write(f"TOP 10 MOST CHANGING LAYERS:\n")
            f.write(f"{'-' * 80}\n")
            for i, item in enumerate(model_rankings[:10], 1):
                f.write(f"{i:2}. {item['dataset']:<8} | {item['type']:<6} | {item['matrix']:<8} | "
                       f"Layer {item['layer']:<2} | Mean|Δw|={item['mean_abs']:.6e}\n")
            
            f.write(f"\nBOTTOM 10 LEAST CHANGING LAYERS:\n")
            f.write(f"{'-' * 80}\n")
            for i, item in enumerate(model_rankings[-10:], len(model_rankings)-9):
                f.write(f"{i:2}. {item['dataset']:<8} | {item['type']:<6} | {item['matrix']:<8} | "
                       f"Layer {item['layer']:<2} | Mean|Δw|={item['mean_abs']:.6e}\n")
            
            # Add per-dataset breakdown
            f.write(f"\n{'=' * 140}\n")
            f.write(f"PER-DATASET BREAKDOWN\n")
            f.write(f"{'=' * 140}\n\n")
            
            unique_datasets = sorted(set(item['dataset'] for item in model_rankings))
            for dataset in unique_datasets:
                dataset_rankings = [item for item in model_rankings if item['dataset'] == dataset]
                f.write(f"\n{dataset}:\n")
                f.write(f"  Top 5 layers: ")
                top5 = [f"Layer {item['layer']} ({item['matrix']}, {item['type']})" for item in dataset_rankings[:5]]
                f.write(", ".join(top5) + "\n")
        
        print(f"✅ Model ranking saved: {summary_file_model}")


def process_single_experiment(args_tuple):
    """Wrapper function for parallel processing."""
    dataset, model, exp_type, matrix_type, base_dir, output_dir = args_tuple
    try:
        print(f"Processing: {dataset}/{model}/{exp_type}/{matrix_type}")
        stats = plot_histogram_for_experiment(dataset, model, exp_type, base_dir, output_dir, matrix_type)
        key = (dataset, model, exp_type, matrix_type)
        return (key, stats)
    except Exception as e:
        print(f"❌ Error processing {dataset}/{model}/{exp_type}/{matrix_type}: {e}")
        return (None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of weight changes at epoch 6"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["imdb", "sst2", "mmlu"],
        help="Datasets to process"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama32_3b", "llama31_8b", "qwen_8b_base"],
        help="Models to process"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["full", "lora"],
        help="Experiment types (full, lora)"
    )
    parser.add_argument(
        "--base-dir",
        default="/home/kadir/topo/numpy_weights",
        help="Base directory containing numpy_weights"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/kadir/topo/plots/weight_histograms",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(8, cpu_count()),
        help="Number of parallel jobs (default: min(8, cpu_count()))"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("PLOTTING WEIGHT CHANGE HISTOGRAMS (PARALLEL)")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Types: {args.types}")
    print(f"Parallel jobs: {args.n_jobs}")
    print("=" * 80)
    
    # Process each matrix type separately (combine MLP into one)
    matrix_types = ['k', 'q', 'v', 'o', 'mlp']
    
    # Create list of all tasks
    tasks = []
    for dataset in args.datasets:
        for model in args.models:
            for exp_type in args.types:
                for matrix_type in matrix_types:
                    tasks.append((dataset, model, exp_type, matrix_type, base_dir, output_dir))
    
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Processing with {args.n_jobs} parallel workers...\n")
    
    # Process in parallel
    all_stats = {}
    with Pool(processes=args.n_jobs) as pool:
        results = pool.map(process_single_experiment, tasks)
    
    # Collect results
    for key, stats in results:
        if key is not None:
            all_stats[key] = stats
    
    # Generate ranking summary
    print("\n" + "=" * 80)
    print("GENERATING RANKING SUMMARY")
    print("=" * 80)
    generate_ranking_summary(all_stats, output_dir)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
