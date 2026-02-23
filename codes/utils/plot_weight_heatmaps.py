#!/usr/bin/env python3
"""
Plot heatmaps of weight changes at epoch 6 for K, Q, V, O, and MLP matrices.
Shows spatial patterns of which parts of each matrix changed most.

Usage:
    python codes/utils/plot_weight_heatmaps.py \
        --datasets imdb \
        --models llama32_3b \
        --types full \
        --output-dir plots/weight_heatmaps
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
from scipy.ndimage import zoom


def downsample_matrix(matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Downsample a matrix to target shape while preserving aspect ratio.
    Uses max pooling to preserve high values (more colorful visualization).
    """
    orig_h, orig_w = matrix.shape
    target_h, target_w = target_shape
    
    # Ensure matrix is float type
    matrix = matrix.astype(np.float32)
    
    # Calculate bin sizes (how many original cells per downsampled cell)
    bin_h = orig_h / target_h
    bin_w = orig_w / target_w
    
    # Create output array
    downsampled = np.zeros(target_shape, dtype=np.float32)
    
    # For each cell in the downsampled matrix, take the MAX of the corresponding region
    # (MAX preserves high values better than MEAN, making it more colorful)
    for i in range(target_h):
        for j in range(target_w):
            # Calculate the region in the original matrix
            row_start = int(i * bin_h)
            row_end = int((i + 1) * bin_h)
            col_start = int(j * bin_w)
            col_end = int((j + 1) * bin_w)
            
            # Take max of this region
            downsampled[i, j] = np.max(matrix[row_start:row_end, col_start:col_end])
    
    return downsampled


def calculate_target_shape(original_shape: Tuple[int, int], max_dim: int = 64, scale_factor: float = None) -> Tuple[int, int]:
    """
    Calculate target shape using FIXED ratio.
    Default: 1/128 for K/Q/V/O
    Can specify custom scale_factor for MLP (e.g., 1/256)
    """
    h, w = original_shape
    
    # Use custom scale factor if provided, otherwise default to 1/128
    if scale_factor is None:
        scale_factor = 1.0 / 128.0
    
    # Apply scale factor to both dimensions
    target_h = max(1, round(h * scale_factor))
    target_w = max(1, round(w * scale_factor))
    
    return (target_h, target_w)


def load_weights_from_safetensors(checkpoint_dir: Path, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """
    Load K, Q, V, O, and MLP weights from safetensors checkpoint.
    
    Returns dict with keys like: 'layer0_k', 'layer0_o', 'layer0_mlp_down', etc.
    """
    weights = {}
    
    # Find safetensors files
    safetensors_files = sorted(checkpoint_dir.glob('*.safetensors'))
    if not safetensors_files:
        return weights
    
    # For LoRA, use adapter_model.safetensors
    if is_lora:
        adapter_file = checkpoint_dir / 'adapter_model.safetensors'
        if not adapter_file.exists():
            return weights
        safetensors_files = [adapter_file]
    
    for st_file in safetensors_files:
        with safe_open(str(st_file), framework='pt', device='cpu') as f:
            keys = f.keys()
            
            for key in keys:
                # Parse layer number
                layer_match = re.search(r'layers\.(\d+)', key)
                if not layer_match:
                    continue
                layer_idx = int(layer_match.group(1))
                
                # Get tensor
                tensor = f.get_tensor(key)
                
                if is_lora:
                    # LoRA: only K, Q, V have adapters
                    if 'k_proj.lora_A' in key:
                        if f'layer{layer_idx}_k_A' not in weights:
                            weights[f'layer{layer_idx}_k_A'] = tensor.float().cpu().numpy()
                    elif 'k_proj.lora_B' in key:
                        if f'layer{layer_idx}_k_B' not in weights:
                            weights[f'layer{layer_idx}_k_B'] = tensor.float().cpu().numpy()
                    elif 'q_proj.lora_A' in key:
                        if f'layer{layer_idx}_q_A' not in weights:
                            weights[f'layer{layer_idx}_q_A'] = tensor.float().cpu().numpy()
                    elif 'q_proj.lora_B' in key:
                        if f'layer{layer_idx}_q_B' not in weights:
                            weights[f'layer{layer_idx}_q_B'] = tensor.float().cpu().numpy()
                    elif 'v_proj.lora_A' in key:
                        if f'layer{layer_idx}_v_A' not in weights:
                            weights[f'layer{layer_idx}_v_A'] = tensor.float().cpu().numpy()
                    elif 'v_proj.lora_B' in key:
                        if f'layer{layer_idx}_v_B' not in weights:
                            weights[f'layer{layer_idx}_v_B'] = tensor.float().cpu().numpy()
                else:
                    # Full finetuning: get O and MLP from safetensors
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


def load_weights_from_numpy(checkpoint_dir: Path, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """
    Load K, Q, V weights from numpy files.
    """
    weights = {}
    numpy_dir = checkpoint_dir / 'numpy_weights'
    
    if not numpy_dir.exists():
        return weights
    
    # Load K, Q, V
    for matrix_type in ['k', 'q', 'v']:
        if is_lora:
            # LoRA: load A and B matrices
            for suffix in ['A', 'B']:
                pattern = f'layer*_{matrix_type}_{suffix}.npy'
                for npy_file in sorted(numpy_dir.glob(pattern)):
                    layer_match = re.search(r'layer(\d+)', npy_file.name)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        key = f'layer{layer_idx}_{matrix_type}_{suffix}'
                        weights[key] = np.load(npy_file)
        else:
            # Full: load weight matrices
            pattern = f'layer*_{matrix_type}.npy'
            for npy_file in sorted(numpy_dir.glob(pattern)):
                layer_match = re.search(r'layer(\d+)', npy_file.name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    key = f'layer{layer_idx}_{matrix_type}'
                    weights[key] = np.load(npy_file)
    
    return weights


def compute_weight_changes(weights_epoch0: Dict, weights_epoch6: Dict, is_lora: bool = False) -> Dict[str, np.ndarray]:
    """
    Compute weight changes: W_6 - W_0 or (B_6 @ A_6) - (B_0 @ A_0) for LoRA.
    """
    changes = {}
    
    if is_lora:
        # For LoRA, compute BA product and then difference
        layers = set()
        for key in weights_epoch0.keys():
            layer_match = re.search(r'layer(\d+)', key)
            if layer_match:
                layers.add(int(layer_match.group(1)))
        
        for layer_idx in sorted(layers):
            for matrix_type in ['k', 'q', 'v']:
                key_A0 = f'layer{layer_idx}_{matrix_type}_A'
                key_B0 = f'layer{layer_idx}_{matrix_type}_B'
                key_A6 = f'layer{layer_idx}_{matrix_type}_A'
                key_B6 = f'layer{layer_idx}_{matrix_type}_B'
                
                if all(k in weights_epoch0 for k in [key_A0, key_B0]) and \
                   all(k in weights_epoch6 for k in [key_A6, key_B6]):
                    # Compute BA products
                    W0 = weights_epoch0[key_B0] @ weights_epoch0[key_A0]
                    W6 = weights_epoch6[key_B6] @ weights_epoch6[key_A6]
                    
                    # Compute change
                    changes[f'layer{layer_idx}_{matrix_type}'] = W6 - W0
    else:
        # For full finetuning, direct subtraction
        for key in weights_epoch0.keys():
            if key in weights_epoch6:
                changes[key] = weights_epoch6[key] - weights_epoch0[key]
    
    return changes


def plot_heatmap_for_experiment(dataset: str, model: str, exp_type: str, 
                                base_dir: Path, output_dir: Path, 
                                matrix_type: str = 'k', max_dim: int = 64,
                                global_scales: Dict[str, Tuple[float, float]] = None) -> bool:
    """
    Generate heatmap for a specific experiment and matrix type.
    """
    # Construct paths
    exp_path = base_dir / dataset / model / exp_type / 'epoch_weights'
    checkpoint_epoch0 = exp_path / 'checkpoint-epoch-0'
    checkpoint_epoch6 = exp_path / 'checkpoint-epoch-6'
    
    if not checkpoint_epoch0.exists() or not checkpoint_epoch6.exists():
        print(f"  ⚠️  Checkpoints not found!")
        return False
    
    is_lora = (exp_type == 'lora')
    
    print(f"  Processing {dataset}/{model}/{exp_type}...")
    
    # Load weights (function will look for numpy_weights subfolder)
    if matrix_type in ['k', 'q', 'v']:
        # Try numpy first
        weights_epoch0 = load_weights_from_numpy(checkpoint_epoch0, is_lora)
        weights_epoch6 = load_weights_from_numpy(checkpoint_epoch6, is_lora)
        
        # If empty, try safetensors
        if not weights_epoch0:
            weights_epoch0_st = load_weights_from_safetensors(checkpoint_epoch0, is_lora)
            weights_epoch6_st = load_weights_from_safetensors(checkpoint_epoch6, is_lora)
            weights_epoch0.update(weights_epoch0_st)
            weights_epoch6.update(weights_epoch6_st)
    else:
        # O and MLP only in safetensors
        weights_epoch0 = load_weights_from_safetensors(checkpoint_epoch0, is_lora)
        weights_epoch6 = load_weights_from_safetensors(checkpoint_epoch6, is_lora)
    
    if not weights_epoch0 or not weights_epoch6:
        print(f"  ⚠️  No weights found!")
        return False
    
    # Compute changes
    changes = compute_weight_changes(weights_epoch0, weights_epoch6, is_lora)
    
    if not changes:
        print(f"  ⚠️  No changes computed!")
        return False
    
    # Filter by matrix type
    if matrix_type == 'mlp':
        matrix_keys = ['mlp_gate', 'mlp_up', 'mlp_down']
    else:
        matrix_keys = [matrix_type]
    
    # Collect layers for this matrix type
    layer_changes = {}
    for key, delta in changes.items():
        for mk in matrix_keys:
            if f'_{mk}' in key:
                layer_match = re.search(r'layer(\d+)', key)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    if mk not in layer_changes:
                        layer_changes[mk] = {}
                    layer_changes[mk][layer_idx] = delta
    
    if not layer_changes:
        print(f"  ⚠️  No layer changes found!")
        return False
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For MLP, combine all 3 subtypes (gate, up, down) into ONE HTML file
    if matrix_type == 'mlp' and len(layer_changes) > 0:
        mlp_subtypes = ['mlp_gate', 'mlp_up', 'mlp_down']
        all_html_parts = []
        
        # Process each MLP subtype exactly like non-MLP matrices
        for mk in mlp_subtypes:
            if mk not in layer_changes:
                continue
                
            layers_dict = layer_changes[mk]
            num_layers = len(layers_dict)
            
            if num_layers == 0:
                continue
            
            # Create subplots - EXACTLY like K/Q/V/O
            fig = make_subplots(
                rows=num_layers, cols=1,
                subplot_titles=[f"Layer {i}" for i in sorted(layers_dict.keys())],
                vertical_spacing=0.01,
                horizontal_spacing=0.05
            )
            
            # FIRST PASS: Calculate min/max for THIS experiment (or use global if provided)
            if global_scales and 'mlp' in global_scales:
                global_min, global_max = global_scales['mlp']
            else:
                global_min = float('inf')
                global_max = float('-inf')
            
            all_downsampled = {}
            
            for layer_idx in sorted(layers_dict.keys()):
                delta = layers_dict[layer_idx]
                abs_delta = np.abs(delta)
                # Use 1/256 scale factor for MLP (smaller plots)
                target_shape = calculate_target_shape(abs_delta.shape, max_dim=max_dim, scale_factor=1.0/256.0)
                downsampled = downsample_matrix(abs_delta, target_shape)
                all_downsampled[layer_idx] = (downsampled, abs_delta, target_shape)
                
                if not global_scales:
                    global_min = min(global_min, np.min(downsampled))
                    global_max = max(global_max, np.max(downsampled))
            
            # SECOND PASS: Plot with shared scale
            for plot_idx, layer_idx in enumerate(sorted(layers_dict.keys()), start=1):
                downsampled, abs_delta, target_shape = all_downsampled[layer_idx]
                
                # Calculate statistics
                mean_val = np.mean(abs_delta)
                median_val = np.median(abs_delta)
                std_val = np.std(abs_delta)
                min_val = np.min(abs_delta)
                max_val = np.max(abs_delta)
                l2_norm = np.linalg.norm(abs_delta)
                
                # Add heatmap - Only FIRST heatmap shows colorbar
                show_colorbar = (plot_idx == 1)
                
                if show_colorbar:
                    row_height = 1.0 / num_layers
                    colorbar_y = 1.0 - 0.5 * row_height
                    colorbar_len = row_height * 0.7
                
                fig.add_trace(
                    go.Heatmap(
                        z=downsampled,
                        zmin=global_min,
                        zmax=global_max,
                        colorscale='Viridis',
                        showscale=show_colorbar,
                        hovertemplate='Row: %{y}<br>Col: %{x}<br>|Δw|: %{z:.6f}<extra></extra>',
                        colorbar=dict(
                            title=dict(text="|Δw|", side="right", font=dict(size=10)),
                            x=1.02,
                            len=colorbar_len,
                            y=colorbar_y,
                            yanchor="middle",
                            thickness=15,
                            tickfont=dict(size=9)
                        ) if show_colorbar else None
                    ),
                    row=plot_idx, col=1
                )
                
                # Add statistics annotation
                stats_text = (
                    f"<b>Stats</b><br>"
                    f"Mean: <b>{mean_val:.2e}</b><br>"
                    f"Median: {median_val:.2e}<br>"
                    f"Std: {std_val:.2e}<br>"
                    f"Min: {min_val:.2e}<br>"
                    f"Max: {max_val:.2e}<br>"
                    f"L2: {l2_norm:.2e}"
                )
                
                if plot_idx == 1:
                    yref_str = "y domain"
                else:
                    yref_str = f"y{plot_idx} domain"
                
                fig.add_annotation(
                    text=stats_text,
                    xref="paper",
                    yref=yref_str,
                    x=0.98,
                    y=0.98,
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=10, family="monospace"),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
                
                # Update axes
                fig.update_xaxes(title_text="Columns", row=plot_idx, col=1)
                fig.update_yaxes(title_text="Rows", row=plot_idx, col=1)
            
            # Get shapes for title
            original_shape = list(layers_dict.values())[0].shape
            target_shape = calculate_target_shape(original_shape, max_dim=max_dim, scale_factor=1.0/256.0)
            
            # Fixed height of 800px per layer
            height_per_layer = 800
            total_height = height_per_layer * num_layers + 200
            
            # Update layout - EXACTLY like K/Q/V/O
            mlp_name = mk.replace('mlp_', '').upper()
            title_text = f"MLP {mlp_name} Weight Changes at Epoch 6 (Per Layer)<br>{dataset.upper()} | {model} | {exp_type.upper()}<br><sub>Original: {original_shape}, Downsampled: {target_shape}</sub>"
            fig.update_layout(
                title_text=title_text,
                title_x=0.5,
                title_font=dict(size=24),
                width=1600,
                height=total_height,
                showlegend=False,
                font=dict(size=16),
                margin=dict(t=200, b=50, l=100, r=350)
            )
            
            # Update all axes for square cells
            for plot_idx, layer_idx in enumerate(sorted(layers_dict.keys()), start=1):
                _, _, target_shape = all_downsampled[layer_idx]
                
                if plot_idx == 1:
                    xref = "x"
                    yref = "y"
                else:
                    xref = f"x{plot_idx}"
                    yref = f"y{plot_idx}"
                
                fig.update_xaxes(
                    scaleanchor=yref,
                    scaleratio=1,
                    row=plot_idx,
                    col=1,
                    tickfont=dict(size=16),
                    title_font=dict(size=16),
                    range=[-0.5, target_shape[1] - 0.5],
                    constrain="domain"
                )
                fig.update_yaxes(
                    constrain="domain",
                    row=plot_idx,
                    col=1,
                    tickfont=dict(size=16),
                    title_font=dict(size=16),
                    range=[-0.5, target_shape[0] - 0.5]
                )
            
            # Convert figure to HTML (first one includes plotlyjs, rest don't)
            if len(all_html_parts) == 0:
                fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            else:
                fig_html = fig.to_html(full_html=False, include_plotlyjs=False)
            all_html_parts.append(fig_html)
        
        # Combine all MLP figures into one HTML file
        if all_html_parts:
            combined_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MLP Weight Changes - {dataset.upper()} | {model} | {exp_type.upper()}</title>
</head>
<body>
    <div style="text-align: center; padding: 30px; background-color: #e8e8e8;">
        <h1>MLP Weight Changes at Epoch 6</h1>
        <h2>{dataset.upper()} | {model} | {exp_type.upper()}</h2>
    </div>
    {''.join(all_html_parts)}
</body>
</html>
"""
            output_file = output_dir / f"{dataset}_{model}_{exp_type}_mlp_heatmap_epoch6.html"
            with open(output_file, 'w') as f:
                f.write(combined_html)
            print(f"  ✅ Saved: {output_file}")
        
        return True
    
    # For non-MLP matrices
    if matrix_type != 'mlp':
        # For non-MLP matrices, create individual plots (NOT subplots)
        for mk, layers_dict in layer_changes.items():
            num_layers = len(layers_dict)
            
            if num_layers == 0:
                continue
            
            # Create subplots (one per layer)
            rows = num_layers
            cols = 1
            
            subplot_titles = [f"Layer {i}" for i in sorted(layers_dict.keys())]
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=subplot_titles,
                vertical_spacing=0.02,
                horizontal_spacing=0.05
            )
            
            # FIRST PASS: Calculate min/max for THIS experiment (or use global if provided)
            if global_scales and mk in global_scales:
                # Use global scale across ALL experiments for this matrix type
                global_min, global_max = global_scales[mk]
            else:
                # Calculate local min/max for this experiment only
                global_min = float('inf')
                global_max = float('-inf')
            
            all_downsampled = {}
            
            for layer_idx in sorted(layers_dict.keys()):
                delta = layers_dict[layer_idx]
                abs_delta = np.abs(delta)
                target_shape = calculate_target_shape(abs_delta.shape, max_dim=max_dim)
                downsampled = downsample_matrix(abs_delta, target_shape)
                all_downsampled[layer_idx] = (downsampled, abs_delta, target_shape)
                
                if not global_scales:
                    # Only calculate if not using global scales
                    global_min = min(global_min, np.min(downsampled))
                    global_max = max(global_max, np.max(downsampled))
            
            # SECOND PASS: Plot with shared scale
            for plot_idx, layer_idx in enumerate(sorted(layers_dict.keys()), start=1):
                downsampled, abs_delta, target_shape = all_downsampled[layer_idx]
                
                # Calculate statistics for the original (not downsampled) data
                mean_val = np.mean(abs_delta)
                median_val = np.median(abs_delta)
                std_val = np.std(abs_delta)
                min_val = np.min(abs_delta)
                max_val = np.max(abs_delta)
                p25 = np.percentile(abs_delta, 25)
                p75 = np.percentile(abs_delta, 75)
                p95 = np.percentile(abs_delta, 95)
                p99 = np.percentile(abs_delta, 99)
                l2_norm = np.linalg.norm(abs_delta)
                
                # Add heatmap - Only FIRST heatmap shows colorbar (shared scale for all)
                show_colorbar = (plot_idx == 1)
                
                # Calculate colorbar position to align with first subplot only
                if show_colorbar:
                    # Position colorbar next to first subplot
                    row_height = 1.0 / num_layers
                    colorbar_y = 1.0 - 0.5 * row_height  # Center of first row
                    colorbar_len = row_height * 0.7  # 70% of first subplot height
                
                fig.add_trace(
                    go.Heatmap(
                        z=downsampled,
                        zmin=global_min,  # SHARED SCALE: same min for all layers
                        zmax=global_max,  # SHARED SCALE: same max for all layers
                        colorscale='Viridis',
                        showscale=show_colorbar,  # Only show ONE colorbar for all layers
                        hovertemplate='Row: %{y}<br>Col: %{x}<br>|Δw|: %{z:.6f}<extra></extra>',
                        colorbar=dict(
                            title=dict(text="|Δw|", side="right", font=dict(size=10)),
                            x=1.02,
                            len=colorbar_len,  # Sized to match first subplot
                            y=colorbar_y,  # Aligned with first subplot
                            yanchor="middle",
                            thickness=15,
                            tickfont=dict(size=9)
                        ) if show_colorbar else None
                    ),
                    row=plot_idx, col=1
                )
                
                # Add statistics annotation to the right of each heatmap
                stats_text = (
                    f"<b>Stats</b><br>"
                    f"Mean: <b>{mean_val:.2e}</b><br>"
                    f"Median: {median_val:.2e}<br>"
                    f"Std: {std_val:.2e}<br>"
                    f"Min: {min_val:.2e}<br>"
                    f"Max: {max_val:.2e}<br>"
                    f"L2: {l2_norm:.2e}"
                )
                
                # Determine xref and yref for the annotation
                if plot_idx == 1:
                    xref_str = "x domain"
                    yref_str = "y domain"
                else:
                    xref_str = f"x{plot_idx} domain"
                    yref_str = f"y{plot_idx} domain"
                
                fig.add_annotation(
                    text=stats_text,
                    xref="paper",  # Use paper coordinates for consistent positioning
                    yref=yref_str,
                    x=0.98,  # Right side of the paper (left of colorbar)
                    y=0.98,
                    xanchor="right",  # LEFT ALIGN (anchor on right means text extends left)
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=10, family="monospace"),
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
                
                # Update axes - ADD X-AXIS LABEL TO ALL PLOTS
                fig.update_xaxes(title_text="Columns", row=plot_idx, col=1)
                fig.update_yaxes(title_text="Rows", row=plot_idx, col=1)
            
            # Get shapes for title and calculate dynamic height
            original_shape = list(layers_dict.values())[0].shape
            target_shape = calculate_target_shape(original_shape, max_dim=max_dim)
            
            # Calculate height based on aspect ratio to maintain square cells
            # Fixed width available for plot area: 1600 - 100 (left) - 350 (right) = 1150
            # Single column gets most of this width: ~1000px
            # Height per layer = (target_shape[0] / target_shape[1]) * 1000
            aspect_ratio = target_shape[0] / target_shape[1]
            height_per_layer = int(aspect_ratio * 1000) + 100  # Dynamic based on aspect ratio
            total_height = height_per_layer * num_layers + 200
            
            # Update layout - MATCH HISTOGRAM STYLE with dynamic height
            title_text = f"{mk.upper()} Weight Changes at Epoch 6 (Per Layer)<br>{dataset.upper()} | {model} | {exp_type.upper()}<br><sub>Original: {original_shape}, Downsampled: {target_shape}</sub>"
            fig.update_layout(
                title_text=title_text,
                title_x=0.5,  # Centered title like histogram
                title_font=dict(size=24),
                title_pad=dict(t=100),
                width=1600,  # FIXED width like histogram
                height=total_height,  # DYNAMIC height based on aspect ratio
            showlegend=False,
            font=dict(size=16),  # Global font size like histogram
            margin=dict(t=200, b=50, l=100, r=350)  # More top margin for title with subtitle
        )
            
            # Update all axes to have equal aspect ratio (square cells) - MATCH HISTOGRAM FONTS
            for row_idx in range(1, num_layers + 1):
                fig.update_xaxes(
                    scaleanchor=f"y{row_idx if row_idx > 1 else ''}", 
                    scaleratio=1, 
                    row=row_idx, 
                    col=1,
                    tickfont=dict(size=16),  # Same as histogram
                    title_font=dict(size=16),  # Same as histogram
                    range=[-0.5, target_shape[1] - 0.5],
                    constrain="domain"
                )
                fig.update_yaxes(
                    constrain="domain", 
                    row=row_idx, 
                    col=1,
                    tickfont=dict(size=16),  # Same as histogram
                    title_font=dict(size=16),  # Same as histogram
                    range=[-0.5, target_shape[0] - 0.5],
                    scaleanchor=f"x{row_idx if row_idx > 1 else ''}"
                )
            
            # Save
            output_file = output_dir / f"{dataset}_{model}_{exp_type}_{matrix_type}_heatmap_epoch6.html"
            
            fig.write_html(str(output_file))
            print(f"  ✅ Saved: {output_file}")
    
    return True


def process_single_experiment(args_tuple):
    """Wrapper function for parallel processing."""
    dataset, model, exp_type, matrix_type, base_dir, output_dir, max_dim, global_scales = args_tuple
    try:
        print(f"Processing: {dataset}/{model}/{exp_type}/{matrix_type}")
        success = plot_heatmap_for_experiment(dataset, model, exp_type, base_dir, output_dir, matrix_type, max_dim, global_scales)
        return success
    except Exception as e:
        import traceback
        print(f"❌ Error processing {dataset}/{model}/{exp_type}/{matrix_type}: {e}")
        traceback.print_exc()
        return False


def calculate_global_scales(datasets, models, types, base_dir, max_dim):
    """
    PASS 1: Calculate global min/max for each matrix type across ALL experiments.
    Returns dict: {'k': (min, max), 'q': (min, max), ...}
    """
    print("\n" + "=" * 80)
    print("PASS 1: Calculating global scales for each matrix type...")
    print("=" * 80)
    
    scales = {
        'k': [float('inf'), float('-inf')],
        'q': [float('inf'), float('-inf')],
        'v': [float('inf'), float('-inf')],
        'o': [float('inf'), float('-inf')],
        'mlp': [float('inf'), float('-inf')]
    }
    
    for dataset in datasets:
        for model in models:
            for exp_type in types:
                exp_path = base_dir / dataset / model / exp_type / 'epoch_weights'
                checkpoint_epoch0 = exp_path / 'checkpoint-epoch-0'
                checkpoint_epoch6 = exp_path / 'checkpoint-epoch-6'
                
                if not checkpoint_epoch0.exists() or not checkpoint_epoch6.exists():
                    continue
                
                is_lora = (exp_type == 'lora')
                
                # Load all weights (function will look for numpy_weights subfolder)
                weights_epoch0 = load_weights_from_numpy(checkpoint_epoch0, is_lora)
                weights_epoch6 = load_weights_from_numpy(checkpoint_epoch6, is_lora)
                
                # Also load from safetensors
                weights_epoch0_st = load_weights_from_safetensors(checkpoint_epoch0, is_lora)
                weights_epoch6_st = load_weights_from_safetensors(checkpoint_epoch6, is_lora)
                weights_epoch0.update(weights_epoch0_st)
                weights_epoch6.update(weights_epoch6_st)
                
                changes = compute_weight_changes(weights_epoch0, weights_epoch6, is_lora=is_lora)
                
                # Reorganize changes by matrix type (same as main plotting function)
                matrix_keys = ['k', 'q', 'v', 'o', 'mlp_gate', 'mlp_up', 'mlp_down']
                layer_changes = {}
                for key, delta in changes.items():
                    for mk in matrix_keys:
                        if f'_{mk}' in key:
                            layer_match = re.search(r'layer(\d+)', key)
                            if layer_match:
                                layer_idx = int(layer_match.group(1))
                                if mk not in layer_changes:
                                    layer_changes[mk] = {}
                                layer_changes[mk][layer_idx] = delta
                
                # Process each matrix type
                for matrix_type in ['k', 'q', 'v', 'o', 'mlp']:
                    if matrix_type == 'mlp':
                        # MLP: check all three sub-matrices
                        for mk in ['mlp_gate', 'mlp_up', 'mlp_down']:
                            if mk in layer_changes:
                                for layer_idx, delta in layer_changes[mk].items():
                                    abs_delta = np.abs(delta)
                                    target_shape = calculate_target_shape(abs_delta.shape, max_dim=max_dim)
                                    downsampled = downsample_matrix(abs_delta, target_shape)
                                    scales['mlp'][0] = min(scales['mlp'][0], np.min(downsampled))
                                    scales['mlp'][1] = max(scales['mlp'][1], np.max(downsampled))
                    else:
                        # K, Q, V, O
                        if matrix_type in layer_changes:
                            for layer_idx, delta in layer_changes[matrix_type].items():
                                abs_delta = np.abs(delta)
                                target_shape = calculate_target_shape(abs_delta.shape, max_dim=max_dim)
                                downsampled = downsample_matrix(abs_delta, target_shape)
                                scales[matrix_type][0] = min(scales[matrix_type][0], np.min(downsampled))
                                scales[matrix_type][1] = max(scales[matrix_type][1], np.max(downsampled))
    
    # Convert to tuples and print
    global_scales = {k: (v[0], v[1]) for k, v in scales.items()}
    
    print("\nGlobal scales calculated:")
    for matrix_type, (vmin, vmax) in global_scales.items():
        print(f"  {matrix_type.upper()}: min={vmin:.6e}, max={vmax:.6e}")
    
    return global_scales


def main():
    parser = argparse.ArgumentParser(
        description="Plot heatmaps of weight changes at epoch 6"
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
        default="/home/kadir/topo/plots/weight_heatmaps",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(8, cpu_count()),
        help="Number of parallel jobs (default: min(8, cpu_count()))"
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=64,
        help="Maximum dimension for downsampling (default: 64, keeps aspect ratio)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("PLOTTING WEIGHT CHANGE HEATMAPS (PARALLEL)")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Types: {args.types}")
    print(f"Max dimension: {args.max_dim} (1/64 ratio)")
    print(f"Parallel jobs: {args.n_jobs}")
    print("=" * 80)
    
    # Skip global scales for now - loading safetensors twice is too slow
    # Each experiment will use its own scale (still shared across layers within that experiment)
    global_scales = None
    
    print("\nGenerating heatmaps (each experiment uses its own scale)...")
    print("=" * 80)
    
    # Process each matrix type separately (combine MLP into one)
    matrix_types = ['k', 'q', 'v', 'o', 'mlp']
    
    # Create list of all tasks
    tasks = []
    for dataset in args.datasets:
        for model in args.models:
            for exp_type in args.types:
                for matrix_type in matrix_types:
                    tasks.append((dataset, model, exp_type, matrix_type, base_dir, output_dir, args.max_dim, global_scales))
    
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Processing with {args.n_jobs} parallel workers...\n")
    
    # Process in parallel
    with Pool(processes=args.n_jobs) as pool:
        results = pool.map(process_single_experiment, tasks)
    
    # Summary
    successful = sum(results)
    print("\n" + "=" * 80)
    print(f"DONE! Successfully generated {successful}/{len(tasks)} heatmaps")
    print("=" * 80)


if __name__ == "__main__":
    main()
