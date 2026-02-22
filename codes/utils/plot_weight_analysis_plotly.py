#!/usr/bin/env python3
"""
Plot weight distance analysis for HotpotQA and SQuAD experiments using Plotly.

Computes normalized L2 distance between epoch-0 (baseline) and epoch-i weights
for Q, K, V projections separately across all layers.

X-axis: Layer index
Y-axis: Normalized weight distance (L2 norm of difference / L2 norm of baseline)
"""

import os
import sys
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def parse_layer_proj(filename: str) -> Optional[Tuple[int, str]]:
    """
    Parse layer number and projection type from filename.
    
    Examples: 
        layer0_q.npy -> (0, 'q')
        layer0_o.npy -> (0, 'o')
        layer0_gate.npy -> (0, 'gate')
        layer0_up.npy -> (0, 'up')
        layer0_down.npy -> (0, 'down')
    """
    # Match q, k, v, o, gate, up, down
    match = re.match(r"layer(\d+)_(q|k|v|o|gate|up|down)\.npy", filename)
    if match:
        layer = int(match.group(1))
        proj = match.group(2)
        return (layer, proj)
    return None


def load_weights_from_dir(npy_dir: str) -> Dict[Tuple[int, str], np.ndarray]:
    """
    Load all weight files (Q/K/V/O/MLP) from a numpy_weights directory.
    
    Returns:
        Dictionary mapping (layer, proj) -> weight array
    """
    weights = {}
    npy_path = Path(npy_dir)
    
    if not npy_path.exists():
        print(f"[warn] Directory not found: {npy_dir}")
        return weights
    
    for filename in npy_path.glob("*.npy"):
        parsed = parse_layer_proj(filename.name)
        if parsed is None:
            continue
        
        layer, proj = parsed
        try:
            weight = np.load(filename)
            weights[(layer, proj)] = weight
        except Exception as e:
            print(f"[warn] Failed to load {filename}: {e}")
            continue
    
    return weights


def normalized_l2_diff(w_i: np.ndarray, w_0: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute normalized L2 distance between two weight matrices.
    
    Formula: ||w_i - w_0|| / (||w_0|| + eps)
    """
    diff = np.linalg.norm(w_i - w_0)
    base = np.linalg.norm(w_0)
    return diff / (base + eps)


def compute_layer_distances(
    weights_0: Dict[Tuple[int, str], np.ndarray],
    weights_i: Dict[Tuple[int, str], np.ndarray],
    proj_type: str
) -> List[Tuple[int, float]]:
    """
    Compute normalized L2 distances for a specific projection type.
    
    Args:
        weights_0: Epoch-0 weights
        weights_i: Epoch-i weights
        proj_type: Projection type (q, k, v, o, gate, up, down)
    
    Returns:
        List of (layer_index, distance) tuples, sorted by layer index
    """
    distances = []
    
    # Get all layers that have this projection type in both epoch 0 and epoch i
    layers_0 = {layer for (layer, proj) in weights_0.keys() if proj == proj_type}
    layers_i = {layer for (layer, proj) in weights_i.keys() if proj == proj_type}
    common_layers = sorted(layers_0 & layers_i)
    
    for layer in common_layers:
        key_0 = (layer, proj_type)
        key_i = (layer, proj_type)
        
        if key_0 in weights_0 and key_i in weights_i:
            w0 = weights_0[key_0]
            wi = weights_i[key_i]
            
            # Flatten for distance calculation
            w0_flat = w0.reshape(-1)
            wi_flat = wi.reshape(-1)
            
            dist = normalized_l2_diff(wi_flat, w0_flat)
            distances.append((layer, dist))
    
    return distances


def plot_weight_analysis_plotly(
    epoch_0_dir: str,
    epoch_i_dir: str,
    output_path: str,
    title: str = "Weight Distance Analysis",
    dataset: str = "",
    experiment: str = "",
    epoch: int = 0
):
    """
    Plot weight distance analysis comparing epoch-0 (baseline) vs epoch-i.
    
    Args:
        epoch_0_dir: Path to epoch-0 numpy_weights directory
        epoch_i_dir: Path to epoch-i numpy_weights directory
        output_path: Output HTML path
        title: Plot title
        dataset: Dataset name (e.g., "HotpotQA", "SQuAD")
        experiment: Experiment name (e.g., "full", "k_o_lowest3")
        epoch: Epoch number for epoch_i
    """
    # Load weights
    print(f"[load] Loading epoch-0 weights from: {epoch_0_dir}")
    weights_0 = load_weights_from_dir(epoch_0_dir)
    print(f"[load] Found {len(weights_0)} weight files in epoch-0")
    
    print(f"[load] Loading epoch-{epoch} weights from: {epoch_i_dir}")
    weights_i = load_weights_from_dir(epoch_i_dir)
    print(f"[load] Found {len(weights_i)} weight files in epoch-{epoch}")
    
    if not weights_0 or not weights_i:
        print(f"[error] Missing weights! epoch-0: {len(weights_0)}, epoch-{epoch}: {len(weights_i)}")
        return
    
    # Compute distances for Q, K, V separately
    q_distances = compute_layer_distances(weights_0, weights_i, 'q')
    k_distances = compute_layer_distances(weights_0, weights_i, 'k')
    v_distances = compute_layer_distances(weights_0, weights_i, 'v')
    
    print(f"[compute] Q distances: {len(q_distances)} layers")
    print(f"[compute] K distances: {len(k_distances)} layers")
    print(f"[compute] V distances: {len(v_distances)} layers")
    
    if not q_distances and not k_distances and not v_distances:
        print("[error] No distances computed!")
        return
    
    # Create plotly figure
    fig = go.Figure()
    
    # Colors for Q, K, V
    colors = {
        'q': '#1f77b4',  # Blue
        'k': '#ff7f0e',  # Orange
        'v': '#2ca02c',  # Green
    }
    
    # Plot Q, K, V separately
    if q_distances:
        layers_q = [l for l, _ in q_distances]
        dists_q = [d for _, d in q_distances]
        fig.add_trace(
            go.Scatter(
                x=layers_q,
                y=dists_q,
                mode='lines+markers',
                name='Query (Q)',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color=colors['q'],
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=colors['q'],
                    width=3.5
                ),
                hovertemplate='<b>Query (Q)</b><br>' +
                             'Layer: %{x}<br>' +
                             'Distance: %{y:.6f}<br>' +
                             '<extra></extra>',
            )
        )
    
    if k_distances:
        layers_k = [l for l, _ in k_distances]
        dists_k = [d for _, d in k_distances]
        fig.add_trace(
            go.Scatter(
                x=layers_k,
                y=dists_k,
                mode='lines+markers',
                name='Key (K)',
                marker=dict(
                    symbol='square',
                    size=10,
                    color=colors['k'],
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=colors['k'],
                    width=3.5
                ),
                hovertemplate='<b>Key (K)</b><br>' +
                             'Layer: %{x}<br>' +
                             'Distance: %{y:.6f}<br>' +
                             '<extra></extra>',
            )
        )
    
    if v_distances:
        layers_v = [l for l, _ in v_distances]
        dists_v = [d for _, d in v_distances]
        fig.add_trace(
            go.Scatter(
                x=layers_v,
                y=dists_v,
                mode='lines+markers',
                name='Value (V)',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color=colors['v'],
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=colors['v'],
                    width=3.5
                ),
                hovertemplate='<b>Value (V)</b><br>' +
                             'Layer: %{x}<br>' +
                             'Distance: %{y:.6f}<br>' +
                             '<extra></extra>',
            )
        )
    
    # Build title
    full_title = title
    if dataset and experiment:
        full_title = f"{dataset} - {experiment}: Weight Distance (Epoch {epoch} vs Epoch 0)"
    elif dataset:
        full_title = f"{dataset}: Weight Distance (Epoch {epoch} vs Epoch 0)"
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=full_title,
            x=0.5,
            font=dict(size=20, family="Arial Black")
        ),
        xaxis=dict(
            title="Layer Index",
            dtick=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Normalized Weight Distance",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        height=700,
        width=1400,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        template="plotly_white",
        font=dict(family="Arial", size=12)
    )
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"\n✅ Plot saved to: {output_path}")


def find_experiment_dirs(base_dir: str, dataset: str, experiment: str, model: str = "llama32_3b") -> Optional[Tuple[str, List[str]]]:
    """
    Find epoch-0 directory and all epoch-i directories for an experiment.
    
    Args:
        base_dir: Base directory (e.g., /home/kadir/topo)
        dataset: Dataset name (e.g., "squad", "hotpotqa")
        experiment: Experiment name (e.g., "full", "k_o_lowest3")
        model: Model name (default: "llama32_3b")
    
    Returns:
        (epoch_0_dir, [epoch_i_dirs]) or None if not found
    """
    base_path = Path(base_dir)
    
    # Primary path: numpy_weights/{dataset}/{model}/{experiment}/epoch_weights
    epoch_weights_dir = base_path / "numpy_weights" / dataset / model / experiment / "epoch_weights"
    
    if not epoch_weights_dir.exists():
        print(f"[error] Could not find epoch_weights directory: {epoch_weights_dir}")
        return None
    
    # Find epoch-0
    epoch_0_dir = epoch_weights_dir / "checkpoint-epoch-0" / "numpy_weights"
    if not epoch_0_dir.exists():
        print(f"[error] Epoch-0 directory not found: {epoch_0_dir}")
        return None
    
    # Find all other epochs
    epoch_dirs = []
    for checkpoint_dir in sorted(epoch_weights_dir.glob("checkpoint-epoch-*")):
        if checkpoint_dir.name == "checkpoint-epoch-0":
            continue
        npy_dir = checkpoint_dir / "numpy_weights"
        if npy_dir.exists():
            epoch_dirs.append(str(npy_dir))
    
    return (str(epoch_0_dir), epoch_dirs)


def plot_all_epochs_separate_proj(
    base_dir: str,
    dataset: str,
    experiment: str,
    output_base_dir: str,
    model: str = "llama32_3b"
):
    """
    Plot weight analysis for all epochs, with separate plots for Q, K, V.
    Each plot shows all epochs as separate lines.
    """
    result = find_experiment_dirs(base_dir, dataset, experiment, model=model)
    if result is None:
        return
    
    epoch_0_dir, epoch_i_dirs = result
    
    print(f"[info] Found {len(epoch_i_dirs)} epochs to plot")
    
    # Load epoch-0 weights once
    print(f"[load] Loading epoch-0 weights from: {epoch_0_dir}")
    weights_0 = load_weights_from_dir(epoch_0_dir)
    print(f"[load] Found {len(weights_0)} weight files in epoch-0")
    
    if not weights_0:
        print("[error] No epoch-0 weights found!")
        return
    
    # Collect all epoch data
    all_epoch_data = {}  # {epoch: {proj: [(layer, dist), ...]}}
    
    for epoch_i_dir in epoch_i_dirs:
        # Extract epoch number from path
        match = re.search(r"checkpoint-epoch-(\d+)", epoch_i_dir)
        if not match:
            continue
        
        epoch = int(match.group(1))
        
        print(f"[load] Loading epoch-{epoch} weights from: {epoch_i_dir}")
        weights_i = load_weights_from_dir(epoch_i_dir)
        print(f"[load] Found {len(weights_i)} weight files in epoch-{epoch}")
        
        if not weights_i:
            continue
        
        # Compute distances for Q, K, V
        q_distances = compute_layer_distances(weights_0, weights_i, 'q')
        k_distances = compute_layer_distances(weights_0, weights_i, 'k')
        v_distances = compute_layer_distances(weights_0, weights_i, 'v')
        # o_distances = compute_layer_distances(weights_0, weights_i, 'o')
        # gate_distances = compute_layer_distances(weights_0, weights_i, 'gate')
        # up_distances = compute_layer_distances(weights_0, weights_i, 'up')
        # down_distances = compute_layer_distances(weights_0, weights_i, 'down')
        
        all_epoch_data[epoch] = {
            'q': q_distances,
            'k': k_distances,
            'v': v_distances,
            # 'o': o_distances,
            # 'gate': gate_distances,
            # 'up': up_distances,
            # 'down': down_distances
        }
    
    if not all_epoch_data:
        print("[error] No epoch data collected!")
        return
    
    # Colors for different epochs (using a color palette)
    epoch_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]
    
    # Markers for different epochs
    epoch_markers = ['circle', 'square', 'triangle-up', 'diamond', 'star', 'hexagon', 'pentagon', 'cross']
    
    # Define all projection types and their display names
    proj_configs = {
        'q': 'Query (Q)',
        'k': 'Key (K)',
        'v': 'Value (V)',
        # 'o': 'Output (O)',
        # 'gate': 'MLP Gate',
        # 'up': 'MLP Up',
        # 'down': 'MLP Down'
    }
    
    # Create separate plots for each projection type
    for proj_type, proj_name in proj_configs.items():
        fig = go.Figure()
        
        # Add a trace for each epoch
        for idx, epoch in enumerate(sorted(all_epoch_data.keys())):
            distances = all_epoch_data[epoch][proj_type]
            if not distances:
                continue
            
            layers = [l for l, _ in distances]
            dists = [d for _, d in distances]
            
            color = epoch_colors[idx % len(epoch_colors)]
            marker = epoch_markers[idx % len(epoch_markers)]
            
            fig.add_trace(
                go.Scatter(
                    x=layers,
                    y=dists,
                    mode='lines+markers',
                    name=f'Epoch {epoch}',
                    marker=dict(
                        symbol=marker,
                        size=10,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    line=dict(
                        color=color,
                        width=3.0
                    ),
                    hovertemplate=f'<b>Epoch {epoch}</b><br>' +
                                 'Layer: %{x}<br>' +
                                 'Distance: %{y:.6f}<br>' +
                                 '<extra></extra>',
                )
            )
        
        # Build title
        full_title = f"{dataset.upper()} - {experiment}: {proj_name} Weight Distance (vs Epoch 0)"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=full_title,
                x=0.5,
                font=dict(size=20, family="Arial Black")
            ),
            xaxis=dict(
                title="Layer Index",
                dtick=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title="Normalized Weight Distance",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            height=700,
            width=1400,
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            template="plotly_white",
            font=dict(family="Arial", size=12)
        )
        
        # Generate output path - use a different folder for final results
        proj_suffix = proj_type
        output_path = os.path.join(
            output_base_dir,
            dataset.lower(),
            "final",  # New folder for final results
            f"{dataset.lower()}_{model}_{experiment}_{proj_suffix}_all_epochs_weight_analysis.html"
        )
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Plot saved to: {output_path}")


def plot_all_experiments_per_epoch(
    base_dir: str,
    dataset: str,
    experiments: List[str],
    output_base_dir: str,
    model: str = "llama32_3b"
):
    """
    Plot all experiments together for each epoch separately.
    Creates plots showing: full, k_o_lowest3, k_o_highest3, etc. all on one plot per epoch.
    Separate plots for Q, K, V.
    """
    # Load epoch-0 weights for all experiments (use full as reference)
    full_result = find_experiment_dirs(base_dir, dataset, "full", model=model)
    if full_result is None:
        print("[error] Could not find 'full' experiment for epoch-0 reference")
        return
    
    epoch_0_dir, _ = full_result
    
    # Collect data for all experiments
    all_experiments_data = {}  # {experiment: {epoch: {proj: [(layer, dist), ...]}}}
    
    for exp_name in experiments:
        result = find_experiment_dirs(base_dir, dataset, exp_name, model=model)
        if result is None:
            print(f"[warn] Skipping {exp_name} - not found")
            continue
        
        _, epoch_i_dirs = result
        
        # Load epoch-0 for this experiment
        exp_epoch_0_dir = result[0]
        weights_0 = load_weights_from_dir(exp_epoch_0_dir)
        
        if not weights_0:
            print(f"[warn] No epoch-0 weights for {exp_name}")
            continue
        
        all_experiments_data[exp_name] = {}
        
        for epoch_i_dir in epoch_i_dirs:
            match = re.search(r"checkpoint-epoch-(\d+)", epoch_i_dir)
            if not match:
                continue
            
            epoch = int(match.group(1))
            weights_i = load_weights_from_dir(epoch_i_dir)
            
            if not weights_i:
                continue
            
            q_distances = compute_layer_distances(weights_0, weights_i, 'q')
            k_distances = compute_layer_distances(weights_0, weights_i, 'k')
            v_distances = compute_layer_distances(weights_0, weights_i, 'v')
            
            all_experiments_data[exp_name][epoch] = {
                'q': q_distances,
                'k': k_distances,
                'v': v_distances
            }
    
    if not all_experiments_data:
        print("[error] No experiment data collected!")
        return
    
    # Get all unique epochs across all experiments
    all_epochs = set()
    for exp_data in all_experiments_data.values():
        all_epochs.update(exp_data.keys())
    all_epochs = sorted(all_epochs)
    
    # Unique colors for each experiment
    exp_colors = {
        'full': '#1f77b4',  # Blue
        'k_o_lowest3': '#ff7f0e',  # Orange
        'k_o_highest3': '#2ca02c',  # Green
        'k_o_lowest15': '#d62728',  # Red
        'k_o_highest15': '#9467bd',  # Purple
        'k_o_mlp_lowest3': '#8c564b',  # Brown
        'k_o_mlp_highest3': '#e377c2',  # Pink
        'k_o_mlp_lowest15': '#7f7f7f',  # Gray
        'k_o_mlp_highest15': '#bcbd22',  # Yellow-green
    }
    
    # Line styles: full=dotted, k_o=solid, k_o_mlp=dashed
    exp_line_styles = {
        'full': 'dot',
        'k_o_lowest3': None,  # solid
        'k_o_highest3': None,
        'k_o_lowest15': None,
        'k_o_highest15': None,
        'k_o_mlp_lowest3': 'dash',
        'k_o_mlp_highest3': 'dash',
        'k_o_mlp_lowest15': 'dash',
        'k_o_mlp_highest15': 'dash',
    }
    
    exp_labels = {
        'full': 'Full',
        'k_o_lowest3': 'K+O Lowest 3',
        'k_o_highest3': 'K+O Highest 3',
        'k_o_lowest15': 'K+O Lowest 15',
        'k_o_highest15': 'K+O Highest 15',
        'k_o_mlp_lowest3': 'K+O+MLP Lowest 3',
        'k_o_mlp_highest3': 'K+O+MLP Highest 3',
        'k_o_mlp_lowest15': 'K+O+MLP Lowest 15',
        'k_o_mlp_highest15': 'K+O+MLP Highest 15',
    }
    
    # Create plots for each epoch, with separate plots for Q, K, V
    for epoch in all_epochs:
        for proj_type in ['q', 'k', 'v']:
            proj_name = {'q': 'Query (Q)', 'k': 'Key (K)', 'v': 'Value (V)'}[proj_type]
            fig = go.Figure()
            
            # Add a trace for each experiment that has data for this epoch
            for exp_name in experiments:
                if exp_name not in all_experiments_data:
                    continue
                if epoch not in all_experiments_data[exp_name]:
                    continue
                
                distances = all_experiments_data[exp_name][epoch][proj_type]
                if not distances:
                    continue
                
                layers = [l for l, _ in distances]
                dists = [d for _, d in distances]
                
                color = exp_colors.get(exp_name, '#000000')
                line_style = exp_line_styles.get(exp_name, None)
                label = exp_labels.get(exp_name, exp_name)
                
                fig.add_trace(
                    go.Scatter(
                        x=layers,
                        y=dists,
                        mode='lines+markers',
                        name=label,
                        marker=dict(
                            symbol='circle',
                            size=8,
                            color=color,
                            line=dict(width=1, color='white')
                        ),
                        line=dict(
                            color=color,
                            width=3.0,
                            dash=line_style  # None=solid, 'dot'=dotted, 'dash'=dashed
                        ),
                        hovertemplate=f'<b>{label}</b><br>' +
                                     'Layer: %{x}<br>' +
                                     'Distance: %{y:.6f}<br>' +
                                     '<extra></extra>',
                    )
                )
            
            # Build title
            full_title = f"{dataset.upper()} - Epoch {epoch}: {proj_name} Weight Distance (vs Epoch 0)"
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=full_title,
                    x=0.5,
                    font=dict(size=20, family="Arial Black")
                ),
                xaxis=dict(
                    title="Layer Index",
                    dtick=1,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title="Normalized Weight Distance",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                height=700,
                width=1400,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                template="plotly_white",
                font=dict(family="Arial", size=12)
            )
            
            # Generate output path - use a different folder for final results
            output_path = os.path.join(
                output_base_dir,
                dataset.lower(),
                "final",  # New folder for final results
                f"{dataset.lower()}_{model}_epoch{epoch}_all_experiments_{proj_type}_weight_analysis.html"
            )
            
            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            print(f"✅ Plot saved to: {output_path}")


def plot_all_epochs_for_experiment(
    base_dir: str,
    dataset: str,
    experiment: str,
    output_base_dir: str,
    model: str = "llama32_3b"
):
    """
    Plot weight analysis for all epochs of an experiment.
    Uses the new format: separate plots for Q, K, V with all epochs on each.
    """
    plot_all_epochs_separate_proj(
        base_dir=base_dir,
        dataset=dataset,
        experiment=experiment,
        output_base_dir=output_base_dir,
        model=model
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot weight distance analysis for HotpotQA and SQuAD experiments"
    )
    parser.add_argument(
        "--epoch-0-dir",
        type=str,
        help="Path to epoch-0 numpy_weights directory"
    )
    parser.add_argument(
        "--epoch-i-dir",
        type=str,
        help="Path to epoch-i numpy_weights directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./plots/weight_analysis/weight_analysis.html",
        help="Output HTML path"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Weight Distance Analysis",
        help="Plot title"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name (e.g., HotpotQA, SQuAD)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        help="Experiment name (e.g., full, k_o_lowest3)"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Epoch number for epoch-i"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/kadir/topo",
        help="Base directory for finding experiments"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama32_3b",
        help="Model name (e.g., llama32_3b, llama31_8b)"
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot all epochs for a given dataset/experiment"
    )
    parser.add_argument(
        "--plot-all-experiments",
        action="store_true",
        help="Plot all experiments together for each epoch"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="List of experiments to plot (e.g., full k_o_lowest3 k_o_highest3)"
    )
    
    args = parser.parse_args()
    
    if args.plot_all_experiments:
        if not args.dataset:
            print("[error] --dataset required for --plot-all-experiments")
            return
        
        # Default experiments if not specified
        if args.experiments is None:
            experiments = [
                "full",
                "k_o_lowest3", "k_o_highest3",
                "k_o_lowest15", "k_o_highest15",
                "k_o_mlp_lowest3", "k_o_mlp_highest3",
                "k_o_mlp_lowest15", "k_o_mlp_highest15"
            ]
        else:
            experiments = args.experiments
        
        # Determine output base directory - remove "final" if it's already in the path
        output_base = os.path.dirname(args.output) or "./plots/weight_analysis"
        if output_base.endswith("/final"):
            output_base = output_base[:-6]  # Remove "/final"
        
        print(f"[info] Plotting all experiments together: {experiments}")
        plot_all_experiments_per_epoch(
            base_dir=args.base_dir,
            dataset=args.dataset,
            experiments=experiments,
            output_base_dir=output_base,
            model=args.model
        )
        
        # Also plot each experiment separately with all epochs
        print(f"[info] Plotting each experiment separately with all epochs")
        for exp in experiments:
            print(f"[info] Processing {exp}...")
            plot_all_epochs_for_experiment(
                base_dir=args.base_dir,
                dataset=args.dataset,
                experiment=exp,
                output_base_dir=output_base,
                model=args.model
            )
    elif args.plot_all:
        if not args.dataset or not args.experiment:
            print("[error] --dataset and --experiment required for --plot-all")
            return
        
        plot_all_epochs_for_experiment(
            base_dir=args.base_dir,
            dataset=args.dataset,
            experiment=args.experiment,
            output_base_dir=os.path.dirname(args.output) or "./plots/weight_analysis",
            model=args.model
        )
    else:
        if not args.epoch_0_dir or not args.epoch_i_dir:
            print("[error] --epoch-0-dir and --epoch-i-dir required (or use --plot-all)")
            return
        
        plot_weight_analysis_plotly(
            epoch_0_dir=args.epoch_0_dir,
            epoch_i_dir=args.epoch_i_dir,
            output_path=args.output,
            title=args.title,
            dataset=args.dataset,
            experiment=args.experiment,
            epoch=args.epoch
        )


if __name__ == "__main__":
    main()
