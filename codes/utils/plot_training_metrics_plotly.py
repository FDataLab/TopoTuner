#!/usr/bin/env python3
"""
Plot training metrics (loss, grad_norm, learning_rate) vs epoch for SQuAD and HotpotQA Llama-3.2-3B experiments.

Experiments:
- Full (1)
- K+O (4: lowest3, highest3, lowest15, highest15)
- K+O+MLP (4: lowest3, highest3, lowest15, highest15)
"""

import re
import json
import ast
import numpy as np
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = Path("/home/kadir/topo")
LOGS_DIR = BASE_DIR / "logs"
MAX_EPOCH = 3.0

# Dataset configurations
DATASET_CONFIGS = {
    "squad": {
        "name": "SQuAD",
        "model": "llama32_3b",
        "experiments": {
            "Full": {
                "log_file": "finetune_squad_llama32_3b_k_o_mlp_full.log",
                "name": "Full",
                "color": "#000000",  # black - bold baseline
                "line_style": "solid",
            },
            "K+O": {
                "experiments": [
                    ("k_o_lowest3", "K+O Lowest 3", "#1f77b4"),  # blue
                    ("k_o_highest3", "K+O Highest 3", "#ff7f0e"),  # orange
                    ("k_o_lowest15", "K+O Lowest 15", "#2ca02c"),  # green
                    ("k_o_highest15", "K+O Highest 15", "#d62728"),  # red
                ],
                "log_pattern": "finetune_squad_llama32_3b_k_o_mlp_k_o_{exp_name}.log",
            },
            "K+O+MLP": {
                "experiments": [
                    ("k_o_mlp_lowest3", "K+O+MLP Lowest 3", "#9467bd"),  # purple (darker blue family)
                    ("k_o_mlp_highest3", "K+O+MLP Highest 3", "#8c564b"),  # brown (darker orange)
                    ("k_o_mlp_lowest15", "K+O+MLP Lowest 15", "#7f7f7f"),  # gray (darker green)
                    ("k_o_mlp_highest15", "K+O+MLP Highest 15", "#e377c2"),  # pink (darker red)
                ],
                "log_pattern": "finetune_squad_llama32_3b_k_o_mlp_k_o_mlp_{exp_name}.log",
            },
        },
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "model": "llama32_3b",
        "experiments": {
            "Full": {
                "log_file": "finetune_hotpotqa_llama32_3b_full.log",
                "name": "Full",
                "color": "#000000",  # black - bold baseline
                "line_style": "solid",
            },
            "K+O": {
                "experiments": [
                    ("k_o_lowest3", "K+O Lowest 3", "#1f77b4"),  # blue
                    ("k_o_highest3", "K+O Highest 3", "#ff7f0e"),  # orange
                    ("k_o_lowest15", "K+O Lowest 15", "#2ca02c"),  # green
                    ("k_o_highest15", "K+O Highest 15", "#d62728"),  # red
                ],
                "log_pattern": "finetune_hotpotqa_llama32_3b_k_o_{exp_name}.log",
            },
            "K+O+MLP": {
                "experiments": [
                    ("k_o_mlp_lowest3", "K+O+MLP Lowest 3", "#9467bd"),  # purple (darker blue family)
                    ("k_o_mlp_highest3", "K+O+MLP Highest 3", "#8c564b"),  # brown (darker orange)
                    ("k_o_mlp_lowest15", "K+O+MLP Lowest 15", "#7f7f7f"),  # gray (darker green)
                    ("k_o_mlp_highest15", "K+O+MLP Highest 15", "#e377c2"),  # pink (darker red)
                ],
                "log_pattern": "finetune_hotpotqa_llama32_3b_k_o_mlp_k_o_mlp_{exp_name}.log",
            },
        },
    },
}


# ==========================
# LOG PARSING
# ==========================
def parse_log_file(log_path: Path) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Parse training log file to extract epoch, loss, grad_norm, and learning_rate.
    
    Returns:
        (epochs, losses, grad_norms, learning_rates)
    """
    epochs = []
    losses = []
    grad_norms = []
    learning_rates = []
    
    if not log_path.exists():
        print(f"⚠️  Warning: Log file not found: {log_path}")
        return epochs, losses, grad_norms, learning_rates
    
    # Pattern to match Python dict-like dictionaries in log lines
    log_pattern = re.compile(r"\{.*?\}")
    
    with open(log_path, 'r') as f:
        for line in f:
            m = log_pattern.search(line)
            if not m:
                continue
            
            try:
                # Try parsing as Python dict (single quotes) first
                dict_str = m.group()
                data = ast.literal_eval(dict_str)
            except (ValueError, SyntaxError):
                try:
                    # Fallback to JSON (double quotes)
                    data = json.loads(dict_str.replace("'", '"'))
                except (json.JSONDecodeError, ValueError):
                    continue
            
            ep = float(data.get("epoch", -1))
            if 0 <= ep <= MAX_EPOCH:
                epochs.append(ep)
                losses.append(float(data.get("loss", np.nan)))
                grad_norms.append(float(data.get("grad_norm", np.nan)))
                learning_rates.append(float(data.get("learning_rate", np.nan)))
    
    return epochs, losses, grad_norms, learning_rates


# ==========================
# DATA COLLECTION
# ==========================
def collect_experiment_data(dataset: str) -> Dict[str, Dict]:
    """Collect data for all experiments for a given dataset."""
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset]
    experiments = config["experiments"]
    all_data = {}
    
    # Full experiment
    full_config = experiments["Full"]
    log_path = LOGS_DIR / full_config["log_file"]
    epochs, losses, grad_norms, lrs = parse_log_file(log_path)
    
    if epochs:
        all_data["Full"] = {
            "name": full_config["name"],
            "color": full_config["color"],
            "line_style": full_config["line_style"],
            "epochs": epochs,
            "losses": losses,
            "grad_norms": grad_norms,
            "learning_rates": lrs,
        }
        print(f"✅ Full: {len(epochs)} data points")
    else:
        print(f"⚠️  Full: No data found")
    
    # K+O experiments
    k_o_config = experiments["K+O"]
    for exp_name, display_name, color in k_o_config["experiments"]:
        # Extract the suffix (lowest3, highest3, etc.)
        suffix = exp_name.replace("k_o_", "")
        log_file = k_o_config["log_pattern"].format(exp_name=suffix)
        log_path = LOGS_DIR / log_file
        epochs, losses, grad_norms, lrs = parse_log_file(log_path)
        
        if epochs:
            all_data[exp_name] = {
                "name": display_name,
                "color": color,
                "line_style": "solid",
                "epochs": epochs,
                "losses": losses,
                "grad_norms": grad_norms,
                "learning_rates": lrs,
            }
            print(f"✅ {display_name}: {len(epochs)} data points")
        else:
            print(f"⚠️  {display_name}: No data found")
    
    # K+O+MLP experiments
    k_o_mlp_config = experiments["K+O+MLP"]
    for exp_name, display_name, color in k_o_mlp_config["experiments"]:
        # Extract the suffix (lowest3, highest3, etc.)
        suffix = exp_name.replace("k_o_mlp_", "")
        log_file = k_o_mlp_config["log_pattern"].format(exp_name=suffix)
        log_path = LOGS_DIR / log_file
        epochs, losses, grad_norms, lrs = parse_log_file(log_path)
        
        if epochs:
            all_data[exp_name] = {
                "name": display_name,
                "color": color,
                "line_style": "dash",
                "epochs": epochs,
                "losses": losses,
                "grad_norms": grad_norms,
                "learning_rates": lrs,
            }
            print(f"✅ {display_name}: {len(epochs)} data points")
        else:
            print(f"⚠️  {display_name}: No data found")
    
    return all_data


# ==========================
# PLOTTING
# ==========================
def create_plots(all_data: Dict[str, Dict], output_dir: Path, dataset: str, model: str = "llama32_3b"):
    """Create three separate plots for loss, grad_norm, and learning_rate.
    
    Args:
        all_data: Dictionary of experiment data
        output_dir: Output directory for plots
        dataset: Dataset name (squad or hotpotqa)
        model: Model name (default: llama32_3b)
    """
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = DATASET_CONFIGS[dataset]["name"]
    
    # Helper function to smooth curves
    def smooth_curve(x, y, num_points=300):
        """Smooth curve using spline interpolation."""
        if len(x) < 3:
            return x, y
        try:
            # Sort by x
            sorted_indices = np.argsort(x)
            x_sorted = np.array([x[i] for i in sorted_indices])
            y_sorted = np.array([y[i] for i in sorted_indices])
            
            # Remove duplicates in x
            unique_indices = np.unique(x_sorted, return_index=True)[1]
            x_unique = x_sorted[unique_indices]
            y_unique = y_sorted[unique_indices]
            
            if len(x_unique) < 3:
                return x, y
            
            # Create smooth curve
            x_smooth = np.linspace(x_unique.min(), x_unique.max(), num_points)
            spline = make_interp_spline(x_unique, y_unique, k=min(3, len(x_unique)-1))
            y_smooth = spline(x_smooth)
            return x_smooth.tolist(), y_smooth.tolist()
        except:
            return x, y
    
    # Separate experiments by type
    full_exps = {k: v for k, v in all_data.items() if k == "Full"}
    k_o_exps = {k: v for k, v in all_data.items() if k.startswith("k_o_") and not k.startswith("k_o_mlp_")}
    k_o_mlp_exps = {k: v for k, v in all_data.items() if k.startswith("k_o_mlp_")}
    
    # Plot configurations (excluding LR - will only be in combined plot)
    plot_configs = [
        {
            "metric": "loss",
            "title": f"{dataset_name} Llama-3.2-3B: Training Loss vs Epoch",
            "y_label": "Loss",
            "y_axis": "losses",
            "filename": f"{dataset}_{model}_training_loss.html",
            "smooth": True,
            "log_scale": False,
        },
        {
            "metric": "grad_norm",
            "title": f"{dataset_name} Llama-3.2-3B: Gradient Norm vs Epoch",
            "y_label": "Gradient Norm",
            "y_axis": "grad_norms",
            "filename": f"{dataset}_{model}_gradient_norm.html",
            "smooth": False,
            "log_scale": False,
        },
    ]
    
    for config in plot_configs:
        fig = go.Figure()
        
        # Add Full experiment first (no offset)
        offset_idx = 0
        for exp_key, exp_data in full_exps.items():
            epochs = exp_data["epochs"]
            y_values = exp_data[config["y_axis"]]
            
            valid_indices = [i for i, v in enumerate(y_values) if not np.isnan(v)]
            if not valid_indices:
                continue
            
            epochs_clean = [epochs[i] for i in valid_indices]
            y_clean = [y_values[i] for i in valid_indices]
            
            # Smooth if requested
            if config["smooth"]:
                epochs_plot, y_plot = smooth_curve(epochs_clean, y_clean)
            else:
                epochs_plot, y_plot = epochs_clean, y_clean
            
            # No offset for Full
            epochs_offset = epochs_plot
            
            fig.add_trace(go.Scatter(
                x=epochs_offset,
                y=y_plot,
                mode='lines',
                name=exp_data["name"],
                line=dict(
                    color=exp_data["color"],
                    width=1.5,
                ),
                opacity=1.0,
                legendgroup="full",
            ))
            offset_idx += 1
        
        # Add K+O experiments (solid lines, slight negative offsets)
        for idx, (exp_key, exp_data) in enumerate(k_o_exps.items()):
            epochs = exp_data["epochs"]
            y_values = exp_data[config["y_axis"]]
            
            valid_indices = [i for i, v in enumerate(y_values) if not np.isnan(v)]
            if not valid_indices:
                continue
            
            epochs_clean = [epochs[i] for i in valid_indices]
            y_clean = [y_values[i] for i in valid_indices]
            
            if config["smooth"]:
                epochs_plot, y_plot = smooth_curve(epochs_clean, y_clean)
            else:
                epochs_plot, y_plot = epochs_clean, y_clean
            
            # Small negative offset to spread out
            offset = -0.015 * (idx + 1)
            epochs_offset = [e + offset for e in epochs_plot]
            
            fig.add_trace(go.Scatter(
                x=epochs_offset,
                y=y_plot,
                mode='lines',
                name=exp_data["name"],
                line=dict(
                    color=exp_data["color"],
                    width=1.5,
                ),
                opacity=0.7,
                legendgroup="k_o",
            ))
        
        # Add K+O+MLP experiments (solid lines, slight positive offsets)
        for idx, (exp_key, exp_data) in enumerate(k_o_mlp_exps.items()):
            epochs = exp_data["epochs"]
            y_values = exp_data[config["y_axis"]]
            
            valid_indices = [i for i, v in enumerate(y_values) if not np.isnan(v)]
            if not valid_indices:
                continue
            
            epochs_clean = [epochs[i] for i in valid_indices]
            y_clean = [y_values[i] for i in valid_indices]
            
            if config["smooth"]:
                epochs_plot, y_plot = smooth_curve(epochs_clean, y_clean)
            else:
                epochs_plot, y_plot = epochs_clean, y_clean
            
            # Small positive offset to spread out
            offset = 0.015 * (idx + 1)
            epochs_offset = [e + offset for e in epochs_plot]
            
            fig.add_trace(go.Scatter(
                x=epochs_offset,
                y=y_plot,
                mode='lines',
                name=exp_data["name"],
                line=dict(
                    color=exp_data["color"],
                    width=1.5,
                ),
                opacity=0.7,
                legendgroup="k_o_mlp",
            ))
        
        # Update layout
        yaxis_config = dict(
            showgrid=True,
            gridcolor='lightgray',
        )
        
        fig.update_layout(
            title=dict(
                text=config["title"],
                x=0.5,
                font=dict(size=18),
            ),
            xaxis_title="Epoch",
            yaxis_title=config["y_label"],
            xaxis=dict(
                range=[0, MAX_EPOCH],
                dtick=0.5,
                showgrid=True,
                gridcolor='lightgray',
            ),
            yaxis=yaxis_config,
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1,
            ),
            width=1000,
            height=600,
            margin=dict(r=200),  # Space for legend
            template='plotly_white',
        )
        
        # Save plot
        output_path = output_dir / config["filename"]
        fig.write_html(str(output_path))
        print(f"✅ Saved: {output_path}")
    
    # Create combined plot with subplots
    create_combined_plot(all_data, output_dir, dataset, model)


def create_combined_plot(all_data: Dict[str, Dict], output_dir: Path, dataset: str, model: str = "llama32_3b"):
    """Create a combined plot with 3 subplots (Loss, Grad Norm, Learning Rate).
    
    Args:
        all_data: Dictionary of experiment data
        output_dir: Output directory for plots
        dataset: Dataset name (squad or hotpotqa)
        model: Model name (default: llama32_3b)
    """
    dataset_name = DATASET_CONFIGS[dataset]["name"]
    
    # Helper function to smooth curves
    def smooth_curve(x, y, num_points=300):
        """Smooth curve using spline interpolation."""
        if len(x) < 3:
            return x, y
        try:
            sorted_indices = np.argsort(x)
            x_sorted = np.array([x[i] for i in sorted_indices])
            y_sorted = np.array([y[i] for i in sorted_indices])
            unique_indices = np.unique(x_sorted, return_index=True)[1]
            x_unique = x_sorted[unique_indices]
            y_unique = y_sorted[unique_indices]
            if len(x_unique) < 3:
                return x, y
            x_smooth = np.linspace(x_unique.min(), x_unique.max(), num_points)
            spline = make_interp_spline(x_unique, y_unique, k=min(3, len(x_unique)-1))
            y_smooth = spline(x_smooth)
            return x_smooth.tolist(), y_smooth.tolist()
        except:
            return x, y
    
    # Separate experiments by type
    full_exps = {k: v for k, v in all_data.items() if k == "Full"}
    k_o_exps = {k: v for k, v in all_data.items() if k.startswith("k_o_") and not k.startswith("k_o_mlp_")}
    k_o_mlp_exps = {k: v for k, v in all_data.items() if k.startswith("k_o_mlp_")}
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Training Loss", "Gradient Norm", "Learning Rate"),
        vertical_spacing=0.08,
        shared_xaxes=True,
    )
    
    # Add traces for each metric
    metrics = [
        ("losses", "Loss", 1, True),   # Loss: smooth
        ("grad_norms", "Gradient Norm", 2, False),  # Grad norm: no smooth
        ("learning_rates", "Learning Rate", 3, False),  # LR: no smooth
    ]
    
    for metric_key, metric_name, row, should_smooth in metrics:
        # Add Full experiment first (no offset)
        for exp_key, exp_data in full_exps.items():
            epochs = exp_data["epochs"]
            y_values = exp_data[metric_key]
            
            valid_indices = [i for i, v in enumerate(y_values) if not np.isnan(v)]
            if not valid_indices:
                continue
            
            epochs_clean = [epochs[i] for i in valid_indices]
            y_clean = [y_values[i] for i in valid_indices]
            
            if should_smooth:
                epochs_plot, y_plot = smooth_curve(epochs_clean, y_clean)
            else:
                epochs_plot, y_plot = epochs_clean, y_clean
            
            # No offset for Full
            epochs_offset = epochs_plot
            
            fig.add_trace(go.Scatter(
                x=epochs_offset,
                y=y_plot,
                mode='lines',
                name=exp_data["name"],
                line=dict(
                    color=exp_data["color"],
                    width=1.5,
                ),
                opacity=1.0,
                legendgroup="full",
                showlegend=(row == 1),
            ), row=row, col=1)
        
        # Add K+O experiments (solid, slight negative offsets)
        for idx, (exp_key, exp_data) in enumerate(k_o_exps.items()):
            epochs = exp_data["epochs"]
            y_values = exp_data[metric_key]
            
            valid_indices = [i for i, v in enumerate(y_values) if not np.isnan(v)]
            if not valid_indices:
                continue
            
            epochs_clean = [epochs[i] for i in valid_indices]
            y_clean = [y_values[i] for i in valid_indices]
            
            if should_smooth:
                epochs_plot, y_plot = smooth_curve(epochs_clean, y_clean)
            else:
                epochs_plot, y_plot = epochs_clean, y_clean
            
            # Small negative offset to spread out
            offset = -0.015 * (idx + 1)
            epochs_offset = [e + offset for e in epochs_plot]
            
            fig.add_trace(go.Scatter(
                x=epochs_offset,
                y=y_plot,
                mode='lines',
                name=exp_data["name"],
                line=dict(
                    color=exp_data["color"],
                    width=1.5,
                ),
                opacity=0.7,
                legendgroup="k_o",
                showlegend=(row == 1),
            ), row=row, col=1)
        
        # Add K+O+MLP experiments (solid, slight positive offsets)
        for idx, (exp_key, exp_data) in enumerate(k_o_mlp_exps.items()):
            epochs = exp_data["epochs"]
            y_values = exp_data[metric_key]
            
            valid_indices = [i for i, v in enumerate(y_values) if not np.isnan(v)]
            if not valid_indices:
                continue
            
            epochs_clean = [epochs[i] for i in valid_indices]
            y_clean = [y_values[i] for i in valid_indices]
            
            if should_smooth:
                epochs_plot, y_plot = smooth_curve(epochs_clean, y_clean)
            else:
                epochs_plot, y_plot = epochs_clean, y_clean
            
            # Small positive offset to spread out
            offset = 0.015 * (idx + 1)
            epochs_offset = [e + offset for e in epochs_plot]
            
            fig.add_trace(go.Scatter(
                x=epochs_offset,
                y=y_plot,
                mode='lines',
                name=exp_data["name"],
                line=dict(
                    color=exp_data["color"],
                    width=1.5,
                ),
                opacity=0.7,
                legendgroup="k_o_mlp",
                showlegend=(row == 1),
            ), row=row, col=1)
    
    # Update axes
    fig.update_xaxes(
        title_text="Epoch",
        range=[0, MAX_EPOCH],
        dtick=0.5,
        showgrid=True,
        gridcolor='lightgray',
        row=3, col=1,
    )
    
    fig.update_yaxes(title_text="Loss", showgrid=True, gridcolor='lightgray', row=1, col=1)
    fig.update_yaxes(title_text="Gradient Norm", showgrid=True, gridcolor='lightgray', row=2, col=1)
    fig.update_yaxes(title_text="Learning Rate", showgrid=True, gridcolor='lightgray', row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{dataset_name} Llama-3.2-3B Training Metrics (Epoch 0-3)",
            x=0.5,
            font=dict(size=18),
        ),
        height=1200,
        width=1000,
        margin=dict(r=200),  # Space for legend
        template='plotly_white',
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
        ),
    )
    
    # Save combined plot
    output_path = output_dir / f"{dataset}_{model}_training_metrics_combined.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved combined plot: {output_path}")


# ==========================
# MAIN
# ==========================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training metrics for SQuAD and HotpotQA")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["squad", "hotpotqa", "all"],
        default=["all"],
        help="Datasets to plot (default: all)"
    )
    args = parser.parse_args()
    
    # Determine which datasets to process
    if "all" in args.datasets:
        datasets_to_process = ["squad", "hotpotqa"]
    else:
        datasets_to_process = args.datasets
    
    for dataset in datasets_to_process:
        print("=" * 70)
        dataset_name = DATASET_CONFIGS[dataset]["name"]
        model = DATASET_CONFIGS[dataset]["model"]
        print(f"{dataset_name} Llama-3.2-3B Training Metrics Plotter")
        print("=" * 70)
        print(f"Max epoch: {MAX_EPOCH}")
        print()
        
        # Collect data
        print(f"Collecting data from log files for {dataset_name}...")
        all_data = collect_experiment_data(dataset)
        print()
        
        if not all_data:
            print(f"❌ No data found for {dataset_name}! Please check log file paths.")
            print()
            continue
        
        # Create plots
        output_dir = BASE_DIR / "plots" / "training_metrics" / dataset
        print(f"Creating plots in: {output_dir}")
        print()
        create_plots(all_data, output_dir, dataset, model)
        
        print()
        print("=" * 70)
        print(f"✅ All plots generated successfully for {dataset_name}!")
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()
