#!/usr/bin/env python3
"""
Plot training metrics (loss, grad_norm, learning_rate) for HotpotQA Llama-3.1-8B O/Q/V Frozen experiments.
Run 1 only, Epochs 0-2.
"""

import re
import json
import ast
import numpy as np
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = Path("/home/kadir/topo")
LOGS_DIR = BASE_DIR / "logs"
MAX_EPOCH = 2.0

# Experiments configuration
EXPERIMENTS = [
    ("k_mlp_full", "Full", "#9467bd"),
    ("k_mlp_lowest3", "L3", "#1f77b4"),
    ("k_mlp_highest3", "H3", "#ff7f0e"),
    ("k_mlp_lowest15", "L15", "#2ca02c"),
    ("k_mlp_highest15", "H15", "#d62728"),
]

# Markers for each experiment
MARKERS = ['star', 'circle', 'square', 'triangle-up', 'diamond']

# Line styles
DASH_STYLES = ["dot", None, "dash", None, "dash"]

# Line widths
LINE_WIDTHS = [3.5, 3.5, 3.0, 3.5, 3.0]


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
def collect_experiment_data() -> Dict[str, Dict]:
    """Collect data for all Run 1 experiments."""
    all_data = {}
    
    for exp_name, display_name, color in EXPERIMENTS:
        log_file = f"finetune_hotpotqa_llama31_8b_oqv_frozen_{exp_name}.log"
        log_path = LOGS_DIR / log_file
        epochs, losses, grad_norms, lrs = parse_log_file(log_path)
        
        if epochs:
            all_data[exp_name] = {
                "name": display_name,
                "color": color,
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
def create_individual_plots(all_data: Dict[str, Dict], output_dir: Path):
    """Create three separate plots for loss, grad_norm, and learning_rate."""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Plot configurations
    plot_configs = [
        {
            "metric": "loss",
            "title": "HotpotQA Llama-3.1-8B O/Q/V Frozen: Training Loss (Run 1)",
            "y_label": "Loss",
            "y_axis": "losses",
            "filename": "hotpotqa_llama31_8b_oqv_frozen_run1_training_loss.html",
            "smooth": True,
        },
        {
            "metric": "grad_norm",
            "title": "HotpotQA Llama-3.1-8B O/Q/V Frozen: Gradient Norm (Run 1)",
            "y_label": "Gradient Norm",
            "y_axis": "grad_norms",
            "filename": "hotpotqa_llama31_8b_oqv_frozen_run1_gradient_norm.html",
            "smooth": False,
        },
        {
            "metric": "learning_rate",
            "title": "HotpotQA Llama-3.1-8B O/Q/V Frozen: Learning Rate (Run 1)",
            "y_label": "Learning Rate",
            "y_axis": "learning_rates",
            "filename": "hotpotqa_llama31_8b_oqv_frozen_run1_learning_rate.html",
            "smooth": False,
        },
    ]
    
    for config in plot_configs:
        fig = go.Figure()
        
        # Add all experiments
        for idx, (exp_key, exp_data) in enumerate(all_data.items()):
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
            
            # Get styling
            marker_shape = MARKERS[idx]
            dash_style = DASH_STYLES[idx]
            line_width = LINE_WIDTHS[idx]
            
            fig.add_trace(go.Scatter(
                x=epochs_plot,
                y=y_plot,
                mode='lines+markers',
                name=exp_data["name"],
                marker=dict(
                    symbol=marker_shape,
                    size=8,
                    color=exp_data["color"],
                    line=dict(width=1, color='white')
                ),
                line=dict(
                    color=exp_data["color"],
                    width=line_width,
                    dash=dash_style
                ),
                hovertemplate=f'<b>{exp_data["name"]}</b><br>' +
                             'Epoch: %{x:.2f}<br>' +
                             f'{config["y_label"]}: %{{y:.4f}}<br>' +
                             '<extra></extra>',
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=config["title"],
                x=0.5,
                font=dict(size=24, family="Arial Black")
            ),
            xaxis_title="Epoch",
            yaxis_title=config["y_label"],
            xaxis=dict(
                range=[0, MAX_EPOCH],
                dtick=0.5,
                showgrid=True,
                gridcolor='lightgray',
                tickfont=dict(size=16),
                title_font=dict(size=18),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                tickfont=dict(size=16),
                title_font=dict(size=18),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=20),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1,
            ),
            width=1400,
            height=700,
            template='plotly_white',
            font=dict(family="Arial", size=16),
        )
        
        # Save plot
        output_path = output_dir / config["filename"]
        fig.write_html(str(output_path))
        print(f"✅ Saved: {output_path}")


def create_combined_plot(all_data: Dict[str, Dict], output_dir: Path):
    """Create a combined plot with 3 subplots (Loss, Grad Norm, Learning Rate)."""
    
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
        # Add all experiments
        for idx, (exp_key, exp_data) in enumerate(all_data.items()):
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
            
            # Get styling
            marker_shape = MARKERS[idx]
            dash_style = DASH_STYLES[idx]
            line_width = LINE_WIDTHS[idx]
            
            fig.add_trace(go.Scatter(
                x=epochs_plot,
                y=y_plot,
                mode='lines+markers',
                name=exp_data["name"],
                marker=dict(
                    symbol=marker_shape,
                    size=8,
                    color=exp_data["color"],
                    line=dict(width=1, color='white')
                ),
                line=dict(
                    color=exp_data["color"],
                    width=line_width,
                    dash=dash_style
                ),
                hovertemplate=f'<b>{exp_data["name"]}</b><br>' +
                             'Epoch: %{x:.2f}<br>' +
                             f'{metric_name}: %{{y:.4f}}<br>' +
                             '<extra></extra>',
                legendgroup=exp_data["name"],
                showlegend=(row == 1),
            ), row=row, col=1)
    
    # Update axes
    fig.update_xaxes(
        title_text="Epoch",
        range=[0, MAX_EPOCH],
        dtick=0.5,
        showgrid=True,
        gridcolor='lightgray',
        tickfont=dict(size=16),
        title_font=dict(size=18),
        row=3, col=1,
    )
    
    fig.update_yaxes(title_text="Loss", showgrid=True, gridcolor='lightgray', tickfont=dict(size=16), title_font=dict(size=18), row=1, col=1)
    fig.update_yaxes(title_text="Gradient Norm", showgrid=True, gridcolor='lightgray', tickfont=dict(size=16), title_font=dict(size=18), row=2, col=1)
    fig.update_yaxes(title_text="Learning Rate", showgrid=True, gridcolor='lightgray', tickfont=dict(size=16), title_font=dict(size=18), row=3, col=1)
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        if annotation['text'] in ['Training Loss', 'Gradient Norm', 'Learning Rate']:
            annotation['y'] = annotation['y'] + 0.02
            annotation['font'] = dict(size=18, family="Arial Black")
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="HotpotQA Llama-3.1-8B O/Q/V Frozen: Training Metrics (Run 1, Epoch 0-2)",
            x=0.5,
            font=dict(size=24, family="Arial Black")
        ),
        height=1200,
        width=1400,
        template='plotly_white',
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=20),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1,
        ),
    )
    
    # Save combined plot
    output_path = output_dir / "hotpotqa_llama31_8b_oqv_frozen_run1_training_metrics_combined.html"
    fig.write_html(str(output_path))
    print(f"✅ Saved combined plot: {output_path}")


# ==========================
# MAIN
# ==========================
def main():
    print("="*80)
    print("HotpotQA Llama-3.1-8B O/Q/V Frozen Training Metrics Plotter")
    print("Run 1, Epochs 0-2")
    print("="*80)
    print()
    
    # Collect data
    print("Collecting data from log files...")
    all_data = collect_experiment_data()
    print()
    
    if not all_data:
        print("❌ No data found! Please check log file paths.")
        return
    
    # Create plots
    output_dir = BASE_DIR / "plots" / "training_metrics" / "hotpotqa_llama31_8b_oqv_frozen"
    print(f"Creating plots in: {output_dir}")
    print()
    
    print("Creating individual plots...")
    create_individual_plots(all_data, output_dir)
    print()
    
    print("Creating combined plot...")
    create_combined_plot(all_data, output_dir)
    print()
    
    print("="*80)
    print("✅ All plots generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
