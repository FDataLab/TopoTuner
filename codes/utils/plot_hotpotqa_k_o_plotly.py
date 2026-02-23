#!/usr/bin/env python3
"""
Plot HotpotQA K+O evaluation results using Plotly.
Plots all 4 K+O experiments + baseline (full) in one interactive plot.

Shows both F1 and EM scores over epochs.
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from codes.utils.eval_plots import infer_epoch


def load_hotpotqa_points(csv_path: str, metric: str = "f1"):
    """
    Load HotpotQA evaluation points from CSV.
    
    Args:
        csv_path: Path to CSV file
        metric: 'em' or 'f1'
    
    Returns:
        List of (epoch, value) tuples
    """
    if not os.path.exists(csv_path):
        print(f"[skip] {csv_path} not found")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[skip] {csv_path} is empty")
            return []
    except Exception as e:
        print(f"[skip] Error reading {csv_path}: {e}")
        return []
    
    if "checkpoint" not in df.columns or metric not in df.columns:
        print(f"[skip] Missing columns in {csv_path} (has: {df.columns.tolist()})")
        return []
    
    points = []
    for _, row in df.iterrows():
        ep = infer_epoch(row["checkpoint"])
        value = float(row[metric])
        if ep >= 0:
            points.append((ep, value))
    
    points.sort(key=lambda x: x[0])
    return points


def plot_hotpotqa_k_o_plotly(csv_dir: str, output_path: str, ymin: float = None, ymax: float = None):
    """
    Plot HotpotQA K+O evaluation results with Plotly.
    
    Args:
        csv_dir: Directory containing CSV files
        output_path: Output HTML path
        ymin, ymax: Y-axis limits (None for automatic scaling)
    """
    # Define experiments: 4 K+O + 1 baseline (Llama-3.1-8B)
    experiments = [
        ("hotpotqa_llama31_8b_k_o_lowest3", "K+O Lowest 3", "#1f77b4"),
        ("hotpotqa_llama31_8b_k_o_highest3", "K+O Highest 3", "#ff7f0e"),
        ("hotpotqa_llama31_8b_k_o_lowest15", "K+O Lowest 15", "#2ca02c"),
        ("hotpotqa_llama31_8b_k_o_highest15", "K+O Highest 15", "#d62728"),
        ("hotpotqa_metrics_llama31_8b", "Baseline (Full)", "#9467bd"),
    ]
    
    # Different markers for each line
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'star']
    
    # Line styles: lowest=solid, highest=dashed, full=dotted
    dash_styles = [None, "dash", None, "dash", "dot"]
    
    # Line widths
    line_widths = [3.5, 3.0, 3.5, 3.0, 3.5]
    
    # Create subplots: top for F1, bottom for EM
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
        shared_xaxes=True,
        subplot_titles=('F1 Score', 'Exact Match (EM) Score'),
        x_title="Epoch"
    )
    
    all_epochs = set()
    all_f1_values = []
    all_em_values = []
    plotted_count = 0
    
    # Plot both F1 and EM for each experiment
    for idx, (exp_name, label, color) in enumerate(experiments):
        # Try different possible CSV paths
        csv_paths = [
            os.path.join(csv_dir, f"{exp_name}.csv"),
            os.path.join(csv_dir, "hotpotqa", f"{exp_name}.csv"),
            os.path.join(os.path.dirname(csv_dir), f"{exp_name}.csv"),
        ]
        
        csv_path = None
        for path in csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path:
            print(f"[skip] {exp_name} not found in any of: {csv_paths}")
            continue
        
        # Load F1 points
        f1_points = load_hotpotqa_points(csv_path, metric="f1")
        if not f1_points:
            print(f"[skip] No F1 data in {csv_path}")
            continue
        
        # Load EM points
        em_points = load_hotpotqa_points(csv_path, metric="em")
        if not em_points:
            print(f"[skip] No EM data in {csv_path}")
            continue
        
        f1_epochs = [e for e, _ in f1_points]
        f1_values = [v for _, v in f1_points]
        em_epochs = [e for e, _ in em_points]
        em_values = [v for _, v in em_points]
        
        all_epochs.update(f1_epochs)
        all_epochs.update(em_epochs)
        all_f1_values.extend(f1_values)
        all_em_values.extend(em_values)
        
        marker_shape = markers[idx]
        dash_style = dash_styles[idx]
        line_width = line_widths[idx]
        
        # Add F1 trace (top subplot, row 1)
        fig.add_trace(
            go.Scatter(
                x=f1_epochs,
                y=f1_values,
                mode='lines+markers',
                name=label,
                marker=dict(
                    symbol=marker_shape,
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=color,
                    width=line_width,
                    dash=dash_style
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Epoch: %{x}<br>' +
                             'F1: %{y:.2f}%<br>' +
                             '<extra></extra>',
                legendgroup=label,
                showlegend=True,
            ),
            row=1, col=1
        )
        
        # Add EM trace (bottom subplot, row 2)
        fig.add_trace(
            go.Scatter(
                x=em_epochs,
                y=em_values,
                mode='lines+markers',
                name=label,
                marker=dict(
                    symbol=marker_shape,
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=color,
                    width=line_width,
                    dash=dash_style
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Epoch: %{x}<br>' +
                             'EM: %{y:.2f}%<br>' +
                             '<extra></extra>',
                legendgroup=label,
                showlegend=False,  # Only show legend once
            ),
            row=2, col=1
        )
        
        # Find and annotate max values
        if f1_values:
            max_f1_idx = max(range(len(f1_values)), key=lambda i: f1_values[i])
            max_f1_epoch = f1_epochs[max_f1_idx]
            max_f1_value = f1_values[max_f1_idx]
            
            fig.add_annotation(
                x=max_f1_epoch,
                y=max_f1_value,
                text=f"{max_f1_value:.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                ax=0,
                ay=-30,
                bgcolor="white",
                bordercolor=color,
                borderwidth=2,
                font=dict(size=10, color=color, family="Arial Black"),
                row=1, col=1
            )
        
        if em_values:
            max_em_idx = max(range(len(em_values)), key=lambda i: em_values[i])
            max_em_epoch = em_epochs[max_em_idx]
            max_em_value = em_values[max_em_idx]
            
            fig.add_annotation(
                x=max_em_epoch,
                y=max_em_value,
                text=f"{max_em_value:.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                ax=0,
                ay=-30,
                bgcolor="white",
                bordercolor=color,
                borderwidth=2,
                font=dict(size=10, color=color, family="Arial Black"),
                row=2, col=1
            )
        
        plotted_count += 1
        print(f"[ok] Plotted {label}: F1 max={max(f1_values):.2f}%, EM max={max(em_values):.2f}%")
    
    if plotted_count == 0:
        print("❌ No data to plot!")
        return
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="HotpotQA Evaluation: Llama-3.1-8B K+O Freezing Experiments",
            x=0.5,
            font=dict(size=20, family="Arial Black")
        ),
        height=900,
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
    
    # Update y-axes with automatic scaling if not specified
    f1_range = None
    em_range = None
    if ymin is not None and ymax is not None:
        f1_range = [ymin, ymax]
        em_range = [ymin, ymax]
    elif all_f1_values and all_em_values:
        # Automatic scaling with small padding
        f1_min, f1_max = min(all_f1_values), max(all_f1_values)
        em_min, em_max = min(all_em_values), max(all_em_values)
        f1_padding = (f1_max - f1_min) * 0.1
        em_padding = (em_max - em_min) * 0.1
        f1_range = [max(0, f1_min - f1_padding), f1_max + f1_padding]
        em_range = [max(0, em_min - em_padding), em_max + em_padding]
    
    fig.update_yaxes(
        title_text="F1 Score (%)",
        range=f1_range,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="EM Score (%)",
        range=em_range,
        row=2, col=1
    )
    
    # Update x-axes
    if all_epochs:
        min_epoch = min(all_epochs)
        max_epoch = max(all_epochs)
        fig.update_xaxes(
            range=[min_epoch - 0.5, max_epoch + 0.5],
            dtick=1,
            row=1, col=1
        )
        fig.update_xaxes(
            range=[min_epoch - 0.5, max_epoch + 0.5],
            dtick=1,
            row=2, col=1
        )
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"\n✅ Plot saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot HotpotQA K+O evaluation results with Plotly")
    parser.add_argument(
        "--csv-dir",
        default="./evaluation_results/hotpotqa",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output",
        default="./plots/hotpotqa/hotpotqa_llama31_8b_k_o_plotly.html",
        help="Output HTML path"
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Y-axis minimum (None for automatic)"
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Y-axis maximum (None for automatic)"
    )
    
    args = parser.parse_args()
    
    plot_hotpotqa_k_o_plotly(args.csv_dir, args.output, args.ymin, args.ymax)


if __name__ == "__main__":
    main()
