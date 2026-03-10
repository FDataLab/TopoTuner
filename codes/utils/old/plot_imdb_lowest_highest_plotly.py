#!/usr/bin/env python3
"""
Plot IMDB lowest/highest evaluation results using Plotly.
Plots all lowest/highest 3,6,9,12,15 experiments + baseline (full) in one interactive plot.

Shows accuracy over epochs.
Line styles: lowest=solid, highest=dashed, full=dotted
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from codes.utils.eval_plots import infer_epoch


def load_imdb_points(csv_path: str):
    """
    Load IMDB evaluation points from CSV.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of (epoch, accuracy) tuples
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
    
    # Check for "acc" column (some might have "accuracy")
    acc_col = None
    if "acc" in df.columns:
        acc_col = "acc"
    elif "accuracy" in df.columns:
        acc_col = "accuracy"
    else:
        print(f"[skip] Missing 'acc' or 'accuracy' column in {csv_path} (has: {df.columns.tolist()})")
        return []
    
    if "checkpoint" not in df.columns:
        print(f"[skip] Missing 'checkpoint' column in {csv_path} (has: {df.columns.tolist()})")
        return []
    
    points = []
    for _, row in df.iterrows():
        ep = infer_epoch(row["checkpoint"])
        acc = float(row[acc_col])
        # Convert to percentage if needed (some CSVs have 0.9455, others have 94.55)
        if acc < 1.0:
            acc = acc * 100
        if ep >= 0:
            points.append((ep, acc))
    
    points.sort(key=lambda x: x[0])
    return points


def plot_imdb_lowest_highest_plotly(csv_dir: str, output_path: str, ymin: float = None, ymax: float = None):
    """
    Plot IMDB lowest/highest evaluation results with Plotly.
    
    Args:
        csv_dir: Directory containing CSV files
        output_path: Output HTML path
        ymin, ymax: Y-axis limits (None for automatic scaling)
    """
    # Define experiments: 3 experiments (k_o_lowest6, k_o_lowest9, full)
    experiments = [
        ("llama32_3b_imdb_k_o_lowest6", "K+O Lowest 6", "#1f77b4"),
        ("llama32_3b_imdb_k_o_lowest9", "K+O Lowest 9", "#ff7f0e"),
        ("llama32_3b_imdb_full", "Baseline (Full)", "#000000"),
    ]
    
    # Different markers for each line
    markers = ['circle', 'square', 'x']
    
    # Line styles: lowest=solid, full=dotted
    dash_styles = [
        None,      # k_o_lowest6 - solid
        None,      # k_o_lowest9 - solid
        "dot",     # full - dotted
    ]
    
    # Line widths
    line_widths = [3.5, 3.0, 3.5]
    
    # Create single subplot for accuracy
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Accuracy (%)',),
        x_title="Epoch"
    )
    
    all_epochs = set()
    all_acc_values = []
    plotted_count = 0
    
    # Plot accuracy for each experiment
    for idx, (exp_name, label, color) in enumerate(experiments):
        # Try different possible CSV paths
        csv_paths = [
            os.path.join(csv_dir, f"{exp_name}.csv"),
            os.path.join(csv_dir, "imdb", f"{exp_name}.csv"),
            os.path.join(os.path.dirname(csv_dir), f"{exp_name}.csv"),
            os.path.join(csv_dir, f"imdb_{exp_name.replace('llama32_3b_imdb_', '')}.csv"),
        ]
        
        # Additional patterns for k_o experiments
        if "k_o" in exp_name.lower():
            # Try k_o_lowest6.csv, k_o_lowest9.csv format
            k_o_name = exp_name.replace("llama32_3b_imdb_", "")
            csv_paths.extend([
                os.path.join(csv_dir, f"{k_o_name}.csv"),
                os.path.join(csv_dir, "imdb", f"{k_o_name}.csv"),
            ])
        
        # Additional patterns for full baseline
        if "full" in exp_name.lower():
            csv_paths.extend([
                os.path.join(csv_dir, "imdb_llama32_3b_full.csv"),  # Actual file name
                os.path.join(csv_dir, "llama32_3b_imdb_full_finetuning.csv"),
                os.path.join(csv_dir, "full.csv"),
                os.path.join(csv_dir, "imdb", "full.csv"),
                os.path.join(csv_dir, "imdb", "imdb_llama32_3b_full.csv"),
            ])
        
        csv_path = None
        for path in csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if not csv_path:
            print(f"[skip] {exp_name} not found in any of: {csv_paths}")
            continue
        
        # Load accuracy points
        acc_points = load_imdb_points(csv_path)
        if not acc_points:
            print(f"[skip] No accuracy data in {csv_path}")
            continue
        
        acc_epochs = [e for e, _ in acc_points]
        acc_values = [v for _, v in acc_points]
        
        all_epochs.update(acc_epochs)
        all_acc_values.extend(acc_values)
        
        marker_shape = markers[idx]
        dash_style = dash_styles[idx]
        line_width = line_widths[idx]
        
        # Add accuracy trace
        fig.add_trace(
            go.Scatter(
                x=acc_epochs,
                y=acc_values,
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
                             'Accuracy: %{y:.2f}%<br>' +
                             '<extra></extra>',
            ),
            row=1, col=1
        )
        
        plotted_count += 1
    
    if plotted_count == 0:
        print("❌ No experiments plotted! Check CSV file paths.")
        return
    
    print(f"✅ Plotted {plotted_count} experiments")
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="IMDB Sentiment Analysis: K+O Freezing Experiments",
            x=0.5,
            font=dict(size=20)
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11)
        ),
        hovermode='x unified',
        template='plotly_white',
    )
    
    # Dynamic Y-axis range calculation
    acc_range = None
    if ymin is not None and ymax is not None:
        acc_range = [ymin, ymax]
    elif all_acc_values:
        acc_min, acc_max = min(all_acc_values), max(all_acc_values)
        acc_padding = (acc_max - acc_min) * 0.1
        acc_range = [max(0, acc_min - acc_padding), acc_max + acc_padding]
    
    fig.update_yaxes(
        title_text="Accuracy (%)",
        range=acc_range,
        row=1, col=1
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
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"\n✅ Plot saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot IMDB lowest/highest evaluation results with Plotly")
    parser.add_argument(
        "--csv-dir",
        default="./evaluation_results/imdb",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output",
        default="./plots/imdb/imdb_llama32_3b_lowest_highest_plotly.html",
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
    
    plot_imdb_lowest_highest_plotly(args.csv_dir, args.output, args.ymin, args.ymax)


if __name__ == "__main__":
    main()
