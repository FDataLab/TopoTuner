#!/usr/bin/env python3
"""
Plot HotpotQA Llama-3.1-8B O/Q/V Frozen Experiments - EM and F1 Scores
Compares Run 1 and Run 2 (if available) across all experiments
Uses the same format as plot_hotpotqa_k_o_plotly.py
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from codes.utils.eval_plots import infer_epoch

# Define experiments and their display names with colors
EXPERIMENTS = [
    ("k_mlp_full", "Full", "#9467bd"),
    ("k_mlp_lowest3", "L3", "#1f77b4"),
    ("k_mlp_highest3", "H3", "#ff7f0e"),
    ("k_mlp_lowest15", "L15", "#2ca02c"),
    ("k_mlp_highest15", "H15", "#d62728"),
]

# Markers for each experiment
MARKERS = ['star', 'circle', 'square', 'triangle-up', 'diamond']

# Line styles: lowest=solid, highest=dashed, full=dotted
DASH_STYLES = ["dot", None, "dash", None, "dash"]

# Line widths
LINE_WIDTHS = [3.5, 3.5, 3.0, 3.5, 3.0]

# Run suffixes for different runs
RUN_SUFFIXES = {
    "run1": "",
    "run2": "_run2",
    "run3": "_run3",
}

def load_hotpotqa_points(csv_path, metric="f1"):
    """
    Load HotpotQA evaluation points from CSV.
    
    Args:
        csv_path: Path to CSV file
        metric: 'em' or 'f1'
    
    Returns:
        List of (epoch, value) tuples
    """
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
    except Exception as e:
        print(f"[skip] Error reading {csv_path}: {e}")
        return []
    
    if "checkpoint" not in df.columns or metric not in df.columns:
        print(f"[skip] Missing columns in {csv_path}")
        return []
    
    points = []
    for _, row in df.iterrows():
        ep = infer_epoch(row["checkpoint"])
        value = float(row[metric]) * 100  # Convert to percentage
        if ep >= 0:
            points.append((ep, value))
    
    points.sort(key=lambda x: x[0])
    return points

def plot_run(run_id, csv_dir="./evaluation_results/hotpotqa"):
    """
    Plot F1 and EM for a specific run.
    
    Args:
        run_id: 'run1', 'run2', or 'run3'
        csv_dir: Directory containing CSV files
    """
    run_suffix = RUN_SUFFIXES.get(run_id, "")
    run_display = run_id.upper().replace("RUN", "Run ")
    
    # Create subplots: top for F1, bottom for EM
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,  # Reduced spacing
        shared_xaxes=True,
        subplot_titles=('F1 Score', 'Exact Match (EM) Score'),
        x_title="Epoch"
    )
    
    all_epochs = set()
    all_f1_values = []
    all_em_values = []
    plotted_count = 0
    
    # Plot both F1 and EM for each experiment
    for idx, (exp_name, label, color) in enumerate(EXPERIMENTS):
        csv_path = os.path.join(csv_dir, f"hotpotqa_llama31_8b_oqv_frozen_{exp_name}{run_suffix}.csv")
        
        if not os.path.exists(csv_path):
            continue
        
        # Load F1 points
        f1_points = load_hotpotqa_points(csv_path, metric="f1")
        if not f1_points:
            continue
        
        # Load EM points
        em_points = load_hotpotqa_points(csv_path, metric="em")
        if not em_points:
            continue
        
        f1_epochs = [e for e, _ in f1_points]
        f1_values = [v for _, v in f1_points]
        em_epochs = [e for e, _ in em_points]
        em_values = [v for _, v in em_points]
        
        all_epochs.update(f1_epochs)
        all_epochs.update(em_epochs)
        all_f1_values.extend(f1_values)
        all_em_values.extend(em_values)
        
        marker_shape = MARKERS[idx]
        dash_style = DASH_STYLES[idx]
        line_width = LINE_WIDTHS[idx]
        
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
                font=dict(size=16, color=color, family="Arial Black"),
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
                font=dict(size=16, color=color, family="Arial Black"),
                row=2, col=1
            )
        
        plotted_count += 1
        print(f"[ok] Plotted {label}: F1 max={max(f1_values):.2f}%, EM max={max(em_values):.2f}%")
    
    if plotted_count == 0:
        print(f"⚠️  No data to plot for {run_display}!")
        return None
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"HotpotQA Llama-3.1-8B: O/Q/V Frozen - K+MLP Training ({run_display})",
            x=0.5,
            font=dict(size=24, family="Arial Black")
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
            font=dict(size=20)  # LARGER LEGEND TEXT
        ),
        template="plotly_white",
        font=dict(family="Arial", size=16)  # Larger base font
    )
    
    # Update subplot titles positioning and font
    for annotation in fig['layout']['annotations']:
        if annotation['text'] in ['F1 Score', 'Exact Match (EM) Score']:
            annotation['y'] = annotation['y'] + 0.03  # Move up
            annotation['font'] = dict(size=18, family="Arial Black")
    
    # Update y-axes with automatic scaling
    if all_f1_values and all_em_values:
        f1_min, f1_max = min(all_f1_values), max(all_f1_values)
        em_min, em_max = min(all_em_values), max(all_em_values)
        f1_padding = (f1_max - f1_min) * 0.1
        em_padding = (em_max - em_min) * 0.1
        f1_range = [max(0, f1_min - f1_padding), f1_max + f1_padding]
        em_range = [max(0, em_min - em_padding), em_max + em_padding]
        
        fig.update_yaxes(
            title_text="F1 Score (%)",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            range=f1_range,
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="EM Score (%)",
            title_font=dict(size=18),
            tickfont=dict(size=16),
            range=em_range,
            row=2, col=1
        )
    
    # Update x-axes
    if all_epochs:
        min_epoch = min(all_epochs)
        max_epoch = max(all_epochs)
        fig.update_xaxes(
            title_font=dict(size=18),
            tickfont=dict(size=16),
            range=[min_epoch - 0.5, max_epoch + 0.5],
            dtick=1,
            row=1, col=1
        )
        fig.update_xaxes(
            title_font=dict(size=18),
            tickfont=dict(size=16),
            range=[min_epoch - 0.5, max_epoch + 0.5],
            dtick=1,
            row=2, col=1
        )
    
    return fig

def create_all_plots(csv_dir="./evaluation_results/hotpotqa", output_dir="./plots"):
    """Create plots for all available runs"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for run_id in ["run1", "run2", "run3"]:
        print(f"\n{'='*80}")
        print(f"Creating plot for {run_id.upper().replace('RUN', 'Run ')}...")
        print('='*80)
        
        fig = plot_run(run_id, csv_dir)
        
        if fig is not None:
            output_file = os.path.join(output_dir, f"hotpotqa_llama31_8b_oqv_frozen_{run_id}.html")
            fig.write_html(output_file)
            print(f"\n✅ Plot saved to: {output_file}")
            
            # Try to save PNG
            try:
                png_file = os.path.join(output_dir, f"hotpotqa_llama31_8b_oqv_frozen_{run_id}.png")
                fig.write_image(png_file, width=1400, height=900, scale=2)
                print(f"✅ PNG saved to: {png_file}")
            except Exception as e:
                print(f"⚠️  Could not save PNG: {e}")

if __name__ == "__main__":
    print("="*80)
    print("Plotting HotpotQA Llama-3.1-8B O/Q/V Frozen Experiments")
    print("="*80)
    
    create_all_plots()
    
    print()
    print("="*80)
    print("✅ All plots created successfully!")
    print("="*80)
