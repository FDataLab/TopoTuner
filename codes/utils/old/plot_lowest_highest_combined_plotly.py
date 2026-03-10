#!/usr/bin/env python3
"""
Plot combined evaluation results for lowest and highest layer freezing experiments using Plotly.
Enhanced version with improved styling, hover, and annotations.

Plots all lowest_3/6/9/12/15 and highest_3/6/9/12/15 experiments:
- Top subplot: Lowest layers (dashed lines)
- Bottom subplot: Highest layers (solid lines)
- Different colors for corresponding numbers (3, 6, 9, 12, 15)
- Enhanced hover, annotations, and styling
"""

import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from codes.utils.eval_plots import load_points, infer_epoch


def load_imdb_points(csv_path: str):
    """Load IMDB evaluation points from CSV."""
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
    except Exception as e:
        print(f"[skip] Error reading {csv_path}: {e}")
        return []
    
    if "checkpoint" not in df.columns or "acc" not in df.columns:
        return []
    
    points = []
    for _, row in df.iterrows():
        ep = infer_epoch(row["checkpoint"])
        acc = float(row["acc"])
        # Convert to percentage if needed
        if acc < 1.0:
            acc = acc * 100
        if ep >= 0:
            points.append((ep, acc))
    
    points.sort(key=lambda x: x[0])
    return points


def plot_lowest_highest_plotly(csv_dir: str, output_path: str):
    """
    Plot lowest and highest experiments with enhanced Plotly styling.
    
    Args:
        csv_dir: Directory containing CSV files (e.g., evaluation_results/imdb/)
        output_path: Output HTML path
    """
    lowest_experiments = [
        ("lowest_3", "Lowest 3", -0.12),
        ("lowest_6", "Lowest 6", -0.06),
        ("lowest_9", "Lowest 9", 0.0),
        ("lowest_12", "Lowest 12", 0.06),
        ("lowest_15", "Lowest 15", 0.12),
    ]
    
    highest_experiments = [
        ("highest_3", "Highest 3", -0.12),
        ("highest_6", "Highest 6", -0.06),
        ("highest_9", "Highest 9", 0.0),
        ("highest_12", "Highest 12", 0.06),
        ("highest_15", "Highest 15", 0.12),
    ]
    
    # Color palette - same colors for corresponding numbers (3, 6, 9, 12, 15)
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
    ]
    
    # Different markers for each line
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down']
    
    # Line styles: different dashes for variety
    dash_styles = [None, "dash", "dot", "dashdot", "longdash"]
    
    # Line widths
    line_widths = [3.5, 3.0, 3.5, 3.0, 3.5]
    
    # Create subplots: top for lowest, bottom for highest
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=('Lowest Layers (Dashed Lines)', 'Highest Layers (Solid Lines)')
    )
    
    all_epochs = set()
    annotations_list = []
    plotted_lowest = 0
    plotted_highest = 0
    
    # Plot lowest experiments (top subplot, row 1)
    for idx, (exp_name, label, x_offset) in enumerate(lowest_experiments):
        csv_path = os.path.join(csv_dir, f"llama32_3b_imdb_{exp_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        
        points = load_imdb_points(csv_path)
        if not points:
            print(f"[skip] No valid data in {csv_path}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        all_epochs.update(epochs)
        
        # Apply x-offset (jitter)
        epochs_jittered = [e + x_offset for e in epochs]
        
        color = colors[idx]
        marker_shape = markers[idx]
        dash_style = dash_styles[idx]
        line_width = line_widths[idx]
        
        # Add to top subplot (lowest)
        fig.add_trace(
            go.Scatter(
                x=epochs_jittered,
                y=accs,
                mode='lines+markers',
                name=label,
                marker=dict(
                    symbol=marker_shape,
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    width=line_width,
                    color=color,
                    dash=dash_style
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Epoch: %{customdata}<br>' +
                             'Accuracy: %{y:.2f}%<extra></extra>',
                customdata=[int(e) for e in epochs],
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add annotation for final point
        if epochs and accs:
            final_epoch_jittered = epochs_jittered[-1]
            final_acc = accs[-1]
            ax_offset = 25 if x_offset <= 0 else -25
            ay_offset = -35
            
            annotations_list.append(dict(
                x=final_epoch_jittered,
                y=final_acc,
                text=f"{label}<br>{final_acc:.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                ax=ax_offset,
                ay=ay_offset,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=color,
                borderwidth=2,
                font=dict(size=10, color=color, family='Arial Black'),
                xref='x',
                yref='y',
                xanchor='left' if x_offset <= 0 else 'right'
            ))
        
        plotted_lowest += 1
        print(f"[ok] Plotted {label}: {len(points)} points")
    
    # Plot highest experiments (bottom subplot, row 2)
    for idx, (exp_name, label, x_offset) in enumerate(highest_experiments):
        csv_path = os.path.join(csv_dir, f"llama32_3b_imdb_{exp_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        
        points = load_imdb_points(csv_path)
        if not points:
            print(f"[skip] No valid data in {csv_path}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        all_epochs.update(epochs)
        
        # Apply x-offset (jitter)
        epochs_jittered = [e + x_offset for e in epochs]
        
        color = colors[idx]  # Same color as corresponding lowest
        marker_shape = markers[idx]
        line_width = line_widths[idx]
        
        # Add to bottom subplot (highest) - solid lines
        fig.add_trace(
            go.Scatter(
                x=epochs_jittered,
                y=accs,
                mode='lines+markers',
                name=label,
                marker=dict(
                    symbol=marker_shape,
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    width=line_width,
                    color=color,
                    dash=None  # Solid lines for highest
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Epoch: %{customdata}<br>' +
                             'Accuracy: %{y:.2f}%<extra></extra>',
                customdata=[int(e) for e in epochs],
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Add annotation for final point
        if epochs and accs:
            final_epoch_jittered = epochs_jittered[-1]
            final_acc = accs[-1]
            ax_offset = 25 if x_offset <= 0 else -25
            ay_offset = 35  # Upward for bottom subplot
            
            annotations_list.append(dict(
                x=final_epoch_jittered,
                y=final_acc,
                text=f"{label}<br>{final_acc:.2f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                ax=ax_offset,
                ay=ay_offset,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor=color,
                borderwidth=2,
                font=dict(size=10, color=color, family='Arial Black'),
                xref='x2',
                yref='y2',
                xanchor='left' if x_offset <= 0 else 'right'
            ))
        
        plotted_highest += 1
        print(f"[ok] Plotted {label}: {len(points)} points")
    
    if plotted_lowest == 0 and plotted_highest == 0:
        print("[error] No data to plot")
        return None
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Llama-3.2-3B IMDB: Lowest vs Highest Layer Freezing",
            font=dict(size=20, family="Arial Black"),
            x=0.5,
            y=0.98
        ),
        hovermode='x unified',
        template='plotly_white',
        width=1400,
        height=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=13),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1.5
        ),
        annotations=annotations_list
    )
    
    # Update x-axes
    sorted_epochs = sorted(list(all_epochs))
    fig.update_xaxes(
        title_text="Epoch",
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        dtick=1,  # Show every epoch
        tickvals=sorted_epochs,
        row=2, col=1
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Accuracy (%)",
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        autorange=True,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Accuracy (%)",
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        autorange=True,
        row=2, col=1
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"[ok] Saved Plotly plot: {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot combined lowest/highest evaluation results with Plotly")
    parser.add_argument("--csv-dir", default="./evaluation_results/imdb",
                        help="Directory containing CSV files")
    parser.add_argument("--output", default="./plots/imdb/eval/llama32_3b_lowest_highest_combined_plotly.html",
                        help="Output HTML path")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Creating Plotly plot: Lowest vs Highest Layer Freezing")
    print("=" * 80)
    
    plot_lowest_highest_plotly(args.csv_dir, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Plot generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
