#!/usr/bin/env python3
"""
Plot IMDB evaluation results: Full Finetuning vs Lowest 3 vs Lowest 15 (no_grad)
Creates multiple Plotly plot variations with different styles.
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
        # Convert to percentage if needed (some CSVs have 0.9455, others have 94.55)
        if acc < 1.0:
            acc = acc * 100
        if ep >= 0:
            points.append((ep, acc))
    
    points.sort(key=lambda x: x[0])
    return points


def create_plot_variant_1(csv_dir: str, output_path: str):
    """
    Variant 1: Enhanced classic line plot with:
    - Different line dashes + widths
    - Distinct marker symbols and larger marker size
    - X-offset (jitter) per series
    - Zoomed view subplot for overlapping region (93-96%)
    - Improved hover with unified mode
    - Final point annotations
    - Legend outside plotting area
    """
    experiments = [
        ("imdb_llama32_3b_full.csv", "Full Finetuning", "circle", "#1f77b4", "solid", 3.5, -0.06),
        ("lowest3_no_grad.csv", "Lowest 3 (total 9)", "square", "#ff7f0e", "dash", 3.0, 0.0),
        ("lowest15_no_grad.csv", "Lowest 15 (total 45)", "diamond", "#2ca02c", "dot", 3.5, 0.06),
    ]
    
    # Create subplots: main plot + zoomed view
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        shared_xaxes=True,
        subplot_titles=('Full View', 'Zoomed View (93-96%)')
    )
    
    annotations_list = []
    
    for csv_file, label, marker_shape, color, dash_style, line_width, x_offset in experiments:
        csv_path = os.path.join(csv_dir, csv_file)
        points = load_imdb_points(csv_path)
        
        if not points:
            print(f"[skip] No data in {csv_file}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        
        # Apply x-offset (jitter) - but keep original epochs for display
        epochs_jittered = [e + x_offset for e in epochs]
        epochs_original = epochs  # Keep original for hover display
        
        # Determine dash pattern
        dash_map = {
            "solid": None,
            "dash": "dash",
            "dot": "dot",
            "dashdot": "dashdot"
        }
        
        # Add to main plot (row 1)
        fig.add_trace(
            go.Scatter(
                x=epochs_jittered,
                y=accs,
                mode='lines+markers',
                name=label,
                marker=dict(
                    symbol=marker_shape,
                    size=14,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    width=line_width,
                    color=color,
                    dash=dash_map[dash_style]
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Epoch: %{customdata}<br>' +
                             'Accuracy: %{y:.2f}%<extra></extra>',
                customdata=[int(e) for e in epochs_original],
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add to zoomed view (row 2)
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
                    line=dict(width=1.5, color='white')
                ),
                line=dict(
                    width=line_width,
                    color=color,
                    dash=dash_map[dash_style]
                ),
                hovertemplate=f'<b>{label}</b><br>' +
                             'Epoch: %{customdata}<br>' +
                             'Accuracy: %{y:.2f}%<extra></extra>',
                customdata=[int(e) for e in epochs_original],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add annotation for final point (on main plot)
        if epochs and accs:
            final_epoch_jittered = epochs_jittered[-1]
            final_epoch_original = epochs[-1]
            final_acc = accs[-1]
            # Position annotation to avoid overlap
            ax_offset = 30 if x_offset <= 0 else -30
            ay_offset = -40 if final_acc > 94 else 40
            
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
                font=dict(size=11, color=color, family='Arial Black'),
                xref='x',
                yref='y',
                xanchor='left' if x_offset <= 0 else 'right'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="IMDB Evaluation: Full Finetuning vs Q/K/V Freezing (no_grad)",
            font=dict(size=20, family="Arial Black"),
            x=0.5,
            y=0.98
        ),
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1.5
        ),
        annotations=annotations_list
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Epoch",
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        dtick=1,  # Show every epoch
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Epoch",
        title_font=dict(size=16),
        tickfont=dict(size=13),
        showgrid=True,
        gridcolor='rgba(200,200,200,0.5)',
        gridwidth=1,
        dtick=1,  # Show every epoch
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
        range=[93, 96],  # Zoomed view
        row=2, col=1
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"[ok] Saved Enhanced Variant 1: {output_path}")
    return output_path


def create_plot_variant_2(csv_dir: str, output_path: str):
    """
    Variant 2: Gradient-filled area under curves with transparency
    """
    experiments = [
        ("imdb_llama32_3b_full.csv", "Full Finetuning", "#1f77b4"),
        ("lowest3_no_grad.csv", "Lowest 3 (total 9)", "#ff7f0e"),
        ("lowest15_no_grad.csv", "Lowest 15 (total 45)", "#2ca02c"),
    ]
    
    fig = go.Figure()
    
    for csv_file, label, color in experiments:
        csv_path = os.path.join(csv_dir, csv_file)
        points = load_imdb_points(csv_path)
        
        if not points:
            print(f"[skip] No data in {csv_file}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        
        # Convert hex to rgba for fill
        hex_color = color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgba_fill = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.2)'
        
        # Add filled area
        fig.add_trace(go.Scatter(
            x=epochs,
            y=accs,
            mode='lines+markers',
            name=label,
            fill='tozeroy',
            fillcolor=rgba_fill,
            marker=dict(size=10, color=color),
            line=dict(width=3, color=color),
            hovertemplate=f'<b>{label}</b><br>' +
                         'Epoch: %{x}<br>' +
                         'Accuracy: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="IMDB Evaluation: Full Finetuning vs Q/K/V Freezing (no_grad)",
            font=dict(size=18, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title="Epoch",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Accuracy (%)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='lightgray',
            autorange=True
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"[ok] Saved Variant 2: {output_path}")
    return output_path


def create_plot_variant_3(csv_dir: str, output_path: str):
    """
    Variant 3: Bar chart style with connecting lines, showing max accuracy annotations
    """
    experiments = [
        ("imdb_llama32_3b_full.csv", "Full Finetuning", "circle", "#1f77b4"),
        ("lowest3_no_grad.csv", "Lowest 3 (total 9)", "square", "#ff7f0e"),
        ("lowest15_no_grad.csv", "Lowest 15 (total 45)", "diamond", "#2ca02c"),
    ]
    
    fig = go.Figure()
    
    for csv_file, label, marker_shape, color in experiments:
        csv_path = os.path.join(csv_dir, csv_file)
        points = load_imdb_points(csv_path)
        
        if not points:
            print(f"[skip] No data in {csv_file}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        
        # Find max accuracy
        max_idx = max(range(len(accs)), key=lambda i: accs[i])
        max_epoch = epochs[max_idx]
        max_acc = accs[max_idx]
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=accs,
            mode='lines+markers',
            name=label,
            marker=dict(
                symbol=marker_shape,
                size=12,
                color=color,
                line=dict(width=2, color='white'),
                # Highlight max point
                colorbar=None
            ),
            line=dict(width=3, color=color, shape='spline'),
            hovertemplate=f'<b>{label}</b><br>' +
                         'Epoch: %{x}<br>' +
                         'Accuracy: %{y:.2f}%<extra></extra>',
            # Add annotation for max
            text=[f'{max_acc:.2f}%' if i == max_idx else '' for i in range(len(epochs))],
            textposition='top center',
            textfont=dict(size=10, color=color, family='Arial Black')
        ))
    
    fig.update_layout(
        title=dict(
            text="IMDB Evaluation: Full Finetuning vs Q/K/V Freezing (no_grad)",
            font=dict(size=18, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title="Epoch",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            gridwidth=1
        ),
        yaxis=dict(
            title="Accuracy (%)",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            gridwidth=1,
            autorange=True
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1.5,
            font=dict(size=12)
        ),
        hovermode='closest',
        template='plotly_white',
        plot_bgcolor='rgba(250,250,250,1)',
        width=1000,
        height=600
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"[ok] Saved Variant 3: {output_path}")
    return output_path


def create_plot_variant_4(csv_dir: str, output_path: str):
    """
    Variant 4: Subplot with individual panels + combined view, with error bands
    """
    experiments = [
        ("imdb_llama32_3b_full.csv", "Full Finetuning", "circle", "#1f77b4"),
        ("lowest3_no_grad.csv", "Lowest 3 (total 9)", "square", "#ff7f0e"),
        ("lowest15_no_grad.csv", "Lowest 15 (total 45)", "diamond", "#2ca02c"),
    ]
    
    # Create subplots: 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Individual Experiments', 'Combined View'),
        vertical_spacing=0.15,
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    for csv_file, label, marker_shape, color in experiments:
        csv_path = os.path.join(csv_dir, csv_file)
        points = load_imdb_points(csv_path)
        
        if not points:
            print(f"[skip] No data in {csv_file}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        
        # Add to both subplots
        for row in [1, 2]:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=accs,
                    mode='lines+markers',
                    name=label,
                    marker=dict(
                        symbol=marker_shape,
                        size=10 if row == 1 else 8,
                        color=color,
                        line=dict(width=1.5, color='white')
                    ),
                    line=dict(width=3 if row == 1 else 2.5, color=color),
                    hovertemplate=f'<b>{label}</b><br>' +
                                 'Epoch: %{x}<br>' +
                                 'Accuracy: %{y:.2f}%<extra></extra>',
                    showlegend=(row == 1)  # Only show legend in first subplot
                ),
                row=row, col=1
            )
    
    fig.update_layout(
        title=dict(
            text="IMDB Evaluation: Full Finetuning vs Q/K/V Freezing (no_grad)",
            font=dict(size=18, family="Arial Black"),
            x=0.5,
            y=0.98
        ),
        height=800,
        width=1000,
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Epoch", row=2, col=1, title_font=dict(size=14))
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1, title_font=dict(size=14), autorange=True)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1, title_font=dict(size=14), autorange=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"[ok] Saved Variant 4: {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot IMDB Full vs Lowest 3 vs Lowest 15 (no_grad) - Multiple Plotly Variants")
    parser.add_argument("--csv-dir", default="./evaluation_results/imdb",
                        help="Directory containing CSV files")
    parser.add_argument("--output-dir", default="./plots/imdb/eval",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create all variants
    variants = [
        (create_plot_variant_1, "imdb_full_vs_no_grad_variant1.html"),
        (create_plot_variant_2, "imdb_full_vs_no_grad_variant2.html"),
        (create_plot_variant_3, "imdb_full_vs_no_grad_variant3.html"),
        (create_plot_variant_4, "imdb_full_vs_no_grad_variant4.html"),
    ]
    
    print("=" * 80)
    print("Creating 4 Plotly plot variants...")
    print("=" * 80)
    
    for variant_func, filename in variants:
        output_path = os.path.join(args.output_dir, filename)
        variant_func(args.csv_dir, output_path)
    
    print("\n" + "=" * 80)
    print("✅ All 4 plot variants generated!")
    print("=" * 80)
    print("\nVariants:")
    print("  1. Classic line plot with markers (different shapes)")
    print("  2. Gradient-filled area under curves")
    print("  3. Spline lines with max accuracy annotations")
    print("  4. Subplot with individual + combined views")
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
