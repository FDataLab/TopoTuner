#!/usr/bin/env python3
"""
Plot IMDB evaluation results for K/Q/V freezing experiments.
Creates separate plots for K, Q, and V experiments.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

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
        if ep >= 0:
            points.append((ep, acc))
    
    points.sort(key=lambda x: x[0])
    return points


def plot_kqv_experiments(csv_dir: str, output_dir: str, projection_type: str, experiments: list, ymin: float = 0.0, ymax: float = 100.0):
    """
    Plot experiments for a specific projection type (K, Q, or V).
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Output directory for plots
        projection_type: 'k', 'q', or 'v'
        experiments: List of (csv_filename, label) tuples
        ymin, ymax: Y-axis limits
    """
    plt.figure(figsize=(10, 6))
    
    # Color palette
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]
    
    # Markers
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Line styles: dashed for O-only, solid for MLP+O
    linestyles = ['--', '--', '--', '-', '-', '-']
    
    all_epochs = set()
    plotted_count = 0
    
    for idx, (csv_file, label) in enumerate(experiments):
        csv_path = os.path.join(csv_dir, csv_file)
        points = load_imdb_points(csv_path)
        
        if not points:
            print(f"[skip] No data in {csv_file}")
            continue
        
        epochs = [e for e, _ in points]
        accs = [a for _, a in points]
        
        all_epochs.update(epochs)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        
        plt.plot(epochs, accs, marker=marker, linewidth=2, linestyle=linestyle,
                color=color, label=label, markersize=8)
        
        # Annotate max value
        if accs:
            max_idx = max(range(len(accs)), key=lambda i: accs[i])
            max_e, max_a = epochs[max_idx], accs[max_idx]
            plt.scatter([max_e], [max_a], s=100, color=color, edgecolors="black", 
                       zorder=3, linewidth=1.5)
            plt.annotate(
                f"{max_a:.2f}%", (max_e, max_a), textcoords="offset points", 
                xytext=(0, -18), ha="center", fontsize=10, fontweight="bold", 
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9),
            )
        
        plotted_count += 1
    
    if plotted_count == 0:
        print(f"[skip] No data to plot for {projection_type.upper()}")
        plt.close()
        return
    
    # Set up plot
    sorted_epochs = sorted(all_epochs)
    plt.xticks(sorted_epochs)
    plt.ylim(ymin, ymax)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(f"IMDB: {projection_type.upper()} Projection Freezing Experiments", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"imdb_{projection_type}_freezing_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ok] Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot IMDB K/Q/V freezing evaluation results")
    parser.add_argument("--csv-dir", default="./evaluation_results/imdb",
                        help="Directory containing CSV files")
    parser.add_argument("--output-dir", default="./plots/imdb/eval",
                        help="Output directory for plots")
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=100.0)
    
    args = parser.parse_args()
    
    # Define experiments for each projection type
    # Format: (csv_filename, display_label)
    k_experiments = [
        ("k_mlp_o_lowest3.csv", "K+MLP+O (lowest 3)"),
        ("k_o_lowest6.csv", "K+O (lowest 6)"),
        ("k_o_lowest9.csv", "K+O (lowest 9)"),
        ("k_mlp_o_lowest9.csv", "K+MLP+O (lowest 9)"),
    ]
    
    q_experiments = [
        ("q_mlp_o_lowest3.csv", "Q+MLP+O (lowest 3)"),
        ("q_o_lowest6.csv", "Q+O (lowest 6)"),
        ("q_o_lowest9.csv", "Q+O (lowest 9)"),
        # q_mlp_o_lowest9.csv is missing, so we skip it
    ]
    
    v_experiments = [
        ("v_mlp_o_lowest3.csv", "V+MLP+O (lowest 3)"),
        ("v_o_lowest6.csv", "V+O (lowest 6)"),
        ("v_o_lowest9.csv", "V+O (lowest 9)"),
        # v_mlp_o_lowest9.csv is missing, so we skip it
    ]
    
    # Plot each projection type
    print("Plotting K experiments...")
    plot_kqv_experiments(args.csv_dir, args.output_dir, "k", k_experiments, 
                        args.ymin, args.ymax)
    
    print("\nPlotting Q experiments...")
    plot_kqv_experiments(args.csv_dir, args.output_dir, "q", q_experiments, 
                        args.ymin, args.ymax)
    
    print("\nPlotting V experiments...")
    plot_kqv_experiments(args.csv_dir, args.output_dir, "v", v_experiments, 
                        args.ymin, args.ymax)
    
    print("\n✅ All plots generated!")


if __name__ == "__main__":
    main()
