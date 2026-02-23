#!/usr/bin/env python3
"""
Plot combined evaluation results for lowest and highest layer freezing experiments.

Plots all lowest_3/6/9/12/15 and highest_3/6/9/12/15 experiments together:
- Different colors for each experiment
- Solid lines for highest experiments
- Dashed/dotted lines for lowest experiments
- Different markers for each line
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from codes.utils.eval_plots import load_points, infer_epoch, to_percent


def plot_combined_lowest_highest_overlay(csv_dir: str, output_path: str, ymin: float = 0.0, ymax: float = 100.0):
    """
    Plot all lowest and highest experiments together on a single plot (overlay version).
    
    Args:
        csv_dir: Directory containing CSV files (e.g., evaluation_results/imdb/)
        output_path: Output PNG path
        ymin, ymax: Y-axis limits
    """
    experiments = [
        ("lowest_3", "Lowest 3"),
        ("lowest_6", "Lowest 6"),
        ("lowest_9", "Lowest 9"),
        ("lowest_12", "Lowest 12"),
        ("lowest_15", "Lowest 15"),
        ("highest_3", "Highest 3"),
        ("highest_6", "Highest 6"),
        ("highest_9", "Highest 9"),
        ("highest_12", "Highest 12"),
        ("highest_15", "Highest 15"),
    ]
    
    # Color palette - different color for each experiment
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    
    # Different markers for each line
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd']
    
    # Line styles: dashed for lowest, solid for highest
    linestyles = ['--', '--', '--', '--', '--', '-', '-', '-', '-', '-']
    
    plt.figure(figsize=(12, 8))
    
    all_epochs = set()
    plotted_count = 0
    
    for idx, (exp_name, label) in enumerate(experiments):
        csv_path = os.path.join(csv_dir, f"llama32_3b_imdb_{exp_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        
        pts = load_points(csv_path)
        if not pts:
            print(f"[skip] No valid data in {csv_path}")
            continue
        
        epochs = [e for e, _ in pts]
        accs = [a for _, a in pts]
        all_epochs.update(epochs)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx]
        
        # Plot with different styles
        plt.plot(
            epochs, accs,
            marker=marker,
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
            label=label,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor='white' if linestyle == '-' else color,
            alpha=0.9 if linestyle == '-' else 0.8
        )
        
        plotted_count += 1
        print(f"[ok] Plotted {label}: {len(pts)} points")
    
    if plotted_count == 0:
        print("[error] No data to plot")
        return None
    
    plt.xticks(sorted(list(all_epochs)))
    plt.ylim(ymin, ymax)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Llama-3.2-3B IMDB: Lowest vs Highest Layer Freezing", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ok] Saved overlay plot: {output_path}")
    return output_path


def plot_combined_lowest_highest(csv_dir: str, output_path: str, ymin: float = 0.0, ymax: float = 100.0):
    """
    Plot all lowest and highest experiments together with better visibility.
    Uses subplots to separate lowest and highest for clarity.
    
    Args:
        csv_dir: Directory containing CSV files (e.g., evaluation_results/imdb/)
        output_path: Output PNG path
        ymin, ymax: Y-axis limits
    """
    lowest_experiments = [
        ("lowest_3", "Lowest 3"),
        ("lowest_6", "Lowest 6"),
        ("lowest_9", "Lowest 9"),
        ("lowest_12", "Lowest 12"),
        ("lowest_15", "Lowest 15"),
    ]
    
    highest_experiments = [
        ("highest_3", "Highest 3"),
        ("highest_6", "Highest 6"),
        ("highest_9", "Highest 9"),
        ("highest_12", "Highest 12"),
        ("highest_15", "Highest 15"),
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
    markers = ['o', 's', '^', 'D', 'v']
    
    # Create subplots: top for lowest, bottom for highest
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, sharey=True)
    
    all_epochs = set()
    plotted_lowest = 0
    plotted_highest = 0
    
    # Plot lowest experiments (top subplot)
    for idx, (exp_name, label) in enumerate(lowest_experiments):
        csv_path = os.path.join(csv_dir, f"llama32_3b_imdb_{exp_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        
        pts = load_points(csv_path)
        if not pts:
            print(f"[skip] No valid data in {csv_path}")
            continue
        
        epochs = [e for e, _ in pts]
        accs = [a for _, a in pts]
        all_epochs.update(epochs)
        
        color = colors[idx]
        marker = markers[idx]
        
        # Dashed line for lowest
        ax1.plot(
            epochs, accs,
            marker=marker,
            linewidth=2.5,
            linestyle='--',
            color=color,
            label=label,
            markersize=10,
            markeredgewidth=2.0,
            markeredgecolor='white',
            markerfacecolor=color,
            alpha=0.85
        )
        
        plotted_lowest += 1
        print(f"[ok] Plotted {label}: {len(pts)} points")
    
    # Plot highest experiments (bottom subplot)
    for idx, (exp_name, label) in enumerate(highest_experiments):
        csv_path = os.path.join(csv_dir, f"llama32_3b_imdb_{exp_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        
        pts = load_points(csv_path)
        if not pts:
            print(f"[skip] No valid data in {csv_path}")
            continue
        
        epochs = [e for e, _ in pts]
        accs = [a for _, a in pts]
        all_epochs.update(epochs)
        
        color = colors[idx]  # Same color as corresponding lowest
        marker = markers[idx]
        
        # Solid line for highest
        ax2.plot(
            epochs, accs,
            marker=marker,
            linewidth=2.5,
            linestyle='-',
            color=color,
            label=label,
            markersize=10,
            markeredgewidth=2.0,
            markeredgecolor='white',
            markerfacecolor=color,
            alpha=0.9
        )
        
        plotted_highest += 1
        print(f"[ok] Plotted {label}: {len(pts)} points")
    
    if plotted_lowest == 0 and plotted_highest == 0:
        print("[error] No data to plot")
        return None
    
    # Configure top subplot (lowest)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax1.set_title("Lowest Layers (Dashed Lines)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xticks(sorted(list(all_epochs)))
    ax1.legend(loc='best', fontsize=11, framealpha=0.95, ncol=5)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    # Configure bottom subplot (highest)
    ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Highest Layers (Solid Lines)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xticks(sorted(list(all_epochs)))
    ax2.legend(loc='best', fontsize=11, framealpha=0.95, ncol=5)
    ax2.grid(True, alpha=0.3, linestyle=':')
    
    # Overall title
    fig.suptitle("Llama-3.2-3B IMDB: Lowest vs Highest Layer Freezing", 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ok] Saved combined plot: {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot combined lowest/highest evaluation results")
    parser.add_argument("--csv-dir", default="./evaluation_results/imdb",
                        help="Directory containing CSV files")
    parser.add_argument("--output", default="./plots/imdb/eval/llama32_3b_lowest_highest_combined.png",
                        help="Output PNG path")
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=100.0)
    parser.add_argument("--overlay", action="store_true",
                        help="Create overlay plot (all lines on one plot) instead of subplots")
    
    args = parser.parse_args()
    
    if args.overlay:
        plot_combined_lowest_highest_overlay(args.csv_dir, args.output, ymin=args.ymin, ymax=args.ymax)
    else:
        plot_combined_lowest_highest(args.csv_dir, args.output, ymin=args.ymin, ymax=args.ymax)


if __name__ == "__main__":
    main()
