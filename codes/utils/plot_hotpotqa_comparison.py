#!/usr/bin/env python3
"""
Plot HotpotQA evaluation results comparing multiple models.
Shows EM (Exact Match) and F1 scores over epochs.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

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


def plot_hotpotqa_comparison(csv_files: dict, output_path: str, metric: str = "f1", ymin: float = 0.0, ymax: float = 100.0):
    """
    Plot HotpotQA evaluation comparison.
    
    Args:
        csv_files: Dict mapping label -> csv_path
        output_path: Output PNG path
        metric: 'em' or 'f1'
        ymin, ymax: Y-axis limits
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    all_epochs = set()
    plotted_count = 0
    
    for idx, (label, csv_path) in enumerate(csv_files.items()):
        pts = load_hotpotqa_points(csv_path, metric=metric)
        if not pts:
            continue
        
        epochs = [e for e, _ in pts]
        values = [v for _, v in pts]
        all_epochs.update(epochs)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(
            epochs, values,
            marker=marker,
            linewidth=2.5,
            color=color,
            label=label,
            markersize=10,
            markeredgewidth=2.0,
            markeredgecolor='white',
            markerfacecolor=color,
            alpha=0.9
        )
        
        # Annotate max value
        if values:
            max_idx = max(range(len(values)), key=lambda i: values[i])
            max_e, max_v = epochs[max_idx], values[max_idx]
            ax.scatter([max_e], [max_v], s=100, color=color, edgecolors="black", zorder=3, linewidth=2)
            ax.annotate(
                f"{max_v:.2f}",
                (max_e, max_v),
                textcoords="offset points",
                xytext=(0, -20),
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9, linewidth=1.5),
            )
        
        plotted_count += 1
        print(f"[ok] Plotted {label}: {len(pts)} points, max {metric.upper()}={max(values):.2f}%")
    
    if plotted_count == 0:
        print("[error] No data to plot")
        return None
    
    ax.set_xticks(sorted(list(all_epochs)))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{metric.upper()} Score (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"HotpotQA Evaluation: {metric.upper()} Score Comparison", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[ok] Saved plot: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot HotpotQA evaluation comparison")
    parser.add_argument(
        "--csv-dir",
        default="./evaluation_results/hotpotqa",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output-dir",
        default="./plots/hotpotqa/eval",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Y-axis minimum"
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=100.0,
        help="Y-axis maximum"
    )
    
    args = parser.parse_args()
    
    # Define CSV files to plot
    csv_files = {}
    
    # Check in the specified csv_dir
    csv_dir_path = Path(args.csv_dir)
    llama32_lora = csv_dir_path / "llama32_3b_hotpotqa_lora.csv"
    qwen_lora = csv_dir_path / "qwen3_8b_base_hotpotqa_lora.csv"
    llama32_full = csv_dir_path / "llama32_3b_hotpotqa_full.csv"
    
    # Also check parent directory for full finetuning
    parent_dir = csv_dir_path.parent
    llama32_full_alt = parent_dir / "llama32_3b_hotpotqa_full.csv"
    
    if llama32_lora.exists():
        csv_files["Llama-3.2-3B LoRA"] = str(llama32_lora)
    if qwen_lora.exists():
        csv_files["Qwen-8B-Base LoRA"] = str(qwen_lora)
    if llama32_full.exists():
        csv_files["Llama-3.2-3B Full"] = str(llama32_full)
    elif llama32_full_alt.exists():
        csv_files["Llama-3.2-3B Full"] = str(llama32_full_alt)
    
    # Plot F1 score
    f1_output = os.path.join(args.output_dir, "hotpotqa_f1_comparison.png")
    plot_hotpotqa_comparison(csv_files, f1_output, metric="f1", ymin=args.ymin, ymax=args.ymax)
    
    # Plot EM score
    em_output = os.path.join(args.output_dir, "hotpotqa_em_comparison.png")
    plot_hotpotqa_comparison(csv_files, em_output, metric="em", ymin=args.ymin, ymax=args.ymax)
    
    print("\n✅ All plots generated!")


if __name__ == "__main__":
    main()
