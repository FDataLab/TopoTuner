#!/usr/bin/env python3
"""Generate overfitting analysis plots from existing overfitting_report.json files."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_reports(base_dir):
    """Load all overfitting reports from experiment dirs."""
    dirs = [
        "overfitting-wass-high6-run1",
        "overfitting-norm-high6-run1",
        "overfitting-norm-high9-run1",
    ]
    data = []
    for d in dirs:
        path = os.path.join(base_dir, d, "overfitting_report.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            r["label"] = d.replace("overfitting-", "").replace("-run1", "")
            data.append(r)
    return data


def plot_loss_vs_epoch(data, out_dir):
    """Combined train/val loss vs epoch for all experiments."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, r in enumerate(data):
        epochs = list(range(1, len(r["per_epoch"]["train_loss"]) + 1))
        ax.plot(epochs, r["per_epoch"]["train_loss"], "-o", color=colors[i],
                label=f"{r['label']} (train)", markersize=5)
        ax.plot(epochs, r["per_epoch"]["val_loss"], "--s", color=colors[i], alpha=0.8,
                label=f"{r['label']} (val)", markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss — All Experiments")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_vs_epoch.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_final_accuracy(data, out_dir):
    """Final Train/Val/Test accuracy comparison across experiments."""
    labels = [r["label"] for r in data]
    x = np.arange(len(labels))
    w = 0.25
    train = [r["final"]["train_accuracy"] * 100 for r in data]
    val = [r["final"]["val_accuracy"] * 100 for r in data]
    test = [r["final"]["test_accuracy"] * 100 for r in data]
    plt.bar(x - w, train, w, label="Train")
    plt.bar(x, val, w, label="Val")
    plt.bar(x + w, test, w, label="Test")
    plt.ylabel("Accuracy (%)")
    plt.title("Final Accuracy (Train / Val / Test) — All Experiments")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "final_accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_generalization_gap(data, out_dir):
    """Train–Val gap across experiments."""
    labels = [r["label"] for r in data]
    gaps = [r["overfitting_gap"]["train_minus_val"] * 100 for r in data]
    plt.bar(labels, gaps, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Train − Val Accuracy (%)")
    plt.title("Generalization Gap (Train − Val) — All Experiments")
    plt.tight_layout()
    path = os.path.join(out_dir, "generalization_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_val_vs_test(data, out_dir):
    """Val vs Test accuracy scatter."""
    val = [r["final"]["val_accuracy"] * 100 for r in data]
    test = [r["final"]["test_accuracy"] * 100 for r in data]
    labels = [r["label"] for r in data]
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (v, t) in enumerate(zip(val, test)):
        ax.scatter(v, t, s=100, label=labels[i])
        ax.annotate(labels[i], (v, t), xytext=(5, 5), textcoords="offset points", fontsize=9)
    lims = [min(val + test) - 2, max(val + test) + 2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Validation Accuracy (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Validation vs Test Accuracy")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "val_vs_test_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_test_vs_trainable(data, out_dir):
    """Test accuracy vs trainable params."""
    params = [r["hyperparameters"]["trainable_params"] / 1e6 for r in data]
    test = [r["final"]["test_accuracy"] * 100 for r in data]
    labels = [r["label"] for r in data]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (p, t) in enumerate(zip(params, test)):
        ax.scatter(p, t, s=120, label=labels[i])
        ax.annotate(labels[i], (p, t), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Trainable params (M)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Trainable Parameters")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "test_acc_vs_trainable_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_gap_vs_trainable(data, out_dir):
    """Generalization gap vs trainable params."""
    params = [r["hyperparameters"]["trainable_params"] / 1e6 for r in data]
    gaps = [r["overfitting_gap"]["train_minus_val"] * 100 for r in data]
    labels = [r["label"] for r in data]
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (p, g) in enumerate(zip(params, gaps)):
        ax.scatter(p, g, s=120, label=labels[i])
        ax.annotate(labels[i], (p, g), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Trainable params (M)")
    ax.set_ylabel("Train − Val Gap (%)")
    ax.set_title("Generalization Gap vs Trainable Parameters")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "gap_vs_trainable_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot overfitting analysis from report JSONs")
    parser.add_argument("--base-dir", default="/home/kadir/topo/numpy_weights/exploration-finetuning",
                        help="Directory containing overfitting-*-run1/")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: base-dir/overfitting_analysis_plots)")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.base_dir, "overfitting_analysis_plots")
    os.makedirs(out_dir, exist_ok=True)

    data = load_reports(args.base_dir)
    if not data:
        print("No reports found.")
        return
    print(f"Loaded {len(data)} reports")

    plot_loss_vs_epoch(data, out_dir)
    plot_final_accuracy(data, out_dir)
    plot_generalization_gap(data, out_dir)
    plot_val_vs_test(data, out_dir)
    plot_test_vs_trainable(data, out_dir)
    plot_gap_vs_trainable(data, out_dir)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
