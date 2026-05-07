#!/usr/bin/env python3
"""Generate overfitting analysis plots from existing overfitting_report.json files."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEGACY_EXPERIMENT_DIRS = [
    "overfitting-wass-high6-run1",
    "overfitting-norm-high6-run1",
    "overfitting-norm-high9-run1",
]

DEFAULT_WASS_ELTWISE_CHECKPOINTS = "checkpoints/overfitting-wass-eltwise"


def load_reports_legacy(base_dir):
    """Load reports from flat overfitting-*-run1 directories under base_dir."""
    data = []
    for d in LEGACY_EXPERIMENT_DIRS:
        path = os.path.join(base_dir, d, "overfitting_report.json")
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            r["label"] = d.replace("overfitting-", "").replace("-run1", "")
            data.append(r)
    return data


def load_reports_tree(tree_root):
    """Load every overfitting_report.json under tree_root (e.g. wass-eltwise sweep)."""
    root = Path(tree_root).resolve()
    if not root.is_dir():
        return []
    paths = sorted(root.rglob("overfitting_report.json"))
    data = []
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        try:
            rel = path.parent.relative_to(root)
            r["label"] = str(rel).replace(os.sep, "/")
        except ValueError:
            r["label"] = path.parent.name
        data.append(r)
    return data


def load_reports(base_dir, reports_root=None):
    """
    Load overfitting_report.json entries.

    If reports_root is set, only scan that directory tree.

    Otherwise: prefer checkpoints/overfitting-wass-eltwise under base_dir when it
    contains any reports (new sweep layout); else fall back to legacy flat dirs.
    """
    if reports_root:
        return load_reports_tree(reports_root)

    tree_path = os.path.join(base_dir, DEFAULT_WASS_ELTWISE_CHECKPOINTS)
    tree = load_reports_tree(tree_path)
    if tree:
        return tree

    return load_reports_legacy(base_dir)


def _series_colors(n):
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def plot_loss_vs_epoch(data, out_dir):
    """Combined train/val loss vs epoch for all experiments."""
    w = 12 if len(data) > 8 else 8
    h = 6 if len(data) > 8 else 5
    fig, ax = plt.subplots(figsize=(w, h))
    colors = _series_colors(len(data))
    for i, r in enumerate(data):
        epochs = list(range(1, len(r["per_epoch"]["train_loss"]) + 1))
        c = colors[i]
        ax.plot(epochs, r["per_epoch"]["train_loss"], "-o", color=c,
                label=f"{r['label']} (train)", markersize=4)
        ax.plot(epochs, r["per_epoch"]["val_loss"], "--s", color=c, alpha=0.8,
                label=f"{r['label']} (val)", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss — All Experiments")
    ncol = min(6, max(2, len(data)))
    fs = 5 if len(data) > 10 else (7 if len(data) > 5 else 8)
    ax.legend(
        ncol=ncol,
        fontsize=fs,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
    )
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
    fig_w = max(10.0, 0.35 * len(labels))
    plt.figure(figsize=(fig_w, 5))
    plt.bar(x - w, train, w, label="Train")
    plt.bar(x, val, w, label="Val")
    plt.bar(x + w, test, w, label="Test")
    plt.ylabel("Accuracy (%)")
    plt.title("Final Accuracy (Train / Val / Test) — All Experiments")
    rot = 45 if len(labels) > 6 else 0
    ha = "right" if rot else "center"
    plt.xticks(x, labels, rotation=rot, ha=ha)
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
    fig_w = max(8.0, 0.35 * len(labels))
    plt.figure(figsize=(fig_w, 5))
    plt.bar(labels, gaps, color=_series_colors(len(gaps)))
    plt.ylabel("Train − Val Accuracy (%)")
    plt.title("Generalization Gap (Train − Val) — All Experiments")
    if len(labels) > 6:
        plt.xticks(rotation=45, ha="right")
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
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = _series_colors(len(data))
    afs = 6 if len(data) > 12 else (8 if len(data) > 6 else 9)
    for i, (v, t) in enumerate(zip(val, test)):
        sc_label = labels[i] if len(data) <= 8 else None
        ax.scatter(v, t, s=100, label=sc_label, color=colors[i])
        ax.annotate(
            labels[i], (v, t), xytext=(5, 5), textcoords="offset points", fontsize=afs
        )
    lims = [min(val + test) - 2, max(val + test) + 2]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")
    ax.set_xlabel("Validation Accuracy (%)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Validation vs Test Accuracy")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)
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
    colors = _series_colors(len(data))
    afs = 6 if len(data) > 12 else 9
    for i, (p, t) in enumerate(zip(params, test)):
        sc_label = labels[i] if len(data) <= 8 else None
        ax.scatter(p, t, s=120, label=sc_label, color=colors[i])
        ax.annotate(labels[i], (p, t), xytext=(5, 5), textcoords="offset points", fontsize=afs)
    ax.set_xlabel("Trainable params (M)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Trainable Parameters")
    if len(data) <= 8:
        ax.legend(fontsize=8)
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
    colors = _series_colors(len(data))
    afs = 6 if len(data) > 12 else 9
    for i, (p, g) in enumerate(zip(params, gaps)):
        sc_label = labels[i] if len(data) <= 8 else None
        ax.scatter(p, g, s=120, label=sc_label, color=colors[i])
        ax.annotate(labels[i], (p, g), xytext=(5, 5), textcoords="offset points", fontsize=afs)
    ax.set_xlabel("Trainable params (M)")
    ax.set_ylabel("Train − Val Gap (%)")
    ax.set_title("Generalization Gap vs Trainable Parameters")
    if len(data) <= 8:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "gap_vs_trainable_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot overfitting analysis from report JSONs")
    parser.add_argument(
        "--base-dir",
        default=str((Path(__file__).resolve().parents[3] / "numpy_weights" / "exploration-finetuning").resolve()),
        help="Project root: scans checkpoints/overfitting-wass-eltwise first, then legacy flat dirs",
    )
    parser.add_argument(
        "--reports-root",
        default=None,
        help="If set, load every overfitting_report.json under this directory (overrides scan order)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: base-dir/analysis/overfitting/overfitting_analysis_plots)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(
        args.base_dir, "analysis", "overfitting", "overfitting_analysis_plots"
    )
    os.makedirs(out_dir, exist_ok=True)

    data = load_reports(args.base_dir, reports_root=args.reports_root)
    if not data:
        tree_hint = os.path.join(args.base_dir, DEFAULT_WASS_ELTWISE_CHECKPOINTS)
        print(
            "No reports found. Train the sweep first, or pass --reports-root PATH "
            f"to a directory tree containing overfitting_report.json files.\n"
            f"  Default tree: {tree_hint}\n"
            f"  Legacy dirs under --base-dir: {', '.join(LEGACY_EXPERIMENT_DIRS)}"
        )
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
