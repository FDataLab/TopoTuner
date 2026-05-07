#!/usr/bin/env python3
"""
Build summary artifacts from ``overfitting_report.json`` files under a checkpoint tree.

For any family × run layout (e.g. ``wass-eltwise``: ``<family>/<run>_.../overfitting_report.json``):

1. ``train_val_loss.pdf`` — all runs: train loss (solid) + val loss (dashed), same color per run.
2. ``val_vs_test_accuracy.pdf`` — scatter of final val vs test accuracy, one point per run.
3. ``metrics_table.csv`` — columns: run, test_acc, val_acc, train_acc, train_val_gap, train_test_gap

Usage::

    cd numpy_weights/exploration-finetuning/scripts
    ./summarize_overfitting_reports.py
    ./summarize_overfitting_reports.py --checkpoint-root ../checkpoints/overfitting-wass-eltwise --out-dir ../analysis/overfitting/wass_eltwise_summary
    ./summarize_overfitting_reports.py --family llama   # only reports under .../llama/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL_FS = 15
TICK_FS = 13
FIG_W, FIG_H = 12, 6


def _run_label(report_path: Path, checkpoint_root: Path) -> str:
    try:
        rel = report_path.parent.resolve().relative_to(checkpoint_root.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return report_path.parent.name


def load_reports(checkpoint_root: Path, family_filter: str | None) -> list[tuple[str, dict]]:
    root = checkpoint_root.resolve()
    paths = sorted(root.rglob("overfitting_report.json"))
    out: list[tuple[str, dict]] = []
    for p in paths:
        if family_filter:
            try:
                rel = p.parent.resolve().relative_to(root)
                first = rel.parts[0] if rel.parts else ""
                if first != family_filter:
                    continue
            except ValueError:
                continue
        with open(p) as f:
            data = json.load(f)
        out.append((_run_label(p, root), data))
    return out


def plot_train_val_loss(rows: list[tuple[str, dict]], out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    cmap = plt.get_cmap("tab20")
    for i, (label, r) in enumerate(rows):
        pe = r.get("per_epoch") or {}
        tl = pe.get("train_loss") or []
        vl = pe.get("val_loss") or []
        if not tl or not vl:
            continue
        ep = list(range(1, len(tl) + 1))
        c = cmap(i % 20)
        ax.plot(ep, tl, "-o", color=c, linewidth=2.0, markersize=5, label=f"{label} (train)")
        ax.plot(ep, vl, "--s", color=c, linewidth=2.0, markersize=5, alpha=0.85, label=f"{label} (val)")
    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Loss", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title("Train vs validation loss (all runs)", fontsize=LABEL_FS)
    ncol = 2 if len(rows) > 4 else 1
    fs = 7 if len(rows) > 6 else 9
    ax.legend(ncol=ncol, fontsize=fs, loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(right=0.72)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_val_vs_test(rows: list[tuple[str, dict]], out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")
    vals, tests, labels = [], [], []
    for label, r in rows:
        fin = r.get("final") or {}
        va = fin.get("val_accuracy")
        ta = fin.get("test_accuracy")
        if va is None or ta is None:
            continue
        vals.append(float(va) * 100)
        tests.append(float(ta) * 100)
        labels.append(label)
    for i, (v, t, lab) in enumerate(zip(vals, tests, labels)):
        c = cmap(i % 20)
        ax.scatter(v, t, s=90, color=c, zorder=3)
        ax.annotate(lab, (v, t), xytext=(4, 4), textcoords="offset points", fontsize=8)
    if vals:
        lo = min(vals + tests) - 2
        hi = max(vals + tests) + 2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation accuracy (%)", fontsize=LABEL_FS)
    ax.set_ylabel("Test accuracy (%)", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_title("Validation vs test accuracy (final)", fontsize=LABEL_FS)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _fmt_float_cell(obj: dict | None, key: str) -> str:
    if not obj:
        return ""
    v = obj.get(key)
    return "" if v is None else f"{float(v):.6f}"


def write_metrics_csv(rows: list[tuple[str, dict]], out_csv: Path) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "test_acc", "val_acc", "train_acc", "train_val_gap", "train_test_gap"])
        for label, r in rows:
            fin = r.get("final") or {}
            gap = r.get("overfitting_gap")
            w.writerow(
                [
                    label,
                    _fmt_float_cell(fin, "test_accuracy"),
                    _fmt_float_cell(fin, "val_accuracy"),
                    _fmt_float_cell(fin, "train_accuracy"),
                    _fmt_float_cell(gap, "train_minus_val"),
                    _fmt_float_cell(gap, "train_minus_test"),
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).resolve().parent
    nw_root = script_dir.parent
    ap.add_argument(
        "--checkpoint-root",
        type=Path,
        default=nw_root / "checkpoints" / "overfitting-wass-eltwise",
        help="Root containing **/overfitting_report.json (default: NW_ROOT/checkpoints/overfitting-wass-eltwise)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=nw_root / "analysis" / "overfitting" / "wass_eltwise_summary",
        help="Output directory for PDFs + CSV",
    )
    ap.add_argument(
        "--family",
        type=str,
        default=None,
        help="If set, only include reports under this first path segment (e.g. llama, qwen-base, mistral-7b-v03)",
    )
    args = ap.parse_args()

    rows = load_reports(args.checkpoint_root, args.family)
    if not rows:
        raise SystemExit(f"No overfitting_report.json under {args.checkpoint_root} (family={args.family!r})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_train_val_loss(rows, args.out_dir / "train_val_loss.pdf")
    plot_val_vs_test(rows, args.out_dir / "val_vs_test_accuracy.pdf")
    write_metrics_csv(rows, args.out_dir / "metrics_table.csv")

    print(f"Wrote {len(rows)} runs to {args.out_dir.resolve()}")
    print(f"  {args.out_dir / 'train_val_loss.pdf'}")
    print(f"  {args.out_dir / 'val_vs_test_accuracy.pdf'}")
    print(f"  {args.out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()
