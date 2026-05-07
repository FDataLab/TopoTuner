#!/usr/bin/env python3
"""Topology stopping vs η (full FT): drift builds CSV (write_table); figures read CSV only.

Outputs: ``topology_stopping_eta_table.csv``, four PDFs of η vs accuracy at t*(η), and four PDFs of
η vs stopping epoch t*(η). See ``plot_topology_stopping_eta`` for data paths.

PDFs use **x-axis η (rho) from 0 to 0.4** by default to emphasize the strict-threshold regime; pass
``--threshold-xmax 1`` for the full 0–1 range. Line weight and typography match ``epoch_accuracies_3``
(15 pt labels / ticks / legend, bold polylines)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import comparison_topological_drift as ctd

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as e:
    raise SystemExit(f"matplotlib required: {e}") from e

from plot_topology_stopping_eta import (
    BACKBONES,
    drift_path_with_manifest,
    intersection_epochs,
    load_accuracy_epochs,
    resolve_accuracy_path,
    stopping_epoch,
)

DRIFT_PROJECTIONS = ["v", "o"]
DATASETS = ("gsm8k", "imdb", "sst2", "mmlu")
MODEL_SLUG = {"llama": "llama", "qwen-base": "qwen-base", "mistral-7b-v03": "mistral"}
# Legend text with concrete checkpoints (full FT grid in this repo).
LEGEND_MODEL_NAME = {
    "llama": "LLaMA-3.1-8B",
    "qwen-base": "Qwen3-8B-Base",
    "mistral-7b-v03": "Mistral-7B-v0.3",
}
_METHOD = "full"
_METRIC = "Wasserstein H0"
_TMIN = 2


def load_mean_normalized_drift_over_monitored(
    path: str, metric_col: str, projections: Sequence[str]
) -> tuple[Dict[int, float], List[int]]:
    csv_path = Path(path)
    eps = ctd.EPS_DRIFT
    per_epoch: Dict[int, List[float]] = {}

    for proj in projections:
        piv = ctd.new_prepare_bar_pivot(csv_path, proj.strip(), metric_col)
        if piv is None or piv.empty:
            continue
        piv = ctd._strip_degenerate_epoch_zero_row(piv.sort_index())
        if piv.empty or 1 not in piv.index:
            continue

        for layer_col in piv.columns:
            s = piv[layer_col].astype(float)
            d1 = s.loc[1]
            if pd.isna(d1):
                continue
            denom = max(float(d1), eps)

            for idx in piv.index:
                try:
                    tp = int(idx)
                except (TypeError, ValueError):
                    continue
                if tp < 2:
                    continue
                tm1 = tp - 1
                if tm1 not in piv.index:
                    continue
                cur = float(s.loc[idx])
                prev = float(s.loc[tm1])
                if pd.isna(cur) or pd.isna(prev):
                    continue
                delta = (cur - prev) / denom
                per_epoch.setdefault(tp, []).append(delta)

    merged = {t: sum(vs) / len(vs) for t, vs in sorted(per_epoch.items())}
    return merged, sorted(merged.keys())


SERIES_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
)
# Match ``plot_wass_full_by_task`` ``epoch_accuracies_3/``: bold curves + 15 pt type everywhere.
LINEWIDTH_LINE = 3.2
ACC_Y_AUTOSCALE_EXPAND = 2
# Default x-axis for η (threshold) curves: focus on the low-η regime where stopping epochs move most.
DEFAULT_THRESHOLD_XMAX = 0.4
LABEL_FS = 15
AXIS_LABEL_FONTSIZE = LABEL_FS
TICK_LABELSIZE = LABEL_FS
LEGEND_FONTSIZE = LABEL_FS
LEGEND_TITLE_FONTSIZE = LABEL_FS


def _spines_left_bottom_only(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_vertical(ax: Any) -> None:
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
        frameon=True,
        loc="best",
        title="model",
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )


def _csv_float3(v: Any) -> str:
    if v == "" or v is None:
        return ""
    x = float(v)
    if x != x:  # NaN
        return ""
    return f"{x:.3f}"


def write_table(repo: str, out_csv: str) -> List[Dict[str, Any]]:
    eta_values = [round(i * 0.05, 2) for i in range(21)]
    rows: List[Dict[str, Any]] = []

    for bb in BACKBONES:
        slug = MODEL_SLUG[bb]
        for ds in DATASETS:
            acc_path = resolve_accuracy_path(repo, ds, bb, _METHOD)
            drift_path = drift_path_with_manifest(repo, ds, bb, _METHOD, {})
            acc_map: Dict[int, float] = {}
            drift_map: Dict[int, float] = {}
            acc_ep: List[int] = []
            drift_ep: List[int] = []

            if acc_path:
                try:
                    acc_map, acc_ep = load_accuracy_epochs(acc_path)
                except (OSError, ValueError) as e:
                    print(f"[warn] accuracy {acc_path}: {e}", file=sys.stderr)
            if drift_path:
                try:
                    drift_map, drift_ep = load_mean_normalized_drift_over_monitored(
                        drift_path, _METRIC, DRIFT_PROJECTIONS
                    )
                except (OSError, ValueError) as e:
                    print(f"[warn] drift {drift_path}: {e}", file=sys.stderr)

            avail = intersection_epochs(acc_ep, drift_ep) if acc_ep and drift_ep else []
            r_eff = {t: drift_map[t] for t in avail} if avail else {}
            acc_e6_s = "" if acc_map.get(6) is None else _csv_float3(acc_map[6])

            for eta in eta_values:
                rec: Dict[str, Any] = {
                    "eta": f"{float(eta):.3f}",
                    "model": slug,
                    "dataset": ds,
                    "stop_epoch": "",
                    "epoch_acc": "",
                    "acc_at_epoch6": acc_e6_s,
                }
                if not r_eff:
                    rows.append(rec)
                    continue
                try:
                    t_star, _ = stopping_epoch(r_eff, float(eta), _TMIN)
                except ValueError:
                    rows.append(rec)
                    continue
                rec["stop_epoch"] = int(t_star)
                av = acc_map.get(t_star)
                rec["epoch_acc"] = "" if av is None else _csv_float3(av)
                rows.append(rec)

    cols = ["eta", "model", "dataset", "stop_epoch", "epoch_acc", "acc_at_epoch6"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows).")
    return rows


def _save_eta_vs_stop_epoch_pdf(
    plot_dir: str,
    fname: str,
    series: List[tuple[Any, Any, str]],
    *,
    threshold_xmax: float = DEFAULT_THRESHOLD_XMAX,
) -> str | None:
    """η-vs-stopping-epoch figure — **CSV only**; drift is **not** recomputed here.

    The CSV columns ``eta``, ``stop_epoch``, … are produced earlier by ``write_table``. This plot is::

        x = η (threshold column),
        y = stop_epoch = t*(η) (column),

    one curve per model for the chosen dataset.

    **Mathematical definition** (how ``stop_epoch`` was computed when the CSV was written):

    For each model–dataset pair, the stopping score is

        r^(t) = (1 / |𝒲|) * Σ_{i ∈ 𝒲} Δ_i^(t),

    where 𝒲 is the monitored set of matrices (here **V** and **O** projections across layers).

    Per-matrix topological drift (paper-style numerator/denominator; CSV generation clamps tiny
    ``D_i^(1)`` with ε for stability — see ``load_mean_normalized_drift_over_monitored``):

        Δ_i^(t) = (D_i^(t) - D_i^(t-1)) / D_i^(1),

    where ``D_i^(t)`` is the Wasserstein H0 distance between the pretrained matrix and the epoch-``t``
    checkpoint.

    For each threshold η in {0, 0.05, …, 1.00},

        t*(η) = min { t ≥ 2 : r^(t) ≤ η }.

    If no epoch satisfies the condition, the table uses the final available aligned epoch.

    **Reading the curve**

    - Smaller η is stricter → usually stop **later** (larger ``t*``).
    - Larger η is more permissive → usually stop **earlier** (smaller ``t*``).
    - η = 0 is very strict; early stopping only if ``r^(t)`` becomes ≤ 0 at some ``t``.
    """
    if not series:
        return None
    nc = len(SERIES_COLORS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (eta_x, stop_y, label) in enumerate(series):
        color = SERIES_COLORS[i % nc]
        # x-axis: η threshold; y-axis: selected stopping epoch t*(η) from the CSV (not recomputed).
        ax.plot(
            eta_x,
            stop_y,
            label=label,
            color=color,
            linestyle="-",
            linewidth=LINEWIDTH_LINE,
            alpha=0.95,
        )

    ax.set_xlabel(r"$\eta$ threshold", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Stopping epoch $t^*(\eta)$", fontsize=AXIS_LABEL_FONTSIZE)
    if threshold_xmax is not None and threshold_xmax > 0:
        ax.set_xlim(0.0, min(1.0, float(threshold_xmax)))
    all_y = [float(y) for _, sy, _ in series for y in sy if not (isinstance(y, float) and y != y)]
    if all_y:
        ylo, yhi = min(all_y), max(all_y)
        ax.set_ylim(ylo - 0.5, yhi + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _spines_left_bottom_only(ax)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    _legend_vertical(ax)
    fig.tight_layout()

    out_pdf = os.path.join(plot_dir, fname)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Wrote {out_pdf} ({len(series)} curves).")
    return out_pdf


def _save_eta_acc_pdf(
    plot_dir: str,
    fname: str,
    series: List[tuple[Any, Any, str]],
    *,
    threshold_xmax: float = DEFAULT_THRESHOLD_XMAX,
) -> str | None:
    """η vs accuracy at ``epoch_acc`` for each model (CSV only; drift not recomputed)."""
    if not series:
        return None
    nc = len(SERIES_COLORS)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (eta_x, acc_y, label) in enumerate(series):
        color = SERIES_COLORS[i % nc]
        ax.plot(
            eta_x,
            acc_y,
            label=label,
            color=color,
            linestyle="-",
            linewidth=LINEWIDTH_LINE,
            alpha=0.95,
        )

    ax.set_xlabel(r"$\eta$ threshold", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Accuracy at selected epoch", fontsize=AXIS_LABEL_FONTSIZE)
    ax.relim()
    ax.autoscale_view()
    lo, hi = ax.get_ylim()
    mid = (lo + hi) / 2.0
    span = hi - lo
    if span <= 0:
        span = 1e-6
    half = span * ACC_Y_AUTOSCALE_EXPAND / 2.0
    nlo = max(0.0, mid - half)
    nhi = min(1.0, mid + half)
    ax.set_ylim(nlo, nhi)
    if threshold_xmax is not None and threshold_xmax > 0:
        ax.set_xlim(0.0, min(1.0, float(threshold_xmax)))
    _spines_left_bottom_only(ax)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE)
    _legend_vertical(ax)
    fig.tight_layout()

    out_pdf = os.path.join(plot_dir, fname)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Wrote {out_pdf} ({len(series)} curves).")
    return out_pdf


def plot_grouped(
    csv_path: str,
    plot_dir: str,
    *,
    threshold_xmax: float = DEFAULT_THRESHOLD_XMAX,
) -> List[str]:
    """Load CSV and write eight PDFs: η vs ``epoch_acc`` and η vs ``stop_epoch`` per dataset."""
    plt.rcParams.update({"figure.facecolor": "white"})
    df = pd.read_csv(csv_path)
    eta_col = "eta" if "eta" in df.columns else "rho"
    df["epoch_acc"] = pd.to_numeric(df["epoch_acc"], errors="coerce")
    df["stop_epoch"] = pd.to_numeric(df["stop_epoch"], errors="coerce")
    os.makedirs(plot_dir, exist_ok=True)

    out: List[str] = []
    for ds in DATASETS:
        series_acc: List[tuple[Any, Any, str]] = []
        series_stop: List[tuple[Any, Any, str]] = []
        for bb in BACKBONES:
            slug = MODEL_SLUG[bb]
            sub = df[(df["model"] == slug) & (df["dataset"] == ds)].sort_values(eta_col)
            curve_acc = sub.dropna(subset=["epoch_acc"])
            if not curve_acc.empty:
                series_acc.append(
                    (
                        curve_acc[eta_col].values,
                        curve_acc["epoch_acc"].values,
                        LEGEND_MODEL_NAME[bb],
                    ),
                )
            curve_stop = sub.dropna(subset=["stop_epoch"])
            if not curve_stop.empty:
                series_stop.append(
                    (
                        curve_stop[eta_col].values,
                        curve_stop["stop_epoch"].values,
                        LEGEND_MODEL_NAME[bb],
                    ),
                )
        if p := _save_eta_acc_pdf(
            plot_dir,
            f"topology_stopping_eta_by_dataset_{ds}.pdf",
            series_acc,
            threshold_xmax=threshold_xmax,
        ):
            out.append(p)
        if p := _save_eta_vs_stop_epoch_pdf(
            plot_dir,
            f"topology_stopping_eta_vs_stop_epoch_{ds}.pdf",
            series_stop,
            threshold_xmax=threshold_xmax,
        ):
            out.append(p)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        default=os.environ.get(
            "NW_EXPLORATION_FINETUNING_ROOT",
            str((Path(__file__).resolve().parents[3] / "numpy_weights" / "exploration-finetuning").resolve()),
        ),
    )
    ap.add_argument("--output-csv", default="topology_stopping_eta_table.csv")
    ap.add_argument(
        "--threshold-xmax",
        type=float,
        default=DEFAULT_THRESHOLD_XMAX,
        help=(
            "Right limit of the η (rho) threshold x-axis on stopping PDFs "
            f"(default {DEFAULT_THRESHOLD_XMAX}; use 1 for full 0–1 range)."
        ),
    )
    args = ap.parse_args(argv)

    repo = os.path.abspath(os.path.expanduser(args.repo_root))
    out_csv = os.path.abspath(os.path.expanduser(args.output_csv))
    plot_dir = os.path.dirname(out_csv) or "."

    write_table(repo, out_csv)
    pdfs = plot_grouped(out_csv, plot_dir, threshold_xmax=args.threshold_xmax)
    print(f"CSV path: {out_csv}")
    for p in pdfs:
        print(f"Plot path: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
