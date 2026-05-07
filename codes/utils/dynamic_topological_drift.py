#!/usr/bin/env python3
"""Plots from ``wass_full_by_task/.../wasserstein_results.csv`` → ``figs_wass_full_by_task/<subdir>/``.

**Defaults (``topological_drift5``):** **H₀ only**, **max-normalized** to ``[0,1]`` with **one divisor per figure**
(lines: max over epochs×layers in that PDF; drift bars: max stacked height in that PDF), so each plot touches ``y=1``.
Pass ``--normalize-across-models`` to restore the legacy divisors: **one divisor per (dataset, projection)** — max over
the **three models** for each ``q/k/v/o``. Pass ``--no-normalize-by-max`` for raw scales. Pass ``--include-h1`` to also
emit ``Wasserstein H1`` panels.

**Normalization (when ``--normalize-by-max`` is on):**

- **Default:** Each line PDF and each drift-bar PDF uses its **own** maximum (within that model/dataset/projection
  figure); ``ylim`` is ``[0, 1]``.
- **With ``--normalize-across-models``:** **Lines:** For each *(dataset, projection)*, Wasserstein values are divided by
  the maximum **H\\ :math:`_0`** across the three models (with ``--include-h1``, divisors pool **H\\ :math:`_0`** and
  **H\\ :math:`_1`** panel maxima across models). **Bars:** Same grouping — max stacked drift across the three models per
  *(dataset, projection)*.

**Drift segments:** absolute ``|W_k - W_{k-1}|`` per layer (see ``comparison_topological_drift.absolute_epoch_to_epoch_drift``).
If epoch ``0`` is missing from the pivot, a zero row is prepended so the **0→1** step is included.

**Caption (drift):** Drift bars show absolute epoch-to-epoch changes in baseline-to-checkpoint Wasserstein distance, i.e.
``|W_k - W_{k-1}|``.

**Axes:** Top and right spines hidden (L-shaped frame only). Legends use a white filled frame (same style family as
``plot_wass_full_by_task``).

Env ``NW_ROOT``: repo root containing ``eval/split/weight_vs_baseline/wass_full_by_task/`` (see
``comparison_topological_drift.resolve_new_csv``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import comparison_topological_drift as ctd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIG_SUBDIR = "topological_drift5"

YLABEL_LINES_H0 = r"Wasserstein Distance in $H_0$"
YLABEL_LINES_H1 = r"Wasserstein Distance in $H_1$"
YLABEL_BARS = r"Topological Drift $\Delta_i$"

# Match ``plot_wass_full_by_task`` inset legend styling (white box, light edge).
LEGEND_BOX_KWARGS = dict(
    fontsize=11,
    title_fontsize=12,
    frameon=True,
    fancybox=False,
    facecolor="white",
    edgecolor="#cccccc",
    framealpha=0.94,
    columnspacing=0.75,
    handlelength=1.05,
    handletextpad=0.5,
    borderpad=0.45,
    labelspacing=0.35,
)

EPOCH_LINE_STYLES = (
    (1, "#1f77b4", "-", "o"),
    (2, "#ff7f0e", "--", "s"),
    (3, "#2ca02c", ":", "^"),
    (4, "#9467bd", (0, (3, 1, 1, 1)), "v"),
    (5, "#8c564b", "--", "D"),
    (6, "#d62728", "-", "D"),
)
LINEWIDTH_LINE = 2.2
MARKERSIZE_LINE = 7

models = ["llama31_8b", "mistral7b", "qwen_8b"]
datasets = ["imdb", "sst2", "mmlu", "gsm8k"]
matrix_types = ["q", "k", "v", "o"]


def _distance_types(include_h1: bool) -> tuple[str, ...]:
    if include_h1:
        return ("Wasserstein H0", "Wasserstein H1")
    return ("Wasserstein H0",)


def _norm_stem_suffix(fig_subdir: str, *, normalize_by_max: bool) -> str:
    if not normalize_by_max:
        return ""
    name = Path(fig_subdir).name
    if name in ("topological_drift3", "topological_drift4", "topological_drift5"):
        return ""
    return "_normmax"


def _open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fig-subdir",
        default=DEFAULT_FIG_SUBDIR,
        help=f"Under figs_wass_full_by_task/ (default: {DEFAULT_FIG_SUBDIR!r})",
    )
    ap.add_argument(
        "--normalize-by-max",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Scale to [0,1] by max (default: per figure). Use --no-normalize-by-max for raw values.",
    )
    ap.add_argument(
        "--normalize-across-models",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "With --normalize-by-max: divide by max over all three models per (dataset, projection) "
            "(legacy comparability). Default is per-figure max so each PDF touches y=1."
        ),
    )
    ap.add_argument(
        "--include-h1",
        action="store_true",
        help="Also plot Wasserstein H1 (default: H0 only).",
    )
    args = ap.parse_args()

    figs_dir = os.path.join(_HERE, "figs_wass_full_by_task", args.fig_subdir)
    os.makedirs(figs_dir, exist_ok=True)

    plt.rcParams.update({"figure.facecolor": "white"})
    styles_map = {ep: (c, ls, m) for ep, c, ls, m in EPOCH_LINE_STYLES}

    dtypes = _distance_types(args.include_h1)

    line_max_by_dataset: dict[tuple[str, str], float]
    bar_max_by_dataset: dict[tuple[str, str], float]
    if args.normalize_by_max and args.normalize_across_models:
        line_max_by_dataset, bar_max_by_dataset = _collect_global_dataset_maxima(dtypes)
    else:
        line_max_by_dataset = {}
        bar_max_by_dataset = {}

    stem_suffix = _norm_stem_suffix(args.fig_subdir, normalize_by_max=args.normalize_by_max)

    for model in models:
        family = ctd.LEGACY_MODEL_TO_FAMILY.get(model)
        if family is None:
            continue
        for dataset in datasets:
            csv_path = ctd.resolve_new_csv(family, dataset)
            if csv_path is None:
                continue

            csv_cols = pd.read_csv(csv_path, nrows=0).columns
            for distance_type in dtypes:
                if distance_type not in csv_cols:
                    continue
                dist_short = "h0" if "H0" in distance_type else "h1"

                for mat in matrix_types:
                    pivot = ctd.new_prepare_line_pivot(csv_path, mat, distance_type)
                    if pivot is None or not ctd.pivot_lines_nonempty(pivot):
                        continue

                    line_denom = (
                        float(line_max_by_dataset[(dataset, mat)])
                        if args.normalize_by_max and args.normalize_across_models
                        else None
                    )
                    _save_lines(
                        figs_dir,
                        styles_map,
                        model,
                        dataset,
                        mat,
                        dist_short,
                        pivot,
                        normalize_by_max=args.normalize_by_max,
                        stem_suffix=stem_suffix,
                        dataset_line_denom=line_denom,
                    )

                    if pivot.index.nunique() <= 2:
                        continue
                    deltas = ctd.absolute_epoch_to_epoch_drift(pivot)
                    if deltas.empty:
                        continue
                    bar_denom = (
                        float(bar_max_by_dataset[(dataset, mat)])
                        if args.normalize_by_max and args.normalize_across_models
                        else None
                    )
                    _save_driftbars(
                        figs_dir,
                        mat,
                        dist_short,
                        model,
                        dataset,
                        deltas,
                        normalize_by_max=args.normalize_by_max,
                        stem_suffix=stem_suffix,
                        dataset_bar_denom=bar_denom,
                    )


def line_maxima_h0_per_dataset_projection() -> dict[tuple[str, str], float]:
    """Maximum ``W`` (H\\ :math:`_0`) over the three models for each *(dataset, projection)* — shared divisors for line plots."""
    line, _ = _collect_global_dataset_maxima(("Wasserstein H0",))
    return line


def _collect_global_dataset_maxima(
    dtypes: tuple[str, ...],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Per (dataset, projection): line max and bar-stack max over the three models (per homology in dtypes)."""
    line_max: dict[tuple[str, str], float] = {
        (d, m): 0.0 for d in datasets for m in matrix_types
    }
    bar_max: dict[tuple[str, str], float] = {
        (d, m): 0.0 for d in datasets for m in matrix_types
    }
    for model in models:
        family = ctd.LEGACY_MODEL_TO_FAMILY.get(model)
        if family is None:
            continue
        for dataset in datasets:
            csv_path = ctd.resolve_new_csv(family, dataset)
            if csv_path is None:
                continue
            csv_cols = pd.read_csv(csv_path, nrows=0).columns
            for distance_type in dtypes:
                if distance_type not in csv_cols:
                    continue
                for mat in matrix_types:
                    pivot = ctd.new_prepare_line_pivot(csv_path, mat, distance_type)
                    if pivot is None or not ctd.pivot_lines_nonempty(pivot):
                        continue
                    epochs_plot = [e for e in pivot.index if e != 0]
                    if epochs_plot:
                        sub = pivot.loc[epochs_plot].astype(float)
                        wloc = float(np.nanmax(sub.values))
                        line_max[(dataset, mat)] = max(line_max[(dataset, mat)], wloc)
                    if pivot.index.nunique() <= 2:
                        continue
                    deltas = ctd.absolute_epoch_to_epoch_drift(pivot)
                    if deltas.empty:
                        continue
                    col_sums = deltas.fillna(0.0).sum(axis=0)
                    sloc = float(np.nanmax(col_sums.values)) if len(col_sums) else 0.0
                    bar_max[(dataset, mat)] = max(bar_max[(dataset, mat)], sloc)
    return line_max, bar_max


def _save_lines(
    figs_dir: str,
    styles_map: dict,
    model: str,
    dataset: str,
    mat: str,
    dist_short: str,
    pivot: pd.DataFrame,
    *,
    normalize_by_max: bool,
    stem_suffix: str,
    dataset_line_denom: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_plot = [e for e in pivot.index if e != 0]
    wmax = 0.0
    if normalize_by_max and epochs_plot:
        sub = pivot.loc[epochs_plot].astype(float)
        local_max = float(np.nanmax(sub.values))
        if dataset_line_denom is not None and dataset_line_denom > 0.0:
            wmax = float(dataset_line_denom)
        else:
            wmax = local_max
    for epoch in epochs_plot:
        y = pivot.loc[epoch].values.astype(float)
        if normalize_by_max and wmax > 0.0:
            y = y / wmax
        c, ls, m = styles_map.get(int(epoch), ("#333333", "-", "o"))
        ax.plot(
            pivot.columns,
            y,
            label=f"Epoch {epoch}",
            color=c,
            linestyle=ls,
            marker=m,
            linewidth=LINEWIDTH_LINE,
            markersize=MARKERSIZE_LINE,
            alpha=0.95,
        )
    ax.set_xlabel("Layer", fontsize=15)
    y_axis = YLABEL_LINES_H0 if dist_short == "h0" else YLABEL_LINES_H1
    ax.set_ylabel(y_axis, fontsize=15)
    ax.legend(title="Epoch", **LEGEND_BOX_KWARGS)
    ax.tick_params(axis="both", labelsize=12)
    _open_axes(ax)
    fig.tight_layout()
    if normalize_by_max:
        ax.set_ylim(0.0, 1.0)
    outp = os.path.join(figs_dir, f"{model}_{dataset}_{mat}_{dist_short}_lines{stem_suffix}.pdf")
    fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Saved: {outp}")


def _save_driftbars(
    figs_dir: str,
    mat: str,
    dist_short: str,
    model: str,
    dataset: str,
    deltas: pd.DataFrame,
    *,
    normalize_by_max: bool,
    stem_suffix: str,
    dataset_bar_denom: float | None,
) -> None:
    deltas_plot = deltas.astype(float).copy()
    if normalize_by_max:
        col_sums = deltas_plot.fillna(0.0).sum(axis=0)
        local_stack_max = float(np.nanmax(col_sums.values)) if len(col_sums) else 0.0
        if dataset_bar_denom is not None and dataset_bar_denom > 0.0:
            stack_max = float(dataset_bar_denom)
        else:
            stack_max = local_stack_max
        if stack_max > 0.0:
            deltas_plot = deltas_plot / stack_max

    epochs = deltas_plot.index.tolist()
    layers = deltas_plot.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0.0] * len(layers)
    for epoch in epochs:
        height = deltas_plot.loc[epoch].fillna(0).values
        ax.bar(
            layers,
            height,
            bottom=bottom,
            label=f"Epoch {int(epoch) - 1}→{int(epoch)}",
            alpha=0.8,
        )
        bottom = [b + float(h) for b, h in zip(bottom, height)]

    ax.set_xlabel("Layer", fontsize=15)
    ax.set_ylabel(YLABEL_BARS, fontsize=15)
    ax.legend(title="Δ Between Epochs", **LEGEND_BOX_KWARGS)
    ax.tick_params(axis="both", labelsize=12)
    _open_axes(ax)
    fig.tight_layout()
    if normalize_by_max:
        ax.set_ylim(0.0, 1.0)
    outp = os.path.join(figs_dir, f"{model}_{dataset}_{mat}_{dist_short}_driftbars{stem_suffix}.pdf")
    fig.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Saved: {outp}")


if __name__ == "__main__":
    main()
