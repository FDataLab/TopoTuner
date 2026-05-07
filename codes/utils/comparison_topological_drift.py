#!/usr/bin/env python3
"""
Side-by-side PDFs using **identical** line + drift-bar formulas as ``dynamic_topological_drift.py``
for **both** legacy pickle CSVs and ``wass_full_by_task`` CSVs.

**Same quantity on both sides (when checkpoints match)**

For each layer and training epoch ``k``, ``compute_wasserstein.compute_baseline_vs_epochs`` stores
``Wasserstein H{0,1}`` = Gudhi's ``wasserstein_distance`` between persistence diagrams at **baseline**
(pretrained / epoch 0) and **checkpoint epoch k** (epoch 6 = end of full finetune when six epochs exist).

Legacy CSV rows labeled ``Type == "Baseline vs Full Finetuned"`` are meant to be that same baseline-vs-epoch-k
comparison for full finetuning (your exported pickle workflow). ``wass_full_by_task`` CSVs spell this as
``Baseline vs epoch_k (proj)``. Large scale mismatches vs legacy usually mean **different diagrams**, baseline
paths, or an older CSV — not just plotting.

**Drift bars (current pipeline)**

Stacked bars use **absolute** epoch-to-epoch changes in baseline-to-checkpoint distance:

  ``|W_k - W_{k-1}|``

per layer (same ``W_k`` as line plots). If epoch ``0`` is missing from the pivot, a row of zeros is
prepended so the **0→1** step is ``|W_1 - 0|``.

The helper ``normalized_epoch_to_epoch_drift`` remains for legacy relative plots:

  ``|W_k - W_{k-1}| / (W_{k-1} + eps)``

Comparison drift-bar panels use **independent** y limits per side when scales differ.

Environment:
  WASS_CSV_FOLDER — legacy CSV directory (default like dynamic_topological_drift).
  NW_ROOT — repo root(s) with ``eval/split/weight_vs_baseline/wass_full_by_task/``.
            Default ``<repo>/numpy_weights``; multiple roots separated by ``:` or ``;``.
            Also probes ``<topo>/numpy_weights/exploration-finetuning`` and ancestor dirs of
            ``WASS_CSV_FOLDER``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
FIG_DIR = _HERE / "figs_wass_full_by_task" / "comparison_topological_drift"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WASS_CSV_FOLDER = Path(
    os.environ.get(
        "WASS_CSV_FOLDER",
        str((_HERE.parents[2] / "results" / "wasserstein_results").resolve()),
    )
)

WASS_REL = Path("eval/split/weight_vs_baseline/wass_full_by_task")

LEGACY_MODEL_TO_FAMILY = {
    "llama31_8b": "llama",
    "mistral7b": "mistral-7b-v03",
    "qwen_8b": "qwen-base",
}

LEGACY_MATRIX_SUFFIX = re.compile(r"_([qkvo])\.pkl$", re.IGNORECASE)
# Regex must match ``dynamic_topological_drift.py`` exactly for legacy pivots:
LEGACY_MATRIX_DYNAMIC = r"_([kqv])\.pkl$"

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

LABEL_FS = 15
TICK_FS = 13

EPS_DRIFT = 1e-8

LEGEND_KW = dict(fontsize=11, title_fontsize=12)

DISTANCE_COLUMNS = ("Wasserstein H0", "Wasserstein H1")

models = ["llama31_8b", "mistral7b", "qwen_8b"]
datasets = ["imdb", "sst2", "mmlu", "gsm8k"]
max_epoch = 6


def candidate_repo_roots() -> list[Path]:
    """Directories that might contain ``eval/split/weight_vs_baseline/...``."""
    roots: list[Path] = []
    seen: set[str] = set()

    def push(p: Path) -> None:
        try:
            p = p.expanduser().resolve()
        except Exception:
            return
        if not p.is_dir():
            return
        key = str(p)
        if key not in seen:
            roots.append(p)
            seen.add(key)

    raw = os.environ.get("NW_ROOT", str((_HERE.parents[2] / "numpy_weights").resolve())).strip()
    for part in re.split(r"[;:]", raw):
        if part.strip():
            push(Path(part.strip()))

    topo = _HERE.parents[2]
    for extra in (
        topo / "numpy_weights",
        topo / "numpy_weights" / "exploration-finetuning",
    ):
        push(extra)

    seeds = [
        WASS_CSV_FOLDER,
        topo,
        Path.cwd(),
    ]
    for seed in seeds:
        try:
            s = seed.resolve()
        except Exception:
            continue
        if not s.exists():
            continue
        push(s)
        for anc in s.parents:
            push(anc)
    return roots


def resolve_new_csv(family: str, dataset: str) -> Path | None:
    tail = WASS_REL / family / dataset / "wasserstein_results.csv"
    for root in candidate_repo_roots():
        for p in (root / tail, root / "exploration-finetuning" / tail):
            p = p.resolve()
            if p.is_file():
                return p
    return None


def legacy_projection_set(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path, usecols=["File"], dtype=str)
    s = df["File"].str.extract(LEGACY_MATRIX_SUFFIX)[0].dropna().str.lower()
    return set(s.unique())


def new_projection_set(csv_path: Path) -> set[str]:
    df = pd.read_csv(csv_path)
    if "Projection" not in df.columns:
        return set()
    return set(df["Projection"].astype(str).str.strip().str.lower().unique())


def csv_distance_columns(path: Path) -> set[str]:
    cols = pd.read_csv(path, nrows=0).columns
    return {c for c in DISTANCE_COLUMNS if c in cols}


def intersect_distance_types(legacy_path: Path, new_path: Path) -> list[str]:
    leg = csv_distance_columns(legacy_path)
    new = csv_distance_columns(new_path)
    out = [c for c in DISTANCE_COLUMNS if c in leg and c in new]
    return out


def _add_epoch_num(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ep = out["Epoch"]
    if pd.api.types.is_numeric_dtype(ep):
        out["epoch_num"] = ep.astype(int)
    else:
        out["epoch_num"] = (
            ep.astype(str).str.replace("epoch_", "", regex=False).astype(int)
        )
    out["layer"] = pd.to_numeric(
        out["File"].astype(str).str.extract(r"layer(\d+)_", expand=False),
        errors="coerce",
    )
    out = out.dropna(subset=["layer"])
    out["layer"] = out["layer"].astype(int)
    return out


def legacy_prepare_line_pivot(
    csv_path: Path,
    proj: str,
    distance_col: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    df = df[df["Type"] == "Baseline vs Full Finetuned"]
    df = df[df["Epoch"] <= max_epoch]
    df["matrix"] = df["File"].astype(str).str.extract(LEGACY_MATRIX_DYNAMIC)[0].str.lower()
    df["layer"] = pd.to_numeric(
        df["File"].astype(str).str.extract(r"layer(\d+)_")[0],
        errors="coerce",
    )
    df = df.dropna(subset=["matrix", "layer"])
    df["layer"] = df["layer"].astype(int)
    sub = df[df["matrix"] == proj.lower()].copy()
    if sub.empty or distance_col not in sub.columns:
        return None
    pivot = sub.pivot_table(index="Epoch", columns="layer", values=distance_col)
    pivot = pivot.sort_index()
    return pivot


def new_prepare_line_pivot(
    csv_path: Path,
    proj: str,
    distance_col: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    df = _add_epoch_num(df)
    sub = df[
        (df["Projection"].astype(str).str.strip().str.lower() == proj.lower())
        & (df["epoch_num"] <= max_epoch)
    ].copy()
    if sub.empty or distance_col not in sub.columns:
        return None
    pivot = sub.pivot_table(
        index="epoch_num",
        columns="layer",
        values=distance_col,
        aggfunc="first",
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def pivot_lines_nonempty(pivot: pd.DataFrame) -> bool:
    """Epoch curve exists for some epoch ≠ 0 (same notion as dynamic_topological_drift.py)."""
    return len([e for e in pivot.index if e != 0]) > 0


def plot_dynamic_lines(
    ax: Axes,
    pivot: pd.DataFrame,
    styles_map: dict[int, tuple],
) -> list:
    """Same loop as ``dynamic_topological_drift.py`` line plots."""
    epochs = [e for e in pivot.index if e != 0]
    handles = []
    for epoch in epochs:
        y = pivot.loc[epoch].values
        c, ls, m = styles_map.get(int(epoch), ("#333333", "-", "o"))
        (ln,) = ax.plot(
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
        handles.append(ln)
    return handles


def _strip_degenerate_epoch_zero_row(pivot: pd.DataFrame) -> pd.DataFrame:
    """Drop epoch ``0`` row when it is all zeros (baseline-vs-baseline).

    Some legacy CSVs include epoch ``0`` with ``Wasserstein H{0,1}=0`` everywhere. Using that row in
    ``|W_1-W_0|/(W_0+eps)`` blows up because ``W_0≈0`` leaves only ``eps`` in the denominator.
    Finetuned CSVs often omit epoch ``0`` entirely; stripping keeps drift comparable.
    """
    pivot = pivot.sort_index()
    if pivot.empty:
        return pivot
    idx0 = pivot.index[0]
    try:
        is_epoch_zero = int(idx0) == 0
    except (TypeError, ValueError):
        is_epoch_zero = False
    if not is_epoch_zero:
        return pivot
    row0 = pivot.iloc[0].astype(float)
    if not row0.notna().all():
        return pivot
    if float(row0.abs().max()) > 1e-15:
        return pivot
    return pivot.iloc[1:].copy()


def absolute_epoch_to_epoch_drift(pivot: pd.DataFrame) -> pd.DataFrame:
    """Absolute epoch-to-epoch drift: ``|W_k - W_{k-1}|`` per layer.

    Rows of ``pivot`` are baseline-vs-checkpoint Wasserstein distances ``W_k(layer)``. Output row
    indexed ``k`` is the magnitude of change from epoch ``k-1`` to ``k``.

    If epoch ``0`` is missing, a zero row is inserted so the first training step includes **0→1**.
    """
    p = pivot.astype(float).sort_index()
    idx = np.asarray(p.index, dtype=float)
    if not np.any(idx == 0):
        p.loc[0] = 0.0
        p = p.sort_index()
    return p.diff().abs().dropna(how="all")


def normalized_epoch_to_epoch_drift(pivot: pd.DataFrame, eps: float = EPS_DRIFT) -> pd.DataFrame:
    """Relative magnitude of successive changes along the epoch axis (nonnegative).

    Rows of ``pivot`` are ``W(baseline, ckpt_at_epoch_k)`` per layer (from ``compute_wasserstein``).
    Each output row indexed ``k`` is ``|W_k - W_{k-1}| / (W_{k-1} + eps)``. The raw difference before
    ``abs()`` can be negative when ``W`` decreases; plotted drift uses ``abs``.

    We only drop diff rows that are entirely undefined (typically epoch index ``1``); older code used
    ``iloc[2:]``, which wrongly dropped the **epoch 1→2** slice as well.
    """
    pivot = _strip_degenerate_epoch_zero_row(pivot)
    raw = (pivot.diff().abs() / (pivot.shift(1) + eps))
    return raw.dropna(how="all")


def plot_dynamic_stacked_bars(ax: Axes, pivot: pd.DataFrame, eps: float = EPS_DRIFT) -> None:
    """Same stacked drift bars as ``dynamic_topological_drift.py`` (absolute |ΔW|)."""
    _ = eps  # unused; kept for backward-compatible call signature
    deltas = absolute_epoch_to_epoch_drift(pivot)
    epochs = deltas.index.tolist()
    layers = deltas.columns.tolist()
    bottom = [0.0] * len(layers)
    for epoch in epochs:
        height = deltas.loc[epoch].fillna(0).values
        ax.bar(
            layers,
            height,
            bottom=bottom,
            label=f"Epoch {epoch - 1}→{epoch}",
            alpha=0.8,
        )
        bottom = [b + float(h) for b, h in zip(bottom, height)]


def driftbars_have_segments(pivot: pd.DataFrame, eps: float = EPS_DRIFT) -> bool:
    _ = eps
    deltas = absolute_epoch_to_epoch_drift(pivot)
    return len(deltas.index) > 0


def _drift_stack_max_height(pivot: pd.DataFrame, eps: float = EPS_DRIFT) -> float:
    """Max stacked-bar height across layers (sum of epoch-delta slices per layer)."""
    _ = eps
    deltas = absolute_epoch_to_epoch_drift(pivot)
    if deltas.empty:
        return 0.0
    return float(deltas.fillna(0).sum(axis=0).max())


def _union_layer_xlim(
    pivot_left: pd.DataFrame | None,
    pivot_right: pd.DataFrame | None,
) -> tuple[float | None, float | None]:
    xs: list[int] = []
    for pv in (pivot_left, pivot_right):
        if pv is not None and len(pv.columns):
            xs.extend(pv.columns.astype(int).tolist())
    if not xs:
        return None, None
    lo, hi = min(xs), max(xs)
    return float(lo) - 0.5, float(hi) + 0.5


def legacy_prepare_bar_pivot(
    csv_path: Path,
    proj: str,
    distance_col: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    df = df[df["Type"] == "Baseline vs Full Finetuned"]
    df = df[df["Epoch"] <= max_epoch]
    df["matrix"] = df["File"].astype(str).str.extract(LEGACY_MATRIX_DYNAMIC)[0].str.lower()
    df["layer"] = pd.to_numeric(
        df["File"].astype(str).str.extract(r"layer(\d+)_")[0],
        errors="coerce",
    )
    df = df.dropna(subset=["matrix", "layer"])
    df["layer"] = df["layer"].astype(int)
    sub = df[df["matrix"] == proj.lower()].copy()
    if (
        sub.empty
        or distance_col not in sub.columns
        or len(sub["Epoch"].unique()) <= 2
    ):
        return None
    return sub.pivot_table(index="Epoch", columns="layer", values=distance_col).sort_index()


def new_prepare_bar_pivot(
    csv_path: Path,
    proj: str,
    distance_col: str,
) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path)
    df = _add_epoch_num(df)
    sub = df[
        (df["Projection"].astype(str).str.strip().str.lower() == proj.lower())
        & (df["epoch_num"] <= max_epoch)
    ].copy()
    if (
        sub.empty
        or distance_col not in sub.columns
        or len(sub["epoch_num"].unique()) <= 2
    ):
        return None
    piv = sub.pivot_table(
        index="epoch_num",
        columns="layer",
        values=distance_col,
        aggfunc="first",
    ).sort_index()
    return piv.reindex(sorted(piv.columns), axis=1)


def _legend_if_labels(ax: Axes, **kwargs) -> None:
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def main() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    styles_map = {ep: (c, ls, m) for ep, c, ls, m in EPOCH_LINE_STYLES}

    roots_all = candidate_repo_roots()
    roots_preview = roots_all[:8]
    print(
        "candidate_repo_roots (first few): "
        + ", ".join(str(r) for r in roots_preview)
        + (" …" if len(roots_all) > 8 else "")
    )

    saved_lines = saved_bars = 0

    for model in models:
        family = LEGACY_MODEL_TO_FAMILY.get(model)
        if family is None:
            continue

        for dataset in datasets:
            legacy_path = WASS_CSV_FOLDER / f"wasserstein_{dataset}_{model}.csv"
            new_path = resolve_new_csv(family, dataset)

            if not legacy_path.is_file():
                print(f"[SKIP] missing legacy CSV: {legacy_path.name}")
                continue
            if new_path is None:
                print(
                    f"[SKIP] no wass_full_by_task CSV for family={family} dataset={dataset} "
                    f"(searched under NW_ROOT / ancestors of WASS_CSV_FOLDER)"
                )
                continue

            legacy_proj = legacy_projection_set(legacy_path)
            new_proj = new_projection_set(new_path)
            common_proj = sorted(legacy_proj & new_proj)
            if not common_proj:
                print(
                    f"[SKIP] no overlapping projections legacy={sorted(legacy_proj)} "
                    f"new={sorted(new_proj)} | {model} {dataset}"
                )
                continue

            distance_types = intersect_distance_types(legacy_path, new_path)
            if not distance_types:
                print(f"[SKIP] no overlapping distance columns | {model} {dataset}")
                continue

            for proj in common_proj:
                for distance_type in distance_types:
                    dist_short = "h0" if "H0" in distance_type else "h1"

                    piv_l = legacy_prepare_line_pivot(
                        legacy_path, proj, distance_type
                    )
                    piv_r = new_prepare_line_pivot(new_path, proj, distance_type)
                    if piv_l is None or piv_r is None:
                        continue
                    if not pivot_lines_nonempty(piv_l) or not pivot_lines_nonempty(piv_r):
                        continue

                    fig, axes = plt.subplots(
                        1,
                        2,
                        figsize=(18.5, 5.25),
                        constrained_layout=False,
                    )

                    axes[0].set_title("Legacy (pickle CSV)", fontsize=13)
                    h_left = plot_dynamic_lines(axes[0], piv_l, styles_map)

                    axes[1].set_title("wass_full_by_task", fontsize=13)
                    h_right = plot_dynamic_lines(axes[1], piv_r, styles_map)

                    if not h_left or not h_right:
                        plt.close(fig)
                        continue

                    handles_for_legend = h_left
                    x0, x1 = _union_layer_xlim(piv_l, piv_r)
                    for ax in axes:
                        ax.set_xlabel("Layer", fontsize=LABEL_FS)
                        ax.tick_params(axis="both", labelsize=TICK_FS)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        if x0 is not None and x1 is not None:
                            ax.set_xlim(x0, x1)
                        ax.set_ylabel(r"Topological Drift $\Delta_i$", fontsize=LABEL_FS)

                    y_vals = []
                    for pv in (piv_l, piv_r):
                        for e in pv.index:
                            if e == 0:
                                continue
                            row = pv.loc[e].astype(float)
                            y_vals.extend(row[np.isfinite(row)].tolist())
                    if y_vals:
                        lo, hi = min(y_vals), max(y_vals)
                        pad = (hi - lo) * 0.06 + 1e-9
                        axes[0].set_ylim(lo - pad, hi + pad)
                        axes[1].set_ylim(lo - pad, hi + pad)

                    uniq = {}
                    for h in handles_for_legend:
                        uniq.setdefault(h.get_label(), h)
                    fig.legend(
                        uniq.values(),
                        uniq.keys(),
                        title="Epoch",
                        loc="lower center",
                        bbox_to_anchor=(0.5, -0.02),
                        ncol=min(6, len(uniq)),
                        frameon=True,
                        fancybox=False,
                        facecolor="white",
                        edgecolor="#cccccc",
                        **LEGEND_KW,
                    )

                    fig.align_ylabels(axes)
                    plt.subplots_adjust(bottom=0.22)
                    outp = FIG_DIR / (
                        f"{model}_{dataset}_{proj}_{dist_short}_comparison_lines.pdf"
                    )
                    plt.savefig(outp, dpi=300, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig)
                    saved_lines += 1
                    print(f"Saved: {outp}")

                    piv_bar_l = legacy_prepare_bar_pivot(
                        legacy_path, proj, distance_type
                    )
                    piv_bar_r = new_prepare_bar_pivot(new_path, proj, distance_type)
                    if piv_bar_l is None or piv_bar_r is None:
                        continue
                    if not driftbars_have_segments(piv_bar_l) or not driftbars_have_segments(
                        piv_bar_r
                    ):
                        continue

                    fig2, axes2 = plt.subplots(1, 2, figsize=(18.5, 5.25))

                    axes2[0].set_title("Legacy (pickle CSV)", fontsize=13)
                    plot_dynamic_stacked_bars(axes2[0], piv_bar_l)

                    axes2[1].set_title("wass_full_by_task", fontsize=13)
                    plot_dynamic_stacked_bars(axes2[1], piv_bar_r)

                    bx0, bx1 = _union_layer_xlim(piv_bar_l, piv_bar_r)
                    drift_ylabel = (
                        r"Drift in $H_0$"
                        if "H0" in distance_type
                        else r"Drift in $H_1$"
                    )
                    for ax in axes2:
                        ax.set_xlabel("Layer", fontsize=LABEL_FS)
                        ax.tick_params(axis="both", labelsize=TICK_FS)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        if bx0 is not None and bx1 is not None:
                            ax.set_xlim(bx0, bx1)
                        ax.set_ylabel(drift_ylabel, fontsize=LABEL_FS)

                    # Independent y limits: legacy vs new amplitudes can differ legitimately; sharing
                    # max() hid the new pipeline when legacy had bogus epoch-0→1 spikes (zeros row).
                    for ax, piv_bar in zip(axes2, (piv_bar_l, piv_bar_r)):
                        ymax_local = _drift_stack_max_height(piv_bar)
                        if ymax_local > 0:
                            pad = ymax_local * 0.06 + 1e-12
                            ax.set_ylim(0.0, ymax_local + pad)

                    _legend_if_labels(
                        axes2[0],
                        title="Δ Between Epochs",
                        loc="upper right",
                        fontsize=LEGEND_KW["fontsize"],
                        title_fontsize=LEGEND_KW["title_fontsize"],
                    )
                    _legend_if_labels(
                        axes2[1],
                        title="Δ Between Epochs",
                        loc="upper right",
                        fontsize=LEGEND_KW["fontsize"],
                        title_fontsize=LEGEND_KW["title_fontsize"],
                    )

                    fig2.align_ylabels(axes2)
                    plt.subplots_adjust(bottom=0.18)
                    outp2 = FIG_DIR / (
                        f"{model}_{dataset}_{proj}_{dist_short}_comparison_driftbars.pdf"
                    )
                    plt.savefig(outp2, dpi=300, bbox_inches="tight", pad_inches=0.05)
                    plt.close(fig2)
                    saved_bars += 1
                    print(f"Saved: {outp2}")

    print(f"Done: {saved_lines} line PDFs, {saved_bars} drift-bar PDFs.")


if __name__ == "__main__":
    main()
