#!/usr/bin/env python3
"""
Plots from ``wass_full_by_task/.../wasserstein_results.csv`` (``compute_wasserstein`` output).

Figures go under ``figs_wass_full_by_task/`` (or ``--out-dir`` as that tree root) in **flat** sibling folders (no per‑experiment subfolders; filenames encode ``<family>_<task>``):

- ``topological_drift_delta/`` — drift vs layer (**q/k/v/o** where available).
- ``topological_drift_bars/`` — stacked bars (**q/k/v/o** where available).
- ``wasserstein_distance_dataset_comparison3/`` (override with ``--dataset-comparison-subdir``) — four‑dataset snapshots (**q/k/v/o** where available).
- ``eltwise_mean_absolute_delta_dataset_comparison2/`` — eltwise dataset snapshots (same four‑dataset
  layout as Wasserstein; **v/o** only in numpy); baseline =
  ``wass/<family>/baseline/numpy/epoch_0``, finetuned = ``wass_full_by_task/.../numpy/epoch_*``.
  Metric **mean(|new-old|)** (**raw** ``mean(|Δ|)`` on the axis; matplotlib's default tick formatting).
- ``epoch_accuracies/`` — accuracy vs epoch (**Full**, **LoRA**, **Wass‑3**), **y-axis fixed 0–100%** (avoids misleading
  zoom when accuracies are close, e.g. 94–96%); GSM8K uses ``eval/split/gsm8k/<family>/json/`` (Llama **Wass‑6** JSON; qwen-base **Wass‑3**).
- ``epoch_accuracies_3/`` — same curves and **same 0–100% y-axis**, **bolder** linewidth/marker; **15 pt** on axis labels, tick values, and legend (entry + title). The third curve is labeled **TDA-High3** wherever the standard plot uses **Wass-3** (GSM8K qwen/mistral and SST2/IMDb/MMLU wass-high-3); Llama GSM8K still shows **Wass-6**.

CSV columns for Wasserstein: ``Type``, ``File``, ``Wasserstein H0``, ``Epoch``, ``Projection``.

**Margins:** Figure ``(10, 5)``, labels **15 pt**, ticks **13 pt**, ``LEGEND_BOX_KWARGS`` for inset legends.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter, ScalarFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_BASE_DEFAULT = SCRIPT_DIR / "figs_wass_full_by_task"
SUBDIR_TOPO_DRIFT_DELTA = "topological_drift_delta"
SUBDIR_TOPO_DRIFT_BARS = "topological_drift_bars"
SUBDIR_WASS_DATASET_CMP = "wasserstein_distance_dataset_comparison3"
SUBDIR_ELTWISE_MEAN_ABSOLUTE_DELTA = "eltwise_mean_absolute_delta_dataset_comparison2"
SUBDIR_EPOCH_ACCURACIES = "epoch_accuracies"
SUBDIR_EPOCH_ACCURACIES_3 = "epoch_accuracies_3"
# epoch_accuracies_3/: bolder lines/markers than epoch_accuracies/ (both use y 0–100%).
LINEWIDTH_ACC_FULLSCALE = 3.2
MARKERSIZE_ACC_FULLSCALE = 8.5

_NPY_QKVO = re.compile(r"^layer(\d+)_(q|k|v|o)\.npy$", re.IGNORECASE)
PROJECTION_ORDER: tuple[str, ...] = ("q", "k", "v", "o")

# --- Legacy parity: plot_wasserstein_topological_drift.py ---
LABEL_FS = 15
TICK_FS = 13
FIGSIZE_LEGACY = (10, 5)
# Inset legends (snapshot, epochs, drift bars): shared box + BAR_* font sizes.
BAR_LEGEND_FS = 11
BAR_LEGEND_TITLE_FS = 12
DRIFT_BAR_LEGEND_TITLE = r"$\Delta$ For Epochs"
LEGEND_BOX_KWARGS = dict(
    fontsize=BAR_LEGEND_FS,
    title_fontsize=BAR_LEGEND_TITLE_FS,
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
# Epoch accuracy (epoch_accuracies/ and epoch_accuracies_3/): legend lower-right, above x-axis, larger type.
EPOCH_ACC_LEGEND_FS = 13
EPOCH_ACC_LEGEND_TITLE_FS = 14
LEGEND_BOX_KWARGS_EPOCH_ACCURACY = {
    **LEGEND_BOX_KWARGS,
    "fontsize": EPOCH_ACC_LEGEND_FS,
    "title_fontsize": EPOCH_ACC_LEGEND_TITLE_FS,
    "handlelength": 1.35,
    "handletextpad": 0.55,
    "borderpad": 0.5,
    "labelspacing": 0.45,
}
# epoch_accuracies_3/: axis labels, tick numbers, and legend entries/title all 15 pt (``LABEL_FS``).
LEGEND_BOX_KWARGS_EPOCH_ACCURACY_2 = {
    **LEGEND_BOX_KWARGS_EPOCH_ACCURACY,
    "fontsize": LABEL_FS,
    "title_fontsize": LABEL_FS,
}
LINEWIDTH_LEGACY = 2.2
MARKERSIZE_LEGACY = 7
YLABEL_DELTA_I = r"Topological Drift $\Delta_i$"
# Snapshot multi-dataset figures only (explicit $H_0$ wording).
YLABEL_SNAPSHOT_H0 = r"Wasserstein Distance in $H_0$"
YLABEL_ELTWISE_MEAN_ABSOLUTE = r"Mean absolute $\Delta$"

ALL_TASKS: tuple[str, ...] = ("sst2", "imdb", "mmlu", "gsm8k")

# Same dataset colors / legend prefixes as plot_wasserstein_topological_drift.py
DATASET_SNAPSHOT_COLORS = {
    "gsm8k": "#d62728",
    "mmlu": "#2ca02c",
    "imdb": "#1f77b4",
    "sst2": "#ff7f0e",
}
DATASET_LEGEND_LABEL = {
    "gsm8k": "QA:GSM8K",
    "mmlu": "QA:MMLU",
    "imdb": "SA:IMDB",
    "sst2": "SA:SST-2",
}


def _open_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def _save_publication(
    fig: plt.Figure,
    path: Path,
    *,
    tight_layout: bool = True,
    bbox_inches: str | None = "tight",
    bbox_extra_artists: list | None = None,
    pad_inches: float = 0,
) -> None:
    if tight_layout:
        fig.tight_layout(pad=0.0)
    save_kw: dict = {"dpi": 300, "pad_inches": pad_inches}
    if bbox_inches is not None:
        save_kw["bbox_inches"] = bbox_inches
        if bbox_extra_artists:
            save_kw["bbox_extra_artists"] = bbox_extra_artists
    fig.savefig(path, **save_kw)

TASK_TITLE = {
    "sst2": "SST-2",
    "imdb": "IMDb",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
}

FAMILY_TITLE = {
    "llama": "Llama",
    "qwen-base": "Qwen",
    "mistral-7b-v03": "Mistral 7B v0.3",
}

# Epoch line plots: epochs 1–6 — color, linestyle, marker (matches reference figure style)
EPOCH_LINE_STYLES = (
    # ep  color       linestyle   marker
    (1, "#1f77b4", "-", "o"),
    (2, "#ff7f0e", "--", "s"),
    (3, "#2ca02c", ":", "^"),
    (4, "#9467bd", (0, (3, 1, 1, 1)), "v"),
    (5, "#8c564b", "--", "D"),
    (6, "#d62728", "-", "D"),
)

# Stacked drift bars: transitions ending at epoch 3,4,5,6 → labelled epoch 2→3 … 5→6
BAR_TRANSITION_LABELS = ("Epoch 2→3", "Epoch 3→4", "Epoch 4→5", "Epoch 5→6")
BAR_SEGMENT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
BAR_TRANSITION_END_EPOCH = (3, 4, 5, 6)

FAMILY_RESULTS_SUBDIR = {
    "llama": "llama",
    "qwen-base": "qwen",
    "mistral-7b-v03": "mistral",
}

ACC_EVAL_RUN_BY_TASK = {
    "sst2": ("results", "sst2_epoch_eval_py_20260427_134006"),
    "imdb": ("results", "imdb_epoch_eval_py_20260427_145301"),
    "mmlu": ("results", "mmlu_epoch_eval_py_20260427_134320"),
}

# GSM8K epoch curves live under nw-root/eval/split/gsm8k/<family>/json/ (gsm8k_epoch_accuracy_split.py output).
# Llama freeze export uses high-6 only; qwen-base uses wass-high-3.
GSM8K_WASS_JSON_BY_FAMILY = {
    "llama": "wass-high6_epoch_accuracy.json",
    "qwen-base": "wass-high3_epoch_accuracy.json",
    "mistral-7b-v03": "wass-high3_epoch_accuracy.json",
}
GSM8K_WASS_LEGEND_BY_FAMILY = {
    "llama": "Wass-6",
    "qwen-base": "Wass-3",
    "mistral-7b-v03": "Wass-3",
}

def wass_csv_path(
    nw_root: Path, family: str, task: str, *, variant: str = "default"
) -> Path:
    """
    Path to Wasserstein CSV for ``wass_full_by_task``.

    ``variant``:
    - ``default`` — ``wasserstein_results.csv`` (often **v/o** only).
    - ``kqvo`` — newest ``wasserstein_results_kqvo_*.csv`` in the task folder (**q/k/v/o** when present).
    """
    base = (
        nw_root
        / "eval/split/weight_vs_baseline/wass_full_by_task"
        / family
        / task
    )
    if variant == "default":
        return (base / "wasserstein_results.csv").resolve()
    if variant == "kqvo":
        kqvo = sorted(
            base.glob("wasserstein_results_kqvo_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if kqvo:
            return kqvo[0].resolve()
        return (base / "wasserstein_results.csv").resolve()
    raise ValueError(f"unknown Wasserstein CSV variant {variant!r}")


def default_csv(nw_root: Path, family: str, task: str) -> Path:
    return wass_csv_path(nw_root, family, task, variant="default")


def _combined_snapshot_projection_filter(raw: str | None) -> frozenset[str] | None:
    if raw is None or not str(raw).strip():
        return None
    return frozenset(x.strip().lower() for x in str(raw).split(",") if x.strip())


def load_wass_full_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Wasserstein H0" not in df.columns:
        raise SystemExit(f"Missing Wasserstein H0 column in {path}")
    df = df.copy()
    df["epoch_num"] = df["Epoch"].astype(str).str.replace("epoch_", "", regex=False).astype(int)
    df["layer"] = df["File"].str.extract(r"layer(\d+)_", expand=False).astype(int)
    df = df.sort_values(["Projection", "epoch_num", "layer"])
    return df


def available_projections_from_df(df: pd.DataFrame) -> list[str]:
    """Projection codes present in CSV, sorted as q/k/v/o."""
    vals = set(df["Projection"].astype(str).str.lower().unique().tolist())
    return [p for p in PROJECTION_ORDER if p in vals]


def available_projections_from_dfs(dfs_by_task: dict[str, pd.DataFrame]) -> list[str]:
    """Union of projections across task CSVs, sorted as q/k/v/o."""
    vals: set[str] = set()
    for df in dfs_by_task.values():
        vals.update(df["Projection"].astype(str).str.lower().unique().tolist())
    return [p for p in PROJECTION_ORDER if p in vals]


def baseline_numpy_epoch0_dir(nw_root: Path, family: str) -> Path:
    """Shared pretrained snapshot used by ``wass_full_by_task`` (V/O ``layer*_{{v,o}}.npy``)."""
    return (
        nw_root
        / "eval/split/weight_vs_baseline/wass"
        / family
        / "baseline/numpy/epoch_0"
    ).resolve()


def task_numpy_epoch_dir(nw_root: Path, family: str, task: str, epoch: int) -> Path:
    return (
        nw_root
        / "eval/split/weight_vs_baseline/wass_full_by_task"
        / family
        / task
        / "numpy"
        / f"epoch_{epoch}"
    ).resolve()


def eltwise_mean_absolute_scalar_per_tensor(w_ref: np.ndarray, w_new: np.ndarray) -> float:
    """One scalar per attention slice (flattened): ``mean(|new-old|)``."""
    ref = w_ref.astype(np.float64, copy=False).ravel()
    new = w_new.astype(np.float64, copy=False).ravel()
    if ref.shape != new.shape:
        return float("nan")
    diff_abs = np.abs(new - ref)
    return float(np.mean(diff_abs))


def load_eltwise_snapshot_series(
    baseline_dir: Path,
    finetune_dir: Path,
    proj: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-layer scalar drift vs pretrained for each ``layer*_{proj}.npy`` pair.

    Returns ``(layer_indices, values)`` sorted by layer; empty arrays if dirs missing or no pairs.
    """
    xs: list[int] = []
    ys: list[float] = []
    if not baseline_dir.is_dir() or not finetune_dir.is_dir():
        return np.array([], dtype=int), np.array([], dtype=float)

    for f in sorted(finetune_dir.glob(f"layer*_{proj}.npy")):
        m = _NPY_QKVO.match(f.name)
        if not m:
            continue
        li = int(m.group(1))
        bf = baseline_dir / f.name
        if not bf.is_file():
            continue
        w0 = np.load(bf)
        w1 = np.load(f)
        ys.append(eltwise_mean_absolute_scalar_per_tensor(w0, w1))
        xs.append(li)

    if not xs:
        return np.array([], dtype=int), np.array([], dtype=float)

    order = np.argsort(xs)
    xs_arr = np.array(xs, dtype=int)[order]
    ys_arr = np.array(ys, dtype=float)[order]
    return xs_arr, ys_arr


def experiment_title(family: str, task: str) -> str:
    ft = FAMILY_TITLE.get(family, family)
    tt = TASK_TITLE.get(task, task.upper())
    return f"{ft} · {tt}"


def infer_nw_root_from_wass_csv(csv_path: Path) -> Path | None:
    """Infer repo root containing ``eval/`` from a ``wass_full_by_task`` CSV path."""
    for anc in csv_path.resolve().parents:
        if anc.name == "eval":
            return anc.parent
    return None


def epoch_accuracy_json_paths(
    nw_root: Path, family: str, task: str
) -> tuple[Path | None, Path | None, Path | None]:
    """Paths to Full, LoRA, Wasserstein epoch-accuracy JSON files (tuple of ``None`` if unsupported)."""
    if task == "gsm8k":
        jdir = nw_root / "eval/split/gsm8k" / family / "json"
        if not jdir.is_dir():
            return None, None, None
        wass_name = GSM8K_WASS_JSON_BY_FAMILY.get(family)
        if not wass_name:
            return None, None, None
        return (
            jdir / "full_epoch_accuracy.json",
            jdir / "lora_epoch_accuracy.json",
            jdir / wass_name,
        )
    run = ACC_EVAL_RUN_BY_TASK.get(task)
    sub = FAMILY_RESULTS_SUBDIR.get(family)
    if run is None or sub is None:
        return None, None, None
    base = nw_root.joinpath(*run) / sub
    if task in ("imdb", "mmlu"):
        return (
            base / "full_epoch_accuracy.json",
            base / "lora_epoch_accuracy.json",
            base / "wass-high-3_epoch_accuracy.json",
        )
    if task == "sst2":
        return (
            base / f"{sub}-full_epoch_accuracy.json",
            base / f"{sub}-lora_epoch_accuracy.json",
            base / f"{sub}-wass_epoch_accuracy.json",
        )
    return None, None, None


def load_epoch_accuracy_series(path: Path) -> tuple[list[int], list[float]] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    epochs = sorted(int(k) for k in raw.keys())
    vals = [float(raw[str(e)][0]) if raw[str(e)] else float("nan") for e in epochs]
    return epochs, vals


def plot_accuracy_epochs_proj(
    nw_root: Path,
    family: str,
    task: str,
    out_path: Path,
    *,
    linewidth: float = LINEWIDTH_LEGACY,
    markersize: float = MARKERSIZE_LEGACY,
    legend_box_kwargs: dict | None = None,
    axis_label_fs: float | None = None,
    tick_label_fs: float | None = None,
    use_tda_high3_legend: bool = False,
) -> bool:
    """Validation accuracy vs epoch: Full, LoRA, Wass (High‑3 or ‑6 per family for GSM8K).

    Y-axis is fixed **0–100%** (data in [0, 1]). ``linewidth`` / ``markersize`` override stroke
    and markers. ``epoch_accuracies_3/`` passes larger type: ``LEGEND_BOX_KWARGS_EPOCH_ACCURACY_2``,
    ``axis_label_fs`` / ``tick_label_fs`` defaulting to ``LABEL_FS`` (15 pt), and
    ``use_tda_high3_legend=True`` so the wass-high-3 curve reads **TDA-High3** instead of **Wass-3**.
    """
    leg_kw = legend_box_kwargs or LEGEND_BOX_KWARGS_EPOCH_ACCURACY
    lab_fs = axis_label_fs if axis_label_fs is not None else LABEL_FS
    tk_fs = tick_label_fs if tick_label_fs is not None else TICK_FS
    full_p, lora_p, wass_p = epoch_accuracy_json_paths(nw_root, family, task)
    if full_p is None or lora_p is None or wass_p is None:
        print(f"[SKIP] accuracy_epochs {family}/{task}: unsupported combination")
        return False
    wass_label = (
        GSM8K_WASS_LEGEND_BY_FAMILY.get(family, "Wass-3")
        if task == "gsm8k"
        else "Wass-3"
    )
    if use_tda_high3_legend and wass_label == "Wass-3":
        wass_label = "TDA-High3"
    sf = load_epoch_accuracy_series(full_p)
    sl = load_epoch_accuracy_series(lora_p)
    sw = load_epoch_accuracy_series(wass_p)
    if sf is None:
        print(f"[SKIP] accuracy_epochs missing full JSON {full_p}")
        return False
    if sl is None:
        print(f"[SKIP] accuracy_epochs missing LoRA JSON {lora_p}")
        return False
    if sw is None:
        print(f"[SKIP] accuracy_epochs missing wass JSON {wass_p}")
        return False

    ep_f, acc_f = sf
    ep_l, acc_l = sl
    ep_w, acc_w = sw

    fig, ax = plt.subplots(figsize=FIGSIZE_LEGACY)
    ax.plot(
        ep_f,
        acc_f,
        color="#1f77b4",
        linestyle="-",
        marker="o",
        linewidth=linewidth,
        markersize=markersize,
        label="Full",
    )
    ax.plot(
        ep_l,
        acc_l,
        color="#2ca02c",
        linestyle="-.",
        marker="^",
        linewidth=linewidth,
        markersize=markersize,
        label="LoRA",
    )
    ax.plot(
        ep_w,
        acc_w,
        color="#ff7f0e",
        linestyle="--",
        marker="s",
        linewidth=linewidth,
        markersize=markersize,
        label=wass_label,
    )
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("Epoch", fontsize=lab_fs)
    ax.set_ylabel("Accuracy", fontsize=lab_fs)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="both", labelsize=tk_fs)
    ax.legend(
        title="Method",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.07),
        **leg_kw,
    )
    _open_axes(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_publication(fig, out_path)
    plt.close(fig)
    return True


def plot_snapshot_family_multi(
    dfs_by_task: dict[str, pd.DataFrame],
    epoch_num: int,
    family: str,
    proj: str,
    out_path: Path,
    normalize: bool,
    drift4_line_div: dict[tuple[str, str], float] | None = None,
    *,
    drift4_per_panel: bool = False,
) -> None:
    """One axes: all tasks (datasets) for this family/projection at a fixed epoch."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LEGACY)
    entries: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []
    for task in ALL_TASKS:
        df = dfs_by_task.get(task)
        if df is None:
            continue
        sub = df[(df["epoch_num"] == epoch_num) & (df["Projection"] == proj)].sort_values("layer")
        if sub.empty:
            print(f"    [SKIP] snapshot family={family} proj={proj} task={task} epoch={epoch_num}: no rows")
            continue
        y = sub["Wasserstein H0"].values.astype(float)
        x = sub["layer"].values
        lab = DATASET_LEGEND_LABEL.get(task, TASK_TITLE.get(task, task))
        color = DATASET_SNAPSHOT_COLORS.get(task, "#333333")
        entries.append((task, x, y, lab, color))

    if not entries:
        plt.close(fig)
        raise SystemExit(
            f"No snapshot curves for family={family!r} projection={proj!r} epoch={epoch_num}. "
            "Check CSV paths and epoch column."
        )

    curves: list[tuple[np.ndarray, np.ndarray, str, str]] = []

    if drift4_per_panel:
        panel_max = max(float(np.nanmax(np.abs(y))) for _, _, y, _, _ in entries)
        if not np.isfinite(panel_max) or panel_max <= 0.0:
            panel_max = 1e-12
        for _task, x, y, lab, color in entries:
            curves.append((x, y / panel_max, lab, color))
    else:
        for task, x, y, lab, color in entries:
            yy = y.astype(float)
            if drift4_line_div is not None:
                denom = float(drift4_line_div.get((task, proj), 0.0))
                if denom > 0.0:
                    yy = yy / denom
                elif normalize:
                    yy = yy / (np.nanmax(np.abs(yy)) + 1e-12)
            elif normalize:
                yy = yy / (np.nanmax(np.abs(yy)) + 1e-12)
            curves.append((x, yy, lab, color))

    for x, y, lab, color in curves:
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=LINEWIDTH_LEGACY,
            markersize=MARKERSIZE_LEGACY,
            label=lab,
            color=color,
        )

    ax.set_xlabel("Layer", fontsize=LABEL_FS)
    ax.set_ylabel(YLABEL_SNAPSHOT_H0, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(title="Dataset", loc="best", **LEGEND_BOX_KWARGS)
    _open_axes(ax)
    if drift4_line_div is not None or normalize or drift4_per_panel:
        ax.set_ylim(0.0, 1.0)
    _save_publication(fig, out_path)
    plt.close(fig)


def plot_eltwise_snapshot_family_multi(
    curves_by_task: dict[str, tuple[np.ndarray, np.ndarray]],
    family: str,
    proj: str,
    out_path: Path,
    normalize: bool,
    *,
    ylabel: str,
) -> None:
    """One axes: all tasks (datasets) for this family/projection — eltwise drift vs pretrained."""
    fig, ax = plt.subplots(figsize=FIGSIZE_LEGACY)
    plotted = 0
    for task in ALL_TASKS:
        tup = curves_by_task.get(task)
        if tup is None:
            continue
        x, y = tup
        if x.size == 0 or y.size == 0:
            print(f"    [SKIP] eltwise snapshot family={family} proj={proj} task={task}: no numpy pairs")
            continue
        y = y.astype(float)
        if normalize:
            m = np.nanmax(np.abs(y)) + 1e-12
            y = y / m
        lab = DATASET_LEGEND_LABEL.get(task, TASK_TITLE.get(task, task))
        color = DATASET_SNAPSHOT_COLORS.get(task, "#333333")
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=LINEWIDTH_LEGACY,
            markersize=MARKERSIZE_LEGACY,
            label=lab,
            color=color,
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise SystemExit(
            f"No eltwise snapshot curves for family={family!r} projection={proj!r}. "
            "Check baseline numpy epoch_0 and wass_full_by_task numpy epoch dirs."
        )

    ax.set_xlabel("Layer", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    if normalize:
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.legend(title="Dataset", loc="best", **LEGEND_BOX_KWARGS)
    _open_axes(ax)
    _save_publication(fig, out_path)
    plt.close(fig)


def plot_epochs_proj(
    df: pd.DataFrame,
    family: str,
    task: str,
    proj: str,
    out_path: Path,
    max_epoch: int | None,
) -> None:
    emax = df["epoch_num"].max() if max_epoch is None else min(max_epoch, int(df["epoch_num"].max()))
    sub = df[(df["Projection"] == proj) & (df["epoch_num"] <= emax)]

    fig, ax = plt.subplots(figsize=FIGSIZE_LEGACY)

    styles_map = {ep: (c, ls, m) for ep, c, ls, m in EPOCH_LINE_STYLES}
    for ep in range(1, emax + 1):
        dd = sub[sub["epoch_num"] == ep].sort_values("layer")
        if dd.empty:
            continue
        c, ls, m = styles_map.get(ep, ("#333333", "-", "o"))
        ax.plot(
            dd["layer"],
            dd["Wasserstein H0"],
            label=f"Epoch {ep}",
            color=c,
            linestyle=ls,
            marker=m,
            linewidth=LINEWIDTH_LEGACY,
            markersize=MARKERSIZE_LEGACY,
            alpha=0.95,
        )

    ax.set_xlabel("Layer", fontsize=LABEL_FS)
    ax.set_ylabel(YLABEL_DELTA_I, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.legend(title="Epoch", loc="best", **LEGEND_BOX_KWARGS)
    _open_axes(ax)
    _save_publication(fig, out_path)
    plt.close(fig)


def plot_bars_proj(
    df: pd.DataFrame,
    family: str,
    task: str,
    proj: str,
    out_path: Path,
) -> None:
    """Absolute Δ(H₀) between consecutive checkpoints; stack transitions 2→3 … 5→6."""
    d = df[df["Projection"] == proj]
    pivot = d.pivot_table(
        index="epoch_num", columns="layer", values="Wasserstein H0", aggfunc="first"
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    delta = pivot.diff()
    layers = [int(c) for c in pivot.columns]
    bottom = np.zeros(len(layers))

    fig, ax = plt.subplots(figsize=FIGSIZE_LEGACY)

    for lab, color, end_ep in zip(
        BAR_TRANSITION_LABELS, BAR_SEGMENT_COLORS, BAR_TRANSITION_END_EPOCH
    ):
        if end_ep not in delta.index:
            continue
        height = delta.loc[end_ep].fillna(0).values.astype(float)
        ax.bar(
            layers,
            height,
            bottom=bottom,
            label=lab,
            color=color,
            alpha=0.92,
            width=0.92,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom = bottom + height

    ax.set_xlabel("Layer", fontsize=LABEL_FS)
    ax.set_ylabel(r"Drift in $H_0$", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    _open_axes(ax)
    # Lay out axes first; tight_layout after legend can reflow multi-column legends.
    fig.tight_layout(pad=0.0)
    ax.legend(
        title=DRIFT_BAR_LEGEND_TITLE,
        loc="upper right",
        ncols=1,
        alignment="right",
        **LEGEND_BOX_KWARGS,
    )
    _save_publication(fig, out_path, tight_layout=False)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--csv", type=Path, default=None, help="Explicit path to wasserstein_results.csv")
    src.add_argument("--nw-root", type=Path, default=None)

    p.add_argument("--family", choices=("llama", "qwen-base", "mistral-7b-v03"))
    p.add_argument("--task", choices=("sst2", "imdb", "mmlu", "gsm8k"))

    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Root for figure tree: topological_drift_delta/, topological_drift_bars/, "
            "wasserstein_distance_dataset_comparison3/, eltwise_mean_absolute_delta_dataset_comparison2/, "
            "epoch_accuracies/, epoch_accuracies_3/. "
            f"Default: {FIG_BASE_DEFAULT}"
        ),
    )
    p.add_argument("--snapshot-epoch", type=int, default=6)
    p.add_argument(
        "--normalize-snapshot",
        action="store_true",
        help="Normalize each dataset curve by its own max within each projection PDF",
    )
    p.add_argument(
        "--normalize-snapshot-drift4",
        action="store_true",
        help=(
            "Normalize combined snapshot panels to [0, 1] using one divisor per PDF: max |H₀| over all dataset curves "
            "on that axes (so some point touches y=1)."
        ),
    )
    p.add_argument(
        "--normalize-snapshot-drift4-across-models",
        action="store_true",
        help=(
            "Legacy: same H₀ divisors as old topological_drift4 — max over three models per (dataset, projection); "
            "y-axis [0, 1]. Mutually exclusive with --normalize-snapshot and --normalize-snapshot-drift4."
        ),
    )
    p.add_argument(
        "--dataset-comparison-subdir",
        type=str,
        default=SUBDIR_WASS_DATASET_CMP,
        help=(
            "Folder under --out-dir for combined four-dataset Wasserstein snapshots "
            f"(default: {SUBDIR_WASS_DATASET_CMP!r})"
        ),
    )
    p.add_argument(
        "--combined-snapshot-wass-csv",
        choices=("default", "kqvo"),
        default="default",
        help=(
            "Which CSV to load per task for --combined-family-snapshot: "
            "default file or newest wasserstein_results_kqvo_*.csv (full q/k/v/o rows when exported)."
        ),
    )
    p.add_argument(
        "--snapshot-projections",
        type=str,
        default=None,
        help=(
            "For --combined-family-snapshot only: comma-separated projections to plot "
            "(e.g. q,k). Default: all projections present in the CSVs."
        ),
    )
    p.add_argument("--max-epoch", type=int, default=None, help="Cap epochs line plot (default: data max)")
    p.add_argument("--skip-bars", action="store_true", help="Skip stacked‑bar figures")
    p.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip Full / LoRA / Wass-3 accuracy vs epoch plots (non‑GSM8K)",
    )
    p.add_argument(
        "--all-experiments",
        action="store_true",
        help="Run all 12 (family × task) cells under nw-root (requires --nw-root)",
    )
    p.add_argument(
        "--combined-family-snapshot",
        action="store_true",
        help=(
            "Only emit family snapshot PDFs: four datasets on one plot per projection "
            "(requires --nw-root and --family); writes PDFs under <root>/<--dataset-comparison-subdir>/"
        ),
    )
    p.add_argument(
        "--combined-family-snapshot-eltwise",
        action="store_true",
        help=(
            "Same layout as --combined-family-snapshot but **eltwise** mean(|Δ|) vs pretrained "
            f"(numpy under wass/.../baseline and wass_full_by_task/.../numpy); "
            f"writes PDFs under <root>/{SUBDIR_ELTWISE_MEAN_ABSOLUTE_DELTA}/ "
            "(raw mean |Δ| vs baseline numpy)."
        ),
    )
    return p.parse_args()


def _run_family_combined_snapshots(
    nw_root: Path,
    family: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """One snapshot PDF per available projection, each with up to four dataset curves."""
    out_dir.mkdir(parents=True, exist_ok=True)
    drift4_div: dict[tuple[str, str], float] | None = None
    drift4_per_panel = bool(getattr(args, "normalize_snapshot_drift4", False))
    if getattr(args, "normalize_snapshot_drift4_across_models", False):
        from dynamic_topological_drift import line_maxima_h0_per_dataset_projection

        drift4_div = line_maxima_h0_per_dataset_projection()
        drift4_per_panel = False
    csv_variant = getattr(args, "combined_snapshot_wass_csv", "default")
    dfs_by_task: dict[str, pd.DataFrame] = {}
    for task in ALL_TASKS:
        p = wass_csv_path(nw_root, family, task, variant=csv_variant)
        if not p.is_file():
            print(f"[SKIP] combined snapshot {family}: missing {p}")
            continue
        dfs_by_task[task] = load_wass_full_csv(p)
    if not dfs_by_task:
        print(f"[SKIP] combined snapshot {family}: no CSVs found")
        return
    projs = available_projections_from_dfs(dfs_by_task)
    want = _combined_snapshot_projection_filter(getattr(args, "snapshot_projections", None))
    if want is not None:
        projs = [pr for pr in projs if pr in want]
        unknown = want.difference(set(PROJECTION_ORDER))
        if unknown:
            print(
                f"[WARN] combined snapshot {family}: ignoring unknown --snapshot-projections "
                f"entries {sorted(unknown)!r} (valid: {list(PROJECTION_ORDER)})"
            )
    if not projs:
        hint = " (after --snapshot-projections filter)" if want is not None else ""
        print(f"[SKIP] combined snapshot {family}: no projections to plot{hint}")
        return
    for proj in projs:
        outp = out_dir / f"{family}_snapshot_epoch{args.snapshot_epoch}_{proj}.pdf"
        plot_snapshot_family_multi(
            dfs_by_task,
            args.snapshot_epoch,
            family,
            proj,
            outp,
            normalize=args.normalize_snapshot,
            drift4_line_div=drift4_div,
            drift4_per_panel=drift4_per_panel,
        )
        print(f"Saved {outp}")


def _run_family_combined_snapshots_eltwise(
    nw_root: Path,
    family: str,
    fig_base: Path,
    args: argparse.Namespace,
) -> None:
    """Four-dataset overlay PDFs for eltwise mean(|Δ|) (same layout as Wasserstein snapshots)."""
    baseline_dir = baseline_numpy_epoch0_dir(nw_root, family)
    if not baseline_dir.is_dir():
        print(f"[SKIP] eltwise snapshot {family}: missing baseline numpy {baseline_dir}")
        return

    out_dir = fig_base / SUBDIR_ELTWISE_MEAN_ABSOLUTE_DELTA
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = args.snapshot_epoch

    for proj in PROJECTION_ORDER:
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for task in ALL_TASKS:
            ft_dir = task_numpy_epoch_dir(nw_root, family, task, epoch)
            xs, ys = load_eltwise_snapshot_series(baseline_dir, ft_dir, proj)
            if xs.size > 0:
                curves[task] = (xs, ys)
            else:
                print(
                    f"[SKIP] eltwise mean(|Δ|) {family}/{task} proj={proj} epoch_{epoch}: "
                    f"empty or missing pairs under {ft_dir}"
                )

        if not curves:
            print(f"[SKIP] eltwise mean(|Δ|) {family} proj={proj}: no task curves")
            continue

        outp = out_dir / f"{family}_snapshot_epoch{epoch}_{proj}.pdf"
        plot_eltwise_snapshot_family_multi(
            curves,
            family,
            proj,
            outp,
            normalize=args.normalize_snapshot,
            ylabel=YLABEL_ELTWISE_MEAN_ABSOLUTE,
        )
        print(f"Saved {outp}")


def main() -> None:
    args = parse_args()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

    fig_base = Path(args.out_dir).resolve() if args.out_dir else FIG_BASE_DEFAULT

    if args.combined_family_snapshot:
        if args.nw_root is None or args.family is None:
            raise SystemExit("--combined-family-snapshot requires --nw-root and --family")
        snap_norm_modes = (
            args.normalize_snapshot,
            args.normalize_snapshot_drift4,
            getattr(args, "normalize_snapshot_drift4_across_models", False),
        )
        if sum(bool(m) for m in snap_norm_modes) > 1:
            raise SystemExit(
                "Use at most one of --normalize-snapshot, --normalize-snapshot-drift4, "
                "--normalize-snapshot-drift4-across-models"
            )
        if args.combined_snapshot_wass_csv == "kqvo" and (
            args.normalize_snapshot_drift4 or getattr(args, "normalize_snapshot_drift4_across_models", False)
        ):
            raise SystemExit(
                "--combined-snapshot-wass-csv kqvo is meant for per-curve scaling; "
                "use --normalize-snapshot (not drift4 snapshot normalization flags)"
            )
        nw = Path(args.nw_root).resolve()
        snap_dir = fig_base / args.dataset_comparison_subdir
        _run_family_combined_snapshots(nw, args.family, snap_dir, args)
        return

    if args.combined_family_snapshot_eltwise:
        if args.nw_root is None or args.family is None:
            raise SystemExit(
                "--combined-family-snapshot-eltwise requires --nw-root and --family"
            )
        nw = Path(args.nw_root).resolve()
        _run_family_combined_snapshots_eltwise(nw, args.family, fig_base, args)
        return

    if args.all_experiments:
        if args.nw_root is None:
            raise SystemExit("--all-experiments requires --nw-root")
        nw = Path(args.nw_root).resolve()
        families = ("llama", "qwen-base", "mistral-7b-v03")
        tasks = ("sst2", "imdb", "mmlu", "gsm8k")
        for fam in families:
            for task in tasks:
                csv_path = default_csv(nw, fam, task)
                _run_one(csv_path, fam, task, nw, fig_base, args)
            snap_out = fig_base / args.dataset_comparison_subdir
            _run_family_combined_snapshots(nw, fam, snap_out, args)
            _run_family_combined_snapshots_eltwise(nw, fam, fig_base, args)
        print("Finished all 12 experiments.")
        return

    if args.csv is not None:
        csv_path = args.csv.resolve()
        stem = csv_path.parent.name
        grand = csv_path.parent.parent.name
        family = args.family or grand
        task = args.task or stem
    elif args.nw_root and args.family and args.task:
        csv_path = default_csv(Path(args.nw_root), args.family, args.task)
        family, task = args.family, args.task
    else:
        raise SystemExit(
            "Provide (--nw-root --family --task) or --csv, or --all-experiments, "
            "or --combined-family-snapshot / --combined-family-snapshot-eltwise with (--nw-root --family)"
        )

    nw = Path(args.nw_root).resolve() if args.nw_root else infer_nw_root_from_wass_csv(csv_path)
    _run_one(csv_path, family, task, nw, fig_base, args)


def _run_one(
    csv_path: Path,
    family: str,
    task: str,
    nw_root: Path | None,
    fig_base: Path,
    args: argparse.Namespace,
) -> None:
    if not csv_path.is_file():
        print(f"[SKIP] missing {csv_path}")
        return

    stem = f"{family}_{task}"
    epochs_dir = fig_base / SUBDIR_TOPO_DRIFT_DELTA
    bars_dir = fig_base / SUBDIR_TOPO_DRIFT_BARS
    epochs_dir.mkdir(parents=True, exist_ok=True)
    bars_dir.mkdir(parents=True, exist_ok=True)

    df = load_wass_full_csv(csv_path)

    for proj in available_projections_from_df(df):
        ep_path = epochs_dir / f"{stem}_topological_drift_delta_{proj}.pdf"
        plot_epochs_proj(df, family, task, proj, ep_path, args.max_epoch)
        print(f"Saved {ep_path}")

        if not args.skip_bars:
            bar_path = bars_dir / f"{stem}_topological_drift_bars_{proj}.pdf"
            plot_bars_proj(df, family, task, proj, bar_path)
            print(f"Saved {bar_path}")

    if not args.skip_accuracy and nw_root is not None:
        acc_dir = fig_base / SUBDIR_EPOCH_ACCURACIES
        acc_dir.mkdir(parents=True, exist_ok=True)
        acc_path = acc_dir / f"{stem}_epoch_accuracies.pdf"
        if plot_accuracy_epochs_proj(nw_root, family, task, acc_path):
            print(f"Saved {acc_path}")
        acc2_dir = fig_base / SUBDIR_EPOCH_ACCURACIES_3
        acc2_dir.mkdir(parents=True, exist_ok=True)
        acc2_path = acc2_dir / f"{stem}_epoch_accuracies.pdf"
        if plot_accuracy_epochs_proj(
            nw_root,
            family,
            task,
            acc2_path,
            linewidth=LINEWIDTH_ACC_FULLSCALE,
            markersize=MARKERSIZE_ACC_FULLSCALE,
            legend_box_kwargs=LEGEND_BOX_KWARGS_EPOCH_ACCURACY_2,
            axis_label_fs=LABEL_FS,
            tick_label_fs=LABEL_FS,
            use_tda_high3_legend=True,
        ):
            print(f"Saved {acc2_path}")
    elif not args.skip_accuracy and nw_root is None:
        print(f"[SKIP] accuracy_epochs {stem}: set --nw-root to locate epoch-accuracy JSON files")


if __name__ == "__main__":
    main()
