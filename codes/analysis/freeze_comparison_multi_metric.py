import os
import re
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from safetensors.torch import load_file as safetensors_load_file

EPS = 1e-6

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

PROJ_PATTERNS = {
    "k": ".k_proj.",
    "q": ".q_proj.",
    "v": ".v_proj.",
    "o": ".o_proj.",
}

OUT_DIR = "/home/vepaul/results"
HF_MODEL_ID = "meta-llama/Llama-3.1-8B"

CHKPT_BASE = "/home/kadir/topo/numpy_weights/exploration-finetuning/checkpoints"

MODEL_DIRS = {
    "llama": {
        "norm": os.path.join(CHKPT_BASE, "llama", "norm-freeze"),
        "wass": os.path.join(CHKPT_BASE, "llama", "wass-freeze"),
    },
    "qwen": {
        "norm": os.path.join(CHKPT_BASE, "qwen", "norm-freeze"),
        "wass": os.path.join(CHKPT_BASE, "qwen", "wass-freeze"),
    },
    "qwen-base": {
        "norm": os.path.join(CHKPT_BASE, "qwen-base", "norm-freeze"),
        "wass": os.path.join(CHKPT_BASE, "qwen-base", "wass-freeze"),
    },
}

PRETRAINED_DIRS = {
    "llama":     "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/llama31-8b/",
    "qwen":      "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/qwen3-8b/",
    "qwen-base": "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/qwen3-8b-base/",
}

FULL_FT_DIRS = {
    "llama":     "/data/cuneyt-topo/numpy_weights/exploration-finetuning/checkpoints/llama/full/",
    "qwen":      "/data/cuneyt-topo/numpy_weights/exploration-finetuning/checkpoints/qwen/full/gsm8k-qwen-full-finetuned/",
    "qwen-base": "/data/cuneyt-topo/numpy_weights/exploration-finetuning/checkpoints/qwen-base/full/run2",
}

VARIANT_NAMES = {
    "llama": {
        "norm": ["high-3","high-6","high-9","low-3","low-6","low-9","low-15"],
        "wass": ["high-6-frozen","high-9-frozen","low-3-frozen","low-6-frozen","low-9-frozen","low-15-frozen"],
    },
    "qwen": {
        "norm": ["high-3","high-6","high-9","low-3","low-6","low-9","low-15"],
        "wass": ["high-3","high-6","high-9","low-3","low-6","low-9","low-15"],
    },
    "qwen-base": {
        "norm": ["high-3","high-6","high-9","low-3","low-6","low-9","low-15"],
        "wass": ["high-3","high-6","high-9","low-3","low-6","low-9","low-15"],
    },
}

RUN_PREFERENCE = ["run3", "run2", "run1"]

METRICS = [
    ("frac_changed",    "Fraction of weights changed"),
    ("mean_abs_change", "Mean |Δw|"),
    ("mean_pct_change", "Mean |Δw| / (|w| + ε)  [%]"),
    ("cosine_dist",     "Cosine distance (1 - cos sim)"),
    ("frobenius_norm",  "Frobenius norm of Δw"),
]

_LOW_COLORS  = {"3": "#a8c8e8", "6": "#5b9bd5", "9": "#1f6fb2", "15": "#0b3d6b"}
_HIGH_COLORS = {"3": "#f5a8a8", "6": "#e05c5c", "9": "#c0392b", "15": "#7b1a10"}

# ------------------------------------------------------------------
# EXPLICIT expected frozen layers from your qwen-base table
# Only V/O are specified because that is what your sheet gives.
# K/Q still fall back to inference if ever needed.
# ------------------------------------------------------------------
EXPECTED_FROZEN_LAYERS = {
    "qwen-base": {
        "wass": {
            "low-3": {
                "v": [25, 10, 4],
                "o": [6, 34, 29],
            },
            "low-6": {
                "v": [25, 10, 4, 6, 12, 17],
                "o": [6, 34, 29, 25, 28, 4],
            },
            "low-9": {
                "v": [25, 10, 4, 6, 12, 17, 8, 15, 7],
                "o": [6, 34, 29, 25, 28, 4, 30, 26, 31],
            },
            "low-15": {
                "v": [25, 10, 4, 6, 12, 17, 8, 15, 7, 19, 26, 34, 24, 28, 30],
                "o": [6, 34, 29, 25, 28, 4, 30, 26, 31, 32, 9, 35, 27, 24, 5],
            },
            "high-3": {
                "v": [1, 5, 0],
                "o": [18, 17, 15],
            },
            "high-6": {
                "v": [31, 29, 1, 5, 0, 13],
                "o": [14, 19, 17, 18, 15, 16],
            },
            "high-9": {
                "v": [32, 3, 21, 13, 31, 29, 1, 5, 0],
                "o": [20, 13, 22, 16, 14, 19, 17, 18, 15],
            },
            "high-15": {
                "v": [14, 20, 11, 16, 33, 35, 32, 3, 21, 13, 31, 29, 1, 5, 0],
                "o": [1, 12, 23, 2, 11, 0, 20, 13, 22, 16, 14, 19, 17, 18, 15],
            },
        },
        "norm": {
            "low-3": {
                "v": [3, 12, 11],
                "o": [10, 8, 12],
            },
            "low-6": {
                "v": [3, 12, 11, 10, 9, 13],
                "o": [10, 8, 12, 9, 11, 7],
            },
            "low-9": {
                "v": [3, 12, 11, 10, 9, 13, 7, 14, 15],
                "o": [10, 8, 12, 9, 11, 7, 4, 6, 3],
            },
            "low-15": {
                "v": [8, 12, 11, 10, 9, 13, 7, 14, 15, 17, 16, 18, 6, 4, 3],
                "o": [10, 8, 12, 9, 11, 7, 4, 6, 3, 2, 14, 5, 17, 13, 16],
            },
            "high-3": {
                "v": [24, 29, 0],
                "o": [21, 35, 0],
            },
            "high-6": {
                "v": [27, 23, 1, 24, 29, 0],
                "o": [29, 20, 28, 21, 35, 0],
            },
            "high-9": {
                "v": [26, 22, 35, 27, 23, 1, 24, 29, 0],
                "o": [27, 30, 34, 29, 20, 28, 21, 35, 0],
            },
            "high-15": {
                "v": [31, 28, 32, 21, 2, 30, 26, 22, 35, 27, 23, 1, 24, 29, 0],
                "o": [15, 24, 25, 26, 32, 22, 27, 30, 34, 29, 20, 28, 21, 35, 0],
            },
        },
    }
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def layer_index(param_name):
    m = LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def projection_type(param_name):
    for proj, pattern in PROJ_PATTERNS.items():
        if pattern in param_name:
            return proj
    return None


def load_weight_map(folder):
    index_path = os.path.join(folder, "model.safetensors.index.json")
    single_path = os.path.join(folder, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)
        wm = idx.get("weight_map")
        if not wm:
            raise RuntimeError(f"No weight_map in {index_path}")
        return wm
    if os.path.isfile(single_path):
        sd = safetensors_load_file(single_path, device="cpu")
        return {k: "model.safetensors" for k in sd.keys()}
    raise RuntimeError(f"No safetensors weights found in: {folder}")


class ShardCache:
    def __init__(self, folder, max_cached=4):
        self.folder, self.max_cached = folder, max_cached
        self.cache, self.order = {}, []

    def get(self, shard):
        if shard in self.cache:
            self.order.remove(shard)
            self.order.append(shard)
            return self.cache[shard]
        path = os.path.join(self.folder, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        sd = safetensors_load_file(path, device="cpu")
        self.cache[shard] = sd
        self.order.append(shard)
        while len(self.order) > self.max_cached:
            del self.cache[self.order.pop(0)]
        return sd


def _has_weights(d):
    return (
        os.path.isfile(os.path.join(d, "model.safetensors.index.json")) or
        os.path.isfile(os.path.join(d, "model.safetensors"))
    )


def _resolve_weights_dir(d):
    if not os.path.isdir(d):
        return None
    if _has_weights(d):
        return d
    ckpt_dirs = [
        (int(m.group(1)), entry)
        for entry in os.listdir(d)
        for m in [re.match(r"checkpoint-(\d+)$", entry)]
        if m and os.path.isdir(os.path.join(d, entry))
    ]
    if ckpt_dirs:
        _, latest = max(ckpt_dirs)
        ckpt_path = os.path.join(d, latest)
        if _has_weights(ckpt_path):
            return ckpt_path
    return None


def _resolve_exp_dir(model, criterion, variant):
    base = MODEL_DIRS[model][criterion]
    variant_dir = os.path.join(base, variant)
    if not os.path.isdir(variant_dir):
        return None
    for run_id in RUN_PREFERENCE:
        run_dir = os.path.join(variant_dir, run_id)
        result = _resolve_weights_dir(run_dir)
        if result:
            return result
    return None


def _short_name(variant, criterion):
    base = variant.replace("-frozen", "")
    base = re.sub(r"-(\d+)$", r"\1", base)
    return f"{criterion}-{base}"


def _plot_style(exp_name):
    if "full-ft" in exp_name:
        return ("black", "-", 2.5, 0.5)
    m = re.search(r"(\d+)$", exp_name)
    n = m.group(1) if m else "6"
    ls = "-"
    if "low" in exp_name:
        return (_LOW_COLORS.get(n, "#1f6fb2"), ls, 1.5, 1.0)
    if "high" in exp_name:
        return (_HIGH_COLORS.get(n, "#c0392b"), ls, 1.5, 1.0)
    return ("gray", "-", 1.0, 0.8)


def _n_from_label(label):
    m = re.search(r"(\d+)$", label)
    return int(m.group(1)) if m else 0


def _infer_frozen_layers(exp_metrics, proj):
    return sorted(
        li for li, v in exp_metrics.get("frac_changed", {}).get(proj, {}).items()
        if v == 0.0
    )


def _get_expected_frozen_layers(model, criterion, variant, proj):
    return (
        EXPECTED_FROZEN_LAYERS
        .get(model, {})
        .get(criterion, {})
        .get(variant, {})
        .get(proj, None)
    )


# ── Metric computation ────────────────────────────────────────────────────────

def compute_all_metrics(dir_a, dir_b):
    map_a = load_weight_map(dir_a)
    map_b = load_weight_map(dir_b)
    common_keys = sorted(set(map_a.keys()).intersection(map_b.keys()))
    if not common_keys:
        return {}

    cache_a = ShardCache(dir_a)
    cache_b = ShardCache(dir_b)

    acc = {}

    for k in common_keys:
        li   = layer_index(k)
        proj = projection_type(k)
        if li is None or proj is None:
            continue
        try:
            t0 = cache_a.get(map_a[k])[k].to(torch.float32).reshape(-1)
            t1 = cache_b.get(map_b[k])[k].to(torch.float32).reshape(-1)
        except (FileNotFoundError, KeyError):
            continue
        if t0.shape != t1.shape:
            continue

        delta = t1 - t0
        abs_delta = delta.abs()

        key = (li, proj)
        if key not in acc:
            acc[key] = {
                "n":             0,
                "n_changed":     0,
                "sum_abs":       0.0,
                "sum_pct":       0.0,
                "dot_t0_t1":     0.0,
                "norm_t0_sq":    0.0,
                "norm_t1_sq":    0.0,
                "sum_sq_delta":  0.0,
            }
        a = acc[key]
        n = t0.numel()

        a["n_changed"]    += (abs_delta > 0).sum().item()
        a["sum_abs"]      += abs_delta.sum().item()
        a["sum_pct"]      += (abs_delta / (t0.abs() + EPS)).sum().item()
        a["dot_t0_t1"]    += (t0 * t1).sum().item()
        a["norm_t0_sq"]   += (t0 * t0).sum().item()
        a["norm_t1_sq"]   += (t1 * t1).sum().item()
        a["sum_sq_delta"] += (delta * delta).sum().item()
        a["n"] += n

    metrics = {mk: {p: {} for p in "kqvo"} for mk, _ in METRICS}

    for (li, proj), a in acc.items():
        n = a["n"]
        if n == 0:
            continue

        metrics["frac_changed"][proj][li] = a["n_changed"] / n
        metrics["mean_abs_change"][proj][li] = a["sum_abs"] / n
        metrics["mean_pct_change"][proj][li] = (a["sum_pct"] / n) * 100.0

        denom = (a["norm_t0_sq"] ** 0.5) * (a["norm_t1_sq"] ** 0.5) + EPS
        metrics["cosine_dist"][proj][li] = 1.0 - a["dot_t0_t1"] / denom
        metrics["frobenius_norm"][proj][li] = a["sum_sq_delta"] ** 0.5

    return metrics


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_unfrozen_scatter(ax, layers, vals, frozen_set, color, marker, label, ls="-", lw=1.5, alpha=1.0):
    ux = [li for li in layers if li not in frozen_set]
    uy = [vals[layers.index(li)] for li in ux]
    if not ux:
        return

    ax.scatter(ux, uy, color=color, s=12, zorder=4, alpha=alpha)

    seg_x, seg_y = [], []
    first_segment = True
    for li, v in zip(layers, vals):
        if li in frozen_set:
            if seg_x:
                ax.plot(seg_x, seg_y, color=color, linestyle=ls,
                        linewidth=lw, alpha=alpha, zorder=3,
                        label=label if first_segment else "_nolegend_")
                first_segment = False
                seg_x, seg_y = [], []
        else:
            seg_x.append(li)
            seg_y.append(v)
    if seg_x:
        ax.plot(seg_x, seg_y, color=color, linestyle=ls,
                linewidth=lw, alpha=alpha, zorder=3,
                label=label if first_segment else "_nolegend_")


# ── Plot A ────────────────────────────────────────────────────────────────────

def save_split_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model):
    PANELS = [
        ("norm", "low",  "Norm - Low layer freezing"),
        ("norm", "high", "Norm - High layer freezing"),
        ("wass", "low",  "Wasserstein - Low layer freezing"),
        ("wass", "high", "Wasserstein - High layer freezing"),
    ]
    all_x = [li for layers, _ in all_series.values() for li in layers]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)

    fig, axes_grid = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
    axes = [axes_grid[0,0], axes_grid[0,1], axes_grid[1,0], axes_grid[1,1]]

    is_kq = proj in ("k", "q")

    for ax, (criterion, direction, title) in zip(axes, PANELS):
        panel_labels = sorted(
            [l for l in all_series if criterion in l and direction in l and "full-ft" not in l],
            key=lambda l: _n_from_label(l)
        )

        if "full-ft" in all_series:
            ft_layers, ft_vals = all_series["full-ft"]
            ax.plot(ft_layers, ft_vals, color="black", linewidth=2.0,
                    alpha=0.3, label="full-ft", zorder=1)

        if not is_kq:
            for label in panel_labels:
                layers, vals = all_series[label]
                color, ls, lw, alpha = _plot_style(label)
                frozen_set = set(all_frozen_layers.get(label, []))
                m = re.search(r"(\d+)", label)
                short = f"N={m.group(1)}" if m else label
                _plot_unfrozen_scatter(ax, layers, vals, frozen_set,
                                       color, "o", short, ls=ls, lw=lw, alpha=alpha)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes_grid[0,0].set_ylabel(metric_title, fontsize=10)
    axes_grid[1,0].set_ylabel(metric_title, fontsize=10)

    note = f"{proj.upper()}: freeze variants had 0 weight change" if is_kq else \
           "blue=low layers, red=high layers | scatter=unfrozen layers only"
    fig.suptitle(f"[{model}] {proj.upper()} — {metric_title}\n{note}", fontsize=12)
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "split"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"split__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot B ────────────────────────────────────────────────────────────────────

def save_collapsed_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model):
    all_x = [li for layers, _ in all_series.values() for li in layers]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)

    fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    is_kq = proj in ("k", "q")

    for ax, direction, panel_title in [
        (ax_low,  "low",  "Low layer freezing"),
        (ax_high, "high", "High layer freezing"),
    ]:
        panel_labels = sorted(
            [l for l in all_series if direction in l and "full-ft" not in l],
            key=lambda l: _n_from_label(l)
        )

        if "full-ft" in all_series:
            ft_layers, ft_vals = all_series["full-ft"]
            ax.plot(ft_layers, ft_vals, color="black", linewidth=2.0,
                    alpha=0.3, label="full-ft", zorder=1)

        if not is_kq:
            for label in panel_labels:
                layers, vals = all_series[label]
                color, ls, lw, alpha = _plot_style(label)
                frozen_set = set(all_frozen_layers.get(label, []))
                m = re.search(r"(\d+)", label)
                crit = "norm" if "norm" in label else "wass"
                short = f"{crit} N={m.group(1)}" if m else label
                _plot_unfrozen_scatter(ax, layers, vals, frozen_set,
                                       color, "o", short, ls=ls, lw=lw, alpha=alpha)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_xlabel("Layer index", fontsize=11)
        ax.set_title(panel_title, fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    ax_low.set_ylabel(metric_title, fontsize=11)
    fig.suptitle(f"[{model}] {proj.upper()} — {metric_title}", fontsize=12)
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "collapsed"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"collapsed__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot C ────────────────────────────────────────────────────────────────────

def save_summary_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model):
    is_kq = proj in ("k", "q")
    if is_kq:
        return

    records = []
    for label, (layers, vals) in all_series.items():
        if "full-ft" in label:
            continue
        frozen_set = set(all_frozen_layers.get(label, []))
        unfrozen_vals = [v for li, v in zip(layers, vals) if li not in frozen_set]
        if not unfrozen_vals:
            continue
        mean_val  = sum(unfrozen_vals) / len(unfrozen_vals)
        direction = "low"  if "low"  in label else "high"
        criterion = "norm" if "norm" in label else "wass"
        n = _n_from_label(label)
        records.append((criterion, direction, n, mean_val))

    if not records:
        return

    fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, direction, color, panel_title in [
        (ax_low,  "low",  "#1f77b4", "Low layer freezing"),
        (ax_high, "high", "#c0392b", "High layer freezing"),
    ]:
        ns = sorted(set(r[2] for r in records if r[1] == direction))
        if not ns:
            ax.set_title(panel_title, fontsize=12)
            continue
        x = np.arange(len(ns))
        width = 0.35

        norm_vals = [next((r[3] for r in records if r[0]=="norm" and r[1]==direction and r[2]==n), 0.0) for n in ns]
        wass_vals = [next((r[3] for r in records if r[0]=="wass" and r[1]==direction and r[2]==n), 0.0) for n in ns]

        ax.bar(x - width/2, norm_vals, width, label="Norm",
               color=color, alpha=0.85, edgecolor="white")
        ax.bar(x + width/2, wass_vals, width, label="Wasserstein",
               color=color, alpha=0.45, hatch="//", edgecolor=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in ns], fontsize=10)
        ax.set_xlabel("Frozen layers (N)", fontsize=11)
        ax.set_title(panel_title, fontsize=12)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9)

    ax_low.set_ylabel(f"Mean {metric_title} (unfrozen layers only)", fontsize=10)
    fig.suptitle(
        f"[{model}] {proj.upper()} — Mean {metric_title} per experiment",
        fontsize=12
    )
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "summary"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"summary__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot D ────────────────────────────────────────────────────────────────────

def save_delta_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model):
    is_kq = proj in ("k", "q")
    if is_kq or "full-ft" not in all_series:
        return

    ft_layers, ft_vals = all_series["full-ft"]
    ft_lookup = dict(zip(ft_layers, ft_vals))
    all_x = [li for layers, _ in all_series.values() for li in layers]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)

    PANELS = [
        ("norm", "low",  "Norm — Low layer freezing"),
        ("norm", "high", "Norm — High layer freezing"),
        ("wass", "low",  "Wasserstein — Low layer freezing"),
        ("wass", "high", "Wasserstein — High layer freezing"),
    ]

    fig, axes_grid = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
    axes = [axes_grid[0,0], axes_grid[0,1], axes_grid[1,0], axes_grid[1,1]]

    for ax, (criterion, direction, panel_title) in zip(axes, PANELS):
        panel_labels = sorted(
            [l for l in all_series if criterion in l and direction in l and "full-ft" not in l],
            key=lambda l: _n_from_label(l)
        )

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)

        for label in panel_labels:
            layers, vals = all_series[label]
            color, ls, lw, alpha = _plot_style(label)
            frozen_set = set(all_frozen_layers.get(label, []))
            m = re.search(r"(\d+)", label)
            short = f"N={m.group(1)}" if m else label

            delta_x = [li for li in layers if li not in frozen_set and li in ft_lookup]
            delta_y = [vals[layers.index(li)] - ft_lookup[li] for li in delta_x]

            if delta_x:
                ax.scatter(delta_x, delta_y, color=color, s=12, zorder=4, alpha=alpha)
                ax.plot(delta_x, delta_y, color=color, linestyle=ls,
                        linewidth=lw, alpha=alpha, zorder=3, label=short)

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_title(panel_title, fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes_grid[0,0].set_ylabel(f"Δ {metric_title} (variant − full-ft)", fontsize=10)
    axes_grid[1,0].set_ylabel(f"Δ {metric_title} (variant − full-ft)", fontsize=10)
    fig.suptitle(
        f"[{model}] {proj.upper()} — {metric_title} deviation from full fine-tuning",
        fontsize=12
    )
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "delta"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"delta__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot E: FIXED HEATMAP ─────────────────────────────────────────────────────

def save_heatmap_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model):
    """
    Fixed behavior:
      - gray   = frozen
      - pink   = missing from metric output / not matched in checkpoint
      - viridis = observed metric value
    """
    is_kq = proj in ("k", "q")
    if is_kq:
        return

    variant_keys = [l for l in all_series if "full-ft" not in l]

    # Include both observed layers and explicitly expected frozen layers
    all_x = sorted(
        set(
            li for label in variant_keys
            for li in all_series[label][0]
        ).union(
            li for label in variant_keys
            for li in all_frozen_layers.get(label, [])
        )
    )

    if not all_x:
        return

    def _draw_heatmap(ax, labels, all_series, all_frozen_layers, all_x, metric_title, title):
        n_layers = len(all_x)
        layer_to_col = {li: i for i, li in enumerate(all_x)}

        observed = np.full((len(labels), n_layers), np.nan)
        frozen_overlay = np.full((len(labels), n_layers), np.nan)
        missing_overlay = np.full((len(labels), n_layers), np.nan)

        for row_i, label in enumerate(labels):
            layers, vals = all_series[label]
            val_by_layer = dict(zip(layers, vals))
            present_layers = set(layers)
            frozen_set = set(all_frozen_layers.get(label, []))

            for li in all_x:
                col_i = layer_to_col[li]
                if li in frozen_set:
                    frozen_overlay[row_i, col_i] = 1.0
                elif li in present_layers:
                    observed[row_i, col_i] = val_by_layer[li]
                else:
                    missing_overlay[row_i, col_i] = 1.0

        extent = [-0.5, n_layers - 0.5, len(labels) - 0.5, -0.5]

        missing_masked = np.ma.masked_invalid(missing_overlay)
        frozen_masked = np.ma.masked_invalid(frozen_overlay)
        observed_masked = np.ma.masked_invalid(observed)

        ax.imshow(
            missing_masked,
            aspect="auto",
            cmap=mcolors.ListedColormap(["#f6d6d6"]),
            extent=extent,
            interpolation="none",
        )

        ax.imshow(
            frozen_masked,
            aspect="auto",
            cmap=mcolors.ListedColormap(["#dddddd"]),
            extent=extent,
            interpolation="none",
        )

        cmap = plt.cm.viridis.copy()
        cmap.set_bad((1, 1, 1, 0))
        im = ax.imshow(
            observed_masked,
            aspect="auto",
            cmap=cmap,
            extent=extent,
            interpolation="none",
        )

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels(all_x, fontsize=7, rotation=90)
        ax.set_yticks(range(len(labels)))
        short_labels = [re.sub(r"(norm|wass)-", "", l) for l in labels]
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Layer index", fontsize=9)
        ax.set_title(title, fontsize=10)

        legend_handles = [
            Patch(facecolor="#dddddd", edgecolor="none", label="Frozen"),
            Patch(facecolor="#f6d6d6", edgecolor="none", label="Missing / not matched"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

        plt.colorbar(im, ax=ax, shrink=0.8, label=metric_title)

    PANELS = [
        ("norm", "low",  "Norm — Low freezing"),
        ("norm", "high", "Norm — High freezing"),
        ("wass", "low",  "Wasserstein — Low freezing"),
        ("wass", "high", "Wasserstein — High freezing"),
    ]

    fig, axes_grid = plt.subplots(2, 2, figsize=(22, 10))
    axes = [axes_grid[0,0], axes_grid[0,1], axes_grid[1,0], axes_grid[1,1]]

    for ax, (criterion, direction, title) in zip(axes, PANELS):
        labels = sorted(
            [l for l in all_series if criterion in l and direction in l and "full-ft" not in l],
            key=lambda l: _n_from_label(l)
        )
        if not labels:
            ax.set_visible(False)
            continue
        _draw_heatmap(ax, labels, all_series, all_frozen_layers, all_x, metric_title, title)

    fig.suptitle(
        f"[{model}] {proj.upper()} — {metric_title} heatmap\n"
        "gray = frozen | pink = missing / not matched | rows = N value | columns = layer index",
        fontsize=12
    )
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "heatmap"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"heatmap__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

VENV_PYTHON = "/home/kadir/topo/topo-env/bin/python"

def main():
    ap = argparse.ArgumentParser(
        description="Compare pretrained baseline vs freeze experiments across multiple metrics."
    )
    ap.add_argument("--model", choices=["llama", "qwen", "qwen-base"], default="llama")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pretrained_dir", default=None)
    ap.add_argument("--full_ft_dir", default=None)
    ap.add_argument("--metrics", nargs="+",
                    choices=[mk for mk, _ in METRICS] + ["all"],
                    default=["all"],
                    help="Which metrics to compute (default: all)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(OUT_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)

    pretrained_dir = args.pretrained_dir or PRETRAINED_DIRS.get(args.model)
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        raise RuntimeError(f"Pretrained dir not found: {pretrained_dir}")

    metrics_to_run = METRICS if "all" in args.metrics else \
                     [(mk, mt) for mk, mt in METRICS if mk in args.metrics]

    print(f"Model    : {args.model}")
    print(f"Baseline : {pretrained_dir}")
    print(f"Metrics  : {[mk for mk, _ in metrics_to_run]}")
    print(f"Output   : {out_dir}\n")

    all_combined   = {mk: {p: {} for p in "kqvo"} for mk, _ in metrics_to_run}
    frozen_by_exp  = {p: {} for p in "kqvo"}

    full_ft_dir = args.full_ft_dir or FULL_FT_DIRS.get(args.model, "")
    full_ft_weights = _resolve_weights_dir(full_ft_dir) if full_ft_dir else None
    if full_ft_weights:
        print(f"Loading full-ft reference: {full_ft_weights}")
        ft_metrics = compute_all_metrics(pretrained_dir, full_ft_weights)
        for mk, _ in metrics_to_run:
            for proj in "kqvo":
                if ft_metrics.get(mk, {}).get(proj):
                    layers = sorted(ft_metrics[mk][proj].keys())
                    vals   = [ft_metrics[mk][proj][li] for li in layers]
                    all_combined[mk][proj]["full-ft"] = (layers, vals)
        for proj in "kqvo":
            frozen_by_exp[proj]["full-ft"] = []
    else:
        print(f"  [skip] full-ft — not found or no weights ({full_ft_dir})")
        full_ft_weights = None

    for criterion, variants in VARIANT_NAMES[args.model].items():
        for variant in variants:
            exp_dir = _resolve_exp_dir(args.model, criterion, variant)
            if not exp_dir:
                print(f"  [skip] {criterion}/{variant} — not found")
                continue

            exp_label = _short_name(variant, criterion)
            print(f"Computing: pretrained vs {exp_label}  ({exp_dir})")
            exp_metrics = compute_all_metrics(pretrained_dir, exp_dir)
            if not exp_metrics:
                print("  [WARNING] No data — skipping.")
                continue

            for mk, _ in metrics_to_run:
                for proj in "kqvo":
                    if exp_metrics.get(mk, {}).get(proj):
                        layers = sorted(exp_metrics[mk][proj].keys())
                        vals   = [exp_metrics[mk][proj][li] for li in layers]
                        all_combined[mk][proj][exp_label] = (layers, vals)

            for proj in "kqvo":
                expected = _get_expected_frozen_layers(args.model, criterion, variant, proj)
                inferred = _infer_frozen_layers(exp_metrics, proj)

                if expected is not None:
                    frozen_by_exp[proj][exp_label] = sorted(expected)

                    present_layers = sorted(exp_metrics.get("frac_changed", {}).get(proj, {}).keys())
                    missing_layers = [li for li in frozen_by_exp[proj][exp_label] if li not in present_layers]
                    if missing_layers:
                        print(f"  [coverage] {exp_label} {proj.upper()} expected frozen but absent from metric output: {missing_layers}")

                    if inferred != sorted(expected):
                        print(f"  [freeze-map] {exp_label} {proj.upper()} inferred={inferred} expected={sorted(expected)}")
                else:
                    frozen_by_exp[proj][exp_label] = inferred

                # Extra visibility for your qwen-base tail-layer issue
                if args.model == "qwen-base" and proj in ("v", "o"):
                    present_layers = set(exp_metrics.get("frac_changed", {}).get(proj, {}).keys())
                    tail = [32, 33, 34, 35]
                    tail_missing = [li for li in tail if li not in present_layers]
                    if tail_missing:
                        print(f"  [tail-missing] {exp_label} {proj.upper()} missing layers: {tail_missing}")

    print(f"\n{'='*55}")
    print("Saving plots ...")
    for mk, mt in metrics_to_run:
        for proj in "kqvo":
            series = all_combined[mk][proj]
            if not series:
                continue
            frozen = frozen_by_exp[proj]
            save_split_plot(mk, mt, proj, series, frozen, out_dir, args.model)
            save_collapsed_plot(mk, mt, proj, series, frozen, out_dir, args.model)
            save_summary_plot(mk, mt, proj, series, frozen, out_dir, args.model)
            save_delta_plot(mk, mt, proj, series, frozen, out_dir, args.model)
            save_heatmap_plot(mk, mt, proj, series, frozen, out_dir, args.model)

    print("\nDone.")


if __name__ == "__main__":
    main()