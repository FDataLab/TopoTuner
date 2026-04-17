import os
import re
import json
import argparse
import csv
from collections import defaultdict
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
# exploration-finetuning repo root (parent of checkpoints/)
REPO_ROOT = os.path.dirname(CHKPT_BASE)

MODEL_DIRS = {
    "llama": {
        "norm": os.path.join(CHKPT_BASE, "llama", "norm-freeze"),
        "wass": os.path.join(CHKPT_BASE, "llama", "wass-freeze"),
    },
    "qwen-base": {
        "norm": os.path.join(CHKPT_BASE, "qwen-base", "norm-freeze"),
        "wass": os.path.join(CHKPT_BASE, "qwen-base", "wass-freeze"),
    },
}

PRETRAINED_DIRS = {
    "llama":     "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/llama31-8b/",
    "qwen-base": "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/qwen3-8b-base/",
}

# Multi-seed roots: pick first run with weights using RUN_PREFERENCE_BY_MODEL.
FULL_MULTI_RUN_ROOT = {
    "llama":     os.path.join(CHKPT_BASE, "llama", "full"),
    "qwen-base": os.path.join(CHKPT_BASE, "qwen-base", "full"),
}

# Rows in heatmap that are not norm/wass freeze sweeps (no expected-freeze overlay).
_HEATMAP_BASELINE_ROWS = frozenset({"full"})

_VARIANT_KS = ["3", "6", "9", "12", "15"]
VARIANT_NAMES = {
    "llama": {
        "norm": [f"high-{k}" for k in _VARIANT_KS] + [f"low-{k}" for k in _VARIANT_KS],
        "wass": [f"high-{k}-frozen" for k in _VARIANT_KS] + [f"low-{k}-frozen" for k in _VARIANT_KS],
    },
    "qwen-base": {
        "norm": [f"high-{k}" for k in _VARIANT_KS] + [f"low-{k}" for k in _VARIANT_KS],
        "wass": [f"high-{k}" for k in _VARIANT_KS] + [f"low-{k}" for k in _VARIANT_KS],
    },
}

# Per-model checkpoint run: only this run is used (no run2/run3 fallback for llama, etc.).
RUN_PREFERENCE_BY_MODEL = {
    "llama":     ["run1"],
    "qwen-base": ["run3"],
}

METRICS = [
    ("frac_changed",    "Fraction of weights changed"),
    ("mean_abs_change", "Mean |Δw|"),
    ("mean_pct_change", "Mean |Δw| / (|w| + ε)  [%]"),
    ("cosine_dist",     "Cosine distance (1 - cos sim)"),
    ("frobenius_norm",  "Frobenius norm of Δw"),
]

_LOW_COLORS  = {"3": "#a8c8e8", "6": "#5b9bd5", "9": "#1f6fb2", "12": "#0d528c", "15": "#0b3d6b"}
_HIGH_COLORS = {"3": "#f5a8a8", "6": "#e05c5c", "9": "#c0392b", "12": "#9c2f23", "15": "#7b1a10"}

# Layer index convention: 0 .. num_hidden_layers-1 (same as HF).
MODEL_NUM_LAYERS = {"llama": 32, "qwen-base": 36}

# ------------------------------------------------------------------
# Experiment layer orderings (low metric → high), **avg / mean — not final-start**:
#
#   Llama — config/layer_orderings_llama.txt
#     • Llama_Wasserstein: Wasserstein H0 **mean over finetuning epochs**
#     • Llama_Norm: normalized L2 **avg over 6 epochs**
#
#   Qwen-base — config/layer_orderings_qwen_base.txt
#     • QwenBase_Wasserstein: H0 from wasserstein_results.csv (same file as training)
#     • QwenBase_Norm: order_layers_by_norm.py **--mode avg** (6 epochs vs baseline)
#
# V/O lines only below (match those config files). Expected frozen: low-k = first k,
# high-k = last k. K/Q heatmaps use inferred layers only.
#
# Inferred: checkpoints with frac_changed == 0; [freeze-map] on mismatch.
# ------------------------------------------------------------------
EXPERIMENT_LAYER_ORDER = {
    "llama": {
        "wass": {
            "v": [29, 26, 30, 28, 31, 27, 25, 22, 23, 15, 24, 21, 17, 3, 1, 19, 20, 14, 9, 4, 12, 18, 13, 16, 5, 6, 10, 11, 7, 8, 2, 0],
            "o": [31, 30, 26, 29, 25, 24, 27, 21, 23, 28, 22, 20, 16, 4, 19, 17, 14, 3, 18, 15, 1, 6, 7, 2, 12, 8, 5, 9, 0, 11, 10, 13],
        },
        "norm": {
            "v": [30, 31, 29, 28, 27, 26, 25, 24, 23, 22, 12, 21, 9, 4, 20, 17, 15, 1, 19, 13, 16, 18, 14, 3, 11, 8, 6, 10, 7, 5, 0, 2],
            "o": [31, 30, 29, 28, 27, 26, 25, 24, 4, 12, 9, 8, 11, 15, 7, 13, 16, 22, 23, 17, 14, 10, 18, 19, 6, 21, 20, 3, 5, 1, 2, 0],
        },
    },
    "qwen-base": {
        "wass": {
            "v": [25, 10, 4, 6, 12, 17, 8, 15, 7, 19, 26, 34, 24, 28, 30, 2, 22, 27, 18, 23, 9, 14, 20, 11, 16, 33, 35, 32, 3, 21, 13, 31, 29, 1, 5, 0],
            "o": [6, 34, 29, 25, 28, 4, 30, 26, 31, 32, 9, 35, 27, 24, 5, 33, 8, 3, 21, 10, 7, 1, 12, 23, 2, 11, 0, 20, 13, 22, 16, 14, 19, 17, 18, 15],
        },
        "norm": {
            "v": [8, 12, 11, 10, 9, 13, 7, 14, 15, 17, 16, 18, 6, 4, 3, 33, 19, 5, 34, 25, 20, 31, 28, 32, 21, 2, 30, 26, 22, 35, 27, 23, 1, 24, 29, 0],
            "o": [10, 8, 12, 9, 11, 7, 4, 6, 3, 2, 14, 5, 17, 13, 16, 33, 19, 18, 23, 31, 1, 15, 24, 25, 26, 32, 22, 27, 30, 34, 29, 20, 28, 21, 35, 0],
        },
    },
}


_VARIANT_LOW_HIGH_K_RE = re.compile(r"(low|high)-(\d+)")


def _variant_low_high_k(variant):
    """e.g. low-12-frozen / high-6 → ('low'|'high', k)."""
    base = variant.replace("-frozen", "")
    m = _VARIANT_LOW_HIGH_K_RE.search(base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))

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


def num_hidden_layers_from_pretrained(pretrained_dir):
    """
    Layer indices in this script are 0 .. N-1 (HF config convention).
    Qwen3-8B-Base has num_hidden_layers=36; Llama-3.1-8B has 32 — do not assume 32 for Qwen plots.
    """
    cfg_path = os.path.join(pretrained_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        n = cfg.get("num_hidden_layers")
        if n is not None:
            return int(n)
    try:
        wm = load_weight_map(pretrained_dir)
    except RuntimeError:
        return None
    best = -1
    for k in wm:
        m = LAYER_RE.search(k)
        if m:
            best = max(best, int(m.group(1)))
    return (best + 1) if best >= 0 else None


def sanitize_layer_indices(layer_list, num_layers, context=""):
    if layer_list is None or num_layers is None:
        return layer_list
    bad = [li for li in layer_list if li < 0 or li >= num_layers]
    if bad:
        print(
            f"  [warn]{context} dropping invalid layer indices {bad} "
            f"(num_hidden_layers={num_layers}, valid 0..{num_layers - 1})"
        )
    return sorted({li for li in layer_list if 0 <= li < num_layers})


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
    for run_id in RUN_PREFERENCE_BY_MODEL.get(model, ["run3", "run2", "run1"]):
        run_dir = os.path.join(variant_dir, run_id)
        result = _resolve_weights_dir(run_dir)
        if result:
            return result
    return None


def _resolve_multi_run_baseline(override_path, default_root, model):
    """Prefer explicit override dir (run folder or folder with shards); else try run*/ under default_root."""
    if override_path and os.path.isdir(override_path):
        hit = _resolve_weights_dir(override_path)
        if hit:
            return hit
    if not default_root or not os.path.isdir(default_root):
        return None
    for run_id in RUN_PREFERENCE_BY_MODEL.get(model, ["run3", "run2", "run1"]):
        run_dir = os.path.join(default_root, run_id)
        if not os.path.isdir(run_dir):
            continue
        hit = _resolve_weights_dir(run_dir)
        if hit:
            return hit
    return None


def _short_name(variant, criterion):
    base = variant.replace("-frozen", "")
    base = re.sub(r"-(\d+)$", r"\1", base)
    return f"{criterion}-{base}"


def _plot_style(exp_name):
    if exp_name == "full" or "full-ft" in exp_name:
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
    if proj not in ("v", "o"):
        return None
    side, k = _variant_low_high_k(variant)
    if side is None or k is None or k < 1:
        return None
    order = EXPERIMENT_LAYER_ORDER.get(model, {}).get(criterion, {}).get(proj)
    if not order:
        return None
    n = len(order)
    exp_n = MODEL_NUM_LAYERS.get(model)
    if exp_n is not None and n != exp_n:
        raise RuntimeError(
            f"EXPERIMENT_LAYER_ORDER[{model!r}][{criterion!r}][{proj!r}] "
            f"has length {n}, expected {exp_n}"
        )
    if k > n:
        return None
    if side == "low":
        return list(order[:k])
    return list(order[-k:])


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


# ── TDA reference signals (same sources as training layer orderings) ─────────

def _avg_by_layer_from_l2_csv(csv_path, proj):
    acc = defaultdict(list)
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("Projection", "").lower() != proj:
                continue
            li = int(row["Layer"])
            acc[li].append(float(row["L2_Normalized"]))
    return {li: sum(v) / len(v) for li, v in acc.items()}


def _avg_by_layer_from_wass_csv(csv_path, proj):
    acc = defaultdict(list)
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("Projection", "").lower() != proj:
                continue
            fn = row.get("File", "")
            m = re.match(r"layer(\d+)_", fn)
            if not m:
                continue
            li = int(m.group(1))
            acc[li].append(float(row["Wasserstein H0"]))
    return {li: sum(v) / len(v) for li, v in acc.items()}


def load_tda_l2_norm_avg_by_layer(model, proj):
    if proj not in ("v", "o"):
        return {}
    if model == "llama":
        p = os.path.join(REPO_ROOT, "analysis/tda/gsm8k-llama-tda-results/l2_results.csv")
        if not os.path.isfile(p):
            return {}
        return _avg_by_layer_from_l2_csv(p, proj)
    jp = os.path.join(REPO_ROOT, "analysis/tda/gsm8k-qwen-base-tda-results/avg_norm_vo.json")
    if not os.path.isfile(jp):
        return {}
    with open(jp) as f:
        d = json.load(f)
    key = proj.upper()
    return {int(k): float(v) for k, v in d.get(key, {}).items()}


def load_tda_wass_avg_by_layer(model, proj):
    if proj not in ("v", "o"):
        return {}
    sub = "gsm8k-llama-tda-results" if model == "llama" else "gsm8k-qwen-base-tda-results"
    p = os.path.join(REPO_ROOT, "analysis/tda", sub, "wasserstein_results.csv")
    if not os.path.isfile(p):
        return {}
    return _avg_by_layer_from_wass_csv(p, proj)


def _rank_smallest_first(val_by_layer):
    """Rank 1 = smallest value; ties broken by layer index."""
    valid = {k: v for k, v in val_by_layer.items() if v == v and np.isfinite(v)}
    ordered = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))
    return {li: r + 1 for r, (li, _) in enumerate(ordered)}


def _fmt_metric_cell(x):
    if x is None:
        return ""
    if isinstance(x, float) and (x != x or not np.isfinite(x)):
        return ""
    return f"{x:.10g}"


def write_layer_metric_ranks_txt(out_dir, model, full_metrics, num_layers):
    """
    Per-layer table for V and O: TDA L2 avg, TDA Wass avg, then weight-diff metrics
    from pretrained vs full-FT (cosine, frac, frobenius, mean_abs, mean_pct), each with rank 1 = smallest.
    """
    sub = os.path.join(out_dir, "rankings")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"layer_metric_ranks__{model}.txt")
    fm = full_metrics or {}
    wf_keys = (
        "cosine_dist",
        "frac_changed",
        "frobenius_norm",
        "mean_abs_change",
        "mean_pct_change",
    )
    lines = [
        f"# model={model}",
        "# Column order: l2_tda_avg | wasserstein_tda_avg | cosine_dist | frac_changed | "
        "frobenius_norm | mean_abs_change | mean_pct_change",
        "# l2 / wass: analysis/tda/... (mean L2_Normalized over epochs; mean Wasserstein H0 over epochs).",
        "# weight metrics: pretrained vs full-FT (same as heatmap row 'full').",
        "# rank_* : 1 = smallest among layers with a finite value for that metric.",
        "",
    ]
    for proj in ("v", "o"):
        l2_d = load_tda_l2_norm_avg_by_layer(model, proj)
        wa_d = load_tda_wass_avg_by_layer(model, proj)
        wf = {k: dict(fm.get(k, {}).get(proj, {})) for k in wf_keys}
        if num_layers is not None:
            layer_ids = list(range(num_layers))
        else:
            layer_ids = sorted(
                set(l2_d) | set(wa_d) | set().union(*(set(wf[k]) for k in wf_keys))
            )
        r_l2 = _rank_smallest_first(l2_d)
        r_wa = _rank_smallest_first(wa_d)
        r_cos = _rank_smallest_first(wf["cosine_dist"])
        r_frac = _rank_smallest_first(wf["frac_changed"])
        r_frob = _rank_smallest_first(wf["frobenius_norm"])
        r_mabs = _rank_smallest_first(wf["mean_abs_change"])
        r_mpct = _rank_smallest_first(wf["mean_pct_change"])
        lines.append(f"## projection={proj.upper()}")
        lines.append(
            "layer\tl2_tda_avg\trank_l2\twass_tda_avg\trank_wass\tcosine\trank_cos\t"
            "frac\trank_frac\tfrobenius\trank_frob\tmean_abs\trank_mabs\tmean_pct\trank_mpct"
        )
        for li in layer_ids:
            lines.append(
                "\t".join(
                    [
                        str(li),
                        _fmt_metric_cell(l2_d.get(li)),
                        str(r_l2.get(li, "")),
                        _fmt_metric_cell(wa_d.get(li)),
                        str(r_wa.get(li, "")),
                        _fmt_metric_cell(wf["cosine_dist"].get(li)),
                        str(r_cos.get(li, "")),
                        _fmt_metric_cell(wf["frac_changed"].get(li)),
                        str(r_frac.get(li, "")),
                        _fmt_metric_cell(wf["frobenius_norm"].get(li)),
                        str(r_frob.get(li, "")),
                        _fmt_metric_cell(wf["mean_abs_change"].get(li)),
                        str(r_mabs.get(li, "")),
                        _fmt_metric_cell(wf["mean_pct_change"].get(li)),
                        str(r_mpct.get(li, "")),
                    ]
                )
            )
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def save_tda_reference_plots(out_dir, model, num_layers):
    subfolder = os.path.join(out_dir, "tda_reference")
    os.makedirs(subfolder, exist_ok=True)
    for proj in ("v", "o"):
        norm_d = load_tda_l2_norm_avg_by_layer(model, proj)
        wass_d = load_tda_wass_avg_by_layer(model, proj)
        if not norm_d and not wass_d:
            print(f"  [skip] tda_reference/{proj}: no TDA files under {REPO_ROOT}/analysis/tda/")
            continue
        xs = list(range(num_layers)) if num_layers is not None else sorted(
            set(norm_d) | set(wass_d)
        )
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        ax0.plot(xs, [norm_d.get(i, float("nan")) for i in xs], "b.-", linewidth=1.2, markersize=4)
        ax0.set_ylabel("TDA L2 norm avg (normalized)")
        ax0.grid(True, alpha=0.3)
        ax1.plot(xs, [wass_d.get(i, float("nan")) for i in xs], "r.-", linewidth=1.2, markersize=4)
        ax1.set_ylabel("TDA Wasserstein H0 avg")
        ax1.set_xlabel("Layer index (0-based)")
        ax1.grid(True, alpha=0.3)
        fig.suptitle(
            f"[{model}] {proj.upper()} — TDA reference (epoch-mean L2 & Wass; GSM8K)",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = os.path.join(subfolder, f"tda_ref__l2_and_wass__{proj}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


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

def save_split_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model, num_layers=None):
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
    if num_layers is not None:
        x_min, x_max = 0, num_layers - 1

    fig, axes_grid = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
    axes = [axes_grid[0,0], axes_grid[0,1], axes_grid[1,0], axes_grid[1,1]]

    is_kq = proj in ("k", "q")

    for ax, (criterion, direction, title) in zip(axes, PANELS):
        panel_labels = sorted(
            [l for l in all_series if criterion in l and direction in l and l != "full"],
            key=lambda l: _n_from_label(l)
        )

        if "full" in all_series:
            ft_layers, ft_vals = all_series["full"]
            ax.plot(ft_layers, ft_vals, color="black", linewidth=2.0,
                    alpha=0.3, label="full", zorder=1)

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

    layer_note = f" | {num_layers} transformer layers (indices 0..{num_layers - 1})" if num_layers is not None else ""
    note = f"{proj.upper()}: freeze variants had 0 weight change" if is_kq else \
           "blue=low layers, red=high layers | scatter=unfrozen layers only"
    fig.suptitle(f"[{model}] {proj.upper()} — {metric_title}{layer_note}\n{note}", fontsize=12)
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "split"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"split__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot B ────────────────────────────────────────────────────────────────────

def save_collapsed_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model, num_layers=None):
    all_x = [li for layers, _ in all_series.values() for li in layers]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)
    if num_layers is not None:
        x_min, x_max = 0, num_layers - 1

    fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    is_kq = proj in ("k", "q")

    for ax, direction, panel_title in [
        (ax_low,  "low",  "Low layer freezing"),
        (ax_high, "high", "High layer freezing"),
    ]:
        panel_labels = sorted(
            [l for l in all_series if direction in l and l != "full"],
            key=lambda l: _n_from_label(l)
        )

        if "full" in all_series:
            ft_layers, ft_vals = all_series["full"]
            ax.plot(ft_layers, ft_vals, color="black", linewidth=2.0,
                    alpha=0.3, label="full", zorder=1)

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
    layer_note = f" ({num_layers} layers)" if num_layers is not None else ""
    fig.suptitle(f"[{model}] {proj.upper()} — {metric_title}{layer_note}", fontsize=12)
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
        if label == "full":
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

def save_delta_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model, num_layers=None):
    is_kq = proj in ("k", "q")
    if is_kq or "full" not in all_series:
        return

    ft_layers, ft_vals = all_series["full"]
    ft_lookup = dict(zip(ft_layers, ft_vals))
    all_x = [li for layers, _ in all_series.values() for li in layers]
    if not all_x:
        return
    x_min, x_max = min(all_x), max(all_x)
    if num_layers is not None:
        x_min, x_max = 0, num_layers - 1

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
            [l for l in all_series if criterion in l and direction in l and l != "full"],
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

    axes_grid[0,0].set_ylabel(f"Δ {metric_title} (variant − full)", fontsize=10)
    axes_grid[1,0].set_ylabel(f"Δ {metric_title} (variant − full)", fontsize=10)
    layer_note = f" ({num_layers} layers)" if num_layers is not None else ""
    fig.suptitle(
        f"[{model}] {proj.upper()} — {metric_title} deviation from full FT{layer_note}",
        fontsize=12
    )
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "delta"); os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"delta__{metric_key}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot E: FIXED HEATMAP ─────────────────────────────────────────────────────

def save_heatmap_plot(metric_key, metric_title, proj, all_series, all_frozen_layers, out_dir, model, num_layers=None):
    """
    Overlays:
      - gray (#dddddd)  = expected frozen (from EXPERIMENT_LAYER_ORDER[v/o] or inferred)
      - light blue      = frac_changed==0 but not in expected frozen (weights match baseline)
      - pink            = no metric / weight key for this layer
      - viridis         = observed value
    When num_hidden_layers is known (from pretrained config), x-axis is always 0..N-1 so Qwen3 (36)
    is not confused with Llama (32).
    """
    is_kq = proj in ("k", "q")
    if is_kq:
        return

    variant_keys = [l for l in all_series if l not in _HEATMAP_BASELINE_ROWS]

    observed_union = set(
        li for label in all_series
        for li in all_series[label][0]
    )
    expected_union = set(
        li for label in variant_keys
        for li in all_frozen_layers.get(label, [])
    )
    all_x = sorted(observed_union.union(expected_union))
    if num_layers is not None:
        bad = [li for li in all_x if li < 0 or li >= num_layers]
        if bad:
            print(f"  [warn] heatmap {proj}: dropping out-of-range layer indices {bad} (num_layers={num_layers})")
        all_x = list(range(num_layers))
    elif not all_x:
        return

    def _draw_heatmap(ax, labels, all_series, all_frozen_layers, all_x, metric_title, title):
        n_layers = len(all_x)
        layer_to_col = {li: i for i, li in enumerate(all_x)}

        observed = np.full((len(labels), n_layers), np.nan)
        frozen_overlay = np.full((len(labels), n_layers), np.nan)
        missing_overlay = np.full((len(labels), n_layers), np.nan)
        unchanged_overlay = np.full((len(labels), n_layers), np.nan)

        for row_i, label in enumerate(labels):
            layers, vals = all_series[label]
            val_by_layer = dict(zip(layers, vals))
            present_layers = set(layers)
            frozen_set = set(all_frozen_layers.get(label, []))

            for li in all_x:
                col_i = layer_to_col[li]
                if li in frozen_set:
                    frozen_overlay[row_i, col_i] = 1.0
                elif li not in present_layers:
                    missing_overlay[row_i, col_i] = 1.0
                elif metric_key == "frac_changed" and val_by_layer[li] == 0.0:
                    unchanged_overlay[row_i, col_i] = 1.0
                else:
                    observed[row_i, col_i] = val_by_layer[li]

        extent = [-0.5, n_layers - 0.5, len(labels) - 0.5, -0.5]

        missing_masked = np.ma.masked_invalid(missing_overlay)
        unchanged_masked = np.ma.masked_invalid(unchanged_overlay)
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
            unchanged_masked,
            aspect="auto",
            cmap=mcolors.ListedColormap(["#c8d8f0"]),
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
        # 1-based layer labels (avoids confusion: last Qwen3 layer is 36, not "35" as 0-based)
        ax.set_xticklabels([str(i + 1) for i in all_x], fontsize=7, rotation=90)
        ax.set_yticks(range(len(labels)))
        short_labels = [re.sub(r"(norm|wass)-", "", l) for l in labels]
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Layer (1-based)", fontsize=9)
        ax.set_title(title, fontsize=10)

        legend_handles = [
            Patch(facecolor="#dddddd", edgecolor="none", label="Expected frozen"),
            Patch(facecolor="#c8d8f0", edgecolor="none", label="Unchanged vs baseline (not in expected)"),
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
            [
                l for l in all_series
                if criterion in l and direction in l and l not in _HEATMAP_BASELINE_ROWS
            ],
            key=lambda l: _n_from_label(l),
        )
        if "full" in all_series:
            labels = ["full"] + labels
        if not labels:
            ax.set_visible(False)
            continue
        _draw_heatmap(ax, labels, all_series, all_frozen_layers, all_x, metric_title, title)

    layer_note = (
        f" | {num_layers} layers (x-axis 1…{num_layers})"
        if num_layers is not None
        else ""
    )
    fig.suptitle(
        f"[{model}] {proj.upper()} — {metric_title} heatmap{layer_note}\n"
        "expected frozen = gray | unchanged vs baseline (not expected) = light blue | missing = pink",
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
    ap.add_argument("--model", choices=["llama", "qwen-base"], default="llama")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pretrained_dir", default=None)
    ap.add_argument(
        "--full_ft_dir",
        default=None,
        help="Optional: explicit full-finetune folder (run dir or shard root); else use model's checkpoints/.../full run*",
    )
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

    num_layers = num_hidden_layers_from_pretrained(pretrained_dir)
    if num_layers is not None:
        print(f"Layers   : {num_layers} transformer blocks (indices 0..{num_layers - 1})")

    print(f"Model    : {args.model}")
    print(f"Run pref : {RUN_PREFERENCE_BY_MODEL.get(args.model, [])}")
    print(f"Baseline : {pretrained_dir}")
    print(f"Metrics  : {[mk for mk, _ in metrics_to_run]}")
    print(f"Output   : {out_dir}\n")

    all_combined   = {mk: {p: {} for p in "kqvo"} for mk, _ in metrics_to_run}
    frozen_by_exp  = {p: {} for p in "kqvo"}
    full_metrics = None

    full_w = _resolve_multi_run_baseline(
        args.full_ft_dir,
        FULL_MULTI_RUN_ROOT.get(args.model),
        args.model,
    )
    if full_w:
        print(f"Loading full finetune vs pretrained: {full_w}")
        full_metrics = compute_all_metrics(pretrained_dir, full_w)
        for mk, _ in metrics_to_run:
            for proj in "kqvo":
                if full_metrics.get(mk, {}).get(proj):
                    layers = sorted(full_metrics[mk][proj].keys())
                    vals = [full_metrics[mk][proj][li] for li in layers]
                    all_combined[mk][proj]["full"] = (layers, vals)
        for proj in "kqvo":
            frozen_by_exp[proj]["full"] = []
    else:
        print(
            f"  [skip] full — no weights "
            f"(override={args.full_ft_dir!r}, root={FULL_MULTI_RUN_ROOT.get(args.model)!r})"
        )

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
                    frozen_by_exp[proj][exp_label] = sanitize_layer_indices(
                        sorted(expected), num_layers, context=f" {exp_label} {proj}"
                    )

                    present_layers = sorted(exp_metrics.get("frac_changed", {}).get(proj, {}).keys())
                    missing_layers = [li for li in frozen_by_exp[proj][exp_label] if li not in present_layers]
                    if missing_layers:
                        print(f"  [coverage] {exp_label} {proj.upper()} expected frozen but absent from metric output: {missing_layers}")

                    if inferred != sorted(expected):
                        print(f"  [freeze-map] {exp_label} {proj.upper()} inferred={inferred} expected={sorted(expected)}")
                else:
                    frozen_by_exp[proj][exp_label] = sanitize_layer_indices(
                        inferred, num_layers, context=f" {exp_label} {proj} inferred"
                    )

                # Extra visibility for your qwen-base tail-layer issue
                if args.model == "qwen-base" and proj in ("v", "o"):
                    present_layers = set(exp_metrics.get("frac_changed", {}).get(proj, {}).keys())
                    # 0-based; 1-based display layers 33–36
                    tail = [32, 33, 34, 35]
                    tail_missing = [li for li in tail if li not in present_layers]
                    if tail_missing:
                        print(
                            f"  [tail-missing] {exp_label} {proj.upper()} "
                            f"missing 0-based {tail_missing} (1-based {[i + 1 for i in tail_missing]})"
                        )

    print(f"\n{'='*55}")
    print("Saving plots + TDA reference + rankings ...")
    for mk, mt in metrics_to_run:
        for proj in "kqvo":
            series = all_combined[mk][proj]
            if not series:
                continue
            frozen = frozen_by_exp[proj]
            save_split_plot(mk, mt, proj, series, frozen, out_dir, args.model, num_layers)
            save_collapsed_plot(mk, mt, proj, series, frozen, out_dir, args.model, num_layers)
            save_summary_plot(mk, mt, proj, series, frozen, out_dir, args.model)
            save_delta_plot(mk, mt, proj, series, frozen, out_dir, args.model, num_layers)
            save_heatmap_plot(mk, mt, proj, series, frozen, out_dir, args.model, num_layers)

    save_tda_reference_plots(out_dir, args.model, num_layers)
    write_layer_metric_ranks_txt(out_dir, args.model, full_metrics, num_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()