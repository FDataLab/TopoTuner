"""
freeze_comparison_l2norm_kadir.py
------------------------------------------------------------------------------
Compares pretrained baseline vs. norm-freeze experiments using:

    normalized_l2(w_i, w_0) = ||w_i - w_0||_2 / (||w_0||_2 + eps)

Outputs heatmap plots only, for v and o projections.
Norm-only experiments (no Wasserstein).
"""

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

EPS_L2 = 1e-12

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

PROJ_PATTERNS = {
    "k": ".k_proj.",
    "q": ".q_proj.",
    "v": ".v_proj.",
    "o": ".o_proj.",
}

METRIC_KEY   = "normalized_l2"
METRIC_TITLE = "Normalized L2  ||dw||_2 / (||w0||_2 + eps)"

OUT_DIR    = "/home/vepaul/results"
CHKPT_BASE = "/home/kadir/topo/numpy_weights/exploration-finetuning/checkpoints"

MODEL_DIRS = {
    "llama": {
        "norm": os.path.join(CHKPT_BASE, "llama", "norm-freeze"),
    },
    "qwen-base": {
        "norm": os.path.join(CHKPT_BASE, "qwen-base", "norm-freeze"),
    },
}

PRETRAINED_DIRS = {
    "llama":     "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/llama31-8b/",
    "qwen-base": "/data/cuneyt-topo/numpy_weights/exploration-finetuning/pretrained/qwen3-8b-base/",
}

FULL_MULTI_RUN_ROOT = {
    "llama":     os.path.join(CHKPT_BASE, "llama", "full"),
    "qwen-base": os.path.join(CHKPT_BASE, "qwen-base", "full"),
}

_HEATMAP_BASELINE_ROWS = frozenset({"full"})

_VARIANT_KS = ["3", "6", "9", "12", "15"]

VARIANT_NAMES = {
    "llama": {
        "norm": [f"high-{k}" for k in _VARIANT_KS] + [f"low-{k}" for k in _VARIANT_KS],
    },
    "qwen-base": {
        "norm": [f"high-{k}" for k in _VARIANT_KS] + [f"low-{k}" for k in _VARIANT_KS],
    },
}

RUN_PREFERENCE_BY_MODEL = {
    "llama":     ["run1"],
    "qwen-base": ["run3"],
}

MODEL_NUM_LAYERS = {"llama": 32, "qwen-base": 36}

EXPERIMENT_LAYER_ORDER = {
    "llama": {
        "norm": {
            "v": [30, 31, 29, 28, 27, 26, 25, 24, 23, 22, 12, 21, 9, 4, 20, 17, 15, 1, 19, 13, 16, 18, 14, 3, 11, 8, 6, 10, 7, 5, 0, 2],
            "o": [31, 30, 29, 28, 27, 26, 25, 24, 4, 12, 9, 8, 11, 15, 7, 13, 16, 22, 23, 17, 14, 10, 18, 19, 6, 21, 20, 3, 5, 1, 2, 0],
        },
    },
    "qwen-base": {
        "norm": {
            "v": [8, 12, 11, 10, 9, 13, 7, 14, 15, 17, 16, 18, 6, 4, 3, 33, 19, 5, 34, 25, 20, 31, 28, 32, 21, 2, 30, 26, 22, 35, 27, 23, 1, 24, 29, 0],
            "o": [10, 8, 12, 9, 11, 7, 4, 6, 3, 2, 14, 5, 17, 13, 16, 33, 19, 18, 23, 31, 1, 15, 24, 25, 26, 32, 22, 27, 30, 34, 29, 20, 28, 21, 35, 0],
        },
    },
}

_VARIANT_LOW_HIGH_K_RE = re.compile(r"(low|high)-(\d+)")


def _variant_low_high_k(variant):
    """e.g. 'low-12' / 'high-6' -> ('low'|'high', k)."""
    base = variant.replace("-frozen", "")
    m = _VARIANT_LOW_HIGH_K_RE.search(base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


# --- Helpers ------------------------------------------------------------------

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
    """LRU cache for safetensors shards - avoids reloading the same file."""
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


def _n_from_label(label):
    m = re.search(r"(\d+)$", label)
    return int(m.group(1)) if m else 0


def _infer_frozen_layers(result_by_proj, proj):
    """Layers where normalized_l2 == 0.0 are treated as frozen."""
    return sorted(li for li, v in result_by_proj.get(proj, {}).items() if v == 0.0)


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
    return list(order[:k]) if side == "low" else list(order[-k:])


# --- Metric computation -------------------------------------------------------

def compute_normalized_l2(dir_a, dir_b):
    """
    Computes normalized L2 per (layer, projection), matching:

        def normalized_l2_diff(w_i, w_0, eps=1e-12):
            diff = np.linalg.norm(w_i - w_0)
            base = np.linalg.norm(w_0)
            return diff / (base + eps)

    w_0 = pretrained weights (dir_a), w_i = finetuned weights (dir_b).

    Because a single projection matrix may be split across multiple safetensors
    shards, squared elements are accumulated across all shards first:
        sum_sq_delta  <- sum of (w_i_j - w_0_j)^2
        sum_sq_base   <- sum of w_0_j^2

    Then norms are reconstructed:
        ||dw||_2  = sqrt(sum_sq_delta)
        ||w0||_2  = sqrt(sum_sq_base)
        result    = ||dw||_2 / (||w0||_2 + EPS_L2)

    Returns: dict { proj: { layer_idx: float } } for proj in "kqvo"
    """
    map_a = load_weight_map(dir_a)
    map_b = load_weight_map(dir_b)
    common_keys = sorted(set(map_a.keys()).intersection(map_b.keys()))
    if not common_keys:
        return {}

    cache_a = ShardCache(dir_a)
    cache_b = ShardCache(dir_b)

    acc = {}  # (layer_idx, proj) -> {"sum_sq_delta": float, "sum_sq_base": float}

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
        key   = (li, proj)
        if key not in acc:
            acc[key] = {"sum_sq_delta": 0.0, "sum_sq_base": 0.0}

        acc[key]["sum_sq_delta"] += (delta * delta).sum().item()
        acc[key]["sum_sq_base"]  += (t0 * t0).sum().item()

    result = {p: {} for p in "kqvo"}
    for (li, proj), a in acc.items():
        diff = a["sum_sq_delta"] ** 0.5   # ||w_i - w_0||_2
        base = a["sum_sq_base"]  ** 0.5   # ||w_0||_2
        result[proj][li] = diff / (base + EPS_L2)

    return result


# --- Heatmap ------------------------------------------------------------------

def save_heatmap_plot(proj, all_series, all_frozen_layers, out_dir, model, num_layers=None):
    """
    1x2 heatmap panels: norm-low | norm-high.
    Rows = experiment variants; columns = layers.

    Cell color encoding:
      viridis    - observed normalized L2 value
      gray       - expected frozen (from EXPERIMENT_LAYER_ORDER)
      light blue - normalized L2 == 0 but not in expected frozen
      pink       - layer absent from metric output entirely

    Skipped for k/q projections.
    """
    if proj in ("k", "q"):
        return

    variant_keys   = [l for l in all_series if l not in _HEATMAP_BASELINE_ROWS]
    observed_union = set(li for label in all_series for li in all_series[label][0])
    expected_union = set(li for label in variant_keys for li in all_frozen_layers.get(label, []))
    all_x = sorted(observed_union.union(expected_union))
    if num_layers is not None:
        bad = [li for li in all_x if li < 0 or li >= num_layers]
        if bad:
            print(f"  [warn] heatmap {proj}: dropping out-of-range indices {bad}")
        all_x = list(range(num_layers))
    elif not all_x:
        return

    def _draw_heatmap(ax, labels, title):
        n_layers     = len(all_x)
        layer_to_col = {li: i for i, li in enumerate(all_x)}

        observed          = np.full((len(labels), n_layers), np.nan)
        frozen_overlay    = np.full((len(labels), n_layers), np.nan)
        missing_overlay   = np.full((len(labels), n_layers), np.nan)
        unchanged_overlay = np.full((len(labels), n_layers), np.nan)

        for row_i, label in enumerate(labels):
            layers, vals   = all_series[label]
            val_by_layer   = dict(zip(layers, vals))
            present_layers = set(layers)
            frozen_set     = set(all_frozen_layers.get(label, []))

            for li in all_x:
                col_i = layer_to_col[li]
                if li in frozen_set:
                    frozen_overlay[row_i, col_i] = 1.0
                elif li not in present_layers:
                    missing_overlay[row_i, col_i] = 1.0
                elif val_by_layer[li] == 0.0:
                    unchanged_overlay[row_i, col_i] = 1.0
                else:
                    observed[row_i, col_i] = val_by_layer[li]

        extent = [-0.5, n_layers - 0.5, len(labels) - 0.5, -0.5]
        ax.imshow(np.ma.masked_invalid(missing_overlay),   aspect="auto",
                  cmap=mcolors.ListedColormap(["#f6d6d6"]), extent=extent, interpolation="none")
        ax.imshow(np.ma.masked_invalid(unchanged_overlay), aspect="auto",
                  cmap=mcolors.ListedColormap(["#c8d8f0"]), extent=extent, interpolation="none")
        ax.imshow(np.ma.masked_invalid(frozen_overlay),    aspect="auto",
                  cmap=mcolors.ListedColormap(["#dddddd"]), extent=extent, interpolation="none")

        cmap = plt.cm.viridis.copy()
        cmap.set_bad((1, 1, 1, 0))
        im = ax.imshow(np.ma.masked_invalid(observed), aspect="auto",
                       cmap=cmap, extent=extent, interpolation="none")

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([str(i + 1) for i in all_x], fontsize=7, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([re.sub(r"(norm|wass)-", "", l) for l in labels], fontsize=8)
        ax.set_xlabel("Layer (1-based)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(handles=[
            Patch(facecolor="#dddddd", edgecolor="none", label="Expected frozen"),
            Patch(facecolor="#c8d8f0", edgecolor="none", label="Unchanged vs baseline (not in expected)"),
            Patch(facecolor="#f6d6d6", edgecolor="none", label="Missing / not matched"),
        ], loc="upper right", fontsize=7, framealpha=0.9)
        plt.colorbar(im, ax=ax, shrink=0.8, label=METRIC_TITLE)

    PANELS = [
        ("norm", "low",  "Norm - Low freezing"),
        ("norm", "high", "Norm - High freezing"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(22, 6))

    for ax, (criterion, direction, title) in zip(axes, PANELS):
        labels = sorted(
            [l for l in all_series if criterion in l and direction in l
             and l not in _HEATMAP_BASELINE_ROWS],
            key=_n_from_label,
        )
        if "full" in all_series:
            labels = ["full"] + labels
        if not labels:
            ax.set_visible(False)
            continue
        _draw_heatmap(ax, labels, title)

    layer_note = f" | {num_layers} layers (x-axis 1..{num_layers})" if num_layers else ""
    fig.suptitle(
        f"[{model}] {proj.upper()} - {METRIC_TITLE} heatmap{layer_note}\n"
        "gray=expected frozen | light blue=unchanged (not expected) | pink=missing",
        fontsize=12
    )
    plt.tight_layout()
    subfolder = os.path.join(out_dir, "heatmap")
    os.makedirs(subfolder, exist_ok=True)
    out_path = os.path.join(subfolder, f"heatmap__{METRIC_KEY}__{proj}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# --- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Normalized L2 weight diff: pretrained vs. norm-freeze experiments."
    )
    ap.add_argument("--model", choices=["llama", "qwen-base"], default="llama")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pretrained_dir", default=None)
    ap.add_argument("--full_ft_dir", default=None,
                    help="Optional: explicit full-finetune folder; "
                         "else uses model's checkpoints/.../full run*")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(OUT_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)

    pretrained_dir = args.pretrained_dir or PRETRAINED_DIRS.get(args.model)
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        raise RuntimeError(f"Pretrained dir not found: {pretrained_dir}")

    num_layers = num_hidden_layers_from_pretrained(pretrained_dir)
    if num_layers is not None:
        print(f"Layers   : {num_layers} transformer blocks (indices 0..{num_layers - 1})")

    print(f"Model    : {args.model}")
    print(f"Run pref : {RUN_PREFERENCE_BY_MODEL.get(args.model, [])}")
    print(f"Baseline : {pretrained_dir}")
    print(f"Metric   : {METRIC_KEY}")
    print(f"Criteria : norm only")
    print(f"Output   : {out_dir}\n")

    # all_combined[proj][exp_label] = (layers, vals)
    all_combined  = {p: {} for p in "kqvo"}
    frozen_by_exp = {p: {} for p in "kqvo"}

    # Full fine-tune baseline
    full_w = _resolve_multi_run_baseline(
        args.full_ft_dir, FULL_MULTI_RUN_ROOT.get(args.model), args.model
    )
    if full_w:
        print(f"Loading full finetune vs pretrained: {full_w}")
        full_result = compute_normalized_l2(pretrained_dir, full_w)
        for proj in "kqvo":
            if full_result.get(proj):
                layers = sorted(full_result[proj].keys())
                all_combined[proj]["full"] = (layers, [full_result[proj][li] for li in layers])
            frozen_by_exp[proj]["full"] = []
    else:
        print(f"  [skip] full - no weights "
              f"(override={args.full_ft_dir!r}, root={FULL_MULTI_RUN_ROOT.get(args.model)!r})")

    # Norm-freeze experiments
    for criterion, variants in VARIANT_NAMES[args.model].items():
        for variant in variants:
            exp_dir = _resolve_exp_dir(args.model, criterion, variant)
            if not exp_dir:
                print(f"  [skip] {criterion}/{variant} - not found")
                continue

            exp_label = _short_name(variant, criterion)
            print(f"Computing: pretrained vs {exp_label}  ({exp_dir})")
            exp_result = compute_normalized_l2(pretrained_dir, exp_dir)
            if not exp_result:
                print("  [WARNING] No data - skipping.")
                continue

            for proj in "kqvo":
                if exp_result.get(proj):
                    layers = sorted(exp_result[proj].keys())
                    all_combined[proj][exp_label] = (layers, [exp_result[proj][li] for li in layers])

            for proj in "kqvo":
                expected = _get_expected_frozen_layers(args.model, criterion, variant, proj)
                inferred = _infer_frozen_layers(exp_result, proj)

                if expected is not None:
                    frozen_by_exp[proj][exp_label] = sanitize_layer_indices(
                        sorted(expected), num_layers, context=f" {exp_label} {proj}"
                    )
                    present = sorted(exp_result.get(proj, {}).keys())
                    missing = [li for li in frozen_by_exp[proj][exp_label] if li not in present]
                    if missing:
                        print(f"  [coverage] {exp_label} {proj.upper()} "
                              f"expected frozen but absent from metric output: {missing}")
                    if inferred != sorted(expected):
                        print(f"  [freeze-map] {exp_label} {proj.upper()} "
                              f"inferred={inferred} expected={sorted(expected)}")
                else:
                    frozen_by_exp[proj][exp_label] = sanitize_layer_indices(
                        inferred, num_layers, context=f" {exp_label} {proj} inferred"
                    )

                if args.model == "qwen-base" and proj in ("v", "o"):
                    present_set  = set(exp_result.get(proj, {}).keys())
                    tail_missing = [li for li in [32, 33, 34, 35] if li not in present_set]
                    if tail_missing:
                        print(f"  [tail-missing] {exp_label} {proj.upper()} "
                              f"missing 0-based {tail_missing} "
                              f"(1-based {[i + 1 for i in tail_missing]})")

    # Save heatmap plots
    print(f"\n{'=' * 55}")
    print("Saving heatmap plots ...")
    for proj in "kqvo":
        series = all_combined[proj]
        if not series:
            continue
        save_heatmap_plot(proj, series, frozen_by_exp[proj], out_dir, args.model, num_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()