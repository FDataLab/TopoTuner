"""
Compare baseline (pretrained) vs finetuned experiments.
- Baseline: pretrained (same for all experiments)
- Experiments: full finetuning + selective-freeze variants, all at final epoch
- Only frac_changed (fraction of weights that changed)
- Splits by K, Q, V, O per layer.

For V and O combined plots: frozen layers are shown as shaded vertical bands
instead of being plotted as zeros (which caused sawtooth artifacts).
For K and Q combined plots: same shading applied; only full-ft line is drawn
since all freeze variants are confirmed zero in those projections.
"""

import os
import re
import json
import argparse
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file as safetensors_load_file


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
CHANGE_THRESH = 0

PROJ_PATTERNS = {
    "k": ".k_proj.",
    "q": ".q_proj.",
    "v": ".v_proj.",
    "o": ".o_proj.",
}

ROOT_DIR = "/home/kadir/topo/numpy_weights/exploration-finetuning"
OUT_DIR = "/home/vepaul/results"
PRETRAINED_DIR = "/home/kadir/topo/numpy_weights/exploration-finetuning/pretrained/llama31-8b"
RUN_QUICK_TEST = True

QUICK_TEST_EXPERIMENTS = ["gsm8k-full-finetuned-tda", "gsm8k-frozen-norm-high6"]
HF_MODEL_ID = "meta-llama/Llama-3.1-8B"

FULL_EXPERIMENTS = [
    "gsm8k-full-finetuned-tda",
    "gsm8k-frozen-norm-low3", "gsm8k-frozen-norm-low6", "gsm8k-frozen-norm-low9", "gsm8k-frozen-norm-low15",
    "gsm8k-frozen-norm-high6", "gsm8k-frozen-norm-high9",
    "gsm8k-frozen-wass-low3", "gsm8k-frozen-wass-low6", "gsm8k-frozen-wass-low9", "gsm8k-frozen-wass-low15",
    "gsm8k-frozen-wass-high6", "gsm8k-frozen-wass-high9",
]

EXPERIMENTS = ["gsm8k-full-finetuned-tda", "gsm8k-frozen-norm-high6"]


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
    """Read-only LRU cache for shard files. Never writes to disk."""
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


def compare_layers_by_proj(dir_a, dir_b):
    """
    Compare baseline (dir_a) vs variant (dir_b).
    Returns: dict proj -> {layer_i: frac_changed}
    """
    map_a = load_weight_map(dir_a)
    map_b = load_weight_map(dir_b)
    common_keys = sorted(set(map_a.keys()).intersection(map_b.keys()))

    if not common_keys:
        return {}

    cache_a = ShardCache(dir_a)
    cache_b = ShardCache(dir_b)

    data = {}  # (layer, proj) -> [n_changed, n_total]

    for k in common_keys:
        li = layer_index(k)
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

        abs_diff = (t1 - t0).abs()
        changed = abs_diff > CHANGE_THRESH

        key = (li, proj)
        if key not in data:
            data[key] = [0, 0]
        data[key][0] += changed.sum().item()
        data[key][1] += t0.numel()

    result = {p: {} for p in "kqvo"}
    for (li, proj), (n_changed, n_total) in data.items():
        result[proj][li] = n_changed / n_total if n_total > 0 else 0.0

    return result


def _frozen_layers_from_stats(proj_stats, proj):
    """
    Infer which layers were frozen for this experiment/projection:
    any layer where frac_changed == 0.0 exactly.
    Returns a sorted list of layer indices.
    """
    return sorted(
        li for li, v in proj_stats.get(proj, {}).items() if v == 0.0
    )


def _contiguous_spans(layer_indices):
    """
    Convert a list of layer indices into contiguous (start, end) spans
    suitable for axvspan, with ±0.5 padding.
    e.g. [2,3,4,8,9] -> [(1.5, 4.5), (7.5, 9.5)]
    """
    if not layer_indices:
        return []
    spans = []
    start = layer_indices[0]
    prev  = layer_indices[0]
    for li in layer_indices[1:]:
        if li == prev + 1:
            prev = li
        else:
            spans.append((start - 0.5, prev + 0.5))
            start = prev = li
    spans.append((start - 0.5, prev + 0.5))
    return spans


def _auto_ylim(all_series, padding=0.03, min_range=0.1):
    """Auto-scale y-axis from data: keeps 0 as floor, adds padding, ensures min visible range."""
    all_vals = []
    for layers, vals in all_series.values():
        all_vals.extend(vals)
    if not all_vals:
        return 0, 0.6
    y_min_data = min(all_vals)
    y_max_data = max(all_vals)
    y_range = max(min_range, y_max_data - y_min_data + 2 * padding)
    y_min = max(0, y_min_data - padding)
    y_max = min(1, y_min + y_range)
    if y_max - y_min < min_range:
        y_max = min(1, y_min + min_range)
    return y_min, y_max


def _spans_by_freeze_count(all_frozen_layers, criterion, x_max):
    """
    Per layer: count how many experiments (of this criterion) froze it.
    Return contiguous spans with same count: [(start_x, end_x, count), ...].
    Used for gradient background: darker = more experiments froze that layer.
    """
    layer_counts = {}
    for li in range(x_max + 1):
        layer_counts[li] = 0
    for label, frozen in all_frozen_layers.items():
        if criterion not in label:
            continue
        for li in frozen:
            layer_counts[li] = layer_counts.get(li, 0) + 1

    spans = []
    i = 0
    while i <= x_max:
        c = layer_counts[i]
        j = i
        while j <= x_max and layer_counts.get(j, 0) == c:
            j += 1
        if c > 0:  # only draw spans where at least one experiment froze
            spans.append((i - 0.5, j - 0.5, c))
        i = j
    return spans


def _compute_insight(all_series, all_frozen_layers, proj):
    """Compute a 1–2 line insight from the data for annotation."""
    if "full-ft" not in all_series:
        return ""
    ft_layers, ft_vals = all_series["full-ft"]
    ft_mean = sum(ft_vals) / len(ft_vals) if ft_vals else 0

    # Compare freeze variants (unfrozen layers only) to full-ft
    diffs = []
    for label in all_series:
        if "full-ft" in label or "norm" not in label and "wass" not in label:
            continue
        layers, vals = all_series[label]
        frozen = set(all_frozen_layers.get(label, []))
        for li, v in zip(layers, vals):
            if li not in frozen and li in ft_layers:
                ft_v = ft_vals[ft_layers.index(li)]
                diffs.append(abs(v - ft_v))
    avg_diff = sum(diffs) / len(diffs) if diffs else 0

    # Layer pattern: early vs late
    early = [v for l, v in zip(ft_layers, ft_vals) if l <= 10]
    late = [v for l, v in zip(ft_layers, ft_vals) if l >= 21]
    early_mean = sum(early) / len(early) if early else 0
    late_mean = sum(late) / len(late) if late else 0

    parts = []
    if avg_diff < 0.03:
        parts.append("Unfrozen layers track full-ft closely (Δ < 0.03).")
    elif avg_diff < 0.08:
        parts.append(f"Unfrozen layers near full-ft (avg Δ = {avg_diff:.2f}).")
    if early_mean < late_mean * 0.9:
        parts.append("Early layers change less than late.")
    elif late_mean < early_mean * 0.9:
        parts.append("Late layers change less than early.")
    return " ".join(parts) if parts else f"Full-ft mean = {ft_mean:.2f}. Darker bg = more experiments froze layer."


def _short_name(exp):
    if "full-finetuned" in exp or "full-ft" in exp:
        return "full-ft"
    return (exp.replace("gsm8k-frozen-", "")
               .replace("-run1", "").replace("-run2", "").replace("-run3", ""))


# Per-experiment palettes for split plots (distinct shades so lines are distinguishable)
NORM_BLUES = ["#aec6e8", "#7eb8da", "#5b9bd5", "#1f6fb2", "#1565c0", "#0b3d6b"]   # low3, low6, low9, low15, high6, high9
WASS_REDS  = ["#f5b89a", "#e8734a", "#c0392b", "#922b21", "#7b1a10", "#4a0e0a"]   # same order

def _split_plot_color(label):
    """Return a unique color per experiment for split plots. Norm=blue shades, wass=red shades."""
    m = re.search(r"(norm|wass)-(low|high)(\d+)", label)
    if not m:
        return "#808080"
    crit, kind, n = m.group(1), m.group(2), int(m.group(3))
    if crit == "norm":
        idx = {"low": {3: 0, 6: 1, 9: 2, 15: 3}, "high": {6: 4, 9: 5}}.get(kind, {}).get(n, 0)
        return NORM_BLUES[min(idx, len(NORM_BLUES) - 1)]
    else:
        idx = {"low": {3: 0, 6: 1, 9: 2, 15: 3}, "high": {6: 4, 9: 5}}.get(kind, {}).get(n, 0)
        return WASS_REDS[min(idx, len(WASS_REDS) - 1)]


def _plot_style(exp_name):
    """Return (color, linestyle, lw, alpha, zorder)."""
    if "full-ft" in exp_name or "full-finetuned" in exp_name:
        return ("black", "-", 3.5, 0.2, 0)
    if "norm-low" in exp_name:
        return ("#1f77b4", "-",  1.5, 1.0, 2)
    if "norm-high" in exp_name:
        return ("#1f77b4", "--", 1.5, 1.0, 2)
    if "wass-low" in exp_name:
        return ("#d62728", "-",  1.5, 1.0, 2)
    if "wass-high" in exp_name:
        return ("#d62728", "--", 1.5, 1.0, 2)
    return ("gray", "-", 1.0, 0.8, 1)



def _resolve_exp_dir(root_dir, exp):
    """Resolve experiment name to directory containing model weights.
    Tries: legacy layout, new checkpoints/ layout, and run-specific dirs.
    For run dirs with checkpoint-* subdirs, returns the highest checkpoint (final epoch).
    """
    # Legacy: exp at root
    d = os.path.join(root_dir, exp)
    if os.path.isdir(d):
        ckpt = _latest_checkpoint_in_dir(d)
        return ckpt if ckpt else d
    if not exp.endswith("-run3"):
        d3 = os.path.join(root_dir, exp + "-run3")
        if os.path.isdir(d3):
            ckpt = _latest_checkpoint_in_dir(d3)
            return ckpt if ckpt else d3

    # New structure: checkpoints/llama/
    if exp == "gsm8k-full-finetuned-tda":
        nd = os.path.join(root_dir, "checkpoints", "llama", exp)
        if os.path.isdir(nd):
            ckpt = _latest_checkpoint_in_dir(nd)
            return ckpt if ckpt else nd

    # Norm-freeze: checkpoints/llama/norm-freeze/{low-3,low-6,...}/run1
    m = re.match(r"gsm8k-frozen-norm-(low|high)(\d+)", exp)
    if m:
        kind, n = m.group(1), m.group(2)
        sub = f"{kind}-{n}"
        for run in ["run1", "run2", "run3"]:
            nd = os.path.join(root_dir, "checkpoints", "llama", "norm-freeze", sub, run)
            if os.path.isdir(nd):
                ckpt = _latest_checkpoint_in_dir(nd)
                return ckpt if ckpt else nd

    # Wass-freeze: checkpoints/llama/wass-freeze/{low-3-frozen,...,high-9-frozen}/run1
    m = re.match(r"gsm8k-frozen-wass-(low|high)(\d+)", exp)
    if m:
        kind, n = m.group(1), m.group(2)
        sub = f"{kind}-{n}-frozen"
        for run in ["run1", "run2", "run3"]:
            nd = os.path.join(root_dir, "checkpoints", "llama", "wass-freeze", sub, run)
            if os.path.isdir(nd):
                ckpt = _latest_checkpoint_in_dir(nd)
                return ckpt if ckpt else nd
    return None


def _latest_checkpoint_in_dir(d):
    """If d contains checkpoint-* subdirs, return path to the one with highest step."""
    import glob
    ckpts = glob.glob(os.path.join(d, "checkpoint-*"))
    if not ckpts:
        return None
    def step(p):
        try:
            return int(p.split("-")[-1])
        except ValueError:
            return 0
    best = max(ckpts, key=step)
    return best if os.path.isfile(os.path.join(best, "model.safetensors.index.json")) or os.path.isfile(os.path.join(best, "model.safetensors")) else None


def save_kqvo_plot(experiment_name, proj_stats, out_dir):
    """One plot per experiment: 4 lines (K, Q, V, O), x=layer, y=frac_changed."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"k": "#1f77b4", "q": "#ff7f0e", "v": "#2ca02c", "o": "#d62728"}
    for proj in "kqvo":
        if proj not in proj_stats or not proj_stats[proj]:
            continue
        layers = sorted(proj_stats[proj].keys())
        vals = [proj_stats[proj][li] for li in layers]
        ax.plot(layers, vals, color=colors[proj], linewidth=1.5,
                marker="o", markersize=2.5, label=f"{proj.upper()}", zorder=2)

    all_layers = [li for p in proj_stats for li in proj_stats[p]]
    x_max = max(all_layers) if all_layers else 31
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Fraction of weights changed (0–1)", fontsize=11)
    ax.set_title(f"pretrained vs {experiment_name}\nfrac_changed by layer (K,Q,V,O)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{experiment_name}__frac_changed_kqvo.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def save_single_proj_plot(label_a, label_b, proj, layers, vals, out_dir):
    """One PNG for a single projection. Y: 0–1."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, vals, color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    ax.set_xlim(0, max(layers) if layers else 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Fraction of weights changed (0–1)", fontsize=11)
    ax.set_title(f"{label_a}  vs  {label_b}\n{proj.upper()} — frac_changed by layer", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{label_a}_vs_{label_b}__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def save_combined_proj_plot(proj, all_series, all_frozen_layers, out_dir):
    """
    Combined overlay plot for one projection (K, Q, V, or O).
    Lines only — no shading, no annotations.
    Lines are drawn through unfrozen segments only (no sawtooth).
    K and Q: only full-ft line drawn; freeze variants confirmed 0.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    all_x = [li for layers, _ in all_series.values() for li in layers]
    x_min = min(all_x) if all_x else 0
    x_max = max(all_x) if all_x else 31

    def _sort_key(label):
        if "full-ft" in label:
            return (0, 0, label)
        m = re.search(r"(\d+)", label)
        n = int(m.group(1)) if m else 0
        if "norm-low"  in label: return (1, n, label)
        if "norm-high" in label: return (2, n, label)
        if "wass-low"  in label: return (3, n, label)
        if "wass-high" in label: return (4, n, label)
        return (5, 0, label)

    is_kq = proj in ("k", "q")

    for label in sorted(all_series.keys(), key=_sort_key):
        layers, vals = all_series[label]
        color, ls, lw, alpha, zorder = _plot_style(label)
        is_full_ft = "full-ft" in label

        if is_kq and not is_full_ft:
            continue

        frozen_set = set(all_frozen_layers.get(label, []))

        seg_x, seg_y = [], []
        segments_x, segments_y = [], []
        for li, v in zip(layers, vals):
            if li in frozen_set:
                if seg_x:
                    segments_x.append(seg_x)
                    segments_y.append(seg_y)
                    seg_x, seg_y = [], []
            else:
                seg_x.append(li)
                seg_y.append(v)
        if seg_x:
            segments_x.append(seg_x)
            segments_y.append(seg_y)

        first = True
        for sx, sy in zip(segments_x, segments_y):
            ax.plot(sx, sy,
                    color=color, linestyle=ls, linewidth=lw,
                    alpha=alpha, zorder=zorder,
                    marker="o", markersize=2.0 if is_full_ft else 1.5,
                    label=label if first else "_nolegend_")
            first = False

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    y_min, y_max = _auto_ylim(all_series)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Fraction of weights changed (0–1)", fontsize=11)

    proj_label = proj.upper()
    if is_kq:
        subtitle = f"{proj_label} — all freeze variants had 0 weight change"
    else:
        subtitle = "blue=norm, red=wass  |  solid=low, dashed=high"
    ax.set_title(
        f"pretrained vs all experiments — {proj_label} frac_changed by layer\n{subtitle}",
        fontsize=12
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"combined__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved combined plot: {out_path}")


def _compute_insight(all_series, all_frozen_layers):
    """Compute a 1-line insight from the data for annotation."""
    if "full-ft" not in all_series:
        return ""
    ft_layers, ft_vals = all_series["full-ft"]
    ft_mean = sum(ft_vals) / len(ft_vals) if ft_vals else 0
    # Compare freeze variants (unfrozen layers only) to full-ft
    freeze_labels = [l for l in all_series if l != "full-ft"]
    if not freeze_labels:
        return ""
    diffs = []
    for label in freeze_labels:
        layers, vals = all_series[label]
        frozen = set(all_frozen_layers.get(label, []))
        for li, v in zip(layers, vals):
            if li not in frozen and li in ft_layers:
                idx = ft_layers.index(li)
                diffs.append(abs(v - ft_vals[idx]))
    if not diffs:
        return "Insight: Unfrozen layers (lines) show weight change; shaded = frozen (no change)."
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff < 0.05:
        return "Insight: Unfrozen regions closely match full-ft (avg Δ ≈ {:.2f}). Darker shade = more experiments froze that layer.".format(avg_diff)
    return "Insight: Unfrozen mean Δ from full-ft = {:.2f}. Darker background = more experiments froze that layer.".format(avg_diff)


def save_split_proj_plot(proj, all_series, all_frozen_layers, out_dir):
    """
    Option B: side-by-side norm | wass panels for one projection.
    Each panel contains:
      - full-ft reference line (black, transparent)
      - only that criterion's variants (norm or wass), lines through unfrozen segments
      - gradient background: darker = more experiments froze that layer
    Saves as split__{proj}.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

    criteria   = ["norm", "wass"]
    colors     = {"norm": "#1f77b4", "wass": "#d62728"}
    pan_titles = {"norm": "Norm-based freezing", "wass": "Wasserstein-based freezing"}

    all_x = [li for layers, _ in all_series.values() for li in layers]
    x_min = min(all_x) if all_x else 0
    x_max = max(all_x) if all_x else 31

    def _sort_key(label):
        if "full-ft" in label:
            return (0, 0, label)
        m = re.search(r"(\d+)", label)
        n = int(m.group(1)) if m else 0
        if "low"  in label: return (1, n, label)
        if "high" in label: return (2, n, label)
        return (3, 0, label)

    for ax, criterion in zip(axes, criteria):
        color = colors[criterion]

        # Gradient background: darker = more experiments froze this layer (visible alphas)
        for s, e, count in _spans_by_freeze_count(all_frozen_layers, criterion, x_max):
            alpha = 0.18 + 0.10 * min(count - 1, 4)  # 1→0.18, 2→0.28, 3→0.38, 4→0.48, 5+→0.58
            ax.axvspan(s, e, color=color, alpha=min(alpha, 0.6), linewidth=0, zorder=0)

        # full-ft reference
        if "full-ft" in all_series:
            ft_layers, ft_vals = all_series["full-ft"]
            ax.plot(ft_layers, ft_vals, color="black", linewidth=2.5,
                    alpha=0.2, marker="o", markersize=2,
                    label="full-ft", zorder=1)

        # Criterion variants only — each experiment gets a distinct blue/red shade
        crit_labels = [l for l in all_series if criterion in l]
        for label in sorted(crit_labels, key=_sort_key):
            layers, vals = all_series[label]
            _, ls, lw, alpha, zorder = _plot_style(label)
            line_color = _split_plot_color(label)
            frozen_set = set(all_frozen_layers.get(label, []))

            # Split into unfrozen segments
            seg_x, seg_y = [], []
            segments_x, segments_y = [], []
            for li, v in zip(layers, vals):
                if li in frozen_set:
                    if seg_x:
                        segments_x.append(seg_x)
                        segments_y.append(seg_y)
                        seg_x, seg_y = [], []
                else:
                    seg_x.append(li)
                    seg_y.append(v)
            if seg_x:
                segments_x.append(seg_x)
                segments_y.append(seg_y)

            # Short label: e.g. "norm-low6" -> "low6"
            short = re.sub(r"^(norm|wass)-", "", label)
            first = True
            for sx, sy in zip(segments_x, segments_y):
                ax.plot(sx, sy, color=line_color, linestyle=ls, linewidth=lw,
                        alpha=alpha, zorder=zorder, marker="o", markersize=1.5,
                        label=short if first else "_nolegend_")
                first = False

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_xlabel("Layer index", fontsize=11)
        ax.set_title(pan_titles[criterion], fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    y_min, y_max = _auto_ylim(all_series)
    for ax in axes:
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel("Fraction of weights changed (0–1)", fontsize=11)

    proj_label = proj.upper()
    fig.suptitle(
        f"pretrained vs all experiments — {proj_label} frac_changed by layer\n"
        "Background: darker = more experiments froze this layer.",
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"split__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved split plot: {out_path}")


VENV_PYTHON = "/home/kadir/topo/topo-env/bin/python"


def _do_save_pretrained(save_dir, model_id):
    from transformers import AutoModelForCausalLM
    print(f"Saving pretrained {model_id} to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    print(f"  Saved. Use: --pretrained_dir {save_dir}")


def save_pretrained_weights(save_dir, model_id=HF_MODEL_ID):
    if os.path.isfile(VENV_PYTHON):
        import subprocess
        code = (
            "import os\n"
            "from transformers import AutoModelForCausalLM\n"
            f"save_dir = {repr(save_dir)}\n"
            f"model_id = {repr(model_id)}\n"
            "os.makedirs(save_dir, exist_ok=True)\n"
            "model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype='bfloat16', trust_remote_code=True)\n"
            "model.save_pretrained(save_dir, safe_serialization=True)\n"
            f'print("  Saved. Use: --pretrained_dir", save_dir)\n'
        )
        print(f"Saving pretrained {model_id} to {save_dir} (via topo-env)...")
        subprocess.run([VENV_PYTHON, "-c", code], check=True)
    else:
        _do_save_pretrained(save_dir, model_id)


def run_quick_test(root_dir, out_dir, pretrained_dir):
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        raise RuntimeError("Pretrained baseline required. Use --pretrained_dir or --save_pretrained")
    summary = {}
    for exp in QUICK_TEST_EXPERIMENTS:
        exp_dir = os.path.join(root_dir, exp)
        if not os.path.isdir(exp_dir):
            raise RuntimeError(f"Experiment folder not found: {exp_dir}")
        exp_label = _short_name(exp)
        print(f"\n{'='*55}")
        print(f"pretrained vs {exp_label}")
        proj_stats = compare_layers_by_proj(pretrained_dir, exp_dir)
        if proj_stats:
            for proj in "kqvo":
                if proj in proj_stats and proj_stats[proj]:
                    layers = sorted(proj_stats[proj].keys())
                    vals = [proj_stats[proj][li] for li in layers]
                    save_single_proj_plot("pretrained", exp_label, proj, layers, vals, out_dir)
            summary[f"pretrained_vs_{exp_label}"] = proj_stats
    out_json = os.path.join(out_dir, "frac_changed_quick_test.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON: {out_json}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare baseline vs freeze experiments, frac_changed by K,Q,V,O per layer."
    )
    ap.add_argument("--root_dir", default=ROOT_DIR)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--experiments", nargs="+", default=None)
    ap.add_argument("--quick_test", action="store_true", default=RUN_QUICK_TEST)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pretrained_dir", default=None)
    ap.add_argument("--save_pretrained", type=str, metavar="DIR", default=None)
    args = ap.parse_args()
    if args.full:
        args.quick_test = False

    if args.save_pretrained:
        save_pretrained_weights(args.save_pretrained)
        print("Done. Now run with: --pretrained_dir", args.save_pretrained)
        return

    out_dir = args.out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    pretrained_dir = args.pretrained_dir or (PRETRAINED_DIR if os.path.isdir(PRETRAINED_DIR) else None)

    if args.quick_test:
        print(f"Quick test: pretrained vs {len(QUICK_TEST_EXPERIMENTS)} experiments")
        run_quick_test(args.root_dir, out_dir, pretrained_dir=pretrained_dir)
        print("Done.")
        return

    # Full run
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        raise RuntimeError("Pretrained baseline required. Use --pretrained_dir or --save_pretrained")
    print(f"Baseline (pretrained): {pretrained_dir}")
    print(f"Output: {out_dir}\n")

    summary       = {}
    combined      = {p: {} for p in "kqvo"}   # proj -> {label: (layers, vals)}
    frozen_by_exp = {p: {} for p in "kqvo"}   # proj -> {label: [frozen layer indices]}

    exp_list = args.experiments if args.experiments else FULL_EXPERIMENTS
    for exp in exp_list:
        exp_dir = _resolve_exp_dir(args.root_dir, exp)
        if not exp_dir:
            print(f"  [skip] {exp} — folder not found")
            continue
        has_w = (
            os.path.isfile(os.path.join(exp_dir, "model.safetensors.index.json")) or
            os.path.isfile(os.path.join(exp_dir, "model.safetensors"))
        )
        if not has_w:
            print(f"  [skip] {exp} — no weights")
            continue

        exp_label = _short_name(exp)
        print(f"Comparing: pretrained vs {exp_label}")
        proj_stats = compare_layers_by_proj(pretrained_dir, exp_dir)
        if not proj_stats:
            print("  [WARNING] No layer data — skipping.")
            continue

        summary[exp_label] = {p: proj_stats.get(p, {}) for p in "kqvo"}

        for proj in "kqvo":
            if proj in proj_stats and proj_stats[proj]:
                layers = sorted(proj_stats[proj].keys())
                vals   = [proj_stats[proj][li] for li in layers]
                combined[proj][exp_label]      = (layers, vals)
                frozen_by_exp[proj][exp_label] = _frozen_layers_from_stats(proj_stats, proj)

    # Save combined overlay plots (one per projection)
    print(f"\n{'='*55}")
    print("Saving combined overlay plots …")
    for proj in "kqvo":
        series = combined[proj]
        if not series:
            continue
        save_combined_proj_plot(proj, series, frozen_by_exp[proj], out_dir)
        save_split_proj_plot(proj, series, frozen_by_exp[proj], out_dir)

    out_json = os.path.join(out_dir, "frac_changed_kqvo_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()


"""
Usage:

python freeze_kqvo_comparison.py \
  --root_dir "/home/kadir/topo/numpy_weights/exploration-finetuning/" \
  --full \
  --pretrained_dir "/home/kadir/topo/numpy_weights/exploration-finetuning/llama31-8b-pretrained"

Combined output (4 + 4 PNGs):
  combined__k.png        — all variants overlaid, lines only
  combined__q.png
  combined__v.png
  combined__o.png
  split__k.png           — side-by-side norm | wass panels
  split__q.png
  split__v.png
  split__o.png
"""