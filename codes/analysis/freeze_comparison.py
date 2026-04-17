import os
import re
import json
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from safetensors.torch import load_file as safetensors_load_file


RESULTS_DIR = "/home/vepaul/results"


BASELINE_DEFAULT = "gsm8k-full-finetuned-tda"

# instead of these, we will use the split freezing methods
# gsm8k-freese-norm
# gsm8k-freeze-wass
FREEZE_PREFIXES = (
    "gsm8k-freeze-highest3-",
    "gsm8k-freeze-highest6-",
    "gsm8k-freeze-highest9-",
    "gsm8k-freeze-highest12-",
    "gsm8k-freeze-lowest3-",
    "gsm8k-freeze-lowest6-",
    "gsm8k-freeze-lowest9-",
    "gsm8k-freeze-lowest12-",
)

LAYER_RE      = re.compile(r"model\.layers\.(\d+)\.")
EPS           = 1e-6
CHANGE_THRESH = 1e-7

# Total: 100.000
# Change: 10.000
METRICS = [
    ("mean_all_pct",     "Mean |% change| (all weights)"),
    ("frac_changed",     "Fraction of weights that changed"),
    ("mean_changed_pct", "Mean |% change| (changed weights only)"),
]

# ── Colour palettes (light→dark as N increases: 3,6,9,12) ──────────────────
HIGHEST_COLORS = ["#aec6e8", "#5b9bd5", "#1f6fb2", "#0b3d6b"]  # blue shades
LOWEST_COLORS  = ["#f5b89a", "#e8734a", "#c0392b", "#7b1a10"]  # red shades

# Number of frozen layers for each step
FREEZE_NS = [3, 6, 9, 12]


def layer_index(param_name):
    m = LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def load_weight_map(folder):
    index_path  = os.path.join(folder, "model.safetensors.index.json")
    single_path = os.path.join(folder, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)
        wm = idx.get("weight_map")
        if not wm:
            raise RuntimeError(f"No weight_map in {index_path}")
        print(f"    [loader] {len(wm)} keys  <- {os.path.basename(folder)}")
        return wm
    if os.path.isfile(single_path):
        sd = safetensors_load_file(single_path, device="cpu")
        print(f"    [loader] {len(sd)} keys  <- {os.path.basename(folder)}")
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
        sd = safetensors_load_file(path, device="cpu")  # read-only, RAM only
        self.cache[shard] = sd
        self.order.append(shard)
        while len(self.order) > self.max_cached:
            del self.cache[self.order.pop(0)]
        return sd


def compare_layers(dir_a, dir_b, verbose):
    map_a = load_weight_map(dir_a)
    map_b = load_weight_map(dir_b)
    common_keys = sorted(set(map_a.keys()).intersection(map_b.keys()))

    if not common_keys:
        print("    [WARNING] No common keys — cannot compare.")
        return [], {}

    cache_a = ShardCache(dir_a)
    cache_b = ShardCache(dir_b)

    layer_sum_pct, layer_total         = {}, {}
    layer_changed_sum, layer_changed_n = {}, {}

    for k in common_keys:
        li = layer_index(k)
        if li is None:
            continue
        try:
            t0 = cache_a.get(map_a[k])[k].to(torch.float32).reshape(-1)
            t1 = cache_b.get(map_b[k])[k].to(torch.float32).reshape(-1)
        except (FileNotFoundError, KeyError):
            continue
        if t0.shape != t1.shape:
            continue

        abs_diff = (t1 - t0).abs()
        pct      = 100.0 * abs_diff / (t0.abs() + EPS)
        changed  = abs_diff > CHANGE_THRESH

        if li not in layer_sum_pct:
            layer_sum_pct[li] = layer_total[li] = 0
            layer_changed_sum[li] = layer_changed_n[li] = 0

        layer_sum_pct[li]     += pct.sum().item()
        layer_total[li]       += t0.numel()
        layer_changed_n[li]   += changed.sum().item()
        layer_changed_sum[li] += pct[changed].sum().item() if changed.any() else 0.0

    
    xs = sorted(layer_sum_pct.keys())
    stats = {
        li: {
            "mean_all_pct":    layer_sum_pct[li]     / layer_total[li],
            "frac_changed":    layer_changed_n[li]   / layer_total[li],
            "mean_changed_pct":layer_changed_sum[li] / layer_changed_n[li]
                               if layer_changed_n[li] > 0 else 0.0,
        }
        for li in xs
    }

    if verbose and xs:
        for key, label in METRICS:
            vals = [stats[i][key] for i in xs]
            print(f"    {label}: [{min(vals):.3e}, {max(vals):.3e}]")

    return xs, stats


def save_metric_plot(label_a, label_b, xs, stats, metric_key, metric_title, out_dir):
    floor  = 1e-9
    values = [max(stats[i][metric_key], floor) for i in xs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, values, color="steelblue", linewidth=1.8, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.set_title(f"{label_a}  vs  {label_b}\n{metric_title}", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    filename = f"{label_a}_vs_{label_b}__{metric_key}.png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def _parse_variant(variant_name):
    """Return ('highest'|'lowest', N) or (None, None)."""
    m = re.search(r"freeze-(highest|lowest)(\d+)-run", variant_name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def save_combined_metric_plot(run_id, baseline_label, all_results,
                               metric_key, metric_title, out_dir):
    """
    Overlay all freeze variants on one plot, with shaded freeze-region bands.

    all_results: dict  variant_name -> (xs, stats)
    Shading convention (matches reference image):
      - highest-N  -> red band   over layers [0 .. N-1]     (frozen = highest-TDA early layers)
      - lowest-N   -> blue band  over layers [max-N .. max]  (frozen = lowest-TDA late layers)
    Band alpha increases with N so that highest12 is the most opaque shade.
    """
    floor = 1e-9

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Determine x range ─────────────────────────────────────────────────
    all_xs = []
    for xs, _ in all_results.values():
        all_xs.extend(xs)
    if not all_xs:
        plt.close()
        return
    x_min, x_max = min(all_xs), max(all_xs)
    n_layers = x_max + 1          # total number of layers (0-indexed)

    # ── Draw freeze-region shading ─────────────────────────────────────────
    # Alpha ramp: [3,6,9,12] -> [0.10, 0.18, 0.26, 0.34]
    alpha_map = {3: 0.10, 6: 0.18, 9: 0.26, 12: 0.34}

    for n in sorted(FREEZE_NS, reverse=True):   # draw largest (most opaque) first
        a = alpha_map[n]
        # highest-N: freeze the N layers with highest TDA score → left side
        ax.axvspan(x_min - 0.5, x_min + n - 0.5,
                   color="red",  alpha=a, linewidth=0, zorder=0)
        # lowest-N: freeze the N layers with lowest TDA score → right side
        ax.axvspan(x_max - n + 0.5, x_max + 0.5,
                   color="blue", alpha=a, linewidth=0, zorder=0)

    # ── Plot each variant ──────────────────────────────────────────────────
    # Sort so legend is ordered: highest3…12, lowest3…12
    def _sort_key(name):
        kind, n = _parse_variant(name)
        if kind is None:
            return (9, 999)
        return (0 if kind == "highest" else 1, n)

    sorted_variants = sorted(all_results.keys(), key=_sort_key)

    for variant_name in sorted_variants:
        xs, stats = all_results[variant_name]
        if not xs:
            continue
        kind, n = _parse_variant(variant_name)
        if kind is None:
            color, lw = "gray", 1.2
            label = variant_name
        else:
            idx = FREEZE_NS.index(n)
            color = HIGHEST_COLORS[idx] if kind == "highest" else LOWEST_COLORS[idx]
            lw    = 1.2 + idx * 0.25
            label = f"{kind}{n}"

        values = [max(stats[i][metric_key], floor) for i in xs]
        ax.plot(xs, values, color=color, linewidth=lw,
                marker="o", markersize=2.5, label=label, zorder=2)

    ax.set_yscale("log")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.set_title(
        f"{baseline_label}  –  freeze variants (run{run_id})\n{metric_title}",
        fontsize=12
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    # ── Legend: two columns (highest | lowest) ────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
              ncol=2, fontsize=9,
              loc="lower center",
              bbox_to_anchor=(0.5, 0.02),
              framealpha=0.85)

    plt.tight_layout()

    filename = f"freeze_run{run_id}__{metric_key}.png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved combined plot: {out_path}")


def sort_key(name):
    m = re.search(r"freeze-(highest|lowest)(\d+)-run", name)
    if not m:
        return (9, 999)
    return (0 if m.group(1) == "highest" else 1, int(m.group(2)))


def main():
    ap = argparse.ArgumentParser(
        description="Compare full fine-tuned baseline vs each freeze-highest/lowest variant."
    )
    ap.add_argument("--root_dir", required=True,
                    help="Directory containing all run folders")
    ap.add_argument("--baseline", default=BASELINE_DEFAULT,
                    help=f"Baseline folder name (default: {BASELINE_DEFAULT})")
    ap.add_argument("--run_id", type=int, default=1,
                    help="Which run number to use (default: 1)")
    ap.add_argument("--out_dir", default=RESULTS_DIR,
                    help="Where to save outputs (never touches weight folders)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Validate baseline
    baseline_dir = os.path.join(args.root_dir, args.baseline)
    if not os.path.isdir(baseline_dir):
        raise RuntimeError(f"Baseline not found: {baseline_dir}")
    has_w = (
        os.path.isfile(os.path.join(baseline_dir, "model.safetensors.index.json")) or
        os.path.isfile(os.path.join(baseline_dir, "model.safetensors"))
    )
    if not has_w:
        raise RuntimeError(f"No weights found in baseline: {baseline_dir}")
    print(f"Baseline : {args.baseline}")

    # Discover freeze variants for the given run_id
    all_dirs = sorted([
        d for d in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, d))
    ])

    variants = []
    for d in all_dirs:
        if not d.endswith(f"run{args.run_id}"):
            continue
        if not any(d.startswith(p) for p in FREEZE_PREFIXES):
            continue
        folder = os.path.join(args.root_dir, d)
        has_w = (
            os.path.isfile(os.path.join(folder, "model.safetensors.index.json")) or
            os.path.isfile(os.path.join(folder, "model.safetensors"))
        )
        if has_w:
            variants.append(d)
        else:
            print(f"  [skip] {d} — no weights found")

    if not variants:
        raise RuntimeError(
            f"No freeze variants found for run_id={args.run_id}.\n"
            f"Expected folder names matching: gsm8k-freeze-highest/lowest<N>-run{args.run_id}"
        )

    variants.sort(key=sort_key)
    print(f"Variants : {len(variants)} found for run{args.run_id}")
    for v in variants:
        print(f"  {v}")
    print()

    # ── Run baseline vs each variant, collect all results ─────────────────
    summary      = {}
    all_results  = {}   # variant_name -> (xs, stats)

    for variant in variants:
        variant_dir    = os.path.join(args.root_dir, variant)
        baseline_label = args.baseline
        variant_label  = variant

        print(f"\n{'='*55}")
        print(f"Comparing: {baseline_label}  vs  {variant_label}")

        xs, stats = compare_layers(baseline_dir, variant_dir, args.verbose)
        if not xs:
            print("  [WARNING] No layer data — skipping.")
            continue

        all_results[variant_label] = (xs, stats)

        # Per-variant individual plots (unchanged behaviour)
        for metric_key, metric_title in METRICS:
            save_metric_plot(baseline_label, variant_label, xs, stats,
                             metric_key, metric_title, args.out_dir)

        summary[f"{baseline_label}_vs_{variant_label}"] = {
            "layers":          xs,
            "mean_all_pct":    [stats[i]["mean_all_pct"]      for i in xs],
            "frac_changed":    [stats[i]["frac_changed"]       for i in xs],
            "mean_changed_pct":[stats[i]["mean_changed_pct"]   for i in xs],
        }

    # ── Save 3 combined overlay plots (one per metric) ────────────────────
    print(f"\n{'='*55}")
    print("Saving combined overlay plots …")
    for metric_key, metric_title in METRICS:
        save_combined_metric_plot(
            run_id         = args.run_id,
            baseline_label = args.baseline,
            all_results    = all_results,
            metric_key     = metric_key,
            metric_title   = metric_title,
            out_dir        = args.out_dir,
        )

    out_json = os.path.join(args.out_dir, "freeze_comparison_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()


"""
Usage:

python freeze_comparison.py \
  --root_dir "/data/cuneyt-topo/numpy_weights/exploration-finetuning/" \
  --run_id 1 \
  --verbose

Per-variant PNGs (unchanged):
  gsm8k-full-finetuned-tda_vs_gsm8k-freeze-highest3-run1__mean_all_pct.png
  ...

Combined overlay PNGs (3 new files):
  freeze_run1__mean_all_pct.png
  freeze_run1__frac_changed.png
  freeze_run1__mean_changed_pct.png
"""
