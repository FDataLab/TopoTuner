"""
Compare baseline (pretrained) vs finetuned experiments.
- Baseline: pretrained (same for all experiments)
- Experiments: full finetuning + selective-freeze variants, all at final epoch
- Only frac_changed (fraction of weights that changed)
- Splits by K, Q, V, O per layer.
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

# Projection types: match param names
PROJ_PATTERNS = {
    "k": ".k_proj.",
    "q": ".q_proj.",
    "v": ".v_proj.",
    "o": ".o_proj.",
}

ROOT_DIR = "/home/kadir/topo/numpy_weights/exploration-finetuning"
OUT_DIR = "/home/kadir/topo/numpy_weights/exploration-finetuning/freeze_comparison_vepaul/freeze_comparison_frac"  # All outputs (plots, JSON) go here
PRETRAINED_DIR = "/home/kadir/topo/numpy_weights/exploration-finetuning/llama31-8b-pretrained"  # Same baseline for all experiments
# Quick test: pretrained vs 2 experiments (full-ft, norm-high6) → 8 PNGs
RUN_QUICK_TEST = True

QUICK_TEST_EXPERIMENTS = ["gsm8k-full-finetuned-tda", "gsm8k-frozen-norm-high6"]
HF_MODEL_ID = "meta-llama/Llama-3.1-8B"

# All experiments for combined plots (tries both with and without -run3)
FULL_EXPERIMENTS = [
    "gsm8k-full-finetuned-tda",
    "gsm8k-frozen-norm-low3", "gsm8k-frozen-norm-low6", "gsm8k-frozen-norm-low9", "gsm8k-frozen-norm-low15",
    "gsm8k-frozen-norm-high6", "gsm8k-frozen-norm-high9",
    "gsm8k-frozen-wass-low3", "gsm8k-frozen-wass-low6", "gsm8k-frozen-wass-low9", "gsm8k-frozen-wass-low15",
    "gsm8k-frozen-wass-high6", "gsm8k-frozen-wass-high9",
]
# With -run3 suffix (fallback if base not found)
FULL_EXPERIMENTS_RUN3 = [
    "gsm8k-frozen-norm-low3-run3", "gsm8k-frozen-norm-low6-run3", "gsm8k-frozen-norm-low9-run3", "gsm8k-frozen-norm-low15-run3",
    "gsm8k-frozen-norm-high6-run3", "gsm8k-frozen-norm-high9-run3",
    "gsm8k-frozen-wass-low3-run3", "gsm8k-frozen-wass-low6-run3", "gsm8k-frozen-wass-low9-run3", "gsm8k-frozen-wass-low15-run3",
    "gsm8k-frozen-wass-high6-run3", "gsm8k-frozen-wass-high9-run3",
]
EXPERIMENTS = ["gsm8k-full-finetuned-tda", "gsm8k-frozen-norm-high6"]


def layer_index(param_name):
    m = LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def projection_type(param_name):
    """Return 'k','q','v','o' or None if not an attention projection."""
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

    # per (layer, proj): total params, count changed
    data = {}  # (layer, proj) -> (n_changed, n_total)

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
        n_total = t0.numel()
        n_changed = changed.sum().item()

        key = (li, proj)
        if key not in data:
            data[key] = [0, 0]
        data[key][0] += n_changed
        data[key][1] += n_total

    # Build result: proj -> {layer: frac_changed}
    result = {p: {} for p in "kqvo"}
    for (li, proj), (n_changed, n_total) in data.items():
        result[proj][li] = n_changed / n_total if n_total > 0 else 0.0

    return result


def save_kqvo_plot(experiment_name, proj_stats, out_dir):
    """One plot per experiment: 4 lines (K, Q, V, O), x=layer, y=frac_changed (0-1)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"k": "#1f77b4", "q": "#ff7f0e", "v": "#2ca02c", "o": "#d62728"}
    for proj in "kqvo":
        if proj not in proj_stats or not proj_stats[proj]:
            continue
        layers = sorted(proj_stats[proj].keys())
        vals = [proj_stats[proj][li] for li in layers]
        ax.plot(layers, vals, color=colors[proj], linewidth=1.5,
                marker="o", markersize=2.5, label=f"{proj.upper()}", zorder=2)

    all_layers = []
    for p in proj_stats:
        if proj_stats[p]:
            all_layers.extend(proj_stats[p].keys())
    x_max = max(all_layers) if all_layers else 31
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Fraction of weights changed (0–1)", fontsize=11)
    ax.set_title(f"pretrained vs {experiment_name}\nfrac_changed by layer (K,Q,V,O)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    filename = f"{experiment_name}__frac_changed_kqvo.png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def save_single_proj_plot(label_a, label_b, proj, layers, vals, out_dir, suffix=""):
    """Save one PNG for a single projection (K, Q, V, or O). Y: 0-1 (percentage)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, vals, color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    ax.set_xlim(0, max(layers) if layers else 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Fraction of weights changed (0–1)", fontsize=11)
    ax.set_title(f"{label_a}  vs  {label_b}\n{proj.upper()} — frac_changed by layer", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    name = f"{label_a}_vs_{label_b}__{proj}{suffix}.png"
    out_path = os.path.join(out_dir, name)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


VENV_PYTHON = "/home/kadir/topo/topo-env/bin/python"  # Use for --save_pretrained (needs transformers/Llama)


def _do_save_pretrained(save_dir, model_id):
    """Inner: load and save. Called directly if imports work, else via subprocess."""
    from transformers import AutoModelForCausalLM
    print(f"Saving pretrained {model_id} to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    print(f"  Saved. Use: --pretrained_dir {save_dir}")


def save_pretrained_weights(save_dir, model_id=HF_MODEL_ID):
    """Download pretrained model and save to safetensors for use as baseline.
    Uses topo-env if available (has transformers/Llama); else tries current Python.
    """
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


def _short_name(exp):
    """e.g. gsm8k-frozen-norm-high6 -> norm-high6, gsm8k-full-finetuned-tda -> full-ft"""
    if "full-finetuned" in exp or "full-ft" in exp:
        return "full-ft"
    for p in ["norm-high", "norm-low", "wass-high", "wass-low"]:
        if p in exp:
            return exp.replace("gsm8k-frozen-", "").replace("-run1", "").replace("-run2", "").replace("-run3", "")
    return exp.replace("gsm8k-frozen-", "").replace("gsm8k-", "")


def _plot_style(exp_name):
    """Return (color, linestyle, lw, alpha, zorder) for combined plot.
    Full-ft: black, thick, transparent, background.
    Norm: blue (low=solid, high=dashed).
    Wass: red (low=solid, high=dashed).
    """
    if "full-ft" in exp_name or "full-finetuned" in exp_name:
        return ("black", "-", 3.5, 0.2, 0)  # more transparent so others are visible
    if "norm-low" in exp_name:
        return ("#1f77b4", "-", 1.5, 1.0, 2)   # blue solid
    if "norm-high" in exp_name:
        return ("#1f77b4", "--", 1.5, 1.0, 2)  # blue dashed
    if "wass-low" in exp_name:
        return ("#d62728", "-", 1.5, 1.0, 2)   # red solid
    if "wass-high" in exp_name:
        return ("#d62728", "--", 1.5, 1.0, 2)  # red dashed
    return ("gray", "-", 1.0, 0.8, 1)


def _resolve_exp_dir(root_dir, exp):
    """Try exp, then exp with -run3 if base not found."""
    d = os.path.join(root_dir, exp)
    if os.path.isdir(d):
        return d
    if not exp.endswith("-run3"):
        d3 = os.path.join(root_dir, exp + "-run3")
        if os.path.isdir(d3):
            return d3
    return None


def save_combined_proj_plot(proj, all_series, out_dir):
    """One plot per projection: all experiments overlaid. Full-ft black/thick/transparent, norm=blue, wass=red."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, (layers, vals) in all_series.items():
        color, ls, lw, alpha, zorder = _plot_style(label)
        ms = 2.5 if "full-ft" in label else 1.5
        ax.plot(layers, vals, color=color, linestyle=ls, linewidth=lw, alpha=alpha, zorder=zorder,
                marker="o", markersize=ms, label=label)
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Fraction of weights changed (0–1)", fontsize=11)
    ax.set_title(f"pretrained vs all experiments — {proj.upper()} frac_changed by layer", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"combined__{proj}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def run_quick_test(root_dir, out_dir, pretrained_dir):
    """Pretrained (baseline) vs each quick-test experiment. 4 PNGs per experiment."""
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        raise RuntimeError("Pretrained baseline required. Use --pretrained_dir or --save_pretrained")
    experiments = QUICK_TEST_EXPERIMENTS
    summary = {}
    for exp in experiments:
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
    ap.add_argument("--root_dir", default=ROOT_DIR,
                    help="Root directory containing model folders")
    ap.add_argument("--out_dir", default=None,
                    help=f"Output directory (default: {OUT_DIR})")
    ap.add_argument("--experiments", nargs="+", default=None,
                    help="Override experiment list (default: hardcoded)")
    ap.add_argument("--quick_test", action="store_true", default=RUN_QUICK_TEST,
                    help="Quick test: pretrained vs 2 experiments")
    ap.add_argument("--full", action="store_true",
                    help="Run full experiment list (overrides --quick_test)")
    ap.add_argument("--pretrained_dir", default=None,
                    help=f"Pretrained baseline (same for all experiments). Default: {PRETRAINED_DIR}")
    ap.add_argument("--save_pretrained", type=str, metavar="DIR", default=None,
                    help="Download and save pretrained Llama-3.1-8B to DIR, then exit. Use that path for --pretrained_dir")
    args = ap.parse_args()
    if args.full:
        args.quick_test = False

    # Save pretrained and exit
    if args.save_pretrained:
        save_pretrained_weights(args.save_pretrained)
        print("Done. Now run with: --pretrained_dir", args.save_pretrained)
        return

    out_dir = args.out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    pretrained_dir = args.pretrained_dir or (PRETRAINED_DIR if os.path.isdir(PRETRAINED_DIR) else None)

    if args.quick_test:
        print(f"Quick test: pretrained vs {len(QUICK_TEST_EXPERIMENTS)} experiments")
        print(f"  Root: {args.root_dir}")
        print(f"  Baseline (pretrained): {pretrained_dir}")
        print(f"  Output: {out_dir}")
        run_quick_test(args.root_dir, out_dir, pretrained_dir=pretrained_dir)
        print("Done.")
        return

    # Full run: pretrained vs all experiments → 4 combined plots (K, Q, V, O)
    if not pretrained_dir or not os.path.isdir(pretrained_dir):
        raise RuntimeError("Pretrained baseline required for full run. Use --pretrained_dir or --save_pretrained")
    print(f"Baseline (pretrained): {pretrained_dir}")
    print(f"Output: {out_dir}\n")

    summary = {}
    combined = {p: {} for p in "kqvo"}  # proj -> {label: (layers, vals)}

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
                vals = [proj_stats[proj][li] for li in layers]
                combined[proj][exp_label] = (layers, vals)

    # Sort: full-ft first, then norm-low*, norm-high*, wass-low*, wass-high*
    def _sort_key(label):
        if "full-ft" in label:
            return (0, 0, label)
        m = re.search(r"(\d+)", label)
        n = int(m.group(1)) if m else 0
        if "norm-low" in label:
            return (1, n, label)
        if "norm-high" in label:
            return (2, n, label)
        if "wass-low" in label:
            return (3, n, label)
        if "wass-high" in label:
            return (4, n, label)
        return (5, 0, label)

    for proj in "kqvo":
        series = combined[proj]
        if not series:
            continue
        sorted_series = {k: series[k] for k in sorted(series.keys(), key=_sort_key)}
        save_combined_proj_plot(proj, sorted_series, out_dir)

    out_json = os.path.join(out_dir, "frac_changed_kqvo_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
