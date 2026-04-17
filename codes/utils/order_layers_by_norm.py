#!/usr/bin/env python3
"""
Order layers by normalized L2 distance of weight change (baseline vs finetuned).

Formula: ||w_i - w_0||_2 / ||w_0||_2  (flattened weight vectors)

Two modes:
  --mode final: baseline vs final epoch only (e.g. epoch_6)
  --mode avg:   baseline vs each epoch, then average per layer (smoother ranking)

Model/dataset agnostic: works with any checkpoint that produces layer{N}_{q|k|v|o}.npy
(e.g. from extract_o_mlp_weights.py).

Usage:
  # Final only (baseline vs epoch_6):
  python order_layers_by_norm.py --mode final \
    --baseline-dir gsm8k-tda-results/baseline/numpy_weights \
    --final-dir gsm8k-tda-results/epoch_6/numpy_weights \
    --output norm_final.txt

  # Averaged across epochs:
  python order_layers_by_norm.py --mode avg \
    --baseline-dir gsm8k-tda-results/baseline/numpy_weights \
    --epochs-dir gsm8k-tda-results \
    --output norm_avg.txt
"""

import os
import re
import argparse
import numpy as np


def load_layer_weights(npy_dir, proj):
    """Load {layer: weight_matrix} for a projection from numpy_weights dir."""
    result = {}
    suffix = f"_{proj}.npy"
    for fname in os.listdir(npy_dir):
        if fname.endswith(suffix):
            m = re.match(r"layer(\d+)_", fname)
            if m:
                layer = int(m.group(1))
                arr = np.load(os.path.join(npy_dir, fname))
                result[layer] = arr.astype(np.float64)
    return result


def normalized_l2_diff(w_i, w_0, eps=1e-12):
    """Normalized L2: ||w_i - w_0||_2 / (||w_0||_2 + eps)."""
    diff = np.linalg.norm(w_i - w_0)
    base = np.linalg.norm(w_0)
    return diff / (base + eps)


def order_from_norm_final(baseline_dir, final_dir, projections=None, output_file=None, label=None):
    """Order layers by normalized L2 (baseline vs final only)."""
    if projections is None:
        projections = ["q", "k", "v", "o"]

    base_weights = {p: load_layer_weights(baseline_dir, p) for p in projections}
    final_weights = {p: load_layer_weights(final_dir, p) for p in projections}

    results = {}
    for proj in projections:
        b = base_weights[proj]
        f = final_weights[proj]
        common_layers = sorted(set(b) & set(f))
        if not common_layers:
            print(f"  ⚠️ No common layers for projection {proj}")
            continue

        norms = {}
        for layer in common_layers:
            w0 = b[layer].reshape(-1)
            wi = f[layer].reshape(-1)
            norms[layer] = normalized_l2_diff(wi, w0)

        ordered = sorted(norms.keys(), key=lambda l: norms[l])
        results[proj] = (ordered, norms)
        print(f"  {proj.upper()}: least={ordered[0]}, most={ordered[-1]}")
        print(f"    Norms (low→high): {[round(norms[l], 6) for l in ordered[:5]]} ... {[round(norms[l], 6) for l in ordered[-3:]]}")

    return results, {"baseline": baseline_dir, "final": final_dir, "mode": "final"}


def order_from_norm_avg(baseline_dir, epochs_dir, projections=None, output_file=None, label=None):
    """Order layers by avg normalized L2 across finetuning epochs (baseline vs epoch_1, epoch_2, ...).

    Directories named epoch_0 under epochs_dir are ignored (pretrained duplicate of baseline_dir).
    """
    if projections is None:
        projections = ["q", "k", "v", "o"]

    # Discover finetuning epoch subdirs: epoch_1, epoch_2, ... (skip epoch_0 — same as pretrained)
    names = [
        d
        for d in os.listdir(epochs_dir)
        if os.path.isdir(os.path.join(epochs_dir, d)) and re.fullmatch(r"epoch_\d+", d)
    ]

    def _ep_num(name: str) -> int:
        return int(name.split("_", 1)[1])

    names = [d for d in names if _ep_num(d) >= 1]
    names.sort(key=_ep_num)
    epoch_dirs = [os.path.join(epochs_dir, d, "numpy_weights") for d in names]
    epoch_dirs = [d for d in epoch_dirs if os.path.isdir(d)]

    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_*/numpy_weights dirs found in {epochs_dir}")

    base_weights = {p: load_layer_weights(baseline_dir, p) for p in projections}
    epoch_weights = [{p: load_layer_weights(ed, p) for p in projections} for ed in epoch_dirs]

    results = {}
    for proj in projections:
        b = base_weights[proj]
        common_layers = sorted(b.keys())
        for ew in epoch_weights:
            common_layers = sorted(set(common_layers) & set(ew[proj].keys()))

        # For each layer: average normalized L2 across epochs
        norms_avg = {}
        for layer in common_layers:
            w0 = b[layer].reshape(-1)
            vals = []
            for ew in epoch_weights:
                wi = ew[proj][layer].reshape(-1)
                vals.append(normalized_l2_diff(wi, w0))
            norms_avg[layer] = float(np.mean(vals))

        ordered = sorted(norms_avg.keys(), key=lambda l: norms_avg[l])
        results[proj] = (ordered, norms_avg)
        print(f"  {proj.upper()}: least={ordered[0]}, most={ordered[-1]} (avg over {len(epoch_dirs)} epochs)")
        print(f"    Norms (low→high): {[round(norms_avg[l], 6) for l in ordered[:5]]} ... {[round(norms_avg[l], 6) for l in ordered[-3:]]}")

    return results, {"baseline": baseline_dir, "epochs": len(epoch_dirs), "mode": "avg"}


def write_orderings(results, meta, output_file=None, append_to=None, section_label=None):
    """Write orderings to file in standard format."""
    lines = []
    if section_label:
        lines.append(section_label)
    lines.append("# Normalized L2: ||w_i - w_0||_2 / ||w_0||_2")
    lines.append(f"# Mode: {meta['mode']}")
    if "final" in meta:
        lines.append(f"# Baseline: {meta['baseline']}")
        lines.append(f"# Final:    {meta['final']}")
    else:
        lines.append(f"# Baseline: {meta['baseline']}")
        lines.append(f"# Epochs:   {meta['epochs']} (baseline vs each, averaged)")
    lines.append("")
    for proj in sorted(results.keys()):
        ordered, _ = results[proj]
        lines.append(f"{proj.upper()}_ORDERED_LAYERS=({' '.join(map(str, ordered))})")
    lines.append("")

    content = "\n".join(lines)
    if append_to:
        with open(append_to, "a") as f:
            f.write(content)
        print(f"  ✅ Appended to {append_to}")
    elif output_file:
        with open(output_file, "w") as f:
            f.write(content)
        print(f"  ✅ Saved to {output_file}")
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Order layers by normalized L2 weight change")
    parser.add_argument("--mode", required=True, choices=["final", "avg"],
                        help="final: baseline vs epoch_N only; avg: baseline vs each epoch, average")
    parser.add_argument("--baseline-dir", required=True,
                        help="Path to baseline numpy_weights")
    parser.add_argument("--final-dir", default=None,
                        help="Path to final numpy_weights (required for --mode final)")
    parser.add_argument("--epochs-dir", default=None,
                        help="Path to dir containing epoch_1, epoch_2, ... (required for --mode avg)")
    parser.add_argument("--projections", default="q,k,v,o",
                        help="Comma-separated projections (default: q,k,v,o)")
    parser.add_argument("--output", default=None,
                        help="Output file (writes or overwrites)")
    parser.add_argument("--append-to", default=None,
                        help="Append to this file (for combined output)")
    parser.add_argument("--label", default=None,
                        help="Section label (for combined output)")
    args = parser.parse_args()

    projs = [p.strip() for p in args.projections.split(",")]

    if args.mode == "final":
        if not args.final_dir:
            parser.error("--final-dir required for --mode final")
        results, meta = order_from_norm_final(
            args.baseline_dir, args.final_dir,
            projections=projs, output_file=None, label=args.label
        )
    else:
        if not args.epochs_dir:
            parser.error("--epochs-dir required for --mode avg")
        results, meta = order_from_norm_avg(
            args.baseline_dir, args.epochs_dir,
            projections=projs, output_file=None, label=args.label
        )

    write_orderings(results, meta, output_file=args.output, append_to=args.append_to, section_label=args.label)
