#!/usr/bin/env python3
"""
Order layers by Wasserstein distance for Q, K, V, O projections.

Two modes:
  1. --csv <wasserstein_results.csv>  (reads CSV from compute_wasserstein.py direct mode)
  2. Legacy mode using generate_wasserstein_boxplots_plotly_prior (old pipeline)

The "lowest" layers have the smallest Wasserstein distance (least change),
and "highest" layers have the largest Wasserstein distance (most change).
"""

import os
import re
import argparse
import pandas as pd


def order_from_csv(csv_path, projections=None, wasserstein_type="H0", output_file=None):
    """Order layers from a Wasserstein CSV produced by compute_wasserstein.py --baseline-dir mode.

    Expected CSV columns: Type, File, Wasserstein H0, Wasserstein H1, Epoch, Projection
    The 'File' column has names like 'layer5_o.pkl' from which we extract the layer index.
    """
    df = pd.read_csv(csv_path)
    if projections is None:
        projections = sorted(df["Projection"].unique()) if "Projection" in df.columns else ["q", "k", "v", "o"]

    wass_col = f"Wasserstein {wasserstein_type}"
    if wass_col not in df.columns:
        print(f"⚠️ Column '{wass_col}' not found. Available: {list(df.columns)}")
        return {}

    results = {}
    for proj in projections:
        sub = df[df["Projection"] == proj].copy() if "Projection" in df.columns else df
        if sub.empty:
            print(f"  ⚠️ No data for projection {proj}")
            continue

        # Extract layer index from File column (e.g. layer5_o.pkl -> 5)
        sub = sub.copy()
        sub["Layer"] = sub["File"].apply(lambda f: int(re.search(r"layer(\d+)", f).group(1)))

        # Average Wasserstein distance per layer across all epochs
        layer_avg = sub.groupby("Layer")[wass_col].mean().sort_values()
        ordered = layer_avg.index.tolist()
        results[proj] = ordered

        print(f"\n{proj.upper()} layers (lowest → highest Wasserstein {wasserstein_type}):")
        print(f"  {ordered}")
        print(f"  Distances: {[round(layer_avg[l], 6) for l in ordered]}")

    # Save to file
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"# Layer orderings by Wasserstein distance (lowest to highest)\n")
            f.write(f"# Source: {csv_path}\n")
            f.write(f"# WassersteinType: {wasserstein_type}\n\n")
            for proj in sorted(results.keys()):
                ordered = results[proj]
                f.write(f"{proj.upper()}_ORDERED_LAYERS=({' '.join(map(str, ordered))})\n")
        print(f"\n✅ Saved layer orderings to: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Order layers by Wasserstein distance")
    parser.add_argument("--csv", required=True,
                        help="Path to Wasserstein results CSV from compute_wasserstein.py")
    parser.add_argument("--projections", default=None,
                        help="Comma-separated projections to order (default: all found in CSV)")
    parser.add_argument("--wasserstein-type", default="H0", choices=["H0", "H1"])
    parser.add_argument("--output", default=None,
                        help="Output file for layer orderings (default: print only)")
    args = parser.parse_args()

    projs = [p.strip() for p in args.projections.split(",")] if args.projections else None
    order_from_csv(args.csv, projections=projs, wasserstein_type=args.wasserstein_type,
                   output_file=args.output)
