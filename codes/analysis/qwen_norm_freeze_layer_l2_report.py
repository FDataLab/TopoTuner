#!/usr/bin/env python3
"""
Per-layer normalized L2 weight change for Qwen-base norm-freeze (run3),
matching analysis/freeze_comparison/freeze_comparison_vepaul/plot_l2norm_heatmap.py.

For each variant (low/high × k in {3,6,9,12,15}):
  - pretrained vs latest checkpoint weights
  - V and O projections per layer (same metric as heatmaps)
  - expected frozen layers from EXPERIMENT_LAYER_ORDER (v/o may differ)
  - ranking layers by mean(V,O) ascending (lowest change first)

Writes a single .txt report (default under eval/split/weight_vs_baseline/norm/).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_plot_module():
    path = (
        REPO_ROOT
        / "analysis"
        / "freeze_comparison"
        / "freeze_comparison_vepaul"
        / "plot_l2norm_heatmap.py"
    )
    spec = importlib.util.spec_from_file_location("plot_l2norm_heatmap", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["plot_l2norm_heatmap"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pretrained_dir",
        default=None,
        help="Override pretrained weights dir (default: plot module PRETRAINED_DIRS)",
    )
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "eval"
            / "split"
            / "weight_vs_baseline"
            / "norm"
            / "qwen_base_normfreeze_vo_per_layer_run3.txt"
        ),
    )
    ap.add_argument("--eps_warn", type=float, default=1e-9)
    args = ap.parse_args()

    plot = _load_plot_module()
    pretrained = args.pretrained_dir or plot.PRETRAINED_DIRS.get("qwen-base")
    if not pretrained or not os.path.isdir(pretrained):
        raise SystemExit(f"pretrained dir missing: {pretrained!r}")

    num_layers = plot.num_hidden_layers_from_pretrained(pretrained)
    if num_layers is None:
        num_layers = plot.MODEL_NUM_LAYERS.get("qwen-base", 36)

    variants = [f"low-{k}" for k in (3, 6, 9, 12, 15)] + [f"high-{k}" for k in (3, 6, 9, 12, 15)]

    lines: list[str] = []
    lines.append("Qwen-base norm-freeze: per-layer normalized L2 vs pretrained")
    lines.append(
        "Metric: ||W - W0||_2 / (||W0||_2 + eps)  (same as plot_l2norm_heatmap.py)"
    )
    lines.append(f"Pretrained: {pretrained}")
    lines.append(f"num_hidden_layers: {num_layers} (indices 0..{num_layers - 1})")
    lines.append(f"Run preference: {plot.RUN_PREFERENCE_BY_MODEL.get('qwen-base', [])}")
    lines.append("")

    for variant in variants:
        exp_dir = plot._resolve_exp_dir("qwen-base", "norm", variant)
        lines.append("=" * 72)
        lines.append(f"experiment: norm-{variant}  (folder {variant}/run*)")
        if not exp_dir:
            lines.append("  [SKIP] weights dir not found")
            lines.append("")
            continue
        lines.append(f"  finetuned weights: {exp_dir}")

        exp_result = plot.compute_normalized_l2(pretrained, exp_dir)
        ev = plot._get_expected_frozen_layers("qwen-base", "norm", variant, "v")
        eo = plot._get_expected_frozen_layers("qwen-base", "norm", variant, "o")
        fv = set(plot.sanitize_layer_indices(ev, num_layers, context=f" {variant} v") or [])
        fo = set(plot.sanitize_layer_indices(eo, num_layers, context=f" {variant} o") or [])

        side, k = plot._variant_low_high_k(variant)
        side = side or "?"
        lines.append(
            f"  expected frozen V ({side}: "
            f"{'first' if side == 'low' else 'last'} {k} in EXPERIMENT_LAYER_ORDER['v']): "
            f"{sorted(fv)}"
        )
        lines.append(
            f"  expected frozen O ({side}: "
            f"{'first' if side == 'low' else 'last'} {k} in EXPERIMENT_LAYER_ORDER['o']): "
            f"{sorted(fo)}"
        )
        lines.append("")

        header = (
            "layer | normL2_V | normL2_O | mean_V_O | exp_frozen_V | exp_frozen_O | "
            "nonzero_when_frozen_V | nonzero_when_frozen_O"
        )
        lines.append(header)
        lines.append("-" * len(header))

        layer_scores: list[tuple[float, int]] = []
        for li in range(num_layers):
            vv = exp_result.get("v", {}).get(li, float("nan"))
            oo = exp_result.get("o", {}).get(li, float("nan"))
            if math.isnan(vv) and math.isnan(oo):
                mean_vo = float("nan")
            elif math.isnan(vv):
                mean_vo = oo
            elif math.isnan(oo):
                mean_vo = vv
            else:
                mean_vo = 0.5 * (vv + oo)

            in_v = li in fv
            in_o = li in fo
            bad_v = in_v and (not math.isnan(vv)) and abs(vv) > args.eps_warn
            bad_o = in_o and (not math.isnan(oo)) and abs(oo) > args.eps_warn
            lines.append(
                f"{li:5d} | {vv:10.6e} | {oo:10.6e} | {mean_vo:10.6e} | "
                f"{'Y' if in_v else 'N':^12} | {'Y' if in_o else 'N':^12} | "
                f"{'Y' if bad_v else 'N':^21} | {'Y' if bad_o else 'N':^21}"
            )
            if not math.isnan(mean_vo):
                layer_scores.append((mean_vo, li))

        layer_scores.sort(key=lambda t: (t[0], t[1]))
        lines.append("")
        lines.append("Ranking (lowest mean(V,O) -> highest):")
        for rank, (score, li) in enumerate(layer_scores, start=1):
            lines.append(f"  {rank:2d}. layer {li:2d}  mean(V,O)={score:.6e}")
        lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
