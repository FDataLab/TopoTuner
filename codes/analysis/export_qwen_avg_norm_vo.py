#!/usr/bin/env python3
"""
Rebuild Qwen V/O **avg normalized L2** scalars using the same logic as
``codes/utils/order_layers_by_norm.py`` (baseline ``numpy_weights`` vs ``epoch_*``).

``layer_orderings_norm.txt`` was produced from this computation but only stores
**order**, not per-layer values. ``scripts/joint_wass_norm_layer_selection.py`` reads
``analysis/tda/gsm8k-qwen-base-tda-results/avg_norm_vo.json`` (this script’s default
output) when ``l2_results.csv`` is absent.

Default layout matches ``layer_orderings_norm.txt`` headers:
  baseline/.../numpy_weights + epoch_1..epoch_N under the TDA root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent


def _codes_utils_candidates() -> list[Path]:
    # exploration-finetuning -> numpy_weights -> topo (sibling of codes/)
    nw = _REPO.parent
    topo = nw.parent
    return [
        topo / "codes" / "utils",
        Path("/home/kadir/topo/codes/utils"),
        Path("/data/cuneyt-topo/codes/utils"),
    ]


def _resolve_codes_utils() -> Path:
    marker = "order_layers_by_norm.py"
    for c in _codes_utils_candidates():
        if (c / marker).is_file():
            return c
    tried = ", ".join(str(x) for x in _codes_utils_candidates())
    sys.exit(f"Cannot find codes/utils with {marker}. Tried: {tried}")


_CODES_UTILS = _resolve_codes_utils()
sys.path.insert(0, str(_CODES_UTILS))

from order_layers_by_norm import order_from_norm_avg  # noqa: E402

DEFAULT_TDA = _REPO / "analysis/tda/gsm8k-qwen-base-tda-results"
DEFAULT_OUT = DEFAULT_TDA / "avg_norm_vo.json"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tda-dir",
        type=Path,
        default=DEFAULT_TDA,
        help="GSM8K Qwen-base TDA root (contains baseline/ and epoch_*/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"JSON path (default: <tda-dir>/avg_norm_vo.json).",
    )
    args = p.parse_args()
    tda = args.tda_dir.resolve()
    baseline_nw = tda / "baseline" / "numpy_weights"
    if not baseline_nw.is_dir():
        sys.exit(f"Missing baseline weights: {baseline_nw}")

    results, meta = order_from_norm_avg(
        str(baseline_nw),
        str(tda),
        projections=["v", "o"],
    )
    out_path = (args.output or (tda / "avg_norm_vo.json")).resolve()
    block: dict[str, dict[str, float]] = {"V": {}, "O": {}}
    for upper, lower in (("V", "v"), ("O", "o")):
        _, norms = results[lower]
        for layer, val in sorted(norms.items()):
            block[upper][str(layer)] = float(val)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(block, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  meta: {meta}")


if __name__ == "__main__":
    main()
