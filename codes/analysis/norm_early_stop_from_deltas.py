#!/usr/bin/env python3
"""
Early stopping from V/O mean step percentages (same scale as facet CSV).

Each ``deltas[i]`` is 100 * mean_{layers×{v,o}}( ‖W_next - W_prev‖_2 / ‖W_prev‖_2 )
for transition i → i+1 (columns 0-1 … 5-6 in the pivot table).

**choose_norm_stop_epoch** returns the smallest K in 1..6 (train **K** full epochs)
such that:

1. **Cumulative:** sum(deltas[0:K]) / sum(deltas) >= cum_threshold  
   (default 0.9 — 90% of total relative motion is already done.)

2. **Relative tail:** either K == 6, or deltas[K] / deltas[0] <= rel_threshold  
   (the *next* marginal step is small vs the first step; larger rel_threshold
   ⇒ easier to satisfy ⇒ **earlier** stop — “aggressive”.)

3. **Decay (optional):** if decay_threshold is not None and K >= 2, require  
   deltas[K-1] / deltas[K-2] <= decay_threshold  
   (last completed step is at most that fraction of the previous step — curve
   has bent; conservative waits for this slowdown before stopping.)

If no K satisfies all active rules, returns K=6.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def choose_norm_stop_epoch(
    deltas: List[float],
    cum_threshold: float,
    rel_threshold: float,
    decay_threshold: Optional[float],
) -> Tuple[int, Dict[str, object]]:
    if len(deltas) != 6:
        raise ValueError(f"expected 6 deltas, got {len(deltas)}")
    T = sum(deltas)
    if T <= 0:
        return 6, {"reason": "nonpositive_sum", "T": T}
    d0 = deltas[0] if deltas[0] > 1e-15 else 1e-15

    for k in range(1, 7):
        cum_ok = sum(deltas[:k]) / T >= cum_threshold
        rel_ok = (k >= 6) or (deltas[k] / d0 <= rel_threshold)
        if k < 2 or decay_threshold is None:
            decay_ok = True
        else:
            prev, pr = deltas[k - 1], deltas[k - 2]
            decay_ok = prev / (pr if pr > 1e-15 else 1e-15) <= decay_threshold
        if cum_ok and rel_ok and decay_ok:
            meta: Dict[str, object] = {
                "k": k,
                "cum_frac": sum(deltas[:k]) / T,
                "next_over_first": None if k >= 6 else deltas[k] / d0,
                "last_over_prev": None
                if k < 2
                else deltas[k - 1] / (deltas[k - 2] if deltas[k - 2] > 1e-15 else 1e-15),
            }
            return k, meta

    return 6, {"k": 6, "reason": "fallback", "cum_frac": 1.0}


def get_stop_candidates(
    deltas: List[float],
) -> Tuple[Dict[str, int], Dict[str, object]]:
    candidates: Dict[str, int] = {}

    k1, m1 = choose_norm_stop_epoch(
        deltas,
        cum_threshold=0.90,
        rel_threshold=0.30,
        decay_threshold=None,
    )
    candidates["aggressive"] = k1

    k2, _m2 = choose_norm_stop_epoch(
        deltas,
        cum_threshold=0.90,
        rel_threshold=0.20,
        decay_threshold=None,
    )
    candidates["balanced"] = k2

    k3, _m3 = choose_norm_stop_epoch(
        deltas,
        cum_threshold=0.90,
        rel_threshold=0.20,
        decay_threshold=0.50,
    )
    candidates["conservative"] = k3

    return candidates, m1


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    nw = script_dir.parent
    default_pivot = nw / "eval" / "split" / "weight_vs_baseline" / "norm" / "norm_vo_transition_pct_pivot.csv"
    default_out = nw / "eval" / "split" / "weight_vs_baseline" / "norm" / "norm_vo_early_stop_epochs.csv"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pivot-csv", type=Path, default=default_pivot)
    ap.add_argument("--out-csv", type=Path, default=default_out)
    args = ap.parse_args()

    pivot = args.pivot_csv.resolve()
    cols = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6"]
    rows_out: List[Dict[str, object]] = []

    with open(pivot, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fam = row["family"]
            exp = row["experiment"]
            deltas = [float(row[c]) for c in cols]
            cand, m1 = get_stop_candidates(deltas)
            rows_out.append(
                {
                    "family": fam,
                    "experiment": exp,
                    "aggressive_stop_epoch": cand["aggressive"],
                    "balanced_stop_epoch": cand["balanced"],
                    "conservative_stop_epoch": cand["conservative"],
                    "aggressive_meta_next_over_first": m1.get("next_over_first"),
                    "aggressive_meta_cum_frac": m1.get("cum_frac"),
                }
            )

    out = args.out_csv.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fn = list(rows_out[0].keys()) if rows_out else []
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {out} ({len(rows_out)} rows)\n")
    for r in rows_out:
        print(
            f"{r['family']:10} {r['experiment']:14}  "
            f"aggr={r['aggressive_stop_epoch']}  bal={r['balanced_stop_epoch']}  "
            f"cons={r['conservative_stop_epoch']}"
        )


if __name__ == "__main__":
    main()
