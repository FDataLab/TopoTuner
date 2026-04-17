#!/usr/bin/env python3
"""
V/O-only layer orderings from **full finetuning** safetensors, using the **same metric** as
``eltwise_rel_llama_full_layer_epoch_curves.py`` (not L2 relative norm):

  * Default: ``mean( |(W_epoch - W_baseline) / max(|W_baseline|, eps)| )`` per tensor (flattened).
  * ``--signed``: ``mean( (W_epoch - W_baseline) / max(|W_baseline|, eps) )`` (can be **negative**);
    layer order = smallest **|mean|** first (then algebraic tie-break).

Output format matches ``codes/utils/order_layers_by_norm.py`` (``V_ORDERED_LAYERS=(...)`` /
``O_ORDERED_LAYERS=(...)``), plus a comment line with ``layer:value`` pairs in that order.

**Modes**
  * ``final`` — baseline vs **last** ``checkpoint-*`` only.
  * ``avg``   — mean of the above over **all** finetuning epochs under the run dir.

**Defaults**
  * Llama: ``checkpoints/llama/full/run1`` vs ``NW_ROOT/pretrained/llama31-8b``.
  * Qwen-base: ``checkpoints/qwen-base/full/run3`` vs ``NW_ROOT/pretrained/qwen3-8b-base``.

Example::

  python scripts/eltwise_rel_vo_orderings_full_finetune.py --nw-root . --mode both --families all

Writes by default::

  eval/split/weight_vs_baseline/eltwise_rel/full_ft_vo_eltwise_rel_orderings.txt
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_eltwise_module():
    script_dir = Path(__file__).resolve().parent
    p = script_dir / "eltwise_rel_llama_full_layer_epoch_curves.py"
    spec = importlib.util.spec_from_file_location("eltwise_rel_llama_full_layer_epoch_curves", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def full_ft_run_dir(ckpt_root: Path, family: str, er) -> Path:
    rid = er.RUN_ID[family]
    return (ckpt_root / family / "full" / rid).resolve()


def collect_vo_vs_baseline_all_epochs(
    er, baseline: str, run_dir: Path, *, signed: bool = False
) -> Dict[int, Dict[str, Dict[int, float]]]:
    cps = er.discover_epoch_checkpoints(run_dir)
    if not cps:
        return {}
    sample = str(cps[0][1])
    if er._has_adapter_only_checkpoint(sample):
        return {}
    return er.collect_vo_vs_baseline_by_layer_epoch(baseline, cps, signed=signed)


def norms_for_mode(
    layer_data: Dict[int, Dict[str, Dict[int, float]]],
    mode: str,
) -> Dict[str, Dict[int, float]]:
    if not layer_data:
        return {}
    epochs = sorted(layer_data.keys())
    if not epochs:
        return {}
    projs = ("v", "o")
    out: Dict[str, Dict[int, float]] = {p: {} for p in projs}

    if mode == "final":
        ep = max(epochs)
        for proj in projs:
            by_layer = layer_data[ep].get(proj, {})
            for li, val in by_layer.items():
                out[proj][li] = float(val)
        return out

    if mode == "avg":
        layers_by_proj: Dict[str, set] = {p: set() for p in projs}
        for ep in epochs:
            for proj in projs:
                layers_by_proj[proj].update(layer_data[ep].get(proj, {}).keys())
        for proj in projs:
            for li in sorted(layers_by_proj[proj]):
                vals = []
                for ep in epochs:
                    d = layer_data[ep].get(proj, {})
                    if li in d:
                        vals.append(float(d[li]))
                if vals:
                    out[proj][li] = float(sum(vals) / len(vals))
        return out

    raise ValueError(mode)


def order_layers(norms_by_layer: Dict[int, float]) -> Tuple[List[int], Dict[int, float]]:
    ordered = sorted(norms_by_layer.keys(), key=lambda li: norms_by_layer[li])
    return ordered, norms_by_layer


def order_layers_signed_least_magnitude_first(
    signed_by_layer: Dict[int, float],
) -> Tuple[List[int], Dict[int, float]]:
    """Smallest |mean signed| first (then more negative mean, for tie-break)."""
    ordered = sorted(
        signed_by_layer.keys(),
        key=lambda li: (abs(signed_by_layer[li]), signed_by_layer[li]),
    )
    return ordered, signed_by_layer


def write_section(
    fh,
    label: str,
    family: str,
    mode: str,
    baseline: str,
    run_dir: Path,
    results: Dict[str, Tuple[List[int], Dict[int, float]]],
    *,
    signed: bool = False,
) -> None:
    fh.write(f"{label}\n")
    if signed:
        fh.write(
            "# Signed eltwise (same denominator as abs): "
            "mean((W_e - W_0) / max(|W_0|, eps)) — can be negative.\n"
            "# Layer order: smallest |mean signed| first (then algebraic tie-break).\n"
        )
    else:
        fh.write(
            "# Eltwise rel (same as eltwise_rel_llama_full_layer_epoch_curves.py): "
            "mean(|W_e - W_0| / max(|W_0|, eps))\n"
        )
    fh.write(f"# Family: {family}  mode: {mode}\n")
    fh.write(f"# Baseline: {baseline}\n")
    fh.write(f"# Run dir:  {run_dir}\n")
    fh.write("\n")
    for proj in ("v", "o"):
        if proj not in results:
            continue
        ordered, vals = results[proj]
        fh.write(f"{proj.upper()}_ORDERED_LAYERS=({' '.join(map(str, ordered))})\n")
        pairs = " ".join(f"{li}:{vals[li]:.6e}" for li in ordered)
        fh.write(f"# {proj.upper()}_SIGNED_MEAN_IN_ORDER=({pairs})\n" if signed else f"# {proj.upper()}_ABS_MEAN_IN_ORDER=({pairs})\n")
    fh.write("\n")


def run_one_family(
    er,
    family: str,
    nw: Path,
    ckpt_root: Path,
    baseline_override: Optional[str],
    mode: str,
    *,
    signed: bool = False,
) -> Optional[Dict[str, Tuple[List[int], Dict[int, float]]]]:
    baseline = er.pick_pretrained_dir(family, baseline_override, nw)
    if not baseline:
        print(f"[skip] no baseline for {family}", flush=True)
        return None
    run_dir = full_ft_run_dir(ckpt_root, family, er)
    if not run_dir.is_dir():
        print(f"[skip] missing run dir: {run_dir}", flush=True)
        return None
    layer_data = collect_vo_vs_baseline_all_epochs(er, baseline, run_dir, signed=signed)
    if not layer_data:
        print(f"[skip] no V/O layer data: {run_dir}", flush=True)
        return None
    norms = norms_for_mode(layer_data, mode)
    results: Dict[str, Tuple[List[int], Dict[int, float]]] = {}
    for proj in ("v", "o"):
        if proj not in norms or not norms[proj]:
            continue
        results[proj] = (
            order_layers_signed_least_magnitude_first(norms[proj])
            if signed
            else order_layers(norms[proj])
        )
        lo, hi = results[proj][0][0], results[proj][0][-1]
        print(
            f"  {family} {mode} {proj.upper()}: least_changed_layer={lo} most_changed_layer={hi} "
            f"(n_layers={len(results[proj][0])})",
            flush=True,
        )
    return results


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    nw_default = script_dir.parent
    default_out = (
        nw_default
        / "eval"
        / "split"
        / "weight_vs_baseline"
        / "eltwise_rel"
        / "full_ft_vo_eltwise_rel_orderings.txt"
    )
    default_out_signed = (
        nw_default
        / "eval"
        / "split"
        / "weight_vs_baseline"
        / "eltwise_rel"
        / "full_ft_vo_eltwise_signed_rel_orderings.txt"
    )

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nw-root", type=Path, default=nw_default)
    ap.add_argument("--ckpt-root", type=Path, default=None, help="Default: NW_ROOT/checkpoints")
    ap.add_argument(
        "--mode",
        choices=("final", "avg", "both"),
        default="both",
        help="final: last epoch only; avg: mean over epochs; both: Llama/Qwen × final/avg.",
    )
    ap.add_argument("--families", choices=("llama", "qwen-base", "all"), default="all")
    ap.add_argument(
        "--signed",
        action="store_true",
        help="Signed mean (W−W0)/max(|W0|,ε) per layer; order by smallest |mean| first. "
        "Default --out becomes full_ft_vo_eltwise_signed_rel_orderings.txt unless overridden.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output .txt path (default depends on --signed)")
    ap.add_argument("--baseline-llama", type=str, default=None)
    ap.add_argument("--baseline-qwen-base", type=str, default=None)
    args = ap.parse_args()

    nw = args.nw_root.resolve()
    ckpt_root = (args.ckpt_root or (nw / "checkpoints")).resolve()
    out_path = (args.out or (default_out_signed if args.signed else default_out)).resolve()
    er = _load_eltwise_module()

    fams: List[str] = ["llama", "qwen-base"] if args.families == "all" else [args.families]
    modes: List[str] = ["final", "avg"] if args.mode == "both" else [args.mode]

    baseline_overrides = {
        "llama": args.baseline_llama,
        "qwen-base": args.baseline_qwen_base,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        if args.signed:
            fh.write(
                "# V/O layer orderings — full finetuning — **signed** eltwise mean vs pretrained\n"
                "# Metric: mean((W - W0) / max(|W0|, eps)) — can be negative.\n"
                "# Order: smallest |mean| first (then algebraic); see eltwise_rel_llama_full_layer_epoch_curves.py\n\n"
            )
        else:
            fh.write(
                "# V/O layer orderings — full finetuning — eltwise relative norm vs pretrained\n"
                "# Order: smallest metric first (least changed layer) → largest (most changed).\n"
                "# Metric: mean(|W - W0| / max(|W0|, eps)) — see eltwise_rel_llama_full_layer_epoch_curves.py\n\n"
            )
        for fam in fams:
            bl = baseline_overrides.get(fam)
            run_dir = full_ft_run_dir(ckpt_root, fam, er)
            base = er.pick_pretrained_dir(fam, bl, nw)
            if not base:
                continue
            for md in modes:
                label = f"========== {fam} — full FT — {md} (V/O, eltwise rel vs baseline) =========="
                res = run_one_family(er, fam, nw, ckpt_root, bl, md, signed=args.signed)
                if not res:
                    continue
                write_section(fh, label, fam, md, base, run_dir, res, signed=args.signed)

    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
