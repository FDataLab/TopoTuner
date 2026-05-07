#!/usr/bin/env python3
"""
Joint Wasserstein + Norm layer selection (AVG): HH/HL/LH/LL × V/O, configurable layer budget (default 15).

Orderings: ``BUILTIN_ORDERINGS`` in this file (V/O, low→high).

Data files (fixed paths under the exploration-finetuning repo):

- Llama: ``analysis/tda/gsm8k-llama-tda-results/wasserstein_results.csv`` and
  ``l2_results.csv`` (columns include ``File`` → ``layer{N}_{q|k|v|o}``, ``Wasserstein H0``,
  and L2: ``Epoch``, ``Layer``, ``Projection``, ``L2_Normalized``).

- Qwen: ``analysis/tda/gsm8k-qwen-base-tda-results/wasserstein_results.csv`` (same Wass schema).
  Norm: ``l2_results.csv`` if present, else ``avg_norm_vo.json`` built by::

      python3 scripts/export_qwen_avg_norm_vo.py

  That JSON must look like ``{"V": {"0": float, ... "35": float}, "O": {...}}`` (36 keys each).

CLI: ``--models``, ``--output-json``, ``--quiet``, plus ``--emit-norm-json-template``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

Projection = Literal["V", "O"]
JointMode = Literal["HH", "HL", "LH", "LL"]

FREEZE_BUDGET = 15


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int


_REPO = Path(__file__).resolve().parent.parent
_LLAMA_TDA = _REPO / "analysis/tda/gsm8k-llama-tda-results"
_QWEN_TDA = _REPO / "analysis/tda/gsm8k-qwen-base-tda-results"

LLAMA_WASS_CSV = _LLAMA_TDA / "wasserstein_results.csv"
LLAMA_L2_CSV = _LLAMA_TDA / "l2_results.csv"
QWEN_WASS_CSV = _QWEN_TDA / "wasserstein_results.csv"
QWEN_L2_CSV = _QWEN_TDA / "l2_results.csv"
# Written by scripts/export_qwen_avg_norm_vo.py (same avg norm L2 as layer_orderings_norm.txt).
QWEN_NORM_SIDECAR = _QWEN_TDA / "avg_norm_vo.json"

# V/O, low → high metric (snapshot of current analysis/tda GSM8K AVG txts).
BUILTIN_ORDERINGS: dict[str, dict[str, dict[Projection, list[int]]]] = {
    "llama": {
        "wass": {
            "V": [29, 26, 30, 28, 31, 27, 25, 22, 23, 15, 24, 21, 17, 3, 1, 19, 20, 14, 9, 4, 12, 18, 13, 16, 5, 6, 10, 11, 7, 8, 2, 0],
            "O": [31, 30, 26, 29, 25, 24, 27, 21, 23, 28, 22, 20, 16, 4, 19, 17, 14, 3, 18, 15, 1, 6, 7, 2, 12, 8, 5, 9, 0, 11, 10, 13],
        },
        "norm": {
            "V": [30, 31, 29, 28, 27, 26, 25, 24, 23, 22, 12, 21, 9, 4, 20, 17, 15, 1, 19, 13, 16, 18, 14, 3, 11, 8, 6, 10, 7, 5, 0, 2],
            "O": [31, 30, 29, 28, 27, 26, 25, 24, 4, 12, 9, 8, 11, 15, 7, 13, 16, 22, 23, 17, 14, 10, 18, 19, 6, 21, 20, 3, 5, 1, 2, 0],
        },
    },
    "qwen": {
        "wass": {
            "V": [25, 10, 4, 6, 12, 17, 8, 15, 7, 19, 26, 34, 24, 28, 30, 2, 22, 27, 18, 23, 9, 14, 20, 11, 16, 33, 35, 32, 3, 21, 13, 31, 29, 1, 5, 0],
            "O": [6, 34, 29, 25, 28, 4, 30, 26, 31, 32, 9, 35, 27, 24, 5, 33, 8, 3, 21, 10, 7, 1, 12, 23, 2, 11, 0, 20, 13, 22, 16, 14, 19, 17, 18, 15],
        },
        "norm": {
            "V": [8, 12, 11, 10, 9, 13, 7, 14, 15, 17, 16, 18, 6, 4, 3, 33, 19, 5, 34, 25, 20, 31, 28, 32, 21, 2, 30, 26, 22, 35, 27, 23, 1, 24, 29, 0],
            "O": [10, 8, 12, 9, 11, 7, 4, 6, 3, 2, 14, 5, 17, 13, 16, 33, 19, 18, 23, 31, 1, 15, 24, 25, 26, 32, 22, 27, 30, 34, 29, 20, 28, 21, 35, 0],
        },
    },
}


MODEL_SPECS: dict[str, ModelSpec] = {
    "llama": ModelSpec(name="Llama-3.1-8B", n_layers=32),
    "qwen": ModelSpec(name="Qwen3-8B-Base", n_layers=36),
}


def validate_ordering(layers: list[int], n_layers: int, label: str) -> None:
    expected = set(range(n_layers))
    got = set(layers)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"{label}: ordering must be a permutation of 0..{n_layers - 1}. "
            f"missing={missing!r} extra={extra!r}"
        )


def _validate_builtin_orderings() -> None:
    for mk, spec in MODEL_SPECS.items():
        for kind in ("wass", "norm"):
            for proj in ("V", "O"):
                layers = BUILTIN_ORDERINGS[mk][kind][proj]
                validate_ordering(layers, spec.n_layers, f"builtin {mk}/{kind}/{proj}")


_validate_builtin_orderings()


def wasserstein_avg_per_layer(csv_path: Path) -> dict[str, dict[int, float]]:
    """
    Mean Wasserstein H0 over epochs per (projection, layer), projections q,k,v,o lowercase.

    Returns dict proj_lower -> {layer_idx: mean_wass}.
    """
    df = pd.read_csv(csv_path)
    if "Wasserstein H0" not in df.columns:
        raise KeyError(f"{csv_path}: expected column 'Wasserstein H0'")
    df = df.copy()
    df["Layer"] = df["File"].astype(str).str.extract(r"layer(\d+)", expand=False).astype(int)
    out: dict[str, dict[int, float]] = {}
    for proj in ("q", "k", "v", "o"):
        sub = df[df["Projection"].astype(str) == proj]
        if sub.empty:
            continue
        g = sub.groupby("Layer", as_index=True)["Wasserstein H0"].mean()
        out[proj] = {int(i): float(v) for i, v in g.items()}
    return out


def norm_avg_per_layer_l2_csv(csv_path: Path) -> dict[str, dict[int, float]]:
    """
    Mean L2_Normalized over epochs per (projection, layer) from l2_results.csv.
    """
    df = pd.read_csv(csv_path)
    need = {"Epoch", "Layer", "Projection", "L2_Normalized"}
    if not need.issubset(df.columns):
        raise KeyError(f"{csv_path}: need columns {sorted(need)}")
    out: dict[str, dict[int, float]] = {}
    for proj in ("q", "k", "v", "o"):
        sub = df[df["Projection"].astype(str) == proj]
        if sub.empty:
            continue
        g = sub.groupby("Layer", as_index=True)["L2_Normalized"].mean()
        out[proj] = {int(i): float(v) for i, v in g.items()}
    return out


def load_norm_values_json(path: Path) -> dict[str, dict[int, float]]:
    """
    Load optional JSON::

        {"V": {"0": 0.01, ...}, "O": {"0": 0.02, ...}}

    Keys must be "V" and "O"; inner keys are layer indices as strings or ints.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[int, float]] = {}
    for proj in ("V", "O"):
        if proj not in data:
            raise KeyError(f"{path}: missing top-level key {proj!r}")
        block = data[proj]
        out[proj.lower()] = {int(k): float(v) for k, v in block.items()}
    return out


_PAIR_RE = re.compile(r"(\d+)\s*:\s*([eE0-9.+\-]+)")


def load_norm_values_text(path: Path) -> dict[str, dict[int, float]]:
    """
    Simple text sidecar (one line per projection)::

        V_VALUES: 0:0.0123 1:0.0118 2:0.0141
        O_VALUES: 0:0.0102 1:0.0099 2:0.0135

    Whitespace between pairs is flexible. Lines may be prefixed with ``#``.
    """
    out: dict[str, dict[int, float]] = {"v": {}, "o": {}}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("V_VALUES:"):
            blob = line.split(":", 1)[1]
            out["v"] = {int(a): float(b) for a, b in _PAIR_RE.findall(blob)}
        elif line.startswith("O_VALUES:"):
            blob = line.split(":", 1)[1]
            out["o"] = {int(a): float(b) for a, b in _PAIR_RE.findall(blob)}
    if not out["v"] or not out["o"]:
        raise ValueError(
            f"{path}: expected non-empty V_VALUES: and O_VALUES: lines "
            f"(got v_layers={len(out['v'])} o_layers={len(out['o'])})"
        )
    return out


def load_norm_values_sidecar(path: Path) -> dict[str, dict[int, float]]:
    """Load norm AVG map from ``.json`` or the ``V_VALUES`` / ``O_VALUES`` text format."""
    suf = path.suffix.lower()
    if suf == ".json":
        return load_norm_values_json(path)
    return load_norm_values_text(path)


def select_pool(ordering_low_to_high: list[int], k: int, high: bool) -> list[int]:
    """``high`` → layers with largest metric (tail of ascending-sorted list)."""
    if len(ordering_low_to_high) < k:
        raise ValueError(f"ordering length {len(ordering_low_to_high)} < k={k}")
    if high:
        return list(ordering_low_to_high[-k:])
    return list(ordering_low_to_high[:k])


def joint_score(mode: JointMode, w_hat: float, n_hat: float) -> float:
    """Combine normalized w_hat, n_hat per mode (both in [0,1] when inputs non-degenerate)."""
    if mode == "HH":
        return (w_hat + n_hat) / 2
    if mode == "HL":
        return (w_hat + (1.0 - n_hat)) / 2
    if mode == "LH":
        return ((1.0 - w_hat) + n_hat) / 2
    if mode == "LL":
        return ((1.0 - w_hat) + (1.0 - n_hat)) / 2
    raise ValueError(mode)


def run_one_joint(
    *,
    model_key: str,
    projection: Projection,
    mode: JointMode,
    wass_ordering: list[int],
    norm_ordering: list[int],
    wass_avg_by_layer: dict[int, float],
    norm_avg_by_layer: dict[int, float],
    k: int = FREEZE_BUDGET,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute steps 1–8 for a single (projection, mode) cell.

    ``wass_avg_by_layer`` / ``norm_avg_by_layer``: full AVG maps (all layers). The
    selected Wass/Norm sets define the candidate union only; W_i and N_i for every
    union layer are the **actual** AVG values from these maps (no masking to zero).
    """
    w_high = mode[0] == "H"
    n_high = mode[1] == "H"

    wass_selected = select_pool(wass_ordering, k, high=w_high)
    norm_selected = select_pool(norm_ordering, k, high=n_high)
    union_layers = sorted(set(wass_selected) | set(norm_selected))

    # Selected Wass/Norm sets define the candidate union only. Ranking inside the union
    # must use the true AVG Wasserstein and AVG Norm for every candidate layer, not
    # zeros for missing-set membership.
    raw_w: dict[int, float] = {}
    raw_n: dict[int, float] = {}
    for layer in union_layers:
        raw_w[layer] = float(wass_avg_by_layer[layer])
        raw_n[layer] = float(norm_avg_by_layer[layer])

    sum_w = sum(raw_w.values())
    sum_n = sum(raw_n.values())
    eps = 1e-12
    norm_w = {L: (raw_w[L] / sum_w) if sum_w > eps else 0.0 for L in union_layers}
    norm_n = {L: (raw_n[L] / sum_n) if sum_n > eps else 0.0 for L in union_layers}

    scores: dict[int, float] = {}
    for L in union_layers:
        scores[L] = joint_score(mode, norm_w[L], norm_n[L])

    ranked = sorted(union_layers, key=lambda L: scores[L], reverse=True)
    if len(ranked) < k:
        raise RuntimeError(
            f"{model_key} {projection} {mode}: |union|={len(ranked)} < {k}. "
            f"Cannot select {k} unique layers."
        )
    final_selected = ranked[:k]
    if len(set(final_selected)) != k:
        raise AssertionError("final_selected must contain unique layers")

    block = {
        "model": model_key,
        "projection": projection,
        "mode": mode,
        "wass_selected": wass_selected,
        "norm_selected": norm_selected,
        "union": union_layers,
        "raw_wass_values": {str(L): raw_w[L] for L in union_layers},
        "raw_norm_values": {str(L): raw_n[L] for L in union_layers},
        "normalized_wass_values": {str(L): norm_w[L] for L in union_layers},
        "normalized_norm_values": {str(L): norm_n[L] for L in union_layers},
        "ranked_layers_with_scores": [
            {
                "layer": L,
                "score": scores[L],
                "raw_wass": raw_w[L],
                "raw_norm": raw_n[L],
                "w_normalized": norm_w[L],
                "n_normalized": norm_n[L],
            }
            for L in ranked
        ],
        "final_selected_layers": final_selected,
    }

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"  {model_key.upper()}  |  {projection}  |  mode={mode}")
        print(f"{'=' * 72}")
        print(f"  wass_selected ({len(wass_selected)}): {wass_selected}")
        print(f"  norm_selected ({len(norm_selected)}): {norm_selected}")
        print(f"  union ({len(union_layers)}): {union_layers}")
        print(f"  raw_wass (union): {raw_w}")
        print(f"  raw_norm (union): {raw_n}")
        print(f"  sum_w={sum_w:.6g} sum_n={sum_n:.6g}")
        print(f"  norm_w: {norm_w}")
        print(f"  norm_n: {norm_n}")
        print(f"  ranked (all in union): {[(L, round(scores[L], 6)) for L in ranked]}")
        print(f"  final_selected_layers ({len(final_selected)}): {final_selected}")

    return block


def build_model_bundle(
    model_key: str,
    wass_csv: Path,
    norm_l2_csv: Path | None,
    norm_sidecar: Path | None,
) -> dict[str, Any]:
    """Load AVG maps from CSV/sidecar; V/O orderings always from ``BUILTIN_ORDERINGS``."""
    spec = MODEL_SPECS[model_key]
    n = spec.n_layers

    orderings: dict[str, dict[Projection, list[int]]] = {
        "wass": {p: list(BUILTIN_ORDERINGS[model_key]["wass"][p]) for p in ("V", "O")},
        "norm": {p: list(BUILTIN_ORDERINGS[model_key]["norm"][p]) for p in ("V", "O")},
    }
    for kind in ("wass", "norm"):
        for proj in ("V", "O"):
            validate_ordering(orderings[kind][proj], n, f"{kind}/{proj}")

    wass_by_proj = wasserstein_avg_per_layer(wass_csv)
    for p in ("v", "o"):
        if p not in wass_by_proj or len(wass_by_proj[p]) != n:
            raise ValueError(
                f"{wass_csv}: expected mean Wasserstein for all {n} layers, projection {p!r}"
            )

    if norm_l2_csv is not None and norm_l2_csv.is_file():
        norm_by_proj = norm_avg_per_layer_l2_csv(norm_l2_csv)
        norm_src = str(norm_l2_csv.resolve())
    elif norm_sidecar is not None and norm_sidecar.is_file():
        norm_by_proj = load_norm_values_sidecar(norm_sidecar)
        norm_src = str(norm_sidecar.resolve())
    else:
        raise FileNotFoundError(
            f"{model_key}: need norm_l2_csv or norm sidecar file. "
            f"l2={norm_l2_csv!s} sidecar={norm_sidecar!s}"
        )

    for p in ("v", "o"):
        if p not in norm_by_proj or len(norm_by_proj[p]) != n:
            raise ValueError(
                f"Norm source: expected AVG norm for all {n} layers, projection {p!r}"
            )

    return {
        "spec": spec,
        "wass_csv": str(wass_csv.resolve()),
        "norm_source": norm_src,
        "wass_avg": {"V": wass_by_proj["v"], "O": wass_by_proj["o"]},
        "norm_avg": {"V": norm_by_proj["v"], "O": norm_by_proj["o"]},
        "orderings": orderings,
    }


def run_all_for_model(
    model_key: str, bundle: dict[str, Any], verbose: bool, *, k: int
) -> dict[str, Any]:
    """Run HH/HL/LH/LL for V and O."""
    spec: ModelSpec = bundle["spec"]
    modes: tuple[JointMode, ...] = ("HH", "HL", "LH", "LL")
    results: dict[str, dict[str, Any]] = {"V": {}, "O": {}}

    for proj in ("V", "O"):
        w_ord = bundle["orderings"]["wass"][proj]
        n_ord = bundle["orderings"]["norm"][proj]
        w_vals = bundle["wass_avg"][proj]
        n_vals = bundle["norm_avg"][proj]
        for mode in modes:
            results[proj][mode] = run_one_joint(
                model_key=model_key,
                projection=proj,
                mode=mode,
                wass_ordering=w_ord,
                norm_ordering=n_ord,
                wass_avg_by_layer=w_vals,
                norm_avg_by_layer=n_vals,
                k=k,
                verbose=verbose,
            )

    return {
        "model_name": spec.name,
        "model_key": model_key,
        "n_layers": spec.n_layers,
        "mode": "avg_only",
        "freeze_budget": k,
        "sources": {
            "orderings": "BUILTIN_ORDERINGS",
            "wass_csv": bundle["wass_csv"],
            "norm_values": bundle["norm_source"],
        },
        "V": results["V"],
        "O": results["O"],
    }


def slim_layers_for_gsm8k(all_out: dict[str, Any]) -> dict[str, Any]:
    """Strip to the nested lists expected by run_gsm8k_joint_wass_norm_freeze.sh."""
    slim: dict[str, Any] = {}
    for mk, block in all_out["joint_layer_selection_avg"].items():
        slim[mk] = {"V": {}, "O": {}}
        for proj in ("V", "O"):
            for mode, cell in block[proj].items():
                slim[mk][proj][mode] = list(cell["final_selected_layers"])
    return slim


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS.keys()),
        default=sorted(MODEL_SPECS.keys()),
        help="Which base models to run (default: both).",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Write full output JSON here.")
    p.add_argument(
        "--output-gsm8k-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write slim JSON for run_gsm8k_joint_wass_norm_freeze.sh "
        "(llama/qwen → V/O → HH/HL/LH/LL → list[int]).",
    )
    p.add_argument(
        "--freeze-budget",
        type=int,
        default=FREEZE_BUDGET,
        metavar="K",
        help=f"How many layers to freeze per projection per mode (default {FREEZE_BUDGET}).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-experiment debug prints.")
    p.add_argument(
        "--emit-norm-json-template",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a JSON template {\"V\": {...}, \"O\": {...}} with 1.0 placeholders "
        "for n_layers (default 36), then exit. Replace values with your AVG norms.",
    )
    p.add_argument(
        "--template-n-layers",
        type=int,
        default=36,
        help="Layer count for --emit-norm-json-template (default 36 = Qwen).",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.freeze_budget < 1:
        print("ERROR: --freeze-budget must be >= 1", file=sys.stderr)
        return 2
    if args.emit_norm_json_template is not None:
        n = args.template_n_layers
        block = {proj: {str(i): 1.0 for i in range(n)} for proj in ("V", "O")}
        args.emit_norm_json_template.parent.mkdir(parents=True, exist_ok=True)
        args.emit_norm_json_template.write_text(
            json.dumps(block, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote template ({n} layers per projection): {args.emit_norm_json_template}")
        return 0

    verbose = not args.quiet
    all_out: dict[str, Any] = {"joint_layer_selection_avg": {}, "schema_version": 1}

    for mk in args.models:
        if mk == "llama":
            if not LLAMA_WASS_CSV.is_file():
                print(f"ERROR: missing {LLAMA_WASS_CSV}", file=sys.stderr)
                return 2
            if not LLAMA_L2_CSV.is_file():
                print(f"ERROR: missing {LLAMA_L2_CSV}", file=sys.stderr)
                return 2
            bundle = build_model_bundle(mk, LLAMA_WASS_CSV, LLAMA_L2_CSV, None)
        else:
            if not QWEN_WASS_CSV.is_file():
                print(f"ERROR: missing {QWEN_WASS_CSV}", file=sys.stderr)
                return 2
            if QWEN_L2_CSV.is_file():
                bundle = build_model_bundle(mk, QWEN_WASS_CSV, QWEN_L2_CSV, None)
            elif QWEN_NORM_SIDECAR.is_file():
                bundle = build_model_bundle(mk, QWEN_WASS_CSV, None, QWEN_NORM_SIDECAR)
            else:
                print(
                    f"ERROR: Qwen needs {QWEN_L2_CSV} or {QWEN_NORM_SIDECAR}",
                    file=sys.stderr,
                )
                return 2
        all_out["joint_layer_selection_avg"][mk] = run_all_for_model(
            mk, bundle, verbose=verbose, k=args.freeze_budget
        )

    text = json.dumps(all_out, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output_json}")
    elif not args.output_gsm8k_json:
        print(text)

    if args.output_gsm8k_json:
        slim = slim_layers_for_gsm8k(all_out)
        args.output_gsm8k_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_gsm8k_json.write_text(
            json.dumps(slim, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output_gsm8k_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
