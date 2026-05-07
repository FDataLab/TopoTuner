#!/usr/bin/env python3
"""
Topology-based stopping threshold experiment.

For each (dataset, backbone, training method), uses exported Wasserstein H0 drift
(`wasserstein_results.csv`) vs baseline per epoch and epoch-wise accuracy JSON files.

Stopping rule (integer epoch t aligned across drift ``epoch_t`` and accuracy key ``"t"``):

    r(t) = mean(Wasserstein H0) over monitored rows for epoch_t (projections configurable).
    t*(eta) = min { t >= T0 | r(t) <= eta }; if none, t* = final epoch with drift.

Outputs:
  - topology_stopping_availability.csv (discovery report)
  - topology_stopping_eta_summary.csv (one row per eta × run)
  - Figures (accuracy vs η), controlled by ``--plot-by``:
      * ``model``: ``topology_stopping_eta_<llama|qwen|mistral>.png`` — legend shows dataset + method.
      * ``dataset``: ``topology_stopping_eta_<imdb|sst2|mmlu|gsm8k>.png`` — legend shows model + method.
      * ``both`` (default): writes both sets.

Examples:
  python plot_topology_stopping_eta.py --repo-root /path/to/exploration-finetuning \\
      --availability-only
  python plot_topology_stopping_eta.py --repo-root ... \\
      --manifest my_paths.json --output-dir ./out

Manifest JSON (optional) can set ``drift_paths`` / ``drift_overrides`` with keys
``"<dataset>|<backbone>|<method>"``, e.g. ``"imdb|llama|lora"``, values = absolute paths
to ``wasserstein_results.csv`` for per-run LoRA/freeze drift when you add exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(f"matplotlib required: {e}") from e


# --- Canonical experiment grid -------------------------------------------------

DATASETS_ORDERED = ("imdb", "sst2", "mmlu", "gsm8k")

BACKBONES: Tuple[str, ...] = ("llama", "qwen-base", "mistral-7b-v03")
METHODS: Tuple[str, ...] = ("full", "lora", "freeze")

DISPLAY_MODEL = {
    "llama": "LLaMA",
    "qwen-base": "Qwen",
    "mistral-7b-v03": "Mistral",
}

DISPLAY_METHOD = {"full": "Full", "lora": "LoRA", "freeze": "Freeze"}

DISPLAY_DATASET = {
    "imdb": "IMDB",
    "sst2": "SST2",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
}

# Filename stems for PNG outputs (avoid awkward folder names).
BACKBONE_OUTPUT_STEM = {
    "llama": "llama",
    "qwen-base": "qwen",
    "mistral-7b-v03": "mistral",
}

RESULTS_SUBDIR = {"llama": "llama", "qwen-base": "qwen", "mistral-7b-v03": "mistral"}

EPOCH_COL_RE = re.compile(r"epoch_(\d+)", re.I)


def _parse_epoch_tag(tag: Any) -> Optional[int]:
    if tag is None or (isinstance(tag, float) and np.isnan(tag)):
        return None
    s = str(tag).strip()
    m = EPOCH_COL_RE.search(s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def curve_label(dataset: str, method: str) -> str:
    return f"{DISPLAY_DATASET.get(dataset, dataset.upper())} {DISPLAY_METHOD[method]}"


def model_method_label(backbone: str, method: str) -> str:
    return f"{DISPLAY_MODEL[backbone]} {DISPLAY_METHOD[method]}"


def discover_latest_results_run(repo_root: str, dataset: str) -> Optional[str]:
    """Pick lexicographically latest ``results/<dataset>_epoch_eval_py_*`` folder."""
    results_root = os.path.join(repo_root, "results")
    if not os.path.isdir(results_root):
        return None
    prefix = f"{dataset}_epoch_eval_py_"
    candidates = [
        os.path.join(results_root, name)
        for name in os.listdir(results_root)
        if name.startswith(prefix) and os.path.isdir(os.path.join(results_root, name))
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def gsm8k_accuracy_path(repo_root: str, backbone: str, method: str) -> str:
    sub = RESULTS_SUBDIR[backbone]
    base = os.path.join(repo_root, "eval", "split", "gsm8k", backbone, "json")
    if method == "full":
        name = "full_epoch_accuracy.json"
    elif method == "lora":
        name = "lora_epoch_accuracy.json"
    else:
        # Freeze exports differ by backbone for GSM8K in this repo.
        if backbone == "llama":
            name = "wass-high6_epoch_accuracy.json"
        else:
            name = "wass-high3_epoch_accuracy.json"
    return os.path.join(base, name)


def standard_accuracy_path(run_root: str, backbone: str, method: str, dataset: str) -> str:
    """Paths under ``results/<dataset>_epoch_eval_py_*/`` for IMDB / SST2 / MMLU."""
    sub = RESULTS_SUBDIR[backbone]
    subdir = os.path.join(run_root, sub)
    if dataset == "sst2":
        if method == "full":
            stem = f"{sub}-{DISPLAY_METHOD[method].lower()}"
        elif method == "lora":
            stem = f"{sub}-{DISPLAY_METHOD[method].lower()}"
        else:
            stem = f"{sub}-wass"
        return os.path.join(subdir, f"{stem}_epoch_accuracy.json")
    # imdb / mmlu
    if method == "full":
        fname = "full_epoch_accuracy.json"
    elif method == "lora":
        fname = "lora_epoch_accuracy.json"
    else:
        fname = "wass-high-3_epoch_accuracy.json"
    return os.path.join(subdir, fname)


def resolve_accuracy_path(repo_root: str, dataset: str, backbone: str, method: str) -> Optional[str]:
    if dataset == "gsm8k":
        p = gsm8k_accuracy_path(repo_root, backbone, method)
        return p if os.path.isfile(p) else None
    run_root = discover_latest_results_run(repo_root, dataset)
    if run_root is None:
        return None
    p = standard_accuracy_path(run_root, backbone, method, dataset)
    return p if os.path.isfile(p) else None


def default_drift_path(repo_root: str, dataset: str, backbone: str, method: str) -> Optional[str]:
    """Built-in drift layout for this repo (extend via manifest overrides)."""
    wvb = os.path.join(repo_root, "eval", "split", "weight_vs_baseline")
    if method == "full":
        p = os.path.join(wvb, "wass_full_by_task", backbone, dataset, "wasserstein_results.csv")
        return p if os.path.isfile(p) else None
    if method == "freeze":
        # GSM8K freeze drift lives next to GSM8K freeze exports (Llama uses high-6 here).
        if dataset != "gsm8k":
            return None
        if backbone == "llama":
            p = os.path.join(wvb, "wass", "llama", "wass_high_6", "wasserstein_results.csv")
        elif backbone == "qwen-base":
            p = os.path.join(wvb, "wass", "qwen-base", "wass_high_3", "wasserstein_results.csv")
        else:
            return None
        return p if os.path.isfile(p) else None
    # LoRA: no shared-by-task exports in-tree; supply manifest overrides when available.
    return None


def load_manifest(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def manifest_drift_key(dataset: str, backbone: str, method: str) -> str:
    return f"{dataset}|{backbone}|{method}"


def drift_path_with_manifest(
    repo_root: str,
    dataset: str,
    backbone: str,
    method: str,
    manifest: Mapping[str, Any],
) -> Optional[str]:
    overrides = manifest.get("drift_paths") or manifest.get("drift_overrides") or {}
    k = manifest_drift_key(dataset, backbone, method)
    if k in overrides and overrides[k]:
        p = os.path.expanduser(str(overrides[k]))
        return p if os.path.isfile(p) else None
    return default_drift_path(repo_root, dataset, backbone, method)


def load_accuracy_epochs(path: str) -> Tuple[Dict[int, float], List[int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    acc: Dict[int, float] = {}
    for ks, vs in raw.items():
        try:
            ei = int(str(ks))
        except ValueError:
            continue
        if isinstance(vs, list) and vs:
            acc[ei] = float(np.mean(np.asarray(vs, dtype=float)))
        elif isinstance(vs, (int, float)):
            acc[ei] = float(vs)
        else:
            continue
    keys = sorted(acc.keys())
    return acc, keys


def load_drift_scores(
    path: str,
    metric_col: str,
    projections: Optional[Sequence[str]],
) -> Tuple[Dict[int, float], List[int]]:
    df = pd.read_csv(path)
    if metric_col not in df.columns:
        raise ValueError(f"Missing column {metric_col!r} in {path}")
    sub = df
    if projections is not None and "Projection" in sub.columns:
        want = {str(p).strip().lower() for p in projections}
        proj_key = sub["Projection"].astype(str).str.strip().str.lower()
        sub = sub[proj_key.isin(want)]
    if "Epoch" not in sub.columns:
        raise ValueError(f"Missing Epoch column in {path}")

    scores: Dict[int, List[float]] = {}
    for _, row in sub.iterrows():
        ei = _parse_epoch_tag(row["Epoch"])
        if ei is None:
            continue
        val = row[metric_col]
        if pd.isna(val):
            continue
        scores.setdefault(ei, []).append(float(val))
    merged = {e: float(np.mean(v)) for e, v in sorted(scores.items())}
    keys = sorted(merged.keys())
    return merged, keys


def symmetric_missing_epochs(a: Iterable[int], b: Iterable[int]) -> List[int]:
    sa, sb = set(a), set(b)
    sym = sorted((sa ^ sb) & set(range(1, max(sa | sb | {1}) + 1)))
    return sym


def intersection_epochs(acc_keys: Sequence[int], drift_keys: Sequence[int]) -> List[int]:
    return sorted(set(acc_keys) & set(drift_keys))


def stopping_epoch(r_by_t: Mapping[int, float], eta: float, t_min: int) -> Tuple[int, float]:
    """Return (t_star, r(t_star)) using only epochs present in r_by_t (aligned drift)."""
    if not r_by_t:
        raise ValueError("empty drift map")
    max_epoch = max(r_by_t.keys())
    candidates = [t for t in range(t_min, max_epoch + 1) if t in r_by_t]
    for t in sorted(candidates):
        if r_by_t[t] <= eta:
            return t, r_by_t[t]
    final_t = max(r_by_t.keys())
    return final_t, r_by_t[final_t]


def print_availability_table(rows: List[Dict[str, Any]]) -> None:
    cols = [
        "dataset",
        "model",
        "method",
        "has_epoch_accuracy",
        "available_epochs",
        "missing_epochs",
        "accuracy_file_path",
        "drift_file_path",
    ]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    hdr = " | ".join(c.ljust(widths[c]) for c in cols)
    print(hdr)
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=str,
        default=os.environ.get(
            "NW_EXPLORATION_FINETUNING_ROOT",
            str((Path(__file__).resolve().parents[3] / "numpy_weights" / "exploration-finetuning").resolve()),
        ),
        help="exploration-finetuning checkout (epoch JSON + eval CSV roots).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="Where to write CSVs and PNGs.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help='JSON file with drift_paths / drift_overrides: { "imdb|llama|lora": "/abs/wasserstein_results.csv", ... }',
    )
    parser.add_argument("--metric-col", type=str, default="Wasserstein H0")
    parser.add_argument(
        "--projections",
        type=str,
        default=None,
        help="Comma-separated projections to average (default: all rows in CSV).",
    )
    parser.add_argument("--t-min", type=int, default=2, help="Minimum epoch in stopping rule.")
    parser.add_argument("--eta-min", type=float, default=0.01)
    parser.add_argument("--eta-max", type=float, default=1.0)
    parser.add_argument("--eta-n", type=int, default=100)
    parser.add_argument(
        "--eta-values",
        type=str,
        default=None,
        help="Comma-separated eta values (overrides eta-min/max/n).",
    )
    parser.add_argument("--availability-only", action="store_true")
    parser.add_argument(
        "--fail-on-incomplete-accuracy",
        action="store_true",
        help="Exit non-zero if any run lacks usable epoch accuracy (needs epoch >= t-min).",
    )
    parser.add_argument(
        "--continue-with-incomplete-accuracy",
        action="store_true",
        help="Plot even when some runs lack usable epoch-wise accuracy (otherwise abort before figures).",
    )
    parser.add_argument(
        "--plot-by",
        choices=("both", "model", "dataset"),
        default="both",
        help="Per-backbone figures (legend=dataset+method), per-dataset figures (legend=model+method), or both.",
    )
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    projections = (
        [p.strip() for p in args.projections.split(",") if p.strip()]
        if args.projections
        else None
    )

    manifest = load_manifest(args.manifest)

    if args.eta_values:
        eta_values = np.asarray([float(x) for x in args.eta_values.split(",") if x.strip()])
    else:
        eta_values = np.linspace(args.eta_min, args.eta_max, args.eta_n)

    availability_rows: List[Dict[str, Any]] = []
    run_context: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for dataset in DATASETS_ORDERED:
        for backbone in BACKBONES:
            for method in METHODS:
                acc_path = resolve_accuracy_path(repo_root, dataset, backbone, method)
                drift_path_resolved = drift_path_with_manifest(
                    repo_root, dataset, backbone, method, manifest
                )
                drift_path_final: Optional[str] = drift_path_resolved

                has_acc_file = acc_path is not None
                acc_epochs: List[int] = []
                drift_epochs: List[int] = []
                available_epochs: List[int] = []
                missing_epochs: List[int] = []
                has_epoch_accuracy = False

                acc_map: Dict[int, float] = {}
                drift_map: Dict[int, float] = {}

                if acc_path:
                    try:
                        acc_map, acc_epochs = load_accuracy_epochs(acc_path)
                        has_epoch_accuracy = any(e >= args.t_min for e in acc_epochs)
                    except (json.JSONDecodeError, OSError, ValueError) as e:
                        print(f"[warn] Could not parse accuracy {acc_path}: {e}", file=sys.stderr)

                if drift_path_final:
                    try:
                        drift_map, drift_epochs = load_drift_scores(
                            drift_path_final, args.metric_col, projections
                        )
                    except (OSError, ValueError) as e:
                        print(
                            f"[warn] Could not parse drift {drift_path_final}: {e}",
                            file=sys.stderr,
                        )
                        drift_path_final = None
                        drift_map, drift_epochs = {}, []

                if drift_path_final and acc_epochs and drift_epochs:
                    available_epochs = intersection_epochs(acc_epochs, drift_epochs)
                    missing_epochs = symmetric_missing_epochs(acc_epochs, drift_epochs)
                elif acc_epochs and not drift_path_final:
                    missing_epochs = ["drift_unavailable"]
                elif drift_path_final and drift_epochs and not acc_epochs:
                    missing_epochs = ["accuracy_unavailable"]

                drift_available = drift_path_final is not None and bool(drift_map)

                row = {
                    "dataset": dataset,
                    "model": DISPLAY_MODEL[backbone],
                    "method": DISPLAY_METHOD[method],
                    "has_epoch_accuracy": has_epoch_accuracy,
                    "available_epochs": ",".join(str(e) for e in available_epochs),
                    "missing_epochs": ",".join(str(e) for e in missing_epochs),
                    "accuracy_file_path": acc_path or "",
                    "drift_file_path": drift_path_final or "",
                }
                availability_rows.append(row)

                run_context[(dataset, backbone, method)] = {
                    "acc_map": acc_map,
                    "drift_map": drift_map,
                    "available_epochs": available_epochs,
                    "drift_available": drift_available,
                    "has_epoch_accuracy": has_epoch_accuracy,
                    "has_acc_file": has_acc_file,
                    "accuracy_path": acc_path,
                    "drift_path": drift_path_final,
                }

    avail_csv = os.path.join(out_dir, "topology_stopping_availability.csv")
    with open(avail_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(availability_rows[0].keys()))
        w.writeheader()
        w.writerows(availability_rows)

    print(f"Wrote {avail_csv}\n")
    print_availability_table(availability_rows)

    bad_acc = [r for r in availability_rows if not r["has_epoch_accuracy"]]

    print("\n--- Availability summary ---")
    missing_acc_files = [
        r for r in availability_rows if not r["accuracy_file_path"]
    ]
    print(f"Missing accuracy files: {len(missing_acc_files)}")
    print(f"Runs without epoch >= {args.t_min} in accuracy JSON: {len(bad_acc)}")
    missing_drift = [r for r in availability_rows if not r["drift_file_path"]]
    print(f"Missing drift files (built-in layout): {len(missing_drift)}")

    if args.fail_on_incomplete_accuracy and bad_acc:
        print("\n[error] Incomplete epoch accuracy for some runs (--fail-on-incomplete-accuracy).", file=sys.stderr)
        for r in bad_acc:
            print(
                f"  {r['dataset']} | {r['model']} | {r['method']} | "
                f"path={r['accuracy_file_path'] or '<missing>'}",
                file=sys.stderr,
            )
        return 2

    if args.availability_only:
        return 0

    summary_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS_ORDERED:
        for backbone in BACKBONES:
            for method in METHODS:
                ctx = run_context[(dataset, backbone, method)]
                if not ctx["drift_available"]:
                    for eta in eta_values:
                        summary_rows.append(
                            {
                                "dataset": dataset,
                                "model": DISPLAY_MODEL[backbone],
                                "method": DISPLAY_METHOD[method],
                                "eta": float(eta),
                                "selected_epoch": "",
                                "stopping_score": "",
                                "accuracy": "",
                                "drift_available": False,
                            }
                        )
                    continue

                acc_map = ctx["acc_map"]
                drift_map = ctx["drift_map"]
                avail_set = set(ctx["available_epochs"]) & set(drift_map.keys())
                avail = sorted(avail_set)
                if not avail:
                    for eta in eta_values:
                        summary_rows.append(
                            {
                                "dataset": dataset,
                                "model": DISPLAY_MODEL[backbone],
                                "method": DISPLAY_METHOD[method],
                                "eta": float(eta),
                                "selected_epoch": "",
                                "stopping_score": "",
                                "accuracy": "",
                                "drift_available": True,
                            }
                        )
                    continue

                r_eff = {t: drift_map[t] for t in avail}

                for eta in eta_values:
                    t_star, r_star = stopping_epoch(r_eff, float(eta), args.t_min)
                    acc_val = acc_map.get(t_star)
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "model": DISPLAY_MODEL[backbone],
                            "method": DISPLAY_METHOD[method],
                            "eta": float(eta),
                            "selected_epoch": int(t_star),
                            "stopping_score": float(r_star),
                            "accuracy": float(acc_val) if acc_val is not None else "",
                            "drift_available": True,
                        }
                    )

    summary_csv = os.path.join(out_dir, "topology_stopping_eta_summary.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {summary_csv}")

    # Warn before plotting if accuracy incomplete for any run user cares about.
    if bad_acc and not args.continue_with_incomplete_accuracy:
        print(
            "\n[abort plotting] Some runs lack usable epoch accuracy "
            f"(need epoch >= {args.t_min}). "
            "Pass --continue-with-incomplete-accuracy to plot anyway.",
            file=sys.stderr,
        )
        return 3

    # --- Figures -----------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("[warn] No summary rows; skipping plots.", file=sys.stderr)
        return 0

    plot_model = args.plot_by in ("both", "model")
    plot_dataset = args.plot_by in ("both", "dataset")

    if plot_model:
        for backbone in BACKBONES:
            model_name = DISPLAY_MODEL[backbone]
            sub = summary_df[summary_df["model"] == model_name]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(11, 6.5))

            plotted = 0
            for dataset in DATASETS_ORDERED:
                for method in METHODS:
                    curve = sub[
                        (sub["dataset"] == dataset)
                        & (sub["method"] == DISPLAY_METHOD[method])
                    ]
                    if curve.empty or not curve["drift_available"].any():
                        continue
                    curve_ok = curve[curve["accuracy"].apply(lambda x: x != "" and pd.notna(x))]
                    if curve_ok.empty:
                        continue
                    label = curve_label(dataset, method)
                    ax.plot(
                        curve_ok["eta"].values,
                        curve_ok["accuracy"].astype(float).values,
                        label=label,
                        linewidth=1.8,
                    )
                    plotted += 1

            ax.set_title(f"Topology-based stopping threshold: {model_name}")
            ax.set_xlabel(r"$\eta$ threshold")
            ax.set_ylabel("Accuracy at selected epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)

            fig.tight_layout()

            stem = BACKBONE_OUTPUT_STEM[backbone]
            path_fig = os.path.join(out_dir, f"topology_stopping_eta_{stem}.png")
            fig.savefig(path_fig, dpi=160)
            plt.close(fig)
            print(f"Wrote {path_fig} [by model] ({plotted} curves with data).")

    if plot_dataset:
        for dataset in DATASETS_ORDERED:
            sub = summary_df[summary_df["dataset"] == dataset]
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(11, 6.5))

            plotted = 0
            for backbone in BACKBONES:
                for method in METHODS:
                    curve = sub[
                        (sub["model"] == DISPLAY_MODEL[backbone])
                        & (sub["method"] == DISPLAY_METHOD[method])
                    ]
                    if curve.empty or not curve["drift_available"].any():
                        continue
                    curve_ok = curve[curve["accuracy"].apply(lambda x: x != "" and pd.notna(x))]
                    if curve_ok.empty:
                        continue
                    label = model_method_label(backbone, method)
                    ax.plot(
                        curve_ok["eta"].values,
                        curve_ok["accuracy"].astype(float).values,
                        label=label,
                        linewidth=1.8,
                    )
                    plotted += 1

            title_ds = DISPLAY_DATASET.get(dataset, dataset.upper())
            ax.set_title(f"Topology-based stopping threshold: {title_ds}")
            ax.set_xlabel(r"$\eta$ threshold")
            ax.set_ylabel("Accuracy at selected epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)

            fig.tight_layout()

            path_fig = os.path.join(out_dir, f"topology_stopping_eta_{dataset}.png")
            fig.savefig(path_fig, dpi=160)
            plt.close(fig)
            print(f"Wrote {path_fig} [by dataset] ({plotted} curves with data).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
