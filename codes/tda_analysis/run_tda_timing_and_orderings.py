#!/usr/bin/env python3
"""Time persistence + Wasserstein + norm; write per-model logs under --output-dir."""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NW_ROOT = SCRIPT_DIR.parent
DEFAULT_CODES = Path(__file__).resolve().parents[3] / "codes"
if not DEFAULT_CODES.is_dir():
    DEFAULT_CODES = Path.cwd() / "codes"

PROJECTIONS = ["q", "k", "v", "o"]

DEFAULT_LLAMA_PRETRAINED_HF = "meta-llama/Llama-3.1-8B"
DEFAULT_LLAMA_FINETUNE_DIR = NW_ROOT / "checkpoints" / "llama" / "full"
DEFAULT_QWEN_PRETRAINED_HF = "Qwen/Qwen3-8B-Base"
DEFAULT_QWEN_FINETUNE_DIR = NW_ROOT / "checkpoints" / "qwen-base" / "full" / "run3"
DEFAULT_RESULTS_ROOT = NW_ROOT / "analysis" / "tda" / "llama-and-qwen-base"

DEFAULT_CODES_FOR_TOPO = Path(__file__).resolve().parents[2]
TOPO_EXPLORE_FINETUNE = DEFAULT_CODES_FOR_TOPO / "numpy_weights" / "exploration-finetuning"
DEFAULT_MISTRAL_PRETRAINED_HF = "mistralai/Mistral-7B-v0.3"
DEFAULT_MISTRAL_FINETUNE_DIR = TOPO_EXPLORE_FINETUNE / "checkpoints" / "mistral-7b-v03" / "full" / "run1"
DEFAULT_MISTRAL_TDA_DIR = TOPO_EXPLORE_FINETUNE / "numpy_weights" / "gsm8k-mistral-7b-v03-tda-results"


def _prepend_sys_path(codes: Path) -> None:
    for sub in ("tda", "utils"):
        p = codes / sub
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


def resolve_pretrained_diagrams(tda_dir: Path) -> Path:
    candidates = [
        tda_dir / "persistence" / "baseline" / "SavedDiagrams",
        tda_dir / "persistence" / "epoch_0" / "SavedDiagrams",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Pretrained (epoch 0) diagrams not found. Tried: {tried}")


def discover_finetuning_epoch_diagrams(tda_dir: Path) -> list[tuple[str, Path]]:
    pers = tda_dir / "persistence"
    if not pers.is_dir():
        raise FileNotFoundError(f"Missing persistence dir: {pers}")
    epochs: list[tuple[int, Path]] = []
    for child in pers.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"epoch_(\d+)", child.name)
        if not m:
            continue
        k = int(m.group(1))
        if k < 1:
            continue
        sd = child / "SavedDiagrams"
        if sd.is_dir():
            epochs.append((k, child))
    epochs.sort(key=lambda x: x[0])
    return [(f"epoch_{n}", p / "SavedDiagrams") for n, p in epochs]


def resolve_pretrained_numpy(tda_dir: Path) -> Path:
    candidates = [
        tda_dir / "baseline" / "numpy_weights",
        tda_dir / "epoch_0" / "numpy_weights",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Pretrained (epoch 0) numpy_weights not found. Tried: {tried}")


def discover_finetuning_numpy_epochs(tda_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for child in tda_dir.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"epoch_(\d+)", child.name)
        if not m:
            continue
        k = int(m.group(1))
        if k < 1:
            continue
        nw = child / "numpy_weights"
        if nw.is_dir():
            out.append((k, nw))
    out.sort(key=lambda x: x[0])
    return out


def discover_numpy_layout(tda_dir: Path) -> tuple[Path, Path]:
    base_nw = resolve_pretrained_numpy(tda_dir)
    fin = discover_finetuning_numpy_epochs(tda_dir)
    if not fin:
        raise FileNotFoundError(
            f"No finetuning epoch_*/numpy_weights (k>=1) under {tda_dir}. "
            "Pretrained should be baseline/ or epoch_0/; training steps are epoch_1+."
        )
    return base_nw, fin[-1][1]


def pretrained_persist_output_dir(tda_dir: Path, base_nw: Path) -> Path:
    b = (tda_dir / "baseline" / "numpy_weights").resolve()
    z = (tda_dir / "epoch_0" / "numpy_weights").resolve()
    if base_nw.resolve() == b:
        return tda_dir / "persistence" / "baseline"
    if base_nw.resolve() == z:
        return tda_dir / "persistence" / "epoch_0"
    raise ValueError(f"Unexpected pretrained numpy path {base_nw}; expected under baseline/ or epoch_0/")


def regenerate_persistence_tree(
    tda_dir: Path,
    base_nw: Path,
    codes_root: Path,
    log: Callable[[str], None],
    timings: dict[str, float],
    key: str,
) -> None:
    script = codes_root / "tda" / "generate_persistence.py"
    if not script.is_file():
        raise FileNotFoundError(f"generate_persistence.py not found: {script}")
    py = Path(sys.executable)
    (tda_dir / "persistence").mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    out0 = pretrained_persist_output_dir(tda_dir, base_nw)
    log(f"  [persistence] pretrained numpy → diagrams: {base_nw} → {out0}")
    t0 = time.perf_counter()
    subprocess.run(
        [str(py), str(script), "--input-dir", str(base_nw), "--output-dir", str(out0), "--maxdim", "0"],
        check=True,
    )
    timings[f"{key}_persistence_pretrained_s"] = time.perf_counter() - t0

    for k, nw in discover_finetuning_numpy_epochs(tda_dir):
        out = tda_dir / "persistence" / f"epoch_{k}"
        log(f"  [persistence] epoch_{k}: {nw} → {out}")
        t0 = time.perf_counter()
        subprocess.run(
            [str(py), str(script), "--input-dir", str(nw), "--output-dir", str(out), "--maxdim", "0"],
            check=True,
        )
        timings[f"{key}_persistence_epoch_{k}_s"] = time.perf_counter() - t0

    timings[f"{key}_persistence_all_s"] = time.perf_counter() - t_all
    log(f"  [timings] {key} persistence total (all folders): {timings[f'{key}_persistence_all_s']:.3f}s\n")


def _last_epoch_from_labels(labels: list[str]) -> str:
    def key(lab: str) -> int:
        m = re.match(r"epoch_(\d+)", str(lab))
        return int(m.group(1)) if m else -1

    return max(labels, key=key)


def _wass_order_mean(df: pd.DataFrame, wass_col: str = "Wasserstein H0") -> dict[str, list[int]]:
    results: dict[str, list[int]] = {}
    for proj in PROJECTIONS:
        sub = df[df["Projection"] == proj].copy()
        if sub.empty:
            continue
        sub["Layer"] = sub["File"].apply(lambda f: int(re.search(r"layer(\d+)", f).group(1)))
        layer_avg = sub.groupby("Layer")[wass_col].mean().sort_values()
        results[proj] = layer_avg.index.tolist()
    return results


def _wass_order_final(df: pd.DataFrame, wass_col: str = "Wasserstein H0") -> tuple[dict[str, list[int]], str]:
    last_ep = _last_epoch_from_labels(df["Epoch"].astype(str).unique().tolist())
    d = df[df["Epoch"].astype(str) == last_ep]
    results: dict[str, list[int]] = {}
    for proj in PROJECTIONS:
        sub = d[d["Projection"] == proj].copy()
        if sub.empty:
            continue
        sub["Layer"] = sub["File"].apply(lambda f: int(re.search(r"layer(\d+)", f).group(1)))
        layer_vals = sub.groupby("Layer")[wass_col].mean().sort_values()
        results[proj] = layer_vals.index.tolist()
    return results, last_ep


def _fmt_wass_block(heading: str, results: dict[str, list[int]]) -> str:
    lines = [heading.rstrip(), ""]
    for proj in sorted(results.keys()):
        layers = results[proj]
        lines.append(f"{proj.upper()}_ORDERED_LAYERS=({' '.join(map(str, layers))})")
    lines.append("")
    return "\n".join(lines)


def run_one_model(
    key: str,
    display_title: str,
    tda_dir: Path,
    compute_baseline_vs_epochs: Any,
    order_from_norm_avg: Any,
    order_from_norm_final: Any,
    write_orderings: Any,
    log: Callable[[str], None],
    skip_persistence: bool,
    codes_root: Path,
) -> dict[str, Any]:
    if key == "qwen_base":
        prefix = "qwen-base"
    elif key == "mistral":
        prefix = "mistral-7b-v03"
    else:
        prefix = "llama"

    base_nw, final_nw = discover_numpy_layout(tda_dir)

    log(f"\n{'=' * 70}\n  {display_title}\n  TDA root: {tda_dir.resolve()}\n{'=' * 70}\n")
    log(f"  [layout] pretrained numpy (epoch 0): {base_nw}")
    log(f"  [layout] final numpy (last epoch):   {final_nw}\n")

    timings: dict[str, float] = {}
    sections: list[str] = []

    if not skip_persistence:
        regenerate_persistence_tree(tda_dir, base_nw, codes_root, log, timings, key)
    else:
        log("  [persistence] skip (--skip-persistence)\n")

    base_diag = resolve_pretrained_diagrams(tda_dir)
    epoch_diag = discover_finetuning_epoch_diagrams(tda_dir)
    if not epoch_diag:
        raise FileNotFoundError(
            f"No finetuning persistence (epoch_k/SavedDiagrams, k>=1) under {tda_dir}/persistence/"
        )

    epoch_dirs_arg = [(lab, str(path)) for lab, path in epoch_diag]

    log(f"  [layout] pretrained diagrams (epoch 0): {base_diag}")
    log(f"  [layout] Wasserstein finetuning epochs: {[lab for lab, _ in epoch_diag]}\n")

    fd, tmp_all = tempfile.mkstemp(prefix="wass_all_", suffix=".csv")
    os.close(fd)
    try:
        t0 = time.perf_counter()
        df_all = compute_baseline_vs_epochs(str(base_diag), epoch_dirs_arg, tmp_all, PROJECTIONS)
        timings[f"{key}_wass_all_epochs_compute_s"] = time.perf_counter() - t0
    finally:
        try:
            os.unlink(tmp_all)
        except OSError:
            pass

    if "Wasserstein H0" not in df_all.columns:
        raise KeyError("Wasserstein H0 missing from Wasserstein dataframe")

    t0 = time.perf_counter()
    w_avg = _wass_order_mean(df_all)
    timings[f"{key}_wass_mean_order_s"] = time.perf_counter() - t0
    sections.append(_fmt_wass_block(f"### {prefix} — Wasserstein H0 — mean over epochs", w_avg))
    log(
        f"  [timings] {key} wass all-epochs compute: {timings[f'{key}_wass_all_epochs_compute_s']:.3f}s, "
        f"mean-rank: {timings[f'{key}_wass_mean_order_s']:.3f}s"
    )

    t0 = time.perf_counter()
    w_fin, last_lab = _wass_order_final(df_all)
    timings[f"{key}_wass_final_order_s"] = time.perf_counter() - t0
    sections.append(_fmt_wass_block(f"### {prefix} — Wasserstein H0 — final ({last_lab})", w_fin))
    log(f"  [timings] {key} wass final rank: {timings[f'{key}_wass_final_order_s']:.3f}s")

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        res_norm_avg, meta_norm_avg = order_from_norm_avg(
            str(base_nw), str(tda_dir), projections=PROJECTIONS, output_file=None, label=None
        )
        norm_avg_txt = write_orderings(res_norm_avg, meta_norm_avg, output_file=None, append_to=None)
    cap = buf.getvalue()
    timings[f"{key}_norm_avg_s"] = time.perf_counter() - t0
    sections.append(f"### {prefix} — Norm L2 (mean over epochs)\n{cap}{norm_avg_txt}\n")
    log(f"  [timings] {key} norm_avg: {timings[f'{key}_norm_avg_s']:.3f}s")

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        res_nf, meta_nf = order_from_norm_final(
            str(base_nw), str(final_nw), projections=PROJECTIONS, output_file=None, label=None
        )
        norm_fin_txt = write_orderings(res_nf, meta_nf, output_file=None, append_to=None)
    cap = buf.getvalue()
    timings[f"{key}_norm_final_s"] = time.perf_counter() - t0
    sections.append(f"### {prefix} — Norm L2 (final)\n{cap}{norm_fin_txt}\n")
    log(f"  [timings] {key} norm_final: {timings[f'{key}_norm_final_s']:.3f}s")

    return {
        "display": display_title,
        "prefix": prefix,
        "tda_dir": str(tda_dir.resolve()),
        "epochs": len(epoch_diag),
        "final_epoch_label": epoch_diag[-1][0],
        "timings": timings,
        "sections": sections,
    }


def _fmt_timing_table(rows: dict[str, dict[str, float]]) -> str:
    lines: list[str] = ["## Timings (s)", ""]
    for model_key, tdict in rows.items():
        lines.append(f"### {model_key}")
        for k in sorted(tdict.keys()):
            lines.append(f"  {k}: {tdict[k]:.6f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--llama-tda-dir", type=Path, default=NW_ROOT / "analysis/tda/gsm8k-llama-tda-results")
    parser.add_argument("--qwen-tda-dir", type=Path, default=NW_ROOT / "analysis/tda/gsm8k-qwen-base-tda-results")
    parser.add_argument("--llama-pretrained-hf", default=DEFAULT_LLAMA_PRETRAINED_HF)
    parser.add_argument("--llama-finetune-dir", type=Path, default=DEFAULT_LLAMA_FINETUNE_DIR)
    parser.add_argument("--qwen-pretrained-hf", default=DEFAULT_QWEN_PRETRAINED_HF)
    parser.add_argument("--qwen-finetune-dir", type=Path, default=DEFAULT_QWEN_FINETUNE_DIR)
    parser.add_argument("--mistral-tda-dir", type=Path, default=DEFAULT_MISTRAL_TDA_DIR)
    parser.add_argument("--mistral-pretrained-hf", default=DEFAULT_MISTRAL_PRETRAINED_HF)
    parser.add_argument("--mistral-finetune-dir", type=Path, default=DEFAULT_MISTRAL_FINETUNE_DIR)
    parser.add_argument("--codes-root", type=Path, default=DEFAULT_CODES)
    parser.add_argument("--skip-persistence", action="store_true")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-mistral", action="store_true")
    args = parser.parse_args()

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    _prepend_sys_path(args.codes_root)
    from compute_wasserstein import compute_baseline_vs_epochs  # noqa: E402
    from order_layers_by_norm import (  # noqa: E402
        order_from_norm_avg,
        order_from_norm_final,
        write_orderings,
    )

    collect: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        collect.append(msg)

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def write_model_file(
        subdir: str,
        title: str,
        timing_key: str,
        pretrained_hf: str,
        finetune_dir: Path,
        tda_dir: Path,
        info: dict[str, Any],
        progress_lines: list[str],
    ) -> Path:
        sub = out_root / subdir
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / "tda_timing_and_orderings.txt"
        body = "\n".join(
            [
                "=" * 72,
                title,
                f"Generated (UTC): {started}",
                "=" * 72,
                "",
                f"pretrained_hf: {pretrained_hf}",
                f"finetune_dir: {finetune_dir.resolve()}",
                f"tda_dir: {tda_dir.resolve()}",
                "",
                "--- log ---",
                "",
                "\n".join(progress_lines).rstrip(),
                "",
                _fmt_timing_table({timing_key: info["timings"]}),
                "",
                "## Layer orderings",
                "",
                "\n\n".join(info["sections"]),
                "",
            ]
        )
        path.write_text(body, encoding="utf-8")
        return path

    if not args.skip_llama:
        log("\n>>> Llama …")
        info = run_one_model(
            "llama",
            "Llama (gsm8k TDA)",
            args.llama_tda_dir.resolve(),
            compute_baseline_vs_epochs,
            order_from_norm_avg,
            order_from_norm_final,
            write_orderings,
            log,
            skip_persistence=args.skip_persistence,
            codes_root=args.codes_root.resolve(),
        )
        prog = collect.copy()
        collect.clear()
        p = write_model_file(
            "llama",
            "Llama — TDA timing and layer orderings",
            "llama",
            args.llama_pretrained_hf,
            args.llama_finetune_dir.resolve(),
            args.llama_tda_dir.resolve(),
            info,
            prog,
        )
        print(f"\n✅ Wrote: {p}", flush=True)

    if not args.skip_qwen:
        log("\n>>> Qwen-base …")
        info = run_one_model(
            "qwen_base",
            "Qwen3-8B-Base (gsm8k TDA)",
            args.qwen_tda_dir.resolve(),
            compute_baseline_vs_epochs,
            order_from_norm_avg,
            order_from_norm_final,
            write_orderings,
            log,
            skip_persistence=args.skip_persistence,
            codes_root=args.codes_root.resolve(),
        )
        prog = collect.copy()
        collect.clear()
        p = write_model_file(
            "qwen-base",
            "Qwen-base — TDA timing and layer orderings",
            "qwen_base",
            args.qwen_pretrained_hf,
            args.qwen_finetune_dir.resolve(),
            args.qwen_tda_dir.resolve(),
            info,
            prog,
        )
        print(f"\n✅ Wrote: {p}", flush=True)

    if not args.skip_mistral:
        log("\n>>> Mistral-7B-v0.3 …")
        info = run_one_model(
            "mistral",
            "Mistral-7B-v0.3 (gsm8k TDA)",
            args.mistral_tda_dir.resolve(),
            compute_baseline_vs_epochs,
            order_from_norm_avg,
            order_from_norm_final,
            write_orderings,
            log,
            skip_persistence=args.skip_persistence,
            codes_root=args.codes_root.resolve(),
        )
        prog = collect.copy()
        collect.clear()
        p = write_model_file(
            "mistral-7b-v03",
            "Mistral-7B-v0.3 — TDA timing and layer orderings",
            "mistral",
            args.mistral_pretrained_hf,
            args.mistral_finetune_dir.resolve(),
            args.mistral_tda_dir.resolve(),
            info,
            prog,
        )
        print(f"\n✅ Wrote: {p}", flush=True)


if __name__ == "__main__":
    main()
