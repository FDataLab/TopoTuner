#!/usr/bin/env python3
"""
Export K/Q/V/O attention weights as .npy for SST2 / IMDB / MMLU full finetunes (9 runs).

Layout (matches GSM8K TDA + ``tda_ranking_eltwise_weights.py``)::

  <out-root>/<dataset>-<model-short>-tda-results/baseline/numpy_weights/layer{L}_{q,k,v,o}.npy
  <out-root>/<dataset>-<model-short>-tda-results/epoch_6/numpy_weights/...

``out-root`` defaults to ``.../exploration-finetuning/numpy_weights`` (next to this ``scripts/`` dir).

Checkpoint resolution (same idea as eval in ``run_sst2_imdb_mmlu_finetune_run1.sh``):

* If ``epoch_weights/checkpoint-epoch-*`` exists: **baseline** = lowest epoch (usually 0),
  **epoch_6** = highest epoch (usually 6).
* Else: **baseline** = smallest-step ``checkpoint-*``, **epoch_6** = largest-step ``checkpoint-*``
  (HF step folders under the run root).

Re-run safely: by default skips a side (baseline or epoch_6) if that ``numpy_weights`` folder
already contains ``layer*_*.npy``. Use ``--force`` to overwrite.

The **paper table** (avg K/Q/V/O ranks by dataset × model) is filled manually from
``tda_ranking_eltwise_weights.py`` (or similar); this script only materializes the npy tree.

Examples::

  # All nine (dataset × model) where checkpoints exist
  ./export_sst2_imdb_mmlu_numpy_weights.py

  ./export_sst2_imdb_mmlu_numpy_weights.py --only-dataset imdb --only-model llama
  CKPT_ROOT=/path/to/exploration-finetuning/checkpoints \\
    ./export_sst2_imdb_mmlu_numpy_weights.py

  # If `export_attn_npy_from_checkpoint` fails (system NumPy 2 + distro torch), use a venv
  # with matching torch+NumPy, e.g. ``topo-env/bin/python3`` at repo root.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_EPOCH_GLOB = re.compile(r"^checkpoint-epoch-(\d+)$")
_STEP_GLOB = re.compile(r"^checkpoint-(\d+)$")

# Order matches the orchestration script: three datasets × three model shorts.
DATASETS = ("sst2", "imdb", "mmlu")
MODEL_SHORTS = ("qwen-base", "llama", "mistral-7b-v03")


def _default_codes_root() -> Path:
    return Path(os.environ.get("CODES_ROOT", Path(__file__).resolve().parents[2]))


def _default_ckpt_root(exploration_root: Path) -> Path:
    env = os.environ.get("CKPT_ROOT", "").strip()
    if env:
        return Path(env)
    # Keep defaults environment/repo-relative for anonymous portability.
    return exploration_root / "checkpoints"


def _sort_epoch_dirs(dirs: list[Path]) -> list[Path]:
    keyed: list[tuple[int, Path]] = []
    for p in dirs:
        m = _EPOCH_GLOB.match(p.name)
        if m:
            keyed.append((int(m.group(1)), p))
    keyed.sort(key=lambda t: t[0])
    return [p for _, p in keyed]


def _sort_step_dirs(dirs: list[Path]) -> list[Path]:
    keyed: list[tuple[int, Path]] = []
    for p in dirs:
        m = _STEP_GLOB.match(p.name)
        if m:
            keyed.append((int(m.group(1)), p))
    keyed.sort(key=lambda t: t[0])
    return [p for _, p in keyed]


def find_baseline_and_final(run_root: Path) -> tuple[Path, Path]:
    """
    Return (baseline_ckpt_dir, final_ckpt_dir) for a finetune output directory.
    """
    ew = run_root / "epoch_weights"
    if ew.is_dir():
        epoch_dirs = [p for p in ew.iterdir() if p.is_dir() and _EPOCH_GLOB.match(p.name)]
        if len(epoch_dirs) >= 2:
            s = _sort_epoch_dirs(epoch_dirs)
            return s[0], s[-1]
        if len(epoch_dirs) == 1:
            raise FileNotFoundError(
                f"Only one epoch checkpoint under {ew}; need at least two for baseline vs final."
            )

    step_dirs = [p for p in run_root.iterdir() if p.is_dir() and _STEP_GLOB.match(p.name)]
    if len(step_dirs) < 2:
        raise FileNotFoundError(
            f"Need at least two checkpoint-* dirs under {run_root} (or epoch_weights with 2+ epochs)."
        )
    s = _sort_step_dirs(step_dirs)
    return s[0], s[-1]


def _npy_count(npy_dir: Path) -> int:
    if not npy_dir.is_dir():
        return 0
    return sum(1 for f in npy_dir.glob("layer*_*.npy") if f.is_file())


def run_export(
    export_script: Path,
    ckpt_dir: Path,
    out_npy_dir: Path,
    float32: bool,
) -> None:
    cmd = [
        sys.executable,
        str(export_script),
        str(ckpt_dir),
        "--out-dir",
        str(out_npy_dir),
    ]
    if float32:
        cmd.append("--float32")
    subprocess.check_call(cmd)


def main() -> None:
    exploration_root = Path(__file__).resolve().parent.parent
    default_out = exploration_root / "numpy_weights"

    ap = argparse.ArgumentParser(
        description="Export Q/K/V/O npy for SST2/IMDB/MMLU full runs into exploration-finetuning/numpy_weights/"
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=default_out,
        help=f"Parent for <dataset>-<model>-tda-results/ (default: {default_out})",
    )
    ap.add_argument(
        "--ckpt-root",
        type=Path,
        default=None,
        help="Override checkpoint root (default: CKPT_ROOT env or data mirror or ./checkpoints)",
    )
    ap.add_argument(
        "--export-script",
        type=Path,
        default=_default_codes_root() / "analysis" / "export_attn_npy_from_checkpoint.py",
        help="Path to export_attn_npy_from_checkpoint.py",
    )
    ap.add_argument(
        "--method",
        default="full",
        help="Subdir under <model>/<dataset>/ (default: full)",
    )
    ap.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number path segment run{N} (default: 1)",
    )
    ap.add_argument(
        "--only-dataset",
        choices=DATASETS,
        action="append",
        default=None,
        help="Restrict to dataset(s); repeatable",
    )
    ap.add_argument(
        "--only-model",
        choices=MODEL_SHORTS,
        action="append",
        default=None,
        help="Restrict to model short(s); repeatable",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even if numpy_weights already has layer*_*.npy",
    )
    ap.add_argument(
        "--float32",
        action="store_true",
        help="Pass through to export script (default float16, same as GSM8K TDA)",
    )
    args = ap.parse_args()

    export_script = args.export_script.resolve()
    if not export_script.is_file():
        raise SystemExit(f"Missing export script: {export_script}")

    ckpt_root = (args.ckpt_root or _default_ckpt_root(exploration_root)).resolve()
    out_root: Path = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = args.only_dataset if args.only_dataset else list(DATASETS)
    models = args.only_model if args.only_model else list(MODEL_SHORTS)
    run_seg = f"run{args.run}"

    ok = 0
    skipped = 0
    failed: list[str] = []

    for dataset in datasets:
        for model_short in models:
            tag = f"{dataset}-{model_short}"
            run_dir = ckpt_root / model_short / dataset / args.method / run_seg
            tda_name = f"{dataset}-{model_short}-tda-results"
            dest = out_root / tda_name
            base_npy = dest / "baseline" / "numpy_weights"
            final_npy = dest / "epoch_6" / "numpy_weights"

            if not run_dir.is_dir():
                print(f"[skip] no checkpoint dir: {run_dir}")
                skipped += 1
                continue

            try:
                b_ckpt, f_ckpt = find_baseline_and_final(run_dir)
            except FileNotFoundError as e:
                print(f"[skip] {tag}: {e}")
                skipped += 1
                continue

            try:
                if not args.force and _npy_count(base_npy) > 0:
                    print(f"[skip] baseline npy already present: {base_npy}")
                else:
                    base_npy.parent.mkdir(parents=True, exist_ok=True)
                    run_export(export_script, b_ckpt, base_npy, args.float32)
                    print(f"[ok] {tag} baseline <- {b_ckpt.name} -> {base_npy}")

                if not args.force and _npy_count(final_npy) > 0:
                    print(f"[skip] epoch_6 npy already present: {final_npy}")
                else:
                    final_npy.parent.mkdir(parents=True, exist_ok=True)
                    run_export(export_script, f_ckpt, final_npy, args.float32)
                    print(f"[ok] {tag} epoch_6 <- {f_ckpt.name} -> {final_npy}")

                if _npy_count(base_npy) > 0 and _npy_count(final_npy) > 0:
                    ok += 1
                else:
                    failed.append(f"{tag}: missing npy after export (baseline={_npy_count(base_npy)} final={_npy_count(final_npy)})")
            except subprocess.CalledProcessError as e:
                failed.append(f"{tag}: export failed ({e})")
            except Exception as e:
                failed.append(f"{tag}: {e}")

    print()
    print(f"Done. ckpt_root={ckpt_root}  out_root={out_root}")
    print(f"  completed run sets: {ok}  skipped/missing: {skipped}  failures: {len(failed)}")
    for line in failed:
        print(f"  ERROR: {line}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
