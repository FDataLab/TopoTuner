#!/usr/bin/env python3
"""
Export **Q/K/V/O** ``.npy`` for **GSM8K full finetunes** into the same TDA layout as SST2/IMDB/MMLU:

  <out-root>/gsm8k-<model-tag>-tda-results/baseline/numpy_weights/layer{L}_{q,k,v,o}.npy
  <out-root>/gsm8k-<model-tag>-tda-results/epoch_6/numpy_weights/...

Uses ``codes/analysis/export_attn_npy_from_checkpoint.py`` on HuggingFace ``safetensors``
(keys ``model.layers.*.self_attn.{q,k,v,o}_proj.weight``).

Checkpoint roots tried (first existing wins), matching typical exploration-finetuning layout::

  <ckpt-root>/<model-short>/gsm8k/full/run{N}
  <ckpt-root>/<model-short>/full/run{N}    # GSM8K-only full fine-tune folder

Baseline / final step resolution matches ``export_sst2_imdb_mmlu_numpy_weights.py``.

Examples::

  cd .../exploration-finetuning/scripts
  ./export_gsm8k_full_ft_numpy_weights.py --force

  CKPT_ROOT=/path/to/checkpoints ./export_gsm8k_full_ft_numpy_weights.py --only-model llama --float32

See also: ``export_sst2_imdb_mmlu_numpy_weights.py`` (already writes full KQVO for IMDB/SST2/MMLU).
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

MODEL_SHORTS = ("qwen-base", "llama", "mistral-7b-v03")

TDA_FOLDER_NAME = {
    "llama": "gsm8k-llama-tda-results",
    "qwen-base": "gsm8k-qwen-base-tda-results",
    "mistral-7b-v03": "gsm8k-mistral-7b-v03-tda-results",
}

DEFAULT_RUN_BY_MODEL = {
    "llama": 1,
    "qwen-base": 3,
    "mistral-7b-v03": 1,
}


def _default_codes_root() -> Path:
    return Path(os.environ.get("CODES_ROOT", Path(__file__).resolve().parents[3] / "codes"))


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


def resolve_gsm8k_run_root(
    ckpt_root: Path, model_short: str, method: str, run_seg: str
) -> Path | None:
    candidates = [
        ckpt_root / model_short / "gsm8k" / method / run_seg,
        ckpt_root / model_short / method / run_seg,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _npy_count(npy_dir: Path) -> int:
    if not npy_dir.is_dir():
        return 0
    return sum(1 for f in npy_dir.glob("layer*_*.npy") if f.is_file())


def run_export(export_script: Path, ckpt_dir: Path, out_npy_dir: Path, float32: bool) -> None:
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
        description="Export Q/K/V/O npy for GSM8K full FT into gsm8k-*-tda-results/"
    )
    ap.add_argument("--out-root", type=Path, default=default_out)
    ap.add_argument("--ckpt-root", type=Path, default=None)
    ap.add_argument(
        "--export-script",
        type=Path,
        default=_default_codes_root() / "analysis" / "export_attn_npy_from_checkpoint.py",
    )
    ap.add_argument("--method", default="full")
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="Uniform run{N} for all models (overrides per-model defaults if set).",
    )
    ap.add_argument("--only-model", choices=MODEL_SHORTS, action="append", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--float32", action="store_true")
    args = ap.parse_args()

    export_script = args.export_script.resolve()
    if not export_script.is_file():
        raise SystemExit(f"Missing export script: {export_script}")

    ckpt_root = (args.ckpt_root or _default_ckpt_root(exploration_root)).resolve()
    out_root: Path = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    models = args.only_model if args.only_model else list(MODEL_SHORTS)

    ok = 0
    skipped = 0
    failed: list[str] = []

    for model_short in models:
        run_num = args.run if args.run is not None else DEFAULT_RUN_BY_MODEL[model_short]
        run_seg = f"run{run_num}"
        tag = f"gsm8k-{model_short}"
        tda_name = TDA_FOLDER_NAME[model_short]
        dest = out_root / tda_name
        base_npy = dest / "baseline" / "numpy_weights"
        final_npy = dest / "epoch_6" / "numpy_weights"

        run_dir = resolve_gsm8k_run_root(ckpt_root, model_short, args.method, run_seg)
        if run_dir is None:
            print(f"[skip] no checkpoint dir for {tag}: tried …/{model_short}/gsm8k/… and …/{model_short}/full/…")
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
                failed.append(
                    f"{tag}: missing npy after export (baseline={_npy_count(base_npy)} final={_npy_count(final_npy)})"
                )
        except subprocess.CalledProcessError as e:
            failed.append(f"{tag}: export failed ({e})")
        except Exception as e:
            failed.append(f"{tag}: {e}")

    print()
    print(f"Done. ckpt_root={ckpt_root}  out_root={out_root}")
    print(f"  completed: {ok}  skipped: {skipped}  failures: {len(failed)}")
    for line in failed:
        print(f"  ERROR: {line}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
