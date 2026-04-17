#!/usr/bin/env python3
"""
GSM8K-style **V/O-only** TDA pipeline for one finetuning run:

  1) **Export** pretrained + each ``checkpoint-*`` → ``numpy/epoch_{0..6}/layer*_{{v,o}}.npy``
  2) **Persistence** → ``persistence/epoch_k/SavedDiagrams/*.pkl`` (``generate_persistence.py``, **H0 only**, ``--maxdim 0``)
  3) **Wasserstein** (baseline = epoch_0 vs epoch_1..6, **H0 only**, ``compute_wasserstein.py --h0-only``) → ``wasserstein_results.csv``

Uses the same checkpoint layout as ``norm_llama_full_layer_epoch_curves.py`` (Llama ``run1``,
Qwen-base ``run3``, full + norm-freeze sweep).

Examples (global ``--nw-root`` must come **before** the subcommand)::

  python scripts/gsm8k_wass_vo_pipeline.py --nw-root . list
  python scripts/gsm8k_wass_vo_pipeline.py --nw-root . run --family llama --tag full
  python scripts/gsm8k_wass_vo_pipeline.py --nw-root . export --family llama --tag full

Env: set ``OMP_NUM_THREADS=1`` (and OPENBLAS/MKL) before ``run`` for sane parallelism.

Each ``export`` / ``persist`` / ``wass`` / ``run`` writes a trace log (default:
``eval/split/weight_vs_baseline/wass/{family}/{tag}/pipeline.log``), including
subprocess output from persistence and Wasserstein steps.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
NW_ROOT_DEFAULT = SCRIPT_DIR.parent
TOPO_TDA = Path(os.environ.get("TOPO_CODES_TDA", "/home/kadir/topo/codes/tda")).resolve()
GEN_PERSISTENCE = TOPO_TDA / "generate_persistence.py"
COMPUTE_WASS = TOPO_TDA / "compute_wasserstein.py"

_LOG = logging.getLogger("gsm8k_wass_vo")


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    _LOG.addHandler(fh)
    _LOG.addHandler(ch)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False
    _LOG.info("log file: %s (UTC %s)", log_path, datetime.now(timezone.utc).isoformat())


def _load_norm_helpers():
    norm_path = SCRIPT_DIR / "norm_llama_full_layer_epoch_curves.py"
    spec = importlib.util.spec_from_file_location("norm_helpers", str(norm_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def resolve_run_dir(nw: Path, family: str, tag: str, ckpt_root: Path) -> Path:
    h = _load_norm_helpers()
    rid = h.RUN_ID[family]
    root = ckpt_root / family
    if tag == "full":
        return root / "full" / rid
    m_norm = __import__("re").fullmatch(r"norm_(high|low)_(\d+)", tag)
    if m_norm:
        side, k = m_norm.group(1), m_norm.group(2)
        return root / "norm-freeze" / f"{side}-{k}" / rid

    m_wass = __import__("re").fullmatch(r"wass_(high|low)_(\d+)", tag)
    if m_wass:
        side, k = m_wass.group(1), m_wass.group(2)
        # Llama uses *-frozen naming; qwen-base currently uses plain high-3/low-3.
        cands = [
            root / "wass-freeze" / f"{side}-{k}-frozen" / rid,
            root / "wass-freeze" / f"{side}-{k}" / rid,
        ]
        for c in cands:
            if c.is_dir():
                return c
        # Return first candidate for a clearer missing-dir error later.
        return cands[0]

    raise SystemExit(f"Unknown tag {tag!r}; use full, norm_high_3, wass_high_3, etc.")


def work_root(nw: Path, family: str, tag: str) -> Path:
    return (nw / "eval" / "split" / "weight_vs_baseline" / "wass" / family / tag).resolve()


def family_baseline_root(nw: Path, family: str) -> Path:
    return (nw / "eval" / "split" / "weight_vs_baseline" / "wass" / family / "baseline").resolve()


def export_numpy_vo(
    nw: Path,
    family: str,
    run_dir: Path,
    out_numpy: Path,
    baseline_dir: str,
    epoch_ckpts: List[Tuple[int, Path]],
    log: Optional[logging.Logger] = None,
    include_baseline_epoch0: bool = True,
) -> None:
    h = _load_norm_helpers()
    if not epoch_ckpts and not include_baseline_epoch0:
        raise SystemExit(f"No checkpoints under {run_dir}")

    def export_dir(weights_dir: str, epoch_label: int) -> None:
        d = out_numpy / f"epoch_{epoch_label}"
        d.mkdir(parents=True, exist_ok=True)
        wm = h.load_weight_map(weights_dir)
        cache = h.ShardCache(weights_dir)
        n = 0
        for name in sorted(wm.keys()):
            proj = h.projection_type(name)
            if proj not in h.VO_PROJS:
                continue
            li = h.layer_index(name)
            if li is None:
                continue
            try:
                w = cache.get(wm[name])[name].to(torch.float32).cpu().numpy()
            except (FileNotFoundError, KeyError):
                continue
            fn = d / f"layer{li}_{proj}.npy"
            np.save(fn, w.astype(np.float32, copy=False))
            n += 1
        msg = f"[export epoch_{epoch_label}] {n} tensors -> {d}"
        if log:
            log.info(msg)
        else:
            print(f"  {msg}", flush=True)

    if epoch_ckpts:
        sample = str(epoch_ckpts[0][1])
        if h._has_adapter_only_checkpoint(sample):
            raise SystemExit("Adapter-only checkpoint; export skipped (merge weights first).")

    if include_baseline_epoch0:
        export_dir(baseline_dir, 0)
    for ep, p in sorted(epoch_ckpts, key=lambda x: x[0]):
        wdir = str(p)
        if not h._has_weights(wdir):
            msg = f"[skip epoch {ep}] no full weights: {p}"
            if log:
                log.warning(msg)
            else:
                print(f"  {msg}", flush=True)
            continue
        export_dir(wdir, ep)


def ensure_shared_baseline(nw: Path, family: str, baseline_dir: str, log: logging.Logger) -> Path:
    base = family_baseline_root(nw, family)
    np_epoch0 = base / "numpy" / "epoch_0"
    pe_epoch0 = base / "persistence" / "epoch_0" / "SavedDiagrams"
    if pe_epoch0.is_dir():
        log.info("[shared-baseline] reuse %s", pe_epoch0)
        return pe_epoch0

    # Guard against 11 parallel jobs trying to build the same baseline simultaneously.
    lock = base / ".baseline.lock"
    while True:
        try:
            lock.mkdir(parents=True, exist_ok=False)
            have_lock = True
            break
        except FileExistsError:
            # Another worker is building baseline; wait until it appears.
            if pe_epoch0.is_dir():
                log.info("[shared-baseline] ready while waiting: %s", pe_epoch0)
                return pe_epoch0
            import time

            log.info("[shared-baseline] waiting on lock %s", lock)
            time.sleep(2.0)
            continue

    try:
        if pe_epoch0.is_dir():
            log.info("[shared-baseline] reuse %s", pe_epoch0)
            return pe_epoch0
        if not np_epoch0.is_dir():
            log.info("[shared-baseline] exporting epoch_0 numpy -> %s", np_epoch0)
            export_numpy_vo(
                nw=nw,
                family=family,
                run_dir=Path("."),
                out_numpy=base / "numpy",
                baseline_dir=baseline_dir,
                epoch_ckpts=[],
                log=log,
                include_baseline_epoch0=True,
            )
        else:
            log.info("[shared-baseline] found existing numpy epoch_0: %s", np_epoch0)

        log.info("[shared-baseline] generating persistence epoch_0")
        run_generate_persistence(np_epoch0, base / "persistence" / "epoch_0", log)
        if not pe_epoch0.is_dir():
            raise SystemExit(f"Failed to build shared baseline diagrams: {pe_epoch0}")
        return pe_epoch0
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def run_generate_persistence(numpy_epoch: Path, persist_epoch: Path, log: logging.Logger) -> None:
    persist_epoch.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(GEN_PERSISTENCE),
        "--input-dir",
        str(numpy_epoch),
        "--output-dir",
        str(persist_epoch),
        "--projections",
        "v,o",
        "--maxdim",
        "0",
    ]
    log.info("RUN %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        log.info("stdout:\n%s", proc.stdout.rstrip())
    if proc.stderr:
        log.warning("stderr:\n%s", proc.stderr.rstrip())
    if proc.returncode != 0:
        log.error("exit code %s", proc.returncode)
        proc.check_returncode()


def run_persistence_all(work: Path, log: logging.Logger) -> None:
    np_root = work / "numpy"
    pe_root = work / "persistence"
    for child in sorted(np_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("epoch_"):
            continue
        label = child.name
        out = pe_root / label
        log.info("[persistence] %s", label)
        run_generate_persistence(child, out, log)


def run_wasserstein(
    work: Path,
    csv_out: Path,
    log: logging.Logger,
    baseline_sd_override: Optional[Path] = None,
) -> None:
    base_sd = (
        baseline_sd_override.resolve()
        if baseline_sd_override is not None
        else (work / "persistence" / "epoch_0" / "SavedDiagrams").resolve()
    )
    if not base_sd.is_dir():
        raise SystemExit(f"Missing baseline diagrams: {base_sd}")
    parts: List[str] = []
    for k in range(1, 7):
        sd = (work / "persistence" / f"epoch_{k}" / "SavedDiagrams").resolve()
        if not sd.is_dir():
            raise SystemExit(f"Missing {sd}")
        parts.append(f"epoch_{k}:{sd}")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(COMPUTE_WASS),
        "--baseline-dir",
        str(base_sd),
        "--epoch-dirs",
        ",".join(parts),
        "--output",
        str(csv_out),
        "--projections",
        "v,o",
        "--h0-only",
    ]
    log.info("RUN %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        log.info("stdout:\n%s", proc.stdout.rstrip())
    if proc.stderr:
        log.warning("stderr:\n%s", proc.stderr.rstrip())
    proc.check_returncode()
    log.info("[wasserstein] wrote %s", csv_out)


def cmd_list(args: argparse.Namespace) -> None:
    h = _load_norm_helpers()
    nw = Path(args.nw_root).resolve()
    ck = Path(args.ckpt_root).resolve() if args.ckpt_root else (nw / "checkpoints").resolve()
    for fam in ("llama", "qwen-base"):
        for tag, _title, run_dir in h.sweep_specs_for_family(fam, ck):
            print(f"{fam}\t{tag}\t{run_dir}")


def _work_and_log(args: argparse.Namespace) -> Tuple[Path, Path]:
    nw = Path(args.nw_root).resolve()
    work = work_root(nw, args.family, args.tag)
    log_path = Path(args.log_file).resolve() if args.log_file else work / "pipeline.log"
    configure_logging(log_path)
    return work, log_path


def cmd_export(args: argparse.Namespace) -> None:
    h = _load_norm_helpers()
    nw = Path(args.nw_root).resolve()
    ck = Path(args.ckpt_root).resolve() if args.ckpt_root else (nw / "checkpoints").resolve()
    family = args.family
    work, log_path = _work_and_log(args)
    run_dir = resolve_run_dir(nw, family, args.tag, ck)
    if not run_dir.is_dir():
        raise SystemExit(f"Missing run dir: {run_dir}")
    bl = h.pick_pretrained_dir(family, args.baseline_override, nw)
    if not bl:
        raise SystemExit("No baseline/pretrained weights")
    cps = h.discover_epoch_checkpoints(run_dir)
    np_root = work / "numpy"
    _LOG.info("[export] %s %s -> %s", family, args.tag, np_root)
    use_shared_baseline = not args.no_shared_baseline
    if use_shared_baseline:
        ensure_shared_baseline(nw, family, bl, _LOG)
    export_numpy_vo(
        nw,
        family,
        run_dir,
        np_root,
        bl,
        cps,
        _LOG,
        include_baseline_epoch0=not use_shared_baseline,
    )
    _LOG.info("export finished (%s)", log_path)


def cmd_persist(args: argparse.Namespace) -> None:
    work, log_path = _work_and_log(args)
    _LOG.info("[persistence] workdir %s", work)
    run_persistence_all(work, _LOG)
    _LOG.info("persistence finished (%s)", log_path)


def cmd_wass(args: argparse.Namespace) -> None:
    work, log_path = _work_and_log(args)
    csv_out = work / "wasserstein_results.csv"
    _LOG.info("[wasserstein] workdir %s", work)
    baseline_sd = None
    if not args.no_shared_baseline:
        baseline_sd = family_baseline_root(Path(args.nw_root).resolve(), args.family) / "persistence" / "epoch_0" / "SavedDiagrams"
    run_wasserstein(work, csv_out, _LOG, baseline_sd_override=baseline_sd)
    _LOG.info("wasserstein finished (%s)", log_path)


def cmd_run(args: argparse.Namespace) -> None:
    cmd_export(args)
    cmd_persist(args)
    cmd_wass(args)
    _LOG.info("run finished (all steps)")


def main() -> None:
    if not GEN_PERSISTENCE.is_file():
        raise SystemExit(f"Missing {GEN_PERSISTENCE} (set TOPO_CODES_TDA)")
    if not COMPUTE_WASS.is_file():
        raise SystemExit(f"Missing {COMPUTE_WASS}")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nw-root", type=Path, default=NW_ROOT_DEFAULT)
    ap.add_argument("--ckpt-root", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Print family, tag, run_dir for 22 experiments")
    p_list.set_defaults(func=cmd_list)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--family", choices=("llama", "qwen-base"), required=True)
        p.add_argument(
            "--tag",
            required=True,
            help="full, norm_high_3, norm_low_15, wass_high_3, wass_low_15, ...",
        )
        p.add_argument("--baseline-override", type=str, default=None)
        p.add_argument(
            "--no-shared-baseline",
            action="store_true",
            help="Do not reuse family baseline epoch_0; export/build epoch_0 inside each experiment folder.",
        )
        p.add_argument(
            "--log-file",
            type=Path,
            default=None,
            help="Append log here (default: wass/{family}/{tag}/pipeline.log under eval/split/weight_vs_baseline)",
        )

    p_e = sub.add_parser("export", help="Only step 1 (numpy v/o)")
    add_common(p_e)
    p_e.set_defaults(func=cmd_export)

    p_p = sub.add_parser("persist", help="Only step 2 (expects numpy/ already)")
    add_common(p_p)
    p_p.set_defaults(func=cmd_persist)

    p_w = sub.add_parser("wass", help="Only step 3 (expects persistence/ already)")
    add_common(p_w)
    p_w.set_defaults(func=cmd_wass)

    p_r = sub.add_parser("run", help="export + persist + wass")
    add_common(p_r)
    p_r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
