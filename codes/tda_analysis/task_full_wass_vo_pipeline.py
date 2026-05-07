#!/usr/bin/env python3
"""
**Full-finetune TDA** for non-sweep runs: one checkpoint tree per (model family × task).

**Tasks (datasets):** ``sst2``, ``imdb``, ``mmlu``, ``gsm8k``  
**Families:** ``llama``, ``qwen-base``, ``mistral-7b-v03`` (same as ``gsm8k_wass_vo_pipeline``).

**Checkpoint layout** (under ``--nw-root/checkpoints`` or ``--ckpt-root``)::

  {family}/full/{run_id}              → task ``gsm8k`` only
  {family}/{task}/full/{run_id}        → task ``sst2`` | ``imdb`` | ``mmlu``

``run_id`` is usually ``run1`` (Llama, Mistral) or, for Qwen, ``run3`` on GSM8K only. Per-task
Qwen finetunes for SST-2 / IMDB / MMLU use ``run1`` on disk (see ``task_run_id``).

**Work directory** (numpy + persistence + ``wasserstein_results.csv``)::

  {nw_root}/eval/split/weight_vs_baseline/wass_full_by_task/{family}/{task}/

This reuses the same export → persistence (H0) → Wasserstein (H0) implementation as
``gsm8k_wass_vo_pipeline.py`` (imported as a module). Projections default to **k,q**
(see ``--projections``); pass ``v,o`` for the older V/O-only diagrams.

This **does not** use GSM8K wass/norm sweep tags (``wass_high_3``, ``norm_low_6``, …).

**Experiment count:** ``3 families × 4 tasks = 12`` full runs.

**How this relates to the old numbers**

- ``run_wass_vo_11x11.sh`` runs **11** tags (``full`` + 10 ``wass_*``) for **Llama** and
  **Qwen** only → **22** jobs (GSM8K freeze sweep, not the 12 task grid here).
- ``gsm8k_wass_vo_pipeline.py list`` uses the **norm** sweep from
  ``sweep_specs_for_family`` → **11** rows per family × **3** families → **33** (if you
  run all three; Mistral is included in that list).

**Parallelism (example, 12 jobs, up to 16 at a time)::

  python task_full_wass_vo_pipeline.py --nw-root "$NW" print-batch \
    | xargs -n1 -P 16 -I% bash -lc '%'

(``print-batch`` emits one full ``python … run --family … --task …`` line per experiment.)

**Skip vs overwrite:** ``run`` / ``batch-run`` **skip** an experiment when its workdir already
has a full chain: all required ``numpy/epoch_*``, ``persistence/epoch_*/SavedDiagrams/*.pkl``,
and a non-trivial ``wasserstein_results.csv``. **Partial** outputs are not complete, so the
pipeline **re-runs** all three steps and overwrites. Use ``--force`` on ``run`` or
``batch-run`` to always recompute.

**Step timings:** Each ``run`` logs one line with ``export_elapsed_s``, ``persist_elapsed_s``,
``wass_elapsed_s``, and ``total_elapsed_s``. Standalone ``export`` / ``persist`` / ``wass`` log a
single step each. Averages are not computed in code; parse logs if needed. To stop a stuck
batch: ``pkill -9 -f task_full_wass_vo_pipeline.py`` and, for orphan Ripser children,
``pkill -9 -f 'codes/tda/generate_persistence.py'`` (can affect any same-path run on the host).

**Env:** set ``OMP_NUM_THREADS=1`` (and OPENBLAS/MKL) for sane CPU use when many processes run.

Use your **fine-tuning / Torch venv**: activate it so ``VIRTUAL_ENV`` is set, or set
``PIPELINE_PYTHON=/path/to/env/bin/python``. The pipeline will ``exec`` into that interpreter when
the entrypoint was accidentally launched with system Python.

**Troubleshooting sparse layers (e.g. Mistral):** ``compute_wasserstein.py`` pairs files present under **both**
the baseline ``SavedDiagrams`` and each epoch's ``SavedDiagrams``. If the shared baseline directory
``eval/split/weight_vs_baseline/wass/<family>/baseline/persistence/epoch_0/SavedDiagrams`` lacks some layer PKLs while the matching ``baseline/numpy/epoch_0`` arrays exist, Ripser/persistence did not produce diagrams for those layers — **re-run persistence on that baseline numpy folder**, then re-run the Wasserstein step (or ``run … --force`` on the experiment).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def maybe_reexec_under_ml_python() -> None:
    """Use Torch-capable interpreter: ``PIPELINE_PYTHON`` or ``$VIRTUAL_ENV/bin/python``."""
    want: Path | None = None
    pip = os.environ.get("PIPELINE_PYTHON")
    if pip:
        p = Path(pip).expanduser().resolve()
        if not p.is_file():
            print(f"PIPELINE_PYTHON={pip!r}: file not found", file=sys.stderr)
            sys.exit(2)
        want = p
    else:
        ve = os.environ.get("VIRTUAL_ENV")
        if ve:
            cand = Path(ve).expanduser().resolve() / "bin" / "python"
            if cand.is_file():
                want = cand.resolve()
    if want is None:
        return
    cur = Path(sys.executable).resolve()
    if cur == want:
        return
    os.execv(str(want), [str(want), *sys.argv])


maybe_reexec_under_ml_python()

# Allow `python /path/to/task_full_wass_vo_pipeline.py` without installing a package
_TDA = Path(__file__).resolve().parent
if str(_TDA) not in sys.path:
    sys.path.insert(0, str(_TDA))
import gsm8k_wass_vo_pipeline as gwp  # noqa: E402

# Default run id per family (Qwen’s per-task SST-2/IMDB/MMLU use run1; see task_run_id).
RUN_ID = {
    "llama": "run1",
    "qwen-base": "run3",
    "mistral-7b-v03": "run1",
}
TASKS: Tuple[str, ...] = ("sst2", "imdb", "mmlu", "gsm8k")
FAMILIES: Tuple[str, ...] = ("llama", "qwen-base", "mistral-7b-v03")

# Total: len(FAMILIES) * len(TASKS) = 12
N_EXPERIMENTS = len(FAMILIES) * len(TASKS)


def task_run_id(family: str, task: str) -> str:
    """
    Finetune run directory name (e.g. ``run1`` / ``run3``) under ``.../full/<run_id>``.

    Qwen-base: GSM8K full FT lives under ``qwen-base/full/run3``; SST-2, IMDB, and MMLU
    use ``run1`` in this repo. Llama and Mistral always use ``run1``.
    """
    if family == "qwen-base" and task in ("sst2", "imdb", "mmlu"):
        return "run1"
    return RUN_ID[family]


def task_checkpoint_dir(ckpt_root: Path, family: str, task: str) -> Path:
    """
    Resolves the *full finetune* run directory for one (family, task).
    GSM8K lives at ``{family}/full/{run_id}``, not under ``{family}/gsm8k/...``.
    """
    rid = task_run_id(family, task)
    if task == "gsm8k":
        return (ckpt_root / family / "full" / rid).resolve()
    return (ckpt_root / family / task / "full" / rid).resolve()


def _add_force_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute numpy + persistence + Wasserstein even if the workdir looks complete (run / batch-run).",
    )


def _renumber_checkpoints_sequential(
    cps: List[Tuple[int, Path]], n_epochs: int
) -> List[Tuple[int, Path]]:
    """
    `norm_llama_full_layer_epoch_curves.discover_epoch_checkpoints` labels saves as
    ``(global_step // min_step, path)``, which can yield ``epoch_9``, ``epoch_74``,
    etc. The Wasserstein step in ``gsm8k_wass_vo_pipeline`` expects
    ``numpy/persistence/epoch_1..epoch_N`` for *consecutive* indices.

    We sort by the discovered index (monotonic in training order), take the **last**
    ``n_epochs`` saves when more than ``n_epochs`` exist, and renumber **1,…,N**.
    """
    if not cps:
        return []
    cps = sorted(cps, key=lambda x: x[0])
    if len(cps) < n_epochs:
        raise SystemExit(
            f"Need at least {n_epochs} checkpoint-* under the run dir; found {len(cps)}. "
            f"Wasserstein expects epoch_1..epoch_{n_epochs} after export."
        )
    if len(cps) > n_epochs:
        cps = cps[-n_epochs:]
    return [(i + 1, p) for i, (step, p) in enumerate(cps)]


def work_root_tasks(nw: Path, family: str, task: str) -> Path:
    return (
        nw
        / "eval"
        / "split"
        / "weight_vs_baseline"
        / "wass_full_by_task"
        / family
        / task
    ).resolve()


def iter_experiments() -> Iterable[Tuple[str, str]]:
    for fam in FAMILIES:
        for task in TASKS:
            yield fam, task


def is_work_complete(work: Path, shared_baseline: bool) -> bool:
    """
    True if this workdir already has a full TDA chain for the default pipeline.

    - With **shared** family baseline: ``numpy/epoch_1..epoch_6``, matching
      ``persistence/epoch_1..epoch_6/SavedDiagrams/*.pkl``, and ``wasserstein_results.csv``.
    - With ``--no-shared-baseline``: also require ``epoch_0`` in numpy + persistence.
    """
    wcsv = work / "wasserstein_results.csv"
    if not wcsv.is_file() or wcsv.stat().st_size < 64:
        return False
    np_r = work / "numpy"
    pe_r = work / "persistence"
    epochs = list(range(1, 7))
    if not shared_baseline:
        epochs = [0] + epochs
    for k in epochs:
        en = f"epoch_{k}"
        nd = np_r / en
        if not nd.is_dir() or not any(nd.glob("*.npy")):
            return False
        pd = pe_r / en / "SavedDiagrams"
        if not pd.is_dir() or not any(pd.glob("*.pkl")):
            return False
    return True


def _task_work_and_log(
    nw: Path, family: str, task: str, log_file: Optional[Path] = None
) -> Tuple[Path, Path]:
    work = work_root_tasks(nw, family, task)
    log_path = (Path(log_file).resolve() if log_file else work / "pipeline.log")
    gwp.configure_logging(log_path)
    return work, log_path


def cmd_list(args: argparse.Namespace) -> None:
    """Print (family, task, run_dir) for all 12 experiments."""
    ck = (Path(args.ckpt_root).resolve() if args.ckpt_root else (Path(args.nw_root) / "checkpoints").resolve())
    n = 0
    for fam, task in iter_experiments():
        rdir = task_checkpoint_dir(ck, fam, task)
        wdir = work_root_tasks(Path(args.nw_root).resolve(), fam, task)
        print(f"{fam}\t{task}\t{rdir}\twork={wdir}")
        n += 1
    print(f"# total {n} (expected {N_EXPERIMENTS})")


def cmd_print_batch(args: argparse.Namespace) -> None:
    """
    Print one shell-ready line per experiment to pipe into xargs -P.
    Each line is a full `python <this_script> run ...` invocation.
    """
    script = Path(__file__).resolve()
    root = Path(args.nw_root).resolve()
    extra: List[str] = []
    if getattr(args, "ckpt_root", None):
        extra.extend(["--ckpt-root", str(Path(args.ckpt_root).resolve())])
    if args.baseline_override:
        extra.extend(["--baseline-override", args.baseline_override])
    if getattr(args, "projections", None):
        extra.extend(["--projections", args.projections])
    if args.no_shared_baseline:
        extra.append("--no-shared-baseline")
    if getattr(args, "force", False):
        extra.append("--force")
    for fam, task in iter_experiments():
        parts = [
            sys.executable,
            str(script),
            "--nw-root",
            str(root),
            *extra,
            "run",
            "--family",
            fam,
            "--task",
            task,
        ]
        # Single line for bash -lc or xargs
        print(_shell_single_quote_cmd(parts))


def _shell_single_quote_cmd(parts: List[str]) -> str:
    def esc(s: str) -> str:
        return "'" + s.replace("'", "'\\''") + "'"

    return " ".join(esc(p) for p in parts)


def task_cmd_export(
    args: argparse.Namespace, *, _emit_step_timing: bool = True
) -> None:
    t0 = time.perf_counter()
    h = gwp._load_norm_helpers()
    projections = gwp.normalize_projection_arg(args.projections)
    proj_csv = gwp.projections_csv(projections)
    proj_set = frozenset(projections)
    nw = Path(args.nw_root).resolve()
    ck = Path(args.ckpt_root).resolve() if args.ckpt_root else (nw / "checkpoints").resolve()
    work, log_path = _task_work_and_log(
        nw, args.family, args.task, args.log_file
    )
    run_dir = task_checkpoint_dir(ck, args.family, args.task)
    if not run_dir.is_dir():
        raise SystemExit(f"Missing run dir: {run_dir}")
    bl = h.pick_pretrained_dir(args.family, args.baseline_override, nw)
    if not bl:
        raise SystemExit("No baseline/pretrained weights (use --baseline-override or place weights under nw/pretrained/)")
    raw_cps = h.discover_epoch_checkpoints(run_dir)
    cps = _renumber_checkpoints_sequential(raw_cps, n_epochs=6)
    gwp._LOG.info(
        "[export] renumbered to epoch_1..epoch_6 (raw step indices were %s)",
        [s for s, _ in sorted(raw_cps, key=lambda x: x[0])],
    )
    np_root = work / "numpy"
    gwp._LOG.info(
        "[export] %s %s -> %s projections=%s (checkpoints: %s)",
        args.family,
        args.task,
        np_root,
        proj_csv,
        run_dir,
    )
    use_shared_baseline = not args.no_shared_baseline
    if use_shared_baseline:
        if getattr(args, "skip_shared_baseline_check", False):
            gwp._LOG.info(
                "[export] skipping shared-baseline ensure (no Ripser on wass/.../baseline); "
                "omit this flag (or fix baseline) before wass.",
            )
        else:
            gwp.ensure_shared_baseline(nw, args.family, bl, gwp._LOG, projections)
    gwp.export_numpy_vo(
        nw,
        args.family,
        run_dir,
        np_root,
        bl,
        cps,
        gwp._LOG,
        include_baseline_epoch0=not use_shared_baseline,
        projections=proj_set,
    )
    gwp._LOG.info("export finished (%s)", log_path)
    if _emit_step_timing:
        gwp._LOG.info(
            "[timing] export_elapsed_s=%.3f", time.perf_counter() - t0
        )


def task_cmd_persist(
    args: argparse.Namespace, *, _emit_step_timing: bool = True
) -> None:
    t0 = time.perf_counter()
    projections = gwp.normalize_projection_arg(args.projections)
    proj_csv = gwp.projections_csv(projections)
    nw = Path(args.nw_root).resolve()
    work, log_path = _task_work_and_log(
        nw, args.family, args.task, args.log_file
    )
    gwp._LOG.info("[persistence] workdir %s projections=%s", work, proj_csv)
    gwp.run_persistence_all(work, gwp._LOG, proj_csv)
    gwp._LOG.info("persistence finished (%s)", log_path)
    if _emit_step_timing:
        gwp._LOG.info(
            "[timing] persist_elapsed_s=%.3f", time.perf_counter() - t0
        )


def task_cmd_wass(
    args: argparse.Namespace, *, _emit_step_timing: bool = True
) -> None:
    t0 = time.perf_counter()
    projections = gwp.normalize_projection_arg(args.projections)
    proj_csv = gwp.projections_csv(projections)
    nw = Path(args.nw_root).resolve()
    work, log_path = _task_work_and_log(
        nw, args.family, args.task, args.log_file
    )
    csv_out = work / "wasserstein_results.csv"
    gwp._LOG.info("[wasserstein] workdir %s projections=%s", work, proj_csv)
    baseline_sd: Optional[Path] = None
    if not args.no_shared_baseline:
        baseline_sd = (
            gwp.family_baseline_root(nw, args.family) / "persistence" / "epoch_0" / "SavedDiagrams"
        )
    gwp.run_wasserstein(work, csv_out, gwp._LOG, baseline_sd_override=baseline_sd, projections_arg=proj_csv)
    gwp._LOG.info("wasserstein finished (%s)", log_path)
    if _emit_step_timing:
        gwp._LOG.info("[timing] wass_elapsed_s=%.3f", time.perf_counter() - t0)


def task_cmd_run(args: argparse.Namespace) -> None:
    nw = Path(args.nw_root).resolve()
    work, _ = _task_work_and_log(
        nw, args.family, args.task, args.log_file
    )
    shared = not args.no_shared_baseline
    if is_work_complete(work, shared_baseline=shared) and not getattr(
        args, "force", False
    ):
        gwp._LOG.info(
            "[run] skip (already complete): %s — use --force to recompute",
            work,
        )
        return
    t0 = time.perf_counter()
    task_cmd_export(args, _emit_step_timing=False)
    t1 = time.perf_counter()
    task_cmd_persist(args, _emit_step_timing=False)
    t2 = time.perf_counter()
    task_cmd_wass(args, _emit_step_timing=False)
    t3 = time.perf_counter()
    gwp._LOG.info(
        "[timing] export_elapsed_s=%.3f persist_elapsed_s=%.3f wass_elapsed_s=%.3f total_elapsed_s=%.3f",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t3 - t0,
    )
    gwp._LOG.info("run finished (all steps)")


def cmd_batch_run(args: argparse.Namespace) -> None:
    """
    Run all 12 ``run`` sub-invocations with subprocess (optional parallelism).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    script = Path(__file__).resolve()
    root = Path(args.nw_root).resolve()
    workers = max(1, int(args.parallel))
    fam_f = str(Path(args.ckpt_root).resolve()) if args.ckpt_root else None
    err = 0

    def one(fam: str, task: str) -> Tuple[str, int]:
        cmd = [sys.executable, str(script), "--nw-root", str(root), "run", "--family", fam, "--task", task]
        if fam_f:
            cmd.extend(["--ckpt-root", fam_f])
        if args.baseline_override:
            cmd.extend(["--baseline-override", args.baseline_override])
        if getattr(args, "projections", None):
            cmd.extend(["--projections", args.projections])
        if args.no_shared_baseline:
            cmd.append("--no-shared-baseline")
        if getattr(args, "force", False):
            cmd.append("--force")
        r = subprocess.run(cmd)
        return f"{fam} {task}", r.returncode

    jobs: List[Tuple[str, str]] = list(iter_experiments())
    if workers == 1:
        for fam, task in jobs:
            label, code = one(fam, task)
            if code != 0:
                err += 1
                print(f"FAIL {label} exit {code}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, f, t): (f, t) for f, t in jobs}
            for fut in as_completed(futs):
                label, code = fut.result()
                if code != 0:
                    err += 1
                    print(f"FAIL {label} exit {code}", file=sys.stderr)
    if err:
        raise SystemExit(err)


def add_task_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--task",
        choices=TASKS,
        required=True,
        help="Dataset / task: sst2, imdb, mmlu, or gsm8k (GSM8K uses {family}/full/…).",
    )


def main() -> None:
    if not gwp.GEN_PERSISTENCE.is_file() or not gwp.COMPUTE_WASS.is_file():
        raise SystemExit("Missing TDA helpers (generate_persistence or compute_wasserstein)")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--nw-root",
        type=Path,
        default=Path.cwd(),
        help="Exploration-finetuning project root (checkpoints, eval/, pretrained). Default: cwd.",
    )
    ap.add_argument(
        "--ckpt-root",
        type=Path,
        default=None,
        help="Checkpoint root (default: nw-root/checkpoints).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help=f"List all {N_EXPERIMENTS} (family, task, run_dir, work_dir) rows")
    p_list.set_defaults(func=cmd_list)

    p_batch = sub.add_parser(
        "print-batch",
        help="Print one `python ... run --family F --task T` per line (for xargs -P N)",
    )
    p_batch.add_argument(
        "--ckpt-root", type=Path, default=None, help="Same as top-level --ckpt-root; forwarded to each run."
    )
    p_batch.add_argument("--baseline-override", type=str, default=None)
    p_batch.add_argument(
        "--projections",
        type=str,
        default="k,q",
        help="Forwarded to each run (default: k,q). Example: v,o",
    )
    p_batch.add_argument("--no-shared-baseline", action="store_true")
    _add_force_flags(p_batch)
    p_batch.set_defaults(func=cmd_print_batch)

    p_brun = sub.add_parser(
        "batch-run",
        help="Run all 12 pipelines (export+persist+wass) with --parallel workers",
    )
    p_brun.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Concurrent subprocesses (default 1 = sequential; try 2–8, not 256).",
    )
    p_brun.add_argument("--baseline-override", type=str, default=None)
    p_brun.add_argument(
        "--projections",
        type=str,
        default="k,q",
        help="Forwarded to each subprocess run (default: k,q).",
    )
    p_brun.add_argument("--no-shared-baseline", action="store_true")
    _add_force_flags(p_brun)
    p_brun.set_defaults(func=cmd_batch_run)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--family",
            choices=FAMILIES,
            required=True,
        )
        add_task_flags(p)
        p.add_argument(
            "--projections",
            type=str,
            default="k,q",
            help="Comma-separated projections for export / Ripser / Wasserstein (default: k,q).",
        )
        p.add_argument("--baseline-override", type=str, default=None)
        p.add_argument(
            "--no-shared-baseline",
            action="store_true",
        )
        p.add_argument(
            "--skip-shared-baseline-check",
            action="store_true",
            help="Export / run only: do not call ensure_shared_baseline (skip baseline epoch_0 "
            "Ripser). Finetuning numpy still exports; complete baseline before wass.",
        )
        p.add_argument("--log-file", type=Path, default=None)

    p_e = sub.add_parser("export", help="Step 1: numpy weights → layer*_*.npy")
    add_common(p_e)
    p_e.set_defaults(func=task_cmd_export)

    p_p = sub.add_parser("persist", help="Step 2: persistence (expects numpy/)")
    add_common(p_p)
    p_p.set_defaults(func=task_cmd_persist)

    p_w = sub.add_parser("wass", help="Step 3: wasserstein_results.csv (expects persistence/)")
    add_common(p_w)
    p_w.set_defaults(func=task_cmd_wass)

    p_r = sub.add_parser("run", help="export + persist + wass (skips if already complete, unless --force)")
    add_common(p_r)
    _add_force_flags(p_r)
    p_r.set_defaults(func=task_cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
