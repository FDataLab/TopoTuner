#!/usr/bin/env python3
"""
Standalone descriptive statistics for weight changes vs pretrained.

**Attention slice (layer_matrix / summaries):** q/k/v/o projection matrices only — same stats per tensor.

**Full-model row** (``record_type=full_model``): ``mean_abs_relative_delta`` and ``changed_fraction`` are computed over **all**
floating weights shared between pretrained and finetuned checkpoints (intersection of keys, matching shapes).
For LoRA runs, Δ is ``scale·(B A)`` on adapted attention projections and **zero** elsewhere — denominator still uses frozen base weights.

Full finetune & TDA-High3 (full weights):  delta = W_finetuned - W_pretrained
LoRA: effective update merged into base (additive LoRA): delta = (alpha/r) * (B @ A)
      at the finetuned checkpoint — equals W_merged - W_base when training starts from zero adapters.

**DropBP + LoRA** (``--method dropbp_lora``): same Δ as LoRA — checkpoints are still PEFT
``adapter_model.safetensors`` under a run directory (e.g. ``competitor/dropbp_tri_task_lora/<stamp>/...``).
Pass ``--dropbp-run-dir`` to that run folder **or** use ``--method lora`` with ``--finetuned-checkpoint`` pointing
at the final ``checkpoint-*`` (equivalent math).

Processes **one checkpoint configuration per invocation** and loads **one weight tensor at a time**
(memory-safe).

Paths mirror ``norm_llama_full_layer_epoch_curves.py`` / ``exploration-finetuning/checkpoints``::

  checkpoints/<family>/<task>/full/<run_id>/checkpoint-*
  checkpoints/<family>/<task>/lora/<run_id>/checkpoint-*   (+ adapter_model.safetensors per checkpoint)
  checkpoints/<family>/<task>/wass-high-3/<run_id>/  (fallback: ``wass-freeze/high-3-frozen``, ``high-3``)

Pretrained base::

  <nw-root>/pretrained/<llama31-8b|qwen3-8b-base|mistral-7b-v03>/   (override with --pretrained-dir)

Typical roots on this machine::

  NW_ROOT=/path/to/topo/numpy_weights/exploration-finetuning
  OUTPUT=/path/to/topo/codes/analysis/weight_change_stats_all

Examples::

  python weight_change_descriptive_stats.py --nw-root ~/numpy_weights/exploration-finetuning \\
      --family llama --task sst2 --method full --out-dir ./stats_out --progress

  python weight_change_descriptive_stats.py --nw-root ... --family llama --task sst2 --method lora

  # DropBP + LoRA (competitor run dir: contains checkpoint-* with adapters)
  python weight_change_descriptive_stats.py --nw-root ~/numpy_weights/exploration-finetuning \\
      --family qwen-base --task imdb --method dropbp_lora \\
      --dropbp-run-dir ~/numpy_weights/exploration-finetuning/competitor/dropbp_tri_task_lora/20260506_163308/Qwen-Qwen3-8B-Base/imdb/imdb-dropbp-lora-dr0p2-e6-20260506_163308 \\
      --out-dir ./stats_out

  python weight_change_descriptive_stats.py --finetuned-checkpoint /path/to/checkpoint-1878 \\
      --pretrained-dir /path/to/pretrained --method full --family llama --task sst2 --out-dir ./out

Batch driver (one consolidated ``weight_change_stats.csv``, one ``weight_change_stats.log``)::

  bash /path/to/topo/codes/analysis/run_weight_change_stats_batch.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open


def torch_float_tensor_to_numpy_f32(t: "torch.Tensor") -> np.ndarray:
    """Convert float tensor to float32 ndarray (handles bf16/f16); avoids broken torch.numpy + NumPy 2 ABI."""
    t = t.detach().cpu().float().contiguous()
    shape = tuple(t.shape)
    try:
        return t.numpy().astype(np.float32, copy=False).reshape(shape)
    except RuntimeError:
        flat = t.view(-1)
        n = int(flat.numel())
        chunk = 4_194_304  # elements per slice (limits Python list sizes)
        parts: List[np.ndarray] = []
        for i in range(0, n, chunk):
            sub = flat[i : i + chunk]
            parts.append(np.asarray(sub.tolist(), dtype=np.float32))
        return np.concatenate(parts).reshape(shape)

DEFAULT_EPS = 1e-6
DEFAULT_THRESHOLD = 1e-6

# ``fork`` after the parent has initialized threaded BLAS/OpenMP can deadlock workers.
_FULL_MODEL_MP_CTX = mp.get_context("spawn")

# CLI methods (folders under checkpoints/<family>/[task]/… except gsm8k omits task segment).
# Finetuned runs are always compared to **pretrained** (NW_ROOT/pretrained/…); there is no separate
# ``baseline`` job — use frozen base weights offline as the reference when comparing tables.
METHOD_CHOICES = (
    "full",
    "lora",
    "dropbp_lora",
    "tda_high3",
    "tda_high6",
    "tda_high9",
    "eltwise_high3",
    "eltwise_high6",
    "eltwise_high9",
)

PROJ_ORDER = ("q", "k", "v", "o")
PROJ_LABEL = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}

LAYER_RE = re.compile(r"(?:.*\.)?layers\.(\d+)\.")
PROJ_PATTERNS = {
    "k": ".k_proj.",
    "q": ".q_proj.",
    "v": ".v_proj.",
    "o": ".o_proj.",
}

# Prefer canonical RUN_ID first. qwen-base tries ``run3`` before ``run1`` (family-root full/lora
# often store under run3/run2; per-dataset folders may only have ``run1`` — order handles both).
RUN_IDS_TRY = {"llama": ("run1",), "qwen-base": ("run3", "run1", "run2"), "mistral-7b-v03": ("run1",)}
_PRETRAINED_SUBDIR = {"llama": "llama31-8b", "qwen-base": "qwen3-8b-base", "mistral-7b-v03": "mistral-7b-v03"}

# LoRA parameter key (PEFT); allow optional .default before .weight
LORA_AB_RE = re.compile(
    r"(?:base_model\.(?:model\.)?model\.|model\.)layers\.(\d+)\.self_attn\.([qkvo])_proj\.lora_([AB])"
    r"(?:\.default)?\.weight"
)


def load_weight_map(folder: Path) -> Dict[str, str]:
    index_path = folder / "model.safetensors.index.json"
    single_path = folder / "model.safetensors"
    if index_path.is_file():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        wm = idx.get("weight_map")
        if not wm:
            raise RuntimeError(f"No weight_map in {index_path}")
        return dict(wm)
    if single_path.is_file():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise RuntimeError(f"No safetensors weights found in {folder}")


def projection_from_param(param_name: str) -> Optional[str]:
    for proj, pattern in PROJ_PATTERNS.items():
        if pattern in param_name:
            return proj
    return None


def layer_index(param_name: str) -> Optional[int]:
    m = LAYER_RE.search(param_name)
    if not m:
        return None
    return int(m.group(1))


def _has_full_weights(d: Path) -> bool:
    return (d / "model.safetensors.index.json").is_file() or (d / "model.safetensors").is_file()


def _has_adapter_ckpt(d: Path) -> bool:
    return (d / "adapter_model.safetensors").is_file()


def discover_step_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (global_step, checkpoint_dir) for ``checkpoint-<step>``."""
    import glob

    cps = sorted(glob.glob(str(run_dir / "checkpoint-*")))
    out: List[Tuple[int, Path]] = []
    for p in cps:
        name = Path(p).name
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            out.append((step, Path(p)))
    out.sort(key=lambda x: x[0])
    return out


def resolve_finetuned_checkpoint(
    run_dir: Path,
    *,
    epoch_tag: str = "6",
    prefer_last_if_no_epoch_folder: bool = True,
) -> Path:
    """
    Prefer ``epoch_weights/checkpoint-epoch-{epoch_tag}``, then ``checkpoint-epoch-{epoch_tag}``
    direct child, then **last** checkpoint by global step.
    """
    ew = run_dir / "epoch_weights"
    candidates = [
        ew / f"checkpoint-epoch-{epoch_tag}",
        run_dir / f"checkpoint-epoch-{epoch_tag}",
    ]
    for c in candidates:
        if not c.is_dir():
            continue
        if _has_full_weights(c) or _has_adapter_ckpt(c):
            return c.resolve()

    steps = discover_step_checkpoints(run_dir)
    if not steps:
        raise FileNotFoundError(f"No checkpoint-* directories under {run_dir}")

    choice = steps[-1][1]
    # Prefer last checkpoint that has weights we need
    for _step, p in reversed(steps):
        if _has_full_weights(p) or _has_adapter_ckpt(p):
            choice = p
            break
    if prefer_last_if_no_epoch_folder:
        return choice.resolve()
    raise FileNotFoundError(f"No epoch checkpoint for epoch_tag={epoch_tag} under {run_dir}")


def pick_pretrained_dir(nw_root: Path, family: str, override: Optional[Path]) -> Path:
    if override is not None and override.is_dir() and _has_full_weights(override):
        return override.resolve()
    sub = _PRETRAINED_SUBDIR.get(family)
    if not sub:
        raise ValueError(f"Unknown family for pretrained lookup: {family}")
    p = (nw_root / "pretrained" / sub).resolve()
    if not _has_full_weights(p):
        raise FileNotFoundError(f"Missing pretrained weights at {p}")
    return p


def _first_existing_subdir(parent: Path, try_names: Sequence[str]) -> Path:
    for name in try_names:
        p = parent / name
        if p.is_dir():
            return p.resolve()
    raise FileNotFoundError(f"No matching run under {parent}; tried {try_names!r}")


def _run_ids_try(family: str, task: str, method: str) -> Tuple[str, ...]:
    """Run subfolder order (``run1``, ``run2``, …). qwen family-root full/lora have run3/run2 only."""
    base_ids = RUN_IDS_TRY.get(family, ("run1",))
    if family == "qwen-base" and task == "gsm8k" and method in ("full", "lora", "dropbp_lora"):
        return ("run3", "run2", "run1")
    return base_ids


def _resolve_wass_high_dir(base: Path, try_names: Tuple[str, ...], hi: str) -> Path:
    """``wass-high-{hi}``, ``wass-freeze/high-{hi}-frozen``, or ``wass-freeze/high-{hi}`` (Llama vs Qwen/Mistral)."""
    for name in try_names:
        cands = [
            base / f"wass-high-{hi}" / name,
            base / "wass-freeze" / f"high-{hi}-frozen" / name,
            base / "wass-freeze" / f"high-{hi}" / name,
        ]
        hit = next((c for c in cands if c.is_dir()), None)
        if hit is not None:
            return hit.resolve()
    raise FileNotFoundError(f"No Wasserstein high-{hi} run under {base} for tries {try_names!r}")


def _resolve_eltwise_high_dir(base: Path, try_names: Tuple[str, ...], hi: str) -> Path:
    return _first_existing_subdir(base / "eltwise-freeze" / f"high-{hi}", try_names)


def resolve_run_dir(nw_root: Path, family: str, task: str, method: str) -> Path:
    try_names = _run_ids_try(family, task, method)
    # gsm8k (and similar) checkpoints live at checkpoints/<family>/{full,lora,wass-…}/run*
    # without a dataset-named segment (unlike sst2/imdb/mmlu).
    if task == "gsm8k":
        base = nw_root / "checkpoints" / family
    else:
        base = nw_root / "checkpoints" / family / task
    if method == "full":
        p = _first_existing_subdir(base / "full", try_names)
    elif method == "lora":
        p = _first_existing_subdir(base / "lora", try_names)
    elif method in ("tda_high3", "tda_high6", "tda_high9"):
        hi = method.replace("tda_high", "")
        p = _resolve_wass_high_dir(base, try_names, hi)
    elif method in ("eltwise_high3", "eltwise_high6", "eltwise_high9"):
        hi = method.replace("eltwise_high", "")
        p = _resolve_eltwise_high_dir(base, try_names, hi)
    else:
        raise ValueError(f"Unknown method {method!r}")
    return p


def load_tensor_numpy(folder: Path, weight_map: Dict[str, str], param_name: str) -> np.ndarray:
    """Load a single parameter tensor (memory-safe: one array at a time). Uses torch for bfloat16."""
    shard = weight_map[param_name]
    path = folder / shard
    with safe_open(str(path), framework="pt", device="cpu") as f:
        t = f.get_tensor(param_name)
    return torch_float_tensor_to_numpy_f32(t)


# Match Llama-style linear weights used for LoRA global Δ overlay (same layer/proj IDs as ``process_lora``).
_ATTN_PROJ_WEIGHT_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.weight$")


def _full_ft_chunk_worker(
    args: Tuple[str, str, Tuple[str, ...], float, float],
) -> Tuple[float, float, int]:
    """ProcessPool worker: partial sums over ``names`` subset (pickle-safe tuple args)."""
    pretrained_dir_s, finetuned_dir_s, names_tuple, eps, thresh = args
    pretrained_dir = Path(pretrained_dir_s)
    finetuned_dir = Path(finetuned_dir_s)
    wm_b = load_weight_map(pretrained_dir)
    wm_f = load_weight_map(finetuned_dir)
    sum_rel = 0.0
    sum_changed = 0.0
    n_tot = 0
    for name in names_tuple:
        if name not in wm_b or name not in wm_f:
            continue
        wb = load_tensor_numpy(pretrained_dir, wm_b, name)
        wf = load_tensor_numpy(finetuned_dir, wm_f, name)
        if wb.shape != wf.shape:
            del wb, wf
            continue
        delta = wf.astype(np.float64, copy=False) - wb.astype(np.float64, copy=False)
        d = delta.ravel()
        wb_r = wb.astype(np.float64, copy=False).ravel()
        ad = np.abs(d)
        ab = np.abs(wb_r)
        denom = np.maximum(ab, eps)
        rel = ad / denom
        sum_rel += float(rel.sum())
        sum_changed += float(np.sum(ad > thresh))
        n_tot += ad.size
        del wb, wf, delta
    return sum_rel, sum_changed, n_tot


def accumulate_global_stats_full_ft(
    pretrained_dir: Path,
    finetuned_dir: Path,
    eps: float,
    thresh: float,
    *,
    progress: bool = False,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Mean relative Δ and changed fraction over **every** intersecting weight tensor (full ~8B scale)."""
    wm_b = load_weight_map(pretrained_dir)
    wm_f = load_weight_map(finetuned_dir)
    common = sorted(set(wm_b.keys()) & set(wm_f.keys()))
    if not common:
        return {}

    nw = max(1, int(num_workers))
    if nw <= 1:
        chunks: List[List[str]] = [common]
    else:
        parts = np.array_split(np.asarray(common, dtype=object), min(nw, len(common)))
        chunks = [p.tolist() for p in parts if p.size > 0]

    if progress:
        print(f"[info] full-model sweep: {len(common)} tensors in {len(chunks)} chunk(s)", flush=True)

    if len(chunks) == 1:
        sum_rel, sum_changed, n_tot = _full_ft_chunk_worker(
            (str(pretrained_dir), str(finetuned_dir), tuple(chunks[0]), eps, thresh)
        )
    else:
        tasks = [(str(pretrained_dir), str(finetuned_dir), tuple(c), eps, thresh) for c in chunks]
        sum_rel = sum_changed = 0.0
        n_tot = 0
        max_proc = len(tasks)
        with ProcessPoolExecutor(max_workers=max_proc, mp_context=_FULL_MODEL_MP_CTX) as ex:
            for i, tup in enumerate(ex.map(_full_ft_chunk_worker, tasks, chunksize=1)):
                sum_rel += tup[0]
                sum_changed += tup[1]
                n_tot += tup[2]
                if progress:
                    print(f"[progress] full-model chunk {i + 1}/{len(tasks)} merged", flush=True)

    if n_tot == 0:
        return {}
    return {
        "mean_abs_relative_delta": sum_rel / n_tot,
        "changed_fraction": sum_changed / n_tot,
        "n_params": int(n_tot),
    }


def _serialize_key_index(key_index: Dict[Tuple[int, str], Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {f"{a}|{b}": dict(v) for (a, b), v in key_index.items()}


def _lora_chunk_worker(
    args: Tuple[str, str, Tuple[str, ...], float, float, float, Dict[str, Dict[str, str]]],
) -> Tuple[float, float, int]:
    pretrained_dir_s, adapter_dir_s, names_tuple, eps, thresh, scale, flat_ki = args
    pretrained_dir = Path(pretrained_dir_s)
    adapter_dir = Path(adapter_dir_s)
    wm_b = load_weight_map(pretrained_dir)
    adapter_file = adapter_dir / "adapter_model.safetensors"

    key_index: Dict[Tuple[int, str], Dict[str, str]] = defaultdict(dict)
    for k, v in flat_ki.items():
        li_s, proj = k.split("|", 1)
        key_index[(int(li_s), proj)] = dict(v)

    sum_rel = 0.0
    sum_changed = 0.0
    n_tot = 0

    with safe_open(str(adapter_file), framework="pt", device="cpu") as af:
        for name in names_tuple:
            wb = load_tensor_numpy(pretrained_dir, wm_b, name)
            n_el = wb.size
            m = _ATTN_PROJ_WEIGHT_RE.search(name)
            delta: Optional[np.ndarray] = None
            if m:
                li = int(m.group(1))
                proj = m.group(2).lower()
                ks = key_index.get((li, proj))
                if ks and "A" in ks and "B" in ks:
                    A = torch_float_tensor_to_numpy_f32(af.get_tensor(ks["A"]))
                    B = torch_float_tensor_to_numpy_f32(af.get_tensor(ks["B"]))
                    ba = np.matmul(B.astype(np.float64), A.astype(np.float64))
                    cand = scale * ba
                    if cand.shape == wb.shape:
                        delta = cand.astype(np.float64, copy=False)
                    del A, B, ba, cand

            if delta is None:
                sum_rel += 0.0
                sum_changed += 0.0
                n_tot += n_el
                del wb
                continue

            d = delta.ravel()
            wb_r = wb.astype(np.float64, copy=False).ravel()
            ad = np.abs(d)
            ab = np.abs(wb_r)
            denom = np.maximum(ab, eps)
            rel = ad / denom
            sum_rel += float(rel.sum())
            sum_changed += float(np.sum(ad > thresh))
            n_tot += n_el
            del wb, delta

    return sum_rel, sum_changed, n_tot


def accumulate_global_stats_lora(
    pretrained_dir: Path,
    adapter_dir: Path,
    eps: float,
    thresh: float,
    *,
    progress: bool = False,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Same pooling as global full-FT, but Δ = scale·(B A) only on adapted attn projections; else Δ = 0."""
    scale, _r = _read_lora_scale(adapter_dir)
    wm_b = load_weight_map(pretrained_dir)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)

    key_index: Dict[Tuple[int, str], Dict[str, str]] = defaultdict(dict)
    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        for key in f.keys():
            m = LORA_AB_RE.search(key)
            if not m:
                continue
            li = int(m.group(1))
            proj = m.group(2).lower()
            ab = m.group(3).upper()
            key_index[(li, proj)][ab] = key

    flat_ki = _serialize_key_index(key_index)
    names = sorted(wm_b.keys())
    nw = max(1, int(num_workers))

    if nw <= 1:
        chunks = [names]
    else:
        parts = np.array_split(np.asarray(names, dtype=object), min(nw, len(names)))
        chunks = [p.tolist() for p in parts if p.size > 0]

    if progress:
        print(f"[info] full-model (LoRA) sweep: {len(names)} tensors in {len(chunks)} chunk(s)", flush=True)

    if len(chunks) == 1:
        sum_rel, sum_changed, n_tot = _lora_chunk_worker(
            (
                str(pretrained_dir),
                str(adapter_dir),
                tuple(chunks[0]),
                eps,
                thresh,
                scale,
                flat_ki,
            )
        )
    else:
        tasks = [
            (str(pretrained_dir), str(adapter_dir), tuple(c), eps, thresh, scale, flat_ki) for c in chunks
        ]
        sum_rel = sum_changed = 0.0
        n_tot = 0
        max_proc = len(tasks)
        with ProcessPoolExecutor(max_workers=max_proc, mp_context=_FULL_MODEL_MP_CTX) as ex:
            for i, tup in enumerate(ex.map(_lora_chunk_worker, tasks, chunksize=1)):
                sum_rel += tup[0]
                sum_changed += tup[1]
                n_tot += tup[2]
                if progress:
                    print(f"[progress] full-model (LoRA) chunk {i + 1}/{len(tasks)} merged", flush=True)

    if n_tot == 0:
        return {}
    return {
        "mean_abs_relative_delta": sum_rel / n_tot,
        "changed_fraction": sum_changed / n_tot,
        "n_params": int(n_tot),
    }


def iter_attn_proj_weight_keys(weight_map: Dict[str, str]) -> Iterator[Tuple[int, str, str]]:
    """Yield (layer_idx, proj, param_name) for model.layers.*.self_attn.{q,k,v,o}_proj.weight."""
    for name in sorted(weight_map.keys()):
        if not name.endswith("_proj.weight"):
            continue
        if ".self_attn." not in name:
            continue
        proj = projection_from_param(name)
        if proj not in PROJ_ORDER:
            continue
        li = layer_index(name)
        if li is None:
            continue
        yield li, proj, name


def compute_stats(delta: np.ndarray, w_base: np.ndarray, eps: float, thresh: float) -> Dict[str, Any]:
    """Per-matrix stats (attention projections only unless caller aggregates globally elsewhere).

    (1) mean_abs_relative_delta — mean_i |Δ_i| / max(|w_{base,i}|, ε).

    (2) changed_fraction — fraction of entries with |Δ_i| > threshold.

    On ``record_type=layer_matrix`` rows these refer **only to that tensor**. On ``record_type=full_model``, same column names mean **whole-model** pooling.

    Trainable-parameter fraction is **not** computed here (depends on mask / LoRA rank).
    """
    d = delta.astype(np.float64, copy=False).ravel()
    wb = w_base.astype(np.float64, copy=False).ravel()
    ad = np.abs(d)
    ab = np.abs(wb)
    denom = np.maximum(ab, eps)
    rel = ad / denom
    return {
        "mean_abs_relative_delta": float(np.mean(rel)),
        "changed_fraction": float(np.mean(ad > thresh)),
        "n_params": int(ad.size),
    }


def process_full_like(
    pretrained_dir: Path,
    finetuned_dir: Path,
    eps: float,
    thresh: float,
    *,
    progress: bool = False,
) -> List[Dict[str, Any]]:
    wm_b = load_weight_map(pretrained_dir)
    wm_f = load_weight_map(finetuned_dir)
    common = set(wm_b.keys()) & set(wm_f.keys())

    rows: List[Dict[str, Any]] = []
    done = 0
    for li, proj, name in iter_attn_proj_weight_keys(wm_b):
        if name not in common:
            continue
        wb = load_tensor_numpy(pretrained_dir, wm_b, name)
        wf = load_tensor_numpy(finetuned_dir, wm_f, name)
        if wb.shape != wf.shape:
            continue
        delta = wf.astype(np.float64, copy=False) - wb.astype(np.float64, copy=False)
        st = compute_stats(delta, wb, eps, thresh)
        rows.append({"layer": li, "projection": proj, **st})
        done += 1
        if progress:
            short = name.split("layers.")[-1] if "layers." in name else name
            print(f"[progress] matrix {done}: layer={li} proj={proj} …{short}", flush=True)
        del delta, wb, wf
    rows.sort(key=lambda r: (r["projection"], r["layer"]))
    return rows


def _read_lora_scale(adapter_dir: Path) -> Tuple[float, int]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    return alpha / float(r), r


def process_lora(
    pretrained_dir: Path,
    finetuned_adapter_dir: Path,
    eps: float,
    thresh: float,
    *,
    progress: bool = False,
) -> List[Dict[str, Any]]:
    """delta = scale * (B @ A) — merged increment vs frozen base (matches additive LoRA forward)."""
    scale, _r = _read_lora_scale(finetuned_adapter_dir)
    wm_b = load_weight_map(pretrained_dir)

    adapter_file = finetuned_adapter_dir / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)

    # Key names only first — then load one (layer,proj) pair at a time.
    key_index: Dict[Tuple[int, str], Dict[str, str]] = defaultdict(dict)
    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        for key in f.keys():
            m = LORA_AB_RE.search(key)
            if not m:
                continue
            li = int(m.group(1))
            proj = m.group(2).lower()
            ab = m.group(3).upper()
            key_index[(li, proj)][ab] = key

    rows: List[Dict[str, Any]] = []
    done = 0
    with safe_open(str(adapter_file), framework="pt", device="cpu") as f:
        for (li, proj), ks in sorted(key_index.items()):
            if proj not in PROJ_ORDER:
                continue
            if "A" not in ks or "B" not in ks:
                continue
            A = torch_float_tensor_to_numpy_f32(f.get_tensor(ks["A"]))
            B = torch_float_tensor_to_numpy_f32(f.get_tensor(ks["B"]))
            ba = np.matmul(B.astype(np.float64), A.astype(np.float64))
            delta = scale * ba

            name = None
            for n in wm_b:
                if re.search(rf"\.layers\.{li}\.self_attn\.{proj}_proj\.weight$", n):
                    name = n
                    break
            if name is None:
                del A, B, ba, delta
                continue
            wb = load_tensor_numpy(pretrained_dir, wm_b, name)
            if wb.shape != delta.shape:
                del A, B, ba, delta, wb
                continue

            st = compute_stats(delta, wb, eps, thresh)
            rows.append({"layer": li, "projection": proj, **st})
            done += 1
            if progress:
                print(f"[progress] LoRA matrix {done}: layer={li} proj={proj}", flush=True)
            del A, B, ba, delta, wb
    rows.sort(key=lambda r: (r["projection"], r["layer"]))
    return rows


def aggregate_summary(rows: Sequence[Dict[str, Any]], group_cols: Sequence[str]) -> List[Dict[str, Any]]:
    """Mean metrics weighted by n_params when aggregating."""
    keyed: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = tuple(r[c] for c in group_cols)
        keyed[key].append(r)

    metrics = (
        "mean_abs_relative_delta",
        "changed_fraction",
    )
    out: List[Dict[str, Any]] = []
    for key, grp in sorted(keyed.items()):
        total_n = sum(int(x["n_params"]) for x in grp)
        row = {c: v for c, v in zip(group_cols, key)}
        row["n_layers"] = len(grp)
        row["n_params_total"] = total_n
        for m in metrics:
            row[m] = sum(float(x[m]) * int(x["n_params"]) for x in grp) / max(total_n, 1)
        out.append(row)
    return out


def aggregate_compact_vo(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Single pooled row over all layers where projection is v or o (weighted means where sensible)."""
    sub = [r for r in rows if r["projection"] in ("v", "o")]
    if not sub:
        return {}
    total_n = sum(int(r["n_params"]) for r in sub)
    wm_metrics = (
        "mean_abs_relative_delta",
        "changed_fraction",
    )
    out: Dict[str, Any] = {"n_rows_vo": len(sub), "n_params_total_vo": total_n}
    for m in wm_metrics:
        out[m] = sum(float(r[m]) * int(r["n_params"]) for r in sub) / max(total_n, 1)
    return out


def plot_layer_curves(
    rows: Sequence[Dict[str, Any]],
    proj: str,
    family: str,
    task: str,
    method: str,
    out_path: Path,
) -> None:
    sub = sorted([r for r in rows if r["projection"] == proj], key=lambda x: x["layer"])
    if len(sub) < 2:
        return
    layers = [r["layer"] for r in sub]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(layers, [r["mean_abs_relative_delta"] for r in sub], marker="o", color="#1f77b4")
    axes[0].set_ylabel(r"mean $|\Delta|/\max(|W_{\mathrm{base}}|,\varepsilon)$")
    axes[0].set_title(f"{family} · {task} · {method} · {PROJ_LABEL[proj]} — mean abs relative Δ")

    axes[1].plot(layers, [r["changed_fraction"] for r in sub], marker="s", color="#ff7f0e")
    axes[1].set_ylabel("changed fraction")
    axes[1].set_xlabel("Layer")
    axes[1].set_title(f"{family} · {task} · {method} · {PROJ_LABEL[proj]} — fraction |Δ|>τ")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


MASTER_COLUMNS: Tuple[str, ...] = (
    "record_type",
    "family",
    "task",
    "method",
    "pretrained_dir",
    "finetuned_checkpoint",
    "eps",
    "threshold",
    "layer",
    "projection",
    "n_layers",
    "n_params",
    "n_params_total",
    "n_rows_vo",
    "mean_abs_relative_delta",
    "changed_fraction",
)


def _meta_row(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family": meta["family"],
        "task": meta["task"],
        "method": meta["method"],
        "pretrained_dir": meta["pretrained_dir"],
        "finetuned_checkpoint": meta["finetuned_checkpoint"],
        "eps": meta["eps"],
        "threshold": meta["threshold"],
    }


def build_master_rows(
    meta: Dict[str, Any],
    detail_raw: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    compact_stats: Dict[str, Any],
    full_model_stats: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Flatten detail, per-projection summary, v+o compact, and optional whole-model row."""
    m = _meta_row(meta)
    out: List[Dict[str, Any]] = []

    metrics = (
        "mean_abs_relative_delta",
        "changed_fraction",
    )

    for r in detail_raw:
        row: Dict[str, Any] = {
            "record_type": "layer_matrix",
            **m,
            "layer": r["layer"],
            "projection": r["projection"],
            "n_layers": "",
            "n_params": r["n_params"],
            "n_params_total": "",
            "n_rows_vo": "",
        }
        for k in metrics:
            row[k] = r[k]
        out.append(row)

    for r in summary_rows:
        row = {
            "record_type": "summary_by_projection",
            **m,
            "layer": "",
            "projection": r["projection"],
            "n_layers": r["n_layers"],
            "n_params": "",
            "n_params_total": r["n_params_total"],
            "n_rows_vo": "",
        }
        for k in metrics:
            row[k] = r[k]
        out.append(row)

    if compact_stats:
        row = {
            "record_type": "compact_vo",
            **m,
            "layer": "",
            "projection": "v_o_pooled",
            "n_layers": "",
            "n_params": "",
            "n_params_total": compact_stats.get("n_params_total_vo", ""),
            "n_rows_vo": compact_stats.get("n_rows_vo", ""),
        }
        for k in metrics:
            row[k] = compact_stats.get(k, "")
        out.append(row)

    if full_model_stats:
        row = {
            "record_type": "full_model",
            **m,
            "layer": "",
            "projection": "FULL_CHECKPOINT",
            "n_layers": "",
            "n_params": full_model_stats.get("n_params", ""),
            "n_params_total": "",
            "n_rows_vo": "",
        }
        for k in metrics:
            row[k] = full_model_stats.get(k, "")
        out.append(row)

    return out


def write_master_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    mode: str = "overwrite",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MASTER_COLUMNS)
    if mode == "append":
        write_header = not path.is_file() or path.stat().st_size == 0
        op = "a"
    else:
        write_header = True
        op = "w"
    with path.open(op, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nw-root", type=Path, default=None, help="Exploration-finetuning root (checkpoints/, pretrained/).")
    p.add_argument("--family", type=str, required=True, choices=tuple(RUN_IDS_TRY.keys()))
    p.add_argument("--task", type=str, required=True, help="Dataset name (e.g. sst2, imdb, mmlu, gsm8k).")
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=METHOD_CHOICES,
        help=(
            "Checkpoint flavor: full, lora, dropbp_lora (same Δ math as lora on PEFT adapters), "
            "Wasserstein (tda_high*), eltwise-freeze (eltwise_high*). All deltas vs pretrained base."
        ),
    )
    p.add_argument("--pretrained-dir", type=Path, default=None, help="Override pretrained weights directory.")
    p.add_argument("--finetuned-checkpoint", type=Path, default=None, help="Explicit finetuned checkpoint directory (skip auto-resolve).")
    p.add_argument(
        "--dropbp-run-dir",
        type=Path,
        default=None,
        help=(
            "For --method dropbp_lora only: directory of one DropBP+LoRA run "
            "(contains checkpoint-* / adapter_model.safetensors). "
            "Resolves final step like resolve_finetuned_checkpoint. Requires --nw-root or --pretrained-dir."
        ),
    )
    p.add_argument("--epoch-tag", type=str, default="6", help="Prefer epoch_weights/checkpoint-epoch-{tag}.")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory for CSV + optional plots.")
    p.add_argument("--tag", type=str, default=None, help="Optional extra tag for plot filenames only.")
    p.add_argument(
        "--master-csv",
        type=Path,
        default=None,
        help="Single consolidated CSV path (default: <out-dir>/weight_change_stats.csv).",
    )
    p.add_argument(
        "--csv-mode",
        type=str,
        choices=("overwrite", "append"),
        default="overwrite",
        help="overwrite: replace master CSV contents; append: add rows (batch jobs truncate once then append).",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Write layerwise v/o PNGs under --out-dir; default is CSV-only.",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print one line per projection matrix as it is processed (stdout; use tee for logs).",
    )
    p.add_argument(
        "--skip-full-model-global",
        action="store_true",
        help="Skip second pass over **all** checkpoint tensors (~full model); keeps attention-only rows.",
    )
    p.add_argument(
        "--full-model-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes for the full-model tensor sweep only (default 1 = sequential). "
        "Ignored when --skip-full-model-global.",
    )
    args = p.parse_args()

    if args.finetuned_checkpoint is not None:
        ft = args.finetuned_checkpoint.resolve()
        if args.pretrained_dir is not None:
            pretrained = args.pretrained_dir.resolve()
        elif args.nw_root is not None:
            pretrained = pick_pretrained_dir(args.nw_root.resolve(), args.family, None)
        else:
            p.error("With --finetuned-checkpoint, pass --pretrained-dir or --nw-root for base weights.")
    elif args.method == "dropbp_lora" and args.dropbp_run_dir is not None:
        dr = args.dropbp_run_dir.resolve()
        if not dr.is_dir():
            p.error(f"--dropbp-run-dir is not a directory: {dr}")
        ft = resolve_finetuned_checkpoint(dr, epoch_tag=args.epoch_tag)
        if args.pretrained_dir is not None:
            pretrained = pick_pretrained_dir(Path("."), args.family, args.pretrained_dir.resolve())
        elif args.nw_root is not None:
            pretrained = pick_pretrained_dir(args.nw_root.resolve(), args.family, None)
        else:
            p.error("--dropbp-run-dir requires --pretrained-dir or --nw-root for pretrained base weights.")
    elif args.nw_root is None:
        p.error(
            "--nw-root is required unless --finetuned-checkpoint is set, "
            "or use --method dropbp_lora with --dropbp-run-dir and --pretrained-dir/--nw-root"
        )
    else:
        nw = args.nw_root.resolve()
        if args.method == "dropbp_lora":
            p.error(
                "For --method dropbp_lora without --finetuned-checkpoint, pass --dropbp-run-dir "
                "(competitor run folder). Standard checkpoints/<family>/<task>/lora/... is not used."
            )
        pretrained = pick_pretrained_dir(nw, args.family, args.pretrained_dir)
        run_dir = resolve_run_dir(nw, args.family, args.task, args.method)
        ft = resolve_finetuned_checkpoint(run_dir, epoch_tag=args.epoch_tag)

    tag_suffix = f"_{args.tag}" if args.tag else ""
    stem = f"{args.family}_{args.task}_{args.method}{tag_suffix}"
    master_csv = (
        args.master_csv.resolve()
        if args.master_csv is not None
        else (args.out_dir.resolve() / "weight_change_stats.csv")
    )

    print(f"[info] pretrained: {pretrained}", flush=True)
    print(f"[info] finetuned checkpoint: {ft}", flush=True)

    if args.method in ("lora", "dropbp_lora"):
        rows = process_lora(
            pretrained, ft, args.eps, args.threshold, progress=args.progress
        )
    else:
        rows = process_full_like(
            pretrained, ft, args.eps, args.threshold, progress=args.progress
        )

    if not rows:
        raise SystemExit("No projection rows computed — check paths and checkpoint compatibility.")

    meta = {
        "family": args.family,
        "task": args.task,
        "method": args.method,
        "pretrained_dir": str(pretrained),
        "finetuned_checkpoint": str(ft),
        "eps": args.eps,
        "threshold": args.threshold,
    }

    summary_rows = [{**meta, **r} for r in aggregate_summary(rows, ("projection",))]
    compact_stats = aggregate_compact_vo(rows)

    full_model_stats: Optional[Dict[str, Any]] = None
    if not args.skip_full_model_global:
        print("[info] full-model sweep (all tensors in checkpoint intersection vs pretrained base) …", flush=True)
        if args.method in ("lora", "dropbp_lora"):
            full_model_stats = accumulate_global_stats_lora(
                pretrained,
                ft,
                args.eps,
                args.threshold,
                progress=args.progress,
                num_workers=args.full_model_workers,
            )
        else:
            full_model_stats = accumulate_global_stats_full_ft(
                pretrained,
                ft,
                args.eps,
                args.threshold,
                progress=args.progress,
                num_workers=args.full_model_workers,
            )
        if full_model_stats:
            print(
                f"[info] full_model n_params={full_model_stats['n_params']} "
                f"mean_abs_relative_delta={full_model_stats['mean_abs_relative_delta']:.6g} "
                f"changed_fraction={full_model_stats['changed_fraction']:.6g}",
                flush=True,
            )

    out_dir = args.out_dir.resolve()
    master_rows = build_master_rows(meta, rows, summary_rows, compact_stats, full_model_stats or None)
    write_master_csv(master_csv, master_rows, mode=args.csv_mode)

    if args.plots:
        for proj in ("v", "o"):
            plot_layer_curves(
                rows, proj, args.family, args.task, args.method, out_dir / f"{stem}_layerwise_{proj}.png"
            )

    print(
        f"[done] master CSV ({args.csv_mode}): {master_csv}  rows={len(master_rows)}"
        + (f"  plots: {stem}_layerwise_*.png" if args.plots else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
