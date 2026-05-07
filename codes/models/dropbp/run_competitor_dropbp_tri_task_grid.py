#!/usr/bin/env python3
"""
3 models × 3 tasks (SST-2, IMDB, MMLU) — **DropBP + LoRA** finetuning, then post-train eval.

Training defaults mirror ``run_competitor_spectrum_tri_task_grid.py`` (subset sizes, per-task batch
sizes aligned with ``run_sst2_imdb_mmlu_finetune_run1.sh``).  Learning rate defaults follow
``codes/gsm8k/finetune_gsm8k_competitor.py`` for ``dropbp_lora``: **2e-4** for all base models unless
``--lr`` is set.

**Protocol defaults (paper-style tri-task runs):** LoRA ``r=16``, ``alpha=32``, target modules
``{q,k,v,o}_proj`` (HF names: ``q_proj k_proj v_proj o_proj``).  DropBP target **average backward drop
rate** ``p`` = ``--dropbp-rate`` (default **0.2**).  **Sensitivity-based layer drop rates** after the
first ~10% of training are implemented **inside** the patched ``Trainer`` in ``transformers_dropbp`` (it calls
``DropBPHandler.sensitivity_based_drop_bp`` once the dataloader step reaches
``floor(0.1 × max_steps × gradient_accumulation_steps)`` (and upstream also requires the **first**
training epoch).  If that index is **larger than batches in epoch 0**, calibration never runs — use
enough data/epochs/steps for your grid.  **Important:** upstream **skips** calibration when
``measure_time_memory=True`` — do **not** use ``--dropbp-profile`` if you need sensitivity adjustment.

**Requirement:** Hugging Face ``Trainer`` must accept DropBP kwargs (``drop_rate``, …).  Install the
official patched package from the DropBP repo (``huggingface/transformers_dropbp``) — typically in a
**dedicated virtualenv**, not stock ``transformers``.

**Peft pin:** current ``transformers_dropbp`` is ``4.46.x``-era; use ``peft==0.13.2`` (newer Peft may
crash with ``torch.distributed.tensor`` / DTensor checks on this stack).

**Repo root on ``PYTHONPATH``:** editable installs of the DropBP package sometimes omit ``dropbp.handler``
from the import map — this driver prepends ``DROPBP_REPO_ROOT`` or ``~/topo/vendor/dropbp`` automatically.

Eval (default ``--eval-backend cf3``): same as the spectrum grid — ``eval_catastrophic_forgetting3.py``
on the matching benchmark only, k-fold mean ± std.  LoRA outputs are loaded via ``adapter_config.json``
(Peft) inside each run directory.

Example
  # After activating your DropBP env:
  python run_competitor_dropbp_tri_task_grid.py --phase train --seed 42
  python run_competitor_dropbp_tri_task_grid.py --phase eval

Smoke (tiny data, 1 epoch, one model/task)
  python run_competitor_dropbp_tri_task_grid.py --models meta-llama/Llama-3.1-8B --tasks sst2 \\
      --epochs 1 --subset-size 512 --batch-imdb 8 --logging-steps 1 --stamp smoke_dropbp

---------------------------------------------------------------------------
Implementation / ops checklist (manual vs automated)
---------------------------------------------------------------------------
1. **Env:** New venv; install DropBP ``transformers_dropbp`` per upstream README (clone
   WooSunghyeon/dropbp, ``pip install -v -e .``, then ``pip install -v -e huggingface/transformers_dropbp``).
   Match CUDA/torch pins to your cluster or accept upstream pins — document versions used.
2. **Deps:** peft, datasets, accelerate, bitsandbytes (if used), same task stacks as topo-env.
3. **Hyperparameters:** defaults match LoRA ``r=16``, ``α=32``, ``q/k/v/o``, DropBP ``p=0.2``; tune
   ``--lora-dropout``, ``--lr``, ``--dropbp-rate``, modules per architecture if needed.
4. **Architecture coverage:** Upstream DropBP patches focus on Llama paths in ``modeling_llama.py``;
   verify behavior on **Qwen3** and **Mistral** under your fork version (may need upstream fixes or fallbacks).
5. **Table metrics:** Train.% / Upd.% for LoRA + DropBP need agreed definitions (adapter-only vs merged-full);
   run ``weight_change_descriptive_stats.py`` or adapter-norm summaries accordingly — not wired here.
6. **Throughput / memory logging:** ``--dropbp-measure-time-memory`` / ``--dropbp-profile`` writes
   ``<run_dir>/dropbp_throughput.txt`` but **disables** upstream sensitivity calibration for that run.

Paths assume this file lives under ``…/numpy_weights/exploration-finetuning/competitor/``.  Code discovery
uses ``CODES_ROOT`` / ``TOPO_CODES`` or ``~/topo/codes`` like the spectrum grid driver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

_THIS = Path(__file__).resolve()
COMPETITOR_ROOT = _THIS.parent


def _find_codes_root() -> Path:
    marker = Path("gsm8k") / "finetune_gsm8k_competitor.py"
    for key in ("CODES_ROOT", "TOPO_CODES"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw).expanduser().resolve()
            if (p / marker).is_file():
                return p
    for p in [_THIS.parent, *list(_THIS.parents)]:
        cand = p / "codes" / marker
        if cand.is_file():
            return p / "codes"
    home_fb = Path.home() / "topo" / "codes"
    if (home_fb / marker).is_file():
        return home_fb.resolve()
    raise RuntimeError(
        f"Could not locate …/codes with gsm8k/finetune_gsm8k_competitor.py. "
        f"Set CODES_ROOT=/abs/path/to/codes or place codes next to an ancestor of: {_THIS}"
    )


CODES_ROOT = _find_codes_root()
_codes = str(CODES_ROOT)
if _codes not in sys.path:
    sys.path.insert(0, _codes)

# Editable ``pip install -e`` for the DropBP repo often registers only ``dropbp.cpp_extention`` in the
# meta-path finder; ``transformers_dropbp`` still does ``from dropbp.handler import DropBPHandler``.
# Put the **repository root** (parent of the ``dropbp/`` package dir) on sys.path.
def _ensure_dropbp_repo_on_path() -> None:
    raw = os.environ.get("DROPBP_REPO_ROOT")
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser().resolve())
    candidates.append(Path.home() / "topo" / "vendor" / "dropbp")
    for root in candidates:
        if (root / "dropbp" / "handler.py").is_file():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


_ensure_dropbp_repo_on_path()

from gsm8k.finetune_gsm8k_competitor import (  # noqa: E402
    count_params,
    default_lr,
    get_gpu_report,
    maybe_make_dropbp_trainer,
    print_trainable_preview,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # noqa: E402
from imdb.data_preprocessing_imdb import (  # noqa: E402
    custom_data_collator as imdb_collator,
    infer_prompt_format_from_model_id as imdb_prompt_fmt,
    preprocess_dataset as imdb_preprocess,
)
from mmlu.data_preprocessing_mmlu import (  # noqa: E402
    custom_data_collator as mmlu_collator,
    infer_prompt_format_from_model_id as mmlu_prompt_fmt,
    preprocess_dataset as mmlu_preprocess,
)
from sst2.data_preprocessing_sst2 import (  # noqa: E402
    custom_data_collator as sst2_collator,
    infer_prompt_format_from_model_id as sst2_prompt_fmt,
    preprocess_dataset as sst2_preprocess,
)

DEFAULT_MODELS: Tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen3-8B-Base",
    "mistralai/Mistral-7B-v0.3",
)
TASKS: Tuple[str, ...] = ("sst2", "imdb", "mmlu")

METHOD_TAG = "dropbp_lora"
REPORT_NAME = "training_report_dropbp_lora.json"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "-").replace("_", "-")


class MetricsCallback(TrainerCallback):
    def __init__(self) -> None:
        self.steps: List[int] = []
        self.losses: List[float] = []
        self.learning_rates: List[float] = []
        self.gradient_norms: List[float] = []
        self.step_times: List[float] = []
        self.epoch_losses: List[float] = []
        self._cur_epoch_losses: List[float] = []
        self._step_start: Optional[float] = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self.step_times.append(time.time() - self._step_start)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(float(logs["loss"]))
            self._cur_epoch_losses.append(float(logs["loss"]))
        if "learning_rate" in logs:
            self.learning_rates.append(float(logs["learning_rate"]))
        if "grad_norm" in logs:
            self.gradient_norms.append(float(logs["grad_norm"]))

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._cur_epoch_losses:
            self.epoch_losses.append(float(np.mean(self._cur_epoch_losses)))
            self._cur_epoch_losses = []


def _task_train_dataset(
    task: str,
    *,
    model_id: str,
    tokenizer,
    max_length: int,
    subset_size: int,
    seed: int,
):
    if task == "sst2":
        ds = load_dataset("stanfordnlp/sst2")
        train_ds = ds["train"]
        if subset_size and subset_size < len(train_ds):
            ds_pos = train_ds.filter(lambda ex: ex["label"] == 1).shuffle(seed=seed)
            ds_neg = train_ds.filter(lambda ex: ex["label"] == 0).shuffle(seed=seed)
            k_pos = min(subset_size // 2, len(ds_pos))
            k_neg = min(subset_size - k_pos, len(ds_neg))
            parts = []
            if k_pos > 0:
                parts.append(ds_pos.select(range(k_pos)))
            if k_neg > 0:
                parts.append(ds_neg.select(range(k_neg)))
            if parts:
                train_ds = concatenate_datasets(parts).shuffle(seed=seed)
        pf = sst2_prompt_fmt(model_id)
        tok = train_ds.map(
            lambda ex: sst2_preprocess(ex, tokenizer, max_len=max_length, prompt_format=pf, is_train=True),
            remove_columns=train_ds.column_names,
        )
        collator = partial(sst2_collator, tokenizer=tokenizer)
        return tok, collator

    if task == "imdb":
        ds = load_dataset("stanfordnlp/imdb")
        train_ds = ds["train"].shuffle(seed=seed)
        if subset_size and subset_size < len(train_ds):
            train_ds = train_ds.select(range(subset_size))
        pf = imdb_prompt_fmt(model_id)
        tok = train_ds.map(
            lambda ex: imdb_preprocess(ex, tokenizer, max_len=max_length, prompt_format=pf, is_train=True),
            remove_columns=train_ds.column_names,
        )
        collator = partial(imdb_collator, tokenizer=tokenizer)
        return tok, collator

    if task == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        split_name = "auxiliary_train" if "auxiliary_train" in ds else "train"
        full_ds: Dataset = ds[split_name]
        n = min(subset_size, len(full_ds))
        train_ds = full_ds.shuffle(seed=seed).select(range(n))
        pf = mmlu_prompt_fmt(model_id)
        tok = train_ds.map(
            lambda ex: mmlu_preprocess(ex, tokenizer, max_len=max_length, prompt_format=pf, is_train=True),
            remove_columns=train_ds.column_names,
        )
        collator = partial(mmlu_collator, tokenizer=tokenizer)
        return tok, collator

    raise ValueError(f"Unknown task: {task}")


def _default_max_length(task: str) -> int:
    return {"sst2": 256, "imdb": 512, "mmlu": 512}[task]


def model_short_for_tri_grid(model_id: str) -> str:
    low = model_id.lower()
    if "qwen" in low:
        return "qwen-base"
    if "mistral" in low:
        return "mistral-7b-v03"
    return "llama"


def train_batch_for_task(task: str, batch_imdb: int) -> int:
    if task == "sst2":
        return 64
    if task == "mmlu":
        return 32
    return batch_imdb


def train_subset_rows(task: str, subset_sst_mmlu: int, subset_imdb: int) -> int:
    if task == "imdb":
        return subset_imdb
    return subset_sst_mmlu


def scaled_train_caps(nominal_sst_mmlu: int, nominal_imdb: int, subset_fraction: float) -> Tuple[int, int]:
    """Scale run1-style subset caps (at least one row each). ``subset_fraction`` in (0, 1]."""
    if subset_fraction <= 0 or subset_fraction > 1.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
    s = max(1, int(round(nominal_sst_mmlu * subset_fraction)))
    i = max(1, int(round(nominal_imdb * subset_fraction)))
    return s, i


def timing_extrapolation_note(training_min: float, subset_fraction: float) -> Dict[str, Any]:
    return {
        "subset_fraction": subset_fraction,
        "approx_equivalent_full_data_training_min": round(training_min / subset_fraction, 2)
        if subset_fraction < 1.0
        else round(training_min, 2),
        "note": "approx_equivalent_full_data_training_min = timing.training_min / subset_fraction "
        "(idealized linear scaling in train rows; ignores fixed model load; CF3/run1 eval cost unchanged "
        "unless you also shrink eval).",
    }


def _collect_checkpoints(out_root: Path) -> List[Path]:
    ew = sorted(out_root.glob("epoch_weights/checkpoint-epoch-*"))
    if ew:

        def _k(p: Path) -> int:
            m = re.search(r"checkpoint-epoch-(\d+)$", p.name)
            return int(m.group(1)) if m else 0

        return sorted(ew, key=_k)
    return sorted(out_root.glob("checkpoint-*"))


def run_final_eval_like_run1(
    *,
    task: str,
    run_dir: Path,
    base_model: str,
    model_short: str,
    eval_batch: int,
    eval_sst2_limit: Optional[int],
    eval_imdb_limit: Optional[int],
    eval_mmlu_limit: Optional[int],
    train_subset_report: int,
    max_new_tokens: int,
    overwrite: bool,
) -> Path:
    out_path = run_dir / f"final_eval_{task}_run1_style.json"
    if out_path.is_file() and not overwrite:
        print(f"[skip eval] exists: {out_path}", flush=True)
        return out_path

    checkpoints = _collect_checkpoints(run_dir)
    if checkpoints:
        ckpt_path = checkpoints[-1]
    elif any((run_dir / f).exists() for f in ("adapter_config.json", "config.json")):
        ckpt_path = run_dir
    else:
        raise FileNotFoundError(f"No checkpoint under {run_dir}")

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    adapter_config = ckpt_path / "adapter_config.json"
    if adapter_config.is_file():
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    _fold_kw: Dict[str, Any] = {}

    if task == "sst2":
        from sst2.eval_sst2 import evaluate_sst2 as evaluate_fn

        split = "validation"
        kws: Dict[str, Any] = dict(
            model=model,
            tokenizer=tok,
            split=split,
            batch_size=eval_batch,
            progress_bar=True,
            prompt_model_id=base_model,
            max_new_tokens=max_new_tokens,
        )
        kws.update(_fold_kw)
        if eval_sst2_limit is not None:
            kws["limit"] = eval_sst2_limit
        metrics = evaluate_fn(**kws)
        acc_percent = float(metrics["accuracy"]) * 100.0
        acc_std = float(metrics.get("accuracy_std", 0.0)) * 100.0
    elif task == "imdb":
        from imdb.eval_imdb import evaluate_imdb as evaluate_fn

        split = "test"
        kw: Dict[str, Any] = dict(
            model=model,
            tokenizer=tok,
            split=split,
            batch_size=eval_batch,
            progress_bar=True,
            prompt_model_id=base_model,
            max_new_tokens=max_new_tokens,
        )
        kw.update(_fold_kw)
        if eval_imdb_limit is not None:
            kw["limit"] = eval_imdb_limit
        metrics = evaluate_fn(**kw)
        acc_percent = float(metrics["accuracy"]) * 100.0
        acc_std = float(metrics.get("accuracy_std", 0.0)) * 100.0
    elif task == "mmlu":
        from mmlu.eval_mmlu import evaluate_mmlu as evaluate_fn

        split = "validation"
        kwm: Dict[str, Any] = dict(
            model=model,
            tokenizer=tok,
            split=split,
            batch_size=eval_batch,
            progress_bar=True,
            save_jsonl=None,
            save_tsv=None,
            prompt_model_id=base_model,
            max_new_tokens=max_new_tokens,
        )
        kwm.update(_fold_kw)
        if eval_mmlu_limit is not None:
            kwm["limit"] = eval_mmlu_limit
        metrics = evaluate_fn(**kwm)
        raw_acc = float(metrics["acc"])
        acc_percent = raw_acc if raw_acc > 1.0 else raw_acc * 100.0
        acc_std = float(metrics.get("acc_std", 0.0))
    else:
        raise ValueError(task)

    rec: Dict[str, Any] = {
        "dataset": task,
        "model_short": model_short,
        "method": METHOD_TAG,
        "out_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "split": split,
        "accuracy_percent": round(acc_percent, 4),
        "accuracy_percent_std": round(acc_std, 4),
        "metrics": metrics,
        "eval_batch": eval_batch,
        "train_subset_size": train_subset_report,
        "eval_sst2_limit": eval_sst2_limit,
        "eval_imdb_limit": eval_imdb_limit,
        "eval_mmlu_limit": eval_mmlu_limit,
        "max_new_tokens": max_new_tokens,
        "eval_pipeline": "run_sst2_imdb_mmlu_finetune_run1.sh:run_final_eval (evaluate_* task modules)",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"  FINAL {task.upper()}  {acc_percent:.2f}% ± {acc_std:.2f}%  (split={split}, ckpt={ckpt_path})", flush=True)
    print(f"  Saved: {out_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def train_one_dropbp_lora(
    *,
    task: str,
    model_id: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_length: Optional[int],
    subset_size: int,
    seed: int,
    lr: Optional[float],
    logging_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    dropbp_rate: float,
    dropbp_measure_time_memory: bool,
    dropbp_time_warmup_steps: int,
    dropbp_time_measure_steps: int,
    dropbp_throughput_path: Optional[str],
    subset_fraction: float = 1.0,
    nominal_subset_sst_mmlu: int = 0,
    nominal_subset_imdb: int = 0,
) -> Path:
    set_seed(seed)
    max_length = max_length if max_length is not None else _default_max_length(task)
    lr_use = float(lr) if lr is not None else default_lr(model_id, METHOD_TAG)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_NAME
    if report_path.is_file():
        print(f"[skip train] exists: {report_path}", flush=True)
        return report_path

    print(f"\n{'=' * 70}\n  TRAIN  {METHOD_TAG}  task={task}  model={model_id}\n{'=' * 70}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model_load_s = time.time() - t0

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_params, trainable_params, trainable_pct = count_params(model)
    trainable_names = print_trainable_preview(model, limit=30)

    thr_default = str(output_dir / "dropbp_throughput.txt")
    throughput_path = dropbp_throughput_path if dropbp_throughput_path else thr_default

    t0 = time.time()
    train_ds, collator = _task_train_dataset(
        task,
        model_id=model_id,
        tokenizer=tokenizer,
        max_length=max_length,
        subset_size=subset_size,
        seed=seed,
    )
    data_prep_s = time.time() - t0
    print(f"  Train rows: {len(train_ds)}  |  data prep {data_prep_s:.1f}s", flush=True)

    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    total_steps_est = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps_est * 0.03))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr_use,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=None,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
        seed=seed,
        data_seed=seed,
    )

    metrics_cb = MetricsCallback()
    trainer = maybe_make_dropbp_trainer(
        method=METHOD_TAG,
        model=model,
        training_args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[metrics_cb],
        drop_rate=dropbp_rate,
        measure_time_memory=dropbp_measure_time_memory,
        time_warmup_steps=dropbp_time_warmup_steps,
        time_measure_steps=dropbp_time_measure_steps,
        throughput_path=throughput_path,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    train_result = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_s = time.time() - t0

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    num_tokens_seen = getattr(trainer.state, "num_input_tokens_seen", None)
    report: Dict[str, Any] = {
        "experiment": f"{task.upper()} DropBP+LoRA finetuning (tri-task grid)",
        "task": task,
        "model": model_id,
        "method": METHOD_TAG,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": lr_use,
            "lr_scheduler": "cosine",
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation": grad_accum,
            "effective_batch_size": eff_batch,
            "max_seq_length": max_length,
            "warmup_steps": warmup_steps,
            "subset_size": subset_size,
            "subset_fraction": subset_fraction,
            "subset_nominal_caps": {
                "sst_mmlu": nominal_subset_sst_mmlu,
                "imdb": nominal_subset_imdb,
            },
            "seed": seed,
        },
        "lora_config": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": list(lora_target_modules),
        },
        "dropbp_config": {
            "drop_rate": dropbp_rate,
            "measure_time_memory": dropbp_measure_time_memory,
            "time_warmup_steps": dropbp_time_warmup_steps,
            "time_measure_steps": dropbp_time_measure_steps,
            "throughput_path": throughput_path,
            "sensitivity_calibration": (
                "upstream Trainer: sensitivity_based_drop_bp at micro-batch step "
                "floor(0.1 * max_steps * gradient_accumulation_steps); skipped if measure_time_memory"
            ),
            "sensitivity_calibration_skipped": bool(dropbp_measure_time_memory),
        },
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_pct": round(trainable_pct, 6),
            "num_trainable_tensors": len(trainable_names),
        },
        "training_results": {
            "trainer_train_runtime": getattr(train_result, "metrics", {}).get("train_runtime"),
            "total_logged_steps": len(metrics_cb.steps),
            "final_loss": round(metrics_cb.losses[-1], 4) if metrics_cb.losses else None,
            "best_loss": round(min(metrics_cb.losses), 4) if metrics_cb.losses else None,
            "epoch_losses": [round(x, 4) for x in metrics_cb.epoch_losses],
        },
        "timing": {
            "model_load_s": round(model_load_s, 1),
            "data_prep_s": round(data_prep_s, 1),
            "training_s": round(training_s, 1),
            "training_min": round(training_s / 60, 2),
        },
        "timing_extrapolation": timing_extrapolation_note(training_s / 60.0, subset_fraction),
        "gpu_memory": get_gpu_report(),
        "metrics_log": {
            "steps": metrics_cb.steps,
            "losses": [round(x, 4) for x in metrics_cb.losses],
            "learning_rates": [round(x, 8) for x in metrics_cb.learning_rates],
            "gradient_norms": [round(x, 4) for x in metrics_cb.gradient_norms],
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Wrote {report_path}", flush=True)
    return report_path


def run_cf3_eval(
    *,
    checkpoint_dir: Path,
    eval_out: Path,
    eval_py: Path,
    seed: Optional[int],
    eval_split_seed: int,
    eval_num_folds: int,
    eval_split_indices_dir: Path,
    batch_size: int,
    max_samples: Optional[int],
    overwrite_report: bool,
    benchmarks: Tuple[str, ...],
) -> int:
    eval_out.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        sys.executable,
        str(eval_py),
        "--model",
        str(checkpoint_dir),
        "--model-name",
        checkpoint_dir.name,
        "--benchmarks",
        *benchmarks,
        "--output-dir",
        str(eval_out),
        "--batch-size",
        str(batch_size),
        "--eval-split-seed",
        str(eval_split_seed),
        "--eval-num-folds",
        str(eval_num_folds),
        "--eval-split-indices-dir",
        str(eval_split_indices_dir),
        "--chat-template",
        "auto",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    if overwrite_report:
        cmd.append("--overwrite-report")
    # PEFT adapter-only run dirs need --is-lora so eval merges adapters; otherwise CF3 loads base only.
    if (checkpoint_dir / "adapter_config.json").is_file():
        cmd.append("--is-lora")
    print(f"\n[eval] {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=str(eval_py.parent)).returncode


def parse_models(raw: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if raw:
        return tuple(raw)
    return DEFAULT_MODELS


def main() -> None:
    ap = argparse.ArgumentParser(description="DropBP+LoRA: 3×3 SST2/IMDB/MMLU train + eval (run1 or CF3).")
    ap.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    ap.add_argument("--models", nargs="+", default=None, help="HF model ids (default: Llama, Qwen3, Mistral v0.3).")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--root-out-dir", type=str, default=None, help="Parent for all runs (default: competitor/dropbp_tri_task_lora/<stamp>/)")
    ap.add_argument("--stamp", type=str, default=None, help="Output folder stamp (default: UTC ymd_HMS).")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--subset-size", type=int, default=20_000, help="SST2 + MMLU train rows (run1 SUBSET_SIZE).")
    ap.add_argument("--imdb-subset-size", type=int, default=25_000, help="IMDB train rows (run1 SUBSET_SIZE_IMDB).")
    ap.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help="Scale SST/MMLU and IMDB train subset caps (e.g. 0.25 for ¼). See timing_extrapolation in report.",
    )
    ap.add_argument("--batch-imdb", type=int, default=32, help="IMDB per-device train batch (run1 BATCH_IMDB).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="If omitted, uses finetune_gsm8k_competitor.default_lr(..., dropbp_lora) → 2e-4 for all models.",
    )
    ap.add_argument("--max-length", type=int, default=None, help="Per-task default if omitted: sst2=256, imdb/mmlu=512.")
    ap.add_argument("--logging-steps", type=int, default=5)

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA target module names (must exist on each architecture).",
    )

    ap.add_argument(
        "--dropbp-rate",
        type=float,
        default=0.2,
        help="Target average backward drop rate p (passed as Trainer drop_rate). Default 0.2.",
    )
    ap.add_argument(
        "--dropbp-measure-time-memory",
        action="store_true",
        help="Trainer throughput/memory logging → <run_dir>/dropbp_throughput.txt. "
        "Upstream skips sensitivity_based_drop_bp when this is on.",
    )
    ap.add_argument(
        "--dropbp-profile",
        action="store_true",
        help="Same as --dropbp-measure-time-memory (disables sensitivity calibration for that run).",
    )
    ap.add_argument("--dropbp-time-warmup-steps", type=int, default=1)
    ap.add_argument("--dropbp-time-measure-steps", type=int, default=3)

    ap.add_argument(
        "--eval-backend",
        choices=("run1", "cf3"),
        default="cf3",
        help="cf3 (default): eval_catastrophic_forgetting3.py, k-fold on matching benchmark only.",
    )
    ap.add_argument("--eval-batch-llama-qwen", type=int, default=64)
    ap.add_argument("--eval-batch-mistral", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--eval-max-new-tokens", type=int, default=8)
    ap.add_argument("--eval-sst2-limit", type=int, default=None)
    ap.add_argument("--eval-imdb-limit", type=int, default=None)
    ap.add_argument("--eval-mmlu-limit", type=int, default=None)
    ap.add_argument("--eval-split-seed", type=int, default=42)
    ap.add_argument("--eval-num-folds", type=int, default=3)
    ap.add_argument("--eval-split-indices-dir", type=str, default=None)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--overwrite-eval-report", action="store_true")
    ap.add_argument(
        "--skip-train-if-complete",
        action="store_true",
        help=f"Skip training when `{REPORT_NAME}` already exists in the run directory (resume grids without redoing finished models).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dropbp_profile:
        args.dropbp_measure_time_memory = True

    if args.subset_fraction <= 0 or args.subset_fraction > 1.0:
        raise SystemExit("--subset-fraction must be in (0, 1]")
    cap_sst, cap_imdb = scaled_train_caps(args.subset_size, args.imdb_subset_size, args.subset_fraction)

    models = parse_models(args.models)
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.root_out_dir:
        root = Path(args.root_out_dir)
    else:
        root = COMPETITOR_ROOT / "dropbp_tri_task_lora" / stamp
    root.mkdir(parents=True, exist_ok=True)

    eval_split_dir = Path(args.eval_split_indices_dir) if args.eval_split_indices_dir else (root / "eval_split_indices")
    eval_py = CODES_ROOT / "gsm8k" / "eval_catastrophic_forgetting3.py"
    if args.eval_backend == "cf3" and not eval_py.is_file():
        raise FileNotFoundError(f"Missing eval script: {eval_py}")

    dr_tag = str(args.dropbp_rate).replace(".", "p")
    runs_meta: List[Dict[str, Any]] = []
    for m in models:
        for task in args.tasks:
            slug = model_slug(m)
            run_name = f"{task}-dropbp-lora-dr{dr_tag}-e{args.epochs}-{stamp}"
            run_dir = root / slug / task / run_name
            runs_meta.append({"model": m, "task": task, "run_dir": str(run_dir)})

    manifest: Dict[str, Any] = {
        "stamp": stamp,
        "method": METHOD_TAG,
        "models": list(models),
        "tasks": list(args.tasks),
        "root": str(root),
        "eval_backend": args.eval_backend,
        "dropbp_rate": args.dropbp_rate,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "target_modules": args.lora_target_modules},
        "subset_fraction": args.subset_fraction,
        "subset_nominal_caps": {"sst_mmlu": args.subset_size, "imdb": args.imdb_subset_size},
        "subset_effective_caps": {"sst_mmlu": cap_sst, "imdb": cap_imdb},
        "runs": runs_meta,
    }

    if args.phase in ("train", "all"):
        for entry in runs_meta:
            m = entry["model"]
            task = entry["task"]
            run_dir = Path(entry["run_dir"])
            if args.dry_run:
                print(f"[dry-run] train -> {run_dir}", flush=True)
                continue
            report_path = run_dir / REPORT_NAME
            if args.skip_train_if_complete and report_path.is_file():
                print(f"[skip train] already complete: {report_path}", flush=True)
                continue
            rows = train_subset_rows(task, cap_sst, cap_imdb)
            tbatch = train_batch_for_task(task, args.batch_imdb)
            train_one_dropbp_lora(
                task=task,
                model_id=m,
                output_dir=run_dir,
                epochs=args.epochs,
                batch_size=tbatch,
                grad_accum=args.grad_accum,
                max_length=args.max_length,
                subset_size=rows,
                seed=args.seed,
                lr=args.lr,
                logging_steps=args.logging_steps,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=list(args.lora_target_modules),
                dropbp_rate=args.dropbp_rate,
                dropbp_measure_time_memory=args.dropbp_measure_time_memory,
                dropbp_time_warmup_steps=args.dropbp_time_warmup_steps,
                dropbp_time_measure_steps=args.dropbp_time_measure_steps,
                dropbp_throughput_path=None,
                subset_fraction=args.subset_fraction,
                nominal_subset_sst_mmlu=args.subset_size,
                nominal_subset_imdb=args.imdb_subset_size,
            )

    if args.phase in ("eval", "all") and not args.dry_run:
        for entry in runs_meta:
            run_dir = Path(entry["run_dir"])
            task = entry["task"]
            m = entry["model"]
            mshort = model_short_for_tri_grid(m)
            if args.eval_backend == "run1":
                done_marker = run_dir / f"final_eval_{task}_run1_style.json"
                if done_marker.is_file() and not args.overwrite_eval_report:
                    print(f"[skip eval] exists: {done_marker}", flush=True)
                    continue
                if args.eval_batch_size is not None:
                    ebs = int(args.eval_batch_size)
                elif mshort == "mistral-7b-v03":
                    ebs = args.eval_batch_mistral
                else:
                    ebs = args.eval_batch_llama_qwen
                train_n = train_subset_rows(task, cap_sst, cap_imdb)
                run_final_eval_like_run1(
                    task=task,
                    run_dir=run_dir,
                    base_model=m,
                    model_short=mshort,
                    eval_batch=ebs,
                    eval_sst2_limit=args.eval_sst2_limit,
                    eval_imdb_limit=args.eval_imdb_limit,
                    eval_mmlu_limit=args.eval_mmlu_limit,
                    train_subset_report=train_n,
                    max_new_tokens=args.eval_max_new_tokens,
                    overwrite=args.overwrite_eval_report,
                )
                entry["eval_json"] = str(done_marker)
            else:
                eval_out = run_dir / f"cf3_eval_{task}"
                done_marker = eval_out / "catastrophic_forgetting3_report.json"
                if done_marker.is_file() and not args.overwrite_eval_report:
                    print(f"[skip eval] exists: {done_marker}", flush=True)
                    continue
                if args.eval_batch_size is not None:
                    cf3_bs = int(args.eval_batch_size)
                elif mshort == "mistral-7b-v03":
                    cf3_bs = args.eval_batch_mistral
                else:
                    cf3_bs = args.eval_batch_llama_qwen
                rc = run_cf3_eval(
                    checkpoint_dir=run_dir,
                    eval_out=eval_out,
                    eval_py=eval_py,
                    seed=args.eval_seed,
                    eval_split_seed=args.eval_split_seed,
                    eval_num_folds=args.eval_num_folds,
                    eval_split_indices_dir=eval_split_dir,
                    batch_size=cf3_bs,
                    max_samples=args.max_samples,
                    overwrite_report=args.overwrite_eval_report,
                    benchmarks=(task,),
                )
                entry["eval_rc"] = rc
                if rc == 0:
                    entry["eval_report"] = str(done_marker)
                if rc != 0:
                    print(f"[warn] eval non-zero rc={rc} for {run_dir}", flush=True)

    manifest_path = root / "tri_task_manifest.json"
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nWrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
