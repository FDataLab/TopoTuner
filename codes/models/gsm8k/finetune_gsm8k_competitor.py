#!/usr/bin/env python3
"""
GSM8K competitor finetuning script.

This file keeps your working finetune_gsm8k.py untouched and adds competitor methods:

  --method spectrum       Spectrum matched-budget freezing
  --method dropbp_full    DropBP + full finetuning
  --method dropbp_lora    DropBP + LoRA

It also keeps the original baselines:

  --method full
  --method lora

Main logged comparison metrics:
  - trainable params and trainable %
  - peak GPU allocated/reserved memory
  - training time
  - samples/sec
  - tokens/sec when Trainer exposes num_input_tokens_seen

Checkpoints live under ``<output-dir>/checkpoint-*``; ``trainer.save_model`` also writes weights
tokenizer into ``<output-dir>`` — load that path with ``eval_gsm8k.py`` for 8-shot GSM8K test accuracy
(same methodology as your other runs). Optionally pass ``--eval-after-train`` to run ``eval_gsm8k.py``
automatically and store accuracy + timing in ``training_report_<method>.json``.

Default ``--output-dir`` (when omitted)::

    <TOPO>/numpy_weights/exploration-finetuning/competitor/<method>/gsm8k-<method>-finetuned

Example for ``--method spectrum``::

    .../exploration-finetuning/competitor/spectrum/gsm8k-spectrum-finetuned
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
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


def default_output_dir_for_method(method: str) -> str:
    """HF checkpoints, JSON report, and plots under exploration-finetuning/competitor/<method>/."""
    topo = Path(__file__).resolve().parents[2]
    out = topo / "numpy_weights" / "exploration-finetuning" / "competitor" / method
    out.mkdir(parents=True, exist_ok=True)
    return str(out / f"gsm8k-{method}-finetuned")


def _gsm8k_eval_result_json_path(checkpoint_dir: str, eval_results_dir: str) -> Path:
    """Matches eval_gsm8k.py naming: gsm8k_<sanitized-model-path>.json"""
    slug = checkpoint_dir.replace("/", "_").replace(".", "-")
    return Path(eval_results_dir) / f"gsm8k_{slug}.json"


def run_eval_gsm8k_after_train(
    *,
    checkpoint_dir: str,
    eval_results_dir: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    """Subprocess canonical 8-shot CoT GSM8K eval; return summary for ``training_report``."""
    eval_py = Path(__file__).resolve().parent / "eval_gsm8k.py"
    os.makedirs(eval_results_dir, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_py),
        "--model",
        checkpoint_dir,
        "--output-dir",
        eval_results_dir,
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    print(f"\n  [Eval] Running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(eval_py.parent))
    wrapper_wall = time.time() - t0
    out_json = _gsm8k_eval_result_json_path(checkpoint_dir, eval_results_dir)
    summary: dict = {
        "subprocess_rc": proc.returncode,
        "wrapper_wall_s": round(wrapper_wall, 1),
        "eval_batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "result_json": str(out_json) if out_json.is_file() else None,
    }
    if proc.returncode != 0:
        summary["error"] = "eval_gsm8k.py exited non-zero"
        return summary
    if not out_json.is_file():
        summary["error"] = "eval output json not found"
        return summary
    with open(out_json) as f:
        data = json.load(f)
    summary["accuracy"] = data.get("accuracy")
    summary["correct"] = data.get("correct")
    summary["total"] = data.get("total")
    summary["eval_time_s"] = data.get("eval_time_s")
    summary["benchmark"] = data.get("benchmark")
    summary["setting"] = data.get("setting")
    return summary


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────

class GSM8KDataset(torch.utils.data.Dataset):
    """Causal LM dataset for GSM8K with question tokens masked from loss."""

    def __init__(self, data, tokenizer, max_length: int = 512):
        self.samples = []
        for example in data:
            prompt = f"Question: {example['question']}\nAnswer: "
            completion = f"{example['answer']}{tokenizer.eos_token}"
            full_text = prompt + completion

            full_enc = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            prompt_enc = tokenizer(
                prompt,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[: prompt_enc["input_ids"].shape[1]] = -100

            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Right-pad batch tensors."""
    max_len = max(s["input_ids"].shape[0] for s in batch)

    def pad(t, val, n):
        return torch.cat([t, torch.full((n,), val, dtype=t.dtype)])

    input_ids, attention_mask, labels = [], [], []
    for s in batch:
        n = max_len - s["input_ids"].shape[0]
        input_ids.append(pad(s["input_ids"], 0, n))
        attention_mask.append(pad(s["attention_mask"], 0, n))
        labels.append(pad(s["labels"], -100, n))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


# ──────────────────────────────────────────────────────────────────────
#  Metrics callback
# ──────────────────────────────────────────────────────────────────────

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.steps = []
        self.losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.step_times = []
        self.epoch_losses = []
        self._cur_epoch_losses = []
        self._step_start = None

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


# ──────────────────────────────────────────────────────────────────────
#  Competitor helpers: Spectrum
# ──────────────────────────────────────────────────────────────────────

def load_spectrum_unfrozen_parameters(path: str | os.PathLike) -> list[str]:
    """
    Load Spectrum-selected unfrozen parameter patterns.

    Supported formats:
      - YAML with `unfrozen_parameters:` list
      - plain txt, one pattern per line
      - JSON list
    """
    if path is None:
        raise ValueError("--spectrum-unfrozen-file is required for --method spectrum")

    path = str(path)
    with open(path, "r") as f:
        text = f.read()

    if path.endswith(".json"):
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("Spectrum JSON file must contain a list of strings.")
        return [str(x).strip() for x in items if str(x).strip()]

    patterns = []
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("unfrozen_parameters:"):
            in_block = True
            continue

        if line.startswith("-"):
            item = line[1:].strip().strip("'").strip('"')
            if item:
                patterns.append(item)
            continue

        # Plain txt mode; avoid collecting unrelated yaml keys.
        if not in_block and ":" not in line:
            patterns.append(line.strip("'").strip('"'))

    if not patterns:
        raise ValueError(f"No Spectrum unfrozen parameters found in: {path}")
    return patterns


def _match_pattern(name: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    # Allow regex when user explicitly writes regex anchors.
    if pattern.startswith("^") or pattern.endswith("$"):
        return re.search(pattern, name) is not None
    return pattern in name


def apply_spectrum_freezing(
    model,
    unfrozen_patterns: Iterable[str],
    target_modules: Optional[list[str]] = None,
) -> list[str]:
    """Freeze all parameters, then unfreeze Spectrum-selected tensors."""
    patterns = list(unfrozen_patterns)

    for _, p in model.named_parameters():
        p.requires_grad = False

    selected = []
    for name, p in model.named_parameters():
        if target_modules is not None and not any(t in name for t in target_modules):
            continue
        if any(_match_pattern(name, pat) for pat in patterns):
            p.requires_grad = True
            selected.append(name)

    if not selected:
        raise ValueError(
            "Spectrum selected 0 trainable tensors. Check the Spectrum YAML/list "
            "and compare its names with model.named_parameters()."
        )
    return selected


# ──────────────────────────────────────────────────────────────────────
#  Competitor helpers: DropBP
# ──────────────────────────────────────────────────────────────────────

def maybe_make_dropbp_trainer(
    *,
    method: str,
    model,
    training_args,
    train_dataset,
    data_collator,
    callbacks,
    drop_rate: float,
    measure_time_memory: bool,
    time_warmup_steps: int,
    time_measure_steps: int,
    throughput_path: str,
):
    """
    Create a Trainer.

    For DropBP methods, this first tries the official DropBP HuggingFace integration,
    where a patched transformers.Trainer accepts extra keyword args such as drop_rate.

    If the installed transformers is not the DropBP-patched one, this raises a clear
    error instead of silently running without DropBP.
    """
    common_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if not method.startswith("dropbp"):
        return Trainer(**common_kwargs)

    try:
        return Trainer(
            **common_kwargs,
            drop_rate=drop_rate,
            measure_time_memory=measure_time_memory,
            time_warmup_steps=time_warmup_steps,
            time_measure_steps=time_measure_steps,
            throughput_path=throughput_path,
        )
    except TypeError as e:
        raise RuntimeError(
            "DropBP method was requested, but your installed transformers.Trainer "
            "does not accept DropBP arguments. Install the official DropBP repo and "
            "its transformers_dropbp package, then rerun. Original error: " + str(e)
        ) from e


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def generate_training_plots(mcb: MetricsCallback, output_dir: str, method: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"GSM8K Finetuning — {method.upper()}", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    if mcb.losses:
        ax.plot(mcb.steps, mcb.losses, alpha=0.25, label="Step loss")
        w = max(1, len(mcb.losses) // 20)
        if w > 1:
            smooth = np.convolve(mcb.losses, np.ones(w) / w, mode="valid")
            ax.plot(mcb.steps[w - 1 :], smooth, lw=2, label=f"Smoothed (w={w})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if mcb.learning_rates:
        lr_steps = mcb.steps[: len(mcb.learning_rates)]
        ax.plot(lr_steps, mcb.learning_rates, lw=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    ax = axes[1, 0]
    if mcb.gradient_norms:
        gn_steps = mcb.steps[: len(mcb.gradient_norms)]
        ax.plot(gn_steps, mcb.gradient_norms, alpha=0.5, label="Grad norm")
        avg_gn = np.mean(mcb.gradient_norms)
        ax.axhline(y=avg_gn, ls="--", lw=2, label=f"Mean: {avg_gn:.2f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient L2 Norm")
        ax.set_title("Gradient Norms")
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if mcb.step_times:
        ax.plot(mcb.step_times, alpha=0.25)
        avg_t = np.mean(mcb.step_times)
        ax.axhline(y=avg_t, ls="--", lw=2, label=f"Avg: {avg_t:.2f}s")
        ax.set_xlabel("Step")
        ax.set_ylabel("Time (s)")
        ax.set_title("Step Duration")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"training_metrics_{method}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_epoch_loss_plot(mcb: MetricsCallback, output_dir: str, method: str):
    if not mcb.epoch_losses:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(mcb.epoch_losses) + 1))
    x = np.arange(len(epochs))
    bars = ax.bar(x, mcb.epoch_losses, edgecolor="black")
    for bar, loss in zip(bars, mcb.epoch_losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{loss:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title(f"Loss per Epoch — {method.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(epochs)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"epoch_loss_{method}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ──────────────────────────────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────────────────────────────

def default_lr(model_name: str, method: str) -> float:
    if method in {"lora", "dropbp_lora"}:
        return 2e-4
    if "mistral" in model_name.lower():
        return 5e-6
    return 2e-5


def get_gpu_report() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }


def count_params(model) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = trainable / total * 100 if total else 0.0
    return total, trainable, pct


def print_trainable_preview(model, limit: int = 40):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"  Trainable tensors: {len(names)}", flush=True)
    print(f"  First {min(limit, len(names))} trainable tensors:", flush=True)
    for name in names[:limit]:
        print(f"    {name}", flush=True)
    return names


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K competitor finetuning")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "lora", "spectrum", "dropbp_full", "dropbp_lora"],
        default="spectrum",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Training outputs (checkpoints, report, plots). Default: NW exploration-finetuning/competitor/<method>/gsm8k-<method>-finetuned",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA args
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Spectrum args
    parser.add_argument("--spectrum-unfrozen-file", type=str, default=None)
    parser.add_argument(
        "--spectrum-target-modules",
        nargs="+",
        default=None,
        help="Optional filter, e.g. --spectrum-target-modules v_proj o_proj",
    )

    # DropBP args. These require the official DropBP transformers_dropbp install.
    # Default p=0.2 aligns with tri-task DropBP+LoRA protocol; upstream Trainer runs sensitivity calibration ~10%.
    parser.add_argument("--dropbp-rate", type=float, default=0.2)
    parser.add_argument("--dropbp-measure-time-memory", action="store_true")
    parser.add_argument("--dropbp-time-warmup-steps", type=int, default=1)
    parser.add_argument("--dropbp-time-measure-steps", type=int, default=3)
    parser.add_argument("--dropbp-throughput-path", type=str, default=None)

    # Post-train GSM8K test eval (same as eval_gsm8k.py; optional)
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help="After save, run eval_gsm8k.py on full GSM8K test set; merge into training_report.",
    )
    parser.add_argument(
        "--eval-results-dir",
        type=str,
        default=None,
        help="Directory for gsm8k_*.json from eval_gsm8k. Default: <output-dir>/gsm8k_8shot_eval",
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.lr is None:
        args.lr = default_lr(args.model, args.method)

    if args.output_dir is None:
        args.output_dir = default_output_dir_for_method(args.method)

    if args.dropbp_throughput_path is None:
        args.dropbp_throughput_path = os.path.join(args.output_dir, "dropbp_throughput.txt")

    if args.method == "spectrum":
        if not args.spectrum_unfrozen_file:
            raise SystemExit("--method spectrum requires --spectrum-unfrozen-file (YAML list, txt, or JSON).")
        if not Path(args.spectrum_unfrozen_file).is_file():
            raise SystemExit(f"Spectrum unfrozen file not found: {args.spectrum_unfrozen_file}")

    timing = {}
    spectrum_selected_params = None

    print(f"\n{'=' * 70}", flush=True)
    print("  GSM8K COMPETITOR FINETUNING", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Method:          {args.method}", flush=True)
    print(f"  Model:           {args.model}", flush=True)
    print(f"  LR:              {args.lr}", flush=True)
    print(f"  Epochs:          {args.epochs}", flush=True)
    print(f"  Batch:           {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}", flush=True)
    print(f"  Max length:      {args.max_length}", flush=True)
    print(f"  Output dir:      {args.output_dir}", flush=True)
    print(f"  Seed:            {args.seed}", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB\n", flush=True)

    # [1/5] Load model/tokenizer
    print("[1/5] Loading model/tokenizer...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    timing["model_load_s"] = time.time() - t0
    print(f"  Loaded in {timing['model_load_s']:.1f}s", flush=True)

    # [2/5] Apply method
    print("\n[2/5] Applying finetuning method...", flush=True)
    is_lora_method = args.method in {"lora", "dropbp_lora"}
    is_dropbp_method = args.method.startswith("dropbp")

    if is_lora_method:
        from peft import LoraConfig, TaskType, get_peft_model

        print("  Applying LoRA...", flush=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif args.method == "full" or args.method == "dropbp_full":
        print("  Setting all parameters trainable...", flush=True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        for _, p in model.named_parameters():
            p.requires_grad = True

    elif args.method == "spectrum":
        print("  Applying Spectrum freezing...", flush=True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        patterns = load_spectrum_unfrozen_parameters(args.spectrum_unfrozen_file)
        spectrum_selected_params = apply_spectrum_freezing(
            model,
            patterns,
            target_modules=args.spectrum_target_modules,
        )
        print(f"  Spectrum file: {args.spectrum_unfrozen_file}", flush=True)
        print(f"  Spectrum patterns loaded: {len(patterns)}", flush=True)
        print(f"  Spectrum selected tensors: {len(spectrum_selected_params)}", flush=True)
        if args.spectrum_target_modules:
            print(f"  Spectrum target-module filter: {args.spectrum_target_modules}", flush=True)

    if is_dropbp_method:
        print("  DropBP requested. This requires official DropBP transformers_dropbp install.", flush=True)
        print(f"  DropBP rate: {args.dropbp_rate}", flush=True)

    total_params, trainable_params, trainable_pct = count_params(model)
    trainable_names = print_trainable_preview(model)
    print(f"  Total params:     {total_params:,}", flush=True)
    print(f"  Trainable params: {trainable_params:,} ({trainable_pct:.4f}%)", flush=True)

    # [3/5] Dataset
    print("\n[3/5] Preparing GSM8K dataset...", flush=True)
    t0 = time.time()
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]
    train_dataset = GSM8KDataset(train_data, tokenizer, max_length=args.max_length)
    timing["data_prep_s"] = time.time() - t0
    lengths = [s["input_ids"].shape[0] for s in train_dataset.samples]
    print(f"  Examples: {len(train_dataset)}", flush=True)
    print(
        f"  Token lengths — min: {min(lengths)}, max: {max(lengths)}, "
        f"mean: {np.mean(lengths):.0f}, median: {np.median(lengths):.0f}",
        flush=True,
    )

    # [4/5] Trainer
    print("\n[4/5] Configuring trainer...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // eff_batch
    total_steps_est = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps_est * 0.03))

    print(f"  Steps/epoch approx: {steps_per_epoch}", flush=True)
    print(f"  Total steps approx: {total_steps_est}", flush=True)
    print(f"  Warmup steps:       {warmup_steps}", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=None,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
        seed=args.seed,
        data_seed=args.seed,
    )

    metrics_cb = MetricsCallback()
    trainer = maybe_make_dropbp_trainer(
        method=args.method,
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[metrics_cb],
        drop_rate=args.dropbp_rate,
        measure_time_memory=args.dropbp_measure_time_memory,
        time_warmup_steps=args.dropbp_time_warmup_steps,
        time_measure_steps=args.dropbp_time_measure_steps,
        throughput_path=args.dropbp_throughput_path,
    )

    print("\n  Training...", flush=True)
    print("─" * 70, flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    train_result = trainer.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timing["training_s"] = time.time() - t0
    print("─" * 70, flush=True)
    print(f"  Training completed in {timing['training_s']:.1f}s ({timing['training_s'] / 60:.1f} min)", flush=True)

    # Speed/memory metrics
    num_samples_seen = len(train_dataset) * args.epochs
    samples_per_sec = num_samples_seen / timing["training_s"] if timing["training_s"] > 0 else None
    num_tokens_seen = getattr(trainer.state, "num_input_tokens_seen", None)
    tokens_per_sec = (
        num_tokens_seen / timing["training_s"]
        if num_tokens_seen is not None and timing["training_s"] > 0
        else None
    )
    gpu_report = get_gpu_report()

    # [5/5] Save
    print("\n[5/5] Saving model/report...", flush=True)
    t0 = time.time()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    timing["save_s"] = time.time() - t0

    generate_training_plots(metrics_cb, args.output_dir, args.method)
    generate_epoch_loss_plot(metrics_cb, args.output_dir, args.method)

    gsm8k_eval_summary = None
    if args.eval_after_train:
        eval_dir = args.eval_results_dir or os.path.join(args.output_dir, "gsm8k_8shot_eval")
        gsm8k_eval_summary = run_eval_gsm8k_after_train(
            checkpoint_dir=args.output_dir,
            eval_results_dir=eval_dir,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
        )

    report = {
        "experiment": f"GSM8K {args.method} finetuning",
        "model": args.model,
        "method": args.method,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": args.lr,
            "lr_scheduler": "cosine",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": eff_batch,
            "max_seq_length": args.max_length,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "precision": "bf16",
            "seed": args.seed,
        },
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules,
        } if is_lora_method else None,
        "spectrum_config": {
            "unfrozen_file": args.spectrum_unfrozen_file,
            "target_modules_filter": args.spectrum_target_modules,
            "num_selected_tensors": len(spectrum_selected_params) if spectrum_selected_params else None,
            "selected_tensors_preview": spectrum_selected_params[:50] if spectrum_selected_params else None,
        } if args.method == "spectrum" else None,
        "dropbp_config": {
            "drop_rate": args.dropbp_rate,
            "measure_time_memory": args.dropbp_measure_time_memory,
            "time_warmup_steps": args.dropbp_time_warmup_steps,
            "time_measure_steps": args.dropbp_time_measure_steps,
            "throughput_path": args.dropbp_throughput_path,
        } if is_dropbp_method else None,
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_pct": round(trainable_pct, 6),
            "num_trainable_tensors": len(trainable_names),
        },
        "training_results": {
            "trainer_train_runtime": getattr(train_result, "metrics", {}).get("train_runtime") if train_result else None,
            "total_logged_steps": len(metrics_cb.steps),
            "final_loss": round(metrics_cb.losses[-1], 4) if metrics_cb.losses else None,
            "best_loss": round(min(metrics_cb.losses), 4) if metrics_cb.losses else None,
            "epoch_losses": [round(l, 4) for l in metrics_cb.epoch_losses],
            "avg_gradient_norm": round(float(np.mean(metrics_cb.gradient_norms)), 4) if metrics_cb.gradient_norms else None,
        },
        "timing": {
            "model_load_s": round(timing["model_load_s"], 1),
            "data_prep_s": round(timing["data_prep_s"], 1),
            "training_s": round(timing["training_s"], 1),
            "training_min": round(timing["training_s"] / 60, 2),
            "save_s": round(timing["save_s"], 1),
            "avg_step_s": round(float(np.mean(metrics_cb.step_times)), 3) if metrics_cb.step_times else None,
            "samples_per_sec": round(samples_per_sec, 3) if samples_per_sec else None,
            "tokens_seen": int(num_tokens_seen) if num_tokens_seen is not None else None,
            "tokens_per_sec": round(tokens_per_sec, 3) if tokens_per_sec else None,
        },
        "gpu_memory": gpu_report,
        "metrics_log": {
            "steps": metrics_cb.steps,
            "losses": [round(l, 4) for l in metrics_cb.losses],
            "learning_rates": [round(lr, 8) for lr in metrics_cb.learning_rates],
            "gradient_norms": [round(g, 4) for g in metrics_cb.gradient_norms],
            "step_times": [round(s, 4) for s in metrics_cb.step_times],
        },
        "gsm8k_eval_8shot": gsm8k_eval_summary,
    }

    report_path = os.path.join(args.output_dir, f"training_report_{args.method}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}", flush=True)
    print("  TRAINING SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Method:            {args.method}", flush=True)
    print(f"  Trainable params:  {trainable_params:,} ({trainable_pct:.4f}%)", flush=True)
    if metrics_cb.losses:
        print(f"  Final loss:        {metrics_cb.losses[-1]:.4f}", flush=True)
        print(f"  Best loss:         {min(metrics_cb.losses):.4f}", flush=True)
    print(f"  Training time:     {timing['training_s'] / 60:.2f} min", flush=True)
    if samples_per_sec:
        print(f"  Samples/sec:       {samples_per_sec:.3f}", flush=True)
    if tokens_per_sec:
        print(f"  Tokens/sec:        {tokens_per_sec:.3f}", flush=True)
    if gpu_report:
        print(f"  Peak allocated:    {gpu_report.get('max_memory_allocated_gb')} GB", flush=True)
        print(f"  Peak reserved:     {gpu_report.get('max_memory_reserved_gb')} GB", flush=True)
    print(f"  Report:            {report_path}", flush=True)
    print(f"  Checkpoints:       {args.output_dir}/checkpoint-*", flush=True)
    if gsm8k_eval_summary and gsm8k_eval_summary.get("accuracy") is not None:
        print(
            f"  GSM8K test acc:    {gsm8k_eval_summary['accuracy'] * 100:.2f}%  "
            f"({gsm8k_eval_summary.get('correct')}/{gsm8k_eval_summary.get('total')})",
            flush=True,
        )
        if gsm8k_eval_summary.get("eval_time_s") is not None:
            print(f"  GSM8K eval time:   {gsm8k_eval_summary['eval_time_s']}s (inside eval script)", flush=True)
    elif not args.eval_after_train:
        print(f"\n  Next accuracy eval: python eval_gsm8k.py --model {args.output_dir}", flush=True)
    print(f"{'=' * 70}\n", flush=True)


if __name__ == "__main__":
    main()
