"""
GSM8K Selective Freezing Finetuning

Finetunes Llama 3.1 8B on GSM8K with most layers frozen.

Plan A: Q+K frozen everywhere. V+O+MLP trainable in first 10 layers (0-9).
Plan B: Q+K+MLP frozen everywhere. Only V+O trainable in first 10 layers (0-9).
Plan C: Q+K frozen everywhere. V+O+MLP trainable in last 10 layers (22-31).
Plan D: Q+K+MLP frozen everywhere. Only V+O trainable in last 10 layers (22-31).

Usage:
  python finetune_gsm8k_frozen.py --plan A --epochs 6
  python finetune_gsm8k_frozen.py --plan C --epochs 6
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ──────────────────────────────────────────────────────────────────────
#  Freezing
# ──────────────────────────────────────────────────────────────────────

def apply_freezing(model, plan, trainable_layers=None):
    """Freeze parameters according to the chosen plan.

    Strategy: freeze everything first, then selectively unfreeze.

    Plan A/C: Unfreeze V + O + MLP (gate/up/down) in trainable_layers
    Plan B/D: Unfreeze V + O only in trainable_layers
    Plan C/D: Also unfreeze lm_head
    """
    if trainable_layers is None:
        if plan in ("C", "D"):
            trainable_layers = list(range(22, 32))
        else:
            trainable_layers = list(range(10))
    trainable_layers = set(trainable_layers)

    train_head = plan in ("C", "D")

    if plan in ("A", "C"):
        trainable_projs = {".v_proj.", ".o_proj.", ".gate_proj.", ".up_proj.", ".down_proj."}
        desc = "V + O + MLP"
    else:
        trainable_projs = {".v_proj.", ".o_proj."}
        desc = "V + O"

    frozen_range = [i for i in range(32) if i not in trainable_layers]
    print(f"\n  Plan {plan}: {desc} trainable in layers {sorted(trainable_layers)}", flush=True)
    print(f"  Q + K frozen everywhere. Layers {frozen_range[0]}-{frozen_range[-1]} fully frozen.", flush=True)
    if train_head:
        print(f"  lm_head: trainable", flush=True)

    for p in model.parameters():
        p.requires_grad = False

    unfrozen_count = 0
    unfrozen_params = 0
    for name, p in model.named_parameters():
        # Unfreeze lm_head for plans C/D
        if train_head and "lm_head" in name:
            p.requires_grad = True
            unfrozen_count += 1
            unfrozen_params += p.numel()
            continue

        layer_idx = None
        for i in trainable_layers:
            if f".layers.{i}." in name:
                layer_idx = i
                break
        if layer_idx is None:
            continue

        if any(proj in name for proj in trainable_projs):
            p.requires_grad = True
            unfrozen_count += 1
            unfrozen_params += p.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Unfroze {unfrozen_count} tensors ({unfrozen_params:,} params, "
          f"{unfrozen_params/total_params*100:.2f}%)", flush=True)

    for i in sorted(trainable_layers):
        layer_trainable = []
        for name, p in model.named_parameters():
            if f".layers.{i}." in name and p.requires_grad:
                short = name.split(".")[-2]
                layer_trainable.append(short)
        if layer_trainable:
            print(f"    Layer {i:2d}: {', '.join(sorted(set(layer_trainable)))}", flush=True)

    # Show non-layer trainable params
    for name, p in model.named_parameters():
        if p.requires_grad and "layers." not in name:
            print(f"    {name} ({p.numel():,} params)", flush=True)

    return unfrozen_params, total_params


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────

class GSM8KDataset(torch.utils.data.Dataset):
    """GSM8K examples with loss masking (loss only on answer tokens)."""

    def __init__(self, data, tokenizer, max_length=512):
        self.samples = []
        for example in data:
            prompt = f"Question: {example['question']}\nAnswer: "
            completion = f"{example['answer']}{tokenizer.eos_token}"
            full_text = prompt + completion

            full_enc = tokenizer(full_text, max_length=max_length,
                                 truncation=True, return_tensors="pt")
            prompt_enc = tokenizer(prompt, max_length=max_length,
                                   truncation=True, return_tensors="pt")

            input_ids = full_enc['input_ids'].squeeze(0)
            attention_mask = full_enc['attention_mask'].squeeze(0)
            labels = input_ids.clone()
            labels[:prompt_enc['input_ids'].shape[1]] = -100

            self.samples.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(s['input_ids'].shape[0] for s in batch)
    pad = lambda t, val, n: torch.cat([t, torch.full((n,), val, dtype=t.dtype)])

    input_ids, attention_mask, labels = [], [], []
    for s in batch:
        n = max_len - s['input_ids'].shape[0]
        input_ids.append(pad(s['input_ids'], 0, n))
        attention_mask.append(pad(s['attention_mask'], 0, n))
        labels.append(pad(s['labels'], -100, n))

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
    }


# ──────────────────────────────────────────────────────────────────────
#  Callbacks
# ──────────────────────────────────────────────────────────────────────

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.steps, self.losses, self.learning_rates = [], [], []
        self.gradient_norms, self.step_times, self.epoch_losses = [], [], []
        self._cur_epoch_losses = []
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start:
            self.step_times.append(time.time() - self._step_start)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if 'loss' in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs['loss'])
            self._cur_epoch_losses.append(logs['loss'])
        if 'learning_rate' in logs:
            self.learning_rates.append(logs['learning_rate'])
        if 'grad_norm' in logs:
            self.gradient_norms.append(float(logs['grad_norm']))

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._cur_epoch_losses:
            self.epoch_losses.append(float(np.mean(self._cur_epoch_losses)))
            self._cur_epoch_losses = []


class ParamSnapshotCallback(TrainerCallback):
    """Saves trainable weight snapshots per epoch for later analysis."""
    def __init__(self, out_dir, keys=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")):
        self.out_dir = out_dir
        self.keys = keys

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        m = model.module if hasattr(model, "module") else model
        snap = {}
        for name, p in m.named_parameters():
            if any(k in name for k in self.keys):
                snap[name] = p.detach().float().cpu()
        epoch_str = f"{state.epoch:.2f}" if state.epoch is not None else "NA"
        path = os.path.join(self.out_dir, f"param_snapshot_epoch{epoch_str}_step{state.global_step}.pt")
        torch.save(snap, path)
        print(f"  Saved param snapshot: {path}", flush=True)


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def generate_training_plots(mcb, output_dir, label):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'GSM8K — {label}', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    if mcb.losses:
        ax.plot(mcb.steps, mcb.losses, 'b-', alpha=0.25, label='Step loss')
        w = max(1, len(mcb.losses) // 20)
        if w > 1:
            smooth = np.convolve(mcb.losses, np.ones(w) / w, mode='valid')
            ax.plot(mcb.steps[w - 1:], smooth, 'b-', lw=2, label=f'Smoothed (w={w})')
        ax.set_xlabel('Step'); ax.set_ylabel('Loss')
        ax.set_title('Training Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if mcb.learning_rates:
        ax.plot(mcb.steps[:len(mcb.learning_rates)], mcb.learning_rates, 'r-', lw=2)
        ax.set_xlabel('Step'); ax.set_ylabel('Learning Rate')
        ax.set_title('LR Schedule'); ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    ax = axes[1, 0]
    if mcb.gradient_norms:
        ax.plot(mcb.steps[:len(mcb.gradient_norms)], mcb.gradient_norms, 'g-', alpha=0.5)
        avg_gn = np.mean(mcb.gradient_norms)
        ax.axhline(y=avg_gn, color='g', ls='--', lw=2, label=f'Mean: {avg_gn:.2f}')
        ax.set_xlabel('Step'); ax.set_ylabel('Grad Norm')
        ax.set_title('Gradient Norms'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if mcb.step_times:
        ax.plot(mcb.step_times, 'm-', alpha=0.25)
        avg_t = np.mean(mcb.step_times)
        ax.axhline(y=avg_t, color='m', ls='--', lw=2, label=f'Avg: {avg_t:.2f}s')
        ax.set_xlabel('Step'); ax.set_ylabel('Time (s)')
        ax.set_title('Step Duration'); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'training_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}", flush=True)


def generate_epoch_loss_plot(mcb, output_dir, label):
    if not mcb.epoch_losses:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(mcb.epoch_losses) + 1))
    x = np.arange(len(epochs))
    bars = ax.bar(x, mcb.epoch_losses, color='steelblue', edgecolor='black')
    for bar, loss in zip(bars, mcb.epoch_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Average Loss')
    ax.set_title(f'Loss per Epoch — {label}')
    ax.set_xticks(x); ax.set_xticklabels(epochs)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'epoch_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Epoch loss plot saved: {path}", flush=True)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM8K Frozen Finetuning")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--plan", type=str, choices=["A", "B", "C", "D"], required=True,
                        help="A: V+O+MLP layers 0-9 | B: V+O layers 0-9 | C: V+O+MLP layers 22-31+head | D: V+O layers 22-31+head")
    parser.add_argument("--trainable-layers", type=int, nargs="+", default=None,
                        help="Which layers to unfreeze (default: 0-9)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to custom JSON dataset (e.g. perturbed). Uses HF gsm8k if not set.")
    args = parser.parse_args()

    if args.trainable_layers is None:
        if args.plan in ("C", "D"):
            args.trainable_layers = list(range(22, 32))
        else:
            args.trainable_layers = list(range(10))
    if args.output_dir is None:
        args.output_dir = f"gsm8k-frozen-plan{args.plan}"

    timing = {}
    label = f"Plan {args.plan} (layers {min(args.trainable_layers)}-{max(args.trainable_layers)})"

    # Banner
    print(f"\n{'='*60}", flush=True)
    print(f"  GSM8K FROZEN FINETUNING — PLAN {args.plan}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:           {args.model}", flush=True)
    print(f"  Plan:            {args.plan}", flush=True)
    print(f"  Trainable layers:{args.trainable_layers}", flush=True)
    print(f"  LR:              {args.lr} (cosine)", flush=True)
    print(f"  Epochs:          {args.epochs}", flush=True)
    print(f"  Batch size:      {args.batch_size} x {args.grad_accum} "
          f"(eff {args.batch_size * args.grad_accum})", flush=True)
    print(f"  Output:          {args.output_dir}", flush=True)
    if args.dataset_path:
        print(f"  Dataset:         {args.dataset_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {gpu_mem:.1f} GB\n", flush=True)

    # [1] Load model
    print("[1/5] Loading model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    timing['model_load'] = time.time() - t0
    print(f"  Loaded in {timing['model_load']:.1f}s", flush=True)

    # [2] Apply freezing
    print("\n[2/5] Applying freezing...", flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable_params, total_params = apply_freezing(model, args.plan, args.trainable_layers)

    # [3] Dataset
    print("\n[3/5] Preparing dataset...", flush=True)
    t0 = time.time()
    if args.dataset_path:
        import json as _json
        with open(args.dataset_path) as _f:
            raw_data = _json.load(_f)
        print(f"  Custom dataset: {args.dataset_path}", flush=True)
    else:
        dataset = load_dataset("openai/gsm8k", "main")
        raw_data = list(dataset['train'])
    train_dataset = GSM8KDataset(raw_data, tokenizer, max_length=args.max_length)
    timing['data_prep'] = time.time() - t0
    print(f"  {len(train_dataset)} examples ({timing['data_prep']:.1f}s)", flush=True)

    # [4] Train
    print("\n[4/5] Configuring trainer...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // eff_batch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.03))

    print(f"  Steps/epoch: {steps_per_epoch}  |  Total: {total_steps}  |  Warmup: {warmup_steps}", flush=True)

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
    )

    metrics_cb = MetricsCallback()
    param_cb = ParamSnapshotCallback(args.output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[metrics_cb, param_cb],
    )

    print(f"\n  Training...", flush=True)
    print(f"{'─'*60}", flush=True)
    t0 = time.time()
    trainer.train()
    timing['training'] = time.time() - t0
    print(f"{'─'*60}", flush=True)
    print(f"  Done in {timing['training']/60:.1f} min", flush=True)

    # [5] Save
    print("\n[5/5] Saving...", flush=True)
    t0 = time.time()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    timing['save'] = time.time() - t0

    generate_training_plots(metrics_cb, args.output_dir, label)
    generate_epoch_loss_plot(metrics_cb, args.output_dir, label)

    report = {
        "experiment": f"GSM8K Frozen Plan {args.plan}",
        "model": args.model,
        "plan": args.plan,
        "trainable_layers": args.trainable_layers,
        "trainable_projections": ("V+O+MLP" if args.plan in ("A", "C") else "V+O") + ("+head" if args.plan in ("C", "D") else ""),
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": args.lr, "lr_scheduler": "cosine",
            "epochs": args.epochs, "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": eff_batch,
            "warmup_steps": warmup_steps,
        },
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_pct": round(trainable_params / total_params * 100, 4),
        },
        "training_results": {
            "total_steps": len(metrics_cb.steps),
            "final_loss": round(metrics_cb.losses[-1], 4) if metrics_cb.losses else None,
            "best_loss": round(min(metrics_cb.losses), 4) if metrics_cb.losses else None,
            "epoch_losses": [round(l, 4) for l in metrics_cb.epoch_losses],
            "avg_gradient_norm": round(float(np.mean(metrics_cb.gradient_norms)), 4)
                if metrics_cb.gradient_norms else None,
        },
        "timing": {
            "training_min": round(timing['training'] / 60, 1),
            "avg_step_s": round(float(np.mean(metrics_cb.step_times)), 3)
                if metrics_cb.step_times else None,
        },
        "metrics_log": {
            "steps": metrics_cb.steps,
            "losses": [round(l, 4) for l in metrics_cb.losses],
            "learning_rates": [round(lr, 8) for lr in metrics_cb.learning_rates],
            "gradient_norms": [round(g, 4) for g in metrics_cb.gradient_norms],
        },
    }

    report_path = os.path.join(args.output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"  TRAINING SUMMARY — PLAN {args.plan}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Trainable:  {trainable_params:,} / {total_params:,} "
          f"({trainable_params/total_params*100:.2f}%)", flush=True)
    if metrics_cb.epoch_losses:
        for i, el in enumerate(metrics_cb.epoch_losses):
            print(f"    Epoch {i+1}: {el:.4f}", flush=True)
    print(f"  Time: {timing['training']/60:.1f} min", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n  Next: python eval_gsm8k.py --model {args.output_dir}\n", flush=True)


if __name__ == "__main__":
    main()
