"""
GSM8K Finetuning — LoRA & Full Finetuning

Finetunes Llama 3.1 8B on GSM8K math questions.
Saves per-epoch checkpoints and training metrics (loss, lr, grad norms, timing)
so you can analyze weight changes in a separate script later.

Usage:
  python finetune_gsm8k.py --method lora --batch-size 16 --grad-accum 1
  python finetune_gsm8k.py --method full --batch-size 16 --grad-accum 1
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
#  Dataset
# ──────────────────────────────────────────────────────────────────────

class GSM8KDataset(torch.utils.data.Dataset):
    """Prepares GSM8K examples for causal language model finetuning.

    Each example becomes: "Question: {q}\nAnswer: {a}<eos>"

    KEY CONCEPT — LOSS MASKING:
      We only compute loss on the ANSWER tokens. The question/prompt tokens
      are masked with label=-100 (ignored by CrossEntropyLoss). This teaches
      the model to GENERATE good answers, not to memorize questions.

      Example:
        Input:  "Question: What is 2+3?\nAnswer: 2+3=5. The answer is 5.<eos>"
        Labels: [-100, -100, ..., -100,  "2", "+", "3", "=", "5", ...]
                 ^^^^ question masked ^^^^  ^^^^ answer: loss computed ^^^^
    """

    def __init__(self, data, tokenizer, max_length=512):
        self.samples = []
        for example in data:
            prompt = f"Question: {example['question']}\nAnswer: "
            completion = f"{example['answer']}{tokenizer.eos_token}"
            full_text = prompt + completion

            full_enc = tokenizer(
                full_text, max_length=max_length,
                truncation=True, return_tensors="pt")
            prompt_enc = tokenizer(
                prompt, max_length=max_length,
                truncation=True, return_tensors="pt")

            input_ids = full_enc['input_ids'].squeeze(0)
            attention_mask = full_enc['attention_mask'].squeeze(0)

            # Mask prompt tokens so loss ignores them
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
    """Pads variable-length sequences to the same length within a batch.

    Right-pads with 0 (input_ids, attention_mask) and -100 (labels).
    """
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
#  Metrics Callback — Records training dynamics per step
# ──────────────────────────────────────────────────────────────────────

class ParamSnapshotCallback(TrainerCallback):
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
                snap[name] = p.detach().float().cpu()  # float for stable diffing

        epoch_str = f"{state.epoch:.2f}" if state.epoch is not None else "NA"
        path = os.path.join(self.out_dir, f"param_snapshot_epoch{epoch_str}_step{state.global_step}.pt")
        torch.save(snap, path)
        print(f"  Saved param snapshot: {path}", flush=True)



class MetricsCallback(TrainerCallback):
    """Records key training metrics at each logging step.

    What each metric means:
      - loss:          How wrong the model's predictions are (lower = better)
      - learning_rate: Current step size for weight updates (cosine schedule)
      - grad_norm:     Magnitude of gradients (how much weights want to change)
      - step_time:     Wall-clock seconds per training step
      - epoch_losses:  Average loss per epoch (for convergence tracking)
    """

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
        if logs is None:
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


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def generate_training_plots(mcb, output_dir, method):
    """4-panel plot of core training metrics.

    Panel 1 — Training Loss:  Prediction error over time.
    Panel 2 — Learning Rate:  Cosine schedule (warmup -> peak -> decay).
    Panel 3 — Gradient Norms: Magnitude of updates (stability indicator).
    Panel 4 — Step Duration:  Compute time per step.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'GSM8K Finetuning — {method.upper()}', fontsize=16, fontweight='bold')

    # Training loss
    ax = axes[0, 0]
    if mcb.losses:
        ax.plot(mcb.steps, mcb.losses, 'b-', alpha=0.25, label='Step loss')
        w = max(1, len(mcb.losses) // 20)
        if w > 1:
            smooth = np.convolve(mcb.losses, np.ones(w) / w, mode='valid')
            ax.plot(mcb.steps[w - 1:], smooth, 'b-', lw=2, label=f'Smoothed (w={w})')
        ax.set_xlabel('Step');  ax.set_ylabel('Loss')
        ax.set_title('Training Loss');  ax.legend();  ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[0, 1]
    if mcb.learning_rates:
        lr_steps = mcb.steps[:len(mcb.learning_rates)]
        ax.plot(lr_steps, mcb.learning_rates, 'r-', lw=2)
        ax.set_xlabel('Step');  ax.set_ylabel('Learning Rate')
        ax.set_title('LR Schedule (cosine)')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Gradient norms
    ax = axes[1, 0]
    if mcb.gradient_norms:
        gn_steps = mcb.steps[:len(mcb.gradient_norms)]
        ax.plot(gn_steps, mcb.gradient_norms, 'g-', alpha=0.5, label='Grad norm')
        avg_gn = np.mean(mcb.gradient_norms)
        ax.axhline(y=avg_gn, color='g', ls='--', lw=2, label=f'Mean: {avg_gn:.2f}')
        ax.set_xlabel('Step');  ax.set_ylabel('Gradient L2 Norm')
        ax.set_title('Gradient Norms');  ax.legend();  ax.grid(True, alpha=0.3)

    # Step time
    ax = axes[1, 1]
    if mcb.step_times:
        ax.plot(mcb.step_times, 'm-', alpha=0.25)
        avg_t = np.mean(mcb.step_times)
        ax.axhline(y=avg_t, color='m', ls='--', lw=2, label=f'Avg: {avg_t:.2f}s')
        ax.set_xlabel('Step');  ax.set_ylabel('Time (s)')
        ax.set_title('Step Duration');  ax.legend();  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'training_metrics_{method}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}", flush=True)
    return path


def generate_epoch_loss_plot(mcb, output_dir, method):
    """Bar chart of average training loss per epoch."""
    if not mcb.epoch_losses:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(mcb.epoch_losses) + 1))
    x = np.arange(len(epochs))

    bars = ax.bar(x, mcb.epoch_losses, color='steelblue', edgecolor='black')
    for bar, loss in zip(bars, mcb.epoch_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xlabel('Epoch');  ax.set_ylabel('Average Loss')
    ax.set_title(f'Loss per Epoch — {method.upper()}')
    ax.set_xticks(x);  ax.set_xticklabels(epochs)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'epoch_loss_{method}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Epoch loss plot saved: {path}", flush=True)
    return path


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM8K Finetuning (LoRA / Full)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B") # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--method", type=str, choices=["lora", "full"], default="lora")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 2e-4 for LoRA, 2e-5 for full)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=5)
    args = parser.parse_args()

    # Defaults
    if args.lr is None:
        args.lr = 2e-4 if args.method == "lora" else 2e-5
    if args.output_dir is None:
        args.output_dir = f"gsm8k-{args.method}-finetuned"

    timing = {}

    # Banner
    print(f"\n{'='*60}", flush=True)
    print(f"  GSM8K FINETUNING", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Method:          {args.method.upper()}", flush=True)
    print(f"  Model:           {args.model}", flush=True)
    print(f"  Learning rate:   {args.lr}", flush=True)
    print(f"  LR scheduler:    cosine", flush=True)
    print(f"  Epochs:          {args.epochs}", flush=True)
    print(f"  Batch size:      {args.batch_size} x {args.grad_accum} "
          f"(effective {args.batch_size * args.grad_accum})", flush=True)
    print(f"  Max seq length:  {args.max_length}", flush=True)
    print(f"  Max grad norm:   1.0", flush=True)
    print(f"  Weight decay:    0.01", flush=True)
    print(f"  Precision:       bf16", flush=True)
    if args.method == "lora":
        print(f"  LoRA r:          {args.lora_r}", flush=True)
        print(f"  LoRA alpha:      {args.lora_alpha}", flush=True)
        print(f"  LoRA dropout:    0.05", flush=True)
        print(f"  LoRA targets:    q_proj, k_proj, v_proj, o_proj", flush=True)
    print(f"  Output dir:      {args.output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}  |  VRAM: {gpu_mem:.1f} GB\n", flush=True)

    # ── [1/5] Load model ──
    print("[1/5] Loading model...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    timing['model_load'] = time.time() - t0
    print(f"  Loaded in {timing['model_load']:.1f}s", flush=True)

    # ── [2/5] Apply LoRA or setup Full FT ──
    is_peft = args.method == "lora"
    if is_peft:
        from peft import LoraConfig, get_peft_model, TaskType
        print("\n[2/5] Applying LoRA...", flush=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("\n[2/5] Setting up full finetuning...", flush=True)
        # Gradient checkpointing: trades compute for memory (recomputes
        # activations during backward instead of storing them all)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}", flush=True)
    print(f"  Trainable params: {trainable_params:,} "
          f"({trainable_params / total_params * 100:.2f}%)", flush=True)

    # ── [3/5] Prepare dataset ──
    print("\n[3/5] Preparing dataset...", flush=True)
    t0 = time.time()

    dataset = load_dataset("openai/gsm8k", "main")
    train_data = dataset['train']
    train_dataset = GSM8KDataset(train_data, tokenizer, max_length=args.max_length)

    timing['data_prep'] = time.time() - t0
    print(f"  {len(train_dataset)} examples ({timing['data_prep']:.1f}s)", flush=True)

    # Token length stats
    lengths = [s['input_ids'].shape[0] for s in train_dataset.samples]
    print(f"  Token lengths — min: {min(lengths)}, max: {max(lengths)}, "
          f"mean: {np.mean(lengths):.0f}, median: {np.median(lengths):.0f}", flush=True)

    # ── [4/5] Train ──
    print("\n[4/5] Configuring trainer...", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_dataset) // eff_batch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.03))

    print(f"  Steps/epoch: {steps_per_epoch}", flush=True)
    print(f"  Total steps: {total_steps}", flush=True)
    print(f"  Warmup steps: {warmup_steps}", flush=True)

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
        save_strategy="epoch",       # save checkpoint at end of each epoch
        save_total_limit=None,       # keep all epoch checkpoints
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
    print(f"  Training completed in {timing['training']:.1f}s "
          f"({timing['training'] / 60:.1f} min)", flush=True)

    # ── [5/5] Save final model & report ──
    print("\n[5/5] Saving model & report...", flush=True)
    t0 = time.time()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    timing['save'] = time.time() - t0
    print(f"  Model saved to {args.output_dir}", flush=True)

    # Generate plots
    generate_training_plots(metrics_cb, args.output_dir, args.method)
    generate_epoch_loss_plot(metrics_cb, args.output_dir, args.method)

    # Save training report (metrics JSON for later analysis)
    report = {
        "experiment": f"GSM8K {args.method.upper()} Finetuning",
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
        },
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        } if is_peft else None,
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
            "model_load_s": round(timing['model_load'], 1),
            "data_prep_s": round(timing['data_prep'], 1),
            "training_s": round(timing['training'], 1),
            "training_min": round(timing['training'] / 60, 1),
            "save_s": round(timing['save'], 1),
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

    report_path = os.path.join(args.output_dir, f"training_report_{args.method}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"  TRAINING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Method:            {args.method.upper()}", flush=True)
    print(f"  Model:             {args.model}", flush=True)
    print(f"  Epochs:            {args.epochs}", flush=True)
    print(f"  LR:                {args.lr}", flush=True)
    print(f"  Effective batch:   {eff_batch}", flush=True)
    print(f"  Trainable params:  {trainable_params:,} "
          f"({trainable_params/total_params*100:.2f}%)", flush=True)
    if metrics_cb.losses:
        print(f"  Final loss:        {metrics_cb.losses[-1]:.4f}", flush=True)
        print(f"  Best loss:         {min(metrics_cb.losses):.4f}", flush=True)
    if metrics_cb.epoch_losses:
        for i, el in enumerate(metrics_cb.epoch_losses):
            print(f"    Epoch {i+1} avg:    {el:.4f}", flush=True)
    if metrics_cb.gradient_norms:
        print(f"  Avg grad norm:     {np.mean(metrics_cb.gradient_norms):.4f}", flush=True)
    print(f"  Training time:     {timing['training']/60:.1f} min", flush=True)
    print(f"  Report:            {report_path}", flush=True)
    print(f"  Checkpoints:       {args.output_dir}/checkpoint-*", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n  Next: python eval_gsm8k.py --model {args.output_dir}\n", flush=True)


if __name__ == "__main__":
    main()
