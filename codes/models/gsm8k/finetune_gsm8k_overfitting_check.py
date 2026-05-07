"""
GSM8K Overfitting Check — 70% Finetune / 15% Validation / 15% Test

Finetunes Llama 3.1 8B on GSM8K with selective layer freezing (V+O).
Uses 70/15/15 split to detect overfitting:
  - Per epoch: train loss, validation loss (optionally train/val accuracy)
  - At end: test accuracy

Split: GSM8K train → 82.4% finetune + 17.6% validation; test → 15% held-out.

Usage:
  python finetune_gsm8k_overfitting_check.py \
      --v-frozen-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
      --o-frozen-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 \
      --output-dir overfitting-wass-high6-run1
"""

import argparse
import json
import os
import re
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
#  Model depth (Llama / Qwen / Mistral)
# ──────────────────────────────────────────────────────────────────────

def _num_transformer_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        return len(layers)
    n = getattr(model.config, "num_hidden_layers", None)
    if n is not None:
        return int(n)
    raise ValueError(
        "Cannot determine transformer depth: no model.model.layers and "
        "config.num_hidden_layers is missing."
    )


# ──────────────────────────────────────────────────────────────────────
#  Freezing (selective V/O)
# ──────────────────────────────────────────────────────────────────────

def apply_selective_vo_freezing(model, v_frozen_layers, o_frozen_layers):
    """K+Q+MLP frozen everywhere; V and O trainable except in specified layers."""
    v_frozen = set(v_frozen_layers)
    o_frozen = set(o_frozen_layers)
    n_layers = _num_transformer_layers(model)

    for p in model.parameters():
        p.requires_grad = False

    unfrozen_count = 0
    unfrozen_params = 0
    for name, p in model.named_parameters():
        if "layers." not in name:
            continue
        try:
            layer_idx = int(name.split(".layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if ".v_proj." in name and layer_idx not in v_frozen:
            p.requires_grad = True
            unfrozen_count += 1
            unfrozen_params += p.numel()
        elif ".o_proj." in name and layer_idx not in o_frozen:
            p.requires_grad = True
            unfrozen_count += 1
            unfrozen_params += p.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Selective V/O freezing:", flush=True)
    print(f"  K + Q + MLP: frozen in all {n_layers} layers", flush=True)
    print(f"  V frozen layers: {sorted(v_frozen)}  ({len(v_frozen)} frozen, {n_layers - len(v_frozen)} trainable)", flush=True)
    print(f"  O frozen layers: {sorted(o_frozen)}  ({len(o_frozen)} frozen, {n_layers - len(o_frozen)} trainable)", flush=True)
    print(f"  Trainable: {unfrozen_params:,} params ({unfrozen_params/total_params*100:.2f}%)", flush=True)
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
#  Accuracy evaluation (8-shot CoT — matches eval_gsm8k.py for comparable results)
# ──────────────────────────────────────────────────────────────────────

# Official 8-shot examples (lm-eval-harness gsm8k_cot) — same as eval_gsm8k.py
FEW_SHOT_EXAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]


def build_prompt_8shot(question):
    """8-shot CoT format — matches eval_gsm8k.py for comparable ~60% baseline."""
    prompt = ""
    for q, a in FEW_SHOT_EXAMPLES:
        prompt += f"Q: {q}\nA: {a}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(text):
    match = re.findall(r'[Tt]he answer is\s*\$?\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match[-1].replace(',', '').strip()
    numbers = re.findall(r'(-?\d[\d,]*\.?\d*)', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return None


def extract_gold(answer_text):
    if '####' in answer_text:
        return answer_text.split('####')[-1].strip().replace(',', '')
    return None


def normalize(s):
    if s is None:
        return None
    return s.strip().replace(',', '').replace('$', '').rstrip('.')


def compute_accuracy(model, tokenizer, data, batch_size=32, max_new_tokens=512):
    """Exact-match accuracy using 8-shot CoT (matches eval_gsm8k.py for ~60% baseline)."""
    model.eval()
    # Left-padding for batched generation (same as eval_gsm8k.py)
    _prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    correct, total = 0, 0
    all_prompts = [build_prompt_8shot(ex['question']) for ex in data]
    all_examples = list(data)
    n_batches = (len(all_prompts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for bi in range(n_batches):
            bs, be = bi * batch_size, min((bi + 1) * batch_size, len(all_prompts))
            batch_prompts = all_prompts[bs:be]
            batch_examples = all_examples[bs:be]

            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
            ).to(model.device)
            input_len = inputs['input_ids'].shape[1]

            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j in range(len(batch_prompts)):
                gen_ids = out[j][input_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                for stop in ["\nQ:", "Q:"]:
                    si = gen_text.find(stop)
                    if si != -1:
                        gen_text = gen_text[:si]
                        break
                gold = normalize(extract_gold(batch_examples[j]['answer']))
                pred = normalize(extract_answer(gen_text))
                if gold is not None and pred is not None and gold == pred:
                    correct += 1
                total += 1

    model.train()
    return correct / total if total > 0 else 0.0, correct, total


# ──────────────────────────────────────────────────────────────────────
#  Callbacks
# ──────────────────────────────────────────────────────────────────────

class OverfittingCallback(TrainerCallback):
    """Tracks train/val loss per epoch, optionally train/val accuracy."""

    def __init__(self, compute_accuracy_each_epoch=False, train_data=None, val_data=None,
                 tokenizer=None, model_ref=None, eval_batch_size=32, eval_max_new_tokens=512):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self._cur_epoch_losses = []
        self.compute_accuracy_each_epoch = compute_accuracy_each_epoch
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        self.model_ref = model_ref
        self.eval_batch_size = eval_batch_size
        self.eval_max_new_tokens = eval_max_new_tokens

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self._cur_epoch_losses.append(logs['loss'])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.val_losses.append(float(metrics['eval_loss']))

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._cur_epoch_losses:
            self.train_losses.append(float(np.mean(self._cur_epoch_losses)))
            self._cur_epoch_losses = []

        if self.compute_accuracy_each_epoch and self.model_ref is not None:
            # Sample 300 from train for speed; full val
            train_sample = self.train_data[:300] if len(self.train_data) > 300 else self.train_data
            train_acc, _, _ = compute_accuracy(
                self.model_ref, self.tokenizer, train_sample,
                batch_size=self.eval_batch_size,
                max_new_tokens=self.eval_max_new_tokens,
            )
            val_acc, _, _ = compute_accuracy(
                self.model_ref, self.tokenizer, self.val_data,
                batch_size=self.eval_batch_size,
                max_new_tokens=self.eval_max_new_tokens,
            )
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            print(f"  [Epoch {len(self.train_losses)}] Train acc: {train_acc*100:.1f}%  |  Val acc: {val_acc*100:.1f}%", flush=True)


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def _backfill_epoch_losses_from_log_history(trainer, cb: OverfittingCallback, n_epochs: int) -> None:
    """Restore per-epoch train/val losses from Trainer.state.log_history (e.g. after resume with empty callbacks)."""
    hist = getattr(trainer.state, "log_history", None) or []
    if not hist or n_epochs < 1:
        return

    if len(cb.val_losses) < n_epochs:
        by_ep: dict[int, float] = {}
        for entry in hist:
            if "eval_loss" not in entry or "epoch" not in entry:
                continue
            ep = float(entry["epoch"])
            if abs(ep - round(ep)) > 1e-3:
                continue
            ei = int(round(ep))
            if 1 <= ei <= n_epochs:
                by_ep[ei] = float(entry["eval_loss"])
        ordered_val = [by_ep[i] for i in range(1, n_epochs + 1) if i in by_ep]
        if len(ordered_val) >= len(cb.val_losses):
            cb.val_losses = ordered_val[:n_epochs]

    if len(cb.train_losses) < n_epochs:
        trains: list[float] = []
        for k in range(1, n_epochs + 1):
            losses_k = [
                float(e["loss"])
                for e in hist
                if "loss" in e
                and "eval_loss" not in e
                and "epoch" in e
                and (k - 1) < float(e["epoch"]) <= k + 1e-9
            ]
            if losses_k:
                trains.append(float(np.mean(losses_k)))
        if len(trains) >= len(cb.train_losses):
            cb.train_losses = trains[:n_epochs]


def generate_overfitting_plots(cb, output_dir, label):
    """Train/val loss vs epoch; optionally accuracy vs epoch."""
    n_epochs = len(cb.train_losses)
    if n_epochs == 0:
        return

    fig, axes = plt.subplots(1, 2 if cb.train_accs else 1, figsize=(10 if cb.train_accs else 6, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    epochs = list(range(1, n_epochs + 1))
    ax = axes[0]
    ax.plot(epochs, cb.train_losses, 'b-o', label='Train loss', markersize=6)
    if cb.val_losses:
        ax.plot(epochs, cb.val_losses, 'r-s', label='Val loss', markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Train vs Val Loss — {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if cb.train_accs and len(axes) > 1:
        ax = axes[1]
        ax.plot(epochs, cb.train_accs, 'b-o', label='Train acc', markersize=6)
        ax.plot(epochs, cb.val_accs, 'r-s', label='Val acc', markersize=6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Train vs Val Accuracy — {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'overfitting_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {path}", flush=True)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM8K Overfitting Check")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--method",
        type=str,
        choices=["vo_freeze", "full", "lora"],
        default="vo_freeze",
        help="vo_freeze: selective V/O (requires --v-frozen-layers / --o-frozen-layers). "
        "full: full finetune. lora: LoRA on q,k,v,o.",
    )
    parser.add_argument(
        "--v-frozen-layers",
        type=str,
        default="",
        help="Comma-separated layer indices to freeze V (vo_freeze only).",
    )
    parser.add_argument(
        "--o-frozen-layers",
        type=str,
        default="",
        help="Comma-separated layer indices to freeze O (vo_freeze only).",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-4 lora; full 5e-6 Mistral else 2e-5; vo_freeze 2e-5).",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--accuracy-each-epoch", action="store_true",
                        help="Compute train/val accuracy each epoch (slower; loss is the main overfitting signal)")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                        help="Batch size for accuracy evaluation")
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens per example during 8-shot GSM8K accuracy (lower = faster, risk truncated answers).",
    )
    parser.add_argument(
        "--skip-final-accuracy",
        action="store_true",
        help="Skip slow 8-shot GSM8K generation on train/val/test after training; save model + loss-only report.",
    )
    parser.add_argument(
        "--resume-from-last-checkpoint",
        action="store_true",
        help="Resume Trainer from latest checkpoint under --output-dir (e.g. after interrupt during final accuracy).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.method == "vo_freeze":
        if not (args.v_frozen_layers.strip() and args.o_frozen_layers.strip()):
            raise SystemExit(
                "vo_freeze requires non-empty --v-frozen-layers and --o-frozen-layers"
            )
        v_frozen = [int(x.strip()) for x in args.v_frozen_layers.split(",") if x.strip()]
        o_frozen = [int(x.strip()) for x in args.o_frozen_layers.split(",") if x.strip()]
    else:
        v_frozen = []
        o_frozen = []

    if args.lr is None:
        if args.method == "lora":
            args.lr = 2e-4
        elif args.method == "full" and "mistral" in args.model.lower():
            args.lr = 5e-6
        else:
            args.lr = 2e-5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 70/15/15 split ──────────────────────────────────────────────
    # We keep the official GSM8K test split held out, then split the official
    # train split into finetune/val using the corresponding train-space ratio.
    split_finetune = 0.70
    split_val = 0.15
    split_test = 0.15
    print("\n" + "=" * 60, flush=True)
    print("  GSM8K OVERFITTING CHECK — 70/15/15 split", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model:       {args.model}", flush=True)
    print(f"  Method:      {args.method}", flush=True)
    if args.method == "vo_freeze":
        print(f"  V frozen:    {len(v_frozen)} layers", flush=True)
        print(f"  O frozen:    {len(o_frozen)} layers", flush=True)
    print(f"  LR:          {args.lr}", flush=True)
    print(f"  Epochs:      {args.epochs}", flush=True)
    print(f"  Output:      {args.output_dir}", flush=True)
    print("=" * 60 + "\n", flush=True)

    dataset = load_dataset("openai/gsm8k", "main")
    train_raw = list(dataset['train'])
    test_raw = list(dataset['test'])

    n_train = len(train_raw)
    # Convert global 70/15/15 to train-space finetune ratio (70 / (70 + 15)),
    # then round to 3 decimals for stable, deterministic integer slicing.
    finetune_train_ratio = round(split_finetune / (split_finetune + split_val), 3)
    n_finetune = int(finetune_train_ratio * n_train)
    n_val = n_train - n_finetune
    n_test = len(test_raw)

    finetune_data = train_raw[:n_finetune]
    val_data = train_raw[n_finetune:]
    assert len(finetune_data) == n_finetune and len(val_data) == n_val

    print(
        f"  Split target: {split_finetune:.2f}/{split_val:.2f}/{split_test:.2f} "
        "(finetune/val/test)",
        flush=True,
    )
    print(f"  Split: finetune={n_finetune}, val={n_val}, test={n_test}", flush=True)
    print(f"  Ratios: {n_finetune/(n_finetune+n_val+n_test)*100:.1f}% / "
          f"{n_val/(n_finetune+n_val+n_test)*100:.1f}% / "
          f"{n_test/(n_finetune+n_val+n_test)*100:.1f}% (finetune/val/test)\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.enable_input_require_grads()

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        print("\n  Applying LoRA (q_proj, k_proj, v_proj, o_proj)...", flush=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
    else:
        model.gradient_checkpointing_enable()
        if args.method == "vo_freeze":
            trainable_params, total_params = apply_selective_vo_freezing(model, v_frozen, o_frozen)
        else:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n  Full finetuning: all {trainable_params:,} trainable params "
                  f"({100 * trainable_params / total_params:.2f}% of {total_params:,})", flush=True)

    train_dataset = GSM8KDataset(finetune_data, tokenizer, max_length=args.max_length)
    val_dataset = GSM8KDataset(val_data, tokenizer, max_length=args.max_length)

    eff_batch = args.batch_size * args.grad_accum
    total_steps = (len(train_dataset) // eff_batch) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.03))

    overfitting_cb = OverfittingCallback(
        compute_accuracy_each_epoch=args.accuracy_each_epoch,
        train_data=finetune_data,
        val_data=val_data,
        tokenizer=tokenizer,
        model_ref=model,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
    )

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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[overfitting_cb],
    )

    print("  Training...", flush=True)
    t0 = time.time()
    train_kw = {}
    if args.resume_from_last_checkpoint:
        train_kw["resume_from_checkpoint"] = True
    trainer.train(**train_kw)
    train_time = time.time() - t0
    print(f"  Done in {train_time/60:.1f} min\n", flush=True)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    _backfill_epoch_losses_from_log_history(trainer, overfitting_cb, args.epochs)

    if args.skip_final_accuracy:
        print("  Skipping final train/val/test accuracy (--skip-final-accuracy).", flush=True)
        train_acc = val_acc = test_acc = None
        train_correct = val_correct = test_correct = None
        train_total = n_finetune
        val_total = n_val
        test_total = n_test
    else:
        emax = args.eval_max_new_tokens
        print("  Computing final train / val / test accuracy...", flush=True)
        train_acc, train_correct, train_total = compute_accuracy(
            model, tokenizer, finetune_data,
            batch_size=args.eval_batch_size, max_new_tokens=emax)
        val_acc, val_correct, val_total = compute_accuracy(
            model, tokenizer, val_data, batch_size=args.eval_batch_size, max_new_tokens=emax)
        test_acc, test_correct, test_total = compute_accuracy(
            model, tokenizer, test_raw, batch_size=args.eval_batch_size, max_new_tokens=emax)

    report = {
        "experiment": "GSM8K Overfitting Check",
        "model": args.model,
        "method": args.method,
        "output_dir": args.output_dir,
        "v_frozen_layers": v_frozen if args.method == "vo_freeze" else None,
        "o_frozen_layers": o_frozen if args.method == "vo_freeze" else None,
        "split": {"finetune": n_finetune, "val": n_val, "test": n_test},
        "per_epoch": {
            "train_loss": overfitting_cb.train_losses,
            "val_loss": overfitting_cb.val_losses,
        },
        "final": (
            {
                "train_accuracy": None,
                "train_correct": None,
                "train_total": train_total,
                "val_accuracy": None,
                "val_correct": None,
                "val_total": val_total,
                "test_accuracy": None,
                "test_correct": None,
                "test_total": test_total,
            }
            if args.skip_final_accuracy
            else {
                "train_accuracy": round(train_acc, 4),
                "train_correct": train_correct,
                "train_total": train_total,
                "val_accuracy": round(val_acc, 4),
                "val_correct": val_correct,
                "val_total": val_total,
                "test_accuracy": round(test_acc, 4),
                "test_correct": test_correct,
                "test_total": test_total,
            }
        ),
        "overfitting_gap": (
            None
            if args.skip_final_accuracy
            else {
                "train_minus_val": round(train_acc - val_acc, 4),
                "train_minus_test": round(train_acc - test_acc, 4),
            }
        ),
        **({"accuracy_eval_skipped": True} if args.skip_final_accuracy else {}),
        "hyperparameters": {
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "eval_batch_size": args.eval_batch_size,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "trainable_params": trainable_params,
            "total_params": total_params,
            **(
                {
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": 0.05,
                    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                }
                if args.method == "lora"
                else {}
            ),
        },
        "timing_min": round(train_time / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }
    if overfitting_cb.train_accs:
        report["per_epoch"]["train_accuracy"] = [round(a, 4) for a in overfitting_cb.train_accs]
        report["per_epoch"]["val_accuracy"] = [round(a, 4) for a in overfitting_cb.val_accs]

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "overfitting_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}", flush=True)

    generate_overfitting_plots(overfitting_cb, args.output_dir, args.output_dir)

    print("\n" + "=" * 60, flush=True)
    print("  OVERFITTING SUMMARY", flush=True)
    print("=" * 60, flush=True)
    if args.skip_final_accuracy:
        print("  Final GSM8K accuracy skipped (run without --skip-final-accuracy to evaluate).", flush=True)
    else:
        print(f"  Train acc:  {train_acc*100:.1f}%  ({train_correct}/{train_total})", flush=True)
        print(f"  Val acc:    {val_acc*100:.1f}%  ({val_correct}/{val_total})", flush=True)
        print(f"  Test acc:   {test_acc*100:.1f}%  ({test_correct}/{test_total})", flush=True)
        print(f"  Train−Val gap:  {(train_acc-val_acc)*100:.1f}%", flush=True)
        print(f"  Train−Test gap: {(train_acc-test_acc)*100:.1f}%", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
