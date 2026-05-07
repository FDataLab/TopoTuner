import os
import csv
import time
import datetime
import torch
from functools import partial
from typing import Optional

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from codes.mmlu.data_preprocessing_mmlu import (
    preprocess_dataset, custom_data_collator,
    infer_prompt_format_from_model_id,
)

from codes.utils.args import parse_args
from codes.utils.gpu_train_stats import GpuTrainStatsCallback

from transformers.utils import logging as hf_logging
hf_logging.enable_progress_bar()

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")


class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] Loss: {logs['loss']:.6f}", flush=True)

def load_model_and_tokenizer(model_id: str, use_lora: bool, freeze_layers=None):
    if freeze_layers is None:
        freeze_layers = []

    # --------------------------
    # Tokenizer
    # --------------------------
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        token=HF_TOKEN
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # --------------------------
    # Base Model
    # --------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    model.config.use_cache = False

    # --------------------------
    # Freeze transformer layers (no training on these)
    # --------------------------
    if freeze_layers:
        print(f"🔥 Freezing Transformer layers: {freeze_layers}")

        # Qwen: model.transformer.layers
        # LLaMA: model.model.layers
        try:
            transformer_layers = model.transformer.layers
        except AttributeError:
            transformer_layers = model.model.layers

        for idx, layer in enumerate(transformer_layers):
            if idx in freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"   → Layer {idx} frozen (epoch-0 behavior preserved)")

    # --------------------------
    # LoRA (inject on q/k/v/o, then kill LoRA in frozen layers)
    # --------------------------
    if use_lora:
        print("⚙️  Applying LoRA: q/k/v/o (match codes/gsm8k/finetune_gsm8k.py; disable on frozen layers)")

        # Here we use standard string-based target_modules, since
        # this PEFT version doesn't support callables.
        lcfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lcfg)

        # Now: freeze LoRA parameters *inside* frozen layers
        if freeze_layers:
            for name, param in model.named_parameters():
                if "lora_" not in name:
                    continue

                # Works for both "model.layers.X..." and "transformer.layers.X..."
                for layer_idx in freeze_layers:
                    if f".layers.{layer_idx}." in name:
                        param.requires_grad = False
                        print(f"   → LoRA disabled in frozen layer {layer_idx}: {name}")
                        break

        model.print_trainable_parameters()

    return model, tok

# def load_model_and_tokenizer(model_id: str, use_lora: bool):
#     tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right", token=HF_TOKEN)

#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#         tok.pad_token_id = tok.eos_token_id

#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         device_map={"": 0},
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         token=HF_TOKEN,
#     )

#     if use_lora:
#         lcfg = LoraConfig(
#             r=16,
#             lora_alpha=32,
#             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#             lora_dropout=0.05,
#             bias="none",
#             task_type=TaskType.CAUSAL_LM,
#         )
#         model = get_peft_model(model, lcfg)
#         model.print_trainable_parameters()

#     model.config.use_cache = False
#     return model, tok


def main():
    args = parse_args()

    # Load MMLU (Hendrycks Test)
    ds = load_dataset('cais/mmlu', 'all')
    print(f"Available MMLU splits: {list(ds.keys())}")
    # Use auxiliary_train split for supervised tuning (validation is too small ~1.5k)
    split_name = "auxiliary_train" if "auxiliary_train" in ds else ("train" if "train" in ds else ("validation" if "validation" in ds else list(ds.keys())[0]))
    print(f"Using split: {split_name}")
    full_ds: Dataset = ds[split_name]
    full_ds = full_ds.add_column("orig_idx", list(range(len(full_ds))))

    if getattr(args, "train_csv", ""):
        import pandas as pd

        pert_df = pd.read_csv(args.train_csv)
        if "orig_idx" not in pert_df.columns or "answer" not in pert_df.columns:
            raise ValueError(f"CSV at {args.train_csv} must contain 'orig_idx' and 'answer' columns")
        answer_map = {int(row.orig_idx): int(row.answer) for row in pert_df.itertuples()}
        flag_map = {int(row.orig_idx): int(getattr(row, "perturbed", 0)) for row in pert_df.itertuples()}

        def apply_perturb(example):
            idx = example["orig_idx"]
            if idx in answer_map:
                example["answer"] = answer_map[idx]
                example["perturbed_flag"] = flag_map.get(idx, 0)
            else:
                example["perturbed_flag"] = 0
            return example

        full_ds = full_ds.map(apply_perturb)

    print(f"Full dataset size: {len(full_ds)}")

    # Subset from args: reuse Hotpot subset args for consistency
    subset_size = min(args.subset_train_size, len(full_ds))
    train_ds = full_ds.shuffle(seed=args.subset_seed).select(range(subset_size))
    print(f"MMLU Train {len(train_ds)} (subset {subset_size}, no held-out dev for LM eval)")

    # Model & tokenizer
    model, tok = load_model_and_tokenizer(args.model_name, args.use_lora, args.freeze_layers)
    pf = infer_prompt_format_from_model_id(args.model_name)

    tokenized_train = train_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=512, prompt_format=pf, is_train=True),
        remove_columns=train_ds.column_names,
    )

    _seed = getattr(args, "subset_seed", 42)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./MMLU/logs/{timestamp}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",  # HF checkpoint-* under output_dir; keep all (like finetune_gsm8k.py)
        save_total_limit=None,
        eval_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
        seed=_seed,
        data_seed=_seed,
        label_names=["labels"],
        disable_tqdm=False,
    )

    collator = partial(custom_data_collator, tokenizer=tok)
    # Saving: HF checkpoint-* in output_dir only (same as finetune_gsm8k.py); no custom epoch_weights.
    callbacks = [LossDebugCallback()]

    class MetricsCSVCallback(TrainerCallback):
        def __init__(self, csv_path: str):
            self.csv_path = csv_path
            self.fieldnames = [
                "time", "epoch", "step",
                "loss", "grad_norm", "learning_rate",
                "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
                "train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss",
            ]

        def _append(self, row: dict):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            new_file = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                if new_file:
                    w.writeheader()
                w.writerow({k: row.get(k) for k in self.fieldnames})

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            row = {
                "time": time.time(),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                "step": int(state.global_step),
            }
            row.update(logs)
            self._append(row)

    metrics_csv_path = os.path.join(args.output_dir, "training_metrics.csv")
    callbacks.append(MetricsCSVCallback(metrics_csv_path))
    callbacks.append(GpuTrainStatsCallback(args.output_dir))

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("\n[5/5] Saving model & tokenizer...", flush=True)
    trainer.save_model(args.output_dir)
    if tok:
        tok.save_pretrained(args.output_dir)
    print(f"  Model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()


"""
nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --freeze-layers 18 19 21 22 23 26 \
  --output-dir ./numpy_weights/mmlu/llama31_8b/combined_lora_freezeA \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Llama-3.1-8B_combined_lora_freezeA.log 2>&1 &
"""




"""
export CUDA_VISIBLE_DEVICES=0

Llama-3.1-8B (8-bit Lora / full)

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --output-dir ./numpy_weights/mmlu/llama31_8b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Llama-3.1-8B_lora.log 2>&1 &

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name meta-llama/Llama-3.1-8B \
  --output-dir ./numpy_weights/mmlu/llama31_8b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Llama-3.1-8B_full.log 2>&1 &

---
Llama-3.2-3B (4-bit Lora / full)

  python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --output-dir ./numpy_weights/mmlu/llama32_3b/lora_fixed \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Llama-3.2-3B_lora_fixed.log 2>&1 &

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name meta-llama/Llama-3.2-3B \
  --output-dir ./numpy_weights/mmlu/llama32_3b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Llama-3.2-3B_full.log 2>&1 &

---
Mistral-7B (8-bit Lora / full)

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name mistralai/Mistral-7B-v0.1 \
  --use-lora \
  --output-dir ./numpy_weights/mmlu/mistral7b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Mistral-7B_lora.log 2>&1 &

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name mistralai/Mistral-7B-v0.1 \
  --output-dir ./numpy_weights/mmlu/mistral7b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Mistral-7B_full.log 2>&1 &

---

Qwen-3-8B (8-bit Lora / full)

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name Qwen/Qwen3-8B \
  --use-lora \
  --output-dir ./numpy_weights/mmlu/qwen_8b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Qwen3-8B_lora.log 2>&1 &

nohup python -m codes.mmlu.finetuning_mmlu \
  --dataset-name MMLU \
  --model-name Qwen/Qwen3-8B \
  --output-dir ./numpy_weights/mmlu/qwen_8b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --subset-train-size 20000 --subset-seed 42 \
  --save-every-epoch --save-npy \
  > logs/finetune_MMLU_Qwen3-8B_full.log 2>&1 &

"""
