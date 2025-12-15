import json
import os
import math
import datetime
import torch
from functools import partial
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from .data_preprocessing_gsm8k import (
    preprocess_dataset, custom_data_collator, add_special_tokens_if_missing,
    infer_prompt_format_from_model_id
)
from .eval_gsm8k_OLD import evaluate_gsm8k  # uses the progress + save_jsonl/tsv version

from codes.utils.args import parse_args
from codes.utils.model_saving import SavePeftModelCallback, concise_lora_filename, concise_full_filename

# (optional) make sure HF progress bar is on
from transformers.utils import logging as hf_logging
hf_logging.enable_progress_bar()

import wandb

import shutil
import glob

# force single GPU to avoid DataParallel scatter issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# ---------- GPU Info ----------

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_id": None, "mem_alloc_MB": None, "mem_reserved_MB": None}
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu": props.name,
        "gpu_id": gpu_id,
        "total_mem_GB": round(props.total_memory / 1024**3, 2),
        "mem_alloc_MB": torch.cuda.memory_allocated(gpu_id) // 1024**2,
        "mem_reserved_MB": torch.cuda.memory_reserved(gpu_id) // 1024**2,
    }

# ---------- Callbacks ----------
class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] "
                  f"Loss: {logs['loss']:.6f}", flush=True)

class EMPerEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, run_eval: bool = False, split: str = "test",
                 limit=None, max_new_tokens: int = 256,
                 log_jsonl=None, log_tsv=None, dataset="", model=""):
        self.tok = tokenizer
        self.run_eval = run_eval
        self.split = split
        self.limit = limit
        self.max_new_tokens = max_new_tokens
        self.log_jsonl = log_jsonl
        self.log_tsv = log_tsv
        self.dataset = dataset
        self.model = model

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self.run_eval:
            return
        metrics = evaluate_gsm8k(
            model, self.tok,
            split=self.split,
            limit=self.limit,
            max_new_tokens=self.max_new_tokens,
            debug_print=True
        )
        gpu_info = get_gpu_info()
        record = {
            "epoch": int(state.epoch),
            "dataset": self.dataset,
            "model": self.model,
            "em": metrics["em"],
            "n": metrics["n"],
            **gpu_info
        }
        print(
          f"[Downstream] epoch={record['epoch']} GSM8K {self.split} "
          f"EM={record['em']:.2f}% n={record['n']} "
          f"GPU={gpu_info['gpu']} mem={gpu_info['mem_alloc_MB']}MB",
          flush=True
        )


        # append to jsonl
        if self.log_jsonl:
            os.makedirs(os.path.dirname(self.log_jsonl), exist_ok=True)
            with open(self.log_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
        # append to tsv
        if self.log_tsv:
            import csv
            new_file = not os.path.exists(self.log_tsv)
            with open(self.log_tsv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys(), delimiter="\t")
                if new_file:
                    writer.writeheader()
                writer.writerow(record)

# ---------- Loader ----------
def load_model_and_tokenizer(model_id: str, use_lora: bool):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right", token=HF_TOKEN)

    # Reuse EOS as PAD to avoid adding a new pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Ensure <final-answer> tokens exist (adds 2 tokens if missing)
    add_special_tokens_if_missing(tok)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},           # keep everything on logical cuda:0
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN
    )

    # Resize for the newly added special tokens
    model.resize_token_embeddings(len(tok))

    # Sync pad id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    # training + grad checkpointing = use_cache must be False
    model.config.use_cache = False

    if use_lora:
        lcfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

        # Init the new special tokens to EOS embedding (helps early stability)
        with torch.no_grad():
            eos_id = tok.eos_token_id
            new_ids = tok.convert_tokens_to_ids(["<final-answer>", "</final-answer>"])
            emb = model.get_input_embeddings().weight
            for tid in new_ids:
                if eos_id is not None and 0 <= tid < emb.size(0):
                    emb[tid].copy_(emb[eos_id])

    model.gradient_checkpointing_enable()
    return model, tok

# ---------- Main ----------
def main():
    args = parse_args()

    # W&B
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"gsm8k-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )
    run_eval = False

    # Dataset
    ds = load_dataset("openai/gsm8k", "main")
    train_full, test_ds = ds["train"], ds["test"]

    # Use full dataset for GSM8K (around 8500 examples)
    # train_full = train_full.select(range(len(ds["train"]) // 64))  # removed downsample
    # test_ds = test_ds.select(range(len(ds["test"]) // 64))  # removed downsample

    split = train_full.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    # Model & tokenizer
    model, tok = load_model_and_tokenizer(args.model_name, args.use_lora)
    pf = infer_prompt_format_from_model_id(args.model_name)

    tokenized_train = train_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=True),
        remove_columns=train_ds.column_names
    )
    tokenized_val = val_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=False),
        remove_columns=val_ds.column_names
    )

    # Training args
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./GSM8K/logs/{timestamp}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy="epoch",
        save_strategy="epoch" if args.save_every_epoch else "no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        optim="paged_adamw_32bit",
        bf16=False,
        report_to="wandb",

        # logging + tqdm
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=False,
        dataloader_pin_memory=True,

        # avoid PeftModel label_names warning
        label_names=["labels"],
    )

    # Build safe names
    safe_model = args.model_name.replace("/", "_")
    safe_dataset = args.dataset_name.replace("/", "_")

    log_jsonl = os.path.join(
        args.output_dir,
        f"{safe_dataset}_{safe_model}_downstream_eval.jsonl"
    )
    log_tsv = os.path.join(
        args.output_dir,
        f"{safe_dataset}_{safe_model}_downstream_eval.tsv"
    )
    run_name  = wandb.run.name if wandb.run else ""

    # Callbacks
    callbacks = [
        EMPerEpochCallback(
            tok,
            run_eval=False,
            split="test",
            limit=None,
            max_new_tokens=256,
            log_jsonl=log_jsonl,
            log_tsv=log_tsv,
            dataset=args.dataset_name,
            model=args.model_name
        ),
        LossDebugCallback(),
]
    if args.save_every_epoch or args.save_npy:
        callbacks.insert(0, SavePeftModelCallback(args, tokenizer=tok))  # save first

    # Trainer
    collator = partial(custom_data_collator, tokenizer=tok)
    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        callbacks=callbacks,
    )

    print(">>> Callbacks attached:", trainer.callback_handler.callbacks, flush=True)

    # Save baseline as epoch 0 (before training) when epoch saving is active
    if getattr(args, "save_every_epoch", False) or getattr(args, "save_npy", False):
        base_dir = os.path.join(args.output_dir, "epoch_weights")
        os.makedirs(base_dir, exist_ok=True)
        save_dir = os.path.join(base_dir, "checkpoint-epoch-0")
        
        # Only save epoch-0 if it doesn't already exist (don't overwrite)
        if not os.path.exists(save_dir):
            print(f">>> Saving baseline as epoch-0 to {save_dir}", flush=True)
            os.makedirs(save_dir, exist_ok=True)

            # Save model and tokenizer
            model.save_pretrained(save_dir)
            if tok:
                tok.save_pretrained(save_dir)

            # Save training args
            import torch as _torch
            _torch.save(args, os.path.join(save_dir, "training_args.bin"))

            # Optional numpy dump
            if getattr(args, "save_npy", False):
                import numpy as _np
                npy_dir = os.path.join(save_dir, "numpy_weights")
                os.makedirs(npy_dir, exist_ok=True)
                count = 0
                for name, param in model.named_parameters():
                    if args.use_lora:
                        if "lora_A" in name or "lora_B" in name:
                            # Use the same concise filename logic as MMLU
                            short = concise_lora_filename(name)
                            if short:
                                arr = param.detach().cpu().to(_torch.float16)
                                _np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                                count += 1
                    else:
                        # For full fine-tuning, save trainable parameters with concise names
                        if param.requires_grad:
                            short = concise_full_filename(name)
                            if short:
                                arr = param.detach().cpu().to(_torch.float16)
                                _np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                                count += 1
                print(f">>> Saved {count} numpy weight files to {npy_dir}", flush=True)
        else:
            print(f">>> Epoch-0 already exists at {save_dir}, skipping baseline save", flush=True)

    _ = trainer.create_optimizer_and_scheduler(num_training_steps=training_args.max_steps)
    print("Optimizer:", trainer.optimizer)
    print("Scheduler:", trainer.lr_scheduler)

    # Optional baseline eval before training
    if run_eval:
        print(">>> Running baseline evaluation before training...", flush=True)
        base = evaluate_gsm8k(
            model, tok,
            split="test",
            limit=None,
            max_new_tokens=256,
            progress_bar=True,
            save_jsonl=log_jsonl,
            save_tsv=log_tsv,
            run_name=run_name,
            phase="baseline",
            epoch=0,
            step=0,
            output_dir=training_args.output_dir,
        )
        print(f"[Baseline] GSM8K test EM={base['em']:.2f}%  n={base['n']}", flush=True)
        print(">>> Baseline evaluation finished, starting training...", flush=True)

    # Better step plan print (state.max_steps is 0 before train)
    train_dl = trainer.get_train_dataloader()
    steps_per_epoch = len(train_dl)
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(f">>> Training plan: steps_per_epoch={steps_per_epoch} "
          f"x epochs={training_args.num_train_epochs} "
          f"= total_updates={total_update_steps}", flush=True)

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final
    #final_dir = f"{args.output_dir}/final_model"
    #trainer.save_model(final_dir)
    #tok.save_pretrained(final_dir)

    # Final eval (optional)
    if run_eval:
        final = evaluate_gsm8k(
            model, tok,
            split="test",
            limit=None,
            max_new_tokens=256,
            progress_bar=True,
            save_jsonl=log_jsonl,
            save_tsv=log_tsv,
            run_name=run_name,
            phase="final",
            epoch=int(training_args.num_train_epochs),
            step=int(trainer.state.global_step),
            output_dir=training_args.output_dir,
        )
        print(f"[Final] GSM8K test EM={final['em']:.2f}%  n={final['n']}", flush=True)

    # Delete Hugging Face default checkpoints
    for path in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        print(f"🗑️ Removing default checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)

    # Delete final_model if it exists
    final_model_path = os.path.join(args.output_dir, "final_model")
    if os.path.exists(final_model_path):
        print(f"🗑️ Removing final model folder: {final_model_path}")
        shutil.rmtree(final_model_path, ignore_errors=True)

if __name__ == "__main__":
    main()

"""
export CUDA_VISIBLE_DEVICES=0

Llama-3.1-8B (8-bit Lora / full)

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --output-dir ./numpy_weights/gsm8k/llama31_8b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Llama-3.1-8B_lora.log 2>&1 &

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name meta-llama/Llama-3.1-8B \
  --output-dir ./numpy_weights/gsm8k/llama31_8b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Llama-3.1-8B_full.log 2>&1 &

---
Llama-3.2-3B (4-bit Lora / full)

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --output-dir ./numpy_weights/gsm8k/llama32_3b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Llama-3.2-3B_lora.log 2>&1 &

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name meta-llama/Llama-3.2-3B \
  --output-dir ./numpy_weights/gsm8k/llama32_3b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Llama-3.2-3B_full.log 2>&1 &

---
Mistral-7B (8-bit Lora / full)

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name mistralai/Mistral-7B-v0.1 \
  --use-lora \
  --output-dir ./numpy_weights/gsm8k/mistral7b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Mistral-7B_lora.log 2>&1 &

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name mistralai/Mistral-7B-v0.1 \
  --output-dir ./numpy_weights/gsm8k/mistral7b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Mistral-7B_full.log 2>&1 &

---
Qwen-3-8B (8-bit Lora / full)

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name Qwen/Qwen3-8B \
  --use-lora \
  --output-dir ./numpy_weights/gsm8k/qwen_8b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Qwen3-8B_lora.log 2>&1 &

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name Qwen/Qwen3-8B \
  --output-dir ./numpy_weights/gsm8k/qwen_8b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_Qwen3-8B_full.log 2>&1 &

---
Olmo-7B (8-bit Lora / full)

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name allenai/OLMo-7B-hf \
  --use-lora \
  --output-dir ./numpy_weights/gsm8k/olmo7b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_OLMo-7B_lora.log 2>&1 &

nohup python -m codes.gsm8k.finetuning_gsm8k \
  --dataset-name GSM8K \
  --model-name allenai/OLMo-7B-hf \
  --output-dir ./numpy_weights/gsm8k/olmo7b/full \
  --batch-size 1 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_GSM8K_OLMo-7B_full.log 2>&1 &
"""