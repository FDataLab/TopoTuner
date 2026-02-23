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

from .data_preprocessing_imdb import (
    preprocess_dataset, custom_data_collator,
    infer_prompt_format_from_model_id
)
from .eval_imdb import evaluate_imdb

from codes.utils.args import parse_args
from codes.utils.model_saving import SavePeftModelCallback, concise_lora_filename, concise_full_filename

from transformers.utils import logging as hf_logging
hf_logging.enable_progress_bar()

import wandb
import shutil
import glob

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

class AccuracyPerEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, run_eval: bool = False, split: str = "test",
                 limit=None, max_new_tokens: int = 50,
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
        metrics = evaluate_imdb(
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
            "accuracy": metrics["accuracy"],
            "positive_acc": metrics["positive_acc"],
            "negative_acc": metrics["negative_acc"],
            "n": metrics["n"],
            **gpu_info
        }
        print(
          f"[Downstream] epoch={record['epoch']} IMDB {self.split} "
          f"Accuracy={record['accuracy']:.2f}% Pos={record['positive_acc']:.2f}% Neg={record['negative_acc']:.2f}% n={record['n']} "
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
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
    model.config.use_cache = False
    return model, tok




# ---------- Main ----------
def main():
    args = parse_args()
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"imdb-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )
    run_eval = False

    # Dataset - use full IMDB dataset (25k train examples)
    ds = load_dataset("stanfordnlp/imdb")
    test_ds = ds["test"]

    if getattr(args, "train_csv", ""):
        import pandas as pd
        from datasets import Dataset

        train_df = pd.read_csv(args.train_csv)
        if "text" not in train_df.columns or "label" not in train_df.columns:
            raise ValueError(f"CSV at {args.train_csv} must contain 'text' and 'label' columns")
        train_full = Dataset.from_pandas(train_df, preserve_index=False)
    else:
        train_full = ds["train"]

    # Use ALL 25k training examples (no subset)
    train_full = train_full.shuffle(seed=42)
    
    # Create a small dev split from the full training set
    split = train_full.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    print(f"IMDB Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

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
        logging_dir=f"./IMDB/logs/{timestamp}",
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
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=False,
        dataloader_pin_memory=True,
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
    run_name = wandb.run.name if wandb.run else ""

    # Callbacks
    callbacks = [
        AccuracyPerEpochCallback(
            tok,
            run_eval=False,
            split="test",
            limit=None,
            max_new_tokens=50,
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
        if not os.path.exists(save_dir):
            print(f">>> Saving baseline as epoch-0 to {save_dir}", flush=True)
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            if tok:
                tok.save_pretrained(save_dir)
            import torch as _torch
            _torch.save(args, os.path.join(save_dir, "training_args.bin"))
            if getattr(args, "save_npy", False):
                import numpy as _np
                npy_dir = os.path.join(save_dir, "numpy_weights")
                os.makedirs(npy_dir, exist_ok=True)
                count = 0
                for name, param in model.named_parameters():
                    if args.use_lora:
                        if "lora_A" in name or "lora_B" in name:
                            short = concise_lora_filename(name)
                            if short:
                                arr = param.detach().cpu().to(_torch.float16)
                                _np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                                count += 1
                    else:
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

    train_dl = trainer.get_train_dataloader()
    steps_per_epoch = len(train_dl)
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(f">>> Training plan: steps_per_epoch={steps_per_epoch} "
          f"x epochs={training_args.num_train_epochs} "
          f"= total_updates={total_update_steps}", flush=True)

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # # Final eval (optional)
    # if run_eval:
    #     final = evaluate_imdb(
    #         model, tok,
    #         split="test",
    #         limit=None,
    #         max_new_tokens=50,
    #         progress_bar=True,
    #         save_jsonl=log_jsonl,
    #         save_tsv=log_tsv,
    #         run_name=run_name,
    #         phase="final",
    #         epoch=int(training_args.num_train_epochs),
    #         step=int(trainer.state.global_step),
    #         output_dir=training_args.output_dir,
    #     )
    #     print(f"[Final] IMDB test Accuracy={final['accuracy']:.2f}% Pos={final['positive_acc']:.2f}% Neg={final['negative_acc']:.2f}% n={final['n']}", flush=True)

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
IMDB Sentiment Analysis Examples:

Llama-3.1-8B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --output-dir ./numpy_weights/imdb/llama31_8b/lora \
  --batch-size 8 --epochs 3 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Llama-3.1-8B_lora.log 2>&1 &

Llama-3.2-3B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name meta-llama/Llama-3.2-3B \
  --output-dir ./numpy_weights/imdb/llama32_3b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Llama-3.2-3B_full.log 2>&1 &

Llama-3.2-3B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --output-dir ./numpy_weights/imdb/llama32_3b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Llama-3.2-3B_lora.log 2>&1 &

-----

Mistral-7B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name mistralai/Mistral-7B-v0.1 \
  --use-lora \
  --output-dir ./numpy_weights/imdb/mistral7b/lora \
  --batch-size 8 --epochs 3 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Mistral-7B_lora.log 2>&1 &

Qwen-3-8B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name Qwen/Qwen3-8B \
  --use-lora \
  --output-dir ./numpy_weights/imdb/qwen_8b/lora \
  --batch-size 8 --epochs 3 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Qwen3-8B_lora.log 2>&1 &
"""
