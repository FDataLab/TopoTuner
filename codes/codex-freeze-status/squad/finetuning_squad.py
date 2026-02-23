import json
import os
import glob
import shutil
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

from .data_preprocessing_squad import (
    preprocess_dataset, custom_data_collator,
    infer_prompt_format_from_model_id,
)
from .eval_squad import evaluate_squad

from codes.utils.args import parse_args
from codes.utils.model_saving import SavePeftModelCallback

from transformers.utils import logging as hf_logging
hf_logging.enable_progress_bar()

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

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

class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] "
                  f"Loss: {logs['loss']:.6f}", flush=True)

class EMF1PerEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, run_eval: bool = True, split: str = "validation",
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
        metrics = evaluate_squad(
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
            "f1": metrics["f1"],
            "n": metrics["n"],
            **gpu_info
        }
        print(
          f"[Downstream] epoch={record['epoch']} SQuAD {self.split} "
          f"EM={record['em']:.2f}% F1={record['f1']:.2f}% n={record['n']} "
          f"GPU={gpu_info['gpu']} mem={gpu_info['mem_alloc_MB']}MB",
          flush=True
        )

        if self.log_jsonl:
            os.makedirs(os.path.dirname(self.log_jsonl), exist_ok=True)
            with open(self.log_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
        if self.log_tsv:
            import csv
            new_file = not os.path.exists(self.log_tsv)
            with open(self.log_tsv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys(), delimiter="\t")
                if new_file:
                    writer.writeheader()
                writer.writerow(record)

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
        token=HF_TOKEN
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
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

    model.gradient_checkpointing_enable()
    return model, tok

def main():
    args = parse_args()

    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"squad-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )
    run_eval = True

    subset_dir = args.subset_save_dir
    if subset_dir and os.path.exists(os.path.join(subset_dir, "state.json")):
        from datasets import DatasetDict, load_from_disk
        print(f">>> Loading SQuAD subset from {subset_dir}")
        ds = load_from_disk(subset_dir)
        train_full, val_ds = ds["train"], ds["validation"]
    else:
        ds = load_dataset("squad")
        train_full, val_ds = ds["train"], ds["validation"]
        subset_size = min(args.subset_train_size, len(train_full))
        print(f">>> Sampling subset of {subset_size} from {len(train_full)} with seed {args.subset_seed}")
        train_full = train_full.shuffle(seed=args.subset_seed).select(range(subset_size))
        if subset_dir:
            from datasets import DatasetDict
            os.makedirs(subset_dir, exist_ok=True)
            print(f">>> Saving SQuAD subset to {subset_dir}")
            DatasetDict({"train": train_full, "validation": val_ds}).save_to_disk(subset_dir)

    split = train_full.train_test_split(test_size=0.05, seed=args.subset_seed)
    train_ds, dev_ds = split["train"], split["test"]
    print(f"Train {len(train_ds)} | Dev {len(dev_ds)} | Val {len(val_ds)}")

    model, tok = load_model_and_tokenizer(args.model_name, args.use_lora)
    pf = infer_prompt_format_from_model_id(args.model_name)

    tokenized_train = train_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=True),
        remove_columns=train_ds.column_names
    )
    tokenized_val = dev_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=False),
        remove_columns=dev_ds.column_names
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./SQuAD/logs/{timestamp}",
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

    safe_model = args.model_name.replace("/", "_")
    safe_dataset = args.dataset_name.replace("/", "_")
    log_jsonl = os.path.join(args.output_dir, f"{safe_dataset}_{safe_model}_downstream_eval.jsonl")
    log_tsv = os.path.join(args.output_dir, f"{safe_dataset}_{safe_model}_downstream_eval.tsv")
    run_name  = wandb.run.name if wandb.run else ""

    callbacks = [
        EMF1PerEpochCallback(
            tok,
            run_eval=True,
            split="validation",
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
        callbacks.insert(0, SavePeftModelCallback(args, tokenizer=tok))

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

    if run_eval:
        print(">>> Running baseline evaluation before training...", flush=True)
        base = evaluate_squad(
            model, tok,
            split="validation",
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
        print(f"[Baseline] SQuAD val EM={base['em']:.2f}% F1={base['f1']:.2f}% n={base['n']}", flush=True)
        print(">>> Baseline evaluation finished, starting training...", flush=True)

    train_dl = trainer.get_train_dataloader()
    steps_per_epoch = len(train_dl)
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(f">>> Training plan: steps_per_epoch={steps_per_epoch} x epochs={training_args.num_train_epochs} = total_updates={total_update_steps}", flush=True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    # Delete Hugging Face default checkpoints to save disk
    for path in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        print(f"🗑️ Removing default checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)

    if run_eval:
        final = evaluate_squad(
            model, tok,
            split="validation",
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
        print(f"[Final] SQuAD val EM={final['em']:.2f}% F1={final['f1']:.2f}% n={final['n']}", flush=True)

if __name__ == "__main__":
    main()

