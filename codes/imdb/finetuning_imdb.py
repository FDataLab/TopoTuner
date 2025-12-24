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
        # Verify frozen parameters remain frozen at each epoch
        if hasattr(model, '_frozen_param_names') and model._frozen_param_names:
            still_frozen = 0
            for name in model._frozen_param_names:
                for p_name, p in model.named_parameters():
                    if p_name == name and not p.requires_grad:
                        still_frozen += 1
                        break
            if still_frozen < len(model._frozen_param_names):
                print(f"⚠️  WARNING: Only {still_frozen}/{len(model._frozen_param_names)} frozen params remain frozen at epoch {int(state.epoch)}!", flush=True)
            else:
                print(f"✅ Verified: All {still_frozen} frozen parameters remain frozen at epoch {int(state.epoch)}", flush=True)
        
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

    # ---------- Loader (with freezable layers, matching HotpotQA) ----------
def load_model_and_tokenizer(
    model_id: str,
    use_lora: bool,
    freeze_layers=None,
    freeze_q_layers=None,
    freeze_k_layers=None,
    freeze_v_layers=None,
):
    """
    Load model/tokenizer and optionally:
      - freeze full transformer layers (`freeze_layers`)
      - freeze only Q / K / V projections in selected layers
        (`freeze_q_layers`, `freeze_k_layers`, `freeze_v_layers`)
    Behavior mirrors `hotpotqa/finetuning_hotpotqa_updated.py`.
    
    ⚠️  CRITICAL: This function is called ONCE before training starts.
    Freezing happens here and persists throughout all training epochs.
    """
    print("   [load_model_and_tokenizer] Starting model loading...", flush=True)
    
    freeze_layers = freeze_layers or []
    freeze_q_layers = freeze_q_layers or []
    freeze_k_layers = freeze_k_layers or []
    freeze_v_layers = freeze_v_layers or []
    
    print(f"   [load_model_and_tokenizer] Freeze configuration:", flush=True)
    print(f"      - Full layers: {freeze_layers}", flush=True)
    print(f"      - Q layers: {freeze_q_layers}", flush=True)
    print(f"      - K layers: {freeze_k_layers}", flush=True)
    print(f"      - V layers: {freeze_v_layers}", flush=True)

    # Load tokenizer
    print("   [load_model_and_tokenizer] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        token=HF_TOKEN,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    print(f"   [load_model_and_tokenizer] ✅ Tokenizer loaded: vocab_size={tok.vocab_size}", flush=True)

    # Load model
    print("   [load_model_and_tokenizer] Loading model (this may take a moment)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    print(f"   [load_model_and_tokenizer] ✅ Model loaded: {model_id}", flush=True)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    print("   [load_model_and_tokenizer] ✅ Model config updated", flush=True)

    # ---- Freeze full transformer layers (base weights) ----
    if freeze_layers:
        print(f"🔥 Freezing Transformer layers: {freeze_layers}", flush=True)

        try:
            transformer_layers = model.transformer.layers  # Qwen-like
        except AttributeError:
            transformer_layers = model.model.layers        # LLaMA-like

        for idx, layer in enumerate(transformer_layers):
            if idx in freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
                print(f"   → Base layer {idx} frozen (epoch-0 behavior preserved)", flush=True)

    # ---- Freeze ONLY q/k/v projections in selected layers (base weights) ----
    if freeze_q_layers or freeze_k_layers or freeze_v_layers:
        qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
        print(f"🧊 Freezing projections: Q={sorted(qset)} K={sorted(kset)} V={sorted(vset)}", flush=True)

        frozen_count = {"q": 0, "k": 0, "v": 0}
        checked_params = 0
        matched_layers = set()
        for name, p in model.named_parameters():
            # Match layer index in parameter name (pattern: ".layers.{i}.")
            hit_layer = None
            for i in (qset | kset | vset):
                if f".layers.{i}." in name:
                    hit_layer = i
                    matched_layers.add(i)
                    break
            if hit_layer is None:
                # Only print if it's a q/k/v projection (to avoid too much output)
                if any(proj in name for proj in [".q_proj.", ".k_proj.", ".v_proj."]):
                    checked_params += 1
                continue
            checked_params += 1

            # Freeze selectively by projection
            if hit_layer in qset and ".q_proj." in name:
                p.requires_grad = False
                frozen_count["q"] += 1
                print(f"   → Frozen: layer {hit_layer} Q-proj ({name})", flush=True)
            if hit_layer in kset and ".k_proj." in name:
                p.requires_grad = False
                frozen_count["k"] += 1
                print(f"   → Frozen: layer {hit_layer} K-proj ({name})", flush=True)
            if hit_layer in vset and ".v_proj." in name:
                p.requires_grad = False
                frozen_count["v"] += 1
                print(f"   → Frozen: layer {hit_layer} V-proj ({name})", flush=True)
        
        print(f"🔍 Layer matching summary: checked {checked_params} q/k/v projection parameters, matched layers {sorted(matched_layers)}", flush=True)
        print(f"✅ Frozen parameter counts: Q={frozen_count['q']} K={frozen_count['k']} V={frozen_count['v']} (total={sum(frozen_count.values())})", flush=True)

    # ---- Apply LoRA on q/k/v, and disable LoRA params in frozen layers ----
    if use_lora:
        print("⚙️  Applying LoRA: q/k/v only (disable LoRA in frozen layers)", flush=True)

        lcfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lcfg)

        # Disable LoRA params in fully frozen layers
        if freeze_layers:
            lora_frozen_count = 0
            for name, p in model.named_parameters():
                if "lora_" not in name:
                    continue
                for layer_idx in freeze_layers:
                    if f".layers.{layer_idx}." in name:
                        p.requires_grad = False
                        lora_frozen_count += 1
                        print(f"   → Frozen LoRA: layer {layer_idx} ({name})", flush=True)
                        break
            if lora_frozen_count > 0:
                print(f"✅ Frozen {lora_frozen_count} LoRA parameters in fully frozen layers", flush=True)

        # Disable LoRA params for q/k/v projections in specified layers
        if freeze_q_layers or freeze_k_layers or freeze_v_layers:
            qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
            lora_frozen_count = {"q": 0, "k": 0, "v": 0}

            for name, p in model.named_parameters():
                if "lora_" not in name:
                    continue

                # Find layer idx match
                layer_hit = None
                for i in (qset | kset | vset):
                    if f".layers.{i}." in name:
                        layer_hit = i
                        break
                if layer_hit is None:
                    continue

                # Freeze LoRA for the specific projection in that layer
                if layer_hit in qset and "q_proj" in name:
                    p.requires_grad = False
                    lora_frozen_count["q"] += 1
                    print(f"   → Frozen LoRA: layer {layer_hit} Q-proj ({name})", flush=True)
                if layer_hit in kset and "k_proj" in name:
                    p.requires_grad = False
                    lora_frozen_count["k"] += 1
                    print(f"   → Frozen LoRA: layer {layer_hit} K-proj ({name})", flush=True)
                if layer_hit in vset and "v_proj" in name:
                    p.requires_grad = False
                    lora_frozen_count["v"] += 1
                    print(f"   → Frozen LoRA: layer {layer_hit} V-proj ({name})", flush=True)
            
            if sum(lora_frozen_count.values()) > 0:
                print(f"✅ Frozen LoRA counts: Q={lora_frozen_count['q']} K={lora_frozen_count['k']} V={lora_frozen_count['v']} (total={sum(lora_frozen_count.values())})", flush=True)

        model.print_trainable_parameters()

    # Verification: Print summary of frozen parameters
    if freeze_layers or freeze_q_layers or freeze_k_layers or freeze_v_layers:
        frozen_params = []
        trainable_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                frozen_params.append(name)
            else:
                trainable_params.append(name)
        print(f"📊 Freezing verification: {len(frozen_params)} frozen, {len(trainable_params)} trainable", flush=True)
        if frozen_params:
            print(f"   Frozen params (first 10): {frozen_params[:10]}", flush=True)
            if len(frozen_params) > 10:
                print(f"   ... and {len(frozen_params) - 10} more", flush=True)
        # Store frozen param names for epoch verification
        model._frozen_param_names = frozen_params

    # Match HotpotQA behavior: enable gradient checkpointing after freezing / LoRA setup
    print("   [load_model_and_tokenizer] Enabling gradient checkpointing...", flush=True)
    model.gradient_checkpointing_enable()
    print("   [load_model_and_tokenizer] ✅ Gradient checkpointing enabled", flush=True)
    
    print("   [load_model_and_tokenizer] ✅ Model loading complete!", flush=True)
    return model, tok




# ---------- Main ----------
def main():
    print("=" * 80, flush=True)
    print("🚀 STARTING IMDB FINETUNING PIPELINE", flush=True)
    print("=" * 80, flush=True)
    
    # STEP 1: Parse arguments
    print("\n[STEP 1/10] 📋 Parsing command-line arguments...", flush=True)
    args = parse_args()
    print(f"   ✅ Arguments parsed:", flush=True)
    print(f"      - Model: {args.model_name}", flush=True)
    print(f"      - Output dir: {args.output_dir}", flush=True)
    print(f"      - Batch size: {args.batch_size}", flush=True)
    print(f"      - Epochs: {args.epochs}", flush=True)
    print(f"      - Learning rate: {args.learning_rate}", flush=True)
    print(f"      - Use LoRA: {getattr(args, 'use_lora', False)}", flush=True)
    print(f"      - Freeze layers: {getattr(args, 'freeze_layers', [])}", flush=True)
    print(f"      - Freeze Q layers: {getattr(args, 'freeze_q_layers', [])}", flush=True)
    print(f"      - Freeze K layers: {getattr(args, 'freeze_k_layers', [])}", flush=True)
    print(f"      - Freeze V layers: {getattr(args, 'freeze_v_layers', [])}", flush=True)
    
    # STEP 2: Initialize wandb
    print("\n[STEP 2/10] 📊 Initializing Weights & Biases...", flush=True)
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"imdb-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )
    print(f"   ✅ WandB initialized: {wandb.run.name if wandb.run else 'N/A'}", flush=True)
    run_eval = False

    # STEP 3: Load dataset
    print("\n[STEP 3/10] 📚 Loading IMDB dataset...", flush=True)
    ds = load_dataset("stanfordnlp/imdb")
    test_ds = ds["test"]
    print(f"   ✅ Dataset loaded: {len(ds['train'])} train, {len(ds['test'])} test", flush=True)

    # STEP 4: Prepare training dataset
    print("\n[STEP 4/10] 🔄 Preparing training dataset...", flush=True)
    if getattr(args, "train_csv", ""):
        import pandas as pd
        from datasets import Dataset
        print(f"   📄 Loading from CSV: {args.train_csv}", flush=True)
        train_df = pd.read_csv(args.train_csv)
        if "text" not in train_df.columns or "label" not in train_df.columns:
            raise ValueError(f"CSV at {args.train_csv} must contain 'text' and 'label' columns")
        train_full = Dataset.from_pandas(train_df, preserve_index=False)
        print(f"   ✅ Loaded {len(train_full)} samples from CSV", flush=True)
    else:
        train_full = ds["train"]
        print(f"   ✅ Using full IMDB train set: {len(train_full)} samples", flush=True)

    # Shuffle dataset
    print("   🔀 Shuffling dataset (seed=42)...", flush=True)
    train_full = train_full.shuffle(seed=42)
    
    # Limit training samples for testing (if TRAIN_LIMIT env var is set)
    train_limit = int(os.environ.get("TRAIN_LIMIT", "0"))
    if train_limit > 0:
        print(f"   ⚠️  LIMITING training dataset to {train_limit} samples (for testing)", flush=True)
        original_size = len(train_full)
        train_full = train_full.select(range(min(train_limit, len(train_full))))
        print(f"   ✅ Reduced from {original_size} to {len(train_full)} samples", flush=True)
    else:
        print(f"   ✅ Using full dataset: {len(train_full)} samples", flush=True)
    
    # Create train/val split
    print("   ✂️  Creating train/val split (95/5)...", flush=True)
    split = train_full.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    print(f"   ✅ Final split: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}", flush=True)

    # STEP 5: Load model and tokenizer (CRITICAL: Freezing happens here!)
    print("\n[STEP 5/10] 🤖 Loading model and tokenizer (FREEZING WILL OCCUR HERE)...", flush=True)
    print("   ⚠️  CRITICAL: This is where freezing happens - ONCE before training starts!", flush=True)
    model, tok = load_model_and_tokenizer(
        args.model_name,
        args.use_lora,
        freeze_layers=getattr(args, "freeze_layers", []),
        freeze_q_layers=getattr(args, "freeze_q_layers", []),
        freeze_k_layers=getattr(args, "freeze_k_layers", []),
        freeze_v_layers=getattr(args, "freeze_v_layers", []),
    )
    print("   ✅ Model and tokenizer loaded, freezing completed", flush=True)
    
    # STEP 6: Infer prompt format
    print("\n[STEP 6/10] 📝 Inferring prompt format...", flush=True)
    pf = infer_prompt_format_from_model_id(args.model_name)
    print(f"   ✅ Prompt format: {pf}", flush=True)

    # STEP 7: Tokenize datasets
    print("\n[STEP 7/10] 🔤 Tokenizing datasets...", flush=True)
    print("   🔄 Tokenizing training set...", flush=True)
    tokenized_train = train_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=True),
        remove_columns=train_ds.column_names
    )
    print(f"   ✅ Training set tokenized: {len(tokenized_train)} examples", flush=True)
    
    print("   🔄 Tokenizing validation set...", flush=True)
    tokenized_val = val_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=False),
        remove_columns=val_ds.column_names
    )
    print(f"   ✅ Validation set tokenized: {len(tokenized_val)} examples", flush=True)

    # STEP 8: Setup training arguments
    print("\n[STEP 8/10] ⚙️  Setting up training arguments...", flush=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./IMDB/logs/{timestamp}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy="no",  # Disabled to speed up training
        save_strategy="epoch" if args.save_every_epoch else "no",
        load_best_model_at_end=False,  # Disabled since eval is off
        metric_for_best_model=None,  # Disabled since eval is off
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
    print(f"   ✅ Training arguments configured:", flush=True)
    print(f"      - Epochs: {args.epochs} (VERIFY THIS IS CORRECT!)", flush=True)
    print(f"      - Batch size: {args.batch_size}", flush=True)
    print(f"      - Learning rate: {args.learning_rate}", flush=True)
    print(f"      - Save strategy: {'epoch' if args.save_every_epoch else 'no'}", flush=True)

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

    # STEP 9: Create Trainer
    print("\n[STEP 9/10] 🏋️  Creating Trainer...", flush=True)
    collator = partial(custom_data_collator, tokenizer=tok)
    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None,  # Disabled to speed up training
        data_collator=collator,
        callbacks=callbacks,
    )
    print("   ✅ Trainer created", flush=True)
    print(f"   📋 Callbacks attached: {len(trainer.callback_handler.callbacks)} callbacks", flush=True)
    for i, cb in enumerate(trainer.callback_handler.callbacks):
        print(f"      {i+1}. {type(cb).__name__}", flush=True)

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

    # Create optimizer and scheduler
    print("\n   🔧 Creating optimizer and scheduler...", flush=True)
    _ = trainer.create_optimizer_and_scheduler(num_training_steps=training_args.max_steps)
    print(f"   ✅ Optimizer: {type(trainer.optimizer).__name__}", flush=True)
    print(f"   ✅ Scheduler: {type(trainer.lr_scheduler).__name__}", flush=True)

    # Calculate training steps
    train_dl = trainer.get_train_dataloader()
    steps_per_epoch = len(train_dl)
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"\n   📊 TRAINING PLAN (VERIFY THESE NUMBERS!):", flush=True)
    print(f"      - Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"      - Number of epochs: {training_args.num_train_epochs} ⚠️  VERIFY THIS!", flush=True)
    print(f"      - Total update steps: {total_update_steps}", flush=True)
    
    # Verify frozen parameters before training
    if hasattr(model, '_frozen_param_names') and model._frozen_param_names:
        print(f"\n   🧊 PRE-TRAINING FREEZE VERIFICATION:", flush=True)
        print(f"      - Expected frozen params: {len(model._frozen_param_names)}", flush=True)
        still_frozen = sum(1 for name in model._frozen_param_names 
                          for p_name, p in model.named_parameters() 
                          if p_name == name and not p.requires_grad)
        print(f"      - Actually frozen params: {still_frozen}", flush=True)
        if still_frozen == len(model._frozen_param_names):
            print(f"      ✅ All frozen parameters confirmed frozen before training!", flush=True)
        else:
            print(f"      ⚠️  WARNING: {len(model._frozen_param_names) - still_frozen} params became unfrozen!", flush=True)

    # STEP 10: Start training
    print("\n[STEP 10/10] 🚀 STARTING TRAINING...", flush=True)
    print("=" * 80, flush=True)
    print("⚠️  REMINDER: Freezing happened ONCE in Step 5, not each epoch!", flush=True)
    print("⚠️  Frozen parameters should remain frozen throughout all epochs!", flush=True)
    print("=" * 80, flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("\n" + "=" * 80, flush=True)
    print("✅ TRAINING COMPLETED", flush=True)
    print("=" * 80, flush=True)

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
