import os
import json
import glob
import shutil
import datetime
from functools import partial
from collections import Counter
import string

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging

from peft import LoraConfig, get_peft_model, TaskType

from .data_preprocessing_hotpotqa import (
    preprocess_dataset,
    custom_data_collator,
    infer_prompt_format_from_model_id,
)
from .eval_hotpotqa import evaluate_hotpotqa

from codes.utils.args import parse_args
from codes.utils.model_saving import (
    SavePeftModelCallback,
    concise_lora_filename,
    concise_full_filename,
)

import wandb

hf_logging.enable_progress_bar()

# If you want, keep these here; otherwise I'd move them to your bash script
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")


# =========================================================
# GPU / Logging Helpers
# =========================================================
def get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_id": None, "total_mem_GB": None, "mem_alloc_MB": None, "mem_reserved_MB": None}
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
        if logs and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] Loss: {logs['loss']:.6f}", flush=True)


class EMF1PerEpochCallback(TrainerCallback):
    """
    Kept exactly as your functionality, but grouped here.
    """
    def __init__(
        self,
        tokenizer,
        run_eval: bool = False,
        split: str = "validation",
        limit=None,
        max_new_tokens: int = 256,
        log_jsonl=None,
        log_tsv=None,
        dataset="",
        model="",
    ):
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

        metrics = evaluate_hotpotqa(
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
            **gpu_info,
        }

        print(
            f"[Downstream] epoch={record['epoch']} HotpotQA {self.split} "
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


# =========================================================
# EM/F1 Helpers (kept functional, slightly cleaner)
# =========================================================
def _extract_answer_start(labels):
    for i, x in enumerate(labels):
        if x != -100:
            return i
    return None


def _clean_answer_text(text: str) -> str:
    """
    For display + for your simple EM logic.
    - strips
    - if "Answer:" exists, keeps content after it
    - takes only the first line
    """
    if text is None:
        return ""
    text = text.strip()
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    text = text.splitlines()[0]
    return text.strip()


def _normalize_text(s: str) -> str:
    """
    For EM/F1 normalization. Keeps your current behavior.
    """
    s = (s or "").lower().strip()
    if "answer:" in s:
        s = s.split("answer:", 1)[1].strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize_text(pred) == _normalize_text(gold)


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# =========================================================
# Post-train Sample Debug (kept, but cleaned)
# =========================================================
def _configure_generation_for_debug(model, tok):
    """
    Keep generation consistent and avoid gradient checkpoint/cache mismatch warnings.
    """
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # Prefer llama3 chat end token if it exists
    eot_id = None
    try:
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None

    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = (eot_id if (eot_id is not None and eot_id != tok.eos_token_id) else tok.eos_token_id)

    return eot_id


@torch.no_grad()
def debug_print_samples_after_training(model, tok, raw_ds, tokenized_ds, output_dir: str, n: int = 3):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hotpot_posttrain_samples.txt")

    idxs = list(range(min(n, len(tokenized_ds))))
    eot_id = _configure_generation_for_debug(model, tok)

    with open(out_path, "w") as f:
        for k, i in enumerate(idxs, start=1):
            raw_ex = raw_ds[i]
            ex = tokenized_ds[i]
            input_ids = ex["input_ids"]
            labels = ex["labels"]

            ans_start = _extract_answer_start(labels)
            if ans_start is None:
                msg = f"[DEBUG] sample {k} (idx={i}): labels all -100, skipping\n"
                print(msg, flush=True)
                f.write(msg)
                continue

            prompt_ids_list = input_ids[:ans_start]
            gold_ids_list = input_ids[ans_start:]

            prompt_text = tok.decode(prompt_ids_list, skip_special_tokens=False)
            gold_text = tok.decode(gold_ids_list, skip_special_tokens=False)

            prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=model.device)
            attn = torch.ones_like(prompt_ids)

            gen = model.generate(
                input_ids=prompt_ids,
                attention_mask=attn,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=(eot_id if eot_id is not None else tok.eos_token_id),
                pad_token_id=tok.pad_token_id,
            )

            pred_completion = tok.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=False)

            gold_short = _clean_answer_text("Answer: " + str(raw_ex.get("answer", "")))
            pred_short = _clean_answer_text(pred_completion)

            em = _exact_match(pred_short, gold_short)
            f1 = _f1_score(pred_short, gold_short)

            block = (
                f"\n================ POST-TRAIN SAMPLE {k} (idx={i}) ================\n"
                f"[RAW Q] {raw_ex.get('question','')}\n"
                f"[RAW GOLD SHORT] {raw_ex.get('answer','')}\n"
                f"[LEN] prompt={len(prompt_ids_list)} gold={len(gold_ids_list)} total={len(input_ids)}\n\n"
                f"--- INPUT PROMPT (FULL) ---\n{prompt_text}\n\n"
                f"--- GOLD (SUPERVISED TEXT) ---\n{gold_text}\n\n"
                f"--- PRED (completion from prompt-only) ---\n{pred_completion}\n\n"
                f"--- SANITY ---\n"
                f"pred_short = {pred_short}\n"
                f"gold_short = {gold_short}\n"
                f"EM(normalized) = {em}\n"
                f"F1(normalized) = {f1:.4f}\n"
                f"===============================================================\n"
            )

            print(block, flush=True)
            f.write(block)

    print(f"[DEBUG] Saved post-train samples to: {out_path}", flush=True)


# =========================================================
# Model / Tokenizer
# =========================================================
def load_model_and_tokenizer(model_id: str, use_lora: bool, freeze_layers=None, freeze_q_layers=None, freeze_k_layers=None, freeze_v_layers=None):
    freeze_layers = freeze_layers or []
    freeze_q_layers = freeze_q_layers or []
    freeze_k_layers = freeze_k_layers or []
    freeze_v_layers = freeze_v_layers or []

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        token=HF_TOKEN
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        dtype=torch.bfloat16,   # torch_dtype deprecated -> dtype
        trust_remote_code=True,
        token=HF_TOKEN
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    # Freeze base transformer layers
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
    
    # --------------------------
    # Freeze ONLY q/k/v projections in selected layers (base weights)
    # Works for LLaMA-family naming (q_proj/k_proj/v_proj).
    # --------------------------
    if freeze_q_layers or freeze_k_layers or freeze_v_layers:
        qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
        print(f"🧊 Freezing projections: Q={sorted(qset)} K={sorted(kset)} V={sorted(vset)}", flush=True)

        for name, p in model.named_parameters():
            # Match layer index in parameter name (common patterns: ".layers.{i}.")
            hit_layer = None
            for i in (qset | kset | vset):
                if f".layers.{i}." in name:
                    hit_layer = i
                    break
            if hit_layer is None:
                continue

            # Freeze selectively by projection
            if hit_layer in qset and ".q_proj." in name:
                p.requires_grad = False
            if hit_layer in kset and ".k_proj." in name:
                p.requires_grad = False
            if hit_layer in vset and ".v_proj." in name:
                p.requires_grad = False


    # LoRA inject q/k/v and disable LoRA params in frozen layers
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

        if freeze_layers:
            for name, p in model.named_parameters():
                if "lora_" not in name:
                    continue
                for layer_idx in freeze_layers:
                    if f".layers.{layer_idx}." in name:
                        p.requires_grad = False
                        break
        # Also disable LoRA params for q/k/v projections in specified layers
        if freeze_q_layers or freeze_k_layers or freeze_v_layers:
            qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)

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
                if layer_hit in kset and "k_proj" in name:
                    p.requires_grad = False
                if layer_hit in vset and "v_proj" in name:
                    p.requires_grad = False


        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    return model, tok


# =========================================================
# Dataset loading (subset caching preserved)
# =========================================================
def load_hotpotqa_with_optional_subset_cache(args):
    subset_dir = args.subset_save_dir
    state_path = os.path.join(subset_dir, "state.json") if subset_dir else None

    if subset_dir and os.path.exists(state_path):
        from datasets import load_from_disk
        print(f">>> Loading HotpotQA subset from {subset_dir}", flush=True)
        ds = load_from_disk(subset_dir)
        return ds["train"], ds["validation"]

    ds = load_dataset("hotpot_qa", "distractor")
    train_full, val_ds = ds["train"], ds["validation"]

    subset_size = min(args.subset_train_size, len(train_full))
    print(f">>> Sampling subset of {subset_size} from {len(train_full)} with seed {args.subset_seed}", flush=True)
    train_full = train_full.shuffle(seed=args.subset_seed).select(range(subset_size))

    if subset_dir:
        from datasets import DatasetDict
        os.makedirs(subset_dir, exist_ok=True)
        print(f">>> Saving HotpotQA subset to {subset_dir}", flush=True)
        DatasetDict({"train": train_full, "validation": val_ds}).save_to_disk(subset_dir)

    return train_full, val_ds


# =========================================================
# Baseline saving (epoch-0) preserved
# =========================================================
def save_epoch0_baseline_if_needed(args, model, tok):
    if not (getattr(args, "save_every_epoch", False) or getattr(args, "save_npy", False)):
        return

    base_dir = os.path.join(args.output_dir, "epoch_weights")
    os.makedirs(base_dir, exist_ok=True)
    save_dir = os.path.join(base_dir, "checkpoint-epoch-0")

    if os.path.exists(save_dir):
        print(f">>> Epoch-0 already exists at {save_dir}, skipping baseline save", flush=True)
        return

    print(f">>> Saving baseline as epoch-0 to {save_dir}", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    import torch as _torch
    _torch.save(args, os.path.join(save_dir, "training_args.bin"))

    if not getattr(args, "save_npy", False):
        return

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

    print(f"✅ [baseline] Saved {count} tensors (float16) to: {npy_dir}", flush=True)


def cleanup_checkpoints(args):
    # Delete Hugging Face default checkpoints
    for path in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        print(f"🗑️ Removing default checkpoint: {path}", flush=True)
        shutil.rmtree(path, ignore_errors=True)

    # Delete final_model if it exists (align with your IMDB behavior)
    final_model_path = os.path.join(args.output_dir, "final_model")
    if os.path.exists(final_model_path):
        print(f"🗑️ Removing final model folder: {final_model_path}", flush=True)
        shutil.rmtree(final_model_path, ignore_errors=True)


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    # Debug prints you were using
    import sys
    print("[DEBUG] sys.argv:", sys.argv, flush=True)
    print("[DEBUG] has debug_hotpot?", hasattr(args, "debug_hotpot"), flush=True)
    print("[DEBUG] args.debug_hotpot =", getattr(args, "debug_hotpot", "MISSING"), flush=True)
    print("[DEBUG] args keys =", sorted(vars(args).keys()), flush=True)

    # W&B (kept as-is)
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"hotpotqa-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )

    # Dataset
    train_full, val_ds = load_hotpotqa_with_optional_subset_cache(args)

    split = train_full.train_test_split(test_size=0.05, seed=args.subset_seed)
    train_ds, dev_ds = split["train"], split["test"]
    print(f"Train {len(train_ds)} | Dev {len(dev_ds)} | Val {len(val_ds)}", flush=True)

    # Model + tokenizer
    model, tok = load_model_and_tokenizer(
        args.model_name,
        args.use_lora,
        freeze_layers=args.freeze_layers,
        freeze_q_layers=getattr(args, "freeze_q_layers", []),
        freeze_k_layers=getattr(args, "freeze_k_layers", []),
        freeze_v_layers=getattr(args, "freeze_v_layers", []),
    )

    # Prompt format and evidence mode
    pf = infer_prompt_format_from_model_id(args.model_name)
    evidence_mode = getattr(args, "hotpot_evidence", "supporting")

    tokenized_train = train_ds.map(
        lambda ex: preprocess_dataset(
            ex, tok,
            max_len=1024,
            prompt_format=pf,
            is_train=True,
            evidence_mode=evidence_mode,
        ),
        remove_columns=train_ds.column_names
    )

    tokenized_val = dev_ds.map(
        lambda ex: preprocess_dataset(
            ex, tok,
            max_len=1024,
            prompt_format=pf,
            is_train=False,
            evidence_mode=evidence_mode,
        ),
        remove_columns=dev_ds.column_names
    )

    # Debug one tokenized sample (kept)
    if getattr(args, "debug_hotpot", False):
        ex = tokenized_train[0]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        first_answer_idx = next((i for i, x in enumerate(labels) if x != -100), None)

        print("\n================ HOTPOT FINETUNE DEBUG ================")
        print(f"hotpot_evidence = {evidence_mode}")
        print(f"input_len = {len(input_ids)}")

        print("\n--- FULL DECODE (truncated) ---")
        print(tok.decode(input_ids[:800], skip_special_tokens=False))

        if first_answer_idx is not None:
            print("\n--- PROMPT TAIL (before answer) ---")
            print(tok.decode(input_ids[max(0, first_answer_idx - 300):first_answer_idx], skip_special_tokens=False))

            print("\n--- ANSWER HEAD (supervised) ---")
            print(tok.decode(input_ids[first_answer_idx:first_answer_idx + 200], skip_special_tokens=False))
        else:
            print("❌ ERROR: No supervised tokens found (labels all -100)")

        masked_ok = all(x == -100 for x in labels[:first_answer_idx]) if first_answer_idx is not None else False
        supervised_ok = all(x != -100 for x in labels[first_answer_idx:]) if first_answer_idx is not None else False
        print(f"\nMask check → masked_ok={masked_ok}, supervised_ok={supervised_ok}")
        print("=======================================================\n", flush=True)

    # Training args
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./HotpotQA/logs/{timestamp}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no",
        save_strategy="epoch" if args.save_every_epoch else "no",
        load_best_model_at_end=False,
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

    callbacks = [LossDebugCallback()]
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

    # Save baseline (epoch 0)
    save_epoch0_baseline_if_needed(args, model, tok)

    # Print training plan
    steps_per_epoch = len(trainer.get_train_dataloader())
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(
        f">>> Training plan: steps_per_epoch={steps_per_epoch} x epochs={training_args.num_train_epochs} = total_updates={total_update_steps}",
        flush=True
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Post-train samples (kept)
    if getattr(args, "debug_hotpot", False):
        debug_print_samples_after_training(
            model=trainer.model,
            tok=tok,
            raw_ds=train_ds,
            tokenized_ds=tokenized_train,
            output_dir=args.output_dir,
            n=3
        )

    # Save final then delete (kept)
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    cleanup_checkpoints(args)

    print("[Training] Final evaluation disabled. Will evaluate manually later.", flush=True)


if __name__ == "__main__":
    main()