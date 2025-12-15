import os
import csv
import json
import time
from typing import Optional, Dict, List

import torch
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from .data_preprocessing_mmlu import (
    create_prompt_llama2,
    create_prompt_mistral,
    create_prompt_qwen,
    create_prompt_olmo,
    create_prompt_llama3,
    infer_prompt_format_from_model_id,
    eos_for_model,
    extract_choice_letter,
)

def _get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_mem_alloc": None, "gpu_mem_reserved": None}
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu": props.name,
        "gpu_id": gpu_id,
        "gpu_mem_alloc": torch.cuda.memory_allocated(gpu_id) // 1024**2,
        "gpu_mem_reserved": torch.cuda.memory_reserved(gpu_id) // 1024**2,
    }

def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _append_tsv(path: str, row: dict, field_order: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order, delimiter="\t")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in field_order})

@torch.no_grad()
def evaluate_mmlu(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "validation",
    subjects: Optional[List[str]] = None,
    limit_per_subject: Optional[int] = None,
    limit: Optional[int] = None,
    max_new_tokens: int = 64,
    batch_size: int = 32,
    *,
    progress_bar: bool = True,
    save_jsonl: Optional[str] = True,
    save_tsv: Optional[str] = True,
    run_name: str = "",
    phase: str = "adhoc",
    epoch: int = -1,
    step: int = -1,
    output_dir: str = "",
    debug_print: bool = True,
    tokenization_debug: bool = False,
    few_shot_k: int = 0,
    few_shot_split: str = "auxiliary_train",
) -> Dict[str, float]:

    start_time = time.time()

    # Load Hendrycks Test (MMLU)
    # Default config: full set; subjects correspond to dataset subsets
    ds_dict: DatasetDict = load_dataset('cais/mmlu', 'all')

    # Choose split
    if split not in ds_dict:
        # some versions have 'validation' and 'test' only; fall back if needed
        if split == "test" and "test" in ds_dict:
            pass
        elif "validation" in ds_dict:
            split = "validation"
        else:
            split = list(ds_dict.keys())[0]

    # Collect available subjects from the dataset
    # hendrycks_test exposes multiple configs; when loaded without a config it returns a mapping with features including 'subject'
    ds = ds_dict[split]
    available_subjects = sorted(set(ds["subject"])) if "subject" in ds.column_names else None

    # Filter subjects
    if subjects:
        subjects_set = set(subjects)
        if available_subjects is not None:
            ds = ds.filter(lambda ex: ex["subject"] in subjects_set)
        else:
            # If no subject column, best-effort: leave as-is
            pass

    # Downsample per subject if requested
    if limit_per_subject and "subject" in ds.column_names:
        # group by subject and take first N from each
        shards = []
        by_subject = {}
        for idx, s in enumerate(ds["subject"]):
            by_subject.setdefault(s, []).append(idx)
        for s, idxs in by_subject.items():
            take = idxs[: min(limit_per_subject, len(idxs))]
            shards.append(ds.select(take))
        from datasets import concatenate_datasets
        ds = concatenate_datasets(shards) if shards else ds

    # Global cap irrespective of subjects if requested
    if limit is not None:
        take_n = min(limit, len(ds))
        ds = ds.select(range(take_n))

    total = 0
    correct = 0

    model_id = getattr(model, "name_or_path", None) or getattr(model.config, "_name_or_path", "")
    prompt_format = infer_prompt_format_from_model_id(str(model_id))
    eos_id = eos_for_model(tokenizer, str(model_id))

    # Ensure pad token is set (some tokenizers lack it)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.eval()

    print(
        f"[Eval][MMLU] split={split} n={len(ds)} subjects=ALL batch_size={batch_size}"
        f" few_shot_k={few_shot_k}",
        flush=True,
    )

    few_shot_examples: List[dict] = []

    def _normalize_choices(example) -> List[str]:
        choices = example.get("choices")
        if isinstance(choices, list):
            normalized = choices
        elif isinstance(choices, dict):
            normalized = choices.get("text", [])
        else:
            normalized = []
        if not normalized or len(normalized) < 4:
            alt = [example.get("A"), example.get("B"), example.get("C"), example.get("D")]
            if all(isinstance(x, str) and x for x in alt):
                normalized = alt
        return normalized

    if few_shot_k and few_shot_k > 0:
        if few_shot_split not in ds_dict:
            available = ", ".join(sorted(ds_dict.keys()))
            raise ValueError(
                f"few_shot_split '{few_shot_split}' not found. Available: {available}"
            )
        support_ds = ds_dict[few_shot_split]
        for example in support_ds:
            choices = _normalize_choices(example)
            if not choices or len(choices) < 4:
                continue
            ans = example.get("answer", "")
            if isinstance(ans, int):
                ans_letter = chr(ord("A") + ans)
            else:
                ans_letter = str(ans).strip().upper()
            few_shot_examples.append(
                {
                    "question": example.get("question", ""),
                    "choices": choices,
                    "answer": ans_letter,
                }
            )
            if len(few_shot_examples) >= few_shot_k:
                break
        if len(few_shot_examples) < few_shot_k:
            print(
                f"[Eval][MMLU] Warning: requested {few_shot_k} shots but only found {len(few_shot_examples)}",
                flush=True,
            )

    # Process in batches
    num_batches = (len(ds) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), disable=not progress_bar, desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(ds))
        batch_ds = ds.select(range(start_idx, end_idx))
        
        # Prepare batch prompts
        batch_prompts = []
        batch_gold_answers = []
        
        for ex in batch_ds:
            q = ex["question"]
            choices = ex["choices"] if isinstance(ex["choices"], list) else ex.get("choices", {}).get("text", [])
            
            # Handle different choice formats
            if not choices or any(c is None for c in choices):
                alt = [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]
                if all(isinstance(x, str) and x for x in alt):
                    choices = alt
            if not choices or len(choices) < 4:
                continue

            ans = ex.get("answer", "")
            if isinstance(ans, int):
                gold_letter = chr(ord("A") + ans)
            else:
                gold_letter = str(ans).strip().upper()

            # Create prompt based on model format
            if prompt_format == "llama2":
                prompt = create_prompt_llama2(
                    q, choices, use_instruction=True, few_shot_examples=few_shot_examples
                )
            elif prompt_format == "llama3":
                prompt = create_prompt_llama3(
                    tokenizer, q, choices, use_instruction=True, few_shot_examples=few_shot_examples
                )
            elif prompt_format == "mistral":
                prompt = create_prompt_mistral(
                    q, choices, use_instruction=True, few_shot_examples=few_shot_examples
                )
            elif prompt_format == "olmo":
                prompt = create_prompt_olmo(
                    q, choices, use_instruction=True, few_shot_examples=few_shot_examples
                )
            else:
                prompt = create_prompt_qwen(
                    tokenizer, q, choices, use_instruction=True, few_shot_examples=few_shot_examples
                )

            batch_prompts.append(prompt)
            batch_gold_answers.append(gold_letter)

        if not batch_prompts:
            continue

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(model.device)

        if tokenization_debug and batch_idx == 0:
            print(f"\n[Eval][Batch {batch_idx}] --- TOKENIZATION DEBUG ---")
            print("First prompt:\n", batch_prompts[0])
            print("Input IDs shape:", inputs["input_ids"].shape)

        # Generate answers for the batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=eos_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode generated text
        generated_texts = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )

        # Extract answers and compute accuracy
        for i, (generated_text, gold_letter) in enumerate(zip(generated_texts, batch_gold_answers)):
            # Extract the predicted letter from generated text
            pred_letter = extract_choice_letter(generated_text)
            
            if pred_letter and gold_letter and pred_letter == gold_letter:
                correct += 1
            total += 1

            if debug_print and batch_idx == 0 and i < 3:  # Show first few examples
                print(f"\n[Eval][Batch {batch_idx}, Item {i}] Gold={gold_letter} Pred={pred_letter}")
                print("   ---- PROMPT ----\n", batch_prompts[i])
                print("   ---- GENERATED ----\n", generated_text)


    acc = 100.0 * correct / max(1, total)

    # restore
    model.config.use_cache = prev_cache
    tokenizer.padding_side = "right"  # Reset to default

    print(f"[Eval][MMLU] Finished {split}: ACC={acc:.2f}% n={total}", flush=True)

    if save_jsonl or save_tsv:
        rec = {
            "timestamp": time.time(),
            "run_name": run_name,
            "model_name": str(model_id),
            "phase": phase,
            "epoch": int(epoch),
            "step": int(step),
            "split": split,
            "acc": float(acc),
            "n": int(total),
            "max_new_tokens": int(max_new_tokens),
            "batch_size": int(batch_size),
            "output_dir": output_dir,
        }
        if save_jsonl:
            _append_jsonl(save_jsonl, rec)
        if save_tsv:
            _append_tsv(
                save_tsv,
                rec,
                field_order=[
                    "timestamp","run_name","model_name","phase","epoch","step",
                    "split","acc","n","max_new_tokens","batch_size","output_dir",
                ],
            )

    elapsed = time.time() - start_time
    metrics = {
        "acc": acc,
        "n": total,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "throughput_qps": total / elapsed if elapsed > 0 else None,
        "batch_size": batch_size,
    }
    metrics.update(_get_gpu_info())
    return metrics
