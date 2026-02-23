import os
import re
import csv
import json
import time
from typing import Optional, Dict, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from .data_preprocessing_squad import (
    build_prompt,
    infer_prompt_format_from_model_id,
)
from codes.hotpotqa.data_preprocessing_hotpotqa import eos_for_model

_ARTICLES = {"a", "an", "the"}

# Number normalization map (0-20 for common SQuAD answers)
_NUM_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
}

# -------------------- per-example CSV logging (Option A: one big file) --------------------
_EXAMPLE_FIELDS = [
    "checkpoint",
    "idx",
    "question",
    "gold",
    "pred",
    "em",
    "f1",
    "raw_gen",
    "prompt",
    "gen_only",
    "full_text",
]

def _append_examples_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_EXAMPLE_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in _EXAMPLE_FIELDS})

def _normalize_text(s: str) -> str:
    s = str(s).lower()

    # digits -> words (0-20)
    tokens = [ _NUM_MAP.get(tok, tok) for tok in s.split() ]
    s = " ".join(tokens)

    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)

    # whitespace
    tokens = [t for t in s.split() if t and t not in _ARTICLES]
    return " ".join(tokens)

def _em_and_f1(prediction: str, ground_truth: str) -> Tuple[float, float]:
    p = _normalize_text(prediction)
    g = _normalize_text(ground_truth)
    em = 1.0 if p == g else 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens and not g_tokens:
        return em, 1.0
    if not p_tokens or not g_tokens:
        return em, 0.0
    common = {}
    for t in p_tokens:
        common[t] = min(common.get(t, 0) + 1, g_tokens.count(t)) if t in g_tokens else common.get(t, 0)
    num_same = sum(common.values())
    if num_same == 0:
        return em, 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1

def _extract_final_answer(text: str) -> str:
    """
    Robust answer extractor for SQuAD.
    Handles unicode quotes, uppercase, extra punctuation, and explanations.
    """
    import unicodedata
    
    if not text:
        return ""
    
    # Normalize unicode (fix ”, weird symbols, etc.)
    text = unicodedata.normalize("NFKD", str(text))
    
    # Remove end-of-text markers and stop at EOS
    # Handle cases where model generates <|end_of_text|><|begin_of_text|> (token corruption)
    # But keep content after begin_of_text if it contains numbered lists or Answer:
    if "<|end_of_text|>" in text:
        parts = text.split("<|end_of_text|>")
        # If we have corruption tokens, try to extract from after begin_of_text
        if len(parts) > 1 and "<|begin_of_text|>" in parts[1]:
            # Extract content after <|begin_of_text|>
            after_begin = parts[1].split("<|begin_of_text|>", 1)
            if len(after_begin) > 1:
                # Use content after begin_of_text (might have numbered lists)
                text = after_begin[1].strip()
            else:
                # Fallback to before end_of_text
                text = parts[0].strip()
        else:
            text = parts[0].strip()
    
    text = text.split("<|eot_id|>")[0].strip()
    if not text:
        return ""
    
    # Find ALL Answer: occurrences (take LAST one in case of repetition)
    # Handle both "Answer: X" and "Answer:\nX" formats
    matches = re.findall(
        r"Answer\s*:\s*(.+?)(?=\n|$|because|The correct|This is)", text, flags=re.IGNORECASE | re.DOTALL
    )
    
    if matches:
        ans = matches[-1].strip()  # Take LAST answer
        # If answer contains quotes, extract the quoted part
        quoted_match = re.search(r'["\']([^"\']+)["\']', ans)
        if quoted_match:
            ans = quoted_match.group(1)
    else:
        # Fallback: Try "The answer is X" pattern
        answer_is_patterns = [
            r"the answer is\s+(.+?)(?:\.|$|\n)",
            r"the answer is\s+['\"](.+?)['\"]",
        ]
        for pattern in answer_is_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ans = match.group(1).strip()
                break
        else:
            # Fallback: Extract from numbered lists (e.g., "1. Denver Broncos")
            # This handles cases where base model generates lists instead of "Answer:"
            numbered_match = re.search(r'^\s*1\.\s*(.+?)(?:\n|$|2\.)', text, re.MULTILINE | re.IGNORECASE)
            if numbered_match:
                ans = numbered_match.group(1).strip()
                # Clean up common suffixes
                ans = re.sub(r'\s*\(.*?\)\s*$', '', ans)  # Remove trailing parentheses
            else:
                # Last resort: first non-empty line that looks like an answer
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                for line in lines[:3]:  # Check first 3 lines
                    # Skip obvious junk
                    if any(skip in line.lower() for skip in [
                        "question:", "context:", "user:", "assistant:", "system:",
                        "step", "reasoning", "explanation", "based on", "from the"
                    ]):
                        continue
                    # Skip if too long (likely not an answer)
                    if len(line.split()) > 15:
                        continue
                    ans = line
                    break
                else:
                    # Last resort: last non-empty line
                    if lines:
                        ans = lines[-1]
                    else:
                        return ""
    
    # Remove explanations if model keeps talking after answer
    ans = ans.split("\n")[0]  # Take first line only
    ans = ans.split(" because")[0]  # Stop at "because" explanations
    ans = ans.split("The correct")[0]  # Stop at "The correct option is..."
    ans = ans.split("This is")[0]  # Stop at "This is because..."
    
    # Strip punctuation and quotes (handles ”, ", ', etc.)
    quote_chars = " .,:;\"'""" + "[]()"
    ans = ans.strip().strip(quote_chars)
    
    # Normalize case for EM matching (SQuAD is case-insensitive)
    return ans.lower()

def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _append_tsv(path: str, row: dict, field_order: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order, delimiter="\t")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in field_order})

@torch.no_grad()
def evaluate_squad(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "validation",
    limit: Optional[int] = None,
    max_new_tokens: int = 12,  # Short answers only (SQuAD answers are typically 1-5 words)
batch_size: int = 128,
    *,
    progress_bar: bool = True,
    save_jsonl: Optional[str] = None,
    save_tsv: Optional[str] = None,
    run_name: str = "",
    phase: str = "adhoc",
    epoch: int = -1,
    step: int = -1,
    output_dir: str = "",
    debug_print: bool = False,
    tokenization_debug: bool = False,
    # Option A: one big file, appends rows across checkpoints
    save_examples_csv: Optional[str] = None,
) -> Dict[str, float]:

    start_time = time.time()

    ds = load_dataset("squad")[split]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

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

    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    # Convert dataset to list for batching
    ds_list = list(ds)
    total_samples = len(ds_list)
    
    # Process in batches
    iterator = tqdm(range(0, total_samples, batch_size), disable=not progress_bar, desc="Evaluating")
    for batch_start in iterator:
        batch_end = min(batch_start + batch_size, total_samples)
        batch_examples = ds_list[batch_start:batch_end]
        batch_size_actual = len(batch_examples)
        
        # Prepare batch data
        batch_prompts = []
        batch_golds = []
        batch_questions = []
        batch_contexts = []
        
        for ex in batch_examples:
            q = ex["question"]
            ctx = ex.get("context", "")
            answers = ex.get("answers", {}).get("text", [])
            gold = (answers[0] if answers else "").strip()
            
            prompt = build_prompt(tokenizer, q, context=ctx, prompt_format=prompt_format, use_instruction=True)
            batch_prompts.append(prompt)
            batch_golds.append(gold)
            batch_questions.append(q)
            batch_contexts.append(ctx)
        
        # Tokenize batch with padding
        # Use large max_length to preserve full context (no truncation for SQuAD)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,  # Don't truncate - preserve full context
            max_length=None,  # No limit - preserve full context
        ).to(model.device)
        
        if tokenization_debug and batch_start == 0:
            print(f"\n[Eval][Batch 0] --- TOKENIZATION DEBUG ---")
            print("Batch size:", batch_size_actual)
            print("Input IDs shape:", inputs["input_ids"].shape)
            print("First prompt (decoded):", tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
        
        # Generate in batch (already in @torch.no_grad() context from decorator)
        outputs = model.generate(
            **inputs,
            max_new_tokens=12,  # SQuAD answers are short (1-5 words)
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id for padding too
        )
        
        # Process each output in the batch
        prompt_lens = inputs["attention_mask"].sum(dim=1).cpu().tolist()
        
        for batch_idx, (ex, prompt, gold, q, ctx, prompt_len) in enumerate(zip(
            batch_examples, batch_prompts, batch_golds, batch_questions, batch_contexts, prompt_lens
        )):
            idx = batch_start + batch_idx
            
            # Extract generated tokens (skip prompt)
            gen_only_ids = outputs[batch_idx][prompt_len:]
            gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=False)
            
            pred = _extract_final_answer(gen_text)
            em, f1 = _em_and_f1(pred, gold)
            
            em_sum += em
            f1_sum += f1
            total += 1
            
            if save_examples_csv:
                full_text = prompt + gen_text
                _append_examples_csv(save_examples_csv, {
                    "checkpoint": run_name,
                    "idx": idx,
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "em": em,
                    "f1": f1,
                    "raw_gen": gen_text,
                    "prompt": prompt,
                    "gen_only": gen_text,
                    "full_text": full_text,
                })
            
            if debug_print:
                print(f"\n[Eval][{idx}] Gold={gold} Pred={pred} EM={em:.3f} F1={f1:.3f}", flush=True)

    # Keep canonical metrics as fractions in [0,1] (consistent with HotpotQA evaluator),
    # and compute percent only for printing / convenience.
    em_avg = em_sum / max(1, total)
    f1_avg = f1_sum / max(1, total)
    em_pct = 100.0 * em_avg
    f1_pct = 100.0 * f1_avg

    # restore
    model.config.use_cache = prev_cache
    tokenizer.padding_side = "right"  # Reset to default

    print(
        f"[Eval][SQuAD] Finished {split}: EM={em_pct:.2f}% F1={f1_pct:.2f}% n={total}",
        flush=True,
    )

    if save_jsonl or save_tsv:
        rec = {
            "timestamp": time.time(),
            "run_name": run_name,
            "model_name": str(model_id),
            "phase": phase,
            "epoch": int(epoch),
            "step": int(step),
            "split": split,
            # Store both fraction and percent for robustness across downstream scripts.
            "em": float(em_avg),
            "f1": float(f1_avg),
            "em_pct": float(em_pct),
            "f1_pct": float(f1_pct),
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
                    "split",
                    "em","f1","em_pct","f1_pct",
                    "n","max_new_tokens","batch_size","output_dir",
                ],
            )

    elapsed = time.time() - start_time
    metrics = {
        # canonical outputs: fractions in [0,1]
        "em": em_avg,
        "f1": f1_avg,
        # convenience: percent versions (0-100)
        "em_pct": em_pct,
        "f1_pct": f1_pct,
        "n": total,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "throughput_qps": total / elapsed if elapsed > 0 else None,
        "batch_size": batch_size,
    }
    # Add GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(gpu_id)
            metrics.update({
                "gpu": props.name,
                "gpu_id": gpu_id,
                "gpu_mem_alloc": torch.cuda.memory_allocated(gpu_id) // 1024**2,
                "gpu_mem_reserved": torch.cuda.memory_reserved(gpu_id) // 1024**2,
            })
    except Exception:
        pass
    return metrics

