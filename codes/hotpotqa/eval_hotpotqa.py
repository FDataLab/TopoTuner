import os
import re
import csv
import json
import time
import string
from typing import Optional, Dict, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from .data_preprocessing_hotpotqa import (
    create_prompt_llama2,
    create_prompt_mistral,
    create_prompt_qwen,
    create_prompt_olmo,
    create_prompt_llama3,
    infer_prompt_format_from_model_id,
    eos_for_model,
)

# -------------------- GPU utils --------------------
def _get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_mem_alloc": None, "gpu_mem_reserved": None}
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu": props.name,
        "gpu_id": gpu_id,
        "gpu_mem_alloc": torch.cuda.memory_allocated(gpu_id) // 1024**2,   # MB
        "gpu_mem_reserved": torch.cuda.memory_reserved(gpu_id) // 1024**2  # MB
    }

# -------------------- text normalization (HotpotQA reference style) --------------------
# Matches hotpot_evaluate_plus.py normalization for fair comparison
def _normalize_text(s: str) -> str:
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    # Apply in same order as reference: lower -> remove_punc -> remove_articles -> fix_whitespace
    return white_space_fix(remove_articles(remove_punc(lower(s))))

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

# -------------------- small utils --------------------
def _extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model output.
    Always takes the FIRST clean answer and stops at repetitive patterns.
    """
    # Repetitive patterns that indicate we should stop processing
    repetitive_patterns = [
        r'#\+#',  # Pattern like "#+# # system"
        r'systematically\s+user',
        r'sistema\w*\s+user',
        r'preferredStyle',
        r'1st\s+century',
        r'\d+st\s+century',
        r'definitely\s+systematically',
        r'^\s*\d+st\s+',  # Lines starting with "1st ", "2nd ", etc.
    ]
    
    # Find FIRST "Answer:" occurrence
    first_answer_idx = text.find("Answer:")
    if first_answer_idx == -1:
        # No "Answer:" found, return empty
        return ""
    
    # Extract text starting from first "Answer:"
    text_from_answer = text[first_answer_idx + 7:].strip()  # +7 to skip "Answer:"
    
    # Find where to stop (first repetitive pattern or reasonable boundary)
    stop_idx = len(text_from_answer)
    for pattern in repetitive_patterns:
        match = re.search(pattern, text_from_answer, re.IGNORECASE | re.MULTILINE)
        if match:
            stop_idx = min(stop_idx, match.start())
    
    # Also stop at second "Answer:" if it appears (indicates repetition)
    second_answer = text_from_answer.find("Answer:", 1)
    if second_answer != -1:
        stop_idx = min(stop_idx, second_answer)
    
    # Extract the answer portion
    answer_text = text_from_answer[:stop_idx].strip()
    
    # Take only the first line (stop at newline)
    first_line = answer_text.split('\n')[0].strip()
    
    # Clean up the answer - detect and remove trailing junk patterns
    # Pattern 1: Remove ".1st", ".2nd", etc. and everything after
    first_line = re.sub(r'\.\d+st.*$', '', first_line, flags=re.IGNORECASE).strip()
    first_line = re.sub(r'\.\d+.*$', '', first_line).strip()  # Remove ".1", ".2", etc.
    first_line = re.sub(r'\.#.*$', '', first_line).strip()  # Remove ".#" patterns
    
    # Pattern 2: Remove patterns like "Famemy", "Famemyประโยค", "Famesystem" etc.
    # These are malformed concatenations where junk got attached to the answer
    # Look for lowercase letters or non-ASCII chars directly after a word
    first_line = re.sub(r'([A-Z][a-zA-Z\s]+?)(?:[a-z]{1,3}(?:system|my|ประโยค|system\d+)).*$', r'\1', first_line, flags=re.IGNORECASE).strip()
    
    # Pattern 3: Remove standalone "system", "my", unicode patterns
    first_line = re.sub(r'(?:my|system|ประโยค|system\d+).*$', '', first_line, flags=re.IGNORECASE).strip()
    
    # Pattern 4: Remove any remaining repetitive patterns
    for pattern in repetitive_patterns:
        first_line = re.sub(pattern, '', first_line, flags=re.IGNORECASE).strip()
    
    # Remove XML tags if present
    first_line = re.sub(r'<final-answer>.*?</final-answer>', '', first_line, flags=re.S).strip()
    first_line = re.sub(r'</final-answer>.*$', '', first_line).strip()
    first_line = re.sub(r'final-answer.*$', '', first_line).strip()
    
    # Remove brackets
    first_line = re.sub(r'\[.*?\]', '', first_line).strip()
    
    # Pattern 5: Handle malformed concatenations like "Famemy", "Wordsystem"
    # Strategy: Find junk and remove it, preserving the word it's attached to
    
    # Look for junk patterns: "my", "system", etc. (case insensitive)
    # Try to find them attached to words first
    junk_attached_pattern = r'([A-Za-z]+)(my|system|ประโยค|system\d+)'
    attached_match = re.search(junk_attached_pattern, first_line, re.IGNORECASE)
    if attached_match:
        # Found junk attached to a word (e.g., "Famemy")
        word_part = attached_match.group(1)  # "Fame" from "Famemy"
        junk_part = attached_match.group(2)   # "my" from "Famemy"
        
        # Replace "word+junk" with just "word" in the string
        # Find the full match and replace it
        full_match = attached_match.group(0)  # "Famemy"
        first_line = first_line.replace(full_match, word_part, 1)  # Replace first occurrence only
        
        # Now remove any remaining junk patterns
        first_line = re.sub(r'(?:my|system|ประโยค|system\d+).*$', '', first_line, flags=re.IGNORECASE).strip()
    else:
        # No attached junk, look for standalone junk patterns
        junk_pattern = r'(?:my|system|ประโยค|system\d+)'
        junk_match = re.search(junk_pattern, first_line, re.IGNORECASE)
        if junk_match:
            # Extract everything before the junk
            first_line = first_line[:junk_match.start()].strip()
    
    # Final cleanup: remove trailing period if it's standalone (not part of abbreviation)
    # But be careful - "U.S." should keep the period
    # Simple heuristic: if period is followed by nothing or whitespace, and answer is > 1 char, remove it
    if len(first_line) > 1 and first_line.endswith('.'):
        # Check if it's likely an abbreviation (short word with periods)
        if not re.match(r'^[A-Z]\.$', first_line) and not re.match(r'^([A-Z]\.)+$', first_line):
            first_line = first_line.rstrip('.')
    
    return first_line

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

# -------------------- main eval --------------------
@torch.no_grad()
def evaluate_hotpotqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "validation",
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    *,
    config: str = "distractor",
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
) -> Dict[str, float]:

    start_time = time.time()

    ds = load_dataset("hotpot_qa", config)[split]
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

    print(f"[Eval][HotpotQA] Starting evaluation on split={split}, n={len(ds)}", flush=True)

    iterator = tqdm(ds, disable=not progress_bar)
    for idx, ex in enumerate(iterator):
        q = ex["question"]
        gold = str(ex.get("answer", "")).strip()

        # Build user message with lightweight context if present
        context = None
        if "context" in ex and isinstance(ex["context"], list):
            parts = []
            for title, sents in ex["context"]:
                parts.append(f"{title}: {' '.join(sents)}")
            context = "\n".join(parts)

        if prompt_format == "llama2":
            prompt = create_prompt_llama2(q, context=context, use_instruction=True)
        elif prompt_format == "llama3":
            prompt = create_prompt_llama3(tokenizer, q, context=context, use_instruction=True)
        elif prompt_format == "mistral":
            prompt = create_prompt_mistral(q, context=context, use_instruction=True)
        elif prompt_format == "olmo":
            prompt = create_prompt_olmo(q, context=context, use_instruction=True)
        else:
            prompt = create_prompt_qwen(tokenizer, q, context=context, use_instruction=True)

        # Match IMDB tokenization behavior: truncate to a reasonable context length
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        if tokenization_debug:
            print(f"\n[Eval][{idx}] --- TOKENIZATION DEBUG ---")
            print("Prompt text:\n", prompt)
            print("Input IDs shape:", inputs["input_ids"].shape)
            print("Decoded back from IDs:\n", tokenizer.decode(inputs["input_ids"][0]))

        out = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, 128),  # Reduce max tokens to prevent long loops
            do_sample=False,
            temperature=1.0,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.3,  # Stronger penalty for repetition
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )

        # Align decoding with IMDB evaluator defaults
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        prompt_len = inputs["input_ids"].shape[1]
        gen_only_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        
        # Early stopping: truncate at first repetition pattern
        if "Answer:" in gen_text:
            # Find first Answer: and take only the first occurrence
            answer_parts = gen_text.split("Answer:", 1)
            if len(answer_parts) > 1:
                first_answer = "Answer:" + answer_parts[1]
                # Stop at first repetition or after reasonable length
                lines = first_answer.split('\n')
                clean_lines = []
                seen_lines = set()
                for line in lines[:10]:  # Take first 10 lines max
                    line_stripped = line.strip()
                    # Stop if we see exact repetition
                    if line_stripped in seen_lines and len(seen_lines) > 0:
                        break
                    if line_stripped:
                        seen_lines.add(line_stripped)
                    clean_lines.append(line)
                    # Stop if we see "</final-answer>" tag
                    if "</final-answer>" in line:
                        break
                gen_text = '\n'.join(clean_lines)

        pred = _extract_final_answer(gen_text)
        em, f1 = _em_and_f1(pred, gold)

        em_sum += em
        f1_sum += f1
        total += 1

        if debug_print:
            print(f"\n[Eval][{idx}]")
            print("   ---- INPUT PROMPT ----")
            print(prompt)
            print("   ---- FULL MODEL OUTPUT ----")
            print(full_text)
            print("   ---- GENERATED ONLY ----")
            print(gen_text)
            print(f"   Gold: {gold}")
            print(f"   Pred (extracted): {pred}")
            print(f"   EM={em:.3f} F1={f1:.3f}", flush=True)

    em_pct = 100.0 * em_sum / max(1, total)
    f1_pct = 100.0 * f1_sum / max(1, total)

    # restore
    model.config.use_cache = prev_cache
    tokenizer.padding_side = "right"  # Reset to default

    print(f"[Eval][HotpotQA] Finished {split}: EM={em_pct:.2f}% F1={f1_pct:.2f}% n={total}", flush=True)

    if save_jsonl or save_tsv:
        rec = {
            "timestamp": time.time(),
            "run_name": run_name,
            "model_name": str(model_id),
            "phase": phase,
            "epoch": int(epoch),
            "step": int(step),
            "split": split,
            "em": float(em_pct),
            "f1": float(f1_pct),
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
                    "split","em","f1","n","max_new_tokens","batch_size","output_dir",
                ],
            )

    elapsed = time.time() - start_time
    metrics = {
        "em": em_pct,
        "f1": f1_pct,
        "n": total,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "throughput_qps": total / elapsed if elapsed > 0 else None,
        "batch_size": batch_size,
    }
    metrics.update(_get_gpu_info())
    return metrics

