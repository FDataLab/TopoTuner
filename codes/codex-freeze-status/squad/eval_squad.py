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

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
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
    text = text.split("<|end_of_text|>")[0]
    if "Answer:" in text:
        _, tail = text.rsplit("Answer:", maxsplit=1)
        return tail.strip().splitlines()[0].strip()
    return text.strip()

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
    max_new_tokens: int = 256,
    batch_size: int = 8,
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

    iterator = tqdm(ds, disable=not progress_bar)
    for idx, ex in enumerate(iterator):
        q = ex["question"]
        ctx = ex.get("context", "")
        answers = ex.get("answers", {}).get("text", [])
        gold = (answers[0] if answers else "").strip()

        prompt = build_prompt(tokenizer, q, context=ctx, prompt_format=prompt_format, use_instruction=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        if tokenization_debug:
            print(f"\n[Eval][{idx}] --- TOKENIZATION DEBUG ---")
            print("Prompt text:\n", prompt)
            print("Input IDs shape:", inputs["input_ids"].shape)
            print("Decoded back from IDs:\n", tokenizer.decode(inputs["input_ids"][0]))

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        gen_only_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=False)

        pred = _extract_final_answer(gen_text)
        em, f1 = _em_and_f1(pred, gold)

        em_sum += em
        f1_sum += f1
        total += 1

        if debug_print:
            print(f"\n[Eval][{idx}] Gold={gold} Pred={pred} EM={em:.3f} F1={f1:.3f}", flush=True)

    em_pct = 100.0 * em_sum / max(1, total)
    f1_pct = 100.0 * f1_sum / max(1, total)

    # restore
    model.config.use_cache = prev_cache
    tokenizer.padding_side = "right"  # Reset to default

    print(f"[Eval][SQuAD] Finished {split}: EM={em_pct:.2f}% F1={f1_pct:.2f}% n={total}", flush=True)

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

