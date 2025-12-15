import os
import re
import csv
import json
import time
from typing import Optional, Dict

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

# use your builders + prompt inference + EOS chooser
from .data_preprocessing_gsm8k import (
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

# -------------------- small utils --------------------
def _extract_final_answer(text: str) -> str:
    # 1. Look for explicit <final-answer> tags
    m = re.search(r"<final-answer>\s*(.*?)\s*</final-answer>", text, flags=re.S)
    if m:
        return m.group(1).strip()

    # 2. Otherwise, cut off any trailing artifacts after <|end_of_text|>
    text = text.split("<|end_of_text|>")[0]

    # 3. Find numbers in the cleaned text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()

def _norm_num(s: str) -> str:
    s = s.strip().replace(",", "")
    if re.match(r"^-?\d+\.$", s):  # trailing dot like "12."
        s = s[:-1]
    return s

def _to_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def _answers_match(pred: str, gold: str) -> bool:
    p, g = _norm_num(pred), _norm_num(gold)
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        return pf == gf
    return p == g

def _gold_from_solution(s: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else s.strip()

def _build_train_prompt(tokenizer, question: str, prompt_format: str) -> str:
    if prompt_format == "llama2":
        return create_prompt_llama2(question, use_instruction=True)
    if prompt_format == "llama3":
        return create_prompt_llama3(tokenizer, question, use_instruction=True)
    if prompt_format == "mistral":
        return create_prompt_mistral(question, use_instruction=True)
    if prompt_format == "olmo":
        return create_prompt_olmo(question, use_instruction=True)
    # default: qwen
    return create_prompt_qwen(tokenizer, question, use_instruction=True)

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
def evaluate_gsm8k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "test",
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
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
    compute_text_metrics: bool = True,
) -> Dict[str, float]:

    start_time = time.time()

    ds = load_dataset("openai/gsm8k", "main")[split]
    # downsample for parity with your finetune debug runs
    ds = ds.select(range(len(ds) // 16))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # decide prompt & eos from model id
    model_id = getattr(model, "name_or_path", None) or getattr(model.config, "_name_or_path", "")
    prompt_format = infer_prompt_format_from_model_id(str(model_id))
    eos_id = eos_for_model(tokenizer, str(model_id))

    # generation-safe toggles
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.eval()

    total = 0
    correct = 0

    # accumulate texts for text-level metrics
    pred_texts = []
    ref_texts = []

    print(f"[Eval] Starting evaluation on split={split}, n={len(ds)}", flush=True)

    iterator = tqdm(ds, disable=not progress_bar)
    for idx, ex in enumerate(iterator):
        q = ex["question"]
        gold = _gold_from_solution(str(ex["answer"]))
        prompt = _build_train_prompt(tokenizer, q, prompt_format)

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

        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        prompt_len = inputs["input_ids"].shape[1]
        gen_only_ids = out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=False)

        pred = _extract_final_answer(gen_text)
        is_correct = _answers_match(pred, gold)
        if is_correct:
            correct += 1
        total += 1

        # store for text metrics (compare raw generated text to ground-truth full solution)
        if compute_text_metrics:
            pred_texts.append(gen_text.strip())
            ref_texts.append(str(ex["answer"]).strip())

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
            print(f"   Correct? {is_correct}", flush=True)

    em = 100.0 * correct / max(1, total)

    # optional text metrics (ROUGE + BERTScore)
    rouge1 = rouge2 = rougeL = rougeLsum = None
    bert_p = bert_r = bert_f1 = None
    if compute_text_metrics and len(pred_texts) == total and total > 0:
        try:
            import evaluate as hf_evaluate  # type: ignore

            # ROUGE
            try:
                rouge_metric = hf_evaluate.load("rouge")
                rouge_res = rouge_metric.compute(
                    predictions=pred_texts,
                    references=ref_texts,
                    use_stemmer=True,
                )
                # convert to percentage
                rouge1 = float(rouge_res.get("rouge1", None))
                rouge2 = float(rouge_res.get("rouge2", None))
                rougeL = float(rouge_res.get("rougeL", None))
                rougeLsum = float(rouge_res.get("rougeLsum", None))
                if rouge1 is not None:
                    rouge1 *= 100.0
                if rouge2 is not None:
                    rouge2 *= 100.0
                if rougeL is not None:
                    rougeL *= 100.0
                if rougeLsum is not None:
                    rougeLsum *= 100.0
            except Exception as e:
                print(f"[Eval][warn] ROUGE metric failed: {e}", flush=True)

            # BERTScore
            try:
                bert_metric = hf_evaluate.load("bertscore")
                bert_res = bert_metric.compute(
                    predictions=pred_texts,
                    references=ref_texts,
                    lang="en",
                )
                # bert_res values are lists; take means and convert to percentage
                if bert_res and "precision" in bert_res:
                    bert_p = float(sum(bert_res["precision"]) / len(bert_res["precision"])) * 100.0
                if bert_res and "recall" in bert_res:
                    bert_r = float(sum(bert_res["recall"]) / len(bert_res["recall"])) * 100.0
                if bert_res and "f1" in bert_res:
                    bert_f1 = float(sum(bert_res["f1"]) / len(bert_res["f1"])) * 100.0
            except Exception as e:
                print(f"[Eval][warn] BERTScore metric failed: {e}", flush=True)
        except Exception as e:
            print(f"[Eval][warn] 'evaluate' package not available or failed: {e}", flush=True)

    # restore config
    model.config.use_cache = prev_cache

    print(f"[Eval] Finished {split}: EM={em:.2f}% n={total}", flush=True)

    # optional persistent logging of the *aggregate* downstream result
    if save_jsonl or save_tsv:
        rec = {
            "timestamp": time.time(),
            "run_name": run_name,
            "model_name": str(model_id),
            "phase": phase,
            "epoch": int(epoch),
            "step": int(step),
            "split": split,
            "em": float(em),
            "n": int(total),
            "max_new_tokens": int(max_new_tokens),
            "output_dir": output_dir,
            # text metrics (may be None if not computed)
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
            "bertscore_precision": bert_p,
            "bertscore_recall": bert_r,
            "bertscore_f1": bert_f1,
        }
        if save_jsonl:
            _append_jsonl(save_jsonl, rec)
        if save_tsv:
            _append_tsv(
                save_tsv,
                rec,
                field_order=[
                    "timestamp","run_name","model_name","phase","epoch","step",
                    "split","em","n","max_new_tokens","output_dir",
                    "rouge1","rouge2","rougeL","rougeLsum",
                    "bertscore_precision","bertscore_recall","bertscore_f1",
                ],
            )

    elapsed = time.time() - start_time

    # add system + timing info
    metrics = {
        "em": em,
        "n": total,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "throughput_qps": total / elapsed if elapsed > 0 else None,
    }
    # attach text metrics if computed
    if compute_text_metrics:
        metrics.update({
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
            "bertscore_precision": bert_p,
            "bertscore_recall": bert_r,
            "bertscore_f1": bert_f1,
        })
    metrics.update(_get_gpu_info())

    return metrics