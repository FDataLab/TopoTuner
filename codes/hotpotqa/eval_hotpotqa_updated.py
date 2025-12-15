# codes/hotpotqa/eval_hotpotqa_updated.py

import os
import csv
import time
import re
import string
from typing import Optional, Dict, Tuple, Any

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


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


# -------------------- robust answer extraction --------------------
def _extract_final_answer(text: str) -> str:
    """
    Robust answer extractor for HotpotQA.
    Goals:
      - never crash
      - handle empty generations
      - prefer content after "Answer:" / "Final answer:" etc.
      - return a short, first-line style answer
    """
    if text is None:
        return ""

    text = str(text).strip()
    if not text:
        return ""

    lowered = text.lower()
    for tag in ["answer:", "final answer:", "final:", "output:"]:
        pos = lowered.find(tag)
        if pos != -1:
            text = text[pos + len(tag):].strip()
            break

    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]

    # If the first line accidentally contains "Context:"/"Question:" etc, trim before it.
    stop_markers = ["context:", "question:", "explanation:", "###", "assistant:"]
    low_first = first.lower()
    for m in stop_markers:
        p = low_first.find(m)
        if p != -1:
            first = first[:p].strip()
            break

    first = first.strip().strip('"').strip("'").strip()
    return first


# -------------------- HotpotQA normalization + metrics --------------------
def _normalize_text(s: str) -> str:
    def lower(text): return text.lower()
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _em_and_f1(pred: str, gold: str) -> Tuple[float, float]:
    pred_n = _normalize_text(pred)
    gold_n = _normalize_text(gold)

    em = 1.0 if pred_n == gold_n else 0.0

    pred_toks = pred_n.split()
    gold_toks = gold_n.split()

    if not pred_toks and not gold_toks:
        return em, 1.0
    if not pred_toks or not gold_toks:
        return em, 0.0

    # token overlap counts
    gold_counts: Dict[str, int] = {}
    for t in gold_toks:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    num_same = 0
    for t in pred_toks:
        if gold_counts.get(t, 0) > 0:
            num_same += 1
            gold_counts[t] -= 1

    if num_same == 0:
        return em, 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def _build_hotpot_context(ex: Dict[str, Any]) -> str:
    """
    ex["context"] in hotpot_qa is typically list of [title, [sentences...]].
    We join into a compact context block.
    """
    ctx = ex.get("context", None)
    if not isinstance(ctx, list):
        return ""

    parts = []
    for item in ctx:
        if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) != 2:
            continue
        title, sents = item
        if isinstance(sents, list):
            text = " ".join(str(x) for x in sents)
        else:
            text = str(sents)
        parts.append(f"{title}: {text}")
    return "\n".join(parts)


@torch.no_grad()
def evaluate_hotpotqa(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "validation",
    limit: Optional[int] = None,
    batch_size: int = 1,          # intentionally ignored (sanity first)
    max_new_tokens: int = 16,
    *,
    config: str = "distractor",
    progress_bar: bool = True,
    debug_print: bool = False,
    # Option A: one big file, appends rows across checkpoints
    save_examples_csv: Optional[str] = None,
    # eval_checkpoints should pass checkpoint name here
    run_name: str = "",
    # safety limits
    max_input_length: int = 1024,
) -> Dict[str, float]:

    start_time = time.time()

    ds = load_dataset("hotpot_qa", config)[split]
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # tokenizer safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    model.eval()

    total = 0
    em_sum = 0.0
    f1_sum = 0.0

    print(f"[Eval][HotpotQA] Starting evaluation on split={split}, n={len(ds)}", flush=True)

    iterator = tqdm(ds, disable=not progress_bar)

    for idx, ex in enumerate(iterator):
        question = str(ex.get("question", "")).strip()
        gold = str(ex.get("answer", "")).strip()

        context = _build_hotpot_context(ex)

        prompt = (
            "You are a question answering assistant.\n"
            "Answer the question using the provided context.\n"
            "Respond with only the final answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # decode only newly generated tokens
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # IMPORTANT: robust extractor must never crash
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
                "question": question,
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
            print("=" * 80)
            print(f"[{idx}] QUESTION:", question)
            print("GOLD:", gold)
            print("RAW GEN:", gen_text)
            print("PRED:", pred)
            print(f"EM={em:.3f} F1={f1:.3f}")
            print("=" * 80, flush=True)

    elapsed = time.time() - start_time

    em_avg = em_sum / max(1, total)
    f1_avg = f1_sum / max(1, total)

    print(
        f"[Eval][HotpotQA] Finished {split}: "
        f"EM={em_avg*100:.2f}% F1={f1_avg*100:.2f}% n={total}",
        flush=True,
    )

    return {
        "em": em_avg,     # fraction (0-1)
        "f1": f1_avg,     # fraction (0-1)
        "n": total,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "throughput_qps": total / elapsed if elapsed > 0 else None,
    }
