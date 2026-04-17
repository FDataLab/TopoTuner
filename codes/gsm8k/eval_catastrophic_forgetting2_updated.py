"""
Catastrophic forgetting evaluation (v2) — longer-form / code benchmarks.

  - HotpotQA (distractor, validation): EM + F1; internal distractor protocol, not fullwiki retrieval.
  - XSum (test): ROUGE; 512-token prompt budget, article truncated by tokens; full summary output.
  - CNN/DailyMail v3.0.0 (test): ROUGE; 1024-token budget; highlights prompt; full output.
  - HumanEval (test, 164): pass@1; completion appended to dataset prompt (minimal fence cleanup).

Dependencies (in addition to torch, transformers, datasets):
  pip install rouge-score human-eval

Usage:
  python eval_catastrophic_forgetting2_updated.py --model Qwen/Qwen3-8B-Base
  python eval_catastrophic_forgetting2_updated.py --model ./ckpt --benchmarks hotpot humaneval --max-samples 32
  python eval_catastrophic_forgetting2_updated.py --model ./lora --is-lora

Gated Hub models (e.g. meta-llama/*): set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, or run ``huggingface-cli login``.

Default generation caps: HotpotQA plain prompt, 3072 ctx (raise with ``--hotpot-max-input-length`` if needed), 16 new tok; XSum/CNN tunable via ``--xsum-max-new-tokens`` / ``--cnndm-max-new-tokens`` (default 256 each); HumanEval 512.
HumanEval always uses the raw dataset ``prompt`` (no chat wrap / no extra instructions). HotpotQA uses a plain string prompt (no chat template), regardless of ``--chat-template``.
Other benchmarks: ``auto`` skips chat wrap for base / finetuned-from-base paths; use ``--chat-template off`` to force plain prompts everywhere those templates apply.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import string
import time
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rouge_score import rouge_scorer
except ImportError as e:
    raise SystemExit(
        "Missing rouge-score. Install: pip install rouge-score"
    ) from e

try:
    from human_eval.execution import check_correctness
except ImportError as e:
    raise SystemExit(
        "Missing human-eval. Install: pip install human-eval"
    ) from e

# Avoid tokenizer deadlock warnings when HumanEval forks workers after generation.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Set from main() so --model-name affects chat-template auto-detection for local checkpoints.
_CHAT_TEMPLATE_MODE: str = "auto"
_EVAL_MODEL_HINT: str = ""


def set_eval_globals(*, chat_template_mode: str, model_hint: str) -> None:
    global _CHAT_TEMPLATE_MODE, _EVAL_MODEL_HINT
    _CHAT_TEMPLATE_MODE = chat_template_mode
    _EVAL_MODEL_HINT = (model_hint or "").lower()


def log(msg: str = ""):
    print(msg, flush=True)


def write_prediction_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log(f"  Saved predictions CSV: {path}")


def chat_model_label(model, tokenizer) -> str:
    """Single string for chat-template detection: CLI hint + config id + tokenizer id."""
    parts = [
        _EVAL_MODEL_HINT,
        getattr(model.config, "_name_or_path", None) or "",
        getattr(tokenizer, "name_or_path", None) or "",
    ]
    return " ".join(str(p).strip() for p in parts if str(p).strip()).lower()


def _looks_like_causal_base_only(label: str) -> bool:
    """Pretrained or finetuned-from-base weights (no instruct/chat SFT): use plain prompts in auto mode."""
    n = (label or "").lower()
    if "instruct" in n:
        return False
    # Merged checkpoints from our Qwen3-8B-Base / Llama 3.1 8B GSM8K runs (paths on disk).
    if "qwen-base" in n or "/checkpoints/qwen-base" in n:
        return True
    if re.search(r"checkpoints/llama(/|$)", n):
        return True
    if "qwen3-8b-base" in n or "qwen3_8b_base" in n:
        return True
    if "qwen3-8b" in n or "qwen3_8b" in n:
        return False  # Hub chat model Qwen/Qwen3-8B (unused in base-only eval scripts)
    if "llama-3.1-8b" in n or "llama_3_1_8b" in n or "meta-llama--llama-3.1-8b" in n:
        return True
    return False


def should_use_chat_template(tokenizer, chat_label: str) -> bool:
    if _CHAT_TEMPLATE_MODE == "off":
        return False
    if not getattr(tokenizer, "chat_template", None):
        return False
    if _CHAT_TEMPLATE_MODE == "on":
        return True
    # auto: wrap instruct/chat; skip only clear causal-base checkpoints.
    return not _looks_like_causal_base_only(chat_label)


def maybe_apply_chat_template(
    tokenizer,
    user_prompt: str,
    system_prompt: str | None = None,
    *,
    chat_label: str = "",
) -> str:
    if not should_use_chat_template(tokenizer, chat_label):
        return user_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if "qwen" in chat_label:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def strip_reasoning_preamble(text: str) -> str:
    """Drop Qwen3 thinking blocks and unclosed openings when generation truncates early."""
    if not text:
        return ""
    t = text.strip()
    ot, ct = chr(60) + "think" + chr(62), chr(60) + "/" + "think" + chr(62)
    t = re.sub(re.escape(ot) + r"[\s\S]*?" + re.escape(ct), "", t, flags=re.IGNORECASE)
    t = re.sub(re.escape(ot) + r"[\s\S]*\Z", "", t, flags=re.IGNORECASE)
    # Some tooling rewrites the think tag name; handle literal redacted placeholder too.
    ro = chr(60) + "redacted_thinking" + chr(62)
    rc = chr(60) + "/" + "redacted_thinking" + chr(62)
    t = re.sub(re.escape(ro) + r"[\s\S]*?" + re.escape(rc), "", t, flags=re.IGNORECASE)
    t = re.sub(re.escape(ro) + r"[\s\S]*\Z", "", t, flags=re.IGNORECASE)
    return t.strip()


def strip_common_generation_prefixes(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    patterns = [
        r"^assistant\s*:?\s*",
        r"^summary\s*:?\s*",
        r"^highlights?\s*:?\s*",
        r"^answer\s*:?\s*",
        r"^final answer\s*:?\s*",
    ]
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new = re.sub(pat, "", text, flags=re.IGNORECASE)
            if new != text:
                text = new.strip()
                changed = True
    return text


# ──────────────────────────────────────────────────────────────────────
#  Shared generation
# ──────────────────────────────────────────────────────────────────────

def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 32,
    batch_size: int = 256,
    max_input_length: int = 4096,
):
    """Batched greedy generation with left-padding.

    New tokens are taken from index ``input_ids.shape[1]`` onward (HF contract). For each row we
    also compute the last non-pad index from ``attention_mask``; with left padding this should
    match ``seq_width - 1``. If it does not, we still slice from ``seq_width`` so we never treat
    pad columns as model-generated text.
    """
    _prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    results: List[str] = []
    n_total = len(prompts)
    if n_total == 0:
        tokenizer.padding_side = _prev_side
        return results
    n_batches = (n_total + batch_size - 1) // batch_size

    for i in range(0, n_total, batch_size):
        batch = prompts[i : i + batch_size]
        device = getattr(model, "device", None) or next(model.parameters()).device
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_width = int(input_ids.shape[1])

        t_batch = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t_batch
        done = min(i + len(batch), n_total)
        bnum = i // batch_size + 1
        log(
            f"  [generate_batch] batch {bnum}/{n_batches}  rows {done}/{n_total}  "
            f"{elapsed:.1f}s  (max_new_tokens={max_new_tokens})"
        )

        for j in range(len(batch)):
            mask_row = attention_mask[j]
            nz = (mask_row == 1).nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                last_real_plus_one = seq_width
            else:
                last_real_plus_one = int(nz[-1].item()) + 1
            # Decoder-only generate: continuations are appended after the full padded input block.
            gen_start = seq_width
            if last_real_plus_one != seq_width and j == 0:
                log(
                    f"  [generate_batch] row0: last non-pad+1={last_real_plus_one} != seq_width "
                    f"{seq_width}; slicing new tokens from seq_width (HF generate contract)."
                )
            gen_ids = out[j, gen_start:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

    tokenizer.padding_side = _prev_side
    return results


# ──────────────────────────────────────────────────────────────────────
#  HotpotQA: normalization + EM/F1 (aligned with hotpotqa/eval_hotpotqa_updated.py)
# ──────────────────────────────────────────────────────────────────────

def hotpot_normalize_text(s: str) -> str:
    def lower(text):
        return text.lower()

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def hotpot_em_and_f1(pred: str, gold: str) -> Tuple[float, float]:
    pred_n = hotpot_normalize_text(pred)
    gold_n = hotpot_normalize_text(gold)

    em = 1.0 if pred_n == gold_n else 0.0

    pred_toks = pred_n.split()
    gold_toks = gold_n.split()

    if not pred_toks and not gold_toks:
        return em, 1.0
    if not pred_toks or not gold_toks:
        return em, 0.0

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


def hotpot_normalize_yes_no_answer(line: str) -> str:
    """If the model leads with yes/no, collapse to 'yes' or 'no' for EM (e.g. 'Yes. Because...' → yes)."""
    m = re.match(r"^\s*(yes|no)\b", line, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return line


def hotpot_extract_final_answer(text: str) -> str:
    """Post-decode: thinking strip, tag split, first line, yes/no collapse, light trim (plain Hotpot)."""
    if text is None:
        return ""

    text = strip_reasoning_preamble(str(text).strip())
    if not text:
        return ""

    lowered = text.lower()
    for tag in ["answer:", "final answer:", "final:", "output:"]:
        pos = lowered.find(tag)
        if pos != -1:
            text = text[pos + len(tag) :].strip()
            break

    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]

    stop_markers = ["context:", "question:", "explanation:", "###", "assistant:"]
    low_first = first.lower()
    for m in stop_markers:
        p = low_first.find(m)
        if p != -1:
            first = first[:p].strip()
            break

    first = hotpot_normalize_yes_no_answer(first.strip())
    first = first.strip().strip('"').strip("'").strip()
    return first


def hotpot_build_context(ex: Dict[str, Any]) -> str:
    """ex['context']: list of [title, [sentences...]]."""
    ctx = ex.get("context", None)
    if not isinstance(ctx, list):
        return ""

    parts: List[str] = []
    for item in ctx:
        if not (isinstance(item, (list, tuple))) or len(item) != 2:
            continue
        title, sents = item
        if isinstance(sents, list):
            text = " ".join(str(x) for x in sents)
        else:
            text = str(sents)
        parts.append(f"{title}: {text}")
    return "\n".join(parts)


def truncate_article_to_token_budget(
    tokenizer, article: str, template_prefix: str, template_suffix: str, max_total_tokens: int
) -> str:
    """Reserve tokens for fixed prompt pieces; truncate **article** by tokens (not words)."""
    pref_ids = tokenizer.encode(template_prefix, add_special_tokens=False)
    suf_ids = tokenizer.encode(template_suffix, add_special_tokens=False)
    overhead = len(pref_ids) + len(suf_ids)
    budget = max(32, max_total_tokens - overhead)
    art_ids = tokenizer.encode(article, add_special_tokens=False, truncation=True, max_length=budget)
    return tokenizer.decode(art_ids, skip_special_tokens=True)


def normal_cleanup_summary(text: str) -> str:
    """Strip + normalize newlines; keep full summary (no first-line cut)."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\r\n?", "\n", text)
    return text.strip()


def light_cleanup_summary(text: str) -> str:
    """Minimal post-process for CNN/DM generations."""
    return normal_cleanup_summary(text)


def mean_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    acc = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(preds)
    if n == 0:
        return acc
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        acc["rouge1"] += s["rouge1"].fmeasure
        acc["rouge2"] += s["rouge2"].fmeasure
        acc["rougeL"] += s["rougeL"].fmeasure
    return {k: round(v / n, 4) for k, v in acc.items()}


# ──────────────────────────────────────────────────────────────────────
#  HotpotQA (distractor)
# ──────────────────────────────────────────────────────────────────────

def build_hotpot_prompt(question: str, context_text: str) -> str:
    """Plain causal prompt (no persona / no chat template); EM+F1 on extracted span."""
    return (
        "Answer the question using the context below. Respond with only the final answer.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def eval_hotpot(
    model,
    tokenizer,
    max_samples=None,
    batch_size=256,
    seed=None,
    *,
    max_input_length: int = 3072,
    max_new_tokens: int = 16,
    **kwargs,
):
    _ = seed
    save_csv = kwargs.pop("save_predictions_path", None)
    _ = kwargs
    log(
        f"  Loading HotpotQA (distractor, validation); "
        f"max_input_length={max_input_length}, max_new_tokens={max_new_tokens}..."
    )
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts: List[str] = []
    golds: List[str] = []
    questions: List[str] = []
    for ex in dataset:
        context = hotpot_build_context(ex)
        q = str(ex.get("question", "")).strip()
        questions.append(q)
        # Always plain text (no chat template / system message) for HotpotQA.
        prompts.append(build_hotpot_prompt(q, context))
        golds.append(str(ex.get("answer", "")).strip())

    log(f"  Generating {len(prompts)} answers (batch_size={batch_size})...")
    raw_preds = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_length=max_input_length,
    )

    em_sum = 0.0
    f1_sum = 0.0
    pred_rows: List[Dict[str, Any]] = []
    for i, (raw, g, q) in enumerate(zip(raw_preds, golds, questions)):
        after_think = strip_reasoning_preamble(raw)
        pred_clean = hotpot_extract_final_answer(after_think)
        em, f1 = hotpot_em_and_f1(pred_clean, g)
        em_sum += em
        f1_sum += f1
        pred_rows.append(
            {
                "idx": i,
                "question": q,
                "gold": g,
                "raw_generation": raw,
                "extracted_answer": pred_clean,
                "em": int(em),
                "f1": round(f1, 4),
            }
        )
    if save_csv:
        write_prediction_csv(
            save_csv,
            ["idx", "question", "gold", "raw_generation", "extracted_answer", "em", "f1"],
            pred_rows,
        )
    n = len(golds)
    return {
        "exact_match": round(em_sum / n, 4) if n else 0.0,
        "f1": round(f1_sum / n, 4) if n else 0.0,
        "total": n,
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "hotpot_config": "distractor",
        "eval_note": (
            "internal_distractor_protocol: fixed paragraphs in split; "
            "not official_fullwiki_retrieval_eval"
        ),
    }


# ──────────────────────────────────────────────────────────────────────
#  XSum
# ──────────────────────────────────────────────────────────────────────

XSUM_PREFIX = "Write a one-sentence news summary for the article below.\n\nArticle:\n"
XSUM_SUFFIX = "\n\nOne-sentence summary:"


def build_xsum_prompt(document: str, tokenizer=None, *, chat_label: str = "") -> str:
    prompt = f"{XSUM_PREFIX}{document}{XSUM_SUFFIX}"
    if tokenizer is not None:
        prompt = maybe_apply_chat_template(
            tokenizer,
            prompt,
            system_prompt="You are a concise news summarizer. Produce one sentence only.",
            chat_label=chat_label,
        )
    return prompt


def eval_xsum(model, tokenizer, max_samples=None, batch_size=64, seed=None, **kwargs):
    _ = seed
    save_csv = kwargs.pop("save_predictions_path", None)
    max_new_tokens = int(kwargs.pop("max_new_tokens", 256))
    _ = kwargs
    chat_label = chat_model_label(model, tokenizer)
    max_input_tokens = 512
    log(
        f"  Loading XSum (test); max_input_tokens={max_input_tokens}, "
        f"max_new_tokens={max_new_tokens} (article fills budget)…"
    )
    dataset = load_dataset("EdinburghNLP/xsum", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    for ex in dataset:
        doc = truncate_article_to_token_budget(
            tokenizer, ex["document"], XSUM_PREFIX, XSUM_SUFFIX, max_input_tokens
        )
        prompts.append(build_xsum_prompt(doc, tokenizer=tokenizer, chat_label=chat_label))
    refs = [ex["summary"].strip() for ex in dataset]

    log(f"  Generating {len(prompts)} summaries (batch_size={batch_size})...")
    preds = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_length=max_input_tokens,
    )
    preds = [
        normal_cleanup_summary(strip_common_generation_prefixes(strip_reasoning_preamble(p)))
        for p in preds
    ]

    if save_csv:
        pred_rows = [{"idx": i, "reference": refs[i], "prediction": preds[i]} for i in range(len(preds))]
        write_prediction_csv(save_csv, ["idx", "reference", "prediction"], pred_rows)

    r = mean_rouge(preds, refs)
    r["total"] = len(preds)
    r["max_input_tokens"] = max_input_tokens
    r["max_new_tokens"] = max_new_tokens
    return r


# ──────────────────────────────────────────────────────────────────────
#  CNN / Daily Mail
# ──────────────────────────────────────────────────────────────────────

CNNDM_PREFIX = "Write highlights for the following article.\n\nArticle:\n"
CNNDM_SUFFIX = "\n\nHighlights:"


def build_cnndm_prompt(article: str, tokenizer=None, *, chat_label: str = "") -> str:
    prompt = f"{CNNDM_PREFIX}{article}{CNNDM_SUFFIX}"
    if tokenizer is not None:
        prompt = maybe_apply_chat_template(
            tokenizer,
            prompt,
            system_prompt="You are a news editor. Write short article highlights only.",
            chat_label=chat_label,
        )
    return prompt


def eval_cnndm(model, tokenizer, max_samples=None, batch_size=64, seed=None, **kwargs):
    _ = seed
    save_csv = kwargs.pop("save_predictions_path", None)
    max_new_tokens = int(kwargs.pop("max_new_tokens", 256))
    _ = kwargs
    chat_label = chat_model_label(model, tokenizer)
    max_input_tokens = 1024
    log(
        f"  Loading CNN/DailyMail 3.0.0 (test); max_input_tokens={max_input_tokens}, "
        f"max_new_tokens={max_new_tokens}…"
    )
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    for ex in dataset:
        art = truncate_article_to_token_budget(
            tokenizer, ex["article"], CNNDM_PREFIX, CNNDM_SUFFIX, max_input_tokens
        )
        prompts.append(build_cnndm_prompt(art, tokenizer=tokenizer, chat_label=chat_label))
    refs = [ex["highlights"].replace("\n", " ").strip() for ex in dataset]

    log(f"  Generating {len(prompts)} summaries (batch_size={batch_size})...")
    preds = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_input_length=max_input_tokens,
    )
    preds = [
        light_cleanup_summary(strip_common_generation_prefixes(strip_reasoning_preamble(p)))
        for p in preds
    ]

    if save_csv:
        pred_rows = [{"idx": i, "reference": refs[i], "prediction": preds[i]} for i in range(len(preds))]
        write_prediction_csv(save_csv, ["idx", "reference", "prediction"], pred_rows)

    r = mean_rouge(preds, refs)
    r["total"] = len(preds)
    r["max_input_tokens"] = max_input_tokens
    r["max_new_tokens"] = max_new_tokens
    return r


# ──────────────────────────────────────────────────────────────────────
#  HumanEval
# ──────────────────────────────────────────────────────────────────────

def humaneval_trim_overgeneration_at_top_level(code: str) -> str:
    """Drop model junk after the stub body: stop at first column-0 def/class/if __name__/assert/print."""
    if not code:
        return code
    out: List[str] = []
    for line in code.splitlines():
        expanded = line.expandtabs(4)
        stripped = expanded.lstrip(" ")
        indent = len(expanded) - len(stripped)
        if not stripped:
            out.append(line)
            continue
        if indent == 0:
            low = stripped.lower()
            if stripped.startswith("def ") or stripped.startswith("class "):
                break
            if low.startswith("if __name__"):
                break
            if stripped.startswith("assert "):
                break
            if stripped.startswith("print(") or stripped.startswith("print "):
                break
        out.append(line)
    return "\n".join(out)


def clean_humaneval_completion(text: str) -> str:
    """Thinking strip, optional fenced code, role prefixes; trim top-level overgeneration; stub newline."""
    text = strip_reasoning_preamble((text or "").strip())
    if "```" in text:
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    text = strip_common_generation_prefixes(text)
    text = text.rstrip()
    text = humaneval_trim_overgeneration_at_top_level(text)
    text = text.rstrip()
    if text and not text.startswith("\n"):
        text = "\n" + text
    return text


def eval_humaneval(model, tokenizer, max_samples=None, batch_size=8, seed=None, **kwargs):
    _ = seed
    save_csv = kwargs.pop("save_predictions_path", None)
    _ = kwargs
    log("  Loading HumanEval (test)...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Benchmark-faithful: always the dataset stub only (no chat template, no system or extra instructions).
    prompts = [ex["prompt"] for ex in dataset]
    problems = [
        {
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "entry_point": ex["entry_point"],
            "test": ex["test"],
        }
        for ex in dataset
    ]

    log(
        f"  Generating {len(prompts)} completions "
        f"(batch_size≤16, then sandbox exec per task)..."
    )
    # Variable-length code prompts: keep batches small
    bs = max(1, min(batch_size, 16))
    preds: List[str] = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i : i + bs]
        preds.extend(
            generate_batch(
                model, tokenizer, chunk, max_new_tokens=512, batch_size=len(chunk)
            )
        )

    passed = 0
    timeout = 10.0
    he_rows: List[Dict[str, Any]] = []
    # check_correctness builds: prompt + completion + tests; completion must extend the stub.
    for prob, raw in zip(problems, preds):
        comp = clean_humaneval_completion(raw)
        res = check_correctness(prob, comp, timeout=timeout)
        ok = bool(res.get("passed"))
        if ok:
            passed += 1
        he_rows.append(
            {
                "task_id": prob["task_id"],
                "raw_generation": raw,
                "completion_passed_to_eval": comp,
                "passed": ok,
                "exec_detail": str(res.get("result", "")),
            }
        )

    if save_csv:
        write_prediction_csv(
            save_csv,
            [
                "task_id",
                "raw_generation",
                "completion_passed_to_eval",
                "passed",
                "exec_detail",
            ],
            he_rows,
        )

    n = len(problems)
    return {
        "pass_at_1": round(passed / n, 4) if n else 0.0,
        "passed": passed,
        "total": n,
    }


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

# Default multi-benchmark order (cnndm last for long runs). Shell dataset-first driver matches this.
BENCHMARK_MAP = {
    "hotpot": ("HotpotQA", eval_hotpot),
    "xsum": ("XSum", eval_xsum),
    "humaneval": ("HumanEval", eval_humaneval),
    "cnndm": ("CNN/DM", eval_cnndm),
}


def _hf_hub_token():
    """Gated Hub models: set HF_TOKEN or HUGGINGFACE_HUB_TOKEN; else use `huggingface-cli login` cache."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True


def load_model_and_tokenizer(args):
    log("Loading model...")
    t0 = time.time()

    tok = _hf_hub_token()

    if args.is_lora:
        from peft import PeftModel

        cfg_path = os.path.join(args.model, "adapter_config.json")
        with open(cfg_path) as f:
            base_id = json.load(f).get("base_model_name_or_path")
        log(f"  LoRA base: {base_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_id, trust_remote_code=True, token=tok
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=tok,
        )
        model = PeftModel.from_pretrained(model, args.model)
        model = model.merge_and_unload()
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True, token=tok
            )
        except (AttributeError, Exception) as e:
            log(f"  Tokenizer load failed ({e.__class__.__name__}), trying config fallback...")
            config_path = os.path.join(args.model, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                model_type = cfg.get("model_type", "")
                base_id = cfg.get("_name_or_path") or ""
                if not base_id or base_id == "None":
                    if "qwen3" in model_type:
                        base_id = "Qwen/Qwen3-8B-Base"
                    elif "llama" in model_type:
                        base_id = "meta-llama/Llama-3.1-8B"
                if base_id:
                    log(f"  Tokenizer fallback to: {base_id}")
                    tokenizer = AutoTokenizer.from_pretrained(
                        base_id, trust_remote_code=True, token=tok
                    )
                else:
                    raise e
            else:
                raise e
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=tok,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    log(f"  Loaded in {time.time() - t0:.1f}s\n")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Catastrophic forgetting evaluation v2")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_MAP.keys()),
        choices=list(BENCHMARK_MAP.keys()),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--humaneval-max-samples",
        type=int,
        default=None,
        help="Cap HumanEval tasks only; other benchmarks still use --max-samples.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--is-lora", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reserved for future stochastic eval hooks (few-shot / shuffling).",
    )
    parser.add_argument(
        "--hotpot-max-input-length",
        type=int,
        default=3072,
        help="HotpotQA: token budget for prompt (plain text); default 3072 to reduce distractor truncation.",
    )
    parser.add_argument(
        "--hotpot-max-new-tokens",
        type=int,
        default=16,
        help="HotpotQA: same default as hotpotqa/eval_hotpotqa_updated.py.",
    )
    parser.add_argument(
        "--xsum-max-new-tokens",
        type=int,
        default=256,
        help="XSum: generation cap (default 256).",
    )
    parser.add_argument(
        "--cnndm-max-new-tokens",
        type=int,
        default=256,
        help="CNN/Daily Mail: generation cap (default 256).",
    )
    parser.add_argument(
        "--merge-existing-report",
        action="store_true",
        help=(
            "If catastrophic_forgetting2_report.json exists in --output-dir, merge new "
            "benchmark results into its 'benchmarks' object (for dataset-by-dataset runs)."
        ),
    )
    parser.add_argument(
        "--save-predictions-dir",
        type=str,
        default=None,
        help=(
            "If set, write per-benchmark CSVs here: "
            "{hotpot,xsum,humaneval,cnndm}_predictions.csv (raw generations + labels / exec details)."
        ),
    )
    parser.add_argument(
        "--chat-template",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto: use chat template when tokenizer has one, except base / our finetuned-from-base "
            "paths (Llama 3.1 8B, Qwen3-8B-Base, checkpoints/llama, qwen-base); "
            "on: always use template if present; off: never wrap (plain prompts)."
        ),
    )
    args = parser.parse_args()

    set_eval_globals(
        chat_template_mode=args.chat_template,
        model_hint=f"{args.model_name or ''} {args.model}",
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_name or os.path.basename(args.model.rstrip("/"))
    if args.output_dir is None:
        args.output_dir = f"catastrophic-forgetting2-{model_name}"

    log(f"\n{'='*60}")
    log("  CATASTROPHIC FORGETTING EVALUATION v2")
    log(f"{'='*60}")
    log(f"  Model:      {args.model}")
    log(f"  Name:       {model_name}")
    log(f"  LoRA:       {args.is_lora}")
    log(f"  Seed:       {args.seed}")
    log(f"  Chat tmpl:  {args.chat_template}")
    log(f"  Benchmarks: {', '.join(args.benchmarks)}")
    log(f"  Batch size: {args.batch_size}")
    if args.max_samples:
        log(f"  Max samples: {args.max_samples}")
    if args.humaneval_max_samples is not None:
        log(f"  HumanEval max samples: {args.humaneval_max_samples}")
    if "hotpot" in args.benchmarks:
        log(
            f"  HotpotQA:    distractor, max_input_length={args.hotpot_max_input_length}, "
            f"max_new_tokens={args.hotpot_max_new_tokens}"
        )
    if "xsum" in args.benchmarks:
        log(f"  XSum:        max_new_tokens={args.xsum_max_new_tokens}")
    if "cnndm" in args.benchmarks:
        log(f"  CNN/DM:      max_new_tokens={args.cnndm_max_new_tokens}")
    log(f"  Output:     {args.output_dir}")
    if args.save_predictions_dir:
        log(f"  Predictions CSV dir: {args.save_predictions_dir}")
    log(f"{'='*60}\n")

    model, tokenizer = load_model_and_tokenizer(args)

    results = {
        "model": args.model,
        "model_name": model_name,
        "seed": args.seed,
        "benchmarks": {},
    }
    os.makedirs(args.output_dir, exist_ok=True)

    for bench_key in args.benchmarks:
        bench_name, eval_fn = BENCHMARK_MAP[bench_key]
        log(f">>> Evaluating {bench_name}...")
        t0 = time.time()
        extra = {}
        if bench_key == "hotpot":
            extra["max_input_length"] = args.hotpot_max_input_length
            extra["max_new_tokens"] = args.hotpot_max_new_tokens
        if bench_key == "xsum":
            extra["max_new_tokens"] = args.xsum_max_new_tokens
        if bench_key == "cnndm":
            extra["max_new_tokens"] = args.cnndm_max_new_tokens
        if args.save_predictions_dir:
            extra["save_predictions_path"] = os.path.join(
                args.save_predictions_dir, f"{bench_key}_predictions.csv"
            )
        max_samples = args.max_samples
        if bench_key == "humaneval" and args.humaneval_max_samples is not None:
            max_samples = args.humaneval_max_samples
        res = eval_fn(
            model,
            tokenizer,
            max_samples=max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            **extra,
        )
        res["time_s"] = round(time.time() - t0, 1)
        results["benchmarks"][bench_key] = res

        if "exact_match" in res:
            log(
                f"  {bench_name}: EM={res['exact_match']*100:.1f}% "
                f"F1={res['f1']*100:.1f}% ({res['total']}) [{res['time_s']}s]"
            )
        elif "rouge1" in res:
            log(
                f"  {bench_name}: R1={res['rouge1']*100:.1f} "
                f"R2={res['rouge2']*100:.1f} RL={res['rougeL']*100:.1f} "
                f"({res['total']}) [{res['time_s']}s]"
            )
        elif "pass_at_1" in res:
            log(
                f"  {bench_name}: pass@1={res['pass_at_1']*100:.1f}% "
                f"({res['passed']}/{res['total']}) [{res['time_s']}s]"
            )
        log("")

    report_path = os.path.join(args.output_dir, "catastrophic_forgetting2_report.json")
    if args.merge_existing_report and os.path.isfile(report_path):
        with open(report_path) as f:
            prev = json.load(f)
        merged = dict(prev.get("benchmarks", {}))
        merged.update(results["benchmarks"])
        results["benchmarks"] = merged
        log(f"  Merged into existing report ({len(args.benchmarks)} new key(s)).")

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Report saved: {report_path}")

    log(f"\n{'='*60}")
    log(f"  SUMMARY v2 — {model_name}")
    log(f"{'='*60}")
    for bench_key, res in results["benchmarks"].items():
        name = BENCHMARK_MAP[bench_key][0]
        if "exact_match" in res:
            log(f"  {name}: EM={res['exact_match']*100:.1f}% F1={res['f1']*100:.1f}%")
        elif "rouge1" in res:
            log(
                f"  {name}: R1={res['rouge1']*100:.1f} "
                f"R2={res['rouge2']*100:.1f} RL={res['rougeL']*100:.1f}"
            )
        elif "pass_at_1" in res:
            log(f"  {name}: pass@1={res['pass_at_1']*100:.1f}%")
    log(f"{'='*60}\n")


if __name__ == "__main__":
    main()
