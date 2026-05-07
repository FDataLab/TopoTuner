"""
Catastrophic forgetting evaluation (v3) — **single file**, greedy HF generation on one GPU.

Eleven downstream benchmarks (GSM8K is the training task — **not** evaluated here):

  sst2 imdb mmlu squad hotpot xsum cnndm dolly alpaca humaneval codefeedback

Optional / twelfth benchmark key:

  oci — HumanEval + MBPP pass@1 plus ``oci_mean_pass_at_1`` (OCI / OpenCodeInterpreter-style pairing).

Rough metric map (see ``metric_note`` / README-style comments in JSON where we diverge):

  SST-2 / IMDB / MMLU — accuracy | Hotpot / SQuAD — EM + token F1 | XSum / CNN-DM — ROUGE |
  Dolly — ROUGE (+ optional BLEU/METEOR) | Alpaca — ROUGE (+ optional BLEU/METEOR; not pairwise win-rate) |
  HumanEval — pass@1 | codefeedback — MBPP-sanitized pass@1 |

Dependencies::
  pip install rouge-score human-eval torch transformers datasets
  # optional: sacrebleu (BLEU), nltk (METEOR — downloads WordNet on first use)

Usage::
  python eval_catastrophic_forgetting3.py --model meta-llama/Llama-3.1-8B --batch-size 64
  python eval_catastrophic_forgetting3.py --model ./ckpt --is-lora --max-samples 256 \\
      --benchmarks sst2 humaneval

Results are checkpointed to ``--output-dir/catastrophic_forgetting3_report.json`` **after each benchmark**
(atomic write). Re-run with the same ``--model``, ``--model-name``, and ``--output-dir`` to resume (existing
benchmarks are skipped). Pass ``--overwrite-report`` to replace that JSON from scratch.

Gated Hub models: HF_TOKEN / huggingface-cli login.

HotpotQA uses plain prompts (no chat template). HumanEval/MBPP use dataset stubs only (no chat wrap).
Summarization / IF tasks follow ``--chat-template`` (auto/on/off) like the former v2 runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import re
import string
import time
from collections import Counter
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from codes.utils.eval_split import random_k_split_indices

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


def _safe_stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.stdev(vals))


def load_or_create_eval_split(
    benchmark_key: str,
    n_total: int,
    max_samples: Optional[int],
    k: int,
    eval_split_seed: int,
    indices_dir: str,
) -> Tuple[List[int], List[List[int]], str, Dict[str, Any]]:
    """Fixed RNG pool over rows ``0..n_total-1``, then ``k`` disjoint folds over pool positions.

    Cache path: ``{indices_dir}/{benchmark_key}_seed{seed}_k{k}_n{n_total}_ms{pool_size}.json``
    """
    pool_size = min(max_samples if max_samples is not None else n_total, n_total)
    os.makedirs(indices_dir, exist_ok=True)
    fname = f"{benchmark_key}_seed{eval_split_seed}_k{k}_n{n_total}_ms{pool_size}.json"
    path = os.path.abspath(os.path.join(indices_dir, fname))

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        ok = (
            cached.get("benchmark_key") == benchmark_key
            and int(cached.get("n_total", -1)) == n_total
            and int(cached.get("pool_size", -1)) == pool_size
            and int(cached.get("k", -1)) == k
            and int(cached.get("eval_split_seed", -1)) == eval_split_seed
            and isinstance(cached.get("pool_indices"), list)
            and isinstance(cached.get("fold_positions"), list)
        )
        if ok:
            log(f"  Loaded cached eval split indices: {path}")
            return cached["pool_indices"], cached["fold_positions"], path, cached
        log(f"  Split cache mismatch — regenerating: {path}")

    rng = np.random.RandomState(eval_split_seed)
    perm = rng.permutation(n_total)
    pool_indices = perm[:pool_size].astype(int).tolist()
    fold_positions = random_k_split_indices(pool_size, k, eval_split_seed)

    payload: Dict[str, Any] = {
        "benchmark_key": benchmark_key,
        "n_total": n_total,
        "pool_size": pool_size,
        "k": k,
        "eval_split_seed": eval_split_seed,
        "format_version": 1,
        "pool_indices": pool_indices,
        "fold_positions": fold_positions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"  Wrote eval split indices: {path}")
    return pool_indices, fold_positions, path, payload


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


# forgetting-codes.zip DATASETS["hotpotqa"]["stop_tokens"] → vLLM ``stop=[...]``; HF uses ``stop_strings`` + tokenizer.
HOTPOT_ZIP_STOP_STRINGS = ("\n\n", "Question:")


def generate_batch_hotpot(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 256,
    max_input_length: int = 8192,
):
    """Batched greedy generation with left-padding — matches zip **decoder** stops via HF ``stop_strings``."""
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
                tokenizer=tokenizer,
                stop_strings=list(HOTPOT_ZIP_STOP_STRINGS),
            )
        elapsed = time.perf_counter() - t_batch
        done = min(i + len(batch), n_total)
        bnum = i // batch_size + 1
        log(
            f"  [generate_batch_hotpot] batch {bnum}/{n_batches}  rows {done}/{n_total}  "
            f"{elapsed:.1f}s  (max_new_tokens={max_new_tokens}; HF stop_strings={list(HOTPOT_ZIP_STOP_STRINGS)})"
        )

        for j in range(len(batch)):
            mask_row = attention_mask[j]
            nz = (mask_row == 1).nonzero(as_tuple=True)[0]
            if nz.numel() == 0:
                last_real_plus_one = seq_width
            else:
                last_real_plus_one = int(nz[-1].item()) + 1
            gen_start = seq_width
            gen_ids = out[j, gen_start:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

    tokenizer.padding_side = _prev_side
    return results


# ──────────────────────────────────────────────────────────────────────
#  HotpotQA: EM + token F1 — **match** ``forgetting-codes.zip`` → ``Untitled/metrics.py``
# ──────────────────────────────────────────────────────────────────────


def hotpot_normalize_text(s: str) -> str:
    """Same as forgetting-codes.zip ``Untitled/metrics.py::_normalize_answer``."""
    text = str(s).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def hotpot_em_and_f1(pred: str, gold: str) -> Tuple[float, float]:
    """EM + multiset token F1 — same formulas as zip ``hotpotqa_metrics`` / ``_token_f1``."""
    pred_n = hotpot_normalize_text(pred)
    gold_n = hotpot_normalize_text(gold)
    em = 1.0 if pred_n == gold_n else 0.0

    pred_tokens = pred_n.split()
    ref_tokens = gold_n.split()
    if not pred_tokens and not ref_tokens:
        return em, 1.0
    if not pred_tokens or not ref_tokens:
        return em, 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return em, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def hotpot_truncate_context_zip(context_text: str, max_ctx_tokens: int) -> str:
    """Char truncation as ``Untitled/datasets_loader.py::_load_hotpotqa`` (×4 heuristic)."""
    max_chars = max(1, int(max_ctx_tokens)) * 4
    text = context_text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"


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
    """ex['context']: list of [title, [sentences...]]. Join paragraphs with blank lines — same as ``forgetting-codes.zip`` → ``Untitled/datasets_loader.py::_load_hotpotqa``."""
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
    return "\n\n".join(parts)


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


def rouge_fold_mean_std(
    fold_positions: List[List[int]],
    preds_pool: List[str],
    refs: List[str],
) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, float]]:
    """Per-fold macro ROUGE (same as ``mean_rouge`` on each fold), then mean/std across folds."""
    keys = ("rouge1", "rouge2", "rougeL")
    fold_lists: Dict[str, List[float]] = {k: [] for k in keys}
    for fp in fold_positions:
        sub_p = [preds_pool[i] for i in fp]
        sub_r = [refs[i] for i in fp]
        mr = mean_rouge(sub_p, sub_r)
        for k in keys:
            fold_lists[k].append(float(mr[k]))
    means = {k: round(float(statistics.mean(fold_lists[k])), 4) if fold_lists[k] else 0.0 for k in keys}
    stds = {k: round(_safe_stdev([float(x) for x in fold_lists[k]]), 4) for k in keys}
    return fold_lists, means, stds


# ──────────────────────────────────────────────────────────────────────
#  HotpotQA (distractor)
# ──────────────────────────────────────────────────────────────────────

def build_hotpot_prompt(question: str, context_text: str) -> str:
    """Plain causal prompt — **match** ``forgetting-codes.zip`` → ``Untitled/datasets_loader.py::_load_hotpotqa`` (no chat template)."""
    q = (question or "").strip()
    return f"Context:\n{context_text}\n\nQuestion: {q}\nAnswer:"


# forgetting-codes.zip → Untitled/config.py FEW_SHOT_EXAMPLES["hotpotqa"]
HOTPOT_ZIP_FEW_SHOT_PREFIX = """\
Context:
Oxygen: Oxygen is a chemical element with symbol O and atomic number 8.
Water: Water is an inorganic compound with the chemical formula H2O.

Question: What is the chemical formula of the compound made from oxygen and hydrogen?
Answer: H2O

Context:
Python (programming language): Python is a high-level, general-purpose programming language created by Guido van Rossum.
Guido van Rossum: Guido van Rossum is a Dutch programmer best known as the creator of Python.

Question: What is the nationality of the person who created Python?
Answer: Dutch

"""


def eval_hotpot(
    model,
    tokenizer,
    max_samples=None,
    batch_size=256,
    seed=None,
    *,
    max_input_length: int = 8192,
    max_new_tokens: int = 256,
    max_ctx_tokens: int = 1500,
    few_shot: bool = True,
    **kwargs,
):
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_csv = kwargs.pop("save_predictions_path", None)
    _ = kwargs
    fs_note = "few_shot=ON (forgetting-codes.zip FEW_SHOT_EXAMPLES)" if few_shot else "few_shot=OFF"
    log(
        f"  Loading HotpotQA (distractor, validation split); forgetting-codes.zip protocol: "
        f"{fs_note}; max_ctx_tokens={max_ctx_tokens} (×4 chars → context truncation), "
        f"tokenizer max_length={max_input_length}, max_new_tokens={max_new_tokens}..."
    )
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "hotpot", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    prompts: List[str] = []
    golds: List[str] = []
    questions: List[str] = []
    for ex in pool_ds:
        context_full = hotpot_build_context(ex)
        context = hotpot_truncate_context_zip(context_full, max_ctx_tokens)
        q = str(ex.get("question", "")).strip()
        questions.append(q)
        instr = build_hotpot_prompt(q, context)
        prompts.append(HOTPOT_ZIP_FEW_SHOT_PREFIX + instr if few_shot else instr)
        golds.append(str(ex.get("answer", "")).strip())

    fold_em: List[float] = []
    fold_f1: List[float] = []
    raw_pool: List[str] = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        raw_preds = generate_batch_hotpot(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        ems = []
        f1s = []
        for pi, pos in enumerate(fp):
            raw_pool[pos] = raw_preds[pi]
            raw = raw_preds[pi]
            g = golds[pos]
            pred_for_metrics = raw.strip()
            em, f1v = hotpot_em_and_f1(pred_for_metrics, g)
            ems.append(em)
            f1s.append(f1v)
        fold_em.append(sum(ems) / len(ems) if ems else 0.0)
        fold_f1.append(sum(f1s) / len(f1s) if f1s else 0.0)

    mean_em = float(statistics.mean([float(x) for x in fold_em]))
    std_em = _safe_stdev([float(x) for x in fold_em])
    mean_f1 = float(statistics.mean([float(x) for x in fold_f1]))
    std_f1 = _safe_stdev([float(x) for x in fold_f1])

    pred_rows = []
    for i in range(len(raw_pool)):
        raw = raw_pool[i]
        g = golds[i]
        qq = questions[i]
        pred_for_metrics = raw.strip()
        em, f1v = hotpot_em_and_f1(pred_for_metrics, g)
        pred_rows.append(
            {
                "idx": i,
                "dataset_row_id": pool_indices[i],
                "question": qq,
                "gold": g,
                "raw_generation": raw,
                "completion_for_metrics": pred_for_metrics,
                "em": int(em),
                "f1": round(f1v, 4),
            }
        )
    if save_csv:
        write_prediction_csv(
            save_csv,
            ["idx", "dataset_row_id", "question", "gold", "raw_generation", "completion_for_metrics", "em", "f1"],
            pred_rows,
        )

    return {
        "exact_match": round(mean_em, 4),
        "f1": round(mean_f1, 4),
        "exact_match_mean": round(mean_em, 4),
        "exact_match_std": round(std_em, 4),
        "f1_mean": round(mean_f1, 4),
        "f1_std": round(std_f1, 4),
        "fold_exact_match": [round(float(x), 4) for x in fold_em],
        "fold_f1": [round(float(x), 4) for x in fold_f1],
        "total": len(golds),
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "max_ctx_tokens": max_ctx_tokens,
        "few_shot": few_shot,
        "hotpot_stop_strings": list(HOTPOT_ZIP_STOP_STRINGS),
        "forgetting_codes_zip_protocol": True,
        "hotpot_config": "distractor",
        "eval_note": (
            "forgetting-codes.zip parity: context truncation max_ctx_tokens×4 chars; Untitled/metrics.py normalize+F1; "
            "HF generate(stop_strings on blank-line and Question:) like zip vLLM; optional FEW_SHOT prefix; "
            "tokenizer max_length defaults to zip-style max_model_len 8192 for Llama. Fixed split pool."
        ),
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF split rows (pool_indices); folds index into pool order.",
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
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_csv = kwargs.pop("save_predictions_path", None)
    max_input_tokens = int(kwargs.pop("max_input_tokens", 512))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 96))
    _ = kwargs
    chat_label = chat_model_label(model, tokenizer)
    log(
        f"  Loading XSum (test split); max_input_tokens={max_input_tokens}, "
        f"max_new_tokens={max_new_tokens} (article fills budget)…"
    )
    dataset = load_dataset("EdinburghNLP/xsum", split="test")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "xsum", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    prompts = []
    refs: List[str] = []
    for ex in pool_ds:
        doc = truncate_article_to_token_budget(
            tokenizer, ex["document"], XSUM_PREFIX, XSUM_SUFFIX, max_input_tokens
        )
        prompts.append(build_xsum_prompt(doc, tokenizer=tokenizer, chat_label=chat_label))
        refs.append(ex["summary"].strip())

    preds_pool = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        raw = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_tokens,
        )
        for j, pos in enumerate(fp):
            preds_pool[pos] = normal_cleanup_summary(
                strip_common_generation_prefixes(strip_reasoning_preamble(raw[j]))
            )

    fold_lists, means, stds = rouge_fold_mean_std(fold_positions, preds_pool, refs)

    if save_csv:
        pred_rows = [
            {"idx": i, "dataset_row_id": pool_indices[i], "reference": refs[i], "prediction": preds_pool[i]}
            for i in range(len(preds_pool))
        ]
        write_prediction_csv(save_csv, ["idx", "dataset_row_id", "reference", "prediction"], pred_rows)

    return {
        "rouge1": means["rouge1"],
        "rouge2": means["rouge2"],
        "rougeL": means["rougeL"],
        "rouge1_mean": means["rouge1"],
        "rouge2_mean": means["rouge2"],
        "rougeL_mean": means["rougeL"],
        "rouge1_std": stds["rouge1"],
        "rouge2_std": stds["rouge2"],
        "rougeL_std": stds["rougeL"],
        "fold_rouge1": [round(float(x), 4) for x in fold_lists["rouge1"]],
        "fold_rouge2": [round(float(x), 4) for x in fold_lists["rouge2"]],
        "fold_rougeL": [round(float(x), 4) for x in fold_lists["rougeL"]],
        "total": len(preds_pool),
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF XSum test rows (pool_indices); folds index into pool order.",
    }


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
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_csv = kwargs.pop("save_predictions_path", None)
    max_input_tokens = int(kwargs.pop("max_input_tokens", 512))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
    _ = kwargs
    chat_label = chat_model_label(model, tokenizer)
    log(
        f"  Loading CNN/DailyMail 3.0.0 (test split); max_input_tokens={max_input_tokens}, "
        f"max_new_tokens={max_new_tokens}…"
    )
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "cnndm", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    prompts = []
    refs = []
    for ex in pool_ds:
        art = truncate_article_to_token_budget(
            tokenizer, ex["article"], CNNDM_PREFIX, CNNDM_SUFFIX, max_input_tokens
        )
        prompts.append(build_cnndm_prompt(art, tokenizer=tokenizer, chat_label=chat_label))
        refs.append(ex["highlights"].replace("\n", " ").strip())

    preds_pool = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        raw = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_tokens,
        )
        for j, pos in enumerate(fp):
            preds_pool[pos] = light_cleanup_summary(
                strip_common_generation_prefixes(strip_reasoning_preamble(raw[j]))
            )

    fold_lists, means, stds = rouge_fold_mean_std(fold_positions, preds_pool, refs)

    if save_csv:
        pred_rows = [
            {"idx": i, "dataset_row_id": pool_indices[i], "reference": refs[i], "prediction": preds_pool[i]}
            for i in range(len(preds_pool))
        ]
        write_prediction_csv(save_csv, ["idx", "dataset_row_id", "reference", "prediction"], pred_rows)

    return {
        "rouge1": means["rouge1"],
        "rouge2": means["rouge2"],
        "rougeL": means["rougeL"],
        "rouge1_mean": means["rouge1"],
        "rouge2_mean": means["rouge2"],
        "rougeL_mean": means["rougeL"],
        "rouge1_std": stds["rouge1"],
        "rouge2_std": stds["rouge2"],
        "rougeL_std": stds["rougeL"],
        "fold_rouge1": [round(float(x), 4) for x in fold_lists["rouge1"]],
        "fold_rouge2": [round(float(x), 4) for x in fold_lists["rouge2"]],
        "fold_rougeL": [round(float(x), 4) for x in fold_lists["rougeL"]],
        "total": len(preds_pool),
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF CNN/DM test rows (pool_indices); folds index into pool order.",
    }


# ──────────────────────────────────────────────────────────────────────
#  SST-2 / IMDB / MMLU / SQuAD (aligned with ``eval_catastrophic_forgetting.py`` v1)
# ──────────────────────────────────────────────────────────────────────

SST2_FEW_SHOT_DEFAULT = [
    ("A stirring, funny and finally transporting re-imagining of Beauty and the Beast.", "positive"),
    ("Unflinchingly bleak and desperate.", "negative"),
]


def _pick_sst2_shots(seed: Optional[int]) -> List[Tuple[str, str]]:
    if seed is None:
        return SST2_FEW_SHOT_DEFAULT
    rng = random.Random(seed)
    train = load_dataset("glue", "sst2", split="train")
    pos = [ex for ex in train if ex["label"] == 1]
    neg = [ex for ex in train if ex["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    return [
        (pos[0]["sentence"], "positive"),
        (neg[0]["sentence"], "negative"),
    ]


def build_sst2_prompt(sentence: str, few_shot: List[Tuple[str, str]]) -> str:
    prompt = ""
    for s, label in few_shot:
        prompt += f"Sentence: {s}\nSentiment: {label}\n\n"
    prompt += f"Sentence: {sentence}\nSentiment:"
    return prompt


def _cls_accuracy_first_token(pred_lines: List[str], gold_labels: List[str]) -> float:
    correct = 0
    for pred, gold in zip(pred_lines, gold_labels):
        pred_lower = pred.lower().strip()
        if pred_lower and gold in pred_lower.split()[0]:
            correct += 1
    return correct / len(gold_labels) if gold_labels else 0.0


def eval_sst2(model, tokenizer, max_samples=None, batch_size=256, seed=None, **kwargs):
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_predictions_path = kwargs.pop("save_predictions_path", None)
    max_input_length = int(kwargs.pop("max_input_length", 256))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 4))
    _ = kwargs
    log("  Loading SST-2 (GLUE validation split)...")
    dataset = load_dataset("glue", "sst2", split="validation")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "sst2", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)
    few_shot = _pick_sst2_shots(seed)
    label_map = {0: "negative", 1: "positive"}
    prompts = [build_sst2_prompt(ex["sentence"], few_shot) for ex in pool_ds]
    gold_labels = [label_map[ex["label"]] for ex in pool_ds]

    fold_acc: List[float] = []
    preds_pool: List[str] = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        sub_g = [gold_labels[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        preds = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        for pi, pos in enumerate(fp):
            preds_pool[pos] = preds[pi]
        fold_acc.append(_cls_accuracy_first_token(preds, sub_g))

    mean_acc = float(statistics.mean(fold_acc))
    std_acc = _safe_stdev(fold_acc)
    correct = sum(
        1
        for p, g in zip(preds_pool, gold_labels)
        if p.lower().strip() and g in p.lower().strip().split()[0]
    )
    out: Dict[str, Any] = {
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "accuracy": round(mean_acc, 4),
        "accuracy_mean": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4),
        "fold_accuracy": [round(a, 4) for a in fold_acc],
        "correct": correct,
        "total": len(prompts),
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF split rows (pool_indices); folds index into pool order.",
    }
    if save_predictions_path:
        rows = [
            {"idx": i, "pool_position": i, "dataset_row_id": pool_indices[i], "prediction": preds_pool[i]}
            for i in range(len(preds_pool))
        ]
        write_prediction_csv(save_predictions_path, ["idx", "pool_position", "dataset_row_id", "prediction"], rows)
    return out


IMDB_FEW_SHOT_DEFAULT = [
    (
        "This film was absolutely wonderful. The acting was superb and the plot was gripping from start to finish.",
        "positive",
    ),
    ("Terrible movie. Poor acting, weak plot, and a waste of time.", "negative"),
]


def _pick_imdb_shots(seed: Optional[int]) -> List[Tuple[str, str]]:
    if seed is None:
        return IMDB_FEW_SHOT_DEFAULT
    rng = random.Random(seed)
    train = load_dataset("imdb", split="train")
    pos = [ex for ex in train if ex["label"] == 1]
    neg = [ex for ex in train if ex["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    return [
        (" ".join(pos[0]["text"].split()[:80]), "positive"),
        (" ".join(neg[0]["text"].split()[:80]), "negative"),
    ]


def build_imdb_prompt(text: str, few_shot: List[Tuple[str, str]]) -> str:
    text_truncated = " ".join(text.split()[:200])
    prompt = ""
    for t, label in few_shot:
        prompt += f"Review: {t}\nSentiment: {label}\n\n"
    prompt += f"Review: {text_truncated}\nSentiment:"
    return prompt


def eval_imdb(model, tokenizer, max_samples=None, batch_size=256, seed=None, **kwargs):
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_predictions_path = kwargs.pop("save_predictions_path", None)
    max_input_length = int(kwargs.pop("max_input_length", 512))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 4))
    _ = kwargs
    log("  Loading IMDB (test split)...")
    dataset = load_dataset("imdb", split="test")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "imdb", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)
    few_shot = _pick_imdb_shots(seed)
    label_map = {0: "negative", 1: "positive"}
    prompts = [build_imdb_prompt(ex["text"], few_shot) for ex in pool_ds]
    gold_labels = [label_map[ex["label"]] for ex in pool_ds]

    fold_acc: List[float] = []
    preds_pool = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        sub_g = [gold_labels[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        preds = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        for pi, pos in enumerate(fp):
            preds_pool[pos] = preds[pi]
        fold_acc.append(_cls_accuracy_first_token(preds, sub_g))

    mean_acc = float(statistics.mean(fold_acc))
    std_acc = _safe_stdev(fold_acc)
    correct = sum(
        1
        for p, g in zip(preds_pool, gold_labels)
        if p.lower().strip() and g in p.lower().strip().split()[0]
    )
    out = {
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "accuracy": round(mean_acc, 4),
        "accuracy_mean": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4),
        "fold_accuracy": [round(a, 4) for a in fold_acc],
        "correct": correct,
        "total": len(prompts),
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF split rows (pool_indices); folds index into pool order.",
    }
    if save_predictions_path:
        rows = [{"idx": i, "dataset_row_id": pool_indices[i], "prediction": preds_pool[i]} for i in range(len(preds_pool))]
        write_prediction_csv(save_predictions_path, ["idx", "dataset_row_id", "prediction"], rows)
    return out


def build_mmlu_prompt(subject: str, few_shot_examples: List[Dict[str, Any]], question: str, choices: List[str]) -> str:
    prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
    for ex in few_shot_examples:
        prompt += f"Question: {ex['question']}\n"
        for i, c in enumerate(ex["choices"]):
            prompt += f"{'ABCD'[i]}. {c}\n"
        prompt += f"Answer: {'ABCD'[ex['answer']]}\n\n"
    prompt += f"Question: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {c}\n"
    prompt += "Answer:"
    return prompt


def _mmlu_accuracy(pred_lines: List[str], gold_labels: List[str]) -> float:
    correct = 0
    for pred, gold in zip(pred_lines, gold_labels):
        pred_clean = pred.strip().upper()
        if pred_clean and pred_clean[0] == gold:
            correct += 1
    return correct / len(gold_labels) if gold_labels else 0.0


def eval_mmlu(model, tokenizer, max_samples=None, batch_size=256, seed=None, **kwargs):
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_predictions_path = kwargs.pop("save_predictions_path", None)
    max_input_length = int(kwargs.pop("max_input_length", 1024))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 2))
    _ = kwargs
    log("  Loading MMLU (cais/mmlu all / test split)...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "mmlu", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    dev_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for ex in dev_dataset:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)
    if seed is not None:
        rng = random.Random(seed)
        for subj in dev_by_subject:
            rng.shuffle(dev_by_subject[subj])
    for subj in dev_by_subject:
        dev_by_subject[subj] = dev_by_subject[subj][:5]

    prompts = []
    gold_labels = []
    for ex in pool_ds:
        few_shot = dev_by_subject.get(ex["subject"], [])
        prompts.append(build_mmlu_prompt(ex["subject"], few_shot, ex["question"], ex["choices"]))
        gold_labels.append("ABCD"[ex["answer"]])

    fold_acc: List[float] = []
    preds_pool = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        sub_g = [gold_labels[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        preds = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        for pi, pos in enumerate(fp):
            preds_pool[pos] = preds[pi]
        fold_acc.append(_mmlu_accuracy(preds, sub_g))

    mean_acc = float(statistics.mean(fold_acc))
    std_acc = _safe_stdev(fold_acc)
    correct = sum(
        1
        for p, g in zip(preds_pool, gold_labels)
        if p.strip().upper() and p.strip().upper()[0] == g
    )
    out = {
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "accuracy": round(mean_acc, 4),
        "accuracy_mean": round(mean_acc, 4),
        "accuracy_std": round(std_acc, 4),
        "fold_accuracy": [round(a, 4) for a in fold_acc],
        "correct": correct,
        "total": len(prompts),
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF split rows (pool_indices); folds index into pool order.",
    }
    if save_predictions_path:
        rows = [{"idx": i, "dataset_row_id": pool_indices[i], "prediction": preds_pool[i]} for i in range(len(preds_pool))]
        write_prediction_csv(save_predictions_path, ["idx", "dataset_row_id", "prediction"], rows)
    return out


SQUAD_FEW_SHOT_DEFAULT = [
    {
        "context": (
            "The Normans were the people who in the 10th and 11th centuries gave their name "
            "to Normandy, a region in France."
        ),
        "question": "In what country is Normandy located?",
        "answer": "France",
    },
    {
        "context": "The Amazon rainforest produces more than 20% of the world's oxygen.",
        "question": "What percentage of the world's oxygen does the Amazon produce?",
        "answer": "more than 20%",
    },
]


def _pick_squad_shots(seed: Optional[int]) -> List[Dict[str, str]]:
    if seed is None:
        return SQUAD_FEW_SHOT_DEFAULT
    rng = random.Random(seed)
    train = load_dataset("rajpurkar/squad", split="train")
    indices = list(range(len(train)))
    rng.shuffle(indices)
    shots: List[Dict[str, str]] = []
    for idx in indices:
        ex = train[idx]
        answers = ex.get("answers", {}).get("text", [])
        if not answers:
            continue
        shots.append(
            {
                "context": " ".join(ex["context"].split()[:150]),
                "question": ex["question"],
                "answer": answers[0],
            }
        )
        if len(shots) >= 2:
            break
    return shots


def normalize_answer_squad(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def squad_compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer_squad(prediction).split()
    gold_tokens = normalize_answer_squad(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_squad_prompt(context: str, question: str, few_shot: List[Dict[str, str]]) -> str:
    prompt = ""
    for ex in few_shot:
        prompt += f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    context_truncated = " ".join(context.split()[:300])
    prompt += f"Context: {context_truncated}\nQuestion: {question}\nAnswer:"
    return prompt


def _squad_em_f1_micro(pred_lines: List[str], gold_answers_lists: List[List[str]]) -> Tuple[float, float]:
    exact_matches = 0
    f1_scores: List[float] = []
    for pred, golds in zip(pred_lines, gold_answers_lists):
        pred_clean = pred.split("\n")[0].strip()
        em = any(normalize_answer_squad(pred_clean) == normalize_answer_squad(g) for g in golds)
        f1 = max(squad_compute_f1(pred_clean, g) for g in golds) if golds else 0.0
        if em:
            exact_matches += 1
        f1_scores.append(f1)
    n = len(gold_answers_lists)
    em_rate = exact_matches / n if n else 0.0
    mean_f1 = sum(f1_scores) / n if n else 0.0
    return em_rate, mean_f1


def eval_squad(model, tokenizer, max_samples=None, batch_size=256, seed=None, **kwargs):
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_predictions_path = kwargs.pop("save_predictions_path", None)
    max_input_length = int(kwargs.pop("max_input_length", 1024))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 24))
    _ = kwargs
    log("  Loading SQuAD v1.1 (validation split)...")
    dataset = load_dataset("rajpurkar/squad", split="validation")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "squad", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)
    few_shot = _pick_squad_shots(seed)
    prompts = []
    gold_answers = []
    for ex in pool_ds:
        prompts.append(build_squad_prompt(ex["context"], ex["question"], few_shot))
        gold_answers.append(ex["answers"]["text"])

    fold_em: List[float] = []
    fold_f1: List[float] = []
    preds_pool = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        sub_g = [gold_answers[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        preds = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        for pi, pos in enumerate(fp):
            preds_pool[pos] = preds[pi]
        em_r, mf1 = _squad_em_f1_micro(preds, sub_g)
        fold_em.append(em_r)
        fold_f1.append(mf1)

    mean_em = float(statistics.mean(fold_em))
    std_em = _safe_stdev(fold_em)
    mean_f1 = float(statistics.mean(fold_f1))
    std_f1 = _safe_stdev(fold_f1)
    rows = []
    for i in range(len(preds_pool)):
        pred_clean = preds_pool[i].split("\n")[0].strip()
        rows.append(
            {"idx": i, "dataset_row_id": pool_indices[i], "question": "", "gold": "|".join(gold_answers[i]), "prediction": pred_clean}
        )
    if save_predictions_path:
        for i, ex in enumerate(pool_ds):
            rows[i]["question"] = ex["question"]
        write_prediction_csv(save_predictions_path, ["idx", "dataset_row_id", "question", "gold", "prediction"], rows)

    return {
        "exact_match": round(mean_em, 4),
        "f1": round(mean_f1, 4),
        "exact_match_mean": round(mean_em, 4),
        "exact_match_std": round(std_em, 4),
        "f1_mean": round(mean_f1, 4),
        "f1_std": round(std_f1, 4),
        "fold_exact_match": [round(x, 4) for x in fold_em],
        "fold_f1": [round(x, 4) for x in fold_f1],
        "total": len(prompts),
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HF split rows (pool_indices); folds index into pool order.",
    }


def build_dolly_prompt(instruction: str, context: str, *, tokenizer=None, chat_label: str = "") -> str:
    ctx = (context or "").strip()
    body = (
        "Below is an instruction that describes a task. Write a response that completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
    )
    if ctx:
        body += f"### Context:\n{ctx}\n\n"
    body += "### Response:\n"
    if tokenizer is not None:
        body = maybe_apply_chat_template(
            tokenizer,
            body,
            system_prompt="You are a helpful, honest assistant.",
            chat_label=chat_label,
        )
    return body


def _optional_corpus_bleu(preds: List[str], refs: List[str]) -> Optional[float]:
    try:
        import sacrebleu  # type: ignore

        return float(sacrebleu.corpus_bleu(preds, [refs]).score / 100.0)
    except Exception:
        return None


def _optional_mean_meteor(preds: List[str], refs: List[str]) -> Optional[float]:
    try:
        import nltk  # type: ignore
        from nltk.translate.meteor_score import meteor_score  # type: ignore

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        scores = []
        for p, r in zip(preds, refs):
            ptoks = p.split()
            rtoks = r.split()
            if not ptoks or not rtoks:
                scores.append(0.0)
            else:
                scores.append(float(meteor_score([rtoks], ptoks)))
        return sum(scores) / len(scores) if scores else None
    except Exception:
        return None


def eval_dolly(model, tokenizer, max_samples=None, batch_size=64, seed=None, **kwargs):
    _ = seed
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_csv = kwargs.pop("save_predictions_path", None)
    max_input_length = int(kwargs.pop("max_input_length", 1024))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
    _ = kwargs
    chat_label = chat_model_label(model, tokenizer)
    log(
        f"  Loading Databricks Dolly 15k (train split, reference responses); "
        f"max_input_length={max_input_length}, max_new_tokens={max_new_tokens}…"
    )
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "dolly", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    prompts = []
    refs = []
    for ex in pool_ds:
        prompts.append(build_dolly_prompt(ex["instruction"], ex.get("context") or "", tokenizer=tokenizer, chat_label=chat_label))
        refs.append(str(ex["response"]).strip())

    preds_pool = [""] * len(prompts)
    log(
        f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}) on train split; "
        f"pool_size={len(prompts)}..."
    )
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        raw = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        for j, pos in enumerate(fp):
            preds_pool[pos] = strip_common_generation_prefixes(strip_reasoning_preamble(raw[j])).strip()

    fold_lists, means, stds = rouge_fold_mean_std(fold_positions, preds_pool, refs)

    out: Dict[str, Any] = {
        "rouge1": means["rouge1"],
        "rouge2": means["rouge2"],
        "rougeL": means["rougeL"],
        "rouge1_mean": means["rouge1"],
        "rouge2_mean": means["rouge2"],
        "rougeL_mean": means["rougeL"],
        "rouge1_std": stds["rouge1"],
        "rouge2_std": stds["rouge2"],
        "rougeL_std": stds["rougeL"],
        "fold_rouge1": [round(float(x), 4) for x in fold_lists["rouge1"]],
        "fold_rouge2": [round(float(x), 4) for x in fold_lists["rouge2"]],
        "fold_rougeL": [round(float(x), 4) for x in fold_lists["rougeL"]],
        "total": len(preds_pool),
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "bleu": _optional_corpus_bleu(preds_pool, refs),
        "meteor_mean": _optional_mean_meteor(preds_pool, refs),
        "metric_note": "Paper BLEU/ROUGE/METEOR; pairwise NLG metrics vs reference (not human ratings). BLEU/METEOR over full pool.",
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to Dolly train rows (pool_indices); folds index into pool order.",
    }
    if save_csv:
        rows = [
            {"idx": i, "dataset_row_id": pool_indices[i], "reference": refs[i], "prediction": preds_pool[i]}
            for i in range(len(preds_pool))
        ]
        write_prediction_csv(save_csv, ["idx", "dataset_row_id", "reference", "prediction"], rows)
    return out


def build_alpaca_prompt(instruction: str, inp: str, *, tokenizer=None, chat_label: str = "") -> str:
    body = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
    )
    if (inp or "").strip():
        body += f"### Input:\n{inp.strip()}\n\n"
    body += "### Response:\n"
    if tokenizer is not None:
        body = maybe_apply_chat_template(
            tokenizer,
            body,
            system_prompt="You are a helpful assistant.",
            chat_label=chat_label,
        )
    return body


def eval_alpaca(model, tokenizer, max_samples=None, batch_size=64, seed=None, **kwargs):
    _ = seed
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_csv = kwargs.pop("save_predictions_path", None)
    max_input_length = int(kwargs.pop("max_input_length", 1024))
    max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
    _ = kwargs
    chat_label = chat_model_label(model, tokenizer)
    log(
        f"  Loading Stanford Alpaca (tatsu-lab/alpaca train split, reference outputs); "
        f"max_input_length={max_input_length}, max_new_tokens={max_new_tokens}…"
    )
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "alpaca", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    prompts = []
    refs = []
    for ex in pool_ds:
        prompts.append(build_alpaca_prompt(ex["instruction"], ex.get("input") or "", tokenizer=tokenizer, chat_label=chat_label))
        refs.append(str(ex["output"]).strip())

    preds_pool = [""] * len(prompts)
    log(
        f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}) on train split; "
        f"pool_size={len(prompts)}..."
    )
    for fi, fp in enumerate(fold_positions):
        sub_p = [prompts[i] for i in fp]
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} examples...")
        raw = generate_batch(
            model,
            tokenizer,
            sub_p,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            max_input_length=max_input_length,
        )
        for j, pos in enumerate(fp):
            preds_pool[pos] = strip_common_generation_prefixes(strip_reasoning_preamble(raw[j])).strip()

    fold_lists, means, stds = rouge_fold_mean_std(fold_positions, preds_pool, refs)

    out = {
        "rouge1": means["rouge1"],
        "rouge2": means["rouge2"],
        "rougeL": means["rougeL"],
        "rouge1_mean": means["rouge1"],
        "rouge2_mean": means["rouge2"],
        "rougeL_mean": means["rougeL"],
        "rouge1_std": stds["rouge1"],
        "rouge2_std": stds["rouge2"],
        "rougeL_std": stds["rougeL"],
        "fold_rouge1": [round(float(x), 4) for x in fold_lists["rouge1"]],
        "fold_rouge2": [round(float(x), 4) for x in fold_lists["rouge2"]],
        "fold_rougeL": [round(float(x), 4) for x in fold_lists["rougeL"]],
        "total": len(preds_pool),
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "bleu": _optional_corpus_bleu(preds_pool, refs),
        "meteor_mean": _optional_mean_meteor(preds_pool, refs),
        "metric_note": "Alpaca pairwise win-rate needs an LLM judge; here ROUGE (+ optional BLEU/METEOR) vs references on train split. BLEU/METEOR over full pool.",
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to Alpaca train rows (pool_indices); folds index into pool order.",
    }
    if save_csv:
        rows = [
            {"idx": i, "dataset_row_id": pool_indices[i], "reference": refs[i], "prediction": preds_pool[i]}
            for i in range(len(preds_pool))
        ]
        write_prediction_csv(save_csv, ["idx", "dataset_row_id", "reference", "prediction"], rows)
    return out


def clean_mbpp_completion(text: str) -> str:
    text = strip_reasoning_preamble((text or "").strip())
    if "```" in text:
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    text = strip_common_generation_prefixes(text)
    return text.strip()


def mbpp_exec_checks(
    code_str: str,
    test_imports: List[str],
    test_list: List[str],
    *,
    timeout_s: float = 30.0,
) -> Tuple[bool, str]:
    """Run MBPP imports + candidate code + asserts in one namespace.

    Uses ``SIGALRM`` / ``setitimer`` on POSIX so buggy generated code cannot spin forever
    (HumanEval uses ``human_eval``'s timeout; MBPP had none).
    """

    class MBPPExecTimeout(Exception):
        pass

    def _handler(signum, frame):
        raise MBPPExecTimeout(f"timeout after {timeout_s}s")

    g: Dict[str, Any] = {}
    use_timer = bool(timeout_s and timeout_s > 0 and hasattr(signal, "SIGALRM"))
    try:
        if use_timer:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        try:
            for imp in test_imports or []:
                exec(imp, g)
            exec(compile(code_str, "<mbpp>", "exec"), g)
            for t in test_list:
                exec(t, g)
            return True, ""
        finally:
            if use_timer:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
    except MBPPExecTimeout as e:
        return False, repr(e)
    except Exception as e:
        return False, repr(e)


def _eval_mbpp_sanitized(
    model,
    tokenizer,
    max_samples=None,
    batch_size=8,
    seed=None,
    *,
    save_predictions_path=None,
    max_new_tokens: int = 512,
    max_input_length: int = 2048,
    eval_split_seed: int = 42,
    eval_num_folds: int = 3,
    eval_split_indices_dir: str = "eval_split_indices",
):
    """MBPP sanitized ``test`` split — shared by ``codefeedback`` and ``oci``."""
    _ = seed
    save_csv = save_predictions_path
    log("  Loading MBPP sanitized (google-research-datasets/mbpp, split=test)...")
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "mbpp", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    prompts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for ex in pool_ds:
        task_prompt = (
            "Use Python 3. Write a complete solution that passes the asserts.\n\n"
            f"Task:\n{ex['prompt'].strip()}\n\nCode:\n"
        )
        prompts.append(task_prompt)
        meta.append({"task_id": ex["task_id"], "test_imports": list(ex.get("test_imports") or []), "test_list": list(ex["test_list"])})
    bs = max(1, min(batch_size, 16))
    preds_pool: List[str] = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} tasks...")
        for i in range(0, len(fp), bs):
            chunk_idx = fp[i : i + bs]
            chunk = [prompts[j] for j in chunk_idx]
            chunk_preds = generate_batch(
                model,
                tokenizer,
                chunk,
                max_new_tokens=max_new_tokens,
                batch_size=len(chunk),
                max_input_length=max_input_length,
            )
            for j, pos in enumerate(chunk_idx):
                preds_pool[pos] = chunk_preds[j]

    passed_flags: List[bool] = []
    rows: List[Dict[str, Any]] = []
    for pos in range(len(meta)):
        m = meta[pos]
        raw = preds_pool[pos]
        code = clean_mbpp_completion(raw)
        ok, err = mbpp_exec_checks(code, m["test_imports"], m["test_list"])
        passed_flags.append(ok)
        rows.append(
            {
                "dataset_row_id": pool_indices[pos],
                "task_id": m["task_id"],
                "passed": ok,
                "exec_error": err,
                "completion": code[:4000],
                "raw_generation": raw[:4000],
            }
        )

    fold_pass_at_1: List[float] = []
    for fp in fold_positions:
        pp = sum(1 for i in fp if passed_flags[i])
        fold_pass_at_1.append(pp / len(fp) if fp else 0.0)

    mean_p = float(statistics.mean(fold_pass_at_1)) if fold_pass_at_1 else 0.0
    std_p = _safe_stdev([float(x) for x in fold_pass_at_1])
    passed_total = sum(passed_flags)

    if save_csv:
        write_prediction_csv(
            save_csv,
            ["dataset_row_id", "task_id", "passed", "exec_error", "completion", "raw_generation"],
            rows,
        )

    n = len(meta)
    return {
        "pass_at_1": round(mean_p, 4),
        "pass_at_1_mean": round(mean_p, 4),
        "pass_at_1_std": round(std_p, 4),
        "fold_pass_at_1": [round(float(x), 4) for x in fold_pass_at_1],
        "passed": passed_total,
        "total": n,
        "dataset": "google-research-datasets/mbpp/sanitized/test",
        "metric_note": "Paper CodeFeedback ≠ MBPP; same spirit (exec vs unit tests). pass_at_1 is mean pass rate across folds.",
        "max_new_tokens": max_new_tokens,
        "max_input_length": max_input_length,
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to MBPP sanitized test rows (pool_indices); folds index into pool order.",
    }


def eval_codefeedback(model, tokenizer, max_samples=None, batch_size=8, seed=None, **kwargs):
    """MBPP sanitized ``test`` only (alias for tables that call this benchmark "CodeFeedback-proxy")."""
    save_csv = kwargs.pop("save_predictions_path", None)
    max_new_tokens = int(kwargs.pop("max_new_tokens", 512))
    max_input_length = int(kwargs.pop("max_input_length", 2048))
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    _ = kwargs
    return _eval_mbpp_sanitized(
        model,
        tokenizer,
        max_samples=max_samples,
        batch_size=batch_size,
        seed=seed,
        save_predictions_path=save_csv,
        max_new_tokens=max_new_tokens,
        max_input_length=max_input_length,
        eval_split_seed=eval_split_seed,
        eval_num_folds=eval_num_folds,
        eval_split_indices_dir=eval_split_indices_dir,
    )


def eval_oci(model, tokenizer, max_samples=None, batch_size=8, seed=None, **kwargs):
    """OpenCodeInterpreter-style coding suite: **HumanEval** + **MBPP** pass@1 (same benchmarks OCI papers report).

    Run with ``--benchmarks oci``. Uses separate sample caps via ``humaneval_max_samples`` /
    ``mbpp_max_samples`` when set; otherwise falls back to ``--max-samples`` for each leg.
    """
    save_base = kwargs.pop("save_predictions_path", None)
    mbpp_new_toks = int(kwargs.pop("max_new_tokens", 512))
    mbpp_max_in = int(kwargs.pop("mbpp_max_input_length", 2048))
    he_cap = kwargs.pop("humaneval_max_samples", None)
    mbpp_cap = kwargs.pop("mbpp_max_samples", None)
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    he_max_nt = int(kwargs.pop("humaneval_max_new_tokens", 512))
    he_max_in = int(kwargs.pop("humaneval_max_input_length", 2048))
    _ = kwargs

    he_csv = mbpp_csv = None
    if save_base:
        he_csv = save_base.replace(".csv", "_humaneval.csv") if save_base.endswith(".csv") else save_base + "_humaneval.csv"
        mbpp_csv = save_base.replace(".csv", "_mbpp.csv") if save_base.endswith(".csv") else save_base + "_mbpp.csv"

    log(
        "  OCI coding suite (HumanEval + MBPP sanitized), aligned with OpenCodeInterpreter-style HE+MBPP reporting..."
    )

    he_limit = he_cap if he_cap is not None else max_samples
    mbpp_limit = mbpp_cap if mbpp_cap is not None else max_samples

    he_res = eval_humaneval(
        model,
        tokenizer,
        max_samples=he_limit,
        batch_size=batch_size,
        seed=seed,
        save_predictions_path=he_csv,
        eval_split_seed=eval_split_seed,
        eval_num_folds=eval_num_folds,
        eval_split_indices_dir=eval_split_indices_dir,
        max_new_tokens=he_max_nt,
        max_input_length=he_max_in,
    )
    mbpp_res = _eval_mbpp_sanitized(
        model,
        tokenizer,
        max_samples=mbpp_limit,
        batch_size=batch_size,
        seed=seed,
        save_predictions_path=mbpp_csv,
        max_new_tokens=mbpp_new_toks,
        max_input_length=mbpp_max_in,
        eval_split_seed=eval_split_seed,
        eval_num_folds=eval_num_folds,
        eval_split_indices_dir=eval_split_indices_dir,
    )

    fold_he = [float(x) for x in he_res["fold_pass_at_1"]]
    fold_mbpp = [float(x) for x in mbpp_res["fold_pass_at_1"]]
    if len(fold_he) != len(fold_mbpp):
        log(
            f"  WARNING: HumanEval folds ({len(fold_he)}) != MBPP folds ({len(fold_mbpp)}); "
            "OCI fold pairing uses the shorter list (check --max-samples / HE/MBPP caps)."
        )
    n_oci = min(len(fold_he), len(fold_mbpp))
    fold_oci = [(fold_he[i] + fold_mbpp[i]) / 2.0 for i in range(n_oci)]
    oci_mean = float(statistics.mean(fold_oci)) if fold_oci else 0.0
    oci_std = _safe_stdev([float(x) for x in fold_oci])

    return {
        "oci_mean_pass_at_1": round(oci_mean, 4),
        "oci_mean_pass_at_1_std": round(oci_std, 4),
        "fold_oci_mean_pass_at_1": [round(float(x), 4) for x in fold_oci],
        "humaneval_pass_at_1": he_res["pass_at_1"],
        "humaneval_pass_at_1_std": he_res.get("pass_at_1_std"),
        "fold_humaneval_pass_at_1": he_res.get("fold_pass_at_1"),
        "humaneval_split_index_path": he_res.get("split_index_path"),
        "mbpp_pass_at_1": mbpp_res["pass_at_1"],
        "mbpp_pass_at_1_std": mbpp_res.get("pass_at_1_std"),
        "fold_mbpp_pass_at_1": mbpp_res.get("fold_pass_at_1"),
        "mbpp_split_index_path": mbpp_res.get("split_index_path"),
        "humaneval": he_res,
        "mbpp": mbpp_res,
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "metric_note": (
            "OCI/OpenCodeInterpreter papers report HumanEval + MBPP (often separately); "
            "oci_mean_pass_at_1 is the mean across folds of (HE_fold + MBPP_fold)/2; "
            "reported HE/MBPP pass@1 are fold-means."
        ),
    }


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
    eval_split_seed = int(kwargs.pop("eval_split_seed", 42))
    eval_num_folds = int(kwargs.pop("eval_num_folds", 3))
    eval_split_indices_dir = kwargs.pop("eval_split_indices_dir", "eval_split_indices")
    save_csv = kwargs.pop("save_predictions_path", None)
    max_new_tokens = int(kwargs.pop("max_new_tokens", 512))
    max_input_length = int(kwargs.pop("max_input_length", 2048))
    _ = kwargs
    log(
        f"  Loading HumanEval (test); max_input_length={max_input_length}, max_new_tokens={max_new_tokens}..."
    )
    dataset = load_dataset("openai/openai_humaneval", split="test")
    n_total = len(dataset)
    pool_indices, fold_positions, split_path, _meta = load_or_create_eval_split(
        "humaneval", n_total, max_samples, eval_num_folds, eval_split_seed, eval_split_indices_dir
    )
    pool_ds = dataset.select(pool_indices)

    # Benchmark-faithful: always the dataset stub only (no chat template, no system or extra instructions).
    prompts = [ex["prompt"] for ex in pool_ds]
    problems = [
        {
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "entry_point": ex["entry_point"],
            "test": ex["test"],
        }
        for ex in pool_ds
    ]

    bs = max(1, min(batch_size, 16))
    preds_pool: List[str] = [""] * len(prompts)
    log(f"  K-fold ({eval_num_folds} folds, eval_split_seed={eval_split_seed}); pool_size={len(prompts)}...")
    for fi, fp in enumerate(fold_positions):
        log(f"    Fold {fi + 1}/{len(fold_positions)}: {len(fp)} tasks...")
        sub_prompts = [prompts[i] for i in fp]
        for i in range(0, len(sub_prompts), bs):
            chunk = sub_prompts[i : i + bs]
            chunk_preds = generate_batch(
                model,
                tokenizer,
                chunk,
                max_new_tokens=max_new_tokens,
                batch_size=len(chunk),
                max_input_length=max_input_length,
            )
            for j, pool_pos in enumerate(fp[i : i + bs]):
                preds_pool[pool_pos] = chunk_preds[j]

    timeout = 10.0
    passed_flags: List[bool] = []
    he_rows: List[Dict[str, Any]] = []
    for pos in range(len(problems)):
        prob = problems[pos]
        raw = preds_pool[pos]
        comp = clean_humaneval_completion(raw)
        res = check_correctness(prob, comp, timeout=timeout)
        ok = bool(res.get("passed"))
        passed_flags.append(ok)
        he_rows.append(
            {
                "dataset_row_id": pool_indices[pos],
                "task_id": prob["task_id"],
                "raw_generation": raw,
                "completion_passed_to_eval": comp,
                "passed": ok,
                "exec_detail": str(res.get("result", "")),
            }
        )

    fold_pass_at_1: List[float] = []
    for fp in fold_positions:
        pp = sum(1 for i in fp if passed_flags[i])
        fold_pass_at_1.append(pp / len(fp) if fp else 0.0)

    mean_p = float(statistics.mean(fold_pass_at_1)) if fold_pass_at_1 else 0.0
    std_p = _safe_stdev([float(x) for x in fold_pass_at_1])
    passed_total = sum(passed_flags)

    if save_csv:
        write_prediction_csv(
            save_csv,
            [
                "dataset_row_id",
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
        "pass_at_1": round(mean_p, 4),
        "pass_at_1_mean": round(mean_p, 4),
        "pass_at_1_std": round(std_p, 4),
        "fold_pass_at_1": [round(float(x), 4) for x in fold_pass_at_1],
        "passed": passed_total,
        "total": n,
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "eval_split_seed": eval_split_seed,
        "eval_num_folds": eval_num_folds,
        "split_index_path": split_path,
        "split_note": "indices relative to HumanEval test rows (pool_indices); folds index into pool order.",
    }


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

# Default order: classification / QA first, summarization / IF, then code (heavy).
DEFAULT_BENCHMARK_KEYS = (
    "sst2",
    "imdb",
    "mmlu",
    "squad",
    "hotpot",
    "xsum",
    "cnndm",
    "dolly",
    "alpaca",
    "humaneval",
    "codefeedback",
)

BENCHMARK_MAP = {
    "sst2": ("SST-2", eval_sst2),
    "imdb": ("IMDB", eval_imdb),
    "mmlu": ("MMLU", eval_mmlu),
    "squad": ("SQuAD-v1.1", eval_squad),
    "hotpot": ("HotpotQA", eval_hotpot),
    "xsum": ("XSum", eval_xsum),
    "cnndm": ("CNN/DM", eval_cnndm),
    "dolly": ("Databricks-Dolly-15k", eval_dolly),
    "alpaca": ("Stanford-Alpaca", eval_alpaca),
    "humaneval": ("HumanEval", eval_humaneval),
    "codefeedback": ("MBPP-sanitized (CodeFeedback-proxy)", eval_codefeedback),
    "oci": ("OCI suite (HumanEval + MBPP)", eval_oci),
}


def _paths_same(a: str, b: str) -> bool:
    try:
        return os.path.abspath(os.path.expanduser(str(a))) == os.path.abspath(os.path.expanduser(str(b)))
    except Exception:
        return str(a) == str(b)


def save_cf3_report_atomic(report_path: str, payload: Dict[str, Any]) -> None:
    """Atomic JSON write so crashes mid-write leave the previous checkpoint intact."""
    path = os.path.abspath(report_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def parse_benchmark_batch_overrides(specs: List[str]) -> Dict[str, int]:
    """Parse ``--benchmark-batch-size KEY=N`` repeatable CLI entries."""
    out: Dict[str, int] = {}
    for raw in specs:
        part = raw.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --benchmark-batch-size {raw!r}; expected KEY=N")
        key, _, val = part.partition("=")
        key = key.strip()
        val = val.strip()
        if key not in BENCHMARK_MAP:
            raise ValueError(f"Unknown benchmark key {key!r} in --benchmark-batch-size (allowed: {sorted(BENCHMARK_MAP)})")
        try:
            n = int(val, 10)
        except ValueError as e:
            raise ValueError(f"Invalid integer batch in --benchmark-batch-size {raw!r}") from e
        if n < 1:
            raise ValueError(f"Batch must be >= 1 for benchmark {key!r}")
        out[key] = n
    return out


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
    parser = argparse.ArgumentParser(
        description=(
            "Catastrophic forgetting evaluation v3 — default 11 benchmarks; add ``oci`` for HE+MBPP suite."
        )
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_KEYS),
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
    parser.add_argument(
        "--mbpp-max-samples",
        type=int,
        default=None,
        help="Cap MBPP / codefeedback tasks only (also used for the MBPP leg of ``oci``).",
    )
    parser.add_argument(
        "--alpaca-max-samples",
        type=int,
        default=20000,
        help=(
            "Cap Alpaca train split pool (default 20000). Use 52002 for the full split. "
            "When ``--max-samples`` is also set, the smaller cap applies."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--benchmark-batch-size",
        action="append",
        default=[],
        metavar="KEY=N",
        help=(
            "Override global --batch-size for one benchmark key only (repeat flag). "
            "Example: --benchmark-batch-size codefeedback=32 --benchmark-batch-size humaneval=16"
        ),
    )
    parser.add_argument("--is-lora", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Few-shot sampling for SST/IMDB/MMLU/SQuAD + MMLU dev shuffle.",
    )
    parser.add_argument(
        "--eval-split-seed",
        type=int,
        default=42,
        help="Seed for eval example pool + k-fold splits (separate from --seed).",
    )
    parser.add_argument(
        "--eval-num-folds",
        type=int,
        default=3,
        help="Number of disjoint eval folds for mean±std (default 3).",
    )
    parser.add_argument(
        "--eval-split-indices-dir",
        type=str,
        default="eval_split_indices",
        help="Directory for cached eval split JSON (pool_indices + fold_positions).",
    )
    parser.add_argument("--sst2-max-input-length", type=int, default=256, help="SST-2: tokenizer truncation budget.")
    parser.add_argument("--sst2-max-new-tokens", type=int, default=4)
    parser.add_argument("--imdb-max-input-length", type=int, default=512)
    parser.add_argument("--imdb-max-new-tokens", type=int, default=4)
    parser.add_argument("--mmlu-max-input-length", type=int, default=1024)
    parser.add_argument("--mmlu-max-new-tokens", type=int, default=2)
    parser.add_argument("--squad-max-input-length", type=int, default=1024)
    parser.add_argument("--squad-max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--hotpot-max-input-length",
        type=int,
        default=8192,
        help=(
            "HotpotQA: tokenizer max_length for full prompt after zip-style context truncation "
            "(forgetting-codes.zip uses max_model_len 8192 for Llama)."
        ),
    )
    parser.add_argument(
        "--hotpot-max-new-tokens",
        type=int,
        default=256,
        help=(
            "HotpotQA: greedy decode budget. forgetting-codes.zip evaluate.py uses EvalConfig.DEFAULT_MAX_NEW_TOKENS (256)."
        ),
    )
    parser.add_argument(
        "--hotpot-max-ctx-tokens",
        type=int,
        default=1500,
        help=(
            "HotpotQA: DATASETS['hotpotqa']['max_ctx_tokens'] in forgetting-codes.zip; "
            "context truncated to max_ctx_tokens×4 characters before building the prompt."
        ),
    )
    parser.add_argument(
        "--no-hotpot-few-shot",
        action="store_true",
        help=(
            "Disable forgetting-codes.zip FEW_SHOT_EXAMPLES['hotpotqa'] prefix for HotpotQA (few-shot is ON by default)."
        ),
    )
    parser.add_argument("--xsum-max-input-tokens", type=int, default=512)
    parser.add_argument("--xsum-max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--cnndm-max-input-tokens",
        type=int,
        default=512,
        help="CNN/DM: article truncation token budget (faster smoke default).",
    )
    parser.add_argument("--cnndm-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--dolly-max-input-length",
        type=int,
        default=1024,
        help="Dolly: tokenizer max_length for prompt (faster smoke default).",
    )
    parser.add_argument("--dolly-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--alpaca-max-input-length",
        type=int,
        default=1024,
        help="Alpaca: tokenizer max_length for prompt (faster smoke default).",
    )
    parser.add_argument("--alpaca-max-new-tokens", type=int, default=128)
    parser.add_argument("--humaneval-max-input-length", type=int, default=2048)
    parser.add_argument("--humaneval-max-new-tokens", type=int, default=512)
    parser.add_argument("--mbpp-max-input-length", type=int, default=2048)
    parser.add_argument("--mbpp-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--merge-existing-report",
        action="store_true",
        help=(
            "Ignored at write time (reports are merged on startup when the checkpoint matches "
            "--model / --model-name). Kept for grid launchers that still pass this flag."
        ),
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help=(
            "Delete catastrophic_forgetting3_report.json under --output-dir before running "
            "(otherwise a matching checkpoint is resumed and finished benchmarks are skipped)."
        ),
    )
    parser.add_argument(
        "--save-predictions-dir",
        type=str,
        default=None,
        help="Write per-benchmark CSVs: {benchmark_key}_predictions.csv",
    )
    parser.add_argument(
        "--chat-template",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "auto: chat template when tokenizer has one (skip for plain-base checkpoints); "
            "on/off: force."
        ),
    )
    args = parser.parse_args()

    try:
        benchmark_batch_overrides = parse_benchmark_batch_overrides(args.benchmark_batch_size)
    except ValueError as e:
        raise SystemExit(str(e))

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
        args.output_dir = f"catastrophic-forgetting3-{model_name}"

    log(f"\n{'='*60}")
    log("  CATASTROPHIC FORGETTING EVALUATION v3 (single-file runner)")
    log(f"{'='*60}")
    log(f"  Model:      {args.model}")
    log(f"  Name:       {model_name}")
    log(f"  LoRA:       {args.is_lora}")
    log(f"  Seed:       {args.seed}")
    log(f"  Eval split: seed={args.eval_split_seed}, folds={args.eval_num_folds}, dir={args.eval_split_indices_dir}")
    log(f"  Chat tmpl:  {args.chat_template}")
    log(f"  Benchmarks: {', '.join(args.benchmarks)} ({len(args.benchmarks)} tasks)")
    log(f"  Batch size: {args.batch_size}")
    if benchmark_batch_overrides:
        log(f"  Per-benchmark batch overrides: {benchmark_batch_overrides}")
    if args.max_samples:
        log(f"  Max samples: {args.max_samples}")
    if args.humaneval_max_samples is not None:
        log(f"  HumanEval max samples: {args.humaneval_max_samples}")
    if args.mbpp_max_samples is not None:
        log(f"  MBPP max samples:      {args.mbpp_max_samples}")
    if "sst2" in args.benchmarks:
        log(f"  SST-2:       max_input_length={args.sst2_max_input_length}, max_new_tokens={args.sst2_max_new_tokens}")
    if "imdb" in args.benchmarks:
        log(f"  IMDB:        max_input_length={args.imdb_max_input_length}, max_new_tokens={args.imdb_max_new_tokens}")
    if "mmlu" in args.benchmarks:
        log(f"  MMLU:        max_input_length={args.mmlu_max_input_length}, max_new_tokens={args.mmlu_max_new_tokens}")
    if "squad" in args.benchmarks:
        log(f"  SQuAD:       max_input_length={args.squad_max_input_length}, max_new_tokens={args.squad_max_new_tokens}")
    if "hotpot" in args.benchmarks:
        log(
            f"  HotpotQA (forgetting-codes.zip): max_ctx_tokens={args.hotpot_max_ctx_tokens}, "
            f"tokenizer max_length={args.hotpot_max_input_length}, max_new_tokens={args.hotpot_max_new_tokens}, "
            f"few_shot={not bool(args.no_hotpot_few_shot)}"
        )
    if "xsum" in args.benchmarks:
        log(
            f"  XSum:        max_input_tokens={args.xsum_max_input_tokens}, "
            f"max_new_tokens={args.xsum_max_new_tokens}"
        )
    if "cnndm" in args.benchmarks:
        log(
            f"  CNN/DM:      max_input_tokens={args.cnndm_max_input_tokens}, "
            f"max_new_tokens={args.cnndm_max_new_tokens}"
        )
    if "dolly" in args.benchmarks:
        log(
            f"  Dolly:       max_input_length={args.dolly_max_input_length}, "
            f"max_new_tokens={args.dolly_max_new_tokens}"
        )
    if "alpaca" in args.benchmarks:
        log(
            f"  Alpaca:      max_input_length={args.alpaca_max_input_length}, "
            f"max_new_tokens={args.alpaca_max_new_tokens}, "
            f"pool_cap={args.alpaca_max_samples} (train rows)"
        )
    if "humaneval" in args.benchmarks:
        log(
            f"  HumanEval:   max_input_length={args.humaneval_max_input_length}, "
            f"max_new_tokens={args.humaneval_max_new_tokens}"
        )
    if "codefeedback" in args.benchmarks:
        log(
            f"  MBPP proxy:  max_input_length={args.mbpp_max_input_length}, "
            f"max_new_tokens={args.mbpp_max_new_tokens}"
        )
    if "oci" in args.benchmarks:
        log(
            f"  OCI suite:   HE sample cap={args.humaneval_max_samples}, "
            f"MBPP sample cap={args.mbpp_max_samples}; "
            f"HE max_in={args.humaneval_max_input_length}, HE max_new={args.humaneval_max_new_tokens}; "
            f"MBPP max_in={args.mbpp_max_input_length}, MBPP max_new={args.mbpp_max_new_tokens}"
        )
    log(f"  Output:     {args.output_dir}")
    if args.save_predictions_dir:
        log(f"  Predictions CSV dir: {args.save_predictions_dir}")
    log(f"{'='*60}\n")

    results = {
        "eval_version": "v3",
        "implementation_file": "eval_catastrophic_forgetting3.py",
        "model": args.model,
        "model_name": model_name,
        "seed": args.seed,
        "eval_split_seed": args.eval_split_seed,
        "eval_num_folds": args.eval_num_folds,
        "eval_split_indices_dir": os.path.abspath(args.eval_split_indices_dir),
        "benchmarks": {},
    }
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.abspath(os.path.join(args.output_dir, "catastrophic_forgetting3_report.json"))

    if args.overwrite_report:
        if os.path.isfile(report_path):
            os.remove(report_path)
            log(f"  Removed existing report (--overwrite-report): {report_path}")
    elif os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            prev = json.load(f)
        pm = prev.get("model")
        pn = prev.get("model_name")
        if pm is None or pn is None:
            raise SystemExit(f"Checkpoint report missing model metadata: {report_path}")
        if not _paths_same(str(pm), args.model) or pn != model_name:
            raise SystemExit(
                "Checkpoint report model mismatch vs this run:\n"
                f"  report: {pm!r} / {pn!r}\n"
                f"  run:    {args.model!r} / {model_name!r}\n"
                "Delete the JSON, pick another --output-dir, or pass --overwrite-report."
            )
        pb = prev.get("benchmarks") or {}
        if pb:
            results["benchmarks"] = dict(pb)
            log(f"  Loaded checkpoint ({len(pb)} benchmark(s)) from {report_path}")

    need_eval = any(bk not in results["benchmarks"] for bk in args.benchmarks)
    model = None
    tokenizer = None
    if need_eval:
        model, tokenizer = load_model_and_tokenizer(args)

    for bench_key in args.benchmarks:
        if bench_key in results["benchmarks"]:
            bench_skip_name = BENCHMARK_MAP[bench_key][0]
            log(f">>> Skipping {bench_skip_name} (checkpoint already has result).")
            log("")
            continue
        bench_name, eval_fn = BENCHMARK_MAP[bench_key]
        assert model is not None and tokenizer is not None
        bs_use = benchmark_batch_overrides.get(bench_key, args.batch_size)
        log(f">>> Evaluating {bench_name}...")
        if bs_use != args.batch_size:
            log(f"  Using batch_size={bs_use} for '{bench_key}' (override; global={args.batch_size}).")
        t0 = time.time()
        extra: Dict[str, Any] = {
            "eval_split_seed": args.eval_split_seed,
            "eval_num_folds": args.eval_num_folds,
            "eval_split_indices_dir": args.eval_split_indices_dir,
        }
        if bench_key == "sst2":
            extra["max_input_length"] = args.sst2_max_input_length
            extra["max_new_tokens"] = args.sst2_max_new_tokens
        if bench_key == "imdb":
            extra["max_input_length"] = args.imdb_max_input_length
            extra["max_new_tokens"] = args.imdb_max_new_tokens
        if bench_key == "mmlu":
            extra["max_input_length"] = args.mmlu_max_input_length
            extra["max_new_tokens"] = args.mmlu_max_new_tokens
        if bench_key == "squad":
            extra["max_input_length"] = args.squad_max_input_length
            extra["max_new_tokens"] = args.squad_max_new_tokens
        if bench_key == "hotpot":
            extra["max_input_length"] = args.hotpot_max_input_length
            extra["max_new_tokens"] = args.hotpot_max_new_tokens
            extra["max_ctx_tokens"] = args.hotpot_max_ctx_tokens
            extra["few_shot"] = not bool(args.no_hotpot_few_shot)
        if bench_key == "xsum":
            extra["max_input_tokens"] = args.xsum_max_input_tokens
            extra["max_new_tokens"] = args.xsum_max_new_tokens
        if bench_key == "cnndm":
            extra["max_input_tokens"] = args.cnndm_max_input_tokens
            extra["max_new_tokens"] = args.cnndm_max_new_tokens
        if bench_key == "dolly":
            extra["max_input_length"] = args.dolly_max_input_length
            extra["max_new_tokens"] = args.dolly_max_new_tokens
        if bench_key == "alpaca":
            extra["max_input_length"] = args.alpaca_max_input_length
            extra["max_new_tokens"] = args.alpaca_max_new_tokens
        if bench_key == "humaneval":
            extra["max_input_length"] = args.humaneval_max_input_length
            extra["max_new_tokens"] = args.humaneval_max_new_tokens
        if bench_key == "codefeedback":
            extra["max_input_length"] = args.mbpp_max_input_length
            extra["max_new_tokens"] = args.mbpp_max_new_tokens
        if bench_key == "oci":
            extra["max_new_tokens"] = args.mbpp_max_new_tokens
            extra["mbpp_max_input_length"] = args.mbpp_max_input_length
            extra["humaneval_max_new_tokens"] = args.humaneval_max_new_tokens
            extra["humaneval_max_input_length"] = args.humaneval_max_input_length
            extra["humaneval_max_samples"] = args.humaneval_max_samples
            extra["mbpp_max_samples"] = args.mbpp_max_samples
        if args.save_predictions_dir:
            extra["save_predictions_path"] = os.path.join(
                args.save_predictions_dir, f"{bench_key}_predictions.csv"
            )
        max_samples = args.max_samples
        if bench_key == "humaneval" and args.humaneval_max_samples is not None:
            max_samples = args.humaneval_max_samples
        if bench_key == "codefeedback" and args.mbpp_max_samples is not None:
            max_samples = args.mbpp_max_samples
        if bench_key == "alpaca":
            ceiling = args.alpaca_max_samples
            if max_samples is None:
                max_samples = ceiling
            else:
                max_samples = min(int(max_samples), int(ceiling))
        res = eval_fn(
            model,
            tokenizer,
            max_samples=max_samples,
            batch_size=bs_use,
            seed=args.seed,
            **extra,
        )
        res["time_s"] = round(time.time() - t0, 1)
        results["benchmarks"][bench_key] = res

        if "accuracy" in res:
            std = f" ±{res['accuracy_std']*100:.2f}" if res.get("accuracy_std") is not None else ""
            log(
                f"  {bench_name}: acc={res['accuracy']*100:.1f}%{std} "
                f"({res.get('correct', '?')}/{res['total']}) [{res['time_s']}s]"
            )
        elif "exact_match" in res:
            em_std = f" ±{res['exact_match_std']*100:.2f}" if res.get("exact_match_std") is not None else ""
            f1_std = f" ±{res['f1_std']*100:.2f}" if res.get("f1_std") is not None else ""
            log(
                f"  {bench_name}: EM={res['exact_match']*100:.1f}%{em_std} "
                f"F1={res['f1']*100:.1f}%{f1_std} ({res['total']}) [{res['time_s']}s]"
            )
        elif "rouge1" in res:
            extra_m = ""
            if res.get("bleu") is not None:
                extra_m += f" BLEU={res['bleu']*100:.1f}"
            if res.get("meteor_mean") is not None:
                extra_m += f" METEOR={res['meteor_mean']*100:.1f}"
            r1s = f" ±{res['rouge1_std']*100:.1f}" if res.get("rouge1_std") is not None else ""
            r2s = f" ±{res['rouge2_std']*100:.1f}" if res.get("rouge2_std") is not None else ""
            rls = f" ±{res['rougeL_std']*100:.1f}" if res.get("rougeL_std") is not None else ""
            log(
                f"  {bench_name}: R1={res['rouge1']*100:.1f}{r1s} "
                f"R2={res['rouge2']*100:.1f}{r2s} RL={res['rougeL']*100:.1f}{rls}{extra_m} "
                f"({res['total']}) [{res['time_s']}s]"
            )
        elif "oci_mean_pass_at_1" in res:
            he_s = f" ±{res['humaneval_pass_at_1_std']*100:.2f}" if res.get("humaneval_pass_at_1_std") is not None else ""
            mb_s = f" ±{res['mbpp_pass_at_1_std']*100:.2f}" if res.get("mbpp_pass_at_1_std") is not None else ""
            oc_s = f" ±{res['oci_mean_pass_at_1_std']*100:.2f}" if res.get("oci_mean_pass_at_1_std") is not None else ""
            log(
                f"  {bench_name}: HE pass@1={res['humaneval_pass_at_1']*100:.1f}%{he_s} "
                f"MBPP pass@1={res['mbpp_pass_at_1']*100:.1f}%{mb_s} "
                f"mean={res['oci_mean_pass_at_1']*100:.1f}%{oc_s} [{res['time_s']}s]"
            )
        elif "pass_at_1" in res:
            pstd = f" ±{res['pass_at_1_std']*100:.2f}" if res.get("pass_at_1_std") is not None else ""
            log(
                f"  {bench_name}: pass@1={res['pass_at_1']*100:.1f}%{pstd} "
                f"({res['passed']}/{res['total']}) [{res['time_s']}s]"
            )
        log("")
        save_cf3_report_atomic(report_path, results)
        log(f"  Checkpoint saved ({len(results['benchmarks'])} benchmarks): {report_path}")

    save_cf3_report_atomic(report_path, results)
    log(f"  Report saved: {report_path}")

    log(f"\n{'='*60}")
    log(f"  SUMMARY v3 — {model_name}")
    log(f"{'='*60}")
    for bench_key, res in results["benchmarks"].items():
        name = BENCHMARK_MAP[bench_key][0]
        if "accuracy" in res:
            std = f" ±{res['accuracy_std']*100:.2f}" if res.get("accuracy_std") is not None else ""
            log(f"  {name}: acc={res['accuracy']*100:.1f}%{std}")
        elif "exact_match" in res:
            em_std = f" ±{res['exact_match_std']*100:.2f}" if res.get("exact_match_std") is not None else ""
            f1_std = f" ±{res['f1_std']*100:.2f}" if res.get("f1_std") is not None else ""
            log(
                f"  {name}: EM={res['exact_match']*100:.1f}%{em_std} "
                f"F1={res['f1']*100:.1f}%{f1_std}"
            )
        elif "rouge1" in res:
            xm = ""
            if res.get("bleu") is not None:
                xm += f" BLEU={res['bleu']*100:.1f}"
            if res.get("meteor_mean") is not None:
                xm += f" METEOR={res['meteor_mean']*100:.1f}"
            r1s = f" ±{res['rouge1_std']*100:.1f}" if res.get("rouge1_std") is not None else ""
            r2s = f" ±{res['rouge2_std']*100:.1f}" if res.get("rouge2_std") is not None else ""
            rls = f" ±{res['rougeL_std']*100:.1f}" if res.get("rougeL_std") is not None else ""
            log(
                f"  {name}: R1={res['rouge1']*100:.1f}{r1s} "
                f"R2={res['rouge2']*100:.1f}{r2s} RL={res['rougeL']*100:.1f}{rls}{xm}"
            )
        elif "oci_mean_pass_at_1" in res:
            he_s = f" ±{res['humaneval_pass_at_1_std']*100:.2f}" if res.get("humaneval_pass_at_1_std") is not None else ""
            mb_s = f" ±{res['mbpp_pass_at_1_std']*100:.2f}" if res.get("mbpp_pass_at_1_std") is not None else ""
            oc_s = f" ±{res['oci_mean_pass_at_1_std']*100:.2f}" if res.get("oci_mean_pass_at_1_std") is not None else ""
            log(
                f"  {name}: HE={res['humaneval_pass_at_1']*100:.1f}%{he_s} "
                f"MBPP={res['mbpp_pass_at_1']*100:.1f}%{mb_s} "
                f"mean={res['oci_mean_pass_at_1']*100:.1f}%{oc_s}"
            )
        elif "pass_at_1" in res:
            pstd = f" ±{res['pass_at_1_std']*100:.2f}" if res.get("pass_at_1_std") is not None else ""
            log(f"  {name}: pass@1={res['pass_at_1']*100:.1f}%{pstd}")
    log(f"{'='*60}\n")


if __name__ == "__main__":
    main()
