"""
GSM8K 8-shot Chain-of-Thought Evaluation

Matches the lm-evaluation-harness gsm8k_cot methodology so our results
are directly comparable to published paper scores.

Key details:
  - 8 fixed few-shot examples (Chain-of-Thought reasoning demos)
  - Greedy decoding (do_sample=False)
  - Final-answer extraction: **last** strict phrase (``The answer is X`` / ``Final answer: X``),
    else last-line ``= num``, else last bare number in full sanitized text; degenerate tails trimmed
  - Numeric equivalence via ``Decimal`` (e.g. 16 vs 16.00); display strings canonicalized
  - Left-padded batched inference for GPU efficiency

Main improvements without changing decoding/model settings: better final-answer extraction,
stronger normalization, and filtering degenerate outputs before scoring.

Usage:
  python eval_gsm8k.py --model meta-llama/Llama-3.1-8B
  python eval_gsm8k.py --model ./gsm8k-lora-finetuned --batch-size 128
  python eval_gsm8k.py --model ./gsm8k-full-finetuned --verbose

Default ``--output-dir`` is ``<topo>/numpy_weights/exploration-finetuning/results`` (two levels
up from this file to ``topo``, then that path). Override with ``--output-dir`` or env
``GSM8K_EVAL_RESULTS_DIR``.

**Canonical eval settings (Llama / Qwen-base pipelines use these defaults):**
``--batch-size 64``, ``--max-new-tokens 512``, full GSM8K test (omit ``--max-samples``).
Override only when you intend a smoke subset or different throughput.
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg=""):
    print(msg, flush=True)


def default_results_dir() -> str:
    """``<topo>/numpy_weights/exploration-finetuning/results`` unless GSM8K_EVAL_RESULTS_DIR is set."""
    override = os.environ.get("GSM8K_EVAL_RESULTS_DIR")
    if override:
        return os.path.abspath(override)
    gsm8k_dir = os.path.dirname(os.path.abspath(__file__))
    topo_root = os.path.normpath(os.path.join(gsm8k_dir, "..", ".."))
    return os.path.join(
        topo_root, "numpy_weights", "exploration-finetuning", "results"
    )


# ──────────────────────────────────────────────
#  Official 8-shot examples (lm-eval-harness)
# ──────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]


# ──────────────────────────────────────────────
#  Answer extraction & generation cleanup
# ──────────────────────────────────────────────

_STRICT_FINAL_PATTERNS = [
    # "The answer is 6" / "the answer is $16.00"
    re.compile(r"[Tt]he answer is\s*\$?\s*(-?[\d,]+\.?\d*)", re.I),
    # "Final answer: 42" / "Final Answer: $3"
    re.compile(r"[Ff]inal\s*answer\s*:\s*\$?\s*(-?[\d,]+\.?\d*)", re.I),
]


def strip_repeated_trailing_lines(text: str) -> tuple[str, bool]:
    """Drop a tail of 3+ identical non-trivial lines (common loop/degeneration)."""
    lines = text.split("\n")
    changed = False
    while len(lines) >= 2:
        t = lines[-1].strip()
        if len(t) < 8:
            break
        run = 1
        i = len(lines) - 2
        while i >= 0 and lines[i].strip() == t:
            run += 1
            i -= 1
        if run >= 3:
            lines = lines[: i + 1]
            changed = True
        else:
            break
    return "\n".join(lines), changed


def truncate_before_absurd_numeric_line(text: str) -> tuple[str, bool]:
    """Truncate before a very long mostly-numeric line (repeating decimals, garbage tails)."""
    lines = text.split("\n")
    out: list[str] = []
    trimmed = False
    for line in lines:
        core = line.strip()
        if len(core) > 100:
            frac = sum(c.isdigit() or c in ".,eE+-" for c in core) / len(core)
            if frac > 0.45 and "answer" not in core.lower():
                trimmed = True
                break
        out.append(line)
    return "\n".join(out), trimmed


def sanitize_generation(text: str) -> tuple[str, dict]:
    """Trim degenerate suffixes before parsing a final answer."""
    flags: dict[str, bool] = {}
    t, a = truncate_before_absurd_numeric_line(text)
    flags["absurd_numeric_line_trimmed"] = a
    t, b = strip_repeated_trailing_lines(t)
    flags["repeated_lines_trimmed"] = b
    return t, flags


def _clean_numeric_token(s: str) -> str:
    """Strip wrappers; remove a stray trailing period after an integer (``10.`` → ``10``)."""
    t = s.replace(",", "").strip().rstrip(".")
    return t


def extract_strict_final_number(text: str) -> str | None:
    """Take the numeric capture from the **last** strict final-answer phrase in document order."""
    best: str | None = None
    best_end = -1
    for rx in _STRICT_FINAL_PATTERNS:
        for m in rx.finditer(text):
            if m.end() > best_end:
                best_end = m.end()
                best = _clean_numeric_token(m.group(1))
    return best


def extract_last_equals_on_last_line(text: str) -> str | None:
    """If the final non-empty line ends an arithmetic chain ``... = 12``, return that number."""
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if len(last) > 200:
        return None
    m = re.search(r"=\s*\$?\s*(-?[\d,]+\.?\d*)\s*\.?\s*$", last)
    if not m:
        return None
    return _clean_numeric_token(m.group(1))


def extract_last_equals_in_tail(text: str, tail_chars: int = 800) -> str | None:
    """Last ``= <num>`` in the tail of the text (diagnostic / consistency check)."""
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    matches = list(re.finditer(r"=\s*\$?\s*(-?[\d,]+\.?\d*)", tail))
    if not matches:
        return None
    return _clean_numeric_token(matches[-1].group(1))


def extract_loose_last_number(text: str) -> str | None:
    """Last bare number in text (fallback after strict / last-line '='; may grab intermediates)."""
    numbers = re.findall(r"(-?\d[\d,]*\.?\d*)", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "").strip()


def extract_gold(answer_text: str):
    """Extract gold answer from GSM8K '#### X' format."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return None


def normalize(s):
    """Light string cleanup for display and fallback comparison."""
    if s is None:
        return None
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def answers_match(gold: str | None, pred: str | None) -> bool:
    """True if extracted gold and predicted strings denote the same number.

    GSM8K gold is usually an integer string; models often emit decimals ($16.00).
    ``Decimal('16') == Decimal('16.00')`` so we avoid false negatives from formatting.
    """
    if gold is None or pred is None:
        return False

    def _prep(x: str) -> str:
        t = x.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
        t = t.rstrip(".")
        return t

    g, p = _prep(gold), _prep(pred)
    if not g or not p:
        return False
    try:
        return Decimal(g) == Decimal(p)
    except InvalidOperation:
        return normalize(gold) == normalize(pred)


def format_answer_for_report(s: str | None) -> str | None:
    """Canonical display: strip clutter; drop trailing zeros (16.00 → 16)."""
    if s is None:
        return None
    t = s.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    t = t.rstrip(".")
    if not t:
        return None
    try:
        d = Decimal(t)
        if d == d.to_integral_value():
            return str(int(d))
        s2 = format(d, "f").rstrip("0").rstrip(".")
        return s2
    except InvalidOperation:
        return s.strip()


def parse_answer_full(
    raw_gen: str,
    *,
    strict_extraction_only: bool = False,
) -> dict:
    """Parse model output into prediction + diagnostics (no gold leakage).

    Fallback order matches historical GSM8K eval: after strict phrases and
    last-line ``= num``, use the **last** bare number in the **full** sanitized
    generation (not a short tail window), which full finetunes often need.
    """
    gen, deg_flags = sanitize_generation(raw_gen)
    strict = extract_strict_final_number(gen)
    last_line_eq = extract_last_equals_on_last_line(gen)
    last_tail_eq = extract_last_equals_in_tail(gen)

    method = "none"
    pred_raw: str | None = None

    if strict is not None:
        pred_raw = strict
        method = "strict_final_phrase"
    elif last_line_eq is not None:
        pred_raw = last_line_eq
        method = "last_line_equals"
    elif not strict_extraction_only:
        loose_full = extract_loose_last_number(gen)
        if loose_full is not None:
            pred_raw = loose_full
            method = "loose_full_text_number"

    last_computed = last_tail_eq
    arith_inconsistent = False
    if pred_raw is not None and last_computed is not None:
        arith_inconsistent = not answers_match(pred_raw, last_computed)

    diagnostic = None
    if pred_raw is not None and last_computed is not None:
        if arith_inconsistent:
            diagnostic = "strict_or_final_pred_differs_from_last_equals_in_tail"

    return {
        "sanitized_generation": gen.strip(),
        "pred_raw": pred_raw,
        "pred_display": format_answer_for_report(pred_raw),
        "method": method,
        "degeneration_flags": deg_flags,
        "last_computed_number": format_answer_for_report(last_computed),
        "arithmetic_inconsistent": arith_inconsistent,
        "diagnostic": diagnostic,
    }


def classify_error_bucket(
    correct: bool,
    parsed: dict,
    gold: str | None,
) -> str | None:
    if correct:
        return None
    flags = parsed.get("degeneration_flags") or {}
    method = parsed.get("method") or "none"
    if method == "none":
        if flags.get("absurd_numeric_line_trimmed") or flags.get("repeated_lines_trimmed"):
            return "truncation_or_looping"
        return "extraction_none"
    pred_cmp = parsed.get("pred_raw") or parsed.get("pred_display") or ""
    if parsed.get("arithmetic_inconsistent") and gold and parsed.get("last_computed_number"):
        if answers_match(gold, parsed["last_computed_number"]) and not answers_match(gold, pred_cmp):
            return "final_answer_mismatch_last_computed"
    if method == "loose_full_text_number":
        return "loose_number_fallback"
    return "logic_or_arithmetic"


# ──────────────────────────────────────────────
#  Prompt
# ──────────────────────────────────────────────

def build_prompt(question):
    """Build the 8-shot Chain-of-Thought prompt.

    Format: Q: {question}\nA: {step-by-step answer}\n\n  (repeated 8 times)
    Then:   Q: {new question}\nA:  (model completes this)

    The 8 examples teach the model to:
      1. Show step-by-step reasoning
      2. End with "The answer is X."
    """
    prompt = ""
    for q, a in FEW_SHOT_EXAMPLES:
        prompt += f"Q: {q}\nA: {a}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


# ──────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    test_data,
    batch_size,
    max_new_tokens,
    verbose=False,
    *,
    strict_extraction_only: bool = False,
):
    """Run batched evaluation on GSM8K test set.

    How batched generation works:
      1. All prompts in a batch are LEFT-PADDED to the same length
         (left-padding so the actual content ends at the right edge,
          and the model generates new tokens to the right)
      2. Model generates up to max_new_tokens for all prompts simultaneously
      3. We extract only the GENERATED tokens (after the input) for each sample
      4. Truncate at "Q:" to stop at the next question boundary
      5. Extract and compare the numerical answer
    """
    correct = 0
    total = 0
    results = []
    t_start = time.time()

    all_prompts = [build_prompt(ex['question']) for ex in test_data]
    all_examples = list(test_data)
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    log(f"  {len(all_prompts)} samples, {num_batches} batches (bs={batch_size})")

    if verbose:
        # Show 1 full prompt so user can verify format
        log(f"\n{'─'*60}")
        log(f"  SAMPLE PROMPT (first test question):")
        log(f"{'─'*60}")
        log(all_prompts[0])
        log(f"{'─'*60}\n")

    for batch_idx in range(num_batches):
        bs = batch_idx * batch_size
        be = min(bs + batch_size, len(all_prompts))
        batch_prompts = all_prompts[bs:be]
        batch_examples = all_examples[bs:be]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=4096,
        ).to(model.device)

        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(batch_prompts)):
            idx = bs + j
            example = batch_examples[j]

            # Generated tokens start after the full padded input (same for all in batch)
            gen_ids = outputs[j][input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Truncate at "Q:" (stop at next question)
            for stop in ["\nQ:", "Q:"]:
                si = gen_text.find(stop)
                if si != -1:
                    gen_text = gen_text[:si]
                    break

            gold_raw = normalize(extract_gold(example["answer"]))
            parsed = parse_answer_full(
                gen_text.strip(),
                strict_extraction_only=strict_extraction_only,
            )
            pred_raw = parsed.get("pred_raw")
            pred_display = parsed.get("pred_display")
            gold_display = format_answer_for_report(gold_raw) if gold_raw else None
            is_correct = answers_match(gold_raw, pred_raw)

            if is_correct:
                correct += 1
            total += 1

            err_bucket = classify_error_bucket(is_correct, parsed, gold_raw)

            results.append({
                "idx": idx,
                "question": example["question"],
                "gold": gold_display,
                "predicted": pred_display,
                "correct": is_correct,
                "generated": gen_text.strip(),
                "generation_sanitized": parsed.get("sanitized_generation", ""),
                "extraction_method": parsed.get("method"),
                "error_bucket": err_bucket,
                "degeneration_trimmed": any(
                    (parsed.get("degeneration_flags") or {}).values()
                ),
                "degeneration_flags": parsed.get("degeneration_flags"),
                "arithmetic_inconsistent": parsed.get("arithmetic_inconsistent"),
                "last_computed_number": parsed.get("last_computed_number"),
                "diagnostic": parsed.get("diagnostic"),
            })

            if verbose:
                mark = "✓" if is_correct else "✗"
                log(f"  [{idx:4d}] {mark}  gold={gold_display}  pred={pred_display}  "
                    f"m={parsed.get('method')}  bucket={err_bucket}")
                if not is_correct:
                    log(f"         Q: {example['question'][:120]}...")
                    log(f"         Gen: {gen_text.strip()[:200]}...")

        elapsed = time.time() - t_start
        sps = total / elapsed if elapsed > 0 else 0
        eta = (len(all_prompts) - total) / sps / 60 if sps > 0 else 0
        log(f"  [{total:4d}/{len(all_prompts)}] "
            f"{correct/total*100:.1f}% ({correct}/{total}) | "
            f"{sps:.1f} s/s | ETA {eta:.1f}m")

    return correct / total if total > 0 else 0.0, results


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for gsm8k_<model_slug>.json. "
            "Default: numpy_weights/exploration-finetuning/results (or GSM8K_EVAL_RESULTS_DIR)."
        ),
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-sample results")
    p.add_argument(
        "--strict-extraction-only",
        action="store_true",
        help=(
            "Only strict 'The answer is' / 'Final answer:' and last-line '= num'; "
            "disable full-text last-number fallback (for ablations)."
        ),
    )
    args = p.parse_args()
    output_dir = args.output_dir or default_results_dir()
    log(f"Results directory: {output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log(f"\nLoading model: {args.model}")
    t0 = time.time()

    is_peft = os.path.exists(os.path.join(args.model, "adapter_config.json"))
    if is_peft:
        from peft import PeftModel
        with open(os.path.join(args.model, "adapter_config.json")) as f:
            cfg = json.load(f)
        base_id = cfg["base_model_name_or_path"]
        log(f"  LoRA adapter, base: {base_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    load_time = time.time() - t0
    log(f"Loaded in {load_time:.1f}s")

    log("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset['test']
    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))

    log(f"Test: {len(test_data)} | Batch: {args.batch_size} | "
        f"MaxTok: {args.max_new_tokens}")
    log()

    t_eval = time.time()
    acc, results = evaluate(
        model,
        tokenizer,
        test_data,
        args.batch_size,
        args.max_new_tokens,
        verbose=args.verbose,
        strict_extraction_only=args.strict_extraction_only,
    )
    eval_time = time.time() - t_eval

    n_correct = sum(1 for r in results if r['correct'])
    log(f"\n{'='*50}")
    log(f"  GSM8K  |  {acc*100:.1f}%  ({n_correct}/{len(results)})")
    log(f"  Model: {args.model}")
    log(f"  Time:  {eval_time:.1f}s  ({len(results)/eval_time:.1f} samples/s)")
    log(f"{'='*50}\n")

    os.makedirs(output_dir, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "-")
    out_path = os.path.join(output_dir, f"gsm8k_{model_short}.json")

    method_ct = Counter(r.get("extraction_method") or "unknown" for r in results)
    err_ct = Counter(
        (r.get("error_bucket") or "ok") for r in results if not r["correct"]
    )
    trimmed_n = sum(1 for r in results if r.get("degeneration_trimmed"))
    inconsistent_n = sum(1 for r in results if r.get("arithmetic_inconsistent"))

    output = {
        "model": args.model,
        "benchmark": "gsm8k",
        "setting": "8-shot CoT, greedy",
        "scoring_notes": (
            "Sanitize degenerate tails; then last strict 'The answer is' / 'Final answer:'; "
            "else last-line '= num'; else last bare number in full sanitized text (legacy GSM8K "
            "fallback, needed for long full-finetune CoT). Decimal equality. "
            "Use --strict-extraction-only to disable the full-text fallback."
        ),
        "strict_extraction_only": bool(args.strict_extraction_only),
        "loose_fallback": "disabled" if args.strict_extraction_only else "full_sanitized_text",
        "accuracy": round(acc, 4),
        "correct": n_correct,
        "total": len(results),
        "eval_time_s": round(eval_time, 1),
        "extraction_method_counts": dict(method_ct),
        "incorrect_error_bucket_counts": dict(err_ct),
        "degeneration_trimmed_count": trimmed_n,
        "arithmetic_inconsistent_count": inconsistent_n,
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
