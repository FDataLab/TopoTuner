"""
GSM8K 8-shot Chain-of-Thought Evaluation

Matches the lm-evaluation-harness gsm8k_cot methodology so our results
are directly comparable to published paper scores.

Key details:
  - 8 fixed few-shot examples (Chain-of-Thought reasoning demos)
  - Greedy decoding (do_sample=False)
  - Exact match on extracted numerical answers
  - Left-padded batched inference for GPU efficiency

Usage:
  python eval_gsm8k.py --model meta-llama/Llama-3.1-8B
  python eval_gsm8k.py --model ./gsm8k-lora-finetuned --batch-size 128
  python eval_gsm8k.py --model ./gsm8k-full-finetuned --verbose
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg=""):
    print(msg, flush=True)


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
#  Answer extraction
# ──────────────────────────────────────────────

def extract_answer(text: str):
    """Extract the numerical answer from model-generated text.

    Two strategies (in priority order):
      1. Strict: Look for "The answer is X" — the pattern few-shot examples teach
      2. Fallback: Take the last number in the text

    The strict approach works when the model follows the CoT format properly.
    The fallback catches cases where the model gives the right number but
    doesn't follow the exact template.
    """
    # Strict
    match = re.findall(r'[Tt]he answer is\s*\$?\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match[-1].replace(',', '').strip()
    # Flexible: last number
    numbers = re.findall(r'(-?\d[\d,]*\.?\d*)', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return None


def extract_gold(answer_text: str):
    """Extract gold answer from GSM8K '#### X' format."""
    if '####' in answer_text:
        return answer_text.split('####')[-1].strip().replace(',', '')
    return None


def normalize(s):
    """Normalize for comparison: strip whitespace, commas, $, trailing period."""
    if s is None:
        return None
    return s.strip().replace(',', '').replace('$', '').rstrip('.')


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

def evaluate(model, tokenizer, test_data, batch_size, max_new_tokens, verbose=False):
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

            gold = normalize(extract_gold(example['answer']))
            pred = normalize(extract_answer(gen_text))
            is_correct = gold is not None and pred is not None and gold == pred

            if is_correct:
                correct += 1
            total += 1

            results.append({
                'idx': idx,
                'question': example['question'],
                'gold': gold,
                'predicted': pred,
                'correct': is_correct,
                'generated': gen_text.strip(),
            })

            if verbose:
                mark = "✓" if is_correct else "✗"
                log(f"  [{idx:4d}] {mark}  gold={gold}  pred={pred}")
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
    p.add_argument("--output-dir", default="results")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-sample results")
    args = p.parse_args()

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
        model, tokenizer, test_data,
        args.batch_size, args.max_new_tokens,
        verbose=args.verbose)
    eval_time = time.time() - t_eval

    n_correct = sum(1 for r in results if r['correct'])
    log(f"\n{'='*50}")
    log(f"  GSM8K  |  {acc*100:.1f}%  ({n_correct}/{len(results)})")
    log(f"  Model: {args.model}")
    log(f"  Time:  {eval_time:.1f}s  ({len(results)/eval_time:.1f} samples/s)")
    log(f"{'='*50}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.replace("/", "_").replace(".", "-")
    out_path = os.path.join(args.output_dir, f"gsm8k_{model_short}.json")

    output = {
        "model": args.model,
        "benchmark": "gsm8k",
        "setting": "8-shot CoT, greedy",
        "accuracy": round(acc, 4),
        "correct": n_correct,
        "total": len(results),
        "eval_time_s": round(eval_time, 1),
        "samples": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
