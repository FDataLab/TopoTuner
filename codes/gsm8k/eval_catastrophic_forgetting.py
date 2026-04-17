"""
Catastrophic Forgetting Evaluation

Evaluates finetuned models on non-GSM8K benchmarks to measure how much
general capability was lost during GSM8K finetuning.

Benchmarks:
  - SST-2:  Sentiment classification (positive/negative). 2-shot.
  - IMDB:   Sentiment classification (positive/negative). 2-shot.
  - MMLU:   Multiple-choice knowledge QA (A/B/C/D). 5-shot (subject-specific dev).
  - SQuAD:  Extractive QA (exact match + F1). 2-shot.

Usage:
  python eval_catastrophic_forgetting.py --model meta-llama/Llama-3.1-8B --benchmarks sst2 imdb mmlu squad
  python eval_catastrophic_forgetting.py --model ./checkpoint --seed 123 --benchmarks sst2
  python eval_catastrophic_forgetting.py --model ./lora-adapter --is-lora --benchmarks sst2 imdb mmlu squad
"""

import argparse
import json
import os
import random
import re
import time
import string
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def log(msg=""):
    print(msg, flush=True)


# ──────────────────────────────────────────────────────────────────────
#  Shared generation utilities
# ──────────────────────────────────────────────────────────────────────

def generate_batch(model, tokenizer, prompts, max_new_tokens=32, batch_size=128):
    """Batched greedy generation with left-padding."""
    _prev_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)
        input_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(batch)):
            gen_ids = out[j][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

    tokenizer.padding_side = _prev_side
    return results


# ──────────────────────────────────────────────────────────────────────
#  SST-2 (Stanford Sentiment Treebank)
# ──────────────────────────────────────────────────────────────────────

SST2_FEW_SHOT_DEFAULT = [
    ("A stirring, funny and finally transporting re-imagining of Beauty and the Beast.", "positive"),
    ("Unflinchingly bleak and desperate.", "negative"),
]


def _pick_sst2_shots(seed):
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


def build_sst2_prompt(sentence, few_shot):
    prompt = ""
    for s, label in few_shot:
        prompt += f"Sentence: {s}\nSentiment: {label}\n\n"
    prompt += f"Sentence: {sentence}\nSentiment:"
    return prompt


def eval_sst2(model, tokenizer, max_samples=None, batch_size=128, seed=None):
    log("  Loading SST-2...")
    dataset = load_dataset("glue", "sst2", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    few_shot = _pick_sst2_shots(seed)
    label_map = {0: "negative", 1: "positive"}
    prompts = [build_sst2_prompt(ex['sentence'], few_shot) for ex in dataset]
    gold_labels = [label_map[ex['label']] for ex in dataset]

    log(f"  Generating {len(prompts)} predictions...")
    preds = generate_batch(model, tokenizer, prompts, max_new_tokens=4, batch_size=batch_size)

    correct = 0
    for pred, gold in zip(preds, gold_labels):
        pred_lower = pred.lower().strip()
        if pred_lower and gold in pred_lower.split()[0]:
            correct += 1
    acc = correct / len(gold_labels) if gold_labels else 0
    return {"accuracy": round(acc, 4), "correct": correct, "total": len(gold_labels)}


# ──────────────────────────────────────────────────────────────────────
#  IMDB
# ──────────────────────────────────────────────────────────────────────

IMDB_FEW_SHOT_DEFAULT = [
    ("This film was absolutely wonderful. The acting was superb and the plot was gripping from start to finish.", "positive"),
    ("Terrible movie. Poor acting, weak plot, and a waste of time.", "negative"),
]


def _pick_imdb_shots(seed):
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


def build_imdb_prompt(text, few_shot):
    text_truncated = " ".join(text.split()[:200])
    prompt = ""
    for t, label in few_shot:
        prompt += f"Review: {t}\nSentiment: {label}\n\n"
    prompt += f"Review: {text_truncated}\nSentiment:"
    return prompt


def eval_imdb(model, tokenizer, max_samples=None, batch_size=128, seed=None):
    log("  Loading IMDB...")
    dataset = load_dataset("imdb", split="test")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    few_shot = _pick_imdb_shots(seed)
    label_map = {0: "negative", 1: "positive"}
    prompts = [build_imdb_prompt(ex['text'], few_shot) for ex in dataset]
    gold_labels = [label_map[ex['label']] for ex in dataset]

    log(f"  Generating {len(prompts)} predictions...")
    preds = generate_batch(model, tokenizer, prompts, max_new_tokens=4, batch_size=batch_size)

    correct = 0
    for pred, gold in zip(preds, gold_labels):
        pred_lower = pred.lower().strip()
        if pred_lower and gold in pred_lower.split()[0]:
            correct += 1
    acc = correct / len(gold_labels) if gold_labels else 0
    return {"accuracy": round(acc, 4), "correct": correct, "total": len(gold_labels)}


# ──────────────────────────────────────────────────────────────────────
#  MMLU (Massive Multitask Language Understanding)
# ──────────────────────────────────────────────────────────────────────

def build_mmlu_prompt(subject, few_shot_examples, question, choices):
    prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
    for ex in few_shot_examples:
        prompt += f"Question: {ex['question']}\n"
        for i, c in enumerate(ex['choices']):
            prompt += f"{'ABCD'[i]}. {c}\n"
        prompt += f"Answer: {'ABCD'[ex['answer']]}\n\n"
    prompt += f"Question: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"{'ABCD'[i]}. {c}\n"
    prompt += "Answer:"
    return prompt


def eval_mmlu(model, tokenizer, max_samples=None, batch_size=128, seed=None):
    log("  Loading MMLU...")
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dev_dataset = load_dataset("cais/mmlu", "all", split="dev")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dev_by_subject = {}
    for ex in dev_dataset:
        subj = ex['subject']
        if subj not in dev_by_subject:
            dev_by_subject[subj] = []
        dev_by_subject[subj].append(ex)

    if seed is not None:
        rng = random.Random(seed)
        for subj in dev_by_subject:
            rng.shuffle(dev_by_subject[subj])

    for subj in dev_by_subject:
        dev_by_subject[subj] = dev_by_subject[subj][:5]

    prompts = []
    gold_labels = []
    for ex in dataset:
        few_shot = dev_by_subject.get(ex['subject'], [])
        prompt = build_mmlu_prompt(ex['subject'], few_shot, ex['question'], ex['choices'])
        prompts.append(prompt)
        gold_labels.append('ABCD'[ex['answer']])

    log(f"  Generating {len(prompts)} predictions...")
    preds = generate_batch(model, tokenizer, prompts, max_new_tokens=2, batch_size=batch_size)

    correct = 0
    for pred, gold in zip(preds, gold_labels):
        pred_clean = pred.strip().upper()
        if pred_clean and pred_clean[0] == gold:
            correct += 1
    acc = correct / len(gold_labels) if gold_labels else 0
    return {"accuracy": round(acc, 4), "correct": correct, "total": len(gold_labels)}


# ──────────────────────────────────────────────────────────────────────
#  SQuAD (Stanford Question Answering Dataset)
# ──────────────────────────────────────────────────────────────────────

SQUAD_FEW_SHOT_DEFAULT = [
    {"context": "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France.",
     "question": "In what country is Normandy located?",
     "answer": "France"},
    {"context": "The Amazon rainforest produces more than 20% of the world's oxygen.",
     "question": "What percentage of the world's oxygen does the Amazon produce?",
     "answer": "more than 20%"},
]


def _pick_squad_shots(seed):
    if seed is None:
        return SQUAD_FEW_SHOT_DEFAULT
    rng = random.Random(seed)
    train = load_dataset("rajpurkar/squad", split="train")
    indices = list(range(len(train)))
    rng.shuffle(indices)
    shots = []
    for idx in indices:
        ex = train[idx]
        answers = ex.get("answers", {}).get("text", [])
        if not answers:
            continue
        shots.append({
            "context": " ".join(ex["context"].split()[:150]),
            "question": ex["question"],
            "answer": answers[0],
        })
        if len(shots) >= 2:
            break
    return shots


def normalize_answer_squad(s):
    """Normalize for SQuAD-style exact match / F1."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = ' '.join(s.split())
    return s


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer_squad(prediction).split()
    gold_tokens = normalize_answer_squad(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def build_squad_prompt(context, question, few_shot):
    prompt = ""
    for ex in few_shot:
        prompt += f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    context_truncated = " ".join(context.split()[:300])
    prompt += f"Context: {context_truncated}\nQuestion: {question}\nAnswer:"
    return prompt


def eval_squad(model, tokenizer, max_samples=None, batch_size=128, seed=None):
    log("  Loading SQuAD...")
    dataset = load_dataset("rajpurkar/squad", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    few_shot = _pick_squad_shots(seed)
    prompts = []
    gold_answers = []
    for ex in dataset:
        prompts.append(build_squad_prompt(ex['context'], ex['question'], few_shot))
        gold_answers.append(ex['answers']['text'])

    log(f"  Generating {len(prompts)} predictions...")
    preds = generate_batch(model, tokenizer, prompts, max_new_tokens=12, batch_size=batch_size)

    exact_matches = 0
    f1_scores = []
    for pred, golds in zip(preds, gold_answers):
        pred_clean = pred.split('\n')[0].strip()
        em = any(normalize_answer_squad(pred_clean) == normalize_answer_squad(g) for g in golds)
        f1 = max(compute_f1(pred_clean, g) for g in golds) if golds else 0.0
        if em:
            exact_matches += 1
        f1_scores.append(f1)

    n = len(gold_answers)
    return {
        "exact_match": round(exact_matches / n, 4) if n else 0,
        "f1": round(sum(f1_scores) / n, 4) if n else 0,
        "total": n,
    }


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

BENCHMARK_MAP = {
    "sst2": ("SST-2", eval_sst2),
    "imdb": ("IMDB", eval_imdb),
    "mmlu": ("MMLU", eval_mmlu),
    "squad": ("SQuAD", eval_squad),
}


def main():
    parser = argparse.ArgumentParser(description="Catastrophic Forgetting Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Human-readable name for reports (default: derived from --model)")
    parser.add_argument("--benchmarks", nargs="+", default=["sst2", "imdb", "mmlu", "squad"],
                        choices=list(BENCHMARK_MAP.keys()))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per benchmark (for quick testing)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--is-lora", action="store_true",
                        help="Load as LoRA adapter (reads base_model from adapter_config.json)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for few-shot example selection (None = use fixed defaults)")
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(args.model.rstrip('/'))
    if args.output_dir is None:
        args.output_dir = f"catastrophic-forgetting-{model_name}"

    log(f"\n{'='*60}")
    log(f"  CATASTROPHIC FORGETTING EVALUATION")
    log(f"{'='*60}")
    log(f"  Model:      {args.model}")
    log(f"  Name:       {model_name}")
    log(f"  LoRA:       {args.is_lora}")
    log(f"  Seed:       {args.seed}")
    log(f"  Benchmarks: {', '.join(args.benchmarks)}")
    log(f"  Batch size: {args.batch_size}")
    if args.max_samples:
        log(f"  Max samples: {args.max_samples}")
    log(f"  Output:     {args.output_dir}")
    log(f"{'='*60}\n")

    log("Loading model...")
    t0 = time.time()

    if args.is_lora:
        from peft import PeftModel
        cfg_path = os.path.join(args.model, "adapter_config.json")
        with open(cfg_path) as f:
            base_id = json.load(f).get("base_model_name_or_path")
        log(f"  LoRA base: {base_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, args.model)
        model = model.merge_and_unload()
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
                    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
                else:
                    raise e
            else:
                raise e
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    load_time = time.time() - t0
    log(f"  Loaded in {load_time:.1f}s\n")

    results = {"model": args.model, "model_name": model_name, "seed": args.seed, "benchmarks": {}}
    os.makedirs(args.output_dir, exist_ok=True)

    for bench_key in args.benchmarks:
        bench_name, eval_fn = BENCHMARK_MAP[bench_key]
        log(f">>> Evaluating {bench_name}...")
        t0 = time.time()
        res = eval_fn(model, tokenizer, max_samples=args.max_samples,
                      batch_size=args.batch_size, seed=args.seed)
        elapsed = time.time() - t0
        res["time_s"] = round(elapsed, 1)
        results["benchmarks"][bench_key] = res

        if "accuracy" in res:
            log(f"  {bench_name}: {res['accuracy']*100:.1f}% ({res['correct']}/{res['total']}) [{elapsed:.1f}s]")
        elif "exact_match" in res:
            log(f"  {bench_name}: EM={res['exact_match']*100:.1f}% F1={res['f1']*100:.1f}% ({res['total']} samples) [{elapsed:.1f}s]")
        log("")

    report_path = os.path.join(args.output_dir, "catastrophic_forgetting_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Report saved: {report_path}")

    log(f"\n{'='*60}")
    log(f"  CATASTROPHIC FORGETTING SUMMARY — {model_name}")
    log(f"{'='*60}")
    for bench_key, res in results["benchmarks"].items():
        bench_name = BENCHMARK_MAP[bench_key][0]
        if "accuracy" in res:
            log(f"  {bench_name:8s}: {res['accuracy']*100:.1f}%")
        elif "exact_match" in res:
            log(f"  {bench_name:8s}: EM={res['exact_match']*100:.1f}%  F1={res['f1']*100:.1f}%")
    log(f"{'='*60}\n")


if __name__ == "__main__":
    main()
