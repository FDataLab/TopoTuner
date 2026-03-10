"""
Evaluate GSM8K accuracy at each epoch checkpoint across multiple runs per plan.

For each plan (A/B/C/D), loads every epoch checkpoint from all runs,
evaluates on GSM8K test, and produces an accuracy-vs-epoch plot with
per-plan curves averaged over runs.

Usage:
  python eval_epoch_accuracy.py --plans A B C D --runs 3 --batch-size 128
  python eval_epoch_accuracy.py --plans A --runs 1 --max-samples 20  # quick test
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Reuse eval logic from eval_gsm8k.py ──────────────────────────

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


def extract_answer(text):
    match = re.findall(r'[Tt]he answer is\s*\$?\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match[-1].replace(',', '').strip()
    numbers = re.findall(r'(-?\d[\d,]*\.?\d*)', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return None


def extract_gold(answer_text):
    if '####' in answer_text:
        return answer_text.split('####')[-1].strip().replace(',', '')
    return None


def normalize(s):
    if s is None:
        return None
    return s.strip().replace(',', '').replace('$', '').rstrip('.')


def build_prompt(question):
    prompt = ""
    for q, a in FEW_SHOT_EXAMPLES:
        prompt += f"Q: {q}\nA: {a}\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def evaluate_model(model, tokenizer, test_data, batch_size, max_new_tokens):
    correct = 0
    total = 0
    all_prompts = [build_prompt(ex['question']) for ex in test_data]
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        bs = batch_idx * batch_size
        be = min(bs + batch_size, len(all_prompts))
        batch_prompts = all_prompts[bs:be]
        batch_examples = list(test_data)[bs:be] if hasattr(test_data, '__getitem__') else list(test_data)[bs:be]

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
            gen_ids = outputs[j][input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            for stop in ["\nQ:", "Q:"]:
                si = gen_text.find(stop)
                if si != -1:
                    gen_text = gen_text[:si]
                    break

            example = batch_examples[j]
            gold = normalize(extract_gold(example['answer']))
            pred = normalize(extract_answer(gen_text))
            if gold is not None and pred is not None and gold == pred:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0, correct, total


# ── Checkpoint discovery ──────────────────────────────────────────

def find_checkpoints(plan, runs, base_dir="."):
    """Find all epoch checkpoint paths for a plan across runs.

    Returns dict: {run_idx: [(epoch, checkpoint_path), ...]}
    """
    result = {}
    for run in range(1, runs + 1):
        if run == 1:
            run_dir = os.path.join(base_dir, f"gsm8k-frozen-plan{plan}")
        else:
            run_dir = os.path.join(base_dir, f"gsm8k-frozen-plan{plan}-run{run}")

        if not os.path.isdir(run_dir):
            print(f"  WARNING: {run_dir} not found, skipping", flush=True)
            continue

        checkpoints = _discover_epoch_checkpoints(run_dir)
        if checkpoints:
            result[run] = checkpoints

    return result


def find_checkpoints_by_dirs(run_dirs):
    """Find epoch checkpoints from explicit directory list.

    Returns dict: {run_idx: [(epoch, checkpoint_path), ...]}
    """
    result = {}
    for idx, run_dir in enumerate(run_dirs, start=1):
        if not os.path.isdir(run_dir):
            print(f"  WARNING: {run_dir} not found, skipping", flush=True)
            continue
        checkpoints = _discover_epoch_checkpoints(run_dir)
        if checkpoints:
            result[idx] = checkpoints
    return result


def _discover_epoch_checkpoints(run_dir):
    """Discover checkpoint-* dirs and assign epoch numbers."""
    checkpoints = []
    for ckpt_dir in sorted(glob.glob(os.path.join(run_dir, "checkpoint-*"))):
        step = int(ckpt_dir.split("-")[-1])
        checkpoints.append((step, ckpt_dir))

    if not checkpoints:
        return []

    checkpoints.sort(key=lambda x: x[0])
    step_gap = checkpoints[0][0]
    return [(step // step_gap, path) for step, path in checkpoints]


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="epoch_accuracy_results")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Base model for tokenizer (checkpoints don't save tokenizer)")
    parser.add_argument("--custom-dirs", nargs="+", default=None,
                        help="Custom run directories (overrides --plans). Format: label:dir1,dir2,...")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer from base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading GSM8K test set...", flush=True)
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset['test']
    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
    print(f"  {len(test_data)} test samples\n", flush=True)

    # Collect all results: {label: {epoch: [acc_run1, acc_run2, ...]}}
    all_results = {}

    # Build experiment list: [(label, ckpt_map), ...]
    experiments = []
    if args.custom_dirs:
        for entry in args.custom_dirs:
            label, dirs_str = entry.split(":", 1)
            dirs = dirs_str.split(",")
            ckpt_map = find_checkpoints_by_dirs(dirs)
            experiments.append((label, ckpt_map))
    else:
        for plan in args.plans:
            ckpt_map = find_checkpoints(plan, args.runs, args.base_dir)
            experiments.append((plan, ckpt_map))

    for label, ckpt_map in experiments:
        print(f"\n{'='*60}", flush=True)
        print(f"  {label}", flush=True)
        print(f"{'='*60}", flush=True)

        if not ckpt_map:
            print(f"  No checkpoints found for {label}", flush=True)
            continue

        plan_results = {}
        for run_idx, epoch_ckpts in sorted(ckpt_map.items()):
            print(f"\n  Run {run_idx}: {len(epoch_ckpts)} epoch checkpoints", flush=True)
            for epoch, ckpt_path in epoch_ckpts:
                print(f"    Epoch {epoch}: {ckpt_path}", flush=True)

                t0 = time.time()
                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, torch_dtype=torch.bfloat16, device_map="auto")
                model.eval()

                acc, correct, total = evaluate_model(
                    model, tokenizer, test_data, args.batch_size, args.max_new_tokens)

                elapsed = time.time() - t0
                print(f"      -> {acc*100:.1f}% ({correct}/{total}) in {elapsed:.0f}s", flush=True)

                if epoch not in plan_results:
                    plan_results[epoch] = []
                plan_results[epoch].append(acc)

                del model
                torch.cuda.empty_cache()

        all_results[label] = plan_results

        save_path = os.path.join(args.output_dir, f"{label}_epoch_accuracy.json")
        with open(save_path, "w") as f:
            json.dump({str(k): v for k, v in plan_results.items()}, f, indent=2)
        print(f"\n  Saved: {save_path}", flush=True)

    # ── Generate plot ─────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print(f"  GENERATING PLOT", flush=True)
    print(f"{'='*60}", flush=True)

    plan_labels = {
        "A": "Plan A (V+O+MLP, layers 0-9)",
        "B": "Plan B (V+O, layers 0-9)",
        "C": "Plan C (V+O+MLP, layers 22-31+head)",
        "D": "Plan D (V+O, layers 22-31+head)",
    }
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 6))

    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']

    for idx, (label, plan_results) in enumerate(all_results.items()):
        if not plan_results:
            continue

        epochs = sorted(plan_results.keys())
        means = [np.mean(plan_results[e]) * 100 for e in epochs]
        stds = [np.std(plan_results[e]) * 100 if len(plan_results[e]) > 1 else 0 for e in epochs]

        display = plan_labels.get(label, label)
        color = colors.get(label, default_colors[idx % len(default_colors)])

        ax.plot(epochs, means, 'o-', label=display, color=color, linewidth=2, markersize=6)
        if any(s > 0 for s in stds):
            ax.fill_between(epochs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=color)

        for e, m in zip(epochs, means):
            ax.annotate(f'{m:.1f}', (e, m), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('GSM8K Accuracy (%)', fontsize=12)
    ax.set_title('GSM8K Epoch Accuracy by Freezing Plan', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 7))

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "epoch_accuracy_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {plot_path}", flush=True)

    # Summary table
    print(f"\n{'='*60}", flush=True)
    print(f"  SUMMARY (mean accuracy %)", flush=True)
    print(f"{'='*60}", flush=True)
    header = f"{'Experiment':<25}" + "".join(f"{'Ep'+str(e):>8}" for e in range(1, 7))
    print(f"  {header}", flush=True)
    for label, pr in all_results.items():
        row = f"  {label:<25}"
        for e in range(1, 7):
            if e in pr:
                row += f"{np.mean(pr[e])*100:>7.1f}%"
            else:
                row += f"{'N/A':>8}"
        print(row, flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
