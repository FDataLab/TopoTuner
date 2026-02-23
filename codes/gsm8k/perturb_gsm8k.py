"""
Create perturbated GSM8K datasets by corrupting a fraction of final answers.

For each selected example, the final numerical answer (after ####) is replaced
with a random wrong number sampled from the distribution of all answers in the dataset.

Usage:
  python perturb_gsm8k.py --ratio 0.25 --output-dir perturbed_datasets
  python perturb_gsm8k.py --ratio 0.50 --output-dir perturbed_datasets
  python perturb_gsm8k.py --ratio 0.25 0.50 --output-dir perturbed_datasets
"""

import argparse
import json
import os
import random
import re

from datasets import load_dataset


def extract_final_answer(answer_text):
    match = re.search(r'####\s*(.+)', answer_text)
    return match.group(1).strip() if match else None


def perturb_dataset(data, ratio, seed=42):
    """Replace the final answer in a random subset of examples with a wrong answer."""
    rng = random.Random(seed)

    all_answers = []
    for ex in data:
        ans = extract_final_answer(ex['answer'])
        if ans is not None:
            all_answers.append(ans)

    n_total = len(data)
    n_perturb = int(n_total * ratio)
    indices_to_perturb = set(rng.sample(range(n_total), n_perturb))

    perturbed = []
    n_changed = 0
    for idx, ex in enumerate(data):
        new_ex = dict(ex)
        if idx in indices_to_perturb:
            original_ans = extract_final_answer(ex['answer'])
            if original_ans is None:
                perturbed.append(new_ex)
                continue

            # Pick a different answer from the pool
            wrong_ans = original_ans
            attempts = 0
            while wrong_ans == original_ans and attempts < 50:
                wrong_ans = rng.choice(all_answers)
                attempts += 1

            new_ex['answer'] = re.sub(
                r'####\s*.+',
                f'#### {wrong_ans}',
                ex['answer']
            )
            new_ex['_perturbed'] = True
            new_ex['_original_answer'] = original_ans
            n_changed += 1
        else:
            new_ex['_perturbed'] = False

        perturbed.append(new_ex)

    return perturbed, n_changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--output-dir", type=str, default="perturbed_datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Show a few examples only")
    args = parser.parse_args()

    print("Loading GSM8K...", flush=True)
    dataset = load_dataset("openai/gsm8k", "main")
    train_data = list(dataset['train'])
    print(f"  {len(train_data)} training examples\n", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    for ratio in args.ratio:
        pct = int(ratio * 100)
        print(f"Creating {pct}% perturbated dataset (seed={args.seed})...", flush=True)
        perturbed, n_changed = perturb_dataset(train_data, ratio, seed=args.seed)
        print(f"  Changed {n_changed}/{len(perturbed)} answers ({n_changed/len(perturbed)*100:.1f}%)", flush=True)

        if args.test:
            print(f"\n  Sample perturbed examples:")
            count = 0
            for ex in perturbed:
                if ex.get('_perturbed'):
                    final = extract_final_answer(ex['answer'])
                    print(f"    Q: {ex['question'][:80]}...")
                    print(f"    Original: {ex['_original_answer']}  ->  Perturbed: {final}")
                    count += 1
                    if count >= 5:
                        break
            print()
        else:
            path = os.path.join(args.output_dir, f"gsm8k_perturbed_{pct}pct.json")
            with open(path, "w") as f:
                json.dump(perturbed, f, indent=2)
            print(f"  Saved: {path}\n", flush=True)


if __name__ == "__main__":
    main()
