# codes/utils/eval_checkpoints.py

import os
import time
import argparse
import gc

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM

# import dataset-specific evaluators
from codes.gsm8k import eval_gsm8k as gsm8k_eval
from codes.mmlu import eval_mmlu as mmlu_eval
from codes.imdb import eval_imdb as imdb_eval
from codes.sst2 import eval_sst2 as sst2_eval
from codes.hotpotqa import eval_hotpotqa_updated as hotpotqa_eval
from codes.squad import eval_squad as squad_eval

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Only pass token to HF loaders if it's present
_HF_KW = {}
if HF_TOKEN:
    _HF_KW["token"] = HF_TOKEN

# registry of datasets → evaluator functions
EVAL_REGISTRY = {
    "gsm8k": gsm8k_eval.evaluate_gsm8k,
    "mmlu": mmlu_eval.evaluate_mmlu,
    "imdb": imdb_eval.evaluate_imdb,
    "sst2": sst2_eval.evaluate_sst2,
    "hotpotqa": hotpotqa_eval.evaluate_hotpotqa,
    "squad": squad_eval.evaluate_squad,
}


def load_model(checkpoint_dir, model_id, tokenizer, use_lora=False):
    """Load checkpoint: full or LoRA (PEFT)."""
    if use_lora:
        print(f"🔹 Loading base model {model_id} and applying LoRA from {checkpoint_dir} ...")
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **_HF_KW
        )
        if base.get_input_embeddings().num_embeddings != len(tokenizer):
            base.resize_token_embeddings(len(tokenizer))

        print(f"🔹 Loading LoRA adapters from {checkpoint_dir}...")
        peft_model = PeftModelForCausalLM.from_pretrained(base, checkpoint_dir)

        print(f"🔹 Using PEFT model directly for evaluation...")
        return peft_model.eval()
    else:
        print(f"🔹 Loading full model from {checkpoint_dir} ...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **_HF_KW
        )
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(EVAL_REGISTRY.keys()))
    parser.add_argument("--checkpoints-dir", required=True, help="Parent dir containing epoch subfolders")
    parser.add_argument("--model-name", required=True, help="Base model id (e.g., meta-llama/Llama-3.2-3B)")
    parser.add_argument("--use-lora", action="store_true", help="Whether checkpoints are LoRA adapters")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-csv", type=str, default="results.csv")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--tokenizer-dir", type=str, default=None,
                        help="Path to tokenizer (e.g., final_model folder with special tokens)")
    parser.add_argument("--debug-print", action="store_true")

    # MMLU options
    parser.add_argument("--mmlu-few-shot-k", type=int, default=0,
                        help="Number of few-shot exemplars for MMLU (0 = zero-shot)")
    parser.add_argument("--mmlu-few-shot-split", type=str, default="auxiliary_train",
                        help="Support split to sample few-shot exemplars from")

    # ✅ Option A: one big per-example CSV across checkpoints
    parser.add_argument("--save-examples-csv", type=str, default=None,
                        help="If set, appends per-example rows (prompt/gen/gold/pred/em/f1) into this CSV.")

    args = parser.parse_args()
    eval_fn = EVAL_REGISTRY[args.dataset]

    # Defer tokenizer loading to each checkpoint
    tok = None

    # Planned size info for SST-2 (optional)
    if args.dataset == "sst2":
        try:
            from datasets import load_dataset
            _ds = load_dataset("stanfordnlp/sst2")[args.split]
            _planned_n = len(_ds) if args.limit is None else min(args.limit, len(_ds))
            print(f"[INFO] SST-2 planned evaluation examples: {_planned_n} (split={args.split}, limit={args.limit})")
        except Exception as _e:
            print(f"[WARN] Unable to determine SST-2 planned size: {_e}")

    results = []

    for subdir in sorted(os.listdir(args.checkpoints_dir)):
        epoch_path = os.path.join(args.checkpoints_dir, subdir)
        if not os.path.isdir(epoch_path):
            continue

        print(f"\n🚀 Evaluating {epoch_path} ...")

        try:
            start_time = time.time()

            # Load tokenizer from checkpoint if present else base model
            if os.path.exists(os.path.join(epoch_path, "tokenizer_config.json")):
                tok = AutoTokenizer.from_pretrained(epoch_path, use_fast=True, **_HF_KW)
            else:
                tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, **_HF_KW)

            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"

            print(f"[DEBUG] Tokenizer vocab size = {len(tok)}")

            model = load_model(epoch_path, args.model_name, tok, args.use_lora)

            # -------------------- dataset routing --------------------
            if args.dataset == "mmlu":
                r = eval_fn(
                    model, tok,
                    split=args.split,
                    subjects=None,
                    limit_per_subject=None,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    progress_bar=True,
                    save_jsonl=None,
                    save_tsv=None,
                    debug_print=args.debug_print,
                    few_shot_k=args.mmlu_few_shot_k,
                    few_shot_split=args.mmlu_few_shot_split,
                )

            elif args.dataset == "gsm8k":
                r = eval_fn(
                    model, tok,
                    split=args.split,
                    limit=args.limit,
                    batch_size=args.batch_size,
                )

            elif args.dataset == "imdb":
                r = eval_fn(
                    model, tok,
                    split=args.split,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    progress_bar=True,
                    debug_print=args.debug_print,
                )

            elif args.dataset == "sst2":
                r = eval_fn(
                    model, tok,
                    split=args.split,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    progress_bar=True,
                    debug_print=args.debug_print,
                )

            elif args.dataset == "hotpotqa":
                # ✅ Option A logging happens here (NOT in MMLU)
                r = eval_fn(
                    model, tok,
                    split=args.split,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    progress_bar=True,
                    debug_print=args.debug_print,
                    save_examples_csv=args.save_examples_csv,
                    run_name=subdir,
                )

            elif args.dataset == "squad":
                r = eval_fn(
                    model, tok,
                    split=args.split,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    progress_bar=True,
                    debug_print=args.debug_print,
                    save_examples_csv=args.save_examples_csv,
                    run_name=subdir,
                )

            else:
                r = eval_fn(model, tok, split=args.split, limit=args.limit)

            elapsed_time = time.time() - start_time

            # -------------------- metric aggregation CSV --------------------
            if args.dataset in ["hotpotqa", "squad"]:
                em_value = r.get("em", 0.0)
                f1_value = r.get("f1", 0.0)
                # Convert to canonical fraction format (0-1) for CSV storage
                em_frac = em_value / 100.0 if em_value > 1.0 else em_value
                f1_frac = f1_value / 100.0 if f1_value > 1.0 else f1_value
                # Calculate percentages for display
                em_pct = em_frac * 100.0
                f1_pct = f1_frac * 100.0

                results.append({
                    "checkpoint": subdir,
                    "em": em_frac,  # Store as fraction (0-1)
                    "f1": f1_frac,   # Store as fraction (0-1)
                    "n": r.get("n", r.get("total", 0)),
                    "time_seconds": round(elapsed_time, 2),
                    "time_minutes": round(elapsed_time / 60, 2),
                })
                print(f"✅ {subdir} → EM={em_pct:.2f}% F1={f1_pct:.2f}% (n={r.get('n', r.get('total', 0))}) time={elapsed_time:.1f}s")

            else:
                acc_value = r.get("accuracy", r.get("acc", 0.0))
                acc_pct = acc_value * 100.0 if acc_value <= 1.0 else acc_value

                results.append({
                    "checkpoint": subdir,
                    "acc": acc_pct,
                    "n": r.get("n", r.get("total", 0)),
                    "time_seconds": round(elapsed_time, 2),
                    "time_minutes": round(elapsed_time / 60, 2),
                })
                print(f"✅ {subdir} → ACC={acc_pct:.2f}% (n={r.get('n', r.get('total', 0))}) time={elapsed_time:.1f}s")

            # cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            import traceback
            print(f"❌ Failed on {subdir}: {e}")
            traceback.print_exc()

            # still cleanup hard
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(results)
    dirn = os.path.dirname(args.save_csv)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    df.to_csv(args.save_csv, index=False)
    print(f"\n📊 All results saved to {args.save_csv}")


if __name__ == "__main__":
    main()

"""
### Evaluation examples with batch processing (batch_size=128)

# HotpotQA evaluation
nohup python -m codes.utils.eval_checkpoints \
  --dataset hotpotqa \
  --split test \
  --checkpoints-dir ./numpy_weights/hotpotqa/llama32_3b/full/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --batch-size 128 \
  --save-csv results/llama32_3b_hotpotqa_full.csv \
  > logs/eval_hotpotqa_llama32_3b_full.log 2>&1 &

nohup python -m codes.utils.eval_checkpoints \
  --dataset hotpotqa \
  --split test \
  --checkpoints-dir ./numpy_weights/hotpotqa/llama32_3b/lora/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --batch-size 8 \
  --save-csv results/llama32_3b_hotpotqa_lora.csv \
  > logs/eval_hotpotqa_llama32_3b_lora.log 2>&1 &

# MMLU evaluation - now uses batch processing
nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/llama32_3b/full/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --batch-size 128 \
  --save-csv results/llama32_3b_mmlu_full.csv \
  > logs/eval_mmlu_llama32_3b_full.log 2>&1 &

nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/llama32_3b/lora/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --batch-size 128 \
  --save-csv results/llama32_3b_mmlu_lora.csv \
  > logs/eval_mmlu_llama32_3b_lora.log 2>&1 &

# Llama-3.1-8B
nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/llama31_8b/full/epoch_weights \
  --model-name meta-llama/Llama-3.1-8B \
  --batch-size 128 \
  --save-csv results/llama31_8b_mmlu_full.csv \
  > logs/eval_mmlu_llama31_8b_full.log 2>&1 &

nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/llama31_8b/lora/epoch_weights \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --batch-size 128 \
  --save-csv results/llama31_8b_mmlu_lora.csv \
  > logs/eval_mmlu_llama31_8b_lora.log 2>&1 &

# Mistral-7B
CUDA_VISIBLE_DEVICES=2 nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/mistral7b/full/epoch_weights \
  --model-name mistralai/Mistral-7B-v0.1 \
  --batch-size 128 \
  --save-csv results/mistral7b_mmlu_full.csv \
  > logs/eval_mmlu_mistral7b_full.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/mistral7b/lora/epoch_weights \
  --model-name mistralai/Mistral-7B-v0.1 \
  --use-lora \
  --batch-size 128 \
  --save-csv results/mistral7b_mmlu_lora.csv \
  > logs/eval_mmlu_mistral7b_lora.log 2>&1 &

# Qwen3-8B
CUDA_VISIBLE_DEVICES=3 nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/qwen_8b/full/epoch_weights \
  --model-name Qwen/Qwen3-8B \
  --batch-size 128 \
  --save-csv results/qwen3_8b_mmlu_full.csv \
  > logs/eval_mmlu_qwen3_8b_full.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -m codes.utils.eval_checkpoints \
  --dataset mmlu \
  --split validation \
  --checkpoints-dir ./numpy_weights/mmlu/qwen_8b/lora/epoch_weights \
  --model-name Qwen/Qwen3-8B \
  --use-lora \
  --batch-size 128 \
  --save-csv results/qwen3_8b_mmlu_lora.csv \
  > logs/eval_mmlu_qwen3_8b_lora.log 2>&1 &

### GSM8K evaluation examples

# Llama-3.2-3B GSM8K
nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/llama32_3b/full/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --batch-size 8 \
  --save-csv results/llama32_3b_gsm8k_full.csv \
  > logs/eval_gsm8k_llama32_3b_full.log 2>&1 &

nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/llama32_3b/lora/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --batch-size 8 \
  --save-csv results/llama32_3b_gsm8k_lora.csv \
  > logs/eval_gsm8k_llama32_3b_lora.log 2>&1 &

# Llama-3.1-8B GSM8K
nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/llama31_8b/full/epoch_weights \
  --model-name meta-llama/Llama-3.1-8B \
  --batch-size 8 \
  --save-csv results/llama31_8b_gsm8k_full.csv \
  > logs/eval_gsm8k_llama31_8b_full.log 2>&1 &

nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/llama31_8b/lora/epoch_weights \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --batch-size 8 \
  --save-csv results/llama31_8b_gsm8k_lora.csv \
  > logs/eval_gsm8k_llama31_8b_lora.log 2>&1 &

# Mistral-7B GSM8K
CUDA_VISIBLE_DEVICES=2 nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/mistral7b/full/epoch_weights \
  --model-name mistralai/Mistral-7B-v0.1 \
  --batch-size 8 \
  --save-csv results/mistral7b_gsm8k_full.csv \
  > logs/eval_gsm8k_mistral7b_full.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/mistral7b/lora/epoch_weights \
  --model-name mistralai/Mistral-7B-v0.1 \
  --use-lora \
  --batch-size 8 \
  --save-csv results/mistral7b_gsm8k_lora.csv \
  > logs/eval_gsm8k_mistral7b_lora.log 2>&1 &

# Qwen-3-8B GSM8K
CUDA_VISIBLE_DEVICES=3 nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/qwen_8b/full/epoch_weights \
  --model-name Qwen/Qwen3-8B \
  --batch-size 8 \
  --save-csv results/qwen3_8b_gsm8k_full.csv \
  > logs/eval_gsm8k_qwen3_8b_full.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -m codes.utils.eval_checkpoints \
  --dataset gsm8k \
  --split test \
  --checkpoints-dir ./numpy_weights/gsm8k/qwen_8b/lora/epoch_weights \
  --model-name Qwen/Qwen3-8B \
  --use-lora \
  --batch-size 8 \
  --save-csv results/qwen3_8b_gsm8k_lora.csv \
  > logs/eval_gsm8k_qwen3_8b_lora.log 2>&1 &

### IMDB evaluation examples (FULL models, sequential recommended)

# Llama-3.2-3B IMDB (FULL)
nohup python -m codes.utils.eval_checkpoints \
  --dataset imdb \
  --split test \
  --checkpoints-dir ./numpy_weights/imdb/llama32_3b/full/epoch_weights \
  --model-name meta-llama/Llama-3.2-3B \
  --batch-size 8 \
  --save-csv results/llama32_3b_imdb_full.csv \
  > logs/eval_imdb_llama32_3b_full.log 2>&1 &

# Llama-3.1-8B IMDB (FULL)
nohup python -m codes.utils.eval_checkpoints \
  --dataset imdb \
  --split test \
  --checkpoints-dir ./numpy_weights/imdb/llama31_8b/full/epoch_weights \
  --model-name meta-llama/Llama-3.1-8B \
  --batch-size 8 \
  --save-csv results/llama31_8b_imdb_full.csv \
  > logs/eval_imdb_llama31_8b_full.log 2>&1 &

# Mistral-7B IMDB (FULL)
nohup python -m codes.utils.eval_checkpoints \
  --dataset imdb \
  --split test \
  --checkpoints-dir ./numpy_weights/imdb/mistral7b/full/epoch_weights \
  --model-name mistralai/Mistral-7B-v0.1 \
  --batch-size 8 \
  --save-csv results/mistral7b_imdb_full.csv \
  > logs/eval_imdb_mistral7b_full.log 2>&1 &

# Qwen-3-8B IMDB (FULL)
nohup python -m codes.utils.eval_checkpoints \
  --dataset imdb \
  --split test \
  --checkpoints-dir ./numpy_weights/imdb/qwen_8b/full/epoch_weights \
  --model-name Qwen/Qwen3-8B \
  --batch-size 8 \
  --save-csv results/qwen3_8b_imdb_full.csv \
  > logs/eval_imdb_qwen3_8b_full.log 2>&1 &
"""