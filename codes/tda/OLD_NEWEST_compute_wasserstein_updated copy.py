import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from persim import wasserstein
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import multiprocessing as mp

# Suppress warnings about non-finite points
warnings.filterwarnings("ignore", category=UserWarning, module="persim")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_single_wasserstein(args):
    """Compute Wasserstein distance for a single pair - for parallel processing."""
    file1, file2, label, epoch = args
    try:
        dgm1 = load_pkl(file1)
        dgm2 = load_pkl(file2)

        # Compute Wasserstein distance for H0 and H1 persistence diagrams
        h0 = wasserstein(dgm1[0], dgm2[0])
        h1 = wasserstein(dgm1[1], dgm2[1])
        
        return {
            "Type": label,
            "Epoch": epoch,
            "File": os.path.basename(file1),
            "Wasserstein H0": h0,
            "Wasserstein H1": h1
        }
    except Exception as e:
        print(f"❌ Skipping {file1} vs {file2}: {e}")
        return None

def compute_direct_comparisons(pairs, label, epoch, max_workers=None):
    """Optimized parallel Wasserstein computation with better memory management."""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    # Prepare arguments for parallel processing
    args_list = [(file1, file2, label, epoch) for file1, file2 in pairs]

    results: list[dict] = []
    if not args_list:
        return results

    # Use ProcessPoolExecutor for CPU-bound Wasserstein computation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once for better parallelization
        futures = [executor.submit(compute_single_wasserstein, args) for args in args_list]
        
        # Collect results with progress bar - use as_completed for better responsiveness
        from concurrent.futures import as_completed
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{label} - Epoch {epoch}"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"❌ Error in {label} Epoch {epoch}: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: min(CPU count, 8))")
    parser.add_argument("--epoch-csv", action="store_true",
                       help="Write/append CSV per epoch to preserve progress incrementally")
    parser.add_argument("--only-q", action="store_true",
                       help="If set, only compute pairs for _q.pkl files (skip _k/_v)")
    args = parser.parse_args()

    base = "/home/kadir/topo"
    dset, model = args.dataset, args.model

    all_results = []
    baseline_path = f"{base}/persistence_diagrams/{dset}/{model}/baseline/SavedDiagrams"

    print(f"🔍 Computing Wasserstein distances for {dset}/{model}")
    print(f"📁 Baseline path: {baseline_path}")
    print(f"🚀 Using {args.max_workers or min(mp.cpu_count(), 8)} parallel workers")
    if args.only_q:
        print(f"⚡ Only-Q mode: Processing only _q.pkl files (skipping _k/_v)")
    
    # Check if baseline path exists
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline path does not exist: {baseline_path}")
        exit(1)

    for epoch in range(0, 7):  # Epochs 0 to 6
        full_path = f"{base}/persistence_diagrams/{dset}/{model}/full/epoch_{epoch}/SavedDiagrams"
        lora_final_path = f"{base}/persistence_diagrams/{dset}/{model}/lora-final/epoch_{epoch}/SavedDiagrams"

        print(f"🔄 Processing epoch {epoch}...")
        
        # Determine which suffixes to process based on --only-q flag
        suffixes = ["_q.pkl"] if args.only_q else ["_q.pkl", "_k.pkl", "_v.pkl"]
        
        for suffix in suffixes:
            # Baseline vs Full Finetuned
            pairs = []
            for fname in os.listdir(baseline_path):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_path, fname)
                    f2 = os.path.join(full_path, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            if pairs:
                res = compute_direct_comparisons(
                    pairs,
                    "Baseline vs Full Finetuned",
                    epoch,
                    max_workers=args.max_workers,
                )
                all_results += res

            # Baseline vs LoRA-final
            pairs = []
            for fname in os.listdir(baseline_path):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_path, fname)
                    f2 = os.path.join(lora_final_path, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            if pairs:
                res = compute_direct_comparisons(
                    pairs,
                    "Baseline vs LoRA-final",
                    epoch,
                    max_workers=args.max_workers,
                )
                all_results += res

        # Write per-epoch CSV if requested
        if args.epoch_csv and all_results:
            df_epoch = pd.DataFrame([r for r in all_results if r["Epoch"] == epoch])
            if not df_epoch.empty:
                os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
                save_path = f"{base}/wasserstein_results/wasserstein_{dset}_{model}.csv"
                # Append or create with header
                if os.path.exists(save_path):
                    df_epoch.to_csv(save_path, mode="a", header=False, index=False)
                else:
                    df_epoch.to_csv(save_path, index=False)
                print(f"✅ Appended epoch {epoch} results to: {save_path}")

    # Save results
    if not args.epoch_csv:
        if all_results:
            df = pd.DataFrame(all_results)
            os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
            save_path = f"{base}/wasserstein_results/wasserstein_{dset}_{model}.csv"
            df.to_csv(save_path, index=False)
            print(f"\n✅ Saved Wasserstein results to: {save_path}")
            print(f"📊 Total comparisons: {len(all_results)}")
        else:
            print("❌ No results generated!")
    else:
        # Per-epoch CSV writes already done above; emit a clear summary instead of a misleading error
        print(f"📊 Per-epoch CSV appends complete. Total comparisons so far: {len(all_results)}")
        print(f"📁 Results file: {base}/wasserstein_results/wasserstein_{dset}_{model}.csv")

"""
# MMLU Wasserstein Distance Commands (OPTIMIZED with parallel processing)
# Run these after persistence diagrams are generated

# Mistral-7B (optimized with 8 parallel workers)
nohup python codes/tda/compute_wasserstein_updated.py \
  --dataset mmlu --model mistral7b --max-workers 8 \
  > logs/wasserstein_mmlu_mistral7b_optimized.log 2>&1 &

# Llama-3.2-3B (optimized)
nohup python codes/tda/compute_wasserstein_updated.py \
  --dataset mmlu --model llama32_3b --max-workers 8 \
  > logs/wasserstein_mmlu_llama32_3b_optimized.log 2>&1 &

# Llama-3.1-8B (optimized)
nohup python codes/tda/compute_wasserstein_updated.py \
  --dataset mmlu --model llama31_8b --max-workers 8 \
  > logs/wasserstein_mmlu_llama31_8b_optimized.log 2>&1 &

# Qwen-3-8B (optimized)
nohup python codes/tda/compute_wasserstein_updated.py \
  --dataset mmlu --model qwen_8b --max-workers 8 \
  > logs/wasserstein_mmlu_qwen_8b_optimized.log 2>&1 &

# For systems with more cores, you can use more workers:
# --max-workers 16  # Use 16 parallel workers
# --max-workers 32  # Use 32 parallel workers
"""

