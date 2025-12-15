import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from persim.sliced_wasserstein import sliced_wasserstein
except ImportError:
    print("❌ persim not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "persim"])
    from persim.sliced_wasserstein import sliced_wasserstein
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp

# Suppress warnings about non-finite points and invalid values
warnings.filterwarnings("ignore", category=UserWarning, module="persim")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.spatial.distance")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_single_sliced_wasserstein(args):
    """Compute Sliced Wasserstein distance for a single pair - for parallel processing."""
    file1, file2, label, epoch, num_projections = args
    try:
        dgm1 = load_pkl(file1)
        dgm2 = load_pkl(file2)
        
        # Helper function to clean persistence diagrams (remove inf/NaN values)
        def clean_diagram(dgm):
            """Remove infinite and NaN values from persistence diagram."""
            if len(dgm) == 0:
                return np.array([[0.0, 0.0]])
            arr = np.array(dgm)
            # Filter out rows with inf or NaN
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            # Remove rows where either column is inf or NaN
            valid_mask = np.isfinite(arr).all(axis=1)
            cleaned = arr[valid_mask]
            if len(cleaned) == 0:
                return np.array([[0.0, 0.0]])
            return cleaned
        
        # Convert to numpy arrays and clean diagrams
        dgm1_h0 = clean_diagram(dgm1[0])
        dgm1_h1 = clean_diagram(dgm1[1])
        dgm2_h0 = clean_diagram(dgm2[0])
        dgm2_h1 = clean_diagram(dgm2[1])
        
        # Ensure 2D arrays (persistence diagrams are birth-death pairs)
        if dgm1_h0.ndim == 1:
            dgm1_h0 = dgm1_h0.reshape(-1, 2)
        if dgm1_h1.ndim == 1:
            dgm1_h1 = dgm1_h1.reshape(-1, 2)
        if dgm2_h0.ndim == 1:
            dgm2_h0 = dgm2_h0.reshape(-1, 2)
        if dgm2_h1.ndim == 1:
            dgm2_h1 = dgm2_h1.reshape(-1, 2)
        
        # Compute Sliced Wasserstein distance for H0 and H1 persistence diagrams
        # num_projections (M) controls the number of directions/projections to use
        try:
            h0 = sliced_wasserstein(dgm1_h0, dgm2_h0, M=num_projections)
            if not np.isfinite(h0):
                h0 = 0.0
        except Exception as e:
            print(f"Warning: H0 computation failed for {os.path.basename(file1)}: {e}")
            h0 = 0.0
            
        try:
            h1 = sliced_wasserstein(dgm1_h1, dgm2_h1, M=num_projections)
            if not np.isfinite(h1):
                h1 = 0.0
        except Exception as e:
            print(f"Warning: H1 computation failed for {os.path.basename(file1)}: {e}")
            h1 = 0.0
        
        return {
            "Type": label,
            "Epoch": epoch,
            "File": os.path.basename(file1),
            "Sliced Wasserstein H0": h0,
            "Sliced Wasserstein H1": h1
        }
    except Exception as e:
        print(f"❌ Skipping {file1} vs {file2}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_direct_comparisons(pairs, label, epoch, max_workers=None, num_projections=50):
    """Optimized parallel Sliced Wasserstein computation with better memory management."""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)

    # Prepare arguments for parallel processing
    args_list = [(file1, file2, label, epoch, num_projections) for file1, file2 in pairs]

    results: list[dict] = []
    if not args_list:
        return results

    # Use ProcessPoolExecutor for CPU-bound Sliced Wasserstein computation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once for better parallelization
        futures = [executor.submit(compute_single_sliced_wasserstein, args) for args in args_list]
        
        # Collect results with progress bar - use as_completed for better responsiveness
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
    parser.add_argument("--num-projections", type=int, default=50,
                       help="Number of projections/directions for Sliced Wasserstein (default: 50)")
    parser.add_argument("--epoch-csv", action="store_true",
                       help="Write/append CSV per epoch to preserve progress incrementally")
    parser.add_argument("--only-q", action="store_true",
                       help="If set, only compute pairs for _q.pkl files (skip _k/_v)")
    args = parser.parse_args()

    base = "/home/kadir/topo"
    dset, model = args.dataset, args.model

    all_results = []
    baseline_path = f"{base}/persistence_diagrams/{dset}/{model}/baseline/SavedDiagrams"

    print(f"🔍 Computing Sliced Wasserstein distances for {dset}/{model}")
    print(f"📁 Baseline path: {baseline_path}")
    print(f"🚀 Using {args.max_workers or min(mp.cpu_count(), 8)} parallel workers")
    print(f"📐 Using {args.num_projections} projections for Sliced Wasserstein")
    
    # Check if baseline path exists
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline path does not exist: {baseline_path}")
        exit(1)

    suffixes = ["_q.pkl", "_k.pkl", "_v.pkl"] if not args.only_q else ["_q.pkl"]
    
    for epoch in range(1, 7):  # Epochs 1 to 6 (skip epoch 0)
        full_path = f"{base}/persistence_diagrams/{dset}/{model}/full/epoch_{epoch}/SavedDiagrams"
        lora_final_path = f"{base}/persistence_diagrams/{dset}/{model}/lora-final/epoch_{epoch}/SavedDiagrams"

        print(f"🔄 Processing epoch {epoch}...")
        
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
                    num_projections=args.num_projections,
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
                    num_projections=args.num_projections,
                )
                all_results += res

        # Write per-epoch CSV if requested
        if args.epoch_csv and all_results:
            df_epoch = pd.DataFrame([r for r in all_results if r["Epoch"] == epoch])
            if not df_epoch.empty:
                os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
                save_path = f"{base}/wasserstein_results/sliced_wasserstein_{dset}_{model}.csv"
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
            save_path = f"{base}/wasserstein_results/sliced_wasserstein_{dset}_{model}.csv"
            df.to_csv(save_path, index=False)
            print(f"\n✅ Saved Sliced Wasserstein results to: {save_path}")
            print(f"📊 Total comparisons: {len(all_results)}")
        else:
            print("❌ No results generated!")
    else:
        # Per-epoch CSV writes already done above; emit a clear summary instead of a misleading error
        print(f"📊 Per-epoch CSV appends complete. Total comparisons so far: {len(all_results)}")
        print(f"📁 Results file: {base}/wasserstein_results/sliced_wasserstein_{dset}_{model}.csv")

"""
# IMDB Sliced Wasserstein Distance Commands (for testing)
# Sliced Wasserstein is faster but approximate compared to full Wasserstein

# Test on a small subset first (Q only, 1 epoch)
python codes/tda/compute_sliced_wasserstein_updated.py \
  --dataset imdb --model llama32_3b --max-workers 8 \
  --num-projections 50 --only-q \
  --epoch-csv

# Full run for Llama-3.2-3B
nohup python codes/tda/compute_sliced_wasserstein_updated.py \
  --dataset imdb --model llama32_3b --max-workers 32 \
  --num-projections 50 \
  > logs/sliced_wasserstein_imdb_llama32_3b.log 2>&1 &

# Full run for Llama-3.1-8B
nohup python codes/tda/compute_sliced_wasserstein_updated.py \
  --dataset imdb --model llama31_8b --max-workers 32 \
  --num-projections 50 \
  > logs/sliced_wasserstein_imdb_llama31_8b.log 2>&1 &

# Full run for Mistral-7B
nohup python codes/tda/compute_sliced_wasserstein_updated.py \
  --dataset imdb --model mistral7b --max-workers 32 \
  --num-projections 50 \
  > logs/sliced_wasserstein_imdb_mistral7b.log 2>&1 &

# Comparison: Higher number of projections (more accurate but slower)
# --num-projections 100  # More accurate approximation
# --num-projections 25   # Faster but less accurate
"""

