import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from persim import wasserstein
import warnings

# Suppress warnings about non-finite points
warnings.filterwarnings("ignore", category=UserWarning, module="persim")

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_persistence_diagram(dgm):
    """Clean persistence diagram by removing non-finite points."""
    if len(dgm) == 0:
        return dgm
    
    # Convert to numpy array if not already
    dgm = np.array(dgm)
    
    # Remove rows with non-finite values
    finite_mask = np.isfinite(dgm).all(axis=1)
    cleaned = dgm[finite_mask]
    
    return cleaned

def compute_wasserstein_optimized(dgm1, dgm2, max_points=1000):
    """Compute Wasserstein distance with optimizations."""
    # Clean both diagrams
    dgm1_clean = clean_persistence_diagram(dgm1)
    dgm2_clean = clean_persistence_diagram(dgm2)
    
    # If either diagram is empty, return 0
    if len(dgm1_clean) == 0 or len(dgm2_clean) == 0:
        return 0.0
    
    # Subsample if too many points (major optimization!)
    if len(dgm1_clean) > max_points:
        indices = np.random.choice(len(dgm1_clean), max_points, replace=False)
        dgm1_clean = dgm1_clean[indices]
    
    if len(dgm2_clean) > max_points:
        indices = np.random.choice(len(dgm2_clean), max_points, replace=False)
        dgm2_clean = dgm2_clean[indices]
    
    try:
        return wasserstein(dgm1_clean, dgm2_clean)
    except Exception as e:
        print(f"⚠️ Wasserstein computation failed: {e}")
        return float('inf')

def compute_direct_comparisons_optimized(pairs, label, epoch, max_points=1000):
    """Optimized version with subsampling and error handling."""
    results = []
    for file1, file2 in tqdm(pairs, desc=f"{label} - Epoch {epoch}"):
        try:
            dgm1 = load_pkl(file1)
            dgm2 = load_pkl(file2)

            # Compute Wasserstein distance for H0 and H1 persistence diagrams
            h0 = compute_wasserstein_optimized(dgm1[0], dgm2[0], max_points)
            h1 = compute_wasserstein_optimized(dgm1[1], dgm2[1], max_points)
            
            results.append({
                "Type": label,
                "Epoch": epoch,
                "File": os.path.basename(file1),
                "Wasserstein H0": h0,
                "Wasserstein H1": h1
            })
        except Exception as e:
            print(f"❌ Skipping {file1} vs {file2}: {e}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-points", type=int, default=1000, 
                       help="Maximum points per persistence diagram (default: 1000)")
    parser.add_argument("--epochs", type=int, default=7, 
                       help="Number of epochs to process (default: 7)")
    args = parser.parse_args()

    base = "/home/kadir/topo"
    dset, model = args.dataset, args.model

    all_results = []
    baseline_path = f"{base}/persistence_diagrams/{dset}/{model}/baseline/SavedDiagrams"

    print(f"🔍 Computing Wasserstein distances for {dset}/{model}")
    print(f"📁 Baseline path: {baseline_path}")
    print(f"🎯 Max points per diagram: {args.max_points}")
    
    # Check if baseline path exists
    if not os.path.exists(baseline_path):
        print(f"❌ Baseline path does not exist: {baseline_path}")
        exit(1)

    for epoch in range(0, args.epochs):  # Epochs 0 to args.epochs-1
        full_path = f"{base}/persistence_diagrams/{dset}/{model}/full/epoch_{epoch}/SavedDiagrams"
        lora_final_path = f"{base}/persistence_diagrams/{dset}/{model}/lora-final/epoch_{epoch}/SavedDiagrams"

        print(f"🔄 Processing epoch {epoch}...")
        
        for suffix in ["_q.pkl", "_k.pkl", "_v.pkl"]:
            # Baseline vs Full Finetuned
            pairs = []
            for fname in os.listdir(baseline_path):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_path, fname)
                    f2 = os.path.join(full_path, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            if pairs:
                all_results += compute_direct_comparisons_optimized(
                    pairs, "Baseline vs Full Finetuned", epoch, args.max_points)

            # Baseline vs LoRA-final
            pairs = []
            for fname in os.listdir(baseline_path):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_path, fname)
                    f2 = os.path.join(lora_final_path, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            if pairs:
                all_results += compute_direct_comparisons_optimized(
                    pairs, "Baseline vs LoRA-final", epoch, args.max_points)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
        save_path = f"{base}/wasserstein_results/wasserstein_{dset}_{model}_optimized.csv"
        df.to_csv(save_path, index=False)
        print(f"\n✅ Saved Wasserstein results to: {save_path}")
        print(f"📊 Total comparisons: {len(all_results)}")
    else:
        print("❌ No results generated!")

"""
# Optimized Wasserstein Distance Commands
# Much faster with subsampling and error handling

# Mistral-7B (optimized with 1000 max points)
nohup python codes/tda/compute_wasserstein_optimized.py \
  --dataset mmlu --model mistral7b --max-points 1000 \
  > logs/wasserstein_mmlu_mistral7b_optimized.log 2>&1 &

# For even faster computation, use fewer points:
nohup python codes/tda/compute_wasserstein_optimized.py \
  --dataset mmlu --model mistral7b --max-points 500 \
  > logs/wasserstein_mmlu_mistral7b_fast.log 2>&1 &

# Process only specific epochs:
nohup python codes/tda/compute_wasserstein_optimized.py \
  --dataset mmlu --model mistral7b --max-points 1000 --epochs 3 \
  > logs/wasserstein_mmlu_mistral7b_epochs0-2.log 2>&1 &
"""
