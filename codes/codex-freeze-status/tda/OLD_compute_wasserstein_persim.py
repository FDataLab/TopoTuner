import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from persim import wasserstein
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp

# Suppress warnings about non-finite points from persim
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

        # Compute Wasserstein distance for H0 and H1 persistence diagrams (Persim, default order)
        # Note: persim.wasserstein does not accept 'order' or 'p' in this version
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagrams-dir", required=True, help="Directory containing .pkl diagrams")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file")
    parser.add_argument("--max-workers", type=int, default=mp.cpu_count(), help="Number of parallel processes")
    parser.add_argument("--only-q", action="store_true", help="Only compare _q.pkl files")
    args = parser.parse_args()

    diagram_files = []
    for root, _, files in os.walk(args.diagrams_dir):
        for f in files:
            if f.endswith(".pkl"):
                if args.only_q and "_q.pkl" not in f:
                    continue
                diagram_files.append(os.path.join(root, f))

    diagram_files = sorted(list(set(diagram_files))) # Ensure unique and sorted

    if not diagram_files:
        print(f"No .pkl files found in {args.diagrams_dir} matching criteria.")
        return

    # Generate all unique pairs for comparison
    tasks = []
    for i in range(len(diagram_files)):
        for j in range(i + 1, len(diagram_files)):
            file1 = diagram_files[i]
            file2 = diagram_files[j]

            # Extract epoch and type from filename (assuming format like 'layerX_q.pkl')
            # This logic might need to be more robust depending on actual filenames
            file1_base = os.path.basename(file1)
            file2_base = os.path.basename(file2)

            # Example: layer0_q.pkl -> epoch 0, type q
            # For baseline vs finetuned, we need to infer 'Type' and 'Epoch'
            # This script is designed for pairwise comparisons within a single model/epoch context
            # For cross-epoch/cross-model comparisons, the 'Type' and 'Epoch' logic needs refinement
            
            # For simplicity, let's assume we are comparing within the same epoch/type for now
            # and the 'label' and 'epoch' will be derived from the directory structure or a more complex regex
            
            # Placeholder for label and epoch extraction
            label = "Pairwise" # Default label
            epoch = 0 # Default epoch

            # Attempt to extract epoch from parent directory name if it follows 'epoch_X' pattern
            import re
            epoch_match = re.search(r'epoch_(\d+)', os.path.dirname(file1))
            if epoch_match:
                epoch = int(epoch_match.group(1))

            # Attempt to extract type from file name (e.g., 'q', 'k', 'v')
            type_match = re.search(r'_(q|k|v)\.pkl$', file1_base)
            if type_match:
                label = f"Type {type_match.group(1).upper()}"

            tasks.append((file1, file2, label, epoch))

    print(f"Planned comparisons: {len(tasks)} from {len(diagram_files)} files")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Write header if file is new or empty
    if not os.path.exists(args.output_csv) or os.path.getsize(args.output_csv) == 0:
        with open(args.output_csv, 'w') as f:
            f.write("Type,Epoch,File,Wasserstein H0,Wasserstein H1\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(compute_single_wasserstein, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Wasserstein"):
            result = future.result()
            if result:
                results.append(result)
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"✅ All results saved to {args.output_csv}")
    else:
        print("No results generated!")

if __name__ == "__main__":
    main()