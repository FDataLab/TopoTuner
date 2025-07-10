import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from gudhi.wasserstein import wasserstein_distance

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_direct_comparisons(pairs, label, epoch):
    results = []
    for file1, file2 in tqdm(pairs, desc=f"{label} - Epoch {epoch}"):
        try:
            dgm1 = load_pkl(file1)
            dgm2 = load_pkl(file2)

            h0 = wasserstein_distance(dgm1[0], dgm2[0])
            h1 = wasserstein_distance(dgm1[1], dgm2[1])
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
    args = parser.parse_args()

    base = "/staging/users/aerol1/tda/Topo-Tuner"
    dset, model = args.dataset, args.model

    all_results = []
    baseline_path = f"{base}/persistence_diagrams/{dset}/{model}/baseline/SavedDiagrams"

    for epoch in range(0, 7):  # Epochs 0 to 6
        full_path = f"{base}/persistence_diagrams/{dset}/{model}/full/epoch_{epoch}/SavedDiagrams"
        lora_final_path = f"{base}/persistence_diagrams/{dset}/{model}/lora-final/epoch_{epoch}/SavedDiagrams"

        for suffix in ["_q.pkl", "_k.pkl", "_v.pkl"]:
            # Baseline vs Full Finetuned
            pairs = []
            for fname in os.listdir(baseline_path):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_path, fname)
                    f2 = os.path.join(full_path, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            all_results += compute_direct_comparisons(pairs, "Baseline vs Full Finetuned", epoch)

            # Baseline vs LoRA-final
            pairs = []
            for fname in os.listdir(baseline_path):
                if fname.endswith(suffix):
                    f1 = os.path.join(baseline_path, fname)
                    f2 = os.path.join(lora_final_path, fname)
                    if os.path.exists(f2):
                        pairs.append((f1, f2))
            all_results += compute_direct_comparisons(pairs, "Baseline vs LoRA-final", epoch)

    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs(f"{base}/wasserstein_results", exist_ok=True)
    save_path = f"{base}/wasserstein_results/wasserstein_{dset}_{model}_baseline_vs_lora-final_fullfinetune.csv"
    df.to_csv(save_path, index=False)
    print(f"\n✅ Saved Wasserstein results to: {save_path}")

"""
nohup python code/tda/compute_wasserstein-final.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  > logs/wasserstein_FinEntity_DeepSeek-Qwen-7B-new.log 2>&1 &
"""