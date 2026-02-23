import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
from persim import plot_diagrams

def compute_cosine_distance_matrix(weights):
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    return squareform(pdist(weights, metric="cosine"))

def compute_persistence_from_dist_matrix(dist_matrix):
    return ripser(dist_matrix, distance_matrix=True, metric="cosine")["dgms"]

def plot_persistence(diagrams, shortname, output_folder):
    plt.figure(figsize=(6, 4))
    plot_diagrams(diagrams)
    plt.title(f"Persistence Diagram - {shortname}")
    output_path = os.path.join(output_folder, f"{shortname}_H0_H1.png")
    plt.savefig(output_path)
    plt.close()

def concise_filename(fname):
    parts = fname.replace(".npy", "").split("_")
    if "layer" in parts[0]:
        layer = parts[0].replace("layer", "")
        proj = parts[1]
        return f"layer{layer}_{proj}"
    return fname.replace(".npy", "")

def process_npy_files_to_persistence(input_folder, output_folder, file_filter, plot=False):
    os.makedirs(output_folder, exist_ok=True)
    diagrams_folder = os.path.join(output_folder, "SavedDiagrams")
    os.makedirs(diagrams_folder, exist_ok=True)

    npy_files = [f for f in os.listdir(input_folder) if f.endswith(".npy") and file_filter(f)]
    print(f"🔍 Found {len(npy_files)} .npy files in {input_folder}")

    for file in tqdm(npy_files, desc=f"[{os.path.basename(input_folder)}] Generating persistence", ncols=100, mininterval=0.5):
        file_path = os.path.join(input_folder, file)
        weights = np.load(file_path)
        dist_matrix = compute_cosine_distance_matrix(weights)
        diagrams = compute_persistence_from_dist_matrix(dist_matrix)

        shortname = concise_filename(file)
        diagram_file = os.path.join(diagrams_folder, f"{shortname}.pkl")
        with open(diagram_file, "wb") as f:
            pickle.dump(diagrams, f)

        if plot:
            plot_persistence(diagrams, shortname, diagrams_folder)

# --------- Entry Point ---------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=["lora_A", "lora_B", "lora_BA", "full", "baseline", "lora-final"])
    args = parser.parse_args()

    # Map modes to base folder paths
    folder_map = {
        "lora_A": "lora",
        "lora_B": "lora",
        "lora_BA": "loraBA",
        "full": "full",
        "baseline": "baseline",
        "lora-final": "lora-final"
    }
    weight_folder = folder_map[args.mode]

    base_path = f"/staging/users/aerol1/tda/Topo-Tuner/numpy_weights/{args.dataset}/{args.model}/{weight_folder}"
    output_base = f"/staging/users/aerol1/tda/Topo-Tuner/persistence_diagrams/{args.dataset}/{args.model}/{args.mode}"

    def lora_A_filter(f): return f.endswith("_A.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_B_filter(f): return f.endswith("_B.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_BA_filter(f): return f.endswith("_BA.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_final_filter(f): return f.endswith("_BAfinal.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def full_filter(f): return f.endswith(".npy") and any(f"_{k}.npy" in f for k in ["k", "q", "v"])
    def baseline_filter(f): return f.endswith(".npy") and any(f"_{k}" in f for k in ["k", "q", "v"])

    mode_filters = {
        "lora_A": lora_A_filter,
        "lora_B": lora_B_filter,
        "lora_BA": lora_BA_filter,
        "full": full_filter,
        "baseline": baseline_filter,
        "lora-final": lora_final_filter
    }

    file_filter = mode_filters[args.mode]

    if args.mode == "baseline":
        if not os.path.exists(base_path):
            print(f"⚠️ Skipping missing folder: {base_path}")
        else:
            process_npy_files_to_persistence(base_path, output_base, file_filter, plot=True)
    else:
        for epoch in range(1, 101):
            input_folder = os.path.join(base_path, f"epoch_{epoch}")
            output_folder = os.path.join(output_base, f"epoch_{epoch}")
            if not os.path.exists(input_folder):
                print(f"⚠️ Skipping missing folder: {input_folder}")
                continue
            process_npy_files_to_persistence(input_folder, output_folder, file_filter, plot=True)

"""
nohup python code/tda/generate_persistence.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  --mode lora_A \
  > logs/persistence_FinEntity_DeepSeek-Qwen-7B_lora_A.log 2>&1 &

nohup python code/tda/generate_persistence.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  --mode lora_B \
  > logs/persistence_FinEntity_DeepSeek-Qwen-7B_lora_B.log 2>&1 &

nohup python code/tda/generate_persistence.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  --mode lora_BA \
  > logs/persistence_FinEntity_DeepSeek-Qwen-7B_lora_BA.log 2>&1 &

nohup python code/tda/generate_persistence.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  --mode full \
  > logs/persistence_FinEntity_DeepSeek-Qwen-7B_full.log 2>&1 &

nohup python code/tda/generate_persistence.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  --mode baseline \
  > logs/persistence_FinEntity_DeepSeek-Qwen-7B_baseline.log 2>&1 &

  nohup python code/tda/generate_persistence.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  --mode lora-final \
  > logs/persistence_FinEntity_DeepSeek-Qwen-7B_lora-final-100.log 2>&1 &
"""