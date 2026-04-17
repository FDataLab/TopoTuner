import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from ripser import ripser
from persim import plot_diagrams

MAX_ROWS = None  # set via CLI; None = no limit

def compute_cosine_distance_matrix(weights):
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    return squareform(pdist(weights, metric="cosine"))

MAXDIM = 1  # set via CLI; 0 = H0 only, 1 = H0+H1

def compute_persistence_from_dist_matrix(dist_matrix):
    return ripser(dist_matrix, distance_matrix=True, metric="cosine", maxdim=MAXDIM)["dgms"]

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
    import time
    os.makedirs(output_folder, exist_ok=True)
    diagrams_folder = os.path.join(output_folder, "SavedDiagrams")
    os.makedirs(diagrams_folder, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy") and file_filter(f)])
    print(f"🔍 Found {len(npy_files)} .npy files in {input_folder}", flush=True)

    for i, file in enumerate(npy_files):
        file_path = os.path.join(input_folder, file)
        weights = np.load(file_path)
        original_shape = weights.shape

        # Subsample rows if matrix is too large
        if MAX_ROWS and weights.ndim == 2 and weights.shape[0] > MAX_ROWS:
            rng = np.random.RandomState(42)
            idx = rng.choice(weights.shape[0], MAX_ROWS, replace=False)
            idx.sort()
            weights = weights[idx]
            print(f"  [{i+1}/{len(npy_files)}] {file} shape {original_shape} → subsampled to {weights.shape}", flush=True)
        else:
            print(f"  [{i+1}/{len(npy_files)}] {file} shape {original_shape}", flush=True)

        t0 = time.time()
        dist_matrix = compute_cosine_distance_matrix(weights)
        t_dist = time.time() - t0
        print(f"    dist_matrix: {dist_matrix.shape} ({t_dist:.1f}s)", flush=True)

        t0 = time.time()
        diagrams = compute_persistence_from_dist_matrix(dist_matrix)
        t_rips = time.time() - t0
        print(f"    ripser: H0={len(diagrams[0])} pts, H1={len(diagrams[1]) if len(diagrams)>1 else 'N/A'} pts ({t_rips:.1f}s)", flush=True)

        shortname = concise_filename(file)
        diagram_file = os.path.join(diagrams_folder, f"{shortname}.pkl")
        with open(diagram_file, "wb") as f:
            pickle.dump(diagrams, f)

        if plot:
            plot_persistence(diagrams, shortname, diagrams_folder)

# --------- Entry Point ---------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", default=None, choices=["lora_A", "lora_B", "lora_BA", "full", "baseline", "lora-final"])
    parser.add_argument("--input-dir", default=None,
                        help="Direct path to folder containing .npy files (bypasses dataset/model/mode path construction)")
    parser.add_argument("--output-dir", default=None,
                        help="Direct output path for persistence diagrams (used with --input-dir)")
    parser.add_argument("--filter", default="full", choices=["full", "baseline", "lora_A", "lora_B", "lora_BA", "lora-final"],
                        help="Which file filter to apply (default: full, matches q/k/v/o)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Subsample matrices with more rows than this (e.g. 1024). Keeps persistence tractable for large matrices.")
    parser.add_argument("--maxdim", type=int, default=1,
                        help="Max homology dimension for ripser (0=H0 only, 1=H0+H1). Default: 1")
    parser.add_argument(
        "--projections",
        type=str,
        default="q,k,v,o",
        help="Direct mode only: comma-separated projection suffixes to load (*.npy with _{proj}.npy). "
        "Example: --projections v,o to skip q and k.",
    )
    args = parser.parse_args()

    if args.max_rows:
        MAX_ROWS = args.max_rows
        print(f"⚙️ Subsampling matrices with > {MAX_ROWS} rows", flush=True)
    MAXDIM = args.maxdim
    print(f"⚙️ Ripser maxdim={MAXDIM} ({'H0 only' if MAXDIM == 0 else 'H0+H1'})", flush=True)

    # Direct path mode: --input-dir + --output-dir
    if args.input_dir:
        if not args.output_dir:
            parser.error("--output-dir is required when using --input-dir")

        projs = [p.strip() for p in args.projections.split(",") if p.strip()]
        if not projs:
            parser.error("--projections must list at least one of q,k,v,o")

        def npy_filter(f: str) -> bool:
            return f.endswith(".npy") and any(f"_{k}.npy" in f for k in projs)

        filt = npy_filter
        if not os.path.exists(args.input_dir):
            print(f"⚠️ Input dir not found: {args.input_dir}")
        else:
            process_npy_files_to_persistence(args.input_dir, args.output_dir, filt, plot=True)
        exit(0)

    if not args.dataset or not args.model or not args.mode:
        parser.error("--dataset, --model, and --mode are required unless using --input-dir/--output-dir")

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

    base_path = f"./numpy_weights/{args.dataset}/{args.model}/{weight_folder}"
    output_base = f"./persistence_diagrams/{args.dataset}/{args.model}/{args.mode}"

    def lora_A_filter(f): return f.endswith("_A.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_B_filter(f): return f.endswith("_B.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_BA_filter(f): return f.endswith("_BA.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def lora_final_filter(f): return f.endswith("_BAfinal.npy") and any(f"_{k}_" in f for k in ["k", "q", "v"])
    def full_filter(f): return f.endswith(".npy") and any(f"_{k}.npy" in f for k in ["k", "q", "v", "o"])
    def baseline_filter(f): return f.endswith(".npy") and any(f"_{k}.npy" in f for k in ["k", "q", "v", "o"])

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
        # Use epoch 0 of full model as baseline (original pre-trained weights)
        baseline_path = os.path.join(f"./numpy_weights/{args.dataset}/{args.model}/full/epoch_weights/checkpoint-epoch-0/numpy_weights")
        if not os.path.exists(baseline_path):
            print(f"⚠️ Skipping missing baseline folder: {baseline_path}")
        else:
            process_npy_files_to_persistence(baseline_path, output_base, file_filter, plot=True)
    else:
        if args.mode == "lora-final":
            # lora-final has a different structure: epoch_0, epoch_1, etc. directly
            for epoch in range(7):  # epochs 0-6
                input_folder = os.path.join(base_path, f"epoch_{epoch}")
                output_folder = os.path.join(output_base, f"epoch_{epoch}")
                if not os.path.exists(input_folder):
                    print(f"⚠️ Skipping missing folder: {input_folder}")
                    continue
                process_npy_files_to_persistence(input_folder, output_folder, file_filter, plot=True)
        else:
            # Look for checkpoint directories (checkpoint-epoch-0, checkpoint-epoch-1, etc.)
            epoch_weights_path = os.path.join(base_path, "epoch_weights")
            if not os.path.exists(epoch_weights_path):
                print(f"⚠️ Skipping missing folder: {epoch_weights_path}")
            else:
                for epoch in range(7):  # epochs 0-6
                    input_folder = os.path.join(epoch_weights_path, f"checkpoint-epoch-{epoch}", "numpy_weights")
                    output_folder = os.path.join(output_base, f"epoch_{epoch}")
                    if not os.path.exists(input_folder):
                        print(f"⚠️ Skipping missing folder: {input_folder}")
                        continue
                    process_npy_files_to_persistence(input_folder, output_folder, file_filter, plot=True)

"""
# MMLU Persistence Generation Commands
# Run these after fine-tuning, evaluation, and generate_ba_final are complete
# We only need 3 modes: lora-final, full, baseline

# Step 1: Generate BA and BAfinal weights first
python codes/tda/generate_ba_final.py --dataset mmlu --model llama32_3b
python codes/tda/generate_ba_final.py --dataset mmlu --model llama31_8b  
python codes/tda/generate_ba_final.py --dataset mmlu --model mistral7b
python codes/tda/generate_ba_final.py --dataset mmlu --model qwen_8b

# Step 2: Generate persistence diagrams for comparison
# Llama-3.2-3B
nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model llama32_3b \
  --mode lora-final \
  > logs/persistence_mmlu_llama32_3b_lora.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model llama32_3b \
  --mode full \
  > logs/persistence_mmlu_llama32_3b_full.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model llama32_3b \
  --mode baseline \
  > logs/persistence_mmlu_llama32_3b_baseline.log 2>&1 &

# Llama-3.1-8B
nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model llama31_8b \
  --mode lora-final \
  > logs/persistence_mmlu_llama31_8b_lora.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model llama31_8b \
  --mode full \
  > logs/persistence_mmlu_llama31_8b_full.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model llama31_8b \
  --mode baseline \
  > logs/persistence_mmlu_llama31_8b_baseline.log 2>&1 &

# Mistral-7B
nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model mistral7b \
  --mode lora-final \
  > logs/persistence_mmlu_mistral7b_lora.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model mistral7b \
  --mode full \
  > logs/persistence_mmlu_mistral7b_full.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model mistral7b \
  --mode baseline \
  > logs/persistence_mmlu_mistral7b_baseline.log 2>&1 &

# Qwen-3-8B
nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model qwen_8b \
  --mode lora-final \
  > logs/persistence_mmlu_qwen_8b_lora.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model qwen_8b \
  --mode full \
  > logs/persistence_mmlu_qwen_8b_full.log 2>&1 &

nohup python codes/tda/generate_persistence.py \
  --dataset mmlu \
  --model qwen_8b \
  --mode baseline \
  > logs/persistence_mmlu_qwen_8b_baseline.log 2>&1 &

# Step 3: After persistence diagrams are generated, run Wasserstein distance analysis
# (This will be implemented in a separate script)
"""