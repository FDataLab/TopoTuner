import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def compute_betti_curve(diagram, scales):
    return [
        sum(birth <= scale < death for birth, death in diagram if death < np.inf)
        for scale in scales
    ]

def process_betti_curves(dataset, model, model_type, epoch, target_file, save_dir):
    min_scale = 0.0
    max_scale = 0.0
    all_betti_curves = {}

    if model_type == "baseline":
        base_path = f"/staging/users/aerol1/tda/Topo-Tuner/persistence_diagrams/{dataset}/{model}/{model_type}/SavedDiagrams"
    else:
        base_path = f"/staging/users/aerol1/tda/Topo-Tuner/persistence_diagrams/{dataset}/{model}/{model_type}/epoch_{epoch}/SavedDiagrams"

    file_path = os.path.join(base_path, target_file)

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Processing: {file_path}")
    with open(file_path, "rb") as f:
        persistence_diagrams = pickle.load(f)

    betti_0, betti_1 = persistence_diagrams[0], persistence_diagrams[1]
    max_betti_0 = max((death for _, death in betti_0 if death < np.inf), default=0)
    max_betti_1 = max((death for _, death in betti_1 if death < np.inf), default=0)
    max_scale = max(max_betti_0, max_betti_1)

    num_scales = 100
    scales = np.linspace(min_scale, max_scale, num_scales)

    all_betti_curves[model_type] = {
        "Betti 0": compute_betti_curve(betti_0, scales),
        "Betti 1": compute_betti_curve(betti_1, scales),
        "scales": scales,
    }

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(
        save_dir, f"{target_file.replace('.pkl', f'_BettiCurves_{model_type}_Epoch{epoch}.png')}"
    )

    plt.figure(figsize=(12, 8))
    for key, curves in all_betti_curves.items():
        plt.plot(curves["scales"], curves["Betti 0"], label=f"{key} - Betti 0")
        plt.plot(curves["scales"], curves["Betti 1"], linestyle="--", label=f"{key} - Betti 1")

    plt.xlabel("Filtration Parameter (Scale)")
    plt.ylabel("Betti Number")
    plt.title(f"Betti Curves - {target_file.replace('.pkl', '')} (Epoch {epoch})")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Saved plot: {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    save_base = f"/staging/users/aerol1/tda/Topo-Tuner/betti_curves_new/{dataset}/{model}/"

    model_types = ["baseline", "lora-final", "full"]

    for layer in range(28):
        for proj in ["k", "q", "v"]:
            target_file = f"layer{layer}_{proj}.pkl"

            # Baseline once (no epoch)
            process_betti_curves(dataset, model, "baseline", "NA", target_file, save_base)

            # Epoched models from 1 to 6
            for epoch in range(1, 7):
                for model_type in ["lora-final", "full"]:
                    process_betti_curves(dataset, model, model_type, epoch, target_file, save_base)

"""
nohup python code/tda/generate_betti.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  > logs/betti_FinEntity_DeepSeek-Qwen-7B.log 2>&1 &
"""