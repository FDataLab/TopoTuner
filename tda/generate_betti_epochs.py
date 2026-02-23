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

def load_persistence_diagram(base_path, target_file):
    file_path = os.path.join(base_path, target_file)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def plot_betti_curve_epochs(baseline_pd, target_pds, model_type, target_file, save_path):
    if baseline_pd is None or any(pd is None for pd in target_pds.values()):
        return

    betti_0_base, betti_1_base = baseline_pd[0], baseline_pd[1]
    max_death = max(
        [death for _, death in list(betti_0_base) + list(betti_1_base) if death < np.inf] +
        [death for pd in target_pds.values() for _, death in list(pd[0]) + list(pd[1]) if death < np.inf],
        default=1.0
    )

    scales = np.linspace(0.0, max_death, 100)

    baseline_curves = {
        "Betti 0": compute_betti_curve(betti_0_base, scales),
        "Betti 1": compute_betti_curve(betti_1_base, scales),
    }

    plt.figure(figsize=(12, 8))

    # Plot baseline
    plt.plot(scales, baseline_curves["Betti 0"], label="Baseline - Betti 0", color="blue", linewidth=1.0)
    plt.plot(scales, baseline_curves["Betti 1"], label="Baseline - Betti 1", color="blue", linestyle="--", linewidth=1.0)

    # Plot epochs
    for epoch, pd in target_pds.items():
        betti_0, betti_1 = pd[0], pd[1]
        curves = {
            "Betti 0": compute_betti_curve(betti_0, scales),
            "Betti 1": compute_betti_curve(betti_1, scales),
        }

        alpha = 0.3 + 0.1 * epoch if epoch < 6 else 1.0
        linewidth = 1.0 if epoch < 6 else 1.8

        plt.plot(scales, curves["Betti 0"], label=f"{model_type} Epoch {epoch} - Betti 0", color="red", alpha=alpha, linewidth=linewidth)
        plt.plot(scales, curves["Betti 1"], label=f"{model_type} Epoch {epoch} - Betti 1", color="red", linestyle="--", alpha=alpha, linewidth=linewidth)

    plt.xlabel("Filtration Parameter (Scale)")
    plt.ylabel("Betti Number")
    plt.title(f"Betti Curves Over Epochs - {target_file.replace('.pkl', '')} - {model_type}")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f"{target_file.replace('.pkl', f'_Betti_Epochs_{model_type}.png')}")
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Saved plot: {fig_path}")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    model = args.model

    base_dir = f"/staging/users/aerol1/tda/Topo-Tuner/persistence_diagrams/{dataset}/{model}"
    save_dir = f"/staging/users/aerol1/tda/Topo-Tuner/betti_curves_new/{dataset}/{model}"

    model_types = ["lora_BA", "full"]
    layers = range(27)
    projections = ["k", "q", "v"]

    for model_type in model_types:
        for layer in layers:
            for proj in projections:
                target_file = f"layer{layer}_{proj}.pkl"
                print(f"\n📌 Working on: {model_type} - {target_file}")

                baseline_path = os.path.join(base_dir, "baseline", "SavedDiagrams")
                baseline_pd = load_persistence_diagram(baseline_path, target_file)

                target_pds = {}
                for epoch in range(1, 7):
                    target_path = os.path.join(base_dir, model_type, f"epoch_{epoch}", "SavedDiagrams")
                    target_pd = load_persistence_diagram(target_path, target_file)
                    if target_pd:
                        target_pds[epoch] = target_pd

                save_path = os.path.join(save_dir, model_type)
                plot_betti_curve_epochs(baseline_pd, target_pds, model_type, target_file, save_path)

"""
nohup python code/tda/generate_betti_epochs.py \
  --dataset FinEntity \
  --model DeepSeek-Qwen-7B \
  > logs/betti_epoch_FinEntity_DeepSeek-Qwen-7B.log 2>&1 &
"""