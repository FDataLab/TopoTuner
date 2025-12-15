import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def _finite_deaths(pd_tuple):
    """Return list of finite death times from a (H0, H1) tuple."""
    if pd_tuple is None:
        return []
    h0, h1 = pd_tuple
    return [d for _, d in list(h0) + list(h1) if d < np.inf]

def plot_betti_curve_epochs(baseline_pd, target_pds, model_type, target_file, save_path):
    # Guard: need baseline and at least one target epoch
    if baseline_pd is None or not target_pds or any(pd is None for pd in target_pds.values()):
        return

    betti_0_base, betti_1_base = baseline_pd[0], baseline_pd[1]

    # Determine a reasonable max scale from finite deaths across baseline + targets
    finite_deaths = _finite_deaths(baseline_pd)
    for pd in target_pds.values():
        finite_deaths.extend(_finite_deaths(pd))

    # Fall back to 1.0 if everything is empty/inf
    max_death = max(finite_deaths, default=1.0)

    # If max_death is 0 (degenerate), expand slightly so we can draw something
    if max_death <= 0:
        max_death = 1.0

    scales = np.linspace(0.0, max_death, 100)

    baseline_curves = {
        "Betti 0": compute_betti_curve(betti_0_base, scales),
        "Betti 1": compute_betti_curve(betti_1_base, scales),
    }

    plt.figure(figsize=(12, 8))

    # Plot baseline (blue, solid/dashed)
    plt.plot(scales, baseline_curves["Betti 0"], label="Baseline - Betti 0", color="blue", linewidth=1.5)
    plt.plot(scales, baseline_curves["Betti 1"], label="Baseline - Betti 1", color="blue", linestyle="--", linewidth=1.5)

    # Prepare epoch colors (gradient across epochs)
    epoch_list = sorted(target_pds.keys())
    cmap = plt.colormaps.get_cmap("viridis").resampled(len(epoch_list))  # change "viridis" to "tab10" if desired

    # Plot each epoch with its own color; solid for Betti 0, dashed for Betti 1
    for idx, epoch in enumerate(epoch_list):
        pd = target_pds[epoch]
        betti_0, betti_1 = pd[0], pd[1]
        curves = {
            "Betti 0": compute_betti_curve(betti_0, scales),
            "Betti 1": compute_betti_curve(betti_1, scales),
        }

        color = cmap(idx)
        plt.plot(scales, curves["Betti 0"], label=f"{model_type} Epoch {epoch} - Betti 0", color=color, linewidth=1.8)
        plt.plot(scales, curves["Betti 1"], label=f"{model_type} Epoch {epoch} - Betti 1", color=color, linestyle="--", linewidth=1.8)

    plt.xlabel("Filtration Parameter (Scale)")
    plt.ylabel("Betti Number")
    plt.title(f"Betti Curves Over Epochs - {target_file.replace('.pkl', '')} - {model_type}")
    plt.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linewidth=0.4, alpha=0.6)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f"{target_file.replace('.pkl', f'_Betti_Epochs_{model_type}.png')}")
    plt.savefig(fig_path, dpi=200)
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

    model_types = ["lora-final", "full"]
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
  > logs/betti_epoch_FinEntity_DeepSeek-Qwen-7B-new.log 2>&1 &
"""
