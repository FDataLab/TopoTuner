import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm
import numpy as np

# Path to your CSV file
#csv_path = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_results/wasserstein_FinEntity_DeepSeek-Qwen-7B_baseline_vs_lora-final_fullfinetune.csv"  # Replace this with your actual path
csv_path = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_results/wasserstein_FinEntity_DeepSeek-Llama-8B_baseline_vs_lora-final_fullfinetune.csv"

# Load CSV data
df = pd.read_csv(csv_path)

# Filter for 'Baseline vs Full Finetune'
df = df[df["Type"].str.contains("Baseline vs Full Finetuned|Baseline vs LoRA-final")]

# Extract layer number and head type (q/k/v)
df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", x).group(1)))
df["HeadType"] = df["File"].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))
df["Method"] = df["Type"].str.replace("Baseline vs ", "").str.strip()

# ✅ Log head type counts
print("=== Number of lines per head type ===")
for head in ["q", "k", "v"]:
    count = df[df["HeadType"] == head]["File"].nunique()
    print(f"{head.upper()}: {count} unique lines")

# ✅ Log method + epoch counts from actual Epoch column
print("\n=== Number of epochs per method (from Epoch column) ===")
if "Epoch" in df.columns:
    for method in df["Method"].unique():
        epoch_count = df[df["Method"] == method]["Epoch"].nunique()
        print(f"{method}: {epoch_count} epochs")
else:
    print("⚠️ Epoch column not found in the CSV.")

# Create output folder for plots
output_dir = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_plots/finentity_llama"
os.makedirs(output_dir, exist_ok=True)

# Unique keys: (Method, Epoch)
unique_keys = sorted(df[["Method", "Epoch"]].drop_duplicates().values.tolist())
color_map = {}

# Generate 12 visually distinct colors using a colormap
cmap = cm.get_cmap("tab20", len(unique_keys))  # tab20 = colorful
for idx, key in enumerate(unique_keys):
    color_map[tuple(key)] = cmap(idx)

# Plot for each head type and metric
for head in ["q", "k", "v"]:
    subset = df[df["HeadType"] == head]

    for metric in ["Wasserstein H0", "Wasserstein H1"]:
        plt.figure()

        # Group by Method + Epoch
        for (method, epoch), group in subset.groupby(["Method", "Epoch"]):
            group = group.sort_values("Layer")
            label = f"{method} - Epoch {epoch}"
            color = color_map.get((method, epoch), "gray")

            plt.plot(group["Layer"], group[metric], marker='o', color=color, label=label)

        plt.title(f"Baseline vs Finetuned - {head.upper()} - {metric}")
        plt.xlabel("Layer")
        plt.ylabel("Wasserstein Distance")
        plt.grid(True)
        plt.legend(fontsize="x-small", ncol=2)
        plt.tight_layout()

        filename = f"{head}_{metric.replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()


print(f"✅ Plots saved in: {output_dir}/")