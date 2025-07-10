import os
import pandas as pd
import plotly.express as px
import plotly.colors as pc
import re

# Load CSV
csv_path = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_results/wasserstein_FinEntity_DeepSeek-Qwen-7B_baseline_vs_lora-final_fullfinetune.csv"
df = pd.read_csv(csv_path)

# Filter for the two methods
df = df[df["Type"].str.contains("Baseline vs Full Finetuned|Baseline vs LoRA-final")]

# Extract fields
df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", x).group(1)))
df["HeadType"] = df["File"].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))
df["Method"] = df["Type"].str.replace("Baseline vs ", "").str.strip()

# Log counts
print("=== Number of lines per head type ===")
for head in ["q", "k", "v"]:
    count = df[df["HeadType"] == head]["File"].nunique()
    print(f"{head.upper()}: {count} unique lines")

print("\n=== Number of epochs per method (from Epoch column) ===")
if "Epoch" in df.columns:
    for method in df["Method"].unique():
        epoch_count = df[df["Method"] == method]["Epoch"].nunique()
        print(f"{method}: {epoch_count} epochs")
else:
    print("⚠️ Epoch column not found in the CSV.")

# Add a line label column
df["LineLabel"] = df.apply(lambda row: f"{row['Method']} - Epoch {row['Epoch']}", axis=1)

# Output folder
output_dir = "wasserstein_plotly"
os.makedirs(output_dir, exist_ok=True)

# Get 12 unique colors
unique_labels = df["LineLabel"].unique()
color_palette = pc.qualitative.Dark24
while len(color_palette) < len(unique_labels):
    color_palette += color_palette  # repeat if needed
label_to_color = dict(zip(unique_labels, color_palette[:len(unique_labels)]))

# Plot per head and metric
for head in ["q", "k", "v"]:
    subset = df[df["HeadType"] == head]

    for metric in ["Wasserstein H0", "Wasserstein H1"]:
        fig = px.line(
            subset,
            x="Layer",
            y=metric,
            color="LineLabel",
            line_dash="Method",  # ✅ dashed for LoRA-final
            title=f"Baseline vs Finetuned - {head.upper()} - {metric}",
            color_discrete_map=label_to_color,
            markers=True
        )

        fig.update_layout(
            xaxis_title="Layer",
            yaxis_title="Wasserstein Distance",
            template="plotly_white",
            legend_title="Run",
            font=dict(size=12)
        )

        filename = f"{head}_{metric.replace(' ', '_')}.html"
        fig.write_html(os.path.join(output_dir, filename))

print(f"✅ Interactive Plotly plots saved in: {output_dir}/")