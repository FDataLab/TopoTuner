import os
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import re

# === CONFIGURATION ===
csv_path = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_results/wasserstein_GSM8K_DeepSeek-Llama-8B.csv"
output_dir = "./wasserstein_plotly/gsm8k_llama/final_headwise_layerwise"
os.makedirs(output_dir, exist_ok=True)

# === LOAD & PROCESS DATA ===
df = pd.read_csv(csv_path)
df = df[df["Type"].str.contains("Baseline vs Full Finetuned|Baseline vs LoRA-final")]

# Extract fields
df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", x).group(1)))
df["HeadType"] = df["File"].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))
df["Method"] = df["Type"].str.replace("Baseline vs ", "").str.strip()

# Compute top 3 layers for each head based on max H1
top_layers_per_head = {}
for head in ["q", "k", "v"]:
    subset = df[df["HeadType"] == head]
    max_h1_per_layer = subset.groupby("Layer")["Wasserstein H1"].max()
    top_layers = max_h1_per_layer.sort_values(ascending=False).head(3).index.tolist()
    top_layers_per_head[head] = top_layers

# Plot for each head type
for head in ["q", "k", "v"]:
    subset = df[(df["HeadType"] == head) & (df["Layer"].isin(top_layers_per_head[head]))]
    metrics = ["Wasserstein H0", "Wasserstein H1"]

    fig = sp.make_subplots(
        rows=3, cols=2,
        subplot_titles=[f"Layer {layer} - {metric}" for layer in top_layers_per_head[head] for metric in metrics],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    color_map = {"Full Finetuned": "blue", "LoRA-final": "red"}

    for i, layer in enumerate(top_layers_per_head[head]):
        for j, metric in enumerate(metrics):
            row = i + 1
            col = j + 1
            for method in ["Full Finetuned", "LoRA-final"]:
                layer_subset = subset[(subset["Layer"] == layer) & (subset["Method"] == method)]
                fig.add_trace(
                    go.Scatter(
                        x=layer_subset["Epoch"],
                        y=layer_subset[metric],
                        mode='lines+markers',
                        name=method,
                        legendgroup=method,
                        marker=dict(color=color_map.get(method, None)),
                        showlegend=(i == 0 and j == 0)
                    ),
                    row=row, col=col
                )

    fig.update_layout(
        height=1000,
        width=1000,
        title_text=f"Top Layers per HeadType: {head.upper()}",
        template="plotly_white"
    )

    fig.write_html(os.path.join(output_dir, f"{head}_top_layers.html"))

print(f"✅ Done. HTML plots saved in: {output_dir}")
