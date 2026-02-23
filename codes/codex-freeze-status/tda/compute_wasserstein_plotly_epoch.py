import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# === CONFIGURATION ===
csv_path = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_results/wasserstein_FinEntity_DeepSeek-Qwen-7B.csv"
output_dir = "/staging/users/aerol1/tda/Topo-Tuner/wasserstein_plotly/finentity_qwen/layer_epochs"
os.makedirs(output_dir, exist_ok=True)

# Specify multiple layers to plot
target_layers = [0, 1, 14, 19, 21, 26]  # ⬅️ Edit this list with any layers you want

# === LOAD CSV ===
df = pd.read_csv(csv_path)
df = df[df["Type"].str.contains("Baseline vs Full Finetuned|Baseline vs LoRA-final")]

# === Extract Fields ===
df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", x).group(1)))
df["HeadType"] = df["File"].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))
df["Method"] = df["Type"].str.replace("Baseline vs ", "").str.strip()

# === Plot Each Layer in a Grid ===
for layer in target_layers:
    df_layer = df[df["Layer"] == layer]
    if df_layer.empty:
        print(f"⚠️ No data for layer {layer}")
        continue

    # Create subplot figure
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "K - H0", "Q - H0", "V - H0",
            "K - H1", "Q - H1", "V - H1"
        ]
    )

    head_to_col = {"k": 1, "q": 2, "v": 3}
    metric_to_row = {"Wasserstein H0": 1, "Wasserstein H1": 2}

    for head in ["k", "q", "v"]:
        df_head = df_layer[df_layer["HeadType"] == head]
        for metric in ["Wasserstein H0", "Wasserstein H1"]:
            row = metric_to_row[metric]
            col = head_to_col[head]

            for method in df_head["Method"].unique():
                df_line = df_head[df_head["Method"] == method]
                if df_line.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=df_line["Epoch"],
                        y=df_line[metric],
                        mode="lines+markers",
                        name=f"{method} ({head.upper()} - {metric[-2:]})",
                        legendgroup=method,
                        showlegend=(row == 1 and col == 1)
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        height=700,
        width=1200,
        title_text=f"Wasserstein Distances at Layer {layer}",
        template="plotly_white",
        legend_title="Method",
        font=dict(size=12)
    )

    filename = f"layer{layer}_grid.html"
    fig.write_html(os.path.join(output_dir, filename))
    print(f"✅ Saved: {filename}")

print(f"\n✅ All grid plots saved to: {output_dir}/")
