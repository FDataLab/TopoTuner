"""
Plot Wasserstein H0 vs layer for GSM8K full finetuning TDA results.
2x2 subplots: K, Q, V, O projections. One line per epoch (epoch_1–6).
"""

import argparse
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def extract_layer(file_name: str) -> int:
    m = re.search(r"layer(\d+)", file_name)
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(description="Plot Wasserstein H0 for full finetuning")
    parser.add_argument(
        "--csv",
        default="gsm8k-tda-results/wasserstein_results.csv",
        help="Path to wasserstein_results.csv",
    )
    parser.add_argument(
        "--out",
        default="gsm8k-tda-results/wasserstein_full_finetuning_plotly.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["Layer"] = df["File"].apply(extract_layer)

    projections = ["k", "q", "v", "o"]
    epochs = ["epoch_1", "epoch_2", "epoch_3", "epoch_4", "epoch_5", "epoch_6"]
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Projection {p.upper()}" for p in projections],
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )

    for idx, proj in enumerate(projections):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        subset = df[df["Projection"] == proj].copy()
        subset = subset.sort_values(["Epoch", "Layer"])

        for e_idx, epoch in enumerate(epochs):
            epoch_data = subset[subset["Epoch"] == epoch]
            color = colors[e_idx % len(colors)]
            label = epoch.replace("epoch_", "Epoch ")

            fig.add_trace(
                go.Scatter(
                    x=epoch_data["Layer"],
                    y=epoch_data["Wasserstein H0"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=5),
                    showlegend=(idx == 0),
                ),
                row=row, col=col,
            )

    fig.update_layout(
        title=dict(text="Wasserstein H0 vs Layer (Baseline vs Full Finetuning)", font=dict(size=18)),
        template="plotly_white",
        width=1100,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=2)
    fig.update_yaxes(title_text="Wasserstein H0", row=1, col=1)
    fig.update_yaxes(title_text="Wasserstein H0", row=1, col=2)
    fig.update_yaxes(title_text="Wasserstein H0", row=2, col=1)
    fig.update_yaxes(title_text="Wasserstein H0", row=2, col=2)

    fig.write_html(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
