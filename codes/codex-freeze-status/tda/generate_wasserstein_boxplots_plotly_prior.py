#!/usr/bin/env python3
"""
Generate boxplots with line plots overlaid for Wasserstein distances using Plotly.

Requested behavior:
1) Use ONLY 3 datasets: IMDB, SST2, MMLU (no GSM8K or others).
2) Generate ONLY the Combined version (Full + LoRA together). No separate Full/LoRA plots.
3) Use REGULAR (linear) y-scale (no log scale).
4) Highlight ONLY the lowest 3 layers (per subplot: per {q/k/v} x {H0/H1}) in RED. No other highlighting tiers.

Output:
- 1 HTML per model (4 models), each with 6 subplots (2 rows x 3 cols):
  Top row:  Q-H0 | K-H0 | V-H0
  Bottom:   Q-H1 | K-H1 | V-H1
"""

import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Configuration (EDIT IF NEEDED)
# ----------------------------
RESULTS_DIR = "wasserstein_results"
OUTPUT_DIR = "wasserstein_boxplots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["imdb", "sst2", "mmlu"]

MODELS = {
    "llama31_8b": "llama31_8b",
    "llama32_3b": "llama32_3b",
    "mistral7b_v03": "mistral7b_v03",
    "qwen_8b_base": "qwen_8b_base",
}

MATRIX_TYPES = ["q", "k", "v"]
WASSERSTEIN_TYPES = ["H0", "H1"]
EPOCHS = list(range(7))  # 0..6

TYPE_FULL = "Baseline vs Full Finetuned"
TYPE_LORA = "Baseline vs LoRA-final"
COMBINED_TYPES = {TYPE_FULL, TYPE_LORA}

# Dataset styling (lines)
DATASET_COLORS = {
    "imdb": "#1f77b4",  # blue
    "sst2": "#2ca02c",  # green
    "mmlu": "#d62728",  # red
}
DATASET_MARKERS = {
    "imdb": "circle",
    "sst2": "square",
    "mmlu": "triangle-up",
}


# ----------------------------
# Helpers
# ----------------------------
def extract_layer_and_matrix(file_value: str):
    """
    Extract layer number and matrix type from a string like 'layer12_q.pkl'
    Returns (layer:int, matrix:str) or (None, None) if not matched.
    """
    if not isinstance(file_value, str):
        return None, None
    m = re.search(r"layer(\d+)_([qkv])\.pkl$", file_value)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def load_and_process_data(dataset, model_pattern):
    csv_path = f"{RESULTS_DIR}/wasserstein_{dataset}_{model_pattern}.csv"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # Ensure Layer exists; if not, extract it
    if "Layer" not in df.columns:
        df["Layer"] = df["File"].apply(lambda x: extract_layer_and_matrix(x)[0])

    # Ensure MatrixType exists; if not, extract it
    if "MatrixType" not in df.columns:
        df["MatrixType"] = df["File"].apply(lambda x: extract_layer_and_matrix(x)[1])

    # Filter out invalid rows
    df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")
    df = df.dropna(subset=["Layer", "MatrixType"]).copy()
    df["Layer"] = df["Layer"].astype(int)
    df["MatrixType"] = df["MatrixType"].astype(str)

    df["Dataset"] = dataset
    return df


def process_all_data():
    """
    Load and combine all CSVs for DATASETS x MODELS.
    """
    all_parts = []
    for dataset in DATASETS:
        for model_key, model_pattern in MODELS.items():
            part = load_and_process_data(dataset, model_pattern)
            if part is None or part.empty:
                continue
            part["Model"] = model_key
            all_parts.append(part)

    if not all_parts:
        raise ValueError("No data files found. Check RESULTS_DIR and filenames.")

    return pd.concat(all_parts, ignore_index=True)


def calculate_combined_averages(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combined = Full + LoRA.
    Compute average Wasserstein per (Layer, MatrixType, WassersteinType, Dataset),
    averaging across epochs 0..6 (EPOCHS).
    """
    if "Type" not in model_df.columns or "Epoch" not in model_df.columns:
        raise ValueError("Expected 'Type' and 'Epoch' columns in CSVs.")

    df = model_df[model_df["Type"].isin(COMBINED_TYPES)].copy()
    df = df[df["Epoch"].isin(EPOCHS)].copy()

    results = []
    for wtype in WASSERSTEIN_TYPES:
        col = f"Wasserstein {wtype}"
        if col not in df.columns:
            print(f"Warning: column '{col}' not found for this model; skipping {wtype}")
            continue

        for mtype in MATRIX_TYPES:
            for dataset in DATASETS:
                sub = df[(df["MatrixType"] == mtype) & (df["Dataset"] == dataset)]
                if sub.empty:
                    continue

                # group by layer: mean across epochs (and any repeats)
                g = sub.groupby("Layer")[col].mean().reset_index()
                for _, row in g.iterrows():
                    results.append(
                        {
                            "Layer": int(row["Layer"]),
                            "MatrixType": mtype,
                            "WassersteinType": wtype,
                            "Dataset": dataset,
                            "AvgDistance": float(row[col]),
                        }
                    )

    return pd.DataFrame(results)


def lowest_k_layers(combo_data: pd.DataFrame, k: int = 3):
    """
    For one subplot (fixed MatrixType + WassersteinType):
    pick the k layers with smallest mean AvgDistance across datasets.
    """
    if combo_data.empty:
        return set()
    layer_mean = combo_data.groupby("Layer")["AvgDistance"].mean().sort_values()
    return set(layer_mean.head(k).index.tolist())


def create_boxplot_with_lines_plotly(model_key: str, avg_data: pd.DataFrame):
    """
    Create 2x3 plot:
      Top row: Q-H0 | K-H0 | V-H0
      Bottom : Q-H1 | K-H1 | V-H1
    Highlight lowest 3 layers (per subplot) in red.
    Linear y-scale.
    """
    subplot_titles = []
    for wtype in WASSERSTEIN_TYPES:
        for mtype in MATRIX_TYPES:
            subplot_titles.append(f"{mtype.upper()}-{wtype}")

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=subplot_titles,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    plot_idx = 0
    for wtype in WASSERSTEIN_TYPES:
        for mtype in MATRIX_TYPES:
            row = 1 if wtype == "H0" else 2
            col = MATRIX_TYPES.index(mtype) + 1

            combo = avg_data[
                (avg_data["MatrixType"] == mtype) & (avg_data["WassersteinType"] == wtype)
            ].copy()

            if combo.empty:
                plot_idx += 1
                continue

            layers = sorted(combo["Layer"].unique())
            low3 = lowest_k_layers(combo, k=3)

            # Build two box groups: normal vs red(low3)
            box_x_normal, box_y_normal = [], []
            box_x_red, box_y_red = [], []

            for layer in layers:
                vals = combo[combo["Layer"] == layer]["AvgDistance"].values
                if len(vals) == 0:
                    continue
                if layer in low3:
                    box_x_red.extend([layer] * len(vals))
                    box_y_red.extend(vals)
                else:
                    box_x_normal.extend([layer] * len(vals))
                    box_y_normal.extend(vals)

            # Normal distribution
            if box_x_normal:
                fig.add_trace(
                    go.Box(
                        x=box_x_normal,
                        y=box_y_normal,
                        name="Distribution",
                        boxmean="sd",
                        showlegend=False,
                        marker_color="lightblue",
                        line=dict(color="rgba(0,0,0,0.5)", width=1),
                        fillcolor="rgba(173,216,230,0.30)",
                    ),
                    row=row,
                    col=col,
                )

            # Lowest 3 layers (red)
            if box_x_red:
                fig.add_trace(
                    go.Box(
                        x=box_x_red,
                        y=box_y_red,
                        name="Lowest 3 layers",
                        boxmean="sd",
                        showlegend=(plot_idx == 0),
                        marker_color="red",
                        line=dict(color="rgba(255,0,0,0.9)", width=2),
                        fillcolor="rgba(255,0,0,0.35)",
                    ),
                    row=row,
                    col=col,
                )

            # Overlay per-dataset line plots
            for dataset in DATASETS:
                ds = combo[combo["Dataset"] == dataset].sort_values("Layer")
                if ds.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=ds["Layer"],
                        y=ds["AvgDistance"],
                        mode="lines+markers",
                        name=dataset.upper(),
                        line=dict(color=DATASET_COLORS[dataset], width=2),
                        marker=dict(
                            symbol=DATASET_MARKERS[dataset],
                            size=6,
                            color=DATASET_COLORS[dataset],
                        ),
                        legendgroup=dataset,
                        showlegend=(plot_idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            # Axes (linear y-scale; NO normalization; NO log)
            fig.update_xaxes(
                title_text="Layer",
                range=[min(layers) - 1, max(layers) + 1],
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text=f"Wasserstein {wtype}",
                row=row,
                col=col,
            )

            plot_idx += 1

    fig.update_layout(
        height=900,
        width=1800,
        template="plotly_white",
        title=f"{model_key} - Combined (Full + LoRA)",
        legend_title="Dataset",
        font=dict(size=13),
        margin=dict(l=60, r=40, t=100, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    outpath = os.path.join(OUTPUT_DIR, f"{model_key}_Combined_boxplots.html")
    fig.write_html(outpath)
    print(f"Saved: {outpath}")

def lowest_k_layers_per_dataset(combo_data: pd.DataFrame, k: int = 3):
    """
    combo_data is filtered to one (MatrixType, WassersteinType).
    Returns dict: dataset -> [lowest k layers] based on AvgDistance within that dataset.
    """
    out = {}
    for ds in DATASETS:
        ds_df = combo_data[combo_data["Dataset"] == ds]
        if ds_df.empty:
            out[ds] = []
            continue
        layer_mean = ds_df.groupby("Layer")["AvgDistance"].mean().sort_values()
        out[ds] = layer_mean.head(k).index.astype(int).tolist()
    return out


def lowest_k_layers_overall(combo_data: pd.DataFrame, k: int = 3):
    """
    Lowest k layers based on mean across datasets (this matches your red highlight logic).
    """
    if combo_data.empty:
        return []
    layer_mean = combo_data.groupby("Layer")["AvgDistance"].mean().sort_values()
    return layer_mean.head(k).index.astype(int).tolist()


def write_analysis_for_model(analysis_f, model_key: str, avg_data: pd.DataFrame, k: int = 3):
    """
    Writes analysis for one model (Combined only) into analysis_f.
    """
    analysis_f.write(f"\n{'='*70}\n")
    analysis_f.write(f"Model: {model_key}\n")
    analysis_f.write(f"{'='*70}\n\n")

    analysis_f.write("Configuration: Combined\n")
    analysis_f.write("-" * 70 + "\n\n")
    analysis_f.write(f"  Analysis for {model_key} - Combined:\n\n")

    for wtype in WASSERSTEIN_TYPES:
        for mtype in MATRIX_TYPES:
            combo = avg_data[
                (avg_data["MatrixType"] == mtype) &
                (avg_data["WassersteinType"] == wtype)
            ].copy()

            analysis_f.write(f"    {mtype.upper()}-{wtype}:\n")

            if combo.empty:
                analysis_f.write("      (no data)\n\n")
                continue

            per_ds = lowest_k_layers_per_dataset(combo, k=k)
            overall = lowest_k_layers_overall(combo, k=k)

            analysis_f.write(f"      Lowest {k} layers per dataset:\n")
            for ds in DATASETS:
                analysis_f.write(f"        {ds.upper()}: {per_ds.get(ds, [])}\n")

            analysis_f.write(f"      Lowest {k} layers overall (mean across datasets): {overall}\n\n")


# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading and processing data...")
    df = process_all_data()

    print(f"Loaded {len(df)} rows")
    print(f"Models found: {sorted(df['Model'].unique())}")
    print(f"Datasets found: {sorted(df['Dataset'].unique())}")
    print(f"Layers found: {sorted(df['Layer'].unique())}")

    for model_key in MODELS.keys():
        model_df = df[df["Model"] == model_key]
        if model_df.empty:
            print(f"Warning: No data for model {model_key}")
            continue

        print(f"\nProcessing {model_key} (Combined only)...")
        avg_data = calculate_combined_averages(model_df)
        if avg_data.empty:
            print(f"  No combined data after filtering Types={COMBINED_TYPES} and Epochs={EPOCHS}")
            continue

        create_boxplot_with_lines_plotly(model_key, avg_data)

    print(f"\nDone. HTML plots saved in: {OUTPUT_DIR}/")

    analysis_path = os.path.join(OUTPUT_DIR, "layer_analysis.txt")
    with open(analysis_path, "w") as analysis_f:
        analysis_f.write("Analysis of Lowest Changing Layers Across Datasets\n")
        analysis_f.write("=" * 70 + "\n\n")

        for model_key in MODELS.keys():
            model_df = df[df["Model"] == model_key]
            if model_df.empty:
                print(f"Warning: No data for model {model_key}")
                continue

            avg_data = calculate_combined_averages(model_df)
            if avg_data.empty:
                print(f"  No combined data for {model_key}")
                continue

            # write analysis
            write_analysis_for_model(analysis_f, model_key, avg_data, k=3)

            # generate plots
            create_boxplot_with_lines_plotly(model_key, avg_data)

    print(f"Analysis saved to: {analysis_path}")



if __name__ == "__main__":
    main()