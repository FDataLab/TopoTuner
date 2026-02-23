import argparse
import glob
import os
import re
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create combined 2x2 Plotly subplots (IMDB, MMLU, SST2 + empty slot) "
            "for every model/head using Wasserstein CSVs."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="wasserstein_results",
        help="Directory containing wasserstein_{dataset}_{model}.csv files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for the combined Plotly HTML files.",
    )
    parser.add_argument(
        "--datasets",
        default="mmlu,imdb,sst2,gsm8k",
        help="Comma-separated dataset names to include (case-insensitive).",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated models to include (lowercase, e.g., qwen_8b). "
             "Defaults to discovering every model present.",
    )
    parser.add_argument(
        "--heads",
        default="q,k,v",
        help="Comma-separated head types to plot.",
    )
    parser.add_argument(
        "--metrics",
        default="Wasserstein H0,Wasserstein H1",
        help="Comma-separated metric column names to plot.",
    )
    parser.add_argument(
        "--filter",
        default="Baseline vs Full Finetuned|Baseline vs LoRA-final",
        help="Regex filter applied to the Type column.",
    )
    return parser.parse_args()


def discover_wasserstein_files(results_dir, datasets, models):
    pattern = os.path.join(results_dir, "wasserstein_*.csv")
    discovered = []

    for path in glob.glob(pattern):
        name = os.path.basename(path)
        if any(tag in name for tag in ("_OLD", "_old", "_OLD2")):
            continue
        if name.endswith(".sorted.csv") or "timeout" in name.lower():
            continue

        parts = name.replace(".csv", "").split("_")
        if len(parts) < 3 or parts[0] != "wasserstein":
            continue

        dataset = parts[1].lower()
        model = "_".join(parts[2:]).lower()

        if datasets and dataset not in datasets:
            continue
        if models and model not in models:
            continue

        discovered.append((path, dataset, model))

    return discovered


def prepare_dataframe(path, heads, filter_regex):
    df = pd.read_csv(path)
    if "Type" in df.columns:
        df = df[df["Type"].str.contains(filter_regex, regex=True)]

    if "Epoch" in df.columns:
        df = df[df["Epoch"] != 0]

    if "Layer" not in df.columns:
        df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", x).group(1)))

    if "HeadType" not in df.columns:
        df["HeadType"] = df["File"].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))

    if "Method" not in df.columns:
        df["Method"] = df["Type"].str.replace("Baseline vs ", "", regex=False).str.strip()

    if "LineLabel" not in df.columns:
        if "Epoch" in df.columns:
            df["LineLabel"] = df.apply(lambda row: f"{row['Method']} - Epoch {row['Epoch']}", axis=1)
        else:
            df["LineLabel"] = df["Method"]

    df = (
        df.sort_values(by=["HeadType", "LineLabel", "Layer"])
          .drop_duplicates(subset=["HeadType", "LineLabel", "Layer"], keep="first")
    )

    if heads:
        df = df[df["HeadType"].isin(heads)]

    return df


def compute_axis_ranges(processed, heads, metrics):
    """
    Compute axis ranges for all models together using 99th percentile.
    Minimum is always set to 0 (Wasserstein distances are non-negative).
    """
    layer_min = float("inf")
    layer_max = float("-inf")
    
    # Collect all values from all models together per head/metric
    metric_values = {head: {metric: [] for metric in metrics} for head in heads}

    for model, datasets in processed.items():
        for dataset, df in datasets.items():
            if df.empty:
                continue
            layer_min = min(layer_min, df["Layer"].min())
            layer_max = max(layer_max, df["Layer"].max())

            for head in heads:
                head_df = df[df["HeadType"] == head]
                if head_df.empty:
                    continue
                for metric in metrics:
                    if metric not in head_df.columns:
                        continue
                    # Collect all values (excluding NaN)
                    values = head_df[metric].dropna().values
                    if len(values) > 0:
                        metric_values[head][metric].extend(values)

    # Compute ranges using 99th percentile for all models together
    metric_ranges = {}
    for head in heads:
        metric_ranges[head] = {}
        for metric in metrics:
            values = metric_values[head][metric]
            if len(values) == 0:
                metric_ranges[head][metric] = (0.0, 1.0)
            else:
                # Use 0 as minimum, 99th percentile for maximum
                high = np.percentile(values, 99)
                padding = 0.02 * high if high > 0 else 0.05
                metric_ranges[head][metric] = (0.0, float(high + padding))

    if layer_min == float("inf"):
        layer_min, layer_max = 0, 1

    # Return the same ranges for all models
    model_ranges = {}
    for model in processed.keys():
        model_ranges[model] = metric_ranges

    return (layer_min, layer_max), model_ranges


def build_label_colors(processed_dfs, heads):
    labels_by_head = defaultdict(set)
    for df in processed_dfs:
        for head in heads:
            head_labels = df[df["HeadType"] == head]["LineLabel"].unique()
            labels_by_head[head].update(head_labels)

    head_color_maps = {}
    for head, labels in labels_by_head.items():
        labels = sorted(labels)
        palette = list(pc.qualitative.Dark24)
        while len(palette) < len(labels):
            palette += palette
        head_color_maps[head] = dict(zip(labels, palette[: len(labels)]))

    return head_color_maps


def add_dataset_traces(fig, subset, metric, row, col, colors, seen_labels):
    sorted_subset = (
        subset.sort_values(by=["LineLabel", "Layer"])
              .drop_duplicates(subset=["LineLabel", "Layer"], keep="first")
    )

    for label in sorted_subset["LineLabel"].unique():
        line_df = sorted_subset[sorted_subset["LineLabel"] == label]
        # Determine dash style based on Method: LoRA-final -> dot, Full Finetuned -> solid
        method = line_df["Method"].iloc[0] if "Method" in line_df.columns else None
        dash_style = "dot" if method and "LoRA" in method else "solid"
        
        fig.add_trace(
            go.Scatter(
                x=line_df["Layer"],
                y=line_df[metric],
                mode="lines+markers",
                name=label,
                line=dict(color=colors.get(label), width=2, dash=dash_style),
                marker=dict(size=6),
                legendgroup=label,
                showlegend=label not in seen_labels,
            ),
            row=row,
            col=col,
        )
        seen_labels.add(label)


def main():
    args = parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    dataset_labels = OrderedDict((d, d.upper()) for d in datasets)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    discovered = discover_wasserstein_files(args.results_dir, set(datasets), set(models) if models else set())
    if not discovered:
        raise SystemExit("No matching wasserstein CSV files found with the provided filters.")

    processed = {}
    processed_dfs = []
    for path, dataset, model in discovered:
        df = prepare_dataframe(path, heads, args.filter)
        if df.empty:
            continue
        processed.setdefault(model, {})[dataset] = df
        processed_dfs.append(df)

    if not processed:
        raise SystemExit("Filtered data is empty after applying heads/filter selections.")

    layer_range, model_ranges = compute_axis_ranges(processed, heads, metrics)
    head_color_maps = build_label_colors(processed_dfs, heads)

    os.makedirs(args.out, exist_ok=True)

    dataset_positions = OrderedDict()
    # Map datasets to positions: mmlu->(1,1), imdb->(1,2), sst2->(2,1), gsm8k->(2,2)
    position_map = {
        "mmlu": (1, 1),
        "imdb": (1, 2),
        "sst2": (2, 1),
        "gsm8k": (2, 2),
    }
    for dataset_key in dataset_labels.keys():
        if dataset_key in position_map:
            dataset_positions[dataset_key] = position_map[dataset_key]

    for model in sorted(processed.keys()):
        for head in heads:
            if head not in head_color_maps or not head_color_maps[head]:
                continue
            colors = head_color_maps[head]

            for metric in metrics:
                title = f"{model} · {head.upper()} · {metric}"
                # Generate subplot titles in position order: (1,1), (1,2), (2,1), (2,2)
                subplot_titles = []
                for pos in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                    # Find dataset at this position
                    dataset_at_pos = None
                    for dkey, dpos in dataset_positions.items():
                        if dpos == pos:
                            dataset_at_pos = dkey
                            break
                    if dataset_at_pos:
                        subplot_titles.append(dataset_labels[dataset_at_pos])
                    else:
                        subplot_titles.append("")

                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=subplot_titles,
                    specs=[[{"type": "scatter"}, {"type": "scatter"}],
                           [{"type": "scatter"}, {"type": "scatter"}]],
                )

                seen_labels = set()
                for dataset_key, (row, col) in dataset_positions.items():
                    dataset_df = processed.get(model, {}).get(dataset_key)
                    if dataset_df is None:
                        # If dataset is missing (e.g., qwen_8b for GSM8K), skip but still set up axes
                        if row == 2 and col == 2:
                            # Bottom-right: set up axes but leave empty
                            fig.update_xaxes(title_text="Layer", range=layer_range, row=row, col=col, showticklabels=False)
                            fig.update_yaxes(
                                title_text="Wasserstein Distance",
                                range=model_ranges[model][head][metric],
                                row=row,
                                col=col,
                                showticklabels=False,
                            )
                        continue
                    head_subset = dataset_df[dataset_df["HeadType"] == head]
                    if head_subset.empty or metric not in head_subset.columns:
                        continue
                    add_dataset_traces(fig, head_subset, metric, row, col, colors, seen_labels)
                    fig.update_xaxes(title_text="Layer", range=layer_range, row=row, col=col)
                    fig.update_yaxes(
                        title_text="Wasserstein Distance",
                        range=model_ranges[model][head][metric],
                        row=row,
                        col=col,
                    )

                fig.update_layout(
                    height=900,
                    width=1400,
                    template="plotly_white",
                    title=title,
                    legend_title="Run",
                    font=dict(size=13),
                    margin=dict(l=60, r=40, t=80, b=60),
                )

                output_model_dir = os.path.join(args.out, model)
                os.makedirs(output_model_dir, exist_ok=True)
                filename = f"{model}_{head}_{metric.replace(' ', '_')}_combined.html"
                fig.write_html(os.path.join(output_model_dir, filename))
                print(f"[save] {os.path.join(output_model_dir, filename)}")

    print("✅ Combined Plotly figures generated successfully.")


if __name__ == "__main__":
    main()

