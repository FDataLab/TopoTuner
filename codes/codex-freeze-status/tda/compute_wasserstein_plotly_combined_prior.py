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
            "Create combined Plotly subplots for every model/head.\n"
            "Layout: 2 rows (H0 top, H1 bottom) x 3 cols (MMLU, IMDB, SST2)."
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
        help="Output directory for the Plotly HTML files.",
    )
    parser.add_argument(
        "--datasets",
        default="mmlu,imdb,sst2",
        help="Comma-separated dataset names to include (case-insensitive).",
    )
    parser.add_argument(
        "--models",
        default="llama3.1-8b,llama3.2-3b,mistral-7b.v0.3,qwen-8b-base",
        help="Comma-separated models to include (lowercase).",
    )
    parser.add_argument(
        "--heads",
        default="k,q,v",
        help="Comma-separated head types to plot.",
    )
    parser.add_argument(
        "--metrics",
        default="Wasserstein H0,Wasserstein H1",
        help="Comma-separated metric column names to plot. Expect exactly 2: H0 and H1.",
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
        df = df[df["Type"].astype(str).str.contains(filter_regex, regex=True, na=False)]

    if "Epoch" in df.columns:
        df = df[df["Epoch"] != 0]

    if "Layer" not in df.columns:
        # expects something like "...layer12..."
        df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", str(x)).group(1)))

    if "HeadType" not in df.columns:
        # expects "..._k.pkl" / "..._q.pkl" / "..._v.pkl"
        df["HeadType"] = df["File"].apply(lambda x: str(x).split("_")[-1].replace(".pkl", ""))

    if "Method" not in df.columns and "Type" in df.columns:
        df["Method"] = df["Type"].str.replace("Baseline vs ", "", regex=False).str.strip()

    if "LineLabel" not in df.columns:
        if "Epoch" in df.columns and "Method" in df.columns:
            df["LineLabel"] = df.apply(lambda row: f"{row['Method']} - Epoch {row['Epoch']}", axis=1)
        elif "Method" in df.columns:
            df["LineLabel"] = df["Method"]
        else:
            df["LineLabel"] = "Run"

    df = (
        df.sort_values(by=["HeadType", "LineLabel", "Layer"])
          .drop_duplicates(subset=["HeadType", "LineLabel", "Layer"], keep="first")
    )

    if heads:
        df = df[df["HeadType"].isin(heads)]

    return df

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

        method = line_df["Method"].iloc[0] if "Method" in line_df.columns else ""
        dash_style = "dot" if ("lora" in str(method).lower()) else "solid"

        fig.add_trace(
            go.Scatter(
                x=line_df["Layer"],
                y=line_df[metric],
                mode="lines+markers",
                name=label,
                line=dict(color=colors.get(label), width=2, dash=dash_style),
                marker=dict(size=6),
                legendgroup=label,
                showlegend=(label not in seen_labels),
            ),
            row=row,
            col=col,
        )
        seen_labels.add(label)


def main():
    args = parse_args()

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    # enforce ordering for columns
    desired_order = ["mmlu", "imdb", "sst2"]
    datasets = [d for d in desired_order if d in datasets]

    dataset_labels = OrderedDict((d, d.upper()) for d in datasets)

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    models_set = set(models)

    heads = [h.strip().lower() for h in args.heads.split(",") if h.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if len(metrics) != 2:
        raise SystemExit("Expected exactly 2 metrics: 'Wasserstein H0' and 'Wasserstein H1'.")

    metric_h0, metric_h1 = metrics[0], metrics[1]

    discovered = discover_wasserstein_files(args.results_dir, set(datasets), models_set)
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

    head_color_maps = build_label_colors(processed_dfs, heads)

    os.makedirs(args.out, exist_ok=True)

    # positions: cols = datasets; rows = H0/H1
    col_map = {d: i + 1 for i, d in enumerate(datasets)}  # mmlu->1, imdb->2, sst2->3

    for model in models:
        if model not in processed:
            # if you want, you can print a warning here
            continue

        for head in heads:
            if head not in head_color_maps or not head_color_maps[head]:
                continue
            colors = head_color_maps[head]

            # ---- one figure per (model, head), with 6 subplots ----
            title = f"{model} · {head.upper()} · H0(top) / H1(bottom)"

            subplot_titles = []
            for row_name in ["H0", "H1"]:
                for d in datasets:
                    subplot_titles.append(f"{dataset_labels[d]} {row_name}")

            fig = make_subplots(
                rows=2,
                cols=3,
                subplot_titles=subplot_titles,
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                ],
                horizontal_spacing=0.07,
                vertical_spacing=0.10,
            )

            seen_labels = set()

            for d in datasets:
                dataset_df = processed.get(model, {}).get(d)
                if dataset_df is None:
                    # still set axes for empty panels
                    c = col_map[d]
                    # top H0
                    fig.update_xaxes(title_text="Layer", row=1, col=c)
                    fig.update_yaxes(title_text="Wasserstein Distance", row=1, col=c)
                    # bottom H1
                    fig.update_xaxes(title_text="Layer", row=2, col=c)
                    fig.update_yaxes(title_text="Wasserstein Distance", row=2, col=c)
                    continue

                head_subset = dataset_df[dataset_df["HeadType"] == head]
                if head_subset.empty:
                    continue

                c = col_map[d]

                # Row 1 = H0
                if metric_h0 in head_subset.columns:
                    add_dataset_traces(fig, head_subset, metric_h0, row=1, col=c, colors=colors, seen_labels=seen_labels)
                fig.update_xaxes(title_text="Layer", row=1, col=c)
                fig.update_yaxes(title_text="Wasserstein Distance", row=1, col=c)

                # Row 2 = H1
                if metric_h1 in head_subset.columns:
                    add_dataset_traces(fig, head_subset, metric_h1, row=2, col=c, colors=colors, seen_labels=seen_labels)
                fig.update_xaxes(title_text="Layer", row=2, col=c)
                fig.update_yaxes(title_text="Wasserstein Distance", row=2, col=c)

            fig.update_layout(
                height=900,
                width=1700,
                template="plotly_white",
                title=title,
                legend_title="Run",
                font=dict(size=13),
                margin=dict(l=60, r=40, t=90, b=60),
            )

            output_model_dir = os.path.join(args.out, model)
            os.makedirs(output_model_dir, exist_ok=True)

            filename = f"{model}_{head}_H0H1_6subplots.html"
            outpath = os.path.join(output_model_dir, filename)
            fig.write_html(outpath)
            print(f"[save] {outpath}")

    print("✅ 12 figures generated (4 models × 3 heads), each with 6 subplots.")


if __name__ == "__main__":
    main()

