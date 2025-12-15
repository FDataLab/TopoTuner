import os
import re
import argparse
import pandas as pd
import plotly.express as px
import plotly.colors as pc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Wasserstein CSV")
    ap.add_argument("--out", required=True, help="Output directory for HTML plots")
    ap.add_argument("--heads", default="q,k,v", help="Comma-separated heads to plot (q,k,v)")
    ap.add_argument("--filter", default="Baseline vs Full Finetuned|Baseline vs LoRA-final",
                    help="Regex to filter Type column")
    ap.add_argument("--write-sorted-csv", default="", help="Optional path to write a sorted/deduped CSV")
    args = ap.parse_args()

    csv_path = args.csv
    output_dir = args.out
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]

    df = pd.read_csv(csv_path)
    df = df[df["Type"].str.contains(args.filter, regex=True)]
    # Skip epoch 0 (baseline) from plots
    if "Epoch" in df.columns:
        df = df[df["Epoch"] != 0]

    # Extract fields
    df["Layer"] = df["File"].apply(lambda x: int(re.search(r"layer(\d+)", x).group(1)))
    df["HeadType"] = df["File"].apply(lambda x: x.split("_")[-1].replace(".pkl", ""))
    df["Method"] = df["Type"].str.replace("Baseline vs ", "", regex=False).str.strip()

    # Log counts
    print("=== Number of lines per head type ===")
    for head in heads:
        count = df[df["HeadType"] == head]["File"].nunique()
        print(f"{head.upper()}: {count} unique lines")

    if "Epoch" in df.columns:
        print("\n=== Number of epochs per method (from Epoch column) ===")
        for method in df["Method"].unique():
            epoch_count = df[df["Method"] == method]["Epoch"].nunique()
            print(f"{method}: {epoch_count} epochs")

    # Add a line label column
    df["LineLabel"] = df.apply(lambda row: f"{row['Method']} - Epoch {row['Epoch']}", axis=1)

    # Sort and de-duplicate like wass_sort: ensure increasing Layer per (HeadType, LineLabel)
    df_sorted = (
        df.sort_values(by=["HeadType", "LineLabel", "Layer"]) 
          .drop_duplicates(subset=["HeadType", "LineLabel", "Layer"], keep="first")
    )

    if args.write_sorted_csv:
        os.makedirs(os.path.dirname(args.write_sorted_csv), exist_ok=True)
        df_sorted.to_csv(args.write_sorted_csv, index=False)
        print(f"[save] Sorted CSV written to: {args.write_sorted_csv}")

    # Derive dataset and model name from CSV filename
    base_csv = os.path.basename(csv_path)
    dataset_name = None
    model_name = None
    # Expected patterns: wasserstein_{dataset}_{model}.csv or sliced_wasserstein_{dataset}_{model}.csv
    parts = base_csv.replace(".csv", "").split("_")
    if len(parts) >= 3:
        # for names like wasserstein_sst2_mistral7b or sliced_wasserstein_sst2_mistral7b
        if parts[0] == "sliced":
            # sliced_wasserstein_sst2_mistral7b -> dataset=sst2, model=mistral7b
            dataset_name = parts[2] if len(parts) > 2 else "dataset"
            model_name = parts[3] if len(parts) > 3 else None
        else:
            # wasserstein_sst2_mistral7b -> dataset=sst2, model=mistral7b
            dataset_name = parts[1] if len(parts) > 1 else "dataset"
            model_name = parts[2] if len(parts) > 2 else None
    else:
        dataset_name = "dataset"

    # Create output subdirectory: output_dir/dataset_name/dataset_name_model_name/
    if model_name:
        model_dir = os.path.join(output_dir, dataset_name, f"{dataset_name}_{model_name}")
    else:
        model_dir = os.path.join(output_dir, dataset_name) if dataset_name else output_dir
    os.makedirs(model_dir, exist_ok=True)

    # Get enough unique colors
    unique_labels = df_sorted["LineLabel"].unique()
    color_palette = pc.qualitative.Dark24
    while len(color_palette) < len(unique_labels):
        color_palette += color_palette  # repeat if needed
    label_to_color = dict(zip(unique_labels, color_palette[:len(unique_labels)]))

    print("\n✅ Sorted layer orders per LineLabel:")
    for label in sorted(df_sorted["LineLabel"].unique()):
        sub_df = df_sorted[df_sorted["LineLabel"] == label].sort_values("Layer")
        print(f"{label}: {list(sub_df['Layer'])}")

    # Plot per head and metric
    for head in heads:
        subset = df_sorted[df_sorted["HeadType"] == head].copy()
        subset["LineLabel"] = subset.apply(lambda row: f"{row['Method']} - Epoch {row['Epoch']}", axis=1)

        for metric in ["Wasserstein H0", "Wasserstein H1"]:
            # ✅ Sort and remove repeated layer points for each run
            subset_sorted = subset.sort_values(by=["LineLabel", "Layer"]).drop_duplicates(subset=["LineLabel", "Layer"])

            # 🔍 Debug: Check if any duplicates existed before drop
            dupe_rows = subset.sort_values(by=["LineLabel", "Layer"]).duplicated(subset=["LineLabel", "Layer"], keep=False)
            if dupe_rows.any():
                print(f"\n🚨 Duplicates found for {head.upper()} - {metric}")
                print(subset[dupe_rows][["LineLabel", "Layer", "File"]].to_string(index=False))

            fig = px.line(
                subset_sorted,
                x="Layer",
                y=metric,
                color="LineLabel",
                line_dash="Method",
                title=f"Baseline vs Finetuned - {head.upper()} - {metric}",
                color_discrete_map=label_to_color,
                markers=True,
            )

            fig.update_layout(
                xaxis_title="Layer",
                yaxis_title="Wasserstein Distance",
                template="plotly_white",
                legend_title="Run",
                font=dict(size=12)
            )

            prefix = f"{dataset_name}_" if dataset_name else ""
            filename = f"{prefix}{head}_{metric.replace(' ', '_')}.html"
            fig.write_html(os.path.join(model_dir, filename))

    print(f"✅ Interactive Plotly plots saved in: {model_dir}/")


if __name__ == "__main__":
    main()