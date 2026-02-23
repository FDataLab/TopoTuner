import os
import argparse

import pandas as pd
import plotly.graph_objects as go


BASE_DIR = "/home/kadir/topo/evaluation_results"


def _load_csv(path):
    df = pd.read_csv(path)
    # Expect columns: checkpoint, em, f1, ...
    if "checkpoint" not in df.columns:
        raise ValueError(f"Missing 'checkpoint' column in {path}: {df.columns}")
    if "f1" not in df.columns and "em" not in df.columns:
        raise ValueError(f"Missing 'f1' or 'em' column in {path}: {df.columns}")

    def _epoch_from_ckpt(name: str) -> int:
        # checkpoint-epoch-3 -> 3
        for part in str(name).replace("_", "-").split("-"):
            if part.isdigit():
                return int(part)
        return -1

    df["epoch"] = df["checkpoint"].apply(_epoch_from_ckpt)
    df = df[df["epoch"] >= 0].sort_values("epoch")
    return df


def make_plot(dataset: str, model_label: str, csv_map: dict, out_dir: str, plot_suffix: str = "", metric: str = "f1"):
    """
    csv_map: exp_key -> (csv_path, display_name, color)
    plot_suffix: Optional suffix for filename (e.g., "k_o", "k_o_mlp")
    metric: "f1" or "em"
    """
    fig = go.Figure()

    for key, value in csv_map.items():
        csv_path, display_name, color = value
        if not os.path.exists(csv_path):
            print(f"[skip] Missing CSV for {dataset} {key}: {csv_path}")
            continue
        df = _load_csv(csv_path)
        if df.empty:
            print(f"[skip] Empty CSV for {dataset} {key}: {csv_path}")
            continue
        
        # Check if metric column exists
        if metric not in df.columns:
            print(f"[skip] Missing '{metric}' column in {csv_path}")
            continue

        # Style: full = bold, lowest = solid, highest = dashed
        if key == "full":
            line_style = dict(color=color, width=3.0)  # Bolder for full
        elif "highest" in key:
            line_style = dict(color=color, width=2.0, dash="dash")  # Dashed for highest
        else:  # lowest
            line_style = dict(color=color, width=2.0)  # Solid for lowest

        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df[metric],
                mode="lines+markers",
                name=display_name,
                line=line_style,
                marker=dict(size=7),
            )
        )

    metric_label = metric.upper()
    fig.update_layout(
        title=f"{model_label} on {dataset}",
        xaxis_title="Epoch",
        yaxis_title=metric_label,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    os.makedirs(out_dir, exist_ok=True)
    # Use plot_suffix if provided (e.g., "k_o", "k_o_mlp"), otherwise default
    if plot_suffix:
        filename = f"llama31_8b_{dataset}_{plot_suffix}_eval_{metric}.html"
    else:
        filename = f"llama31_8b_{dataset}_eval_{metric}.html"
    out_path = os.path.join(out_dir, filename)
    fig.write_html(out_path)
    print(f"[ok] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default="/home/kadir/topo/plots/llama31_8b_eval",
        help="Directory to write HTML plots",
    )
    args = ap.parse_args()

    # Colors consistent with earlier plots (semantic-ish)
    colors = {
        "full": "#000000",
        "lowest3": "#1f77b4",
        "highest3": "#ff7f0e",
        "lowest15": "#2ca02c",
        "highest15": "#d62728",
    }

    # 1) HotpotQA K+MLP (llama31_8b, full + K+MLP variants)
    hotpotqa_dir = os.path.join(BASE_DIR, "hotpotqa")
    hotpotqa_k_mlp_map = {
        "full": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_full.csv"),
            "Full",
            colors["full"],
        ),
        "lowest3": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_mlp_lowest3.csv"),
            "K+MLP Lowest 3",
            colors["lowest3"],
        ),
        "highest3": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_mlp_highest3.csv"),
            "K+MLP Highest 3",
            colors["highest3"],
        ),
        "lowest15": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_mlp_lowest15.csv"),
            "K+MLP Lowest 15",
            colors["lowest15"],
        ),
        "highest15": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_mlp_highest15.csv"),
            "K+MLP Highest 15",
            colors["highest15"],
        ),
    }
    # Generate F1 and EM plots for K+MLP
    make_plot(
        dataset="HotpotQA",
        model_label="Llama-3.1-8B",
        csv_map=hotpotqa_k_mlp_map,
        out_dir=args.out_dir,
        plot_suffix="k_mlp",
        metric="f1",
    )
    make_plot(
        dataset="HotpotQA",
        model_label="Llama-3.1-8B",
        csv_map=hotpotqa_k_mlp_map,
        out_dir=args.out_dir,
        plot_suffix="k_mlp",
        metric="em",
    )

    # 2) SQuAD K+MLP (llama31_8b, full + K+MLP variants)
    squad_dir = os.path.join(BASE_DIR, "squad")
    squad_k_mlp_map = {
        "full": (
            os.path.join(squad_dir, "squad_llama31_8b_full.csv"),
            "Full",
            colors["full"],
        ),
        "lowest3": (
            os.path.join(squad_dir, "squad_llama31_8b_k_mlp_lowest3.csv"),
            "K+MLP Lowest 3",
            colors["lowest3"],
        ),
        "highest3": (
            os.path.join(squad_dir, "squad_llama31_8b_k_mlp_highest3.csv"),
            "K+MLP Highest 3",
            colors["highest3"],
        ),
        "lowest15": (
            os.path.join(squad_dir, "squad_llama31_8b_k_mlp_lowest15.csv"),
            "K+MLP Lowest 15",
            colors["lowest15"],
        ),
        "highest15": (
            os.path.join(squad_dir, "squad_llama31_8b_k_mlp_highest15.csv"),
            "K+MLP Highest 15",
            colors["highest15"],
        ),
    }
    # Generate F1 and EM plots for SQuAD K+MLP
    make_plot(
        dataset="SQuAD",
        model_label="Llama-3.1-8B",
        csv_map=squad_k_mlp_map,
        out_dir=args.out_dir,
        plot_suffix="k_mlp",
        metric="f1",
    )
    make_plot(
        dataset="SQuAD",
        model_label="Llama-3.1-8B",
        csv_map=squad_k_mlp_map,
        out_dir=args.out_dir,
        plot_suffix="k_mlp",
        metric="em",
    )

    # 3) HotpotQA K+O (llama31_8b, full + K+O variants)
    hotpotqa_k_o_map = {
        "full": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_full.csv"),
            "Full",
            colors["full"],
        ),
        "lowest3": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_o_lowest3.csv"),
            "K+O Lowest 3",
            colors["lowest3"],
        ),
        "highest3": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_o_highest3.csv"),
            "K+O Highest 3",
            colors["highest3"],
        ),
        "lowest15": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_o_lowest15.csv"),
            "K+O Lowest 15",
            colors["lowest15"],
        ),
        "highest15": (
            os.path.join(hotpotqa_dir, "hotpotqa_llama31_8b_k_o_highest15.csv"),
            "K+O Highest 15",
            colors["highest15"],
        ),
    }
    # Generate F1 and EM plots for K+O
    make_plot(
        dataset="HotpotQA",
        model_label="Llama-3.1-8B",
        csv_map=hotpotqa_k_o_map,
        out_dir=args.out_dir,
        plot_suffix="k_o",
        metric="f1",
    )
    make_plot(
        dataset="HotpotQA",
        model_label="Llama-3.1-8B",
        csv_map=hotpotqa_k_o_map,
        out_dir=args.out_dir,
        plot_suffix="k_o",
        metric="em",
    )



if __name__ == "__main__":
    main()

