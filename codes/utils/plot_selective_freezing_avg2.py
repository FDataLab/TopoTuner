"""
Generate Plotly epoch-accuracy plot for selective freezing experiments.
Averages run1 (eval-selective-freezing) and run2 (eval-selective-freezing-run2).
Wass: solid lines. Norm: dashed lines.
"""

import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go

BASE = "/home/kadir/topo/numpy_weights/exploration-finetuning"
DEFAULT_RUN1 = os.path.join(BASE, "eval", "qwen", "selective-freezing", "eval-selective-freezing")
DEFAULT_RUN2 = os.path.join(BASE, "eval", "qwen", "selective-freezing", "run2")
DEFAULT_OUTPUT = os.path.join(BASE, "eval", "qwen", "plots", "epoch_accuracy_freeze_avg2.html")

BASELINE_ACC = 0.5618
FULL_FT_ACC = 0.6444
LORA_ACC = 0.5982

# Order: Wass then Norm, Low to High. Wass solid, Norm dashed.
EXPERIMENT_CONFIG = [
    ("Wass-Low3", {"label": "Wass-Low3", "color": "#1f77b4", "dash": None}),
    ("Wass-Low6", {"label": "Wass-Low6", "color": "#17becf", "dash": None}),
    ("Wass-Low9", {"label": "Wass-Low9", "color": "#9467bd", "dash": None}),
    ("Wass-Low15", {"label": "Wass-Low15", "color": "#2ca02c", "dash": None}),
    ("Wass-High6", {"label": "Wass-High6", "color": "#ff7f0e", "dash": None}),
    ("Wass-High9", {"label": "Wass-High9", "color": "#e377c2", "dash": None}),
    ("Norm-Low3", {"label": "Norm-Low3", "color": "#d62728", "dash": "dash"}),
    ("Norm-Low6", {"label": "Norm-Low6", "color": "#8c564b", "dash": "dash"}),
    ("Norm-Low9", {"label": "Norm-Low9", "color": "#bcbd22", "dash": "dash"}),
    ("Norm-Low15", {"label": "Norm-Low15", "color": "#7f7f7f", "dash": "dash"}),
    ("Norm-High6", {"label": "Norm-High6", "color": "#ff9896", "dash": "dash"}),
    ("Norm-High9", {"label": "Norm-High9", "color": "#c5b0d5", "dash": "dash"}),
]


def load_avg_two_runs(run1_dir, run2_dir, name):
    """Load epoch accuracies from run1 and run2, return mean and std."""
    path1 = os.path.join(run1_dir, f"{name}_epoch_accuracy.json")
    path2 = os.path.join(run2_dir, f"{name}_epoch_accuracy.json")
    if not os.path.exists(path1) or not os.path.exists(path2):
        return None
    with open(path1) as f:
        d1 = json.load(f)
    with open(path2) as f:
        d2 = json.load(f)
    epochs = sorted(int(k) for k in d1.keys())
    means = []
    stds = []
    for e in epochs:
        v1 = d1[str(e)][0] * 100
        v2 = d2[str(e)][0] * 100
        means.append((v1 + v2) / 2)
        stds.append(np.std([v1, v2]) if v1 != v2 else 0)
    return [0] + epochs, [BASELINE_ACC * 100] + means, [0] + stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run1-dir", default=DEFAULT_RUN1,
                        help="Run1 eval directory")
    parser.add_argument("--run2-dir", default=DEFAULT_RUN2,
                        help="Run2 eval directory")
    parser.add_argument("--output", default=None,
                        help="Output HTML (default: eval/qwen/plots/epoch_accuracy_freeze_avg2.html)")
    args = parser.parse_args()

    run1 = args.run1_dir if os.path.isabs(args.run1_dir) else os.path.join(BASE, args.run1_dir)
    run2 = args.run2_dir if os.path.isabs(args.run2_dir) else os.path.join(BASE, args.run2_dir)
    out_path = args.output or DEFAULT_OUTPUT

    fig = go.Figure()

    for name, cfg in EXPERIMENT_CONFIG:
        loaded = load_avg_two_runs(run1, run2, name)
        if loaded is None:
            print(f"Warning: {name} not found in both run1 and run2, skipping")
            continue
        epochs, means_pct, stds_pct = loaded

        line_dict = dict(color=cfg["color"], width=2.5)
        if cfg.get("dash"):
            line_dict["dash"] = cfg["dash"]

        fig.add_trace(go.Scatter(
            x=epochs, y=means_pct,
            mode="lines+markers",
            name=cfg["label"],
            line=line_dict,
            marker=dict(size=8),
            hovertemplate="Epoch %{x}<br>Accuracy: %{y:.1f}% (avg of 2 runs)<extra>" + cfg["label"] + "</extra>",
        ))

    fig.add_hline(y=BASELINE_ACC * 100, line_dash="dash", line_color="gray", line_width=2,
                  annotation_text=f"Baseline (no FT): {BASELINE_ACC*100:.1f}%",
                  annotation_position="top left", annotation_font=dict(size=11, color="gray"))
    fig.add_hline(y=FULL_FT_ACC * 100, line_dash="dash", line_color="#9467bd", line_width=2,
                  annotation_text=f"Full FT (no freeze): {FULL_FT_ACC*100:.1f}%",
                  annotation_position="top right", annotation_font=dict(size=11, color="#9467bd"))
    fig.add_hline(y=LORA_ACC * 100, line_dash="dot", line_color="#8c564b", line_width=2,
                  annotation_text=f"LoRA: {LORA_ACC*100:.1f}%",
                  annotation_position="bottom right", annotation_font=dict(size=11, color="#8c564b"))

    fig.update_layout(
        title=dict(text="GSM8K Epoch Accuracy — Selective Freezing (avg of 2 runs, Wass solid, Norm dashed)", font=dict(size=18)),
        xaxis=dict(title="Epoch", dtick=1, range=[-0.3, 6.5], tickfont=dict(size=16), title_font=dict(size=16)),
        yaxis=dict(title="GSM8K Accuracy (%)", range=[44, 68], tickfont=dict(size=16), title_font=dict(size=16)),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        template="plotly_white", width=1000, height=600,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
