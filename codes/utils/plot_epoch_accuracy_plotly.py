"""
Generate Plotly epoch-accuracy plots for all freezing plans.
Epoch 0 = baseline (pre-training) accuracy.
Also generates Qwen plot: Full, LoRA, + 4 freeze experiments with reference lines.
"""

import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE = "/home/kadir/topo/numpy_weights/exploration-finetuning"
RESULTS_DIR = os.path.join(BASE, "eval", "llama", "epoch_accuracy", "epoch_accuracy_results")
BASELINE_ACC = 0.5618  # 56.2% baseline Llama 3.1 8B on GSM8K
FULL_FT_ACC = 0.6444   # 64.4% full finetuning (no freezing)
LORA_ACC = 0.5982       # 59.8% LoRA finetuning
PERTURB_25_ACC = 0.4829 # 48.3% Plan A on 25% perturbed
PERTURB_50_ACC = 0.4898 # 49.0% Plan A on 50% perturbed

PLAN_CONFIG = {
    "A": {"label": "Plan A (V+O+MLP, layers 0-9)", "color": "#1f77b4"},
    "B": {"label": "Plan B (V+O, layers 0-9)", "color": "#ff7f0e"},
    "C": {"label": "Plan C (V+O+MLP, layers 22-31+head)", "color": "#2ca02c"},
    "D": {"label": "Plan D (V+O, layers 22-31+head)", "color": "#d62728"},
}


def load_plan(plan):
    with open(f"{RESULTS_DIR}/plan_{plan}_epoch_accuracy.json") as f:
        data = json.load(f)
    epochs = [0] + sorted(int(k) for k in data.keys())
    means = [BASELINE_ACC]
    stds = [0.0]
    run1 = [BASELINE_ACC]
    run2 = [BASELINE_ACC]
    for e in epochs[1:]:
        vals = data[str(e)]
        means.append(np.mean(vals))
        stds.append(np.std(vals) if len(vals) > 1 else 0)
        run1.append(vals[0])
        run2.append(vals[1] if len(vals) > 1 else None)
    return epochs, means, stds, run1, run2


# ── Combined plot ─────────────────────────────────────────────

fig_combined = go.Figure()

for plan, cfg in PLAN_CONFIG.items():
    epochs, means, stds, run1, run2 = load_plan(plan)
    means_pct = [m * 100 for m in means]
    stds_pct = [s * 100 for s in stds]

    fig_combined.add_trace(go.Scatter(
        x=epochs, y=means_pct,
        mode='lines+markers',
        name=cfg["label"],
        line=dict(color=cfg["color"], width=2.5),
        marker=dict(size=8),
        hovertemplate="Epoch %{x}<br>Accuracy: %{y:.1f}%<extra>" + cfg["label"] + "</extra>",
    ))

    if any(s > 0 for s in stds_pct):
        upper = [m + s for m, s in zip(means_pct, stds_pct)]
        lower = [m - s for m, s in zip(means_pct, stds_pct)]
        fig_combined.add_trace(go.Scatter(
            x=epochs + epochs[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor=f"rgba({int(cfg['color'][1:3],16)},{int(cfg['color'][3:5],16)},{int(cfg['color'][5:7],16)},0.1)",
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip',
        ))

fig_combined.add_hline(y=BASELINE_ACC * 100, line_dash="dash", line_color="gray", line_width=2,
                       annotation_text=f"Baseline (no FT): {BASELINE_ACC*100:.1f}%",
                       annotation_position="top left",
                       annotation_font=dict(size=11, color="gray"))

fig_combined.add_hline(y=FULL_FT_ACC * 100, line_dash="dash", line_color="#9467bd", line_width=2,
                       annotation_text=f"Full FT (no freeze): {FULL_FT_ACC*100:.1f}%",
                       annotation_position="top right",
                       annotation_font=dict(size=11, color="#9467bd"))

fig_combined.add_hline(y=LORA_ACC * 100, line_dash="dot", line_color="#8c564b", line_width=2,
                       annotation_text=f"LoRA: {LORA_ACC*100:.1f}%",
                       annotation_position="bottom right",
                       annotation_font=dict(size=11, color="#8c564b"))

fig_combined.add_trace(go.Scatter(
    x=[6], y=[PERTURB_25_ACC * 100],
    mode='markers',
    name="Plan A (25% perturbed)",
    marker=dict(size=12, symbol="diamond", color="#1f77b4", line=dict(width=2, color="black")),
    hovertemplate="25% perturbed<br>Accuracy: %{y:.1f}%<extra></extra>",
))
fig_combined.add_trace(go.Scatter(
    x=[6], y=[PERTURB_50_ACC * 100],
    mode='markers',
    name="Plan A (50% perturbed)",
    marker=dict(size=12, symbol="diamond-open", color="#1f77b4", line=dict(width=2, color="#1f77b4")),
    hovertemplate="50% perturbed<br>Accuracy: %{y:.1f}%<extra></extra>",
))

fig_combined.update_layout(
    title=dict(text="GSM8K Epoch Accuracy by Freezing Plan", font=dict(size=18)),
    xaxis=dict(title="Epoch", dtick=1, range=[-0.3, 6.5], tickfont=dict(size=16), title_font=dict(size=16)),
    yaxis=dict(title="GSM8K Accuracy (%)", range=[44, 68], tickfont=dict(size=16), title_font=dict(size=16)),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    template="plotly_white",
    width=1000, height=600,
)

out_combined = os.path.join(BASE, "eval", "llama", "plots", "epoch_accuracy_combined.html")
os.makedirs(os.path.dirname(out_combined), exist_ok=True)
fig_combined.write_html(out_combined)
print(f"Saved: {out_combined}")


# ── Qwen: Full, LoRA, + 4 freeze experiments ─────────────────────────────

QWEN_GSM8K = os.path.join(BASE, "eval", "qwen", "gsm8k", "eval-qwen-gsm8k")
QWEN_EPOCH = os.path.join(BASE, "eval", "qwen", "epoch_accuracy", "eval-qwen-epoch-accuracy")

QWEN_CONFIG = [
    ("Qwen-Full_epoch_accuracy.json", "Full", "#9467bd"),
    ("Qwen-LoRA_epoch_accuracy.json", "LoRA", "#8c564b"),
    ("norm-low6_epoch_accuracy.json", "norm-low6", "#1f77b4"),
    ("norm-high6_epoch_accuracy.json", "norm-high6", "#1f6fb2"),
    ("wass-low6_epoch_accuracy.json", "wass-low6", "#d62728"),
    ("wass-high6_epoch_accuracy.json", "wass-high6", "#c0392b"),
]


def load_qwen_json(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    epochs = sorted(int(k) for k in data.keys())
    means = [np.mean(data[str(e)]) * 100 for e in epochs]
    return epochs, means


fig_qwen = go.Figure()

full_final, lora_final = None, None
for fname, label, color in QWEN_CONFIG:
    if "Full" in fname:
        path = os.path.join(QWEN_GSM8K, fname)
    elif "LoRA" in fname:
        path = os.path.join(QWEN_GSM8K, fname)
    else:
        path = os.path.join(QWEN_EPOCH, fname)
    epochs, means = load_qwen_json(path)
    if not epochs:
        continue
    fig_qwen.add_trace(go.Scatter(
        x=epochs, y=means,
        mode='lines+markers',
        name=label,
        line=dict(color=color, width=2.5),
        marker=dict(size=8),
        hovertemplate="Epoch %{x}<br>Accuracy: %{y:.1f}%<extra>" + label + "</extra>",
    ))
    if "Full" in label:
        full_final = means[-1] if means else None
    elif "LoRA" in label:
        lora_final = means[-1] if means else None

if full_final is not None:
    fig_qwen.add_hline(y=full_final, line_dash="dash", line_color="#9467bd", line_width=2,
                       annotation_text=f"Full FT final: {full_final:.1f}%",
                       annotation_position="top right",
                       annotation_font=dict(size=11, color="#9467bd"))
if lora_final is not None:
    fig_qwen.add_hline(y=lora_final, line_dash="dot", line_color="#8c564b", line_width=2,
                       annotation_text=f"LoRA final: {lora_final:.1f}%",
                       annotation_position="bottom right",
                       annotation_font=dict(size=11, color="#8c564b"))

fig_qwen.update_layout(
    title=dict(text="Qwen GSM8K Epoch Accuracy — Full, LoRA, Freeze Experiments", font=dict(size=18)),
    xaxis=dict(title="Epoch", dtick=1, tickfont=dict(size=16), title_font=dict(size=16)),
    yaxis=dict(title="GSM8K Accuracy (%)", tickfont=dict(size=16), title_font=dict(size=16)),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    template="plotly_white",
    width=1000, height=600,
)

out_qwen = os.path.join(QWEN_GSM8K, "epoch_accuracy_qwen_combined.html")
fig_qwen.write_html(out_qwen)
print(f"Saved: {out_qwen}")

print("\nDone!")
