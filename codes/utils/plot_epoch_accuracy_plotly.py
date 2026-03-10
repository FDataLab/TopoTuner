"""
Generate Plotly epoch-accuracy plots for all freezing plans.
Epoch 0 = baseline (pre-training) accuracy.
"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = "epoch_accuracy_results"
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

fig_combined.write_html(f"{RESULTS_DIR}/epoch_accuracy_combined.html")
print(f"Saved: {RESULTS_DIR}/epoch_accuracy_combined.html")




print("\nDone!")
