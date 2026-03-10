"""Generate 4 separate Plotly epoch-accuracy plots."""
import json
import os
import plotly.graph_objects as go

BASE = "/home/kadir/topo/numpy_weights/exploration-finetuning"
BASELINE = 56.2
FULL_FT = 64.4
LORA = 59.8


def load_json(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    epochs = sorted(int(k) for k in data.keys())
    means = [sum(data[str(e)]) / len(data[str(e)]) * 100 for e in epochs]
    return epochs, means


def load_json_split_runs(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    max_runs = max(len(v) for v in data.values())
    result = {}
    for r in range(max_runs):
        run_data = {}
        for k, vals in data.items():
            if r < len(vals):
                run_data[int(k)] = vals[r] * 100
        result[r + 1] = run_data
    return result


def add_ref_lines(fig):
    fig.add_hline(y=BASELINE, line_dash="dot", line_color="gray", line_width=1.5,
                  annotation_text=f"Baseline ({BASELINE}%)", annotation_position="bottom left",
                  annotation_font_size=12)
    fig.add_hline(y=FULL_FT, line_dash="dot", line_color="green", line_width=1.5,
                  annotation_text=f"Full FT ({FULL_FT}%)", annotation_position="top left",
                  annotation_font_size=12)
    fig.add_hline(y=LORA, line_dash="dot", line_color="blue", line_width=1.5,
                  annotation_text=f"LoRA ({LORA}%)", annotation_position="bottom left",
                  annotation_font_size=12)


def style_layout(fig, title, ymin=45, ymax=68):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title=dict(text="Epoch", font=dict(size=14)), tickvals=list(range(7)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(text="Accuracy (%)", font=dict(size=14)), tickfont=dict(size=14),
                   range=[ymin, ymax]),
        legend=dict(font=dict(size=12), x=1.02, y=1, xanchor='left'),
        width=1100, height=600, margin=dict(r=300),
        template="plotly_white",
    )


# ── Config ────────────────────────────────────────────────────────

plan_dir = os.path.join(BASE, "epoch_accuracy_results")
plan_configs = [
    ("A", "Plan A (V+O+MLP, L0-9)",          "31,119,180",  "solid",  "circle"),
    ("B", "Plan B (V+O, L0-9)",              "255,127,14",  "solid",  "square"),
    ("C", "Plan C (V+O+MLP, L22-31+head)",   "44,160,44",   "dash",   "diamond"),
    ("D", "Plan D (V+O, L22-31+head)",        "214,39,40",   "dash",   "triangle-up"),
]

perturb_results = [
    ("25% perturbed (Plan A)", 48.29, "star",    "#636EFA"),
    ("50% perturbed (Plan A)", 48.98, "hexagram", "#EF553B"),
    ("100% perturbed (Plan A)", 47.46, "x",       "#00CC96"),
]

w_dir = os.path.join(BASE, "epoch_accuracy_wasserstein_912")
w_configs = [
    ("freeze-lowest9",   "Lowest 9 V+O",   "31,119,180",  "solid"),
    ("freeze-lowest12",  "Lowest 12 V+O",  "255,127,14",  "solid"),
    ("freeze-highest9",  "Highest 9 V+O",  "44,160,44",   "dash"),
    ("freeze-highest12", "Highest 12 V+O", "214,39,40",   "dash"),
]


def load_and_merge(paths):
    merged = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for k, vals in data.items():
            e = int(k)
            merged.setdefault(e, []).extend(vals)
    epochs = sorted(merged.keys())
    means = [sum(merged[e]) / len(merged[e]) * 100 for e in epochs]
    return epochs, means


# ── Plot 1: Plan A/B/C/D averaged ────────────────────────────────

fig1 = go.Figure()
for plan, label, rgb, dash, sym in plan_configs:
    path = os.path.join(plan_dir, f"plan_{plan}_epoch_accuracy.json")
    epochs, means = load_and_merge([path])
    if not epochs:
        continue
    fig1.add_trace(go.Scatter(
        x=[0] + epochs, y=[BASELINE] + means, mode='lines+markers', name=label,
        line=dict(color=f"rgba({rgb},1.0)", width=2.5, dash=dash),
        marker=dict(size=8, symbol=sym),
    ))
for plabel, pacc, psym, pcolor in perturb_results:
    fig1.add_trace(go.Scatter(
        x=[6], y=[pacc], mode='markers+text', name=plabel,
        marker=dict(size=13, symbol=psym, color=pcolor, line=dict(width=1, color='black')),
        text=[f"{pacc:.1f}%"], textposition="middle right", textfont=dict(size=10),
    ))
add_ref_lines(fig1)
style_layout(fig1, "GSM8K Epoch Accuracy — Plans A/B/C/D (avg of 2 runs)")
out1 = os.path.join(BASE, "epoch_accuracy_plans_avg.html")
fig1.write_html(out1)
print(f"Saved: {out1}")


# ── Plot 2: Wasserstein 9/12 averaged ────────────────────────────

fig2 = go.Figure()
for key, label, rgb, dash in w_configs:
    run1 = os.path.join(w_dir, f"{key}_epoch_accuracy.json")
    run2 = os.path.join(w_dir, f"{key}-r2_epoch_accuracy.json")
    epochs, means = load_and_merge([run1, run2])
    if not epochs:
        continue
    fig2.add_trace(go.Scatter(
        x=[0] + epochs, y=[BASELINE] + means, mode='lines+markers', name=label,
        line=dict(color=f"rgba({rgb},1.0)", width=2.5, dash=dash), marker=dict(size=7),
    ))
add_ref_lines(fig2)
style_layout(fig2, "GSM8K Epoch Accuracy — Wasserstein Freeze 9/12 (avg of 2 runs)", ymin=50)
out2 = os.path.join(BASE, "epoch_accuracy_freeze912_avg.html")
fig2.write_html(out2)
print(f"Saved: {out2}")


# ── Plot 3: All Wasserstein 3/6/9/12 averaged ────────────────────

w36_dir = os.path.join(BASE, "epoch_accuracy_wasserstein_36")
all_w_configs = [
    ("freeze-lowest3",   w36_dir, "Freeze Lowest 3 V+O",   "31,119,180",   "solid",  "circle"),
    ("freeze-lowest6",   w36_dir, "Freeze Lowest 6 V+O",   "255,127,14",   "solid",  "square"),
    ("freeze-lowest9",   w_dir,   "Freeze Lowest 9 V+O",   "44,160,44",    "solid",  "diamond"),
    ("freeze-lowest12",  w_dir,   "Freeze Lowest 12 V+O",  "148,103,189",  "solid",  "triangle-up"),
    ("freeze-highest3",  w36_dir, "Freeze Highest 3 V+O",  "214,39,40",    "dash",   "circle"),
    ("freeze-highest6",  w36_dir, "Freeze Highest 6 V+O",  "140,86,75",    "dash",   "square"),
    ("freeze-highest9",  w_dir,   "Freeze Highest 9 V+O",  "227,119,194",  "dash",   "diamond"),
    ("freeze-highest12", w_dir,   "Freeze Highest 12 V+O", "127,127,127",  "dash",   "triangle-up"),
]

fig3 = go.Figure()
for key, src_dir, label, rgb, dash, sym in all_w_configs:
    run1 = os.path.join(src_dir, f"{key}_epoch_accuracy.json")
    run2 = os.path.join(src_dir, f"{key}-r2_epoch_accuracy.json")
    epochs, means = load_and_merge([run1, run2])
    if not epochs:
        continue
    fig3.add_trace(go.Scatter(
        x=[0] + epochs, y=[BASELINE] + means, mode='lines+markers', name=label,
        line=dict(color=f"rgba({rgb},1.0)", width=2.5, dash=dash),
        marker=dict(size=7, symbol=sym),
    ))
add_ref_lines(fig3)
style_layout(fig3, "GSM8K Epoch Accuracy — Wasserstein Freeze 3/6/9/12 (avg of 2 runs)", ymin=50)
out3 = os.path.join(BASE, "epoch_accuracy_freeze_all_avg.html")
fig3.write_html(out3)
print(f"Saved: {out3}")
