"""Generate Plotly epoch-accuracy plots. Edit RUN below to choose which plots to generate."""
import json
import os
import plotly.graph_objects as go

BASE = "/home/kadir/topo/numpy_weights/exploration-finetuning"
BASELINE_LLAMA = 56.2
FULL_FT_LLAMA = 64.4
LORA_LLAMA = 59.8

# Edit this: add/remove what to run
RUN = ["qwen"]


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
    if not epochs:
        return [], []
    means = [sum(merged[e]) / len(merged[e]) * 100 for e in epochs]
    return epochs, means


def style_layout(fig, title, ymin=45, ymax=68):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title=dict(text="Epoch", font=dict(size=14)), tickvals=list(range(7)), tickfont=dict(size=14)),
        yaxis=dict(title=dict(text="Accuracy (%)", font=dict(size=14)), tickfont=dict(size=14), range=[ymin, ymax]),
        legend=dict(font=dict(size=12), x=1.02, y=1, xanchor='left'),
        width=1100, height=600, margin=dict(r=300),
        template="plotly_white",
    )


# ── Plot definitions: add new model results here ─────────────────────
# Each entry: (model_key, title, out_path, configs, baseline?, ref_lines?, ymin, ymax)
# configs: list of (path_or_paths, label, rgb, dash, symbol) — path can be str or list
# baseline: prepend epoch 0 with this value; ref_lines: add Full/LoRA hlines

def run_llama_plans():
    plan_dir = os.path.join(BASE, "eval", "llama", "epoch_accuracy", "epoch_accuracy_results")
    configs = [
        ([os.path.join(plan_dir, "plan_A_epoch_accuracy.json")], "Plan A (V+O+MLP, L0-9)",   "31,119,180", "solid", "circle"),
        ([os.path.join(plan_dir, "plan_B_epoch_accuracy.json")], "Plan B (V+O, L0-9)",       "255,127,14", "solid", "square"),
        ([os.path.join(plan_dir, "plan_C_epoch_accuracy.json")], "Plan C (V+O+MLP, L22-31+head)", "44,160,44", "dash", "diamond"),
        ([os.path.join(plan_dir, "plan_D_epoch_accuracy.json")], "Plan D (V+O, L22-31+head)",   "214,39,40", "dash", "triangle-up"),
    ]
    perturb = [(48.29, "25% perturbed (Plan A)", "star", "#636EFA"), (48.98, "50% perturbed (Plan A)", "hexagram", "#EF553B"), (47.46, "100% perturbed (Plan A)", "x", "#00CC96")]
    fig = go.Figure()
    for paths, label, rgb, dash, sym in configs:
        epochs, means = load_and_merge(paths)
        if epochs:
            fig.add_trace(go.Scatter(x=[0] + epochs, y=[BASELINE_LLAMA] + means, mode='lines+markers', name=label,
                line=dict(color=f"rgba({rgb},1.0)", width=2.5, dash=dash), marker=dict(size=8, symbol=sym)))
    for pacc, plabel, psym, pcolor in perturb:
        fig.add_trace(go.Scatter(x=[6], y=[pacc], mode='markers+text', name=plabel,
            marker=dict(size=13, symbol=psym, color=pcolor, line=dict(width=1, color='black')),
            text=[f"{pacc:.1f}%"], textposition="middle right", textfont=dict(size=10)))
    fig.add_hline(y=BASELINE_LLAMA, line_dash="dot", line_color="gray", annotation_text=f"Baseline ({BASELINE_LLAMA}%)", annotation_position="bottom left")
    fig.add_hline(y=FULL_FT_LLAMA, line_dash="dot", line_color="green", annotation_text=f"Full FT ({FULL_FT_LLAMA}%)", annotation_position="top left")
    fig.add_hline(y=LORA_LLAMA, line_dash="dot", line_color="blue", annotation_text=f"LoRA ({LORA_LLAMA}%)", annotation_position="bottom left")
    style_layout(fig, "GSM8K Epoch Accuracy — Plans A/B/C/D (Llama)", ymin=45, ymax=68)
    out = os.path.join(BASE, "eval", "llama", "plots", "epoch_accuracy_plans_avg.html")
    fig.write_html(out)
    print(f"Saved: {out}")


def run_llama_freeze912():
    w_dir = os.path.join(BASE, "eval", "llama", "epoch_accuracy", "epoch_accuracy_wasserstein_912")
    configs = [
        ([os.path.join(w_dir, "freeze-lowest9_epoch_accuracy.json"), os.path.join(w_dir, "freeze-lowest9-r2_epoch_accuracy.json")], "Lowest 9 V+O",  "31,119,180", "solid"),
        ([os.path.join(w_dir, "freeze-lowest12_epoch_accuracy.json"), os.path.join(w_dir, "freeze-lowest12-r2_epoch_accuracy.json")], "Lowest 12 V+O", "255,127,14", "solid"),
        ([os.path.join(w_dir, "freeze-highest9_epoch_accuracy.json"), os.path.join(w_dir, "freeze-highest9-r2_epoch_accuracy.json")], "Highest 9 V+O", "44,160,44", "dash"),
        ([os.path.join(w_dir, "freeze-highest12_epoch_accuracy.json"), os.path.join(w_dir, "freeze-highest12-r2_epoch_accuracy.json")], "Highest 12 V+O", "214,39,40", "dash"),
    ]
    fig = go.Figure()
    for paths, label, rgb, dash in configs:
        epochs, means = load_and_merge(paths)
        if epochs:
            fig.add_trace(go.Scatter(x=[0] + epochs, y=[BASELINE_LLAMA] + means, mode='lines+markers', name=label,
                line=dict(color=f"rgba({rgb},1.0)", width=2.5, dash=dash), marker=dict(size=7)))
    fig.add_hline(y=BASELINE_LLAMA, line_dash="dot", line_color="gray", annotation_text=f"Baseline ({BASELINE_LLAMA}%)", annotation_position="bottom left")
    fig.add_hline(y=FULL_FT_LLAMA, line_dash="dot", line_color="green", annotation_text=f"Full FT ({FULL_FT_LLAMA}%)", annotation_position="top left")
    fig.add_hline(y=LORA_LLAMA, line_dash="dot", line_color="blue", annotation_text=f"LoRA ({LORA_LLAMA}%)", annotation_position="bottom left")
    style_layout(fig, "GSM8K Epoch Accuracy — Wasserstein Freeze 9/12 (Llama)", ymin=50, ymax=68)
    out = os.path.join(BASE, "eval", "llama", "plots", "epoch_accuracy_freeze912_avg.html")
    fig.write_html(out)
    print(f"Saved: {out}")


def run_llama_freeze_all():
    w_dir = os.path.join(BASE, "eval", "llama", "epoch_accuracy", "epoch_accuracy_wasserstein_912")
    w36_dir = os.path.join(BASE, "eval", "llama", "epoch_accuracy", "epoch_accuracy_wasserstein_36")
    configs = [
        ("freeze-lowest3",  w36_dir, "Freeze Lowest 3 V+O",  "31,119,180",  "solid", "circle"),
        ("freeze-lowest6",  w36_dir, "Freeze Lowest 6 V+O",  "255,127,14",  "solid", "square"),
        ("freeze-lowest9",  w_dir,   "Freeze Lowest 9 V+O",  "44,160,44",   "solid", "diamond"),
        ("freeze-lowest12", w_dir,   "Freeze Lowest 12 V+O", "148,103,189", "solid", "triangle-up"),
        ("freeze-highest3", w36_dir, "Freeze Highest 3 V+O", "214,39,40",   "dash",  "circle"),
        ("freeze-highest6", w36_dir, "Freeze Highest 6 V+O", "140,86,75",   "dash",  "square"),
        ("freeze-highest9", w_dir,   "Freeze Highest 9 V+O", "227,119,194", "dash",  "diamond"),
        ("freeze-highest12", w_dir,  "Freeze Highest 12 V+O", "127,127,127", "dash",  "triangle-up"),
    ]
    fig = go.Figure()
    for key, src_dir, label, rgb, dash, sym in configs:
        run1 = os.path.join(src_dir, f"{key}_epoch_accuracy.json")
        run2 = os.path.join(src_dir, f"{key}-r2_epoch_accuracy.json")
        epochs, means = load_and_merge([run1, run2])
        if epochs:
            fig.add_trace(go.Scatter(x=[0] + epochs, y=[BASELINE_LLAMA] + means, mode='lines+markers', name=label,
                line=dict(color=f"rgba({rgb},1.0)", width=2.5, dash=dash), marker=dict(size=7, symbol=sym)))
    fig.add_hline(y=BASELINE_LLAMA, line_dash="dot", line_color="gray", annotation_text=f"Baseline ({BASELINE_LLAMA}%)", annotation_position="bottom left")
    fig.add_hline(y=FULL_FT_LLAMA, line_dash="dot", line_color="green", annotation_text=f"Full FT ({FULL_FT_LLAMA}%)", annotation_position="top left")
    fig.add_hline(y=LORA_LLAMA, line_dash="dot", line_color="blue", annotation_text=f"LoRA ({LORA_LLAMA}%)", annotation_position="bottom left")
    style_layout(fig, "GSM8K Epoch Accuracy — Wasserstein Freeze 3/6/9/12 (Llama)", ymin=50, ymax=68)
    out = os.path.join(BASE, "eval", "llama", "plots", "epoch_accuracy_freeze_all_avg.html")
    fig.write_html(out)
    print(f"Saved: {out}")


def run_qwen():
    qwen_gsm8k = os.path.join(BASE, "eval", "qwen", "gsm8k", "eval-qwen-gsm8k")
    qwen_freeze = os.path.join(BASE, "eval", "qwen", "epoch_accuracy", "eval-qwen-epoch-accuracy")
    # Full=black, LoRA=dark gray. Others: distinct colors (blue, orange, green, red). Low=solid, high=dashed.
    configs = [
        (os.path.join(qwen_gsm8k, "Qwen-Full_epoch_accuracy.json"),  "Full",      "0,0,0",         "solid", "circle"),
        (os.path.join(qwen_gsm8k, "Qwen-LoRA_epoch_accuracy.json"),  "LoRA",      "80,80,80",      "solid", "square"),
        (os.path.join(qwen_freeze, "norm-low6_epoch_accuracy.json"),  "norm-low6",  "31,119,180",   "solid", "diamond"),
        (os.path.join(qwen_freeze, "wass-low6_epoch_accuracy.json"),  "wass-low6",  "255,127,14",   "solid", "cross"),
        (os.path.join(qwen_freeze, "norm-high6_epoch_accuracy.json"), "norm-high6", "44,160,44",    "dash",  "triangle-up"),
        (os.path.join(qwen_freeze, "wass-high6_epoch_accuracy.json"), "wass-high6", "214,39,40",    "dash",  "star"),
    ]
    fig = go.Figure()
    for path, label, rgb, dash, sym in configs:
        epochs, means = load_and_merge([path])
        if epochs:
            fig.add_trace(go.Scatter(x=epochs, y=means, mode='lines+markers', name=label,
                line=dict(color=f"rgb({rgb})", width=2.5, dash=dash), marker=dict(size=8, symbol=sym)))
    # Reference lines for Full and LoRA final accuracy
    for fname, label in [("Qwen-Full_epoch_accuracy.json", "Full"), ("Qwen-LoRA_epoch_accuracy.json", "LoRA")]:
        p = os.path.join(qwen_gsm8k, fname)
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            if data:
                last = max(int(k) for k in data.keys())
                acc = sum(data[str(last)]) / len(data[str(last)]) * 100
                fig.add_hline(y=acc, line_dash="dot", line_color="gray", line_width=1.5,
                    annotation_text=f"{label} final: {acc:.1f}%",
                    annotation_position="top right" if label == "Full" else "bottom right", annotation_font_size=11)
    style_layout(fig, "Qwen GSM8K Epoch Accuracy — Full, LoRA, and 4 freeze experiments", ymin=78, ymax=88)
    out = os.path.join(BASE, "eval", "qwen", "gsm8k", "eval-qwen-gsm8k", "epoch_accuracy_qwen_combined.html")
    fig.write_html(out)
    print(f"Saved: {out}")


RUNNERS = {
    "llama_plans": run_llama_plans,
    "llama_freeze912": run_llama_freeze912,
    "llama_freeze_all": run_llama_freeze_all,
    "qwen": run_qwen,
}

if __name__ == "__main__":
    for name in RUN:
        if name in RUNNERS:
            RUNNERS[name]()
        else:
            print(f"Unknown: {name}")
