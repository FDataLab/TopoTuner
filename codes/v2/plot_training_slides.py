#!/usr/bin/env python3
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

RUNS = [
    ("gsm8k-full-finetuned", "FULL / Base",
     "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-full-finetuned/training_report_full.json"),
    ("gsm8k-full-instruct", "FULL / Instruct",
     "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-full-instruct/training_report_full.json"),
    ("gsm8k-lora-finetuned", "LoRA / Base",
     "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-lora-finetuned/training_report_lora.json"),
    ("gsm8k-lora-instruct", "LoRA / Instruct",
     "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-lora-instruct/training_report_lora.json"),
]

OUTDIR = os.getcwd()

# ---- cleaner fonts (big but not chunky) ----
mpl.rcParams.update({
    "figure.figsize": (16, 10),
    "savefig.dpi": 300,

    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "lines.linewidth": 2.2,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

def load_report(path):
    with open(path, "r") as f:
        r = json.load(f)
    steps = np.asarray(r["metrics_log"]["steps"], dtype=float)
    loss  = np.asarray(r["metrics_log"]["losses"], dtype=float)
    lr    = np.asarray(r["metrics_log"]["learning_rates"], dtype=float)
    grad  = np.asarray(r["metrics_log"]["gradient_norms"], dtype=float)
    warmup = int(r.get("hyperparameters", {}).get("warmup_steps", 0))
    avg_step_s = float(r.get("timing", {}).get("avg_step_s", float("nan")))
    return steps, loss, lr, grad, warmup, avg_step_s

def ema(y, alpha=0.10):
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float)
    s = float(y[0]); out[0] = s
    for i in range(1, len(y)):
        s = alpha*float(y[i]) + (1-alpha)*s
        out[i] = s
    return out

def robust_ylim(y, lo_q=0.02, hi_q=0.98, pad_frac=0.05):
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0, 1)
    lo = float(np.quantile(y, lo_q))
    hi = float(np.quantile(y, hi_q))
    if hi <= lo:
        lo = float(y.min()); hi = float(y.max())
    span = hi - lo
    return (lo - pad_frac*span, hi + pad_frac*span)

# shared x-lims only (consistent comparison of training progress)
all_steps = []
for _, _, p in RUNS:
    s, *_ = load_report(p)
    all_steps.append(s)
XMIN = min(s.min() for s in all_steps)
XMAX = max(s.max() for s in all_steps)

def plot_one(name, label, path, alpha=0.10):
    steps, loss, lr, grad, warmup, avg_step_s = load_report(path)

    loss_s = ema(loss, alpha=alpha)
    grad_s = ema(grad, alpha=alpha)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"{label} — Training Metrics", fontweight="bold")

    # Loss
    ax = axs[0, 0]
    ax.plot(steps, loss_s, label=f"EMA (α={alpha})")
    if warmup > 0: ax.axvline(warmup, linestyle="--", alpha=0.4, label="warmup end")
    ax.set_title("Training Loss")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(*robust_ylim(loss_s))
    ax.legend(loc="best")

    # LR
    ax = axs[0, 1]
    ax.plot(steps, lr)
    if warmup > 0: ax.axvline(warmup, linestyle="--", alpha=0.4)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step"); ax.set_ylabel("LR")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(*robust_ylim(lr, lo_q=0.0, hi_q=1.0, pad_frac=0.02))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style="plain", axis="y")

    # Grad norm
    ax = axs[1, 0]
    ax.plot(steps, grad_s, label=f"EMA (α={alpha})")
    if warmup > 0: ax.axvline(warmup, linestyle="--", alpha=0.4)
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Step"); ax.set_ylabel("L2 Norm")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(*robust_ylim(grad_s))
    ax.legend(loc="best")

    # Step duration (avg only)
    ax = axs[1, 1]
    ax.set_title("Step Duration")
    ax.set_xlabel("Step"); ax.set_ylabel("seconds/step")
    ax.set_xlim(XMIN, XMAX)
    if np.isfinite(avg_step_s):
        ax.axhline(avg_step_s, linestyle="--", alpha=0.8, label=f"avg={avg_step_s:.2f}s")
        ax.set_ylim(avg_step_s*0.9, avg_step_s*1.1)
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "avg_step_s not found", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()

    png = os.path.join(OUTDIR, f"{name}_clean.png")
    plt.savefig(png, bbox_inches="tight")
    plt.close(fig)

def main():
    for name, label, path in RUNS:
        plot_one(name, label, path, alpha=0.10)

if __name__ == "__main__":
    main()
