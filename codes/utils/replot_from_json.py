import os
import json
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ===================== SETTINGS =====================
Y_TICK_SIZE = 13          # y-axis tick NUMBER size
X_TICK_SIZE = 13         # x-axis tick NUMBER size
OUTPUT_DIR = "./plots_matplotlib"

RUNS = {
    "full_base": "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-full-finetuned/training_report_full.json",
    "full_instruct": "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-full-instruct/training_report_full.json",
    "lora_base": "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-lora-finetuned/training_report_lora.json",
    "lora_instruct": "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-lora-instruct/training_report_lora.json",
}

# ===================== HELPERS =====================
def bump_ticks(ax):
    # tick label sizes (the numbers)
    ax.tick_params(axis="y", labelsize=Y_TICK_SIZE)
    ax.tick_params(axis="x", labelsize=X_TICK_SIZE)


def pad_range(r, pct=0.05):
    lo, hi = r
    span = hi - lo
    if span <= 0:
        # degenerate case
        return (lo - 1.0, hi + 1.0)
    return (lo - pct * span, hi + pct * span)


# ===================== JSON LOADER =====================
def load_mcb_from_json(path: str):
    """
    Your training_report_*.json stores arrays under:
      d["metrics_log"]["steps", "losses", "learning_rates", "gradient_norms"]
    and epoch losses under:
      d["training_results"]["epoch_losses"]
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    m = d.get("metrics_log", {})  # <- key insight: metrics live here

    steps = m.get("steps", [])
    losses = m.get("losses", [])
    learning_rates = m.get("learning_rates", [])
    gradient_norms = m.get("gradient_norms", [])

    # epoch losses (optional)
    tr = d.get("training_results", {})
    epoch_losses = tr.get("epoch_losses", [])

    return SimpleNamespace(
        steps=steps,
        losses=losses,
        learning_rates=learning_rates,
        gradient_norms=gradient_norms,
        epoch_losses=epoch_losses,
    )


# ===================== GLOBAL RANGES =====================
def compute_global_ranges(runs: dict):
    all_losses = []
    all_grads = []

    for path in runs.values():
        mcb = load_mcb_from_json(path)

        if mcb.losses:
            all_losses.extend(mcb.losses)

        if mcb.gradient_norms:
            all_grads.extend(mcb.gradient_norms)

    if not all_losses:
        raise ValueError("No loss values found across runs. Check JSON keys/paths.")
    if not all_grads:
        raise ValueError("No gradient_norm values found across runs. Check JSON keys/paths.")

    loss_range = (min(all_losses), max(all_losses))
    grad_range = (min(all_grads), max(all_grads))

    # add a little padding for aesthetics
    loss_range = pad_range(loss_range, pct=0.05)
    grad_range = pad_range(grad_range, pct=0.05)

    return loss_range, grad_range


# ===================== PLOTTING =====================
def generate_training_plots(mcb, method, output_dir, loss_range, grad_range):
    """
    3-panel plot:
      1) Training Loss (fixed y-limits across runs)
      2) Learning Rate
      3) Gradient Norm (fixed y-limits across runs)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"GSM8K Finetuning — {method.upper()}", fontsize=16, fontweight="bold")

    # ---------- Panel 1: Training Loss ----------
    ax = axes[0]
    if mcb.losses:
        x = mcb.steps[:len(mcb.losses)] if mcb.steps else list(range(len(mcb.losses)))
        ax.plot(x, mcb.losses, alpha=0.25, label="Step loss")

        w = max(1, len(mcb.losses) // 20)
        if w > 1:
            smooth = np.convolve(mcb.losses, np.ones(w) / w, mode="valid")
            ax.plot(x[w - 1:w - 1 + len(smooth)], smooth, lw=2, label="Smoothed")

        ax.set_ylim(loss_range)
        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        bump_ticks(ax)

    # ---------- Panel 2: Learning Rate ----------
    ax = axes[1]
    if mcb.learning_rates:
        x = mcb.steps[:len(mcb.learning_rates)] if mcb.steps else list(range(len(mcb.learning_rates)))
        ax.plot(x, mcb.learning_rates, lw=2)
        ax.set_title("Learning Rate (Cosine)")
        ax.set_xlabel("Step")
        ax.set_ylabel("LR")
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        bump_ticks(ax)

    # ---------- Panel 3: Gradient Norm ----------
    ax = axes[2]
    if mcb.gradient_norms:
        x = mcb.steps[:len(mcb.gradient_norms)] if mcb.steps else list(range(len(mcb.gradient_norms)))
        ax.plot(x, mcb.gradient_norms, alpha=0.6, label="Grad norm")
        ax.axhline(np.mean(mcb.gradient_norms), linestyle="--", lw=2, label="Mean")

        ax.set_ylim(grad_range)
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("L2 Norm")
        ax.grid(True, alpha=0.3)
        ax.legend()
        bump_ticks(ax)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"training_metrics_{method}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)
    return path


def generate_epoch_plot(mcb, method, output_dir):
    if not mcb.epoch_losses:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(mcb.epoch_losses) + 1)

    ax.bar(epochs, mcb.epoch_losses, edgecolor="black")
    ax.set_title(f"Epoch Loss — {method.upper()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Loss")
    ax.grid(True, axis="y", alpha=0.3)
    bump_ticks(ax)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epoch_loss_{method}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)
    return path


# ===================== MAIN =====================
def main():
    loss_range, grad_range = compute_global_ranges(RUNS)

    for method, path in RUNS.items():
        mcb = load_mcb_from_json(path)
        generate_training_plots(mcb, method, OUTPUT_DIR, loss_range, grad_range)
        generate_epoch_plot(mcb, method, OUTPUT_DIR)


if __name__ == "__main__":
    main()
