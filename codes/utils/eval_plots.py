import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt


DATASET_KEYS = ["mmlu", "imdb", "gsm8k", "sst2"]


def to_percent(acc_value, scale_factor=None):
    """
    Normalize accuracy values to percent scale.
    Rules (based on project CSVs):
    - Values like 0.65 should be read as 0.65% (NOT fraction). → percent = 0.65
    - Values like 57 or 57.0 mean 57% → percent = 57
    Therefore: treat everything as percent already if 0 <= x <= 100.
    If x > 100, ignore. If x < 0, ignore.
    """
    try:
        x = float(acc_value)
    except Exception:
        return None
    if scale_factor is not None:
        x = x * scale_factor
    if x < 0 or x > 100:
        return None
    return x


def infer_epoch(checkpoint_name: str) -> int:
    name = str(checkpoint_name)
    # expect 'checkpoint-epoch-<num>'
    if "epoch" in name:
        for part in name.replace("_", "-").split("-"):
            if part.isdigit():
                return int(part)
    return -1


def infer_dataset_from_filename(path: str) -> str:
    base = os.path.basename(path).lower()
    for key in DATASET_KEYS:
        if key in base:
            return key
    return "misc"


def load_points(csv_path: str):
    df = pd.read_csv(csv_path)
    if "checkpoint" not in df.columns or "acc" not in df.columns:
        return []
    # detect scaling based on max acc in the file
    max_v = df['acc'].max() if not df.empty else None
    scale = 100.0 if max_v is not None and max_v <= 1.0 else None
    points = []
    for _, row in df.iterrows():
        ep = infer_epoch(row["checkpoint"])
        pct = to_percent(row["acc"], scale_factor=scale)
        if ep >= 0 and pct is not None:
            points.append((ep, pct))
    points.sort(key=lambda x: x[0])
    return points


def plot_csv(csv_path: str, out_root: str, ymin: float = 0.0, ymax: float = 100.0):
    pts = load_points(csv_path)
    if not pts:
        print(f"[skip] No valid data in {csv_path}")
        return None

    epochs = [e for e, _ in pts]
    accs_pct = [p for _, p in pts]

    plt.figure(figsize=(7, 4))
    # line + markers
    color = "#1f77b4"
    plt.plot(epochs, accs_pct, marker='o', linewidth=1.8, color=color)

    # annotate only the max value using the same line color
    if accs_pct:
        max_idx = max(range(len(accs_pct)), key=lambda i: accs_pct[i])
        max_e, max_a = epochs[max_idx], accs_pct[max_idx]
        plt.scatter([max_e], [max_a], s=80, color=color, edgecolors="black", zorder=3)
        plt.annotate(
            f"{max_a:.2f}", (max_e, max_a), textcoords="offset points", xytext=(0, -16),
            ha="center", fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
    plt.xticks(epochs)
    plt.ylim(ymin, ymax)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    title = os.path.splitext(os.path.basename(csv_path))[0].replace("_", " ")
    plt.title(title)

    dataset = infer_dataset_from_filename(csv_path)
    out_dir = os.path.join(out_root, dataset, "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(csv_path))[0]}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[ok] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/*.csv")
    ap.add_argument("--out-root", default="plots")
    ap.add_argument("--ymin", type=float, default=0.0)
    ap.add_argument("--ymax", type=float, default=100.0)
    ap.add_argument("--combine", action="store_true", help="Combine FULL and LORA eval results into one plot per model (for ANY dataset)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched: {args.pattern}")
        return

    if args.combine:
        # Group by dataset and model prefix, look for pairs of *_full and *_lora for each dataset/model
        from collections import defaultdict
        grouped = defaultdict(dict)  # grouped[(dataset, model)][series] = csv_path
        for p in files:
            base = os.path.basename(p)
            dataset = infer_dataset_from_filename(base)
            parts = base.split('_')
            # Format: <dataset>_<model>_<style>.csv (IMDB and GSM8K)
            # or: <model>_mmlu_<style>.csv (old MMLU)
            if dataset != 'misc' and len(parts) >= 3:
                if parts[0] in DATASET_KEYS:
                    model = '_'.join(parts[1:-1])
                    style = parts[-1].replace('.csv','')
                    if style in ('full','lora'):
                        grouped[(dataset, model)][style] = p
            elif "mmlu" in base.lower() and len(parts) >= 3:
                model = base.split('_mmlu_')[0]
                style = 'full' if '_mmlu_full' in base else ('lora' if '_mmlu_lora' in base else None)
                if style:
                    grouped(('mmlu',model))[style] = p
        for (dataset, model), d in grouped.items():
            series = []
            for label in ["full","lora"]:
                if label in d:
                    pts = load_points(d[label])
                    if pts:
                        epochs = [e for e, _ in pts]
                        accs = [a for _, a in pts]
                        series.append((label, epochs, accs))
            if not series:
                continue
            plt.figure(figsize=(7, 4))
            colors = {"full": "#1f77b4", "lora": "#ff7f0e"}
            all_epochs = set()
            for label, epochs, accs in series:
                all_epochs.update(epochs)
                c = colors.get(label, None)
                plt.plot(epochs, accs, marker='o', linewidth=1.8, label=label, color=c)
                if accs:
                    s_idx = max(range(len(accs)), key=lambda i: accs[i])
                    se, sa = epochs[s_idx], accs[s_idx]
                    plt.scatter([se], [sa], s=80, color=c, edgecolors="black", zorder=3)
                    voff = -16 if label == "full" else 10
                    plt.annotate(
                        f"{sa:.2f}", (se, sa), textcoords="offset points", xytext=(0, voff),
                        ha="center", fontsize=11, fontweight="bold", color=c,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    )
            plt.xticks(sorted(list(all_epochs)))
            plt.ylim(args.ymin, args.ymax)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title(f"{model} {dataset.upper()} (full vs lora)")
            plt.legend()
            out_dir = os.path.join(args.out_root, dataset, "eval")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{dataset}_{model}_combined.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[ok] {out_path}")
    else:
        for path in files:
            try:
                plot_csv(path, args.out_root, ymin=args.ymin, ymax=args.ymax)
            except Exception as e:
                print(f"[err] {path}: {e}")


if __name__ == "__main__":
    main()


