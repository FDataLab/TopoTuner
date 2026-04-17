#!/usr/bin/env python3
"""
Aggregate per-experiment Wasserstein CSVs (V/O, H0-only) into norm-style facet outputs.

Outputs (default under eval/split/weight_vs_baseline/wass):
  - wass_llama_vo_facets.csv
  - wass_llama_vo_facets.png
  - wass_qwen_base_vo_facets.csv
  - wass_qwen_base_vo_facets.png
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VO_ORDER = ("v", "o")
LAYER_FILE_RE = re.compile(r"layer(\d+)_([a-z]+)\.pkl$")

# Same tag order as scripts/run_wass_vo_11x11.sh (not norm_* sweep tags).
WASS_TAG_ORDER: Tuple[str, ...] = (
    "full",
    "wass_high_3",
    "wass_low_3",
    "wass_high_6",
    "wass_low_6",
    "wass_high_9",
    "wass_low_9",
    "wass_high_12",
    "wass_low_12",
    "wass_high_15",
    "wass_low_15",
)


def wass_experiment_tags(in_root: Path, family: str) -> List[str]:
    fam_dir = in_root / family
    if not fam_dir.is_dir():
        return []
    present = {
        p.name
        for p in fam_dir.iterdir()
        if p.is_dir()
        and p.name != "baseline"
        and (p / "wasserstein_results.csv").is_file()
    }
    ordered = [t for t in WASS_TAG_ORDER if t in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def parse_epoch_index(epoch_s: str) -> int:
    m = re.fullmatch(r"epoch_(\d+)", epoch_s.strip())
    if not m:
        raise ValueError(f"Bad epoch label: {epoch_s!r}")
    return int(m.group(1))


def parse_layer_proj(file_s: str) -> Tuple[int, str]:
    m = LAYER_FILE_RE.fullmatch(file_s.strip())
    if not m:
        raise ValueError(f"Bad File field: {file_s!r}")
    return int(m.group(1)), m.group(2)


def read_experiment_csv(path: Path) -> Tuple[List[Dict[str, object]], Dict[int, Dict[str, Dict[int, float]]]]:
    rows: List[Dict[str, object]] = []
    panel: Dict[int, Dict[str, Dict[int, float]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            proj = str(r.get("Projection", "")).strip().lower()
            if proj not in VO_ORDER:
                continue
            ep = parse_epoch_index(str(r["Epoch"]))
            li, proj_from_file = parse_layer_proj(str(r["File"]))
            if proj_from_file != proj:
                proj = proj_from_file
                if proj not in VO_ORDER:
                    continue
            val = float(r["Wasserstein H0"])
            rows.append(
                {
                    "epoch_index": ep,
                    "layer": li,
                    "proj": proj,
                    "wasserstein_h0_vs_baseline": val,
                }
            )
            panel.setdefault(ep, {}).setdefault(proj, {})[li] = val
    return rows, panel


def write_family_csv(path: Path, family: str, exp_rows: List[Tuple[str, List[Dict[str, object]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = ["family", "experiment", "epoch_index", "layer", "proj", "wasserstein_h0_vs_baseline"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for tag, rows in exp_rows:
            for r in rows:
                w.writerow(
                    {
                        "family": family,
                        "experiment": tag,
                        "epoch_index": r["epoch_index"],
                        "layer": r["layer"],
                        "proj": r["proj"],
                        "wasserstein_h0_vs_baseline": f"{float(r['wasserstein_h0_vs_baseline']):.8g}",
                    }
                )


def plot_facets(path: Path, family: str, panels: List[Tuple[str, Dict[int, Dict[str, Dict[int, float]]]]]) -> None:
    if not panels:
        raise SystemExit(f"No panel data for {family}")
    n = len(panels)
    all_epochs = sorted({e for _, d in panels for e in d})
    # Distinct from norm plots (norm uses viridis); warm purple→yellow for epochs.
    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.12, 0.92, max(len(all_epochs), 1)))
    epoch_color = {ep: colors[j] for j, ep in enumerate(all_epochs)}

    fig, axes = plt.subplots(n, 2, figsize=(11, max(2.0 * n, 3.0)), sharex=True, squeeze=False)
    for row, (label, data) in enumerate(panels):
        for col, proj in enumerate(VO_ORDER):
            ax = axes[row][col]
            for ep in sorted(data.keys()):
                by_layer = data[ep].get(proj)
                if not by_layer:
                    continue
                xs = sorted(by_layer.keys())
                ys = [by_layer[x] for x in xs]
                ax.plot(xs, ys, "o-", ms=2, lw=0.85, label=f"ep{ep}", color=epoch_color[ep])
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(label.replace("_", " "), fontsize=8)
            if row == 0:
                ax.set_title(f"{proj.upper()} vs baseline", fontsize=9)
            if row == n - 1:
                ax.set_xlabel("Layer", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(all_epochs)), fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{family} — V/O Wasserstein H0 vs baseline (rows = experiments)", fontsize=11, y=1.02)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_family(family: str, in_root: Path, out_dir: Path) -> None:
    exp_rows: List[Tuple[str, List[Dict[str, object]]]] = []
    panels: List[Tuple[str, Dict[int, Dict[str, Dict[int, float]]]]] = []
    for tag in wass_experiment_tags(in_root, family):
        p = in_root / family / tag / "wasserstein_results.csv"
        if not p.is_file():
            print(f"[skip] missing {p}", flush=True)
            continue
        rows, panel = read_experiment_csv(p)
        if not rows:
            print(f"[skip] no v/o rows in {p}", flush=True)
            continue
        exp_rows.append((tag, rows))
        panels.append((tag, panel))

    if not exp_rows:
        raise SystemExit(f"No CSVs found for family={family} in {in_root / family}")

    stem = "wass_llama_vo_facets" if family == "llama" else "wass_qwen_base_vo_facets"
    out_csv = out_dir / f"{stem}.csv"
    out_png = out_dir / f"{stem}.png"
    write_family_csv(out_csv, family, exp_rows)
    plot_facets(out_png, family, panels)
    print(f"Wrote {out_csv}", flush=True)
    print(f"Wrote {out_png}", flush=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    nw_default = script_dir.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--nw-root", type=Path, default=nw_default, help="Repo root (used only for default --in-root).")
    ap.add_argument("--in-root", type=Path, default=None, help="Default: NW_ROOT/eval/split/weight_vs_baseline/wass")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: same as --in-root")
    ap.add_argument("--families", choices=("llama", "qwen-base", "all"), default="all")
    args = ap.parse_args()

    nw = args.nw_root.resolve()
    in_root = (args.in_root or (nw / "eval" / "split" / "weight_vs_baseline" / "wass")).resolve()
    out_dir = (args.out_dir or in_root).resolve()

    fams = ["llama", "qwen-base"] if args.families == "all" else [args.families]
    for fam in fams:
        run_family(fam, in_root, out_dir)


if __name__ == "__main__":
    main()
