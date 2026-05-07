#!/usr/bin/env python3
"""
**V & O only** (no k/q): rel. L2 vs pretrained per layer, plus mean V+O **step %** into each epoch.

For each training epoch row, ``mean_vo_pct_step_vs_prev`` is
``100 * mean(||W_next-W_prev||/||W_prev||)`` over layers × {{v,o}} for that transition
(baseline→1, 1→2, … 5→6), repeated on every layer/proj row for that ``epoch_index``.

**Per family, one PNG + one CSV:**
  • Llama — full + norm-freeze ``run1`` (11 experiments; LoRA skipped)
  • Qwen-base — full + norm-freeze ``run3`` (11 experiments; LoRA skipped)

**PNG:** grid — **each row** one experiment, **columns** V | O (layer on x, curves = epochs).

**Default outputs** (override with ``--out-dir``): ``eval/split/weight_vs_baseline/norm/``

  python scripts/norm_llama_full_layer_epoch_curves.py --nw-root .
      (same as --sweep-llama when --finetune-run-dir is left at default)
  python scripts/norm_llama_full_layer_epoch_curves.py --nw-root . --sweep-llama
  python scripts/norm_llama_full_layer_epoch_curves.py --nw-root . --sweep-qwen-base
  python scripts/norm_llama_full_layer_epoch_curves.py --nw-root . --sweep-all
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file as safetensors_load_file

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
PROJ_PATTERNS = {
    "k": ".k_proj.",
    "q": ".q_proj.",
    "v": ".v_proj.",
    "o": ".o_proj.",
}
VO_PROJS = frozenset({"v", "o"})
VO_ORDER = ("v", "o")

_PRETRAINED_SUBDIR = {"llama": "llama31-8b", "qwen-base": "qwen3-8b-base", "mistral-7b-v03": "mistral-7b-v03"}
NORM_KS: Sequence[int] = (3, 6, 9, 12, 15)

RUN_ID = {"llama": "run1", "qwen-base": "run3", "mistral-7b-v03": "run1"}


def layer_index(param_name: str) -> Optional[int]:
    m = LAYER_RE.search(param_name)
    return int(m.group(1)) if m else None


def projection_type(param_name: str) -> Optional[str]:
    for proj, pattern in PROJ_PATTERNS.items():
        if pattern in param_name:
            return proj
    return None


def load_weight_map(folder: str) -> Dict[str, str]:
    index_path = os.path.join(folder, "model.safetensors.index.json")
    single_path = os.path.join(folder, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        wm = idx.get("weight_map")
        if not wm:
            raise RuntimeError(f"No weight_map in {index_path}")
        return wm
    if os.path.isfile(single_path):
        sd = safetensors_load_file(single_path, device="cpu")
        return {k: "model.safetensors" for k in sd.keys()}
    raise RuntimeError(f"No safetensors weights found in: {folder}")


class ShardCache:
    def __init__(self, folder: str, max_cached: int = 4):
        self.folder = folder
        self.max_cached = max_cached
        self.cache: Dict[str, dict] = {}
        self.order: List[str] = []

    def get(self, shard: str) -> dict:
        if shard in self.cache:
            self.order.remove(shard)
            self.order.append(shard)
            return self.cache[shard]
        path = os.path.join(self.folder, shard)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Shard not found: {path}")
        sd = safetensors_load_file(path, device="cpu")
        self.cache[shard] = sd
        self.order.append(shard)
        while len(self.order) > self.max_cached:
            del self.cache[self.order.pop(0)]
        return sd


def _has_weights(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "model.safetensors.index.json")) or os.path.isfile(
        os.path.join(d, "model.safetensors")
    )


def _has_adapter_only_checkpoint(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "adapter_model.safetensors")) and not _has_weights(d)


def discover_epoch_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    cps = sorted(glob.glob(str(run_dir / "checkpoint-*")))
    parsed: List[Tuple[int, Path]] = []
    for p in cps:
        step = int(Path(p).name.split("-")[-1])
        parsed.append((step, Path(p)))
    if not parsed:
        return []
    parsed.sort(key=lambda x: x[0])
    gap = parsed[0][0]
    return [(s // gap, Path(p)) for s, p in parsed]


def pick_pretrained_dir(family: str, override: Optional[str], nw_root: Path) -> Optional[str]:
    if override and os.path.isdir(override) and _has_weights(override):
        return override.rstrip("/")
    sub = _PRETRAINED_SUBDIR.get(family)
    if not sub:
        return None
    p = (nw_root / "pretrained" / sub).resolve()
    if p.is_dir() and _has_weights(str(p)):
        return str(p).rstrip("/")
    return None


def sweep_specs_for_family(family: str, ckpt_root: Path) -> List[Tuple[str, str, Path]]:
    rid = RUN_ID[family]
    root = ckpt_root / family
    rows: List[Tuple[str, str, Path]] = [
        ("full", "full finetune", root / "full" / rid),
    ]
    for k in NORM_KS:
        for side in ("high", "low"):
            tag = f"norm_{side}_{k}"
            rows.append((tag, f"norm-freeze {side}-{k}", root / "norm-freeze" / f"{side}-{k}" / rid))
    return rows


def infer_experiment_tag(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    if "norm-freeze" in parts:
        i = parts.index("norm-freeze")
        variant = parts[i + 1]
        return f"norm_{variant.replace('-', '_')}"
    if "full" in parts:
        return "full"
    return f"{run_dir.parent.name}_{run_dir.name}"


def experiment_title(file_tag: str) -> str:
    if file_tag == "full":
        return "full finetune"
    m = re.fullmatch(r"norm_(high|low)_(\d+)", file_tag)
    if m:
        return f"norm-freeze {m.group(1)}-{m.group(2)}"
    return file_tag


def normalized_l2_rel(w_prev: torch.Tensor, w_next: torch.Tensor, eps: float = 1e-12) -> float:
    p = w_prev.detach().float().reshape(-1)
    n = w_next.detach().float().reshape(-1)
    return (torch.linalg.norm(n - p) / (torch.linalg.norm(p) + eps)).item()


def collect_vo_vs_baseline_by_layer_epoch(
    baseline_dir: str,
    epoch_checkpoints: List[Tuple[int, Path]],
) -> Dict[int, Dict[str, Dict[int, float]]]:
    map_b = load_weight_map(baseline_dir)
    cache_b = ShardCache(baseline_dir)
    data: Dict[int, Dict[str, Dict[int, float]]] = {}

    for epoch_idx, ckpt_path in epoch_checkpoints:
        wdir = str(ckpt_path)
        if not _has_weights(wdir):
            continue
        map_e = load_weight_map(wdir)
        cache_e = ShardCache(wdir)
        common = sorted(set(map_b.keys()).intersection(map_e.keys()))
        ep_data: Dict[str, Dict[int, float]] = {}

        for name in common:
            proj = projection_type(name)
            if proj not in VO_PROJS:
                continue
            li = layer_index(name)
            if li is None:
                continue
            try:
                wb = cache_b.get(map_b[name])[name].to(torch.float32)
                we = cache_e.get(map_e[name])[name].to(torch.float32)
            except (FileNotFoundError, KeyError):
                continue
            if wb.shape != we.shape:
                continue
            ep_data.setdefault(proj, {})[li] = normalized_l2_rel(wb, we)

        if ep_data:
            data[epoch_idx] = ep_data

    return data


def epoch_index_to_step(cps: List[Tuple[int, Path]]) -> Dict[int, int]:
    return {ep: int(p.name.split("-")[-1]) for ep, p in cps}


def mean_vo_rel_step_pct(prev_dir: str, next_dir: str) -> Tuple[float, int]:
    map_p = load_weight_map(prev_dir)
    map_n = load_weight_map(next_dir)
    cache_p = ShardCache(prev_dir)
    cache_n = ShardCache(next_dir)
    common = sorted(set(map_p.keys()).intersection(map_n.keys()))
    vals: List[float] = []
    for name in common:
        if projection_type(name) not in VO_PROJS:
            continue
        if layer_index(name) is None:
            continue
        try:
            wp = cache_p.get(map_p[name])[name].to(torch.float32)
            wn = cache_n.get(map_n[name])[name].to(torch.float32)
        except (FileNotFoundError, KeyError):
            continue
        if wp.shape != wn.shape:
            continue
        vals.append(normalized_l2_rel(wp, wn))
    if not vals:
        return float("nan"), 0
    return 100.0 * float(np.mean(vals)), len(vals)


def transition_label_for_index(i: int) -> str:
    if i == 0:
        return "baseline_to_epoch1"
    return f"epoch{i}_to_epoch{i + 1}"


def summarize_experiment_transitions(
    baseline_dir: str,
    epoch_checkpoints: List[Tuple[int, Path]],
) -> List[Dict[str, object]]:
    cps_sorted = sorted(epoch_checkpoints, key=lambda x: x[0])
    dirs: List[str] = [baseline_dir] + [str(p) for _, p in cps_sorted]
    rows: List[Dict[str, object]] = []
    for i in range(len(dirs) - 1):
        pct, n_t = mean_vo_rel_step_pct(dirs[i], dirs[i + 1])
        rows.append(
            {
                "transition": transition_label_for_index(i),
                "mean_pct_relative_l2": pct,
                "n_tensors_vo": n_t,
            }
        )
    return rows


def layer_rows_for_experiment(
    family: str,
    experiment: str,
    data: Dict[int, Dict[str, Dict[int, float]]],
    ep_to_step: Dict[int, int],
    trans: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Per-layer rows with transition columns for this experiment."""
    ep_to_step_pct: Dict[int, Tuple[str, str]] = {}
    for i, t in enumerate(trans):
        ep = i + 1
        lbl = str(t["transition"])
        mp = t["mean_pct_relative_l2"]
        pct_s = "" if mp != mp else f"{float(mp):.8g}"
        ep_to_step_pct[ep] = (lbl, pct_s)

    rows: List[Dict[str, object]] = []
    for ep in sorted(data.keys()):
        step = ep_to_step.get(ep, -1)
        tl, pst = ep_to_step_pct.get(ep, ("", ""))
        for proj in VO_ORDER:
            by_layer = data[ep].get(proj, {})
            for layer in sorted(by_layer.keys()):
                rows.append(
                    {
                        "family": family,
                        "experiment": experiment,
                        "epoch_index": ep,
                        "checkpoint_step": step,
                        "transition_into_epoch": tl,
                        "mean_vo_pct_step_vs_prev": pst,
                        "layer": layer,
                        "proj": proj,
                        "norm_l2_rel_vs_baseline": by_layer[layer],
                    }
                )
    return rows


def write_facets_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = [
        "family",
        "experiment",
        "epoch_index",
        "checkpoint_step",
        "transition_into_epoch",
        "mean_vo_pct_step_vs_prev",
        "layer",
        "proj",
        "norm_l2_rel_vs_baseline",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "family": r["family"],
                    "experiment": r["experiment"],
                    "epoch_index": r["epoch_index"],
                    "checkpoint_step": r["checkpoint_step"],
                    "transition_into_epoch": r["transition_into_epoch"],
                    "mean_vo_pct_step_vs_prev": r["mean_vo_pct_step_vs_prev"],
                    "layer": r["layer"],
                    "proj": r["proj"],
                    "norm_l2_rel_vs_baseline": f"{float(r['norm_l2_rel_vs_baseline']):.8g}",
                }
            )


def plot_vo_faceted_grid(
    out_png: Path,
    panels: List[Tuple[str, Dict[int, Dict[str, Dict[int, float]]]]],
    suptitle: str,
) -> None:
    """Each row = experiment tag; cols = v, o."""
    n = len(panels)
    if n == 0:
        raise ValueError("no panels")
    all_epochs = sorted({e for _, d in panels for e in d})
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, max(len(all_epochs), 1)))
    epoch_color = {ep: colors[j] for j, ep in enumerate(all_epochs)}

    fig_h = max(2.0 * n, 3.0)
    fig, axes = plt.subplots(n, 2, figsize=(11, fig_h), sharex=True, squeeze=False)

    for row, (label, data) in enumerate(panels):
        epochs = sorted(data.keys())
        for col, proj in enumerate(VO_ORDER):
            ax = axes[row][col]
            for ep in epochs:
                by_layer = data[ep].get(proj)
                if not by_layer:
                    continue
                xs = sorted(by_layer.keys())
                ys = [by_layer[x] for x in xs]
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    ms=2,
                    lw=0.85,
                    label=f"ep{ep}",
                    color=epoch_color[ep],
                )
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(label.replace("_", " "), fontsize=8)
            if row == 0:
                ax.set_title(f"{proj.upper()} vs baseline", fontsize=9)
            if row == n - 1:
                ax.set_xlabel("Layer", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(6, len(all_epochs)),
        fontsize=7,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_experiment(
    baseline: str,
    run_dir: Path,
    experiment: str,
    family: str,
) -> Optional[Tuple[List[Dict[str, object]], Dict[int, Dict[str, Dict[int, float]]]]]:
    cps = discover_epoch_checkpoints(run_dir)
    if not cps:
        print(f"[skip] no checkpoint-* under {run_dir}", flush=True)
        return None
    sample = str(cps[0][1])
    if _has_adapter_only_checkpoint(sample):
        print(f"[skip] {experiment}: PEFT adapter-only. {run_dir}", flush=True)
        return None

    print(f"[{family}/{experiment}] {run_dir}  ({len(cps)} epochs)", flush=True)

    layer_data = collect_vo_vs_baseline_by_layer_epoch(baseline, cps)
    if not layer_data:
        print(f"[skip] no v/o tensors: {run_dir}", flush=True)
        return None

    trans = summarize_experiment_transitions(baseline, cps)
    if not trans or all(t["n_tensors_vo"] == 0 for t in trans):
        print(f"[skip] no v/o transitions: {run_dir}", flush=True)
        return None

    ep_to_step = epoch_index_to_step(cps)
    rows = layer_rows_for_experiment(family, experiment, layer_data, ep_to_step, trans)
    return rows, layer_data


def run_family_sweep(
    family: str,
    ckpt_root: Path,
    baseline: str,
    out_csv: Path,
    out_png: Path,
) -> None:
    specs = sweep_specs_for_family(family, ckpt_root)
    all_rows: List[Dict[str, object]] = []
    plot_panels: List[Tuple[str, Dict[int, Dict[str, Dict[int, float]]]]] = []

    for tag, _title_sfx, run_dir in specs:
        if not run_dir.is_dir():
            print(f"[skip] missing dir: {run_dir}", flush=True)
            continue
        out = process_experiment(baseline, run_dir, tag, family)
        if not out:
            continue
        rows, layer_data = out
        all_rows.extend(rows)
        plot_panels.append((tag, layer_data))

    if not all_rows:
        raise SystemExit(f"No data for family={family}")

    write_facets_csv(out_csv, all_rows)
    plot_vo_faceted_grid(
        out_png,
        plot_panels,
        suptitle=f"{family} — V/O rel. L2 vs pretrained (rows = experiments)",
    )
    print(f"Wrote {out_csv}  ({len(all_rows)} rows)", flush=True)
    print(f"Wrote {out_png}", flush=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    nw_root = script_dir.parent
    default_run = nw_root / "checkpoints" / "llama" / "full" / "run1"
    default_out_dir = nw_root / "eval" / "split" / "weight_vs_baseline" / "norm"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nw-root", type=Path, default=nw_root, help="exploration-finetuning root")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory for facet CSV/PNG (default: eval/split/weight_vs_baseline/norm)",
    )
    ap.add_argument("--ckpt-root", type=Path, default=None, help="Checkpoint root (default: NW_ROOT/checkpoints)")
    ap.add_argument("--sweep-llama", action="store_true", help="One PNG + one CSV for Llama (11 experiments).")
    ap.add_argument("--sweep-qwen-base", action="store_true", help="One PNG + one CSV for Qwen-base (11 experiments).")
    ap.add_argument(
        "--sweep-all",
        action="store_true",
        help="Both families (2 PNG + 2 CSV).",
    )
    ap.add_argument(
        "--sweep-all-llama",
        action="store_true",
        help="Deprecated: same as --sweep-llama.",
    )
    ap.add_argument(
        "--finetune-run-dir",
        type=Path,
        default=default_run,
        help="Single-run: one experiment (use with no sweep flags).",
    )
    ap.add_argument("--baseline-llama", type=str, default=None)
    ap.add_argument("--baseline-qwen-base", type=str, default=None)
    ap.add_argument("--out-csv", type=Path, default=None, help="Single-run CSV")
    ap.add_argument("--out-png", type=Path, default=None, help="Single-run PNG")
    args = ap.parse_args()

    nw = args.nw_root.resolve()
    ckpt_root = (args.ckpt_root or (nw / "checkpoints")).resolve()
    out_dir = args.out_dir.resolve()

    baseline_overrides = {
        "llama": args.baseline_llama,
        "qwen-base": args.baseline_qwen_base,
    }

    sweep_fams: List[str] = []
    if args.sweep_all:
        sweep_fams = ["llama", "qwen-base"]
    else:
        if args.sweep_llama or args.sweep_all_llama:
            sweep_fams.append("llama")
        if args.sweep_qwen_base:
            sweep_fams.append("qwen-base")

    finetune_arg = args.finetune_run_dir.resolve()

    if not sweep_fams and finetune_arg == default_run.resolve():
        sweep_fams = ["llama"]

    if sweep_fams:
        for fam in sweep_fams:
            bl = pick_pretrained_dir(fam, baseline_overrides.get(fam), nw)
            if not bl:
                print(f"[skip] no baseline for {fam}", flush=True)
                continue
            print(f"[baseline {fam}] {bl}", flush=True)
            stem = "norm_llama_vo_facets" if fam == "llama" else "norm_qwen_base_vo_facets"
            run_family_sweep(fam, ckpt_root, bl, out_dir / f"{stem}.csv", out_dir / f"{stem}.png")
        return

    fam = "qwen-base" if "qwen-base" in finetune_arg.parts else "llama"
    baseline = pick_pretrained_dir(fam, baseline_overrides.get(fam), nw)
    if not baseline:
        raise SystemExit(f"No baseline for {fam}")
    print(f"[baseline {fam}] {baseline}", flush=True)
    exp_tag = infer_experiment_tag(finetune_arg)
    out = process_experiment(baseline, finetune_arg, exp_tag, fam)
    if not out:
        raise SystemExit("No data.")
    rows, layer_data = out
    out_csv = (args.out_csv or (out_dir / f"norm_{fam}_vo_single.csv")).resolve()
    out_png = (args.out_png or (out_dir / f"norm_{fam}_vo_single.png")).resolve()
    write_facets_csv(out_csv, rows)
    plot_vo_faceted_grid(out_png, [(exp_tag, layer_data)], suptitle=f"{fam} — {experiment_title(exp_tag)}")
    print(f"Wrote {out_csv}  ({len(rows)} rows)", flush=True)
    print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
