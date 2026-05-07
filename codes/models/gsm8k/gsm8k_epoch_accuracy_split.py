#!/usr/bin/env python3
"""
GSM8K accuracy vs epoch (0 = pretrained baseline, 1–6 = checkpoint-* steps)
for the fixed 10-experiment “split” eval layout (separate from llama(PLANS-AND-NO-SPLIT)).

Reuses evaluate_model / prompts from topo/codes/gsm8k/eval_epoch_accuracy.py.

Outputs (under NW_ROOT/eval/split/gsm8k/ by default):
  llama/json/<slug>_epoch_accuracy.json
  qwen-base/json/<slug>_epoch_accuracy.json
  mistral-7b-v03/json/<slug>_epoch_accuracy.json
  llama/plots/epoch_accuracy_llama.png
  qwen-base/plots/epoch_accuracy_qwen-base.png
  mistral-7b-v03/plots/epoch_accuracy_mistral-7b-v03.png

TDA / norm distances are intentionally not here (CPU-only pipeline, run separately).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- paths ---
SCRIPT_PATH = Path(__file__).resolve()
NW_ROOT = Path(os.environ.get("NW_ROOT", SCRIPT_PATH.parent.parent)).resolve()


def _effective_topo_root() -> Path:
    """
    Resolve a repo root that contains codes/gsm8k/eval_epoch_accuracy.py.
    Prefer explicit env paths, then infer from NW_ROOT.
    """
    explicit = os.environ.get("GSM8K_EEA_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.is_file():
            return p.parents[2]  # .../codes/gsm8k/eval_epoch_accuracy.py -> topo root
    inferred_topo = NW_ROOT.parents[2] if len(NW_ROOT.parents) >= 3 else NW_ROOT
    candidates = [
        Path(os.environ.get("TOPO_ROOT", "")).expanduser().resolve()
        if os.environ.get("TOPO_ROOT")
        else None,
        inferred_topo,
    ]
    gsm8k_rel = Path("codes") / "gsm8k" / "eval_epoch_accuracy.py"
    for c in candidates:
        if c is None:
            continue
        try:
            cand = c.expanduser().resolve()
        except OSError:
            continue
        if (cand / gsm8k_rel).is_file():
            return cand
    return inferred_topo


# Resolve from env/repo structure; avoid machine-specific absolute fallbacks.
TOPO_ROOT = _effective_topo_root()
GSM8K_EEA = TOPO_ROOT / "codes" / "gsm8k" / "eval_epoch_accuracy.py"


def _resolve_gsm8k_eval_epoch_accuracy_path() -> Path:
    candidates = [
        Path(os.environ.get("GSM8K_EEA_PATH", "").strip()),
        GSM8K_EEA,
    ]
    for c in candidates:
        if c and c.parts and c.is_file():
            return c
    return GSM8K_EEA

DEFAULT_CKPT = Path(
    os.environ.get(
        "CHECKPOINT_NW_ROOT",
        str((NW_ROOT / "checkpoints").resolve()),
    )
).resolve()

LLAMA_BASE = "meta-llama/Llama-3.1-8B"
QWEN_BASE = "Qwen/Qwen3-8B-Base"

# (slug, relpath from CHECKPOINT_ROOT, hf_base_id)
LLAMA_EXPERIMENTS: List[Tuple[str, str, str]] = [
    ("full", "llama/full/run1", LLAMA_BASE),
    ("lora", "llama/lora/run1", LLAMA_BASE),
    ("wass-high6", "llama/wass-freeze/high-6-frozen/run1", LLAMA_BASE),
    ("norm-high6", "llama/norm-freeze/high-6/run1", LLAMA_BASE),
    ("norm-high9", "llama/norm-freeze/high-9/run1", LLAMA_BASE),
]

QWEN_EXPERIMENTS: List[Tuple[str, str, str]] = [
    ("full", "qwen-base/full/run3", QWEN_BASE),
    ("lora", "qwen-base/lora/run3", QWEN_BASE),
    ("wass-high3", "qwen-base/wass-freeze/high-3/run3", QWEN_BASE),
    ("norm-low3", "qwen-base/norm-freeze/low-3/run3", QWEN_BASE),
    ("norm-low15", "qwen-base/norm-freeze/low-15/run3", QWEN_BASE),
]

MISTRAL_BASE = "mistralai/Mistral-7B-v0.3"

# Family-root GSM8K checkpoints (same layout as qwen): full/lora + wass-freeze/high-3 for plot_wass_full_by_task.
MISTRAL_EXPERIMENTS: List[Tuple[str, str, str]] = [
    ("full", "mistral-7b-v03/full/run1", MISTRAL_BASE),
    ("lora", "mistral-7b-v03/lora/run1", MISTRAL_BASE),
    ("wass-high3", "mistral-7b-v03/wass-freeze/high-3/run1", MISTRAL_BASE),
]


def _load_eval_epoch_accuracy_module():
    path = _resolve_gsm8k_eval_epoch_accuracy_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing eval_epoch_accuracy.py (set GSM8K_EEA_PATH or TOPO_ROOT). Tried: {path}"
        )
    spec = importlib.util.spec_from_file_location("eval_epoch_accuracy", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_epoch_accuracy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _discover_epoch_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    import glob

    cps = sorted(glob.glob(str(run_dir / "checkpoint-*")))
    parsed = []
    for p in cps:
        step = int(Path(p).name.split("-")[-1])
        parsed.append((step, Path(p)))
    if not parsed:
        return []
    parsed.sort(key=lambda x: x[0])
    gap = parsed[0][0]
    return [(s // gap, Path(p)) for s, p in parsed]


def _load_model_from_checkpoint(ckpt_path: Path, base_model_id: str):
    adapter = ckpt_path / "adapter_config.json"
    if adapter.is_file():
        from peft import PeftModel

        with open(adapter) as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path", base_model_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path), torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return model


_baseline_acc_cache: Dict[str, float] = {}


def _eval_baseline(
    base_model_id: str,
    tokenizer,
    test_data,
    batch_size: int,
    max_new_tokens: int,
    eea,
) -> float:
    if base_model_id in _baseline_acc_cache:
        return _baseline_acc_cache[base_model_id]
    print(f"  [epoch 0] Loading pretrained baseline: {base_model_id}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    acc, _, _ = eea.evaluate_model(
        model, tokenizer, test_data, batch_size, max_new_tokens
    )
    del model
    torch.cuda.empty_cache()
    print(f"  [epoch 0] -> {acc*100:.2f}% in {time.time()-t0:.0f}s", flush=True)
    _baseline_acc_cache[base_model_id] = acc
    return acc


def run_one_experiment(
    slug: str,
    run_relpath: str,
    base_model_id: str,
    ckpt_root: Path,
    tokenizer,
    test_data,
    batch_size: int,
    max_new_tokens: int,
    eea,
    json_out: Path,
    skip_if_exists: bool,
) -> Dict[int, List[float]]:
    run_dir = ckpt_root / run_relpath
    if skip_if_exists and json_out.is_file():
        print(f"SKIP (exists): {json_out}", flush=True)
        with open(json_out) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    if not run_dir.is_dir():
        print(f"MISSING RUN DIR: {run_dir}", flush=True)
        return {}

    ckpts = _discover_epoch_checkpoints(run_dir)
    if not ckpts:
        print(f"NO checkpoint-* under {run_dir}", flush=True)
        return {}

    out: Dict[int, List[float]] = {}
    out[0] = [_eval_baseline(base_model_id, tokenizer, test_data, batch_size, max_new_tokens, eea)]

    for epoch_idx, ckpt_path in ckpts:
        print(f"  [epoch {epoch_idx}] {ckpt_path}", flush=True)
        t0 = time.time()
        model = _load_model_from_checkpoint(ckpt_path, base_model_id)
        acc, cor, tot = eea.evaluate_model(
            model, tokenizer, test_data, batch_size, max_new_tokens
        )
        del model
        torch.cuda.empty_cache()
        print(
            f"      -> {acc*100:.2f}% ({cor}/{tot}) in {time.time()-t0:.0f}s",
            flush=True,
        )
        out.setdefault(epoch_idx, []).append(acc)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump({str(k): v for k, v in sorted(out.items())}, f, indent=2)
    print(f"  Wrote {json_out}", flush=True)
    return out


def plot_family(
    json_dir: Path,
    title: str,
    out_png: Path,
    y_label: str = "GSM8K accuracy (%)",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
):
    files = sorted(json_dir.glob("*_epoch_accuracy.json"))
    if not files:
        print(f"No JSON in {json_dir}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.tab10.colors
    for i, path in enumerate(files):
        slug = path.name.replace("_epoch_accuracy.json", "")
        with open(path) as f:
            data = json.load(f)
        epochs = sorted(int(k) for k in data)
        means = [float(np.mean(data[str(e)])) * 100 for e in epochs]
        color = cmap[i % len(cmap)]
        ax.plot(epochs, means, "o-", label=slug, color=color, linewidth=2, markersize=6)
        for e, m in zip(epochs, means):
            ax.annotate(
                f"{m:.1f}",
                (e, m),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=12,
                color=color,
            )

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks(list(range(0, 7)))
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    if y_min is not None or y_max is not None:
        ymin = y_min if y_min is not None else ax.get_ylim()[0]
        ymax = y_max if y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)
        # With focused y-ranges, show denser, integer-aligned ticks.
        ax.set_yticks(np.arange(int(np.floor(ymin)), int(np.ceil(ymax)) + 1, 2))
    else:
        ax.set_ylim(bottom=0, top=max(100, ax.get_ylim()[1]))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Plot: {out_png}", flush=True)


def mock_preview_plots(out_dir: Path):
    """Deterministic fake curves → PNG preview (no GPU, no checkpoints)."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def fake_family(name: str, slugs: List[str], path: Path):
        fig, ax = plt.subplots(figsize=(11, 6))
        xs = np.arange(0, 7)
        cmap = plt.cm.tab10.colors
        for i, slug in enumerate(slugs):
            base = 35 + i * 5 + name.startswith("Qwen") * 15
            ys = base + 4 * xs + rng.normal(0, 1.2, size=len(xs))
            ys = np.clip(ys, 0, 95)
            color = cmap[i % len(cmap)]
            ax.plot(xs, ys, "o-", label=f"{slug} (MOCK)", color=color, lw=2, ms=6)
        ax.set_xlabel("Epoch (0 = pretrained baseline)", fontsize=12)
        ax.set_ylabel("GSM8K accuracy (%) — MOCK DATA", fontsize=12)
        ax.set_title(
            f"[PREVIEW — NOT REAL] {name}: epoch accuracy (split layout)", fontsize=13
        )
        ax.set_xticks(list(range(0, 7)))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Mock plot saved: {path}", flush=True)

    llama_slugs = [s for s, _, _ in LLAMA_EXPERIMENTS]
    qwen_slugs = [s for s, _, _ in QWEN_EXPERIMENTS]
    fake_family(
        "Llama-3.1-8B",
        llama_slugs,
        out_dir / "mock_epoch_accuracy_llama_preview.png",
    )
    fake_family(
        "Qwen3-8B-Base",
        qwen_slugs,
        out_dir / "mock_epoch_accuracy_qwen-base_preview.png",
    )
    mistral_slugs = [s for s, _, _ in MISTRAL_EXPERIMENTS]
    fake_family(
        "Mistral-7B-v0.3",
        mistral_slugs,
        out_dir / "mock_epoch_accuracy_mistral-7b-v03_preview.png",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nw-root",
        type=Path,
        default=NW_ROOT,
        help="exploration-finetuning repo root",
    )
    ap.add_argument(
        "--ckpt-root",
        type=Path,
        default=DEFAULT_CKPT,
        help="Directory that contains llama/, qwen-base/, mistral-7b-v03/, …",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: NW_ROOT/eval/split/gsm8k",
    )
    ap.add_argument(
        "--families",
        choices=["llama", "qwen-base", "mistral-7b-v03", "all"],
        default="all",
    )
    ap.add_argument("--y-min", type=float, default=None, help="Optional y-axis min (both families if per-family unset).")
    ap.add_argument("--y-max", type=float, default=None, help="Optional y-axis max (both families if per-family unset).")
    ap.add_argument("--llama-y-min", type=float, default=None, help="Llama plot y-axis minimum (overrides --y-min for Llama).")
    ap.add_argument("--llama-y-max", type=float, default=None, help="Llama plot y-axis maximum (overrides --y-max for Llama).")
    ap.add_argument("--qwen-y-min", type=float, default=None, help="Qwen-base plot y-axis minimum (overrides --y-min for Qwen).")
    ap.add_argument("--qwen-y-max", type=float, default=None, help="Qwen-base plot y-axis maximum (overrides --y-max for Qwen).")
    ap.add_argument("--mistral-y-min", type=float, default=None, help="Mistral plot y-axis minimum (overrides --y-min for Mistral).")
    ap.add_argument("--mistral-y-max", type=float, default=None, help="Mistral plot y-axis maximum (overrides --y-max for Mistral).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--mock-plot-only",
        action="store_true",
        help="Only write mock PNG previews (no model loading).",
    )
    ap.add_argument(
        "--skip-existing-json",
        action="store_true",
        help="Skip an experiment if its JSON already exists.",
    )
    ap.add_argument("--plot-only", action="store_true", help="Only rebuild PNGs from JSON.")
    args = ap.parse_args()

    nw = args.nw_root.resolve()
    out_root = (args.out_root or (nw / "eval" / "split" / "gsm8k")).resolve()

    print(
        f"Paths: TOPO_ROOT={TOPO_ROOT} | "
        f"eval_epoch_accuracy={_resolve_gsm8k_eval_epoch_accuracy_path()}",
        flush=True,
    )

    if args.mock_plot_only:
        mock_preview_plots(out_root / "_preview")
        return

    if args.plot_only:
        ly_min = args.llama_y_min if args.llama_y_min is not None else args.y_min
        ly_max = args.llama_y_max if args.llama_y_max is not None else args.y_max
        qy_min = args.qwen_y_min if args.qwen_y_min is not None else args.y_min
        qy_max = args.qwen_y_max if args.qwen_y_max is not None else args.y_max
        my_min = args.mistral_y_min if args.mistral_y_min is not None else args.y_min
        my_max = args.mistral_y_max if args.mistral_y_max is not None else args.y_max
        if args.families in ("llama", "all"):
            plot_family(
                out_root / "llama" / "json",
                "Llama — GSM8K accuracy vs epoch",
                out_root / "llama" / "plots" / "epoch_accuracy_llama.png",
                y_min=ly_min,
                y_max=ly_max,
            )
        if args.families in ("qwen-base", "all"):
            plot_family(
                out_root / "qwen-base" / "json",
                "Qwen-base — GSM8K accuracy vs epoch",
                out_root / "qwen-base" / "plots" / "epoch_accuracy_qwen-base.png",
                y_min=qy_min,
                y_max=qy_max,
            )
        if args.families in ("mistral-7b-v03", "all"):
            plot_family(
                out_root / "mistral-7b-v03" / "json",
                "Mistral-7B-v0.3 — GSM8K accuracy vs epoch",
                out_root / "mistral-7b-v03" / "plots" / "epoch_accuracy_mistral-7b-v03.png",
                y_min=my_min,
                y_max=my_max,
            )
        return

    eea = _load_eval_epoch_accuracy_module()

    print("Loading GSM8K test...", flush=True)
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]
    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))
    print(f"  {len(test_data)} test examples\n", flush=True)

    def make_tok(model_id: str):
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return tok

    if args.families in ("llama", "all"):
        print("--- Llama tokenizer ---", flush=True)
        tokenizer = make_tok(LLAMA_BASE)
        for slug, rel, base in LLAMA_EXPERIMENTS:
            jpath = out_root / "llama" / "json" / f"{slug}_epoch_accuracy.json"
            print(f"\n{'='*60}\n  {slug}  ({rel})\n{'='*60}", flush=True)
            run_one_experiment(
                slug,
                rel,
                base,
                args.ckpt_root,
                tokenizer,
                test_data,
                args.batch_size,
                args.max_new_tokens,
                eea,
                jpath,
                args.skip_existing_json,
            )

    if args.families in ("qwen-base", "all"):
        print("--- Qwen tokenizer ---", flush=True)
        tokenizer = make_tok(QWEN_BASE)
        for slug, rel, base in QWEN_EXPERIMENTS:
            jpath = out_root / "qwen-base" / "json" / f"{slug}_epoch_accuracy.json"
            print(f"\n{'='*60}\n  {slug}  ({rel})\n{'='*60}", flush=True)
            run_one_experiment(
                slug,
                rel,
                base,
                args.ckpt_root,
                tokenizer,
                test_data,
                args.batch_size,
                args.max_new_tokens,
                eea,
                jpath,
                args.skip_existing_json,
            )

    if args.families in ("mistral-7b-v03", "all"):
        print("--- Mistral tokenizer ---", flush=True)
        tokenizer = make_tok(MISTRAL_BASE)
        for slug, rel, base in MISTRAL_EXPERIMENTS:
            jpath = out_root / "mistral-7b-v03" / "json" / f"{slug}_epoch_accuracy.json"
            print(f"\n{'='*60}\n  {slug}  ({rel})\n{'='*60}", flush=True)
            run_one_experiment(
                slug,
                rel,
                base,
                args.ckpt_root,
                tokenizer,
                test_data,
                args.batch_size,
                args.max_new_tokens,
                eea,
                jpath,
                args.skip_existing_json,
            )

    ly_min = args.llama_y_min if args.llama_y_min is not None else args.y_min
    ly_max = args.llama_y_max if args.llama_y_max is not None else args.y_max
    qy_min = args.qwen_y_min if args.qwen_y_min is not None else args.y_min
    qy_max = args.qwen_y_max if args.qwen_y_max is not None else args.y_max
    my_min = args.mistral_y_min if args.mistral_y_min is not None else args.y_min
    my_max = args.mistral_y_max if args.mistral_y_max is not None else args.y_max
    if args.families in ("llama", "all"):
        plot_family(
            out_root / "llama" / "json",
            "Llama — GSM8K accuracy vs epoch",
            out_root / "llama" / "plots" / "epoch_accuracy_llama.png",
            y_min=ly_min,
            y_max=ly_max,
        )
    if args.families in ("qwen-base", "all"):
        plot_family(
            out_root / "qwen-base" / "json",
            "Qwen-base — GSM8K accuracy vs epoch",
            out_root / "qwen-base" / "plots" / "epoch_accuracy_qwen-base.png",
            y_min=qy_min,
            y_max=qy_max,
        )
    if args.families in ("mistral-7b-v03", "all"):
        plot_family(
            out_root / "mistral-7b-v03" / "json",
            "Mistral-7B-v0.3 — GSM8K accuracy vs epoch",
            out_root / "mistral-7b-v03" / "plots" / "epoch_accuracy_mistral-7b-v03.png",
            y_min=my_min,
            y_max=my_max,
        )


if __name__ == "__main__":
    main()
