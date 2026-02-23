#!/usr/bin/env python3
import os
import json
import math
import argparse
from typing import Dict, Any, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ema(x: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def get_series(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ml = d.get("metrics_log", {})
    steps = np.array(ml.get("steps", []), dtype=int)
    losses = np.array(ml.get("losses", []), dtype=float)
    lrs = np.array(ml.get("learning_rates", []), dtype=float)
    gns = np.array(ml.get("gradient_norms", []), dtype=float)
    return steps, losses, lrs, gns


def find_step_duration_series(d: Dict[str, Any], steps: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    ml = d.get("metrics_log", {})
    timing = d.get("timing", {})

    candidate_keys = [
        "step_times",
        "step_time_s",
        "step_durations",
        "step_duration_s",
        "durations",
        "times_s",
    ]

    step_time = None
    for k in candidate_keys:
        if k in ml and isinstance(ml[k], list) and len(ml[k]) == len(steps):
            step_time = np.array(ml[k], dtype=float)
            break

    mean_step = None
    if isinstance(timing, dict) and "avg_step_s" in timing:
        try:
            mean_step = float(timing["avg_step_s"])
        except Exception:
            mean_step = None

    if step_time is not None and step_time.size > 0:
        mean_step = float(np.nanmean(step_time))

    return step_time, mean_step


def sci_str(v: float) -> str:
    """2e-5 style (no 2.0e-5)."""
    if v == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(v))))
    base = v / (10 ** exp)
    sbase = f"{base:.2f}".rstrip("0").rstrip(".")
    return f"{sbase}e{exp}"


def pick_lr_scale(lr_max: float) -> float:
    """
    Choose a nice power-of-10 scale so LR axis can show 0.5,1,1.5,2...
    If lr_max=2e-5 -> scale=1e-5
    If lr_max=2e-4 -> scale=1e-4
    etc.
    """
    if lr_max <= 0:
        return 1e-5
    exp = int(math.floor(math.log10(lr_max)))
    # If lr_max is ~2e-5, exp=-5, choose 1e-5.
    return 10 ** exp


def make_figure(
    name: str,
    steps: np.ndarray,
    losses: np.ndarray,
    lrs: np.ndarray,
    gns: np.ndarray,
    step_time: Optional[np.ndarray],
    mean_step: Optional[float],
    ema_alpha: float,
) -> go.Figure:
    # More space between plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Training Loss", "LR Schedule (cosine)", "Gradient Norms", "Step Duration"),
        horizontal_spacing=0.14,
        vertical_spacing=0.22,
    )

    # Same-family colors: raw is same color with low opacity
    LOSS_COLOR = "#2563EB"   # blue
    GN_COLOR   = "#059669"   # green
    LR_COLOR   = "#DC2626"   # red
    STEP_COLOR = "#7C3AED"   # purple
    RAW_OPACITY = 0.18

    # ---------- Loss ----------
    loss_ema = ema(losses, alpha=ema_alpha)
    fig.add_trace(
        go.Scatter(
            x=steps, y=losses, mode="lines",
            name="Loss (raw)",
            line=dict(width=2, color=LOSS_COLOR),
            opacity=RAW_OPACITY,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=loss_ema, mode="lines",
            name=f"Loss (EMA α={ema_alpha:.2f})",
            line=dict(width=4, color=LOSS_COLOR),
        ),
        row=1, col=1
    )

    # ---------- LR (will rescale later for nicer ticks) ----------
    fig.add_trace(
        go.Scatter(
            x=steps, y=lrs, mode="lines",
            name="LR",
            line=dict(width=4, color=LR_COLOR),
        ),
        row=1, col=2
    )

    # ---------- Grad Norm ----------
    gn_ema = ema(gns, alpha=ema_alpha)
    fig.add_trace(
        go.Scatter(
            x=steps, y=gns, mode="lines",
            name="GradNorm (raw)",
            line=dict(width=2, color=GN_COLOR),
            opacity=RAW_OPACITY,
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=gn_ema, mode="lines",
            name=f"GradNorm (EMA α={ema_alpha:.2f})",
            line=dict(width=4, color=GN_COLOR),
        ),
        row=2, col=1
    )

    # ---------- Step duration ----------
    plotted_step_series = False
    if step_time is not None and step_time.size == len(steps) and step_time.size > 0:
        st_ema = ema(step_time, alpha=ema_alpha)
        fig.add_trace(
            go.Scatter(
                x=steps, y=step_time, mode="lines",
                name="Step time (raw)",
                line=dict(width=2, color=STEP_COLOR),
                opacity=RAW_OPACITY,
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=steps, y=st_ema, mode="lines",
                name=f"Step time (EMA α={ema_alpha:.2f})",
                line=dict(width=4, color=STEP_COLOR),
            ),
            row=2, col=2
        )
        plotted_step_series = True

    # Always draw mean line if we have it (even if no per-step series)
    if mean_step is not None and steps.size > 0:
        fig.add_trace(
            go.Scatter(
                x=[int(steps.min()), int(steps.max())],
                y=[mean_step, mean_step],
                mode="lines",
                name=f"Step mean = {mean_step:.2f}s",
                line=dict(width=3, dash="dash", color=STEP_COLOR),
            ),
            row=2, col=2
        )
        # Put mean label inside the subplot (right side) — not overlapping legend/title
        fig.add_annotation(
            x=float(steps.max()),
            y=float(mean_step),
            xref="x4", yref="y4",
            text=f"mean={mean_step:.2f}s",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=26, color=STEP_COLOR),
        )

    # ---------- Fonts / layout ----------
    TITLE_FONT = 44
    SUBTITLE_FONT = 30
    AXIS_TITLE = 32
    TICK_FONT = 28
    LEGEND_FONT = 26

    fig.update_layout(
        template="plotly_white",
        width=1800,
        height=1000,
        title=dict(
            text=name,
            x=0.5,
            xanchor="center",
            y=0.99,
            font=dict(size=TITLE_FONT),
        ),
        font=dict(size=LEGEND_FONT),
        # Legend above, no overlap with plots
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.20,
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_FONT),
        ),
        # Extra top margin to make room for title + legend + LR label
        margin=dict(l=90, r=60, t=280, b=90),
    )

    # Subplot titles
    fig.update_annotations(font=dict(size=SUBTITLE_FONT))

    # Axes: keep axis lines; avoid boxy look (no mirroring)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor="black",
        mirror=False,
        ticks="outside", tickwidth=2, ticklen=7,
        tickfont=dict(size=TICK_FONT),
        title_font=dict(size=AXIS_TITLE),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor="black",
        mirror=False,
        ticks="outside", tickwidth=2, ticklen=7,
        tickfont=dict(size=TICK_FONT),
        title_font=dict(size=AXIS_TITLE),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
    )

    # Axis titles
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=1, col=2)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=2)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Grad L2 norm", row=2, col=1)
    fig.update_yaxes(title_text="Time (s)", row=2, col=2)

    # ---------- LR formatting exactly how you want ----------
    lr_max = float(np.nanmax(lrs)) if lrs.size > 0 else 0.0
    lr_scale = pick_lr_scale(lr_max)  # e.g., 1e-5 or 1e-4
    lrs_scaled = (lrs / lr_scale) if lr_scale > 0 else lrs

    # Update LR trace y to scaled values
    for tr in fig.data:
        if tr.name == "LR":
            tr.y = lrs_scaled

    # LR axis shows multipliers (1, 1.5, 2...) and label indicates scale
    fig.update_yaxes(title_text=f"LR (×{sci_str(lr_scale)})", row=1, col=2)

    # Ticks like: 0.5, 1, 1.5, 2
    tickvals = [0.5*lr_scale, 1.0*lr_scale, 1.5*lr_scale, 2.0*lr_scale]
    fig.update_yaxes(
        tickvals=tickvals,
        ticktext=[str(t/lr_scale).rstrip("0").rstrip(".") for t in tickvals],
        row=1, col=2
    )



    # If there is no step series at all, keep subplot clean but still show mean line if present
    if not plotted_step_series and mean_step is None:
        # Add a small note so the empty panel isn't confusing
        if steps.size > 0:
            fig.add_annotation(
                x=0.82, y=0.16,
                xref="paper", yref="paper",
                text="(no per-step timing logged)",
                showarrow=False,
                font=dict(size=22, color="#666"),
            )

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reports",
        nargs="+",
        default=[
            "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-full-finetuned/training_report_full.json",
            "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-full-instruct/training_report_full.json",
            "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-lora-finetuned/training_report_lora.json",
            "/home/kadir/topo/numpy_weights/exploration-finetuning/gsm8k-lora-instruct/training_report_lora.json",
        ],
    )
    ap.add_argument("--outdir", default="plotly_training_slides", help="Output directory.")
    ap.add_argument("--ema-alpha", type=float, default=0.12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for rp in args.reports:
        if not os.path.exists(rp):
            print(f"[SKIP] Missing: {rp}")
            continue

        d = load_json(rp)

        method = d.get("method", "UNKNOWN")
        model = d.get("model", os.path.basename(os.path.dirname(rp)))
        nice_name = f"{method} / {model} — GSM8K Training Metrics"

        steps, losses, lrs, gns = get_series(d)
        step_time, mean_step = find_step_duration_series(d, steps)

        fig = make_figure(
            name=nice_name,
            steps=steps,
            losses=losses,
            lrs=lrs,
            gns=gns,
            step_time=step_time,
            mean_step=mean_step,
            ema_alpha=args.ema_alpha,
        )

        parent = os.path.basename(os.path.dirname(rp))
        out_prefix = os.path.join(args.outdir, parent)

        html_path = out_prefix + "_plotly.html"
        fig.write_html(html_path)
        print(f"[OK] Wrote {html_path}")


if __name__ == "__main__":
    main()
