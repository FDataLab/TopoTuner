#!/usr/bin/env python3
import json
import glob
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class EvalRow:
    file: str
    model: str
    accuracy: float
    correct: int
    total: int
    eval_time_s: Optional[float]

    @property
    def acc_pct(self) -> float:
        return 100.0 * self.accuracy

    @property
    def time_per_sample_ms(self) -> Optional[float]:
        if self.eval_time_s is None or self.total == 0:
            return None
        return 1000.0 * (self.eval_time_s / self.total)


def safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def load_eval_json(path: str) -> EvalRow:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = str(safe_get(data, "model", os.path.basename(path)))
    correct = int(safe_get(data, "correct", 0))
    total = int(safe_get(data, "total", 0))
    acc = safe_get(data, "accuracy", None)

    # if accuracy missing, compute it
    if acc is None:
        acc = (correct / total) if total else 0.0
    else:
        acc = float(acc)

    # sanity check: if mismatch is large, prefer correct/total
    if total > 0:
        acc_from_counts = correct / total
        if abs(acc - acc_from_counts) > 1e-3:
            # keep both? we’ll trust counts
            acc = acc_from_counts

    eval_time_s = safe_get(data, "eval_time_s", None)
    eval_time_s = float(eval_time_s) if eval_time_s is not None else None

    return EvalRow(
        file=os.path.basename(path),
        model=model,
        accuracy=acc,
        correct=correct,
        total=total,
        eval_time_s=eval_time_s,
    )


def format_ms(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1f}"


def main():
    # adjust if your files live elsewhere
    paths = sorted(glob.glob("gsm8k*.json"))

    if not paths:
        raise SystemExit("No files matched gsm8k*.json in current directory.")

    rows: List[EvalRow] = [load_eval_json(p) for p in paths]

    # print summary table
    print("\n=== GSM8K Eval Summary (from JSON files) ===\n")
    header = f"{'file':38}  {'acc%':>7}  {'correct/total':>13}  {'t(s)':>7}  {'ms/sample':>10}  model"
    print(header)
    print("-" * len(header))

    rows_sorted = sorted(rows, key=lambda r: r.accuracy, reverse=True)

    for r in rows_sorted:
        msps = r.time_per_sample_ms
        print(
            f"{r.file:38}  "
            f"{r.acc_pct:7.2f}  "
            f"{r.correct:6d}/{r.total:<6d}  "
            f"{(f'{r.eval_time_s:.1f}' if r.eval_time_s is not None else '-'):>7}  "
            f"{format_ms(msps):>10}  "
            f"{r.model}"
        )

    # choose baselines automatically if present
    # You can customize these keywords.
    baseline_base = next((r for r in rows if "Llama-3.1-8B" in r.model and "Instruct" not in r.model), None)
    baseline_inst = next((r for r in rows if "Llama-3.1-8B-Instruct" in r.model), None)

    print("\n=== Deltas vs Baselines ===\n")
    if baseline_base:
        print(f"Baseline (BASE):    {baseline_base.model}  acc={baseline_base.acc_pct:.2f}% ({baseline_base.correct}/{baseline_base.total})")
    else:
        print("Baseline (BASE):    not found automatically (edit script keywords).")

    if baseline_inst:
        print(f"Baseline (INSTRUCT): {baseline_inst.model}  acc={baseline_inst.acc_pct:.2f}% ({baseline_inst.correct}/{baseline_inst.total})")
    else:
        print("Baseline (INSTRUCT): not found automatically (edit script keywords).")

    def delta_line(r: EvalRow, b: EvalRow) -> str:
        delta = 100.0 * (r.accuracy - b.accuracy)
        return f"{r.model:35}  {r.acc_pct:7.2f}%  (Δ {delta:+.2f} pts)"

    if baseline_base:
        print("\n-- vs BASE baseline --")
        for r in rows_sorted:
            print(delta_line(r, baseline_base))

    if baseline_inst:
        print("\n-- vs INSTRUCT baseline --")
        for r in rows_sorted:
            print(delta_line(r, baseline_inst))


if __name__ == "__main__":
    main()
