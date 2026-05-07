"""
Trainer callback: record GPU memory (PyTorch) and utilization (nvidia-smi) during
training, write training_gpu_stats.json under output_dir. Merged into eval JSON
by run_sst2_imdb_mmlu_finetune_run1.sh after training.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerState, TrainingArguments, TrainerControl

TRAINING_GPU_STATS_FILE = "training_gpu_stats.json"


def _interval_steps() -> int:
    raw = (os.environ.get("GPU_TRAIN_STATS_INTERVAL") or "100").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _all_peak_reserved_gib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    out: Dict[str, float] = {}
    for i in range(torch.cuda.device_count()):
        out[str(i)] = round(torch.cuda.max_memory_reserved(i) / (1024**3), 3)
    return out


def _all_peak_allocated_gib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    out: Dict[str, float] = {}
    for i in range(torch.cuda.device_count()):
        out[str(i)] = round(torch.cuda.max_memory_allocated(i) / (1024**3), 3)
    return out


def _nvidia_smi_util_per_gpu() -> Optional[List[int]]:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        vals = [int(x.strip()) for x in p.stdout.strip().splitlines() if x.strip()]
        return vals or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _last_train_runtime_sec(state: TrainerState) -> Optional[float]:
    if not state.log_history:
        return None
    for h in reversed(state.log_history):
        if isinstance(h, dict) and "train_runtime" in h and h["train_runtime"] is not None:
            try:
                return float(h["train_runtime"])
            except (TypeError, ValueError):
                pass
    return None


class GpuTrainStatsCallback(TrainerCallback):
    def __init__(self, output_dir: str, sample_interval_steps: Optional[int] = None):
        self.output_dir = output_dir
        self.sample_interval = int(sample_interval_steps) if sample_interval_steps is not None else _interval_steps()
        self.sample_interval = max(1, self.sample_interval)
        self._util_samples: List[int] = []
        self._util_max_gpus: List[int] = []
        self._mem_snaps: List[Dict[str, Any]] = []

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._util_samples.clear()
        self._util_max_gpus.clear()
        self._mem_snaps.clear()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = int(state.global_step)
        if step < 1 or step % self.sample_interval != 0:
            return
        per = _nvidia_smi_util_per_gpu()
        if per:
            self._util_max_gpus.append(max(per))
            for v in per:
                self._util_samples.append(int(v))
        if torch.cuda.is_available():
            snap: Dict[str, Any] = {"step": step}
            for i in range(torch.cuda.device_count()):
                snap[f"reserved_gib_gpu{i}"] = round(
                    torch.cuda.memory_reserved(i) / (1024**3), 3
                )
            self._mem_snaps.append(snap)
            if len(self._mem_snaps) > 200:
                self._mem_snaps = self._mem_snaps[-200:]

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        out: Dict[str, Any] = {
            "output_dir": self.output_dir,
            "total_steps": int(state.global_step) if state.global_step is not None else None,
            "train_runtime_sec": _last_train_runtime_sec(state),
            "sample_interval_steps": self.sample_interval,
            "torch_peak_reserved_gib": _all_peak_reserved_gib(),
            "torch_peak_allocated_gib": _all_peak_allocated_gib(),
        }
        if self._util_max_gpus:
            out["gpu_utilization_max_across_gpus_percent"] = {
                "per_step_sample_max": self._util_max_gpus,
                "max": max(self._util_max_gpus),
                "mean": round(
                    sum(self._util_max_gpus) / len(self._util_max_gpus), 2
                ),
                "n_step_samples": len(self._util_max_gpus),
                "source": "nvidia-smi utilization.gpu",
            }
        if self._util_samples:
            out["gpu_utilization_per_gpu_samples_percent"] = {
                "all_samples_flat": self._util_samples,
                "max": max(self._util_samples),
                "mean": round(
                    sum(self._util_samples) / len(self._util_samples), 2
                ),
            }
        if self._mem_snaps:
            out["reserved_gib_time_series_last_n"] = self._mem_snaps
        path = os.path.join(self.output_dir, TRAINING_GPU_STATS_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
