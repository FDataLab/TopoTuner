import json
import os
import math
import datetime
import time
import torch
from functools import partial
from contextlib import contextmanager
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import has_length
from peft import LoraConfig, get_peft_model, TaskType

from .data_preprocessing_imdb import (
    preprocess_dataset, custom_data_collator,
    infer_prompt_format_from_model_id
)
from .eval_imdb import evaluate_imdb

from codes.utils.args import parse_args
from codes.utils.model_saving import SavePeftModelCallback, concise_lora_filename, concise_full_filename

from transformers.utils import logging as hf_logging
hf_logging.enable_progress_bar()

import wandb
import shutil
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")


# ---------- Lightweight Timing Utility ----------
class TimingTracker:
    """Lightweight timing tracker with minimal overhead."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings = {}
        self.overhead_samples = []
        self._overhead_measured = False
        self._timing_overhead_total = 0.0  # Track total time spent in timing code itself
    
    def _measure_overhead(self, n_samples=1000):
        """Measure timing overhead once."""
        if self._overhead_measured:
            return
        overhead_times = []
        for _ in range(n_samples):
            start = time.perf_counter()
            end = time.perf_counter()
            overhead_times.append(end - start)
        self.overhead_samples = overhead_times
        avg_overhead = sum(overhead_times) / len(overhead_times)
        self._overhead_measured = True
        print(f"   ⏱️  Timing overhead: {avg_overhead*1e6:.2f} microseconds per call", flush=True)
    
    @contextmanager
    def time_block(self, name: str):
        """Context manager for timing a code block."""
        if not self.enabled:
            yield
            return
        
        # Measure overhead of timing code itself
        timing_start = time.perf_counter()
        
        self._measure_overhead()
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            timing_end = time.perf_counter()
            
            # Calculate actual elapsed time (work being timed)
            elapsed = end - start
            
            # Calculate timing overhead (time spent in timing code)
            timing_overhead = (timing_end - timing_start) - elapsed
            self._timing_overhead_total += max(0, timing_overhead)  # Only count positive overhead
            
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
    
    def format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}min"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}min {secs:.0f}s"
    
    def print_summary(self):
        """Print timing summary."""
        if not self.enabled or not self.timings:
            return
        print("\n" + "=" * 80, flush=True)
        print("⏱️  TIMING SUMMARY", flush=True)
        print("=" * 80, flush=True)
        
        # Calculate total excluding overall pipeline (to avoid double counting)
        timing_keys = [k for k in self.timings.keys() if k != "Pipeline (Overall)"]
        total_time = sum(sum(self.timings[k]) for k in timing_keys)
        
        # Print individual timings
        step_timing_key = "Training (Forward+Backward+Optimizer per step)"
        for name in sorted(timing_keys):
            # Skip step-level timing here, we'll print it separately
            if name == step_timing_key:
                continue
            times = self.timings[name]
            total = sum(times)
            avg = total / len(times) if times else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            count_str = f" (x{len(times)})" if len(times) > 1 else ""
            print(f"   {name:35s}: {self.format_time(total):>12s} (avg: {self.format_time(avg)}{count_str}, {pct:5.1f}%)", flush=True)
        
        # Print detailed training breakdown (forward, backward, optimizer)
        forward_key = "Training (Forward Pass)"
        backward_key = "Training (Backward Pass)"
        optimizer_key = "Training (Optimizer Step)"
        
        if forward_key in self.timings or backward_key in self.timings:
            print(f"\n   {'Training Breakdown (Per Step)':35s}:", flush=True)
            
            if forward_key in self.timings:
                forward_times = self.timings[forward_key]
                total_forward = sum(forward_times)
                avg_forward = total_forward / len(forward_times) if forward_times else 0
                pct_forward = (total_forward / total_time * 100) if total_time > 0 else 0
                print(f"      {'  → Forward Pass':33s}: {self.format_time(total_forward):>12s} (avg: {self.format_time(avg_forward)}, {pct_forward:5.1f}%)", flush=True)
            
            if backward_key in self.timings:
                backward_times = self.timings[backward_key]
                total_backward = sum(backward_times)
                avg_backward = total_backward / len(backward_times) if backward_times else 0
                pct_backward = (total_backward / total_time * 100) if total_time > 0 else 0
                print(f"      {'  → Backward Pass':33s}: {self.format_time(total_backward):>12s} (avg: {self.format_time(avg_backward)}, {pct_backward:5.1f}%)", flush=True)
            
            if optimizer_key in self.timings:
                optimizer_times = self.timings[optimizer_key]
                total_optimizer = sum(optimizer_times)
                avg_optimizer = total_optimizer / len(optimizer_times) if optimizer_times else 0
                pct_optimizer = (total_optimizer / total_time * 100) if total_time > 0 else 0
                print(f"      {'  → Optimizer Step':33s}: {self.format_time(total_optimizer):>12s} (avg: {self.format_time(avg_optimizer)}, {pct_optimizer:5.1f}%)", flush=True)
        
        # Print step-level timing separately with more detail (if available)
        if step_timing_key in self.timings:
            step_times = self.timings[step_timing_key]
            if step_times:
                total_steps_time = sum(step_times)
                avg_step = sum(step_times) / len(step_times)
                min_step = min(step_times)
                max_step = max(step_times)
                pct = (total_steps_time / total_time * 100) if total_time > 0 else 0
                print(f"\n   {'Training (Per Step Total)':35s}: {self.format_time(total_steps_time):>12s} ({len(step_times)} steps, {pct:5.1f}%)", flush=True)
                print(f"      {'  → Avg per step':33s}: {self.format_time(avg_step):>12s}", flush=True)
                print(f"      {'  → Min step':33s}: {self.format_time(min_step):>12s}", flush=True)
                print(f"      {'  → Max step':33s}: {self.format_time(max_step):>12s}", flush=True)
        
        # Print overall pipeline time
        if "Pipeline (Overall)" in self.timings:
            pipeline_time = sum(self.timings["Pipeline (Overall)"])
            print(f"\n   {'Pipeline (Overall)':35s}: {self.format_time(pipeline_time):>12s}", flush=True)
        
        # Print timing overhead (actual measured + estimated perf_counter overhead)
        if self.overhead_samples:
            total_calls = sum(len(times) for times in self.timings.values())
            avg_overhead = sum(self.overhead_samples) / len(self.overhead_samples)
            perf_counter_overhead = avg_overhead * total_calls
            
            # Total overhead = actual timing code overhead + perf_counter overhead
            total_overhead = self._timing_overhead_total + perf_counter_overhead
            overhead_pct = (total_overhead / total_time * 100) if total_time > 0 else 0
            
            print(f"   {'Timing overhead (measured)':35s}: {self.format_time(self._timing_overhead_total):>12s}", flush=True)
            print(f"   {'Timing overhead (perf_counter)':35s}: {self.format_time(perf_counter_overhead):>12s}", flush=True)
            print(f"   {'Timing overhead (total)':35s}: {self.format_time(total_overhead):>12s} ({overhead_pct:.3f}%)", flush=True)
        
        print("=" * 80 + "\n", flush=True)


# Global timing tracker (will be initialized in main() based on args)
_timing_tracker = None


# ---------- GPU Info ----------
def get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_id": None, "mem_alloc_MB": None, "mem_reserved_MB": None}
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu": props.name,
        "gpu_id": gpu_id,
        "total_mem_GB": round(props.total_memory / 1024**3, 2),
        "mem_alloc_MB": torch.cuda.memory_allocated(gpu_id) // 1024**2,
        "mem_reserved_MB": torch.cuda.memory_reserved(gpu_id) // 1024**2,
    }

# ---------- Callbacks ----------
class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] "
                  f"Loss: {logs['loss']:.6f}", flush=True)

class DetailedTimingTrainer(Trainer):
    """Custom Trainer that tracks forward, backward, and optimizer timing separately."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch_forward_times = {}  # epoch_num -> list of forward times
        self._epoch_backward_times = {}   # epoch_num -> list of backward times
        self._epoch_optimizer_times = {} # epoch_num -> list of optimizer times
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to time forward, backward, and optimizer separately."""
        global _timing_tracker
        
        if not (_timing_tracker and _timing_tracker.enabled):
            return super().training_step(model, inputs, num_items_in_batch)
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Time forward pass (includes loss computation)
        forward_start = time.perf_counter()
        loss = self.compute_loss(model, inputs)
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        _timing_tracker.timings.setdefault("Training (Forward Pass)", []).append(forward_time)
        
        # Track per epoch
        epoch_num = int(self.state.epoch) if self.state.epoch is not None else 0
        if epoch_num not in self._epoch_forward_times:
            self._epoch_forward_times[epoch_num] = []
        self._epoch_forward_times[epoch_num].append(forward_time)
        
        # For backward and optimizer, we need to call the parent's training_step
        # but we can't easily separate them. Instead, let's use a simpler approach:
        # Time the backward+optimizer together, then subtract optimizer time separately
        
        # Handle gradient accumulation scaling
        if self.args.gradient_accumulation_steps > 1 and not getattr(self, 'deepspeed', None):
            loss = loss / self.args.gradient_accumulation_steps
        
        # Time backward pass
        backward_start = time.perf_counter()
        do_grad_scaling = getattr(self, 'do_grad_scaling', False)
        use_apex = getattr(self, 'use_apex', False)
        
        if do_grad_scaling and hasattr(self, 'scaler'):
            self.scaler.scale(loss).backward()
        elif use_apex and hasattr(self, 'optimizer'):
            try:
                import apex
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                loss.backward()
        else:
            loss.backward()
        backward_end = time.perf_counter()
        backward_time = backward_end - backward_start
        _timing_tracker.timings.setdefault("Training (Backward Pass)", []).append(backward_time)
        
        # Track per epoch
        if epoch_num not in self._epoch_backward_times:
            self._epoch_backward_times[epoch_num] = []
        self._epoch_backward_times[epoch_num].append(backward_time)
        
        # Time optimizer step (only on accumulation boundary)
        optimizer_time = 0.0
        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer_start = time.perf_counter()
            
            if do_grad_scaling and hasattr(self, 'scaler'):
                self.scaler.unscale_(self.optimizer)
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            optimizer_end = time.perf_counter()
            optimizer_time = optimizer_end - optimizer_start
            _timing_tracker.timings.setdefault("Training (Optimizer Step)", []).append(optimizer_time)
            
            # Track per epoch
            if epoch_num not in self._epoch_optimizer_times:
                self._epoch_optimizer_times[epoch_num] = []
            self._epoch_optimizer_times[epoch_num].append(optimizer_time)
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def log_epoch_timing_breakdown(self, epoch_num: int):
        """Log forward/backward/optimizer breakdown for a specific epoch."""
        global _timing_tracker
        if not (_timing_tracker and _timing_tracker.enabled):
            return
        
        forward_times = self._epoch_forward_times.get(epoch_num, [])
        backward_times = self._epoch_backward_times.get(epoch_num, [])
        optimizer_times = self._epoch_optimizer_times.get(epoch_num, [])
        
        if not forward_times:
            return
        
        total_forward = sum(forward_times)
        total_backward = sum(backward_times)
        total_optimizer = sum(optimizer_times)
        total_epoch = total_forward + total_backward + total_optimizer
        
        avg_forward = total_forward / len(forward_times) if forward_times else 0
        avg_backward = total_backward / len(backward_times) if backward_times else 0
        avg_optimizer = total_optimizer / len(optimizer_times) if optimizer_times else 0
        
        print(f"\n   📊 Epoch {epoch_num} Timing Breakdown:", flush=True)
        print(f"      → Forward Pass:  {_timing_tracker.format_time(total_forward):>12s} (avg: {_timing_tracker.format_time(avg_forward)}, {len(forward_times)} steps)", flush=True)
        print(f"      → Backward Pass: {_timing_tracker.format_time(total_backward):>12s} (avg: {_timing_tracker.format_time(avg_backward)}, {len(backward_times)} steps)", flush=True)
        if optimizer_times:
            print(f"      → Optimizer Step: {_timing_tracker.format_time(total_optimizer):>12s} (avg: {_timing_tracker.format_time(avg_optimizer)}, {len(optimizer_times)} steps)", flush=True)
        print(f"      → Total: {_timing_tracker.format_time(total_epoch):>12s}", flush=True)


class TrainingStepTimingCallback(TrainerCallback):
    """Track per-step timing within training and log detailed breakdown per epoch."""
    def __init__(self):
        self.step_start_time = None
        self.step_times = []
        self.current_epoch = None
        self.epoch_step_times = {}
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Track step start time."""
        global _timing_tracker
        if _timing_tracker and _timing_tracker.enabled:
            self.step_start_time = time.perf_counter()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Track step end time and accumulate."""
        global _timing_tracker
        if _timing_tracker and _timing_tracker.enabled and self.step_start_time is not None:
            step_elapsed = time.perf_counter() - self.step_start_time
            self.step_times.append(step_elapsed)
            
            # Track per epoch
            epoch_num = int(state.epoch) if state.epoch is not None else 0
            if epoch_num not in self.epoch_step_times:
                self.epoch_step_times[epoch_num] = []
            self.epoch_step_times[epoch_num].append(step_elapsed)
    
    def on_epoch_end(self, args, state, control, trainer=None, **kwargs):
        """Log per-epoch step timing summary and detailed breakdown."""
        global _timing_tracker
        if _timing_tracker and _timing_tracker.enabled:
            epoch_num = int(state.epoch) if state.epoch is not None else 0
            if epoch_num in self.epoch_step_times:
                epoch_steps = self.epoch_step_times[epoch_num]
                if epoch_steps:
                    avg_step = sum(epoch_steps) / len(epoch_steps)
                    total_epoch_steps = sum(epoch_steps)
                    _timing_tracker.timings.setdefault("Training (Forward+Backward+Optimizer per step)", []).extend(epoch_steps)
                    print(f"   ⏱️  Epoch {epoch_num}: {len(epoch_steps)} steps, avg {_timing_tracker.format_time(avg_step)}/step, total {_timing_tracker.format_time(total_epoch_steps)}", flush=True)
            
            # Log detailed breakdown if using DetailedTimingTrainer
            if trainer is not None and isinstance(trainer, DetailedTimingTrainer):
                trainer.log_epoch_timing_breakdown(epoch_num)


class AccuracyPerEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, run_eval: bool = False, split: str = "test",
                 limit=None, max_new_tokens: int = 50,
                 log_jsonl=None, log_tsv=None, dataset="", model=""):
        self.tok = tokenizer
        self.run_eval = run_eval
        self.split = split
        self.limit = limit
        self.max_new_tokens = max_new_tokens
        self.log_jsonl = log_jsonl
        self.log_tsv = log_tsv
        self.dataset = dataset
        self.model = model
        self.epoch_start_time = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Track epoch start time."""
        global _timing_tracker
        if _timing_tracker and _timing_tracker.enabled:
            self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Track epoch end time and log to timing tracker."""
        global _timing_tracker
        if _timing_tracker and _timing_tracker.enabled and self.epoch_start_time is not None:
            epoch_elapsed = time.perf_counter() - self.epoch_start_time
            epoch_num = int(state.epoch)
            _timing_tracker.timings.setdefault("Training (Per Epoch)", []).append(epoch_elapsed)
            print(f"   ⏱️  Epoch {epoch_num} completed in {_timing_tracker.format_time(epoch_elapsed)}", flush=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Verify frozen parameters remain frozen at each epoch
        if hasattr(model, '_frozen_param_names') and model._frozen_param_names:
            still_frozen = 0
            for name in model._frozen_param_names:
                for p_name, p in model.named_parameters():
                    if p_name == name and not p.requires_grad:
                        still_frozen += 1
                        break
            if still_frozen < len(model._frozen_param_names):
                print(f"⚠️  WARNING: Only {still_frozen}/{len(model._frozen_param_names)} frozen params remain frozen at epoch {int(state.epoch)}!", flush=True)
            else:
                print(f"✅ Verified: All {still_frozen} frozen parameters remain frozen at epoch {int(state.epoch)}", flush=True)
        
        if not self.run_eval:
            return
        metrics = evaluate_imdb(
            model, self.tok,
            split=self.split,
            limit=self.limit,
            max_new_tokens=self.max_new_tokens,
            debug_print=True
        )
        gpu_info = get_gpu_info()
        record = {
            "epoch": int(state.epoch),
            "dataset": self.dataset,
            "model": self.model,
            "accuracy": metrics["accuracy"],
            "positive_acc": metrics["positive_acc"],
            "negative_acc": metrics["negative_acc"],
            "n": metrics["n"],
            **gpu_info
        }
        print(
          f"[Downstream] epoch={record['epoch']} IMDB {self.split} "
          f"Accuracy={record['accuracy']:.2f}% Pos={record['positive_acc']:.2f}% Neg={record['negative_acc']:.2f}% n={record['n']} "
          f"GPU={gpu_info['gpu']} mem={gpu_info['mem_alloc_MB']}MB",
          flush=True
        )

        # append to jsonl
        if self.log_jsonl:
            os.makedirs(os.path.dirname(self.log_jsonl), exist_ok=True)
            with open(self.log_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
        # append to tsv
        if self.log_tsv:
            import csv
            new_file = not os.path.exists(self.log_tsv)
            with open(self.log_tsv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys(), delimiter="\t")
                if new_file:
                    writer.writeheader()
                writer.writerow(record)

# ---------- Loader (with freezable layers, matching HotpotQA) ----------
def load_model_and_tokenizer(
    model_id: str,
    use_lora: bool,
    freeze_layers=None,
    freeze_q_layers=None,
    freeze_k_layers=None,
    freeze_v_layers=None,
    freeze_qkv_no_grad: bool = False,
    freeze_o_with_qkv: bool = False,
):
    """
    Load model/tokenizer and optionally:
      - freeze full transformer layers (`freeze_layers`)
      - freeze ONLY Q/K/V projections in selected layers
        (`freeze_q_layers`, `freeze_k_layers`, `freeze_v_layers`)
        - Professor's simpler strategy (default): only freezes Q/K/V, not o_proj or MLP
      - optionally ALSO freeze o_proj in those layers when `freeze_o_with_qkv=True`
        (enables K+O, Q+O, V+O experiments)
    
    Freezing strategy:
      - If freeze_k_layers is set: freeze ONLY k_proj (and o_proj if freeze_o_with_qkv=True)
      - If freeze_q_layers is set: freeze ONLY q_proj (and o_proj if freeze_o_with_qkv=True)
      - If freeze_v_layers is set: freeze ONLY v_proj (and o_proj if freeze_o_with_qkv=True)
    
    ⚠️  CRITICAL: This function is called ONCE before training starts.
    Freezing happens here and persists throughout all training epochs.
    """
    freeze_layers = freeze_layers or []
    freeze_q_layers = freeze_q_layers or []
    freeze_k_layers = freeze_k_layers or []
    freeze_v_layers = freeze_v_layers or []

    def _wrap_no_grad(module, label):
        original_forward = module.forward
        def forward_no_grad(*args, **kwargs):
            with torch.no_grad():
                return original_forward(*args, **kwargs)
        module.forward = forward_no_grad
        print(f"   → {label} forward wrapped with no_grad()", flush=True)

    def _apply_no_grad_wrappers(transformer_layers, qset, kset, vset, also_o_proj=False):
        for idx, layer in enumerate(transformer_layers):
            if idx not in (qset | kset | vset):
                continue
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                continue
            if idx in qset and hasattr(attn, "q_proj"):
                _wrap_no_grad(attn.q_proj, f"Layer {idx} q_proj")
            if idx in kset and hasattr(attn, "k_proj"):
                _wrap_no_grad(attn.k_proj, f"Layer {idx} k_proj")
            if idx in vset and hasattr(attn, "v_proj"):
                _wrap_no_grad(attn.v_proj, f"Layer {idx} v_proj")
            # Also wrap o_proj if requested (for K+O, Q+O, V+O experiments)
            if also_o_proj and hasattr(attn, "o_proj"):
                _wrap_no_grad(attn.o_proj, f"Layer {idx} o_proj")

    # Load tokenizer
    print("   [load_model_and_tokenizer] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        token=HF_TOKEN,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    print(f"   [load_model_and_tokenizer] ✅ Tokenizer loaded: vocab_size={tok.vocab_size}", flush=True)

    # Load model
    print("   [load_model_and_tokenizer] Loading model (this may take a moment)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    print(f"   [load_model_and_tokenizer] ✅ Model loaded: {model_id}", flush=True)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    print("   [load_model_and_tokenizer] ✅ Model config updated", flush=True)

    # ---- Freeze full transformer layers (base weights) ----
    if freeze_layers:
        print(f"🔥 Freezing Transformer layers: {freeze_layers}", flush=True)

        try:
            transformer_layers = model.transformer.layers  # Qwen-like
        except AttributeError:
            transformer_layers = model.model.layers        # LLaMA-like

        for idx, layer in enumerate(transformer_layers):
            if idx in freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
                print(f"   → Base layer {idx} frozen (epoch-0 behavior preserved)", flush=True)

    # --------------------------
    # Freeze ONLY q/k/v projections in selected layers (base weights)
    # Works for LLaMA-family naming (q_proj/k_proj/v_proj).
    # Professor's simpler strategy: only freezes Q/K/V, not o_proj or MLP
    # --------------------------
    if freeze_q_layers or freeze_k_layers or freeze_v_layers:
        qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
        print(f"🧊 Freezing projections: Q={sorted(qset)} K={sorted(kset)} V={sorted(vset)}", flush=True)
        if freeze_o_with_qkv:
            print("   ➕ Also freezing o_proj in any layer present in Q/K/V sets (K+O, Q+O, V+O enabled)", flush=True)

        for name, p in model.named_parameters():
            # Match layer index in parameter name (common patterns: ".layers.{i}.")
            hit_layer = None
            for i in (qset | kset | vset):
                if f".layers.{i}." in name:
                    hit_layer = i
                    break
            if hit_layer is None:
                continue

            # Freeze selectively by projection
            if hit_layer in qset and ".q_proj." in name:
                p.requires_grad = False
            if hit_layer in kset and ".k_proj." in name:
                p.requires_grad = False
            if hit_layer in vset and ".v_proj." in name:
                p.requires_grad = False
            # Optionally also freeze o_proj in those layers (no MLP)
            if freeze_o_with_qkv and ".o_proj." in name and hit_layer in (qset | kset | vset):
                p.requires_grad = False

    # ---- Apply LoRA on q/k/v, and disable LoRA params in frozen layers ----
    if use_lora:
        print("⚙️  Applying LoRA: q/k/v only (disable LoRA in frozen layers)", flush=True)

        lcfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lcfg)

        # Disable LoRA params in fully frozen layers
        if freeze_layers:
            lora_frozen_count = 0
            for name, p in model.named_parameters():
                if "lora_" not in name:
                    continue
                for layer_idx in freeze_layers:
                    if f".layers.{layer_idx}." in name:
                        p.requires_grad = False
                        lora_frozen_count += 1
                        print(f"   → Frozen LoRA: layer {layer_idx} ({name})", flush=True)
                        break
            if lora_frozen_count > 0:
                print(f"✅ Frozen {lora_frozen_count} LoRA parameters in fully frozen layers", flush=True)

        # Also disable LoRA params for q/k/v projections in specified layers
        if freeze_q_layers or freeze_k_layers or freeze_v_layers:
            qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)

            for name, p in model.named_parameters():
                if "lora_" not in name:
                    continue

                # Find layer idx match
                layer_hit = None
                for i in (qset | kset | vset):
                    if f".layers.{i}." in name:
                        layer_hit = i
                        break
                if layer_hit is None:
                    continue

                # Freeze LoRA for the specific projection in that layer
                if layer_hit in qset and "q_proj" in name:
                    p.requires_grad = False
                if layer_hit in kset and "k_proj" in name:
                    p.requires_grad = False
                if layer_hit in vset and "v_proj" in name:
                    p.requires_grad = False

        model.print_trainable_parameters()

    # Verification: Print summary of frozen parameters
    if freeze_layers or freeze_q_layers or freeze_k_layers or freeze_v_layers:
        frozen_params = []
        trainable_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                frozen_params.append(name)
            else:
                trainable_params.append(name)
        print(f"📊 Freezing verification: {len(frozen_params)} frozen, {len(trainable_params)} trainable", flush=True)
        if frozen_params:
            print(f"   Frozen params (first 10): {frozen_params[:10]}", flush=True)
            if len(frozen_params) > 10:
                print(f"   ... and {len(frozen_params) - 10} more", flush=True)
        # Store frozen param names for epoch verification
        model._frozen_param_names = frozen_params

    if freeze_qkv_no_grad and (freeze_q_layers or freeze_k_layers or freeze_v_layers):
        qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
        print("⚠️  freeze_qkv_no_grad enabled: detaching q/k/v outputs to skip backward for those ops.", flush=True)
        if freeze_o_with_qkv:
            print("   ➕ Also applying no_grad() wrapper to o_proj in frozen layers", flush=True)
        if "transformer_layers" not in locals():
            try:
                transformer_layers = model.transformer.layers  # Qwen-like
            except AttributeError:
                transformer_layers = model.model.layers        # LLaMA-like
        _apply_no_grad_wrappers(transformer_layers, qset, kset, vset, also_o_proj=freeze_o_with_qkv)

    # Match HotpotQA behavior: enable gradient checkpointing after freezing / LoRA setup
    print("   [load_model_and_tokenizer] Enabling gradient checkpointing...", flush=True)
    model.gradient_checkpointing_enable()
    print("   [load_model_and_tokenizer] ✅ Gradient checkpointing enabled", flush=True)
    
    print("   [load_model_and_tokenizer] ✅ Model loading complete!", flush=True)
    return model, tok




# ---------- Main ----------
def main():
    print("=" * 80, flush=True)
    print("🚀 STARTING IMDB FINETUNING PIPELINE", flush=True)
    print("=" * 80, flush=True)
    
    # Start overall timing
    pipeline_start = time.perf_counter()
    
    # STEP 1: Parse arguments
    print("\n[STEP 1/10] 📋 Parsing command-line arguments...", flush=True)
    args = parse_args()
    
    # Initialize timing tracker based on argument
    global _timing_tracker
    _timing_tracker = TimingTracker(enabled=getattr(args, "enable_timing", False))
    if _timing_tracker.enabled:
        print("   ⏱️  Timing measurements: ENABLED", flush=True)
    else:
        print("   ⏱️  Timing measurements: DISABLED", flush=True)
    
    print(f"   ✅ Arguments parsed:", flush=True)
    print(f"      - Model: {args.model_name}", flush=True)
    print(f"      - Output dir: {args.output_dir}", flush=True)
    print(f"      - Batch size: {args.batch_size}", flush=True)
    print(f"      - Epochs: {args.epochs}", flush=True)
    print(f"      - Learning rate: {args.learning_rate}", flush=True)
    print(f"      - Use LoRA: {getattr(args, 'use_lora', False)}", flush=True)
    print(f"      - Freeze layers: {getattr(args, 'freeze_layers', [])}", flush=True)
    print(f"      - Freeze Q layers: {getattr(args, 'freeze_q_layers', [])}", flush=True)
    print(f"      - Freeze K layers: {getattr(args, 'freeze_k_layers', [])}", flush=True)
    print(f"      - Freeze V layers: {getattr(args, 'freeze_v_layers', [])}", flush=True)
    print(f"      - Freeze o_proj with Q/K/V: {getattr(args, 'freeze_o_with_qkv', False)}", flush=True)
    
    # STEP 2: Initialize wandb
    print("\n[STEP 2/10] 📊 Initializing Weights & Biases...", flush=True)
    # Optional W&B auth via environment (keeps script anonymous/repo-safe).
    wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
    if wandb_key:
        wandb.login(key=wandb_key, relogin=True)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "topotuner"),
        name=f"imdb-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity=os.environ.get("WANDB_ENTITY", None),
    )
    print(f"   ✅ WandB initialized: {wandb.run.name if wandb.run else 'N/A'}", flush=True)
    run_eval = False

    # STEP 3: Load dataset
    print("\n[STEP 3/10] 📚 Loading IMDB dataset...", flush=True)
    ds = load_dataset("stanfordnlp/imdb")
    test_ds = ds["test"]
    print(f"   ✅ Dataset loaded: {len(ds['train'])} train, {len(ds['test'])} test", flush=True)

    # STEP 4: Prepare training dataset
    print("\n[STEP 4/10] 🔄 Preparing training dataset...", flush=True)
    if getattr(args, "train_csv", ""):
        import pandas as pd
        from datasets import Dataset
        print(f"   📄 Loading from CSV: {args.train_csv}", flush=True)
        train_df = pd.read_csv(args.train_csv)
        if "text" not in train_df.columns or "label" not in train_df.columns:
            raise ValueError(f"CSV at {args.train_csv} must contain 'text' and 'label' columns")
        train_full = Dataset.from_pandas(train_df, preserve_index=False)
        print(f"   ✅ Loaded {len(train_full)} samples from CSV", flush=True)
    else:
        train_full = ds["train"]
        print(f"   ✅ Using full IMDB train set: {len(train_full)} samples", flush=True)

    # Shuffle dataset
    print("   🔀 Shuffling dataset (seed=42)...", flush=True)
    train_full = train_full.shuffle(seed=42)
    
    # Limit training samples for testing (if TRAIN_LIMIT env var is set)
    train_limit = int(os.environ.get("TRAIN_LIMIT", "0"))
    if train_limit > 0:
        print(f"   ⚠️  LIMITING training dataset to {train_limit} samples (for testing)", flush=True)
        original_size = len(train_full)
        train_full = train_full.select(range(min(train_limit, len(train_full))))
        print(f"   ✅ Reduced from {original_size} to {len(train_full)} samples", flush=True)
    else:
        print(f"   ✅ Using full dataset: {len(train_full)} samples", flush=True)
    
    # Create train/val split
    print("   ✂️  Creating train/val split (95/5)...", flush=True)
    split = train_full.train_test_split(test_size=0.05, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    print(f"   ✅ Final split: Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}", flush=True)

    # STEP 5: Load model and tokenizer (CRITICAL: Freezing happens here!)
    print("\n[STEP 5/10] 🤖 Loading model and tokenizer (FREEZING WILL OCCUR HERE)...", flush=True)
    print("   ⚠️  CRITICAL: This is where freezing happens - ONCE before training starts!", flush=True)
    with _timing_tracker.time_block("Model Loading"):
    model, tok = load_model_and_tokenizer(
        args.model_name,
        args.use_lora,
        freeze_layers=getattr(args, "freeze_layers", []),
        freeze_q_layers=getattr(args, "freeze_q_layers", []),
        freeze_k_layers=getattr(args, "freeze_k_layers", []),
        freeze_v_layers=getattr(args, "freeze_v_layers", []),
            freeze_qkv_no_grad=getattr(args, "freeze_qkv_no_grad", False),
            freeze_o_with_qkv=getattr(args, "freeze_o_with_qkv", False),
    )
    print("   ✅ Model and tokenizer loaded, freezing completed", flush=True)
    
    # STEP 6: Infer prompt format
    print("\n[STEP 6/10] 📝 Inferring prompt format...", flush=True)
    pf = infer_prompt_format_from_model_id(args.model_name)
    print(f"   ✅ Prompt format: {pf}", flush=True)

    # STEP 7: Tokenize datasets
    print("\n[STEP 7/10] 🔤 Tokenizing datasets...", flush=True)
    print("   🔄 Tokenizing training set...", flush=True)
    with _timing_tracker.time_block("Tokenization (Train)"):
    tokenized_train = train_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=True),
        remove_columns=train_ds.column_names
    )
    print(f"   ✅ Training set tokenized: {len(tokenized_train)} examples", flush=True)
    
    print("   🔄 Tokenizing validation set...", flush=True)
    with _timing_tracker.time_block("Tokenization (Val)"):
    tokenized_val = val_ds.map(
        lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=False),
        remove_columns=val_ds.column_names
    )
    print(f"   ✅ Validation set tokenized: {len(tokenized_val)} examples", flush=True)

    # STEP 8: Setup training arguments
    print("\n[STEP 8/10] ⚙️  Setting up training arguments...", flush=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./IMDB/logs/{timestamp}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy="no",  # Disabled to speed up training
        save_strategy="epoch" if args.save_every_epoch else "no",
        load_best_model_at_end=False,  # Disabled since eval is off
        metric_for_best_model=None,  # Disabled since eval is off
        greater_is_better=False,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        optim="paged_adamw_32bit",
        bf16=False,
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,
        disable_tqdm=False,
        dataloader_pin_memory=True,
        label_names=["labels"],
    )
    print(f"   ✅ Training arguments configured:", flush=True)
    print(f"      - Epochs: {args.epochs} (VERIFY THIS IS CORRECT!)", flush=True)
    print(f"      - Batch size: {args.batch_size}", flush=True)
    print(f"      - Learning rate: {args.learning_rate}", flush=True)
    print(f"      - Save strategy: {'epoch' if args.save_every_epoch else 'no'}", flush=True)

    # Build safe names
    safe_model = args.model_name.replace("/", "_")
    safe_dataset = args.dataset_name.replace("/", "_")
    log_jsonl = os.path.join(
        args.output_dir,
        f"{safe_dataset}_{safe_model}_downstream_eval.jsonl"
    )
    log_tsv = os.path.join(
        args.output_dir,
        f"{safe_dataset}_{safe_model}_downstream_eval.tsv"
    )
    run_name = wandb.run.name if wandb.run else ""

    # Callbacks
    callbacks = [
        AccuracyPerEpochCallback(
            tok,
            run_eval=False,
            split="test",
            limit=None,
            max_new_tokens=50,
            log_jsonl=log_jsonl,
            log_tsv=log_tsv,
            dataset=args.dataset_name,
            model=args.model_name
        ),
        LossDebugCallback(),
]
    # Add step-level timing callback if timing is enabled
    if _timing_tracker and _timing_tracker.enabled:
        callbacks.append(TrainingStepTimingCallback())
    if args.save_every_epoch or args.save_npy:
        callbacks.insert(0, SavePeftModelCallback(args, tokenizer=tok))  # save first

    # STEP 9: Create Trainer
    print("\n[STEP 9/10] 🏋️  Creating Trainer...", flush=True)
    collator = partial(custom_data_collator, tokenizer=tok)
    
    # Use DetailedTimingTrainer if timing is enabled, otherwise use regular Trainer
    TrainerClass = DetailedTimingTrainer if (_timing_tracker and _timing_tracker.enabled) else Trainer
    if TrainerClass == DetailedTimingTrainer:
        print("   ⏱️  Using DetailedTimingTrainer to track forward/backward/optimizer separately", flush=True)
    
    trainer = TrainerClass(
        model=model,
        tokenizer=tok,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=None,  # Disabled to speed up training
        data_collator=collator,
        callbacks=callbacks,
    )
    print("   ✅ Trainer created", flush=True)
    print(f"   📋 Callbacks attached: {len(trainer.callback_handler.callbacks)} callbacks", flush=True)
    for i, cb in enumerate(trainer.callback_handler.callbacks):
        print(f"      {i+1}. {type(cb).__name__}", flush=True)

    # Save baseline as epoch 0 (before training) when epoch saving is active
    if getattr(args, "save_every_epoch", False) or getattr(args, "save_npy", False):
        base_dir = os.path.join(args.output_dir, "epoch_weights")
        os.makedirs(base_dir, exist_ok=True)
        save_dir = os.path.join(base_dir, "checkpoint-epoch-0")
        if not os.path.exists(save_dir):
            print(f">>> Saving baseline as epoch-0 to {save_dir}", flush=True)
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            if tok:
                tok.save_pretrained(save_dir)
            import torch as _torch
            _torch.save(args, os.path.join(save_dir, "training_args.bin"))
            if getattr(args, "save_npy", False):
                import numpy as _np
                npy_dir = os.path.join(save_dir, "numpy_weights")
                os.makedirs(npy_dir, exist_ok=True)
                count = 0
                for name, param in model.named_parameters():
                    if args.use_lora:
                        if "lora_A" in name or "lora_B" in name:
                            short = concise_lora_filename(name)
                            if short:
                                arr = param.detach().cpu().to(_torch.float16)
                                _np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                                count += 1
                    else:
                        if param.requires_grad:
                            short = concise_full_filename(name)
                            if short:
                                arr = param.detach().cpu().to(_torch.float16)
                                _np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                                count += 1
                print(f">>> Saved {count} numpy weight files to {npy_dir}", flush=True)
        else:
            print(f">>> Epoch-0 already exists at {save_dir}, skipping baseline save", flush=True)

    # Create optimizer and scheduler
    print("\n   🔧 Creating optimizer and scheduler...", flush=True)
    _ = trainer.create_optimizer_and_scheduler(num_training_steps=training_args.max_steps)
    print(f"   ✅ Optimizer: {type(trainer.optimizer).__name__}", flush=True)
    print(f"   ✅ Scheduler: {type(trainer.lr_scheduler).__name__}", flush=True)

    # Calculate training steps
    train_dl = trainer.get_train_dataloader()
    steps_per_epoch = len(train_dl)
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(f"\n   📊 TRAINING PLAN (VERIFY THESE NUMBERS!):", flush=True)
    print(f"      - Steps per epoch: {steps_per_epoch}", flush=True)
    print(f"      - Number of epochs: {training_args.num_train_epochs} ⚠️  VERIFY THIS!", flush=True)
    print(f"      - Total update steps: {total_update_steps}", flush=True)
    
    # Verify frozen parameters before training
    if hasattr(model, '_frozen_param_names') and model._frozen_param_names:
        print(f"\n   🧊 PRE-TRAINING FREEZE VERIFICATION:", flush=True)
        print(f"      - Expected frozen params: {len(model._frozen_param_names)}", flush=True)
        still_frozen = sum(1 for name in model._frozen_param_names 
                          for p_name, p in model.named_parameters() 
                          if p_name == name and not p.requires_grad)
        print(f"      - Actually frozen params: {still_frozen}", flush=True)
        if still_frozen == len(model._frozen_param_names):
            print(f"      ✅ All frozen parameters confirmed frozen before training!", flush=True)
        else:
            print(f"      ⚠️  WARNING: {len(model._frozen_param_names) - still_frozen} params became unfrozen!", flush=True)

    # STEP 10: Start training
    print("\n[STEP 10/10] 🚀 STARTING TRAINING...", flush=True)
    print("=" * 80, flush=True)
    print("⚠️  REMINDER: Freezing happened ONCE in Step 5, not each epoch!", flush=True)
    print("⚠️  Frozen parameters should remain frozen throughout all epochs!", flush=True)
    print("=" * 80, flush=True)
    with _timing_tracker.time_block("Training (Total)"):
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print("\n" + "=" * 80, flush=True)
    print("✅ TRAINING COMPLETED", flush=True)
    print("=" * 80, flush=True)
    
    # Calculate and print overall pipeline timing
    pipeline_end = time.perf_counter()
    pipeline_total = pipeline_end - pipeline_start
    _timing_tracker.timings["Pipeline (Overall)"] = [pipeline_total]
    
    # Print timing summary
    _timing_tracker.print_summary()

    # # Final eval (optional)
    # if run_eval:
    #     final = evaluate_imdb(
    #         model, tok,
    #         split="test",
    #         limit=None,
    #         max_new_tokens=50,
    #         progress_bar=True,
    #         save_jsonl=log_jsonl,
    #         save_tsv=log_tsv,
    #         run_name=run_name,
    #         phase="final",
    #         epoch=int(training_args.num_train_epochs),
    #         step=int(trainer.state.global_step),
    #         output_dir=training_args.output_dir,
    #     )
    #     print(f"[Final] IMDB test Accuracy={final['accuracy']:.2f}% Pos={final['positive_acc']:.2f}% Neg={final['negative_acc']:.2f}% n={final['n']}", flush=True)

    # Delete Hugging Face default checkpoints
    for path in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        print(f"🗑️ Removing default checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)

    # Delete final_model if it exists
    final_model_path = os.path.join(args.output_dir, "final_model")
    if os.path.exists(final_model_path):
        print(f"🗑️ Removing final model folder: {final_model_path}")
        shutil.rmtree(final_model_path, ignore_errors=True)

if __name__ == "__main__":
    main()

"""
export CUDA_VISIBLE_DEVICES=0
IMDB Sentiment Analysis Examples:

Llama-3.1-8B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name meta-llama/Llama-3.1-8B \
  --use-lora \
  --output-dir ./numpy_weights/imdb/llama31_8b/lora \
  --batch-size 8 --epochs 3 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Llama-3.1-8B_lora.log 2>&1 &

Llama-3.2-3B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name meta-llama/Llama-3.2-3B \
  --output-dir ./numpy_weights/imdb/llama32_3b/full \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Llama-3.2-3B_full.log 2>&1 &

Llama-3.2-3B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name meta-llama/Llama-3.2-3B \
  --use-lora \
  --output-dir ./numpy_weights/imdb/llama32_3b/lora \
  --batch-size 8 --epochs 6 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Llama-3.2-3B_lora.log 2>&1 &

-----

Mistral-7B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name mistralai/Mistral-7B-v0.1 \
  --use-lora \
  --output-dir ./numpy_weights/imdb/mistral7b/lora \
  --batch-size 8 --epochs 3 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Mistral-7B_lora.log 2>&1 &

Qwen-3-8B LoRA:
nohup python -m codes.imdb.finetuning_imdb \
  --dataset-name IMDB \
  --model-name Qwen/Qwen3-8B \
  --use-lora \
  --output-dir ./numpy_weights/imdb/qwen_8b/lora \
  --batch-size 8 --epochs 3 --gradient_accumulation_steps 1 \
  --learning-rate 1e-5 --patience 2 \
  --save-every-epoch --save-npy \
  > logs/finetune_IMDB_Qwen3-8B_lora.log 2>&1 &
"""
