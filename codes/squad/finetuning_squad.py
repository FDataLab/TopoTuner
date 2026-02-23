import json
import os
import glob
import shutil
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
from peft import LoraConfig, get_peft_model, TaskType

from .data_preprocessing_squad import (
    preprocess_dataset, custom_data_collator,
    infer_prompt_format_from_model_id,
)
from .eval_squad import evaluate_squad

from codes.utils.args import parse_args
from codes.utils.model_saving import (
    SavePeftModelCallback,
    concise_lora_filename,
    concise_full_filename,
)

from transformers.utils import logging as hf_logging
hf_logging.enable_progress_bar()

import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# =========================================================
# Timing Tracker
# =========================================================
class TimingTracker:
    """Lightweight timing tracker with minimal overhead."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings = {}
        self.overhead_samples = []
        self._overhead_measured = False
        self._timing_overhead_total = 0.0
    
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
        
        timing_start = time.perf_counter()
        self._measure_overhead()
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            timing_end = time.perf_counter()
            
            elapsed = end - start
            timing_overhead = (timing_end - timing_start) - elapsed
            self._timing_overhead_total += max(0, timing_overhead)
            
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
        
        timing_keys = [k for k in self.timings.keys() if k != "Pipeline (Overall)"]
        total_time = sum(sum(self.timings[k]) for k in timing_keys)
        
        step_timing_key = "Training (Forward+Backward+Optimizer per step)"
        for name in sorted(timing_keys):
            if name == step_timing_key:
                continue
            times = self.timings[name]
            total = sum(times)
            avg = total / len(times) if times else 0
            pct = (total / total_time * 100) if total_time > 0 else 0
            count_str = f" (x{len(times)})" if len(times) > 1 else ""
            print(f"   {name:35s}: {self.format_time(total):>12s} (avg: {self.format_time(avg)}{count_str}, {pct:5.1f}%)", flush=True)
        
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
                print(f"      {'  → Min per step':33s}: {self.format_time(min_step):>12s}", flush=True)
                print(f"      {'  → Max per step':33s}: {self.format_time(max_step):>12s}", flush=True)
        
        print("=" * 80, flush=True)

# Global timing tracker (initialized in main)
_timing_tracker = None

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

class DetailedTimingTrainer(Trainer):
    """Custom Trainer that tracks forward, backward, and optimizer timing separately."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch_forward_times = {}
        self._epoch_backward_times = {}
        self._epoch_optimizer_times = {}
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to time forward, backward, and optimizer separately."""
        global _timing_tracker
        
        if not (_timing_tracker and _timing_tracker.enabled):
            return super().training_step(model, inputs, num_items_in_batch)
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Time forward pass
        forward_start = time.perf_counter()
        loss = self.compute_loss(model, inputs)
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        _timing_tracker.timings.setdefault("Training (Forward Pass)", []).append(forward_time)
        
        epoch_num = int(self.state.epoch) if self.state.epoch is not None else 0
        if epoch_num not in self._epoch_forward_times:
            self._epoch_forward_times[epoch_num] = []
        self._epoch_forward_times[epoch_num].append(forward_time)
        
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
                import apex  # type: ignore
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                loss.backward()
        else:
            loss.backward()
        backward_end = time.perf_counter()
        backward_time = backward_end - backward_start
        _timing_tracker.timings.setdefault("Training (Backward Pass)", []).append(backward_time)
        
        if epoch_num not in self._epoch_backward_times:
            self._epoch_backward_times[epoch_num] = []
        self._epoch_backward_times[epoch_num].append(backward_time)
        
        # Time optimizer step
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
            
            if trainer is not None and isinstance(trainer, DetailedTimingTrainer):
                trainer.log_epoch_timing_breakdown(epoch_num)


class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] "
                  f"Loss: {logs['loss']:.6f}", flush=True)

class EMF1PerEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, run_eval: bool = True, split: str = "validation",
                 limit=None, max_new_tokens: int = 12,  # Short answers only (SQuAD answers are typically 1-5 words)
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

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self.run_eval:
            return
        metrics = evaluate_squad(
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
            "em": metrics["em"],
            "f1": metrics["f1"],
            "n": metrics["n"],
            **gpu_info
        }
        print(
          f"[Downstream] epoch={record['epoch']} SQuAD {self.split} "
          f"EM={record['em']:.2f}% F1={record['f1']:.2f}% n={record['n']} "
          f"GPU={gpu_info['gpu']} mem={gpu_info['mem_alloc_MB']}MB",
          flush=True
        )

        if self.log_jsonl:
            os.makedirs(os.path.dirname(self.log_jsonl), exist_ok=True)
            with open(self.log_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")
        if self.log_tsv:
            import csv
            new_file = not os.path.exists(self.log_tsv)
            with open(self.log_tsv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys(), delimiter="\t")
                if new_file:
                    writer.writeheader()
                writer.writerow(record)

def load_model_and_tokenizer(
    model_id: str,
    use_lora: bool,
    freeze_layers=None,
    freeze_q_layers=None,
    freeze_k_layers=None,
    freeze_v_layers=None,
    freeze_qkv_no_grad: bool = False,
    freeze_o_with_qkv: bool = False,
    freeze_mlp: bool = False,
    freeze_mlp_no_grad: bool = False,
    freeze_mlp_layers=None,
):
    freeze_layers = freeze_layers or []
    freeze_q_layers = freeze_q_layers or []
    freeze_k_layers = freeze_k_layers or []
    freeze_v_layers = freeze_v_layers or []
    freeze_mlp_layers = freeze_mlp_layers or []

    def _wrap_no_grad(module, label):
        original_forward = module.forward

        def forward_no_grad(*args, **kwargs):
            with torch.no_grad():
                return original_forward(*args, **kwargs)

        module.forward = forward_no_grad
        print(f"   → {label} forward wrapped with no_grad()", flush=True)

    def _apply_no_grad_wrappers(
        transformer_layers,
        q_layers=None,
        k_layers=None,
        v_layers=None,
        mlp_layers=None,
        also_o_proj: bool = False,
    ):
        """
        Wrap selected submodules (Q/K/V/O and MLP) with no_grad() in their forward pass.
        This is the extended professor-style helper, supporting both attention heads
        and MLP blocks for LLaMA/Qwen-style architectures.
        """
        q_layers = q_layers or set()
        k_layers = k_layers or set()
        v_layers = v_layers or set()
        mlp_layers = mlp_layers or set()

        for idx, layer in enumerate(transformer_layers):
            # -------- Attention block (Q/K/V/O) --------
            if idx in (q_layers | k_layers | v_layers):
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
                if attn is not None:
                    if idx in q_layers and hasattr(attn, "q_proj"):
                        _wrap_no_grad(attn.q_proj, f"Layer {idx} q_proj")
                    if idx in k_layers and hasattr(attn, "k_proj"):
                        _wrap_no_grad(attn.k_proj, f"Layer {idx} k_proj")
                    if idx in v_layers and hasattr(attn, "v_proj"):
                        _wrap_no_grad(attn.v_proj, f"Layer {idx} v_proj")
                    # Optionally also wrap o_proj (K+O, Q+O, V+O experiments)
                    if also_o_proj and hasattr(attn, "o_proj"):
                        _wrap_no_grad(attn.o_proj, f"Layer {idx} o_proj")

            # -------- MLP block (gate/up/down) --------
            if idx in mlp_layers:
                mlp = getattr(layer, "mlp", None)
                if mlp is not None:
                    if hasattr(mlp, "gate_proj"):
                        _wrap_no_grad(mlp.gate_proj, f"Layer {idx} mlp.gate_proj")
                    if hasattr(mlp, "up_proj"):
                        _wrap_no_grad(mlp.up_proj, f"Layer {idx} mlp.up_proj")
                    if hasattr(mlp, "down_proj"):
                        _wrap_no_grad(mlp.down_proj, f"Layer {idx} mlp.down_proj")

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        token=HF_TOKEN
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        dtype=torch.bfloat16,   # torch_dtype deprecated -> dtype
        trust_remote_code=True,
        token=HF_TOKEN
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    # Freeze base transformer layers
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
    
    # ===== PROFESSOR'S VERSION (ACTIVE) =====
    # --------------------------
    # Freeze ONLY q/k/v projections in selected layers (base weights)
    # Works for LLaMA-family naming (q_proj/k_proj/v_proj).
    # Professor's simpler strategy (default): only freezes Q/K/V, not o_proj or MLP
    # Optionally also freeze o_proj in those layers when freeze_o_with_qkv=True
    # (enables K+O, Q+O, V+O experiments)
    # Optionally also freeze MLP (gate_proj, up_proj, down_proj) when freeze_mlp=True
    # (enables K+O+MLP, Q+O+MLP, V+O+MLP experiments)
    # --------------------------
    if freeze_q_layers or freeze_k_layers or freeze_v_layers:
        qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
        print(f"🧊 Freezing projections: Q={sorted(qset)} K={sorted(kset)} V={sorted(vset)}", flush=True)
        if freeze_o_with_qkv:
            print("   ➕ Also freezing o_proj in any layer present in Q/K/V sets (K+O, Q+O, V+O enabled)", flush=True)
        if freeze_mlp:
            print("   ➕ Also freezing MLP (gate_proj, up_proj, down_proj) in any layer present in Q/K/V sets (K+O+MLP, Q+O+MLP, V+O+MLP enabled)", flush=True)

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
            # Optionally also freeze o_proj in those layers
            if freeze_o_with_qkv and ".o_proj." in name and hit_layer in (qset | kset | vset):
                p.requires_grad = False
            # Optionally also freeze MLP in those layers (K+O+MLP, Q+O+MLP, V+O+MLP experiments)
            if freeze_mlp and hit_layer in (qset | kset | vset):
                if ".gate_proj." in name or ".up_proj." in name or ".down_proj." in name:
                    p.requires_grad = False

    # LoRA inject q/k/v and disable LoRA params in frozen layers
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

        if freeze_layers:
            for name, p in model.named_parameters():
                if "lora_" not in name:
                    continue
                for layer_idx in freeze_layers:
                    if f".layers.{layer_idx}." in name:
                        p.requires_grad = False
                        break
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

    # ===== PROFESSOR'S VERSION (EXTENDED) =====
    if (
        freeze_qkv_no_grad and (freeze_q_layers or freeze_k_layers or freeze_v_layers)
    ) or (
        freeze_mlp_no_grad and freeze_mlp_layers
    ):
        qset = set(freeze_q_layers or [])
        kset = set(freeze_k_layers or [])
        vset = set(freeze_v_layers or [])
        mlpset = set(freeze_mlp_layers or [])

        print("⚠️  no_grad wrappers enabled:", flush=True)
        if qset or kset or vset:
            print(f"   • QKV layers: {sorted(qset | kset | vset)}", flush=True)
            if freeze_o_with_qkv:
                print("   • o_proj also wrapped with no_grad in those layers", flush=True)
        if mlpset:
            print(f"   • MLP layers: {sorted(mlpset)}", flush=True)

        if "transformer_layers" not in locals():
            try:
                transformer_layers = model.transformer.layers  # Qwen-like
            except AttributeError:
                transformer_layers = model.model.layers        # LLaMA-like

        _apply_no_grad_wrappers(
            transformer_layers,
            q_layers=qset,
            k_layers=kset,
            v_layers=vset,
            mlp_layers=mlpset,
            also_o_proj=freeze_o_with_qkv,
        )

    model.gradient_checkpointing_enable()
    return model, tok


# =========================================================
# Baseline saving (epoch-0) preserved
# =========================================================
def save_epoch0_baseline_if_needed(args, model, tok):
    if not (getattr(args, "save_every_epoch", False) or getattr(args, "save_npy", False)):
        return

    base_dir = os.path.join(args.output_dir, "epoch_weights")
    os.makedirs(base_dir, exist_ok=True)
    save_dir = os.path.join(base_dir, "checkpoint-epoch-0")

    if os.path.exists(save_dir):
        print(f">>> Epoch-0 already exists at {save_dir}, skipping baseline save", flush=True)
        return

    print(f">>> Saving baseline as epoch-0 to {save_dir}", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    import torch as _torch
    _torch.save(args, os.path.join(save_dir, "training_args.bin"))

    if not getattr(args, "save_npy", False):
        return

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

    print(f"✅ [baseline] Saved {count} tensors (float16) to: {npy_dir}", flush=True)


def cleanup_checkpoints(args):
    # Delete Hugging Face default checkpoints
    for path in glob.glob(os.path.join(args.output_dir, "checkpoint-*")):
        print(f"🗑️ Removing default checkpoint: {path}", flush=True)
        shutil.rmtree(path, ignore_errors=True)

    # Delete final_model if it exists (align with your IMDB behavior)
    final_model_path = os.path.join(args.output_dir, "final_model")
    if os.path.exists(final_model_path):
        print(f"🗑️ Removing final model folder: {final_model_path}", flush=True)
        shutil.rmtree(final_model_path, ignore_errors=True)


def main():
    args = parse_args()
    
    # Initialize timing tracker based on argument
    global _timing_tracker
    _timing_tracker = TimingTracker(enabled=getattr(args, "enable_timing", False))
    if _timing_tracker.enabled:
        print("   ⏱️  Timing measurements: ENABLED", flush=True)
    else:
        print("   ⏱️  Timing measurements: DISABLED", flush=True)
    
    pipeline_start = time.perf_counter()

    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"squad-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )
    run_eval = False  # Disable evaluation during training (match HotpotQA)

    subset_dir = args.subset_save_dir
    if subset_dir and os.path.exists(os.path.join(subset_dir, "state.json")):
        from datasets import DatasetDict, load_from_disk
        print(f">>> Loading SQuAD subset from {subset_dir}")
        ds = load_from_disk(subset_dir)
        train_full, val_ds = ds["train"], ds["validation"]
    else:
        ds = load_dataset("squad")
        train_full, val_ds = ds["train"], ds["validation"]
        subset_size = min(args.subset_train_size, len(train_full))
        print(f">>> Sampling subset of {subset_size} from {len(train_full)} with seed {args.subset_seed}")
        train_full = train_full.shuffle(seed=args.subset_seed).select(range(subset_size))
        if subset_dir:
            from datasets import DatasetDict
            os.makedirs(subset_dir, exist_ok=True)
            print(f">>> Saving SQuAD subset to {subset_dir}")
            DatasetDict({"train": train_full, "validation": val_ds}).save_to_disk(subset_dir)

    split = train_full.train_test_split(test_size=0.05, seed=args.subset_seed)
    train_ds, dev_ds = split["train"], split["test"]
    print(f"Train {len(train_ds)} | Dev {len(dev_ds)} | Val {len(val_ds)}", flush=True)

    # Model + tokenizer
    with _timing_tracker.time_block("Model Loading"):
        model, tok = load_model_and_tokenizer(
        args.model_name,
        args.use_lora,
        freeze_layers=args.freeze_layers,
        freeze_q_layers=getattr(args, "freeze_q_layers", []),
        freeze_k_layers=getattr(args, "freeze_k_layers", []),
        freeze_v_layers=getattr(args, "freeze_v_layers", []),
        freeze_qkv_no_grad=getattr(args, "freeze_qkv_no_grad", False),
        freeze_o_with_qkv=getattr(args, "freeze_o_with_qkv", False),
        freeze_mlp=getattr(args, "freeze_mlp", False),
        freeze_mlp_no_grad=getattr(args, "freeze_mlp_no_grad", False),
        freeze_mlp_layers=getattr(args, "freeze_mlp_layers", []),
        )
    pf = infer_prompt_format_from_model_id(args.model_name)

    with _timing_tracker.time_block("Tokenization"):
        tokenized_train = train_ds.map(
            lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=True),  # SQuAD max is ~517 tokens, 512 will truncate ~0.1% of longest examples
            remove_columns=train_ds.column_names
        )
        tokenized_val = dev_ds.map(
            lambda ex: preprocess_dataset(ex, tok, max_len=1024, prompt_format=pf, is_train=False),  # SQuAD max is ~517 tokens, 512 will truncate ~0.1% of longest examples
            remove_columns=dev_ds.column_names
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./SQuAD/logs/{timestamp}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy="no",
        save_strategy="epoch" if args.save_every_epoch else "no",
        load_best_model_at_end=False,
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

    callbacks = [LossDebugCallback()]
    if args.save_every_epoch or args.save_npy:
        callbacks.insert(0, SavePeftModelCallback(args, tokenizer=tok))
    
    if _timing_tracker and _timing_tracker.enabled:
        callbacks.append(TrainingStepTimingCallback())

    collator = partial(custom_data_collator, tokenizer=tok)
    TrainerClass = DetailedTimingTrainer if (_timing_tracker and _timing_tracker.enabled) else Trainer
    trainer = TrainerClass(
        model=model,
        tokenizer=tok,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        callbacks=callbacks,
    )

    print(">>> Callbacks attached:", trainer.callback_handler.callbacks, flush=True)

    # Save baseline (epoch 0)
    save_epoch0_baseline_if_needed(args, model, tok)

    # Print training plan
    steps_per_epoch = len(trainer.get_train_dataloader())
    total_update_steps = steps_per_epoch * training_args.num_train_epochs
    print(
        f">>> Training plan: steps_per_epoch={steps_per_epoch} x epochs={training_args.num_train_epochs} = total_updates={total_update_steps}",
        flush=True
    )

    # Train
    with _timing_tracker.time_block("Training (Total)"):
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Calculate and print overall pipeline timing
    pipeline_end = time.perf_counter()
    pipeline_total = pipeline_end - pipeline_start
    _timing_tracker.timings["Pipeline (Overall)"] = [pipeline_total]
    
    # Print timing summary
    _timing_tracker.print_summary()

    # Save final then delete
    with _timing_tracker.time_block("Model Saving"):
        final_dir = f"{args.output_dir}/final_model"
        trainer.save_model(final_dir)
        tok.save_pretrained(final_dir)

    cleanup_checkpoints(args)

    print("[Training] Final evaluation disabled. Will evaluate manually later.", flush=True)

if __name__ == "__main__":
    main()

