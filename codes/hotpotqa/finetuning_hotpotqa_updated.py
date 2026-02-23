import os
import json
import glob
import shutil
import datetime
import time
from functools import partial
from collections import Counter
from contextlib import contextmanager
import string

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.utils import logging as hf_logging

from peft import LoraConfig, get_peft_model, TaskType

from .data_preprocessing_hotpotqa import (
    preprocess_dataset,
    custom_data_collator,
    infer_prompt_format_from_model_id,
)
from .eval_hotpotqa import evaluate_hotpotqa

from codes.utils.args import parse_args
from codes.utils.model_saving import (
    SavePeftModelCallback,
    concise_lora_filename,
    concise_full_filename,
)

import wandb

hf_logging.enable_progress_bar()

# If you want, keep these here; otherwise I'd move them to your bash script
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
                print(f"      {'  → Min step':33s}: {self.format_time(min_step):>12s}", flush=True)
                print(f"      {'  → Max step':33s}: {self.format_time(max_step):>12s}", flush=True)
        
        if "Pipeline (Overall)" in self.timings:
            pipeline_time = sum(self.timings["Pipeline (Overall)"])
            print(f"\n   {'Pipeline (Overall)':35s}: {self.format_time(pipeline_time):>12s}", flush=True)
        
        if self.overhead_samples:
            total_calls = sum(len(times) for times in self.timings.values())
            avg_overhead = sum(self.overhead_samples) / len(self.overhead_samples)
            perf_counter_overhead = avg_overhead * total_calls
            total_overhead = self._timing_overhead_total + perf_counter_overhead
            overhead_pct = (total_overhead / total_time * 100) if total_time > 0 else 0
            
            print(f"   {'Timing overhead (measured)':35s}: {self.format_time(self._timing_overhead_total):>12s}", flush=True)
            print(f"   {'Timing overhead (perf_counter)':35s}: {self.format_time(perf_counter_overhead):>12s}", flush=True)
            print(f"   {'Timing overhead (total)':35s}: {self.format_time(total_overhead):>12s} ({overhead_pct:.3f}%)", flush=True)
        
        print("=" * 80 + "\n", flush=True)


# Global timing tracker (will be initialized in main() based on args)
_timing_tracker = None


# =========================================================
# GPU / Logging Helpers
# =========================================================
def get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_id": None, "total_mem_GB": None, "mem_alloc_MB": None, "mem_reserved_MB": None}
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu": props.name,
        "gpu_id": gpu_id,
        "total_mem_GB": round(props.total_memory / 1024**3, 2),
        "mem_alloc_MB": torch.cuda.memory_allocated(gpu_id) // 1024**2,
        "mem_reserved_MB": torch.cuda.memory_reserved(gpu_id) // 1024**2,
    }


class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"[Epoch {state.epoch:.2f} | Step {state.global_step}] Loss: {logs['loss']:.6f}", flush=True)


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


class EMF1PerEpochCallback(TrainerCallback):
    """
    Kept exactly as your functionality, but grouped here.
    """
    def __init__(
        self,
        tokenizer,
        run_eval: bool = False,
        split: str = "validation",
        limit=None,
        max_new_tokens: int = 256,
        log_jsonl=None,
        log_tsv=None,
        dataset="",
        model="",
    ):
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

        metrics = evaluate_hotpotqa(
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
            **gpu_info,
        }

        print(
            f"[Downstream] epoch={record['epoch']} HotpotQA {self.split} "
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


# =========================================================
# EM/F1 Helpers (kept functional, slightly cleaner)
# =========================================================
def _extract_answer_start(labels):
    for i, x in enumerate(labels):
        if x != -100:
            return i
    return None


def _clean_answer_text(text: str) -> str:
    """
    For display + for your simple EM logic.
    - strips
    - if "Answer:" exists, keeps content after it
    - takes only the first line
    """
    if text is None:
        return ""
    text = text.strip()
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    text = text.splitlines()[0]
    return text.strip()


def _normalize_text(s: str) -> str:
    """
    For EM/F1 normalization. Keeps your current behavior.
    """
    s = (s or "").lower().strip()
    if "answer:" in s:
        s = s.split("answer:", 1)[1].strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize_text(pred) == _normalize_text(gold)


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# =========================================================
# Post-train Sample Debug (kept, but cleaned)
# =========================================================
def _configure_generation_for_debug(model, tok):
    """
    Keep generation consistent and avoid gradient checkpoint/cache mismatch warnings.
    """
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    # Prefer llama3 chat end token if it exists
    eot_id = None
    try:
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None

    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = (eot_id if (eot_id is not None and eot_id != tok.eos_token_id) else tok.eos_token_id)

    return eot_id


@torch.no_grad()
def debug_print_samples_after_training(model, tok, raw_ds, tokenized_ds, output_dir: str, n: int = 3):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "hotpot_posttrain_samples.txt")

    idxs = list(range(min(n, len(tokenized_ds))))
    eot_id = _configure_generation_for_debug(model, tok)

    with open(out_path, "w") as f:
        for k, i in enumerate(idxs, start=1):
            raw_ex = raw_ds[i]
            ex = tokenized_ds[i]
            input_ids = ex["input_ids"]
            labels = ex["labels"]

            ans_start = _extract_answer_start(labels)
            if ans_start is None:
                msg = f"[DEBUG] sample {k} (idx={i}): labels all -100, skipping\n"
                print(msg, flush=True)
                f.write(msg)
                continue

            prompt_ids_list = input_ids[:ans_start]
            gold_ids_list = input_ids[ans_start:]

            prompt_text = tok.decode(prompt_ids_list, skip_special_tokens=False)
            gold_text = tok.decode(gold_ids_list, skip_special_tokens=False)

            prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=model.device)
            attn = torch.ones_like(prompt_ids)

            gen = model.generate(
                input_ids=prompt_ids,
                attention_mask=attn,
                max_new_tokens=80,
                do_sample=False,
                eos_token_id=(eot_id if eot_id is not None else tok.eos_token_id),
                pad_token_id=tok.pad_token_id,
            )

            pred_completion = tok.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=False)

            gold_short = _clean_answer_text("Answer: " + str(raw_ex.get("answer", "")))
            pred_short = _clean_answer_text(pred_completion)

            em = _exact_match(pred_short, gold_short)
            f1 = _f1_score(pred_short, gold_short)

            block = (
                f"\n================ POST-TRAIN SAMPLE {k} (idx={i}) ================\n"
                f"[RAW Q] {raw_ex.get('question','')}\n"
                f"[RAW GOLD SHORT] {raw_ex.get('answer','')}\n"
                f"[LEN] prompt={len(prompt_ids_list)} gold={len(gold_ids_list)} total={len(input_ids)}\n\n"
                f"--- INPUT PROMPT (FULL) ---\n{prompt_text}\n\n"
                f"--- GOLD (SUPERVISED TEXT) ---\n{gold_text}\n\n"
                f"--- PRED (completion from prompt-only) ---\n{pred_completion}\n\n"
                f"--- SANITY ---\n"
                f"pred_short = {pred_short}\n"
                f"gold_short = {gold_short}\n"
                f"EM(normalized) = {em}\n"
                f"F1(normalized) = {f1:.4f}\n"
                f"===============================================================\n"
            )

            print(block, flush=True)
            f.write(block)

    print(f"[DEBUG] Saved post-train samples to: {out_path}", flush=True)


# =========================================================
# Model / Tokenizer
# =========================================================
# ===== PROFESSOR'S VERSION (ACTIVE) =====
def load_model_and_tokenizer(
    model_id: str,
    use_lora: bool,
    freeze_layers=None,
    freeze_q_layers=None,
    freeze_k_layers=None,
    freeze_v_layers=None,
    freeze_qkv_no_grad: bool = False,  # ❌ DEPRECATED: kept for compatibility but must be False
    freeze_o_with_qkv: bool = False,
    freeze_mlp: bool = False,
    freeze_mlp_no_grad: bool = False,  # ❌ DEPRECATED: kept for compatibility but must be False
    freeze_mlp_layers=None,
):
    # ✅ Safety checks: no_grad flags break gradient flow to K/MLP
    assert not freeze_qkv_no_grad, "❌ ERROR: --freeze-qkv-no-grad blocks gradient flow to K/MLP. Use ONLY requires_grad=False."
    assert not freeze_mlp_no_grad, "❌ ERROR: --freeze-mlp-no-grad blocks gradient flow. Use ONLY requires_grad=False."
    
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

# ===== KADIR'S VERSION (COMMENTED OUT - PRESERVED FOR REFERENCE) =====
# def load_model_and_tokenizer(model_id: str, use_lora: bool, freeze_layers=None, freeze_q_layers=None, freeze_k_layers=None, freeze_v_layers=None):
#     freeze_layers = freeze_layers or []
#     freeze_q_layers = freeze_q_layers or []
#     freeze_k_layers = freeze_k_layers or []
#     freeze_v_layers = freeze_v_layers or []

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
        token=HF_TOKEN
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # ===== PROFESSOR'S VERSION (ACTIVE) =====
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

    # ===== KADIR'S VERSION (COMMENTED OUT - PRESERVED FOR REFERENCE) =====
    # tok = AutoTokenizer.from_pretrained(
    #     model_id,
    #     trust_remote_code=True,
    #     padding_side="right",
    #     token=HF_TOKEN
    # )
    # if tok.pad_token is None:
    #     tok.pad_token = tok.eos_token
    #     tok.pad_token_id = tok.eos_token_id
    #
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map={"": 0},
    #     dtype=torch.bfloat16,   # torch_dtype deprecated -> dtype
    #     trust_remote_code=True,
    #     token=HF_TOKEN
    # )
    #
    # if model.config.pad_token_id is None:
    #     model.config.pad_token_id = tok.pad_token_id
    # model.config.use_cache = False

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
    # --------------------------
    if freeze_q_layers or freeze_k_layers or freeze_v_layers:
        qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
        print(f"🧊 Freezing projections: Q={sorted(qset)} K={sorted(kset)} V={sorted(vset)}", flush=True)
        if freeze_o_with_qkv:
            print("   ➕ Also freezing o_proj in any layer present in Q/K/V sets (K+O, Q+O, V+O enabled)", flush=True)
        if freeze_mlp:
            print("   ➕ Also freezing MLP (gate_proj, up_proj, down_proj) in any layer present in Q/K/V sets (K+MLP, Q+MLP, V+MLP enabled)", flush=True)

        # Get all layer indices (0-31 for Llama-3.1-8B)
        try:
            transformer_layers = model.transformer.layers  # Qwen-like
        except AttributeError:
            transformer_layers = model.model.layers        # LLaMA-like
        all_layer_indices = set(range(len(transformer_layers)))

        for name, p in model.named_parameters():
            # Match layer index in parameter name (common patterns: ".layers.{i}.")
            hit_layer = None
            for i in all_layer_indices:
                if f".layers.{i}." in name:
                    hit_layer = i
                    break
            if hit_layer is None:
                continue

            # Determine if this parameter should be frozen
            should_freeze = False
            
            # Freeze selectively by projection
            if hit_layer in qset and ".q_proj." in name:
                should_freeze = True
            elif hit_layer in kset and ".k_proj." in name:
                should_freeze = True
            elif hit_layer in vset and ".v_proj." in name:
                should_freeze = True
            # Optionally also freeze o_proj in those layers
            elif freeze_o_with_qkv and ".o_proj." in name and hit_layer in (qset | kset | vset):
                should_freeze = True
            # Optionally also freeze MLP in specified layers
            elif hit_layer in set(freeze_mlp_layers):
                if ".gate_proj." in name or ".up_proj." in name or ".down_proj." in name:
                    should_freeze = True
            
            # Apply the freezing decision
            p.requires_grad = not should_freeze
            
            # Debug: Print first few K projections to verify
            if ".k_proj." in name and hit_layer in [0, 1, 31]:
                status = "FROZEN" if should_freeze else "TRAINABLE"
                print(f"   [DEBUG] Layer {hit_layer} k_proj: {status} (requires_grad={p.requires_grad})", flush=True)

    # ===== KADIR'S VERSION (COMMENTED OUT - PRESERVED FOR REFERENCE) =====
    # # --------------------------
    # # Freeze ONLY q/k/v projections in selected layers (base weights)
    # # Works for LLaMA-family naming (q_proj/k_proj/v_proj).
    # # --------------------------
    # if freeze_q_layers or freeze_k_layers or freeze_v_layers:
    #     qset, kset, vset = set(freeze_q_layers), set(freeze_k_layers), set(freeze_v_layers)
    #     print(f"🧊 Freezing projections: Q={sorted(qset)} K={sorted(kset)} V={sorted(vset)}", flush=True)
    #
    #     for name, p in model.named_parameters():
    #         # Match layer index in parameter name (common patterns: ".layers.{i}.")
    #         hit_layer = None
    #         for i in (qset | kset | vset):
    #             if f".layers.{i}." in name:
    #                 hit_layer = i
    #                 break
    #         if hit_layer is None:
    #             continue
    #
    #         # Freeze selectively by projection
    #         if hit_layer in qset and ".q_proj." in name:
    #             p.requires_grad = False
    #         if hit_layer in kset and ".k_proj." in name:
    #             p.requires_grad = False
    #         if hit_layer in vset and ".v_proj." in name:
    #             p.requires_grad = False


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

    # ===== PROFESSOR'S VERSION (ACTIVE) =====
    # ===== PROFESSOR'S VERSION (EXTENDED) =====
    if (
        freeze_qkv_no_grad and (freeze_q_layers or freeze_k_layers or freeze_v_layers)
    ) or (
        freeze_mlp_no_grad and (freeze_mlp_layers or (freeze_mlp and (freeze_q_layers or freeze_k_layers or freeze_v_layers)))
    ):
        qset = set(freeze_q_layers or [])
        kset = set(freeze_k_layers or [])
        vset = set(freeze_v_layers or [])
        # Use freeze_mlp_layers if provided, otherwise use the intersection of Q/K/V if freeze_mlp is True
        if freeze_mlp_layers:
            mlpset = set(freeze_mlp_layers)
        elif freeze_mlp and (freeze_q_layers or freeze_k_layers or freeze_v_layers):
            # Use the union of Q/K/V layers for MLP freezing
            mlpset = qset | kset | vset
        else:
            mlpset = set()

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

    # Debug: Print trainable parameters summary
    total_params = 0
    trainable_params = 0
    k_proj_trainable = 0
    k_proj_total = 0
    
    for name, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
        
        if ".k_proj." in name:
            k_proj_total += 1
            if p.requires_grad:
                k_proj_trainable += 1
    
    print(f"\n📊 Parameter Summary:", flush=True)
    print(f"   Total parameters: {total_params:,}", flush=True)
    print(f"   Trainable parameters: {trainable_params:,} ({100.0*trainable_params/total_params:.2f}%)", flush=True)
    print(f"   K projections: {k_proj_trainable}/{k_proj_total} trainable\n", flush=True)
    
    model.gradient_checkpointing_enable()
    return model, tok


# =========================================================
# Dataset loading (subset caching preserved)
# =========================================================
def load_hotpotqa_with_optional_subset_cache(args):
    subset_dir = args.subset_save_dir
    state_path = os.path.join(subset_dir, "state.json") if subset_dir else None

    if subset_dir and os.path.exists(state_path):
        from datasets import load_from_disk
        print(f">>> Loading HotpotQA subset from {subset_dir}", flush=True)
        ds = load_from_disk(subset_dir)
        return ds["train"], ds["validation"]

    ds = load_dataset("hotpot_qa", "distractor")
    train_full, val_ds = ds["train"], ds["validation"]

    subset_size = min(args.subset_train_size, len(train_full))
    print(f">>> Sampling subset of {subset_size} from {len(train_full)} with seed {args.subset_seed}", flush=True)
    train_full = train_full.shuffle(seed=args.subset_seed).select(range(subset_size))

    if subset_dir:
        from datasets import DatasetDict
        os.makedirs(subset_dir, exist_ok=True)
        print(f">>> Saving HotpotQA subset to {subset_dir}", flush=True)
        DatasetDict({"train": train_full, "validation": val_ds}).save_to_disk(subset_dir)

    return train_full, val_ds


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


# =========================================================
# Main
# =========================================================
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

    # Debug prints you were using
    import sys
    print("[DEBUG] sys.argv:", sys.argv, flush=True)
    print("[DEBUG] has debug_hotpot?", hasattr(args, "debug_hotpot"), flush=True)
    print("[DEBUG] args.debug_hotpot =", getattr(args, "debug_hotpot", "MISSING"), flush=True)
    print("[DEBUG] args keys =", sorted(vars(args).keys()), flush=True)

    # W&B (kept as-is)
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"hotpotqa-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )

    # Dataset
    train_full, val_ds = load_hotpotqa_with_optional_subset_cache(args)

    split = train_full.train_test_split(test_size=0.05, seed=args.subset_seed)
    train_ds, dev_ds = split["train"], split["test"]
    print(f"Train {len(train_ds)} | Dev {len(dev_ds)} | Val {len(val_ds)}", flush=True)

    # Model + tokenizer
    # ===== PROFESSOR'S VERSION (ACTIVE) =====
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
    # ===== KADIR'S VERSION (COMMENTED OUT - PRESERVED FOR REFERENCE) =====
    # model, tok = load_model_and_tokenizer(
    #     args.model_name,
    #     args.use_lora,
    #     freeze_layers=args.freeze_layers,
    #     freeze_q_layers=getattr(args, "freeze_q_layers", []),
    #     freeze_k_layers=getattr(args, "freeze_k_layers", []),
    #     freeze_v_layers=getattr(args, "freeze_v_layers", []),
    # )

    # Prompt format and evidence mode
    pf = infer_prompt_format_from_model_id(args.model_name)
    evidence_mode = getattr(args, "hotpot_evidence", "supporting")

    with _timing_tracker.time_block("Tokenization (Train)"):
        tokenized_train = train_ds.map(
            lambda ex: preprocess_dataset(
                ex, tok,
                    # ===== PROFESSOR'S VERSION (ACTIVE) =====
                    max_len=1024,
                    # ===== KADIR'S VERSION (COMMENTED OUT - PRESERVED FOR REFERENCE) =====
                    # max_len=2048,
                prompt_format=pf,
                is_train=True,
                evidence_mode=evidence_mode,
            ),
            remove_columns=train_ds.column_names
        )

    with _timing_tracker.time_block("Tokenization (Val)"):
        tokenized_val = dev_ds.map(
            lambda ex: preprocess_dataset(
                ex, tok,
                    # ===== PROFESSOR'S VERSION (ACTIVE) =====
                    max_len=1024,
                    # ===== KADIR'S VERSION (COMMENTED OUT - PRESERVED FOR REFERENCE) =====
                    # max_len=2048,
                prompt_format=pf,
                is_train=False,
                evidence_mode=evidence_mode,
            ),
            remove_columns=dev_ds.column_names
        )

    # Debug one tokenized sample (kept)
    if getattr(args, "debug_hotpot", False):
        ex = tokenized_train[0]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        first_answer_idx = next((i for i, x in enumerate(labels) if x != -100), None)

        print("\n================ HOTPOT FINETUNE DEBUG ================")
        print(f"hotpot_evidence = {evidence_mode}")
        print(f"input_len = {len(input_ids)}")

        print("\n--- FULL DECODE (truncated) ---")
        print(tok.decode(input_ids[:800], skip_special_tokens=False))

        if first_answer_idx is not None:
            print("\n--- PROMPT TAIL (before answer) ---")
            print(tok.decode(input_ids[max(0, first_answer_idx - 300):first_answer_idx], skip_special_tokens=False))

            print("\n--- ANSWER HEAD (supervised) ---")
            print(tok.decode(input_ids[first_answer_idx:first_answer_idx + 200], skip_special_tokens=False))
        else:
            print("❌ ERROR: No supervised tokens found (labels all -100)")

        masked_ok = all(x == -100 for x in labels[:first_answer_idx]) if first_answer_idx is not None else False
        supervised_ok = all(x != -100 for x in labels[first_answer_idx:]) if first_answer_idx is not None else False
        print(f"\nMask check → masked_ok={masked_ok}, supervised_ok={supervised_ok}")
        print("=======================================================\n", flush=True)

    # Training args
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=f"./HotpotQA/logs/{timestamp}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    # Post-train samples (kept)
    if getattr(args, "debug_hotpot", False):
        debug_print_samples_after_training(
            model=trainer.model,
            tok=tok,
            raw_ds=train_ds,
            tokenized_ds=tokenized_train,
            output_dir=args.output_dir,
            n=3
        )

    # Save final then delete (kept)
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)

    cleanup_checkpoints(args)

    print("[Training] Final evaluation disabled. Will evaluate manually later.", flush=True)


if __name__ == "__main__":
    main()


# =========================================================
# REFERENCE: Original freezing logic (from codex snapshot)
# =========================================================
#
# def load_model_and_tokenizer(
#     model_id, use_lora,
#     freeze_layers=None,
#     freeze_q_layers=None,
#     freeze_k_layers=None,
#     freeze_v_layers=None,
#     freeze_qkv_no_grad=False,
# ):
#     freeze_layers = freeze_layers or []
#     freeze_q_layers = freeze_q_layers or []
#     freeze_k_layers = freeze_k_layers or []
#     freeze_v_layers = freeze_v_layers or []
#
#     # --- Helper: wrap a module's forward with torch.no_grad() ---
#     def _wrap_no_grad(module, label):
#         original_forward = module.forward
#         def forward_no_grad(*args, **kwargs):
#             with torch.no_grad():
#                 return original_forward(*args, **kwargs)
#         module.forward = forward_no_grad
#
#     def _apply_no_grad_wrappers(transformer_layers, qset, kset, vset):
#         for idx, layer in enumerate(transformer_layers):
#             if idx not in (qset | kset | vset):
#                 continue
#             attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
#             if attn is None:
#                 continue
#             if idx in qset and hasattr(attn, "q_proj"):
#                 _wrap_no_grad(attn.q_proj, f"Layer {idx} q_proj")
#             if idx in kset and hasattr(attn, "k_proj"):
#                 _wrap_no_grad(attn.k_proj, f"Layer {idx} k_proj")
#             if idx in vset and hasattr(attn, "v_proj"):
#                 _wrap_no_grad(attn.v_proj, f"Layer {idx} v_proj")
#
#     # ... (tokenizer + model loading) ...
#
#     # --- Step 1: Freeze entire transformer layers ---
#     if freeze_layers:
#         try:
#             transformer_layers = model.transformer.layers  # Qwen-like
#         except AttributeError:
#             transformer_layers = model.model.layers        # LLaMA-like
#         for idx, layer in enumerate(transformer_layers):
#             if idx in freeze_layers:
#                 for p in layer.parameters():
#                     p.requires_grad = False
#
#     # --- Step 2: Freeze individual Q/K/V projections by layer ---
#     if freeze_q_layers or freeze_k_layers or freeze_v_layers:
#         qset = set(freeze_q_layers)
#         kset = set(freeze_k_layers)
#         vset = set(freeze_v_layers)
#         for name, p in model.named_parameters():
#             hit_layer = None
#             for i in (qset | kset | vset):
#                 if f".layers.{i}." in name:
#                     hit_layer = i
#                     break
#             if hit_layer is None:
#                 continue
#             if hit_layer in qset and ".q_proj." in name:
#                 p.requires_grad = False
#             if hit_layer in kset and ".k_proj." in name:
#                 p.requires_grad = False
#             if hit_layer in vset and ".v_proj." in name:
#                 p.requires_grad = False
#
#     # --- Step 3 (LoRA): Also freeze LoRA params in frozen layers ---
#     if use_lora:
#         lcfg = LoraConfig(r=8, lora_alpha=16,
#                           target_modules=["q_proj","k_proj","v_proj"],
#                           lora_dropout=0.1, bias="none",
#                           task_type=TaskType.CAUSAL_LM)
#         model = get_peft_model(model, lcfg)
#         if freeze_layers:
#             for name, p in model.named_parameters():
#                 if "lora_" not in name:
#                     continue
#                 for layer_idx in freeze_layers:
#                     if f".layers.{layer_idx}." in name:
#                         p.requires_grad = False
#                         break
#         if freeze_q_layers or freeze_k_layers or freeze_v_layers:
#             for name, p in model.named_parameters():
#                 if "lora_" not in name:
#                     continue
#                 layer_hit = None
#                 for i in (qset | kset | vset):
#                     if f".layers.{i}." in name:
#                         layer_hit = i
#                         break
#                 if layer_hit is None:
#                     continue
#                 if layer_hit in qset and "q_proj" in name:
#                     p.requires_grad = False
#                 if layer_hit in kset and "k_proj" in name:
#                     p.requires_grad = False
#                 if layer_hit in vset and "v_proj" in name:
#                     p.requires_grad = False
#
#     # --- Step 4 (optional): no_grad wrapper for speed ---
#     if freeze_qkv_no_grad and (freeze_q_layers or freeze_k_layers or freeze_v_layers):
#         _apply_no_grad_wrappers(transformer_layers, qset, kset, vset)
#
#     model.gradient_checkpointing_enable()
#     return model, tok