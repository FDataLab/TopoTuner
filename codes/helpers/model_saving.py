import os
import re
import numpy as np
from transformers import TrainerCallback
import torch

def concise_lora_filename(param_name: str) -> str | None:
    """
    Match LoRA A/B on q/k/v/o projections and produce a short filename.
    Example: layers.7.self_attn.q_proj.lora_A.default.weight -> layer7_q_A.npy
    """
    m = re.search(r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.lora_(A|B)\.default\.weight", param_name)
    if m:
        layer, proj, ab = m.groups()
        return f"layer{layer}_{proj}_{ab}"
    return None

def concise_full_filename(param_name: str) -> str | None:
    """
    Match q/k/v projection weights in full finetuning and produce a short filename.
    Example:
      model.layers.7.self_attn.q_proj.weight -> layer7_q.npy
    """
    m = re.search(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", param_name)
    if m:
        layer, proj = m.groups()
        return f"layer{layer}_{proj}"
    return None


class SavePeftModelCallback(TrainerCallback):
    """
    Per-epoch saver, close to your reference:
      - LoRA: saves adapters; optionally dumps A/B matrices to numpy_weights/
      - Full model: saves model + tokenizer; optionally dumps trainable tensors to numpy_weights/
    Saves under:  <output_dir>/epoch_weights/checkpoint-epoch-{N}
    """
    def __init__(self, args, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.trainer = None

    def set_trainer(self, trainer):
        print("🔗 SavePeftModelCallback attached to Trainer")
        self.trainer = trainer
        if self.tokenizer is None:
            self.tokenizer = getattr(trainer, "tokenizer", None)

    def on_epoch_end(self, train_args, state, control, **kwargs):
        model = kwargs["model"]
        epoch_number = int(state.epoch)

        base_dir = os.path.join(self.args.output_dir, "epoch_weights")
        os.makedirs(base_dir, exist_ok=True)
        save_dir = os.path.join(base_dir, f"checkpoint-epoch-{epoch_number}")
        os.makedirs(save_dir, exist_ok=True)

        # ---- Save model ----
        model.save_pretrained(save_dir)
        print(f"💾 Model saved at: {save_dir}")

        # ---- Save tokenizer ----
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)
            print(f"💾 Tokenizer saved at: {save_dir}")

        # ---- Save training args ----
        ta_path = os.path.join(save_dir, "training_args.bin")
        torch.save(self.args, ta_path)
        print(f"💾 Training args saved at: {ta_path}")

        # ---- Save trainer state (optimizer, scheduler, rng) ----
        if self.trainer is not None:
            if self.trainer.optimizer is not None:
                opt_path = os.path.join(save_dir, "optimizer.pt")
                torch.save(self.trainer.optimizer.state_dict(), opt_path)
                print(f"💾 Optimizer saved at: {opt_path}")

            if self.trainer.lr_scheduler is not None:
                sch_path = os.path.join(save_dir, "scheduler.pt")
                torch.save(self.trainer.lr_scheduler.state_dict(), sch_path)
                print(f"💾 Scheduler saved at: {sch_path}")

            rng_path = os.path.join(save_dir, "rng_state.pth")
            torch.save(torch.get_rng_state(), rng_path)
            print(f"💾 RNG state saved at: {rng_path}")

            ts_path = os.path.join(save_dir, "trainer_state.json")
            with open(ts_path, "w") as f:
                f.write(self.trainer.state.to_json_string())
            print(f"💾 Trainer state saved at: {ts_path}")

            print(">>> DEBUG: trainer is", self.trainer)
            print(">>> DEBUG: optimizer is", self.trainer.optimizer)
            print(">>> DEBUG: scheduler is", self.trainer.lr_scheduler)
            print(">>> DEBUG: saving to", save_dir)

        # ---- Optional numpy dump ----
        if getattr(self.args, "save_npy", False):
            npy_dir = os.path.join(save_dir, "numpy_weights")
            os.makedirs(npy_dir, exist_ok=True)
            count = 0
            for name, param in model.named_parameters():
                if self.args.use_lora:
                    if "lora_A" in name or "lora_B" in name:
                        short = concise_lora_filename(name)
                        if short:
                            arr = param.detach().cpu().to(torch.float16)
                            np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                            count += 1
                else:
                    if param.requires_grad:
                        short = concise_full_filename(name)
                        if short:
                            arr = param.detach().cpu().to(torch.float16)
                            np.save(os.path.join(npy_dir, f"{short}.npy"), arr.numpy())
                            count += 1
            print(f"✅ Saved {count} tensors (float16) to: {npy_dir}")

        print(f"💾 Full checkpoint saved at: {save_dir}")
        return control