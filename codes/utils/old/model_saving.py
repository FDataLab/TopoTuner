import os
import re
import numpy as np
from transformers import TrainerCallback

class SavePeftModelCallback(TrainerCallback):
    def __init__(self, args):
        self.trainer = None
        self.args = args

    def set_trainer(self, trainer):
        self.trainer = trainer

    def concise_lora_filename(self, param_name: str) -> str:
        match = re.search(r"layers\.(\d+)\.self_attn\.(q|k|v)_proj\.lora_(A|B)\.default\.weight", param_name)
        if match:
            layer, proj, ab = match.groups()
            return f"layer{layer}_{proj}_{ab}.npy"
        return None

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        tokenizer = self.trainer.tokenizer if self.trainer is not None else None
        epoch_number = int(state.epoch)
        dataset = self.args.dataset_name
        model_name = self.args.model_name

        if self.args.use_lora:
            save_dir = os.path.join("numpy_weights", dataset, model_name, "lora", f"epoch_{epoch_number}")
            os.makedirs(save_dir, exist_ok=True)
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    shortname = self.concise_lora_filename(name)
                    if shortname:
                        np.save(os.path.join(save_dir, shortname), param.detach().cpu().numpy())
            print(f"✅ Saved LoRA A/B weights to: {save_dir}")

        else:
            # Save full model
            save_dir = os.path.join("numpy_weights", dataset, model_name, "full", f"epoch_{epoch_number}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            if tokenizer:
                tokenizer.save_pretrained(save_dir)

            if self.args.save_npy:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        np.save(os.path.join(save_dir, f"{name}.npy"), param.detach().cpu().numpy())
                print(f"✅ Saved full model weights in NumPy format to: {save_dir}")

        print(f"📦 Finished saving checkpoint for epoch {epoch_number}")


# import os
# import re
# import numpy as np
# from transformers import TrainerCallback

# class SavePeftModelCallback(TrainerCallback):
#     def __init__(self, args):
#         self.trainer = None
#         self.args = args

#     def set_trainer(self, trainer):
#         self.trainer = trainer

#     def concise_lora_filename(self, param_name: str) -> str:
#         match = re.search(r"layers\.(\d+)\.self_attn\.(q|k|v)_proj\.lora_(A|B)\.default\.weight", param_name)
#         if match:
#             layer, proj, ab = match.groups()
#             return f"layer{layer}_{proj}_{ab}.npy"
#         return None

#     def on_epoch_end(self, args, state, control, **kwargs):
#         model = kwargs["model"]
#         tokenizer = self.trainer.tokenizer if self.trainer is not None else None
#         epoch_number = int(state.epoch)
#         dataset = self.args.dataset_name
#         model_name = self.args.model_name

#         if self.args.use_lora:
#             # Save LoRA A/B weights
#             save_dir = os.path.join(
#                 "numpy_weights",
#                 dataset,
#                 model_name,
#                 "lora",
#                 f"epoch_{epoch_number}"
#             )
#             os.makedirs(save_dir, exist_ok=True)

#             for name, param in model.named_parameters():
#                 if "lora_A" in name or "lora_B" in name:
#                     shortname = self.concise_lora_filename(name)
#                     if shortname:
#                         save_path = os.path.join(save_dir, shortname)
#                         np.save(save_path, param.detach().cpu().numpy())
#             print(f"✅ Saved LoRA A/B weights to: {save_dir}")

#         else:
#             # Full finetune checkpoint
#             output_dir = os.path.join(
#                 "numpy_weights",
#                 self.args.dataset_name,
#                 self.args.model_name,
#                 "full",
#                 f"epoch_{epoch_number}"
#             )
#             os.makedirs(output_dir, exist_ok=True)
#             model.save_pretrained(output_dir)
#             if tokenizer:
#                 tokenizer.save_pretrained(output_dir)

#             if self.args.save_npy:
#                 npy_output_dir = os.path.join(
#                     "Topo-Tuner",
#                     "numpy_weights",
#                     dataset,
#                     model_name,
#                     "full",
#                     f"epoch_{epoch_number}"
#                 )
#                 os.makedirs(npy_output_dir, exist_ok=True)
#                 for name, param in model.named_parameters():
#                     if param.requires_grad:
#                         np.save(os.path.join(npy_output_dir, f"{name}.npy"), param.detach().cpu().numpy())
#                 print(f"✅ Saved full model weights in NumPy format to: {npy_output_dir}")

#         print(f"📦 Finished saving checkpoint for epoch {epoch_number}")