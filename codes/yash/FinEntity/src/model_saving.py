import numpy as np
import os
import datetime
from transformers import TrainerCallback
from accelerate import Accelerator

class SavePeftModelCallback(TrainerCallback):
    def __init__(self,args, ):
        self.trainer = None
        self.args = args
        # self.accelerator = accelerator

    def set_trainer(self, trainer):
        self.trainer = trainer
        
        
    # def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
    #     output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{state.epoch}")
    #     # self.accelerator.wait_for_everyone()
    #     # unwrapped_model = self.accelerator.unwrap_model(model)
    #     # unwrapped_model.save_pretrained(output_dir, save_function=self.accelerator.save)
    #     tokenizer = self.trainer.tokenizer if self.trainer is not None else None

    #     if self.args.use_lora:
    #         output_dir = os.path.join("./Lora_finetune_weights", "epoch_weights", f"checkpoint-epoch-{state.epoch}")
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         unwrapped_model.save_pretrained(output_dir)
    #     else:
    #         unwrapped_model.save_pretrained(output_dir)
    #         if tokenizer:
    #             tokenizer.save_pretrained(output_dir)

    #     if self.args.save_npy:
    #         npy_output_dir = os.path.join(output_dir, "numpy_weights")
    #         os.makedirs(npy_output_dir, exist_ok=True)
    #         for name, param in unwrapped_model.named_parameters():
    #             if param.requires_grad:  # Only save trainable parameters
    #                 np.save(os.path.join(npy_output_dir, f"{name}.npy"), param.detach().cpu().numpy())
    #         print(f"Saved trainable model weights in NumPy format at {npy_output_dir}")

    #     print(f"Saved model checkpoint at {output_dir}")
        

    def on_epoch_end(self, args, state, control, **kwargs):
        output_dir = os.path.join(self.args.output_dir, "epoch_weights", f"checkpoint-epoch-{state.epoch}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model = kwargs["model"]
        tokenizer = self.trainer.tokenizer if self.trainer is not None else None

        if self.args.use_lora:
            output_dir = os.path.join("./FinEntity/Lora_finetune_weights", "epoch_weights", f"checkpoint-epoch-{state.epoch}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir)
            if tokenizer:
                tokenizer.save_pretrained(output_dir)

        if self.args.save_npy:
            npy_output_dir = os.path.join(output_dir, "numpy_weights")
            os.makedirs(npy_output_dir, exist_ok=True)
            for name, param in model.named_parameters():
                if param.requires_grad:  # Only save trainable parameters
                    np.save(os.path.join(npy_output_dir, f"{name}.npy"), param.detach().cpu().numpy())
            print(f"Saved trainable model weights in NumPy format at {npy_output_dir}")

        print(f"Saved model checkpoint at {output_dir}")