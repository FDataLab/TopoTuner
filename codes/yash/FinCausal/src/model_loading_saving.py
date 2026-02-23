import numpy as np
import os
import datetime
from transformers import TrainerCallback
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

def load_model(model_id="MODEL_NAME", device= None, use_lora:bool=False):
    device = device #if device else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            trust_remote_code=True,
                                            padding_side = "right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens_list = ["<cause>", "</cause>", "<effect>", "</effect>"]

    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})
    # print(f"Added {num_added_toks} special tokens: {special_tokens_list}")
    
    model_load_kwargs = {
    "trust_remote_code": True,
    "device_map": device,
    "torch_dtype":torch.float16 if torch.cuda.is_available() else None, # you can change to torch.bfloat16. My GPU does not supports it. 
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    model.resize_token_embeddings(len(tokenizer)) # including new tokens
    
    # to check model context length
    # print(f"Model context length: {model.config.max_position_embeddings}")
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply LoRA if specified
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj","k_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
                                                
    return model, tokenizer


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
            output_dir = os.path.join("./Lora_finetune_weights", "epoch_weights", f"checkpoint-epoch-{state.epoch}")
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