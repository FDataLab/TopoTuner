from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

class Model:
    def __init__(self, model_id, device="cuda", merge=False, torch_dtype=torch.float16):
        """
        Load a model, intelligently handling if it's a LoRA fine-tuned model or a regular model.
        
        Args:
            model_id (str): Path to a saved LoRA adapter directory OR a Hugging Face Hub model ID/path to a regular model.
            device (str): The device to load the model onto (e.g., "cuda:0", "cpu").
            merge (bool): If model_id is a LoRA adapter, whether to merge the adapters into the base model's weights immediately.
            torch_dtype (torch.dtype): The torch.dtype to use for model loading (e.g., torch.float16, torch.bfloat16).
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype

        is_lora_adapter = False
        adapter_config_path = os.path.join(model_id, "adapter_config.json")
        if os.path.isdir(model_id) and os.path.exists(adapter_config_path):
            try:
                # Try to load as a PEFT adapter config; if it fails, it's not a valid adapter path
                peft_config = LoraConfig.from_pretrained(model_id)
                is_lora_adapter = True
            except Exception:
                # If it's a directory but doesn't have a valid peft config, treat as regular model dir
                print(f"Warning: Directory '{model_id}' exists but does not contain a valid PEFT adapter config. Attempting to load as a regular model.")

        if is_lora_adapter:
            print(f"Detected LoRA adapter at: {model_id}. Loading base model and attaching adapter.")
            base_model_name = peft_config.base_model_name_or_path 
            if not base_model_name:
                raise ValueError(
                    f"`base_model_name_or_path` not found in adapter_config.json at {model_id}. "
                    "Ensure the adapter config specifies the original base model name."
                )

            # Load the original base model
            print(f"Loading original base model: {base_model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.torch_dtype, # Use the specified dtype
                trust_remote_code=True,
                # device_map="auto" # Can use "auto" for large models, then .to(device) for final placement
            )

            # Attach the LoRA adapter weights
            self.model = PeftModel.from_pretrained(self.model, model_id)
            print("LoRA adapter successfully attached.")

            if merge:
                print("Merging LoRA adapter into the base model's weights...")
                self.model = self.model.merge_and_unload()
                print("LoRA adapter merged.")
            else:
                print("LoRA adapter kept separate (not merged).")

            # Tokenizer for LoRA models should be from the base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, # Use base model name for tokenizer
                trust_remote_code=True,
                padding_side="right"
            )

        else: # Load as a regular model
            print(f"Loading regular model from: {model_id}.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype, # Use the specified dtype
                trust_remote_code=True,
                device_map=self.device # Can use "auto" for large models
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                padding_side="right"
            )

        # Common tokenizer setup regardless of model type
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer's pad_token set to eos_token: {self.tokenizer.pad_token}")
        self.tokenizer.add_eos_token = True
        self.tokenizer.add_bos_token = True
        self.tokenizer.padding_side = "right"

        # Move the model to the specified device
        if self.device and torch.cuda.is_available():
            self.model.to(self.device)
            print(f"Model moved to {self.device}")
        
        self.model.eval()