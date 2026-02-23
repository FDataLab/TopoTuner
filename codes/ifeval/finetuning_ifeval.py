import os
import re
import torch
import datetime
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from code.ifeval.data_preprocessing_ifeval import (
    custom_data_collator,
    preprocess_ifeval,
)
from code.utils.model_saving import SavePeftModelCallback
from code.utils.args import parse_args
import wandb

from transformers import TrainerCallback
from functools import partial
from accelerate import init_empty_weights

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class LossDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"[Epoch {state.epoch:.2f}] Loss: {logs['loss']:.6f}")


def save_weight_matrix(param, path):
    if hasattr(param, "detach"):
        param = param.detach().cpu().numpy()
    np.save(path, param)

def concise_lora_filename(param_name: str) -> str:
    match = re.search(r"layers\.(\d+)\.self_attn\.(q|k|v)_proj\.lora_(A|B)\.default\.weight", param_name)
    if match:
        layer, proj, ab = match.groups()
        return f"layer{layer}_{proj}_{ab}.npy"
    return None

def save_lora_weights(model, dataset_name, model_name, epoch_tag):
    save_dir = os.path.join("numpy_weights", dataset_name, model_name, "lora", epoch_tag)
    os.makedirs(save_dir, exist_ok=True)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            shortname = concise_lora_filename(name)
            if shortname:
                save_path = os.path.join(save_dir, shortname)
                save_weight_matrix(param, save_path)
    print(f"✅ Saved LoRA A/B weights to: {save_dir}")

def save_baseline_weights(model, dataset_name, model_name):
    save_dir = os.path.join("numpy_weights", dataset_name, model_name, "baseline")
    os.makedirs(save_dir, exist_ok=True)

    for name, param in model.named_parameters():
        if param.requires_grad and any(proj in name for proj in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
            clean_name = name.replace(".", "_") + ".npy"
            save_path = os.path.join(save_dir, clean_name)
            np.save(save_path, param.detach().cpu().numpy())

    print(f"✅ Saved baseline weights (q/k/v) to: {save_dir}")

def load_model(model_id, device=None, use_lora=False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",               
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.resize_token_embeddings(len(tokenizer)) 

    # ✅ Sync config pad token
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Pad token ID:", tokenizer.pad_token_id)
    print("Vocab size:", tokenizer.vocab_size)
    assert tokenizer.pad_token_id < model.get_input_embeddings().num_embeddings

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        print(f"LoRA config: {lora_config}")
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()  # ✅ VERY IMPORTANT
        model.print_trainable_parameters()
        
    model.gradient_checkpointing_enable()

    return model, tokenizer


def main():
    args = parse_args()
    # 🟣 WandB init
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"ifeval-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )

    # 🟡 Load dataset
    dataset = load_dataset("google/IFEval")
    split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_split = split_dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_dataset = train_split["train"] # For faster debugging, take a subset
    val_dataset = train_split["test"] # For faster debugging, take a subset
    test_dataset = split_dataset["test"] # For faster debugging, take a subset
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    model, tokenizer = load_model(
        model_id=model_id,
        use_lora=args.use_lora,
    )

    # 🧮 Tokenize
    tokenized_train = train_dataset.map(
        lambda x: preprocess_ifeval(x, tokenizer, max_len=1024, prompt_format="qwen", is_train=True),
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_ifeval(x, tokenizer, max_len=1024, prompt_format="qwen", is_train=False),
        remove_columns=val_dataset.column_names
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = f"./IFEval/logs/{timestamp}"

    # 🧪 Training args
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        gradient_accumulation_steps=1,  
        per_device_train_batch_size=args.batch_size,
        auto_find_batch_size=False,
        per_device_eval_batch_size=args.batch_size,
        dataloader_pin_memory=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="wandb",
        logging_dir=logging_dir,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        bf16=True,
        label_names=["labels"] 
    )
    
    callbacks = [LossDebugCallback()]
    if args.save_every_epoch:
        callbacks.append(SavePeftModelCallback(args=args))
    
    data_collator = partial(custom_data_collator, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Debug one batch
    from torch.utils.data import DataLoader
    dl = DataLoader(tokenized_train, batch_size=args.batch_size, collate_fn=lambda x: custom_data_collator(x, tokenizer))
    batch = next(iter(dl))
    print("INPUT IDS:", batch["input_ids"][0])
    print("LABELS   :", batch["labels"][0])
    print("ATTN MASK:", batch["attention_mask"][0])

    # Move batch to CUDA
    from accelerate import find_executable_batch_size
    device = model.device if not hasattr(model, "module") else model.module.device
    batch = {k: v.to(device) for k, v in batch.items()}
    print("DECODED INPUT:", tokenizer.decode(batch["input_ids"][0]))

    model.eval()
    with torch.no_grad():
        decoded = tokenizer.decode(batch["input_ids"][0])
        try:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
        except RuntimeError as e:
            print("CUDA ERROR CAUGHT:", e)
            print("🚨 Possible cause: label value out of bounds or tensor shape mismatch")
            print("Labels:", batch["labels"][0])
            print("Max label:", batch["labels"].max().item(), "| Vocab size:", model.get_output_embeddings().num_embeddings)
            raise e

        print("=== Forward Pass Debug ===")
        print("Loss:", output.loss)
        print("Logits:", output.logits.shape)
        print("Grad fn:", output.loss.grad_fn if output.loss is not None else None)

    trainer.train()

    if args.use_lora:
        save_lora_weights(model, args.dataset_name, args.model_name, f"epoch_{int(args.epochs)}")


if __name__ == "__main__":
    main()

"""
  # lora finetuning
nohup python -m code.ifeval.finetuning_ifeval \
  --use-lora \
  --epochs 6 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --dataset-name IFEval \
  --model-name DeepSeek-Qwen-7B \
  --save-every-epoch \
  --save-npy \
  > logs/finetune_IFEval_DeepSeek-Qwen-7B_lora.log 2>&1 &

  # full finetuning
nohup python -m code.ifeval.finetuning_ifeval \
  --epochs 6 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --save-every-epoch \
  --save-npy \
  --dataset-name IFEval \
  --model-name DeepSeek-Qwen-7B \
  --output-dir logs/finetuned_model_IFEval_Qwen7B_full \
  > logs/finetune_IFEval_DeepSeek-Qwen-7B_full.log 2>&1 &
"""