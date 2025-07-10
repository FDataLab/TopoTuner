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

from code.gsm8k.data_preprocessing_gsm8k import (
    custom_data_collator,
    preprocess_dataset,
    add_special_tokens_if_missing
)
from code.utils.model_saving import SavePeftModelCallback
from code.utils.args import parse_args
import wandb


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


def load_model(model_id, device="cuda:0", use_lora=False, special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")
    #print_existing_special_tokens(tokenizer)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    # Now, add the correct approach:
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    ).to("cuda:0") 

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # print("Added a new [PAD] token to the tokenizer.")
        
        # IMPORTANT: Resize model embeddings after adding new tokens
        model.resize_token_embeddings(len(tokenizer))
        # print(f"Resized model embeddings to {len(tokenizer)} tokens.")

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        # print(f"Set model.config.pad_token_id to {tokenizer.pad_token_id}.")

    # ✅ Add special tokens to tokenizer
    if special_tokens:
        add_special_tokens_if_missing(tokenizer)

    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

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
    resume_from_checkpoint = args.resume_from_checkpoint

    # 🟣 WandB init
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",
        name=f"gsm8k-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"
    )

    # 🟡 Load dataset
    dataset = load_dataset("openai/gsm8k", "main")
    split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_split = split_dataset["train"].train_test_split(test_size=0.05, seed=42)
    train_dataset = train_split["train"]  # For faster debugging, take a subset
    val_dataset = train_split["test"]  # For faster debugging, take a subset
    test_dataset = split_dataset["test"]  # For faster debugging, take a subset
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    special_tokens = ["<final-answer>", "</final-answer>"]
    model, tokenizer = load_model(
        model_id=model_id,
        use_lora=args.use_lora,
        special_tokens=special_tokens
    )

    # print(f"Model loaded: {model_id}")
    # print(f"Tokenizer loaded: {tokenizer.name_or_path}")
    # print(f"Special tokens: {tokenizer.special_tokens_map}")

    # 🔘 Collator
    data_collator = lambda features: custom_data_collator(features, tokenizer)

    # 🧮 Tokenize
    tokenized_train = train_dataset.map(
        lambda x: preprocess_dataset(x, tokenizer, max_len=1024, prompt_format="qwen", is_train=True),
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_dataset(x, tokenizer, max_len=1024, prompt_format="qwen", is_train=False),
        remove_columns=val_dataset.column_names
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = f"./GSM8K/logs/{timestamp}"

    # 🧪 Training args
    training_args = TrainingArguments(
        resume_from_checkpoint=resume_from_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        learning_rate=args.learning_rate,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        logging_steps=5,
        report_to="wandb",
        label_names=["labels"],
        logging_dir=logging_dir,
    )

    callbacks = []
    if args.save_every_epoch:
        callbacks.append(SavePeftModelCallback(args=args))

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=callbacks
    )

    trainer.model = model.to("cuda:0")

    # Debug one batch
    # from torch.utils.data import DataLoader
    # dl = DataLoader(tokenized_train, batch_size=1, collate_fn=lambda x: custom_data_collator(x, tokenizer))
    # batch = next(iter(dl))
    # print("input_ids shape:", batch["input_ids"].shape)
    # print("labels shape:", batch["labels"].shape)
    # print("input_ids:", batch["input_ids"][0])
    # print("labels:", batch["labels"][0])

    from torch.utils.data import DataLoader

    # Create a small batch to test forward pass
    dl = DataLoader(tokenized_train, batch_size=1, collate_fn=data_collator)
    batch = next(iter(dl))

    # Move batch to CUDA
    batch = {k: v.to("cuda:0") for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
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
nohup python -m code.gsm8k.finetuning_gsm8k \
  --use-lora \
  --epochs 6 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --dataset-name GSM8K \
  --model-name DeepSeek-Qwen-7B \
  --save-every-epoch \
  --save-npy \
  > logs/finetune_GSM8K_DeepSeek-Qwen-7B_lora.log 2>&1 &

  # full finetuning
nohup python -m code.gsm8k.finetuning_gsm8k \
  --epochs 6 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --save-every-epoch \
  --save-npy \
  --dataset-name GSM8K \
  --model-name DeepSeek-Qwen-7B \
  > logs/finetune_GSM8K_DeepSeek-Qwen-7B_full.log 2>&1 &
"""