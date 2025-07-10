import os
import re
import torch
import numpy as np
import datetime
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from code.ifeval.data_preprocessing_ifeval import (
    preprocess_ifeval,
    custom_data_collator,
    print_existing_special_tokens
)
from code.utils.model_saving import SavePeftModelCallback
from code.utils.args import parse_args
import wandb

print(torch.version.cuda)

def save_weight_matrix(param, path):
    if hasattr(param, "detach"):
        param = param.detach().cpu().numpy()
    np.save(path, param)

def concise_lora_filename(param_name: str) -> str:
    match = re.search(r"layers\\.(\\d+)\\.self_attn\\.(q|k|v)_proj\\.lora_(A|B)\\.default\\.weight", param_name)
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

def main():
    args = parse_args()
    wandb.init(project="IFEval", name=f"{args.model_name}_{args.dataset_name}_finetune")

    dataset = load_dataset("google/IFEval")
    split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=43)
    train_split = split_dataset["train"].train_test_split(test_size=0.05, seed=43)
    train_dataset = train_split["train"].take(3)
    val_dataset = train_split["test"].take(3)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added a new [PAD] token to the tokenizer.")

    print_existing_special_tokens(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = lambda features: custom_data_collator(features, tokenizer)

    tokenized_train = train_dataset.map(
        lambda x: preprocess_ifeval(x, tokenizer, max_len=1024, prompt_format="qwen", use_instruction=True),
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_ifeval(x, tokenizer, max_len=1024, prompt_format="qwen", use_instruction=True),
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = f"./IFEval/logs/{timestamp}"

    training_args = TrainingArguments(
        output_dir=f"outputs/{args.model_name}_{args.dataset_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim="paged_adamw_32bit",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=5,
        save_total_limit=1,
        bf16=True,
        report_to="wandb",
        logging_dir=logging_dir,
        run_name=f"{args.model_name}_{args.dataset_name}_finetune"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SavePeftModelCallback(args)] if args.use_lora else []
    )

    trainer.train()

    if args.use_lora:
        save_lora_weights(model, args.dataset_name, args.model_name, f"epoch_{int(args.epochs)}")

    if args.save_baseline:
        save_baseline_weights(model, args.dataset_name, args.model_name)
        tokenizer.save_pretrained(f"baseline_weights/{args.model_name}_{args.dataset_name}")

if __name__ == "__main__":
    main()