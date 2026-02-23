import os
import re
import torch
import datetime
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from code.finentity.data_preprocessing import preprocess_dataset
from code.utils.model_saving import SavePeftModelCallback
from code.utils.args import parse_args

import wandb

print(torch.version.cuda)

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

def load_model(model_id, device= None, use_lora:bool=False):
    device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            trust_remote_code=True,
                                            padding_side = "right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.config.pad_token_id = tokenizer.config.eos_token_id
    
    model_load_kwargs = {
        "trust_remote_code": True,
        "device_map": device,
        "torch_dtype": torch.float16,
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    # Apply LoRA if specified
    if use_lora:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj","k_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        print(f"Lora config: {lora_config}\n")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()                             
    return model, tokenizer

def main():
    args = parse_args()
    resume_from_checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint else None

    # Initialize wandb
    wandb.login(key="4559d55ae1eb6282f60a6d9a13fbf5c65e9ec215", relogin=True)
    wandb.init(
        project="topotuner",  # Must match exactly your personal project name
        name=f"finentity-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        entity="kadirerol"  # Your username
    )

    data = load_dataset("yixuantt/FinEntity")
    # data = load_dataset("openai/gsm8k", "main")
    split_dataset = data["train"].train_test_split(test_size=0.05, seed=43)
    train_split = split_dataset["train"].train_test_split(test_size=0.05, seed=43)
    
    #train_dataset = train_split["train"].select(range(len(train_split["train"] // 16))
    #val_dataset = train_split["test"].select(range(len(train_split["test"]) // 16))
    train_dataset = train_split["train"]
    val_dataset = train_split["test"]
    test_dataset = split_dataset["test"]
    print(f"train dataset size : {train_dataset.num_rows} \nval dataset size: {val_dataset.num_rows} \ntest dataset size : {test_dataset.num_rows}")

    # Change the model ID - VEPAUL
    # VEPAUL: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model,tokenizer = load_model(model_id, device="cuda:0", use_lora=args.use_lora)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    sample_batch = [preprocess_dataset(train_dataset[i], tokenizer, is_train=True, max_len=512) for i in range(min(4, len(train_dataset)))]
    collated_batch = data_collator(sample_batch)
    print("Collated Input IDs Shape:", collated_batch["input_ids"].shape)
    print("Collated Labels Shape:", collated_batch["labels"].shape)

    #train set
    tokenized_train_dataset = train_dataset.map(lambda sample: preprocess_dataset(sample,tokenizer,max_len=512))
    tokenized_val_dataset = val_dataset.map(lambda sample: preprocess_dataset(sample,tokenizer,max_len=512,is_train=False))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = f"./FinEntity/logs/{timestamp}"

    #if args.save_baseline:
    #    save_baseline_weights(model, args.dataset_name, args.model_name)

    training_args = TrainingArguments(
        resume_from_checkpoint = args.resume_from_checkpoint,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_checkpointing=True, 
        gradient_accumulation_steps=2,  
        per_device_train_batch_size=args.batch_size,
        auto_find_batch_size=False,
        per_device_eval_batch_size=args.batch_size,
        dataloader_pin_memory=True,
        fp16=False,
        bf16=True,   
        
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        optim="paged_adamw_32bit",
        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        
        logging_steps=5,
        # report_to="tensorboard",
        report_to="wandb",
        logging_dir=logging_dir,
        output_dir = args.output_dir or os.path.join("logs", "finetuned_model")
    )
    
    callbacks = []
    if args.save_every_epoch:
        callbacks.append(SavePeftModelCallback(args=args))
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
        # compute_metrics=lambda p: evaluate_during_training(trainer.model, tokenized_val_dataset.select(range(min(len(tokenized_val_dataset), 100))), tokenizer, max_len=1024) # Evaluate using our function
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if args.use_lora:
        save_lora_weights(
            model,
            args.dataset_name,
            args.model_name,
            f"epoch_{int(args.epochs)}"
        )

if __name__ == "__main__":
    main()


"""
  # lora finetuning
nohup python -m code.finentity.finetuning \
  --use-lora \
  --epochs 100 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --save-baseline \
  --dataset-name FinEntity \
  --model-name DeepSeek-Qwen-7B \
  --save-every-epoch \
  > /staging/users/aerol1/tda/Topo-Tuner/logs/finetune_FinEntity_DeepSeek-Qwen-7B_lora-100.log 2>&1 &

  # full finetuning
nohup python -m code.finentity.finetuning \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --save-baseline \
  --save-every-epoch \
  --save-npy \
  --dataset-name FinEntity \
  --model-name DeepSeek-Qwen-7B \
  > logs/finetune_FinEntity_DeepSeek-Qwen-7B_full-50.log 2>&1 &

  nohup python -m code.finentity.finetuning \
  --epochs 50 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --save-baseline \
  --dataset-name FinEntity \
  --model-name DeepSeek-Qwen-7B \
  --save-every-epoch \
  --output-dir logs/finetuned_model_FinEntity_Qwen7B_full-big \
  > logs/finetune_FinEntity_DeepSeek-Qwen-7B_full-50.log 2>&1 &
"""