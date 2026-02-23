import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
import numpy as np
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig
from typing import Dict, Union, List, Any
from functools import partial
import os
import datetime
from data_preprocessing import load_and_preprocess_csv,preprocess_function_for_finetuning
from model_loading_saving import SavePeftModelCallback,load_model
import pprint
import wandb
from args import parse_args
from metric import compute_metrics_for_causal_extraction


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main():
    args = parse_args()
    print(args)
    # model,tokenizer = load_model(device="auto",use_lora=args.use_lora)
    model,tokenizer = load_model(model_id=MODEL_NAME,device=args.device)
    
    #data
    data_path = "/home/yash/Finetunning/FinCausal/data/train.csv"
    dataset = load_and_preprocess_csv(data_path)
    split_dataset = dataset.train_test_split(test_size=0.1,seed=42)
    train_val_split = split_dataset["train"].train_test_split(test_size=0.05,seed=42) 
    train_dataset = train_val_split["train"].take(100) # 1417 samples
    val_dataset = train_val_split["test"].take(2) # 74 samples
    test_dataset = split_dataset["test"].take(2) # 175 samples
    print(f"train dataset size : {train_dataset.num_rows} \nval dataset size: {val_dataset.num_rows} \ntest dataset size : {test_dataset.num_rows}")
    
    tokenized_train_dataset = train_dataset.map(
    preprocess_function_for_finetuning,             
    fn_kwargs={"tokenizer": tokenizer, "max_length":2048},
    batched=True,  # This processes examples in batches
    )
    
    tokenized_val_dataset = val_dataset.map(
    preprocess_function_for_finetuning,             
    fn_kwargs={"tokenizer": tokenizer, "max_length": 2048},
    batched=True,  # This processes examples in batches
    )
    
    tokenized_test_dataset = test_dataset.map(
    preprocess_function_for_finetuning,             
    fn_kwargs={"tokenizer": tokenizer, "max_length": 2048},
    batched=True,  # This processes examples in batches
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = f"use_lora_{timestamp}" if args.use_lora else timestamp
    logging_dir = f"./logs/{run_tag}"
    os.makedirs(logging_dir, exist_ok=True)

    # W&B settings
    os.environ["WANDB_PROJECT"] = "FinCausal-finetuning"
    os.environ["WANDB_RUN_NAME"] = f"run-{run_tag}"
    os.environ["WANDB_DIR"] = os.path.abspath("./wandb_logs")
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_WATCH"] = "false"
    
    training_args = SFTConfig(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        gradient_accumulation_steps=1,  
        per_device_train_batch_size=args.batch_size,
        auto_find_batch_size=False,
        per_device_eval_batch_size=args.batch_size,
        dataloader_pin_memory=True,
        fp16=False,
        bf16=False,  
        # max_seq_length=8192, # select this instead of 1024
        max_seq_length=1024,
        
        
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
        report_to="wandb",
        logging_dir=logging_dir,
        # run_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.output_dir.split('/')[-1]}",
        output_dir=args.output_dir,
        run_name=f"run-{timestamp}",
    )
    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    if args.save_every_epoch:
        callbacks.append(SavePeftModelCallback(args=args))
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,  
        padding="longest",
        label_pad_token_id=-100,
    )
    compute_metrics = partial(compute_metrics_for_causal_extraction, tokenizer_object=tokenizer)
    

    # Initialize Trainer with early stopping
    trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics_for_causal_extraction(p, tokenizer),
            callbacks=callbacks
        )
   
    trainer.train()
    
    
    # Save the model
    save_dir = os.path.join(args.output_dir, "final_model")
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    
    wandb.finish()
    
if __name__ == "__main__":
    main()
    