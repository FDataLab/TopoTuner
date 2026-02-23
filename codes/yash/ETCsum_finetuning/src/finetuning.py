import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig, pipeline
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
import numpy as np
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig
from typing import Dict, Union, List, Any
from functools import partial
import os
import datetime
from args import parse_args
from data_preprocessing import preprocess_sample, create_prompt, DEFAULT_SYSTEM_PROMPT
from model_saving import SavePeftModelCallback
from eval_metric import compute_metrics
import pprint
import wandb


# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
"""DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-llama3
QWen-7b
Llama3 - 7B

"""

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="ETCsum-finetuning"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"



def load_model(model_id=MODEL_NAME, device= None, use_lora:bool=False):
    device = device #if device else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            trust_remote_code=True,
                                            padding_side = "right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.add_eos_token
    
    model_load_kwargs = {
    "trust_remote_code": True,
    "device_map": device,
    "torch_dtype":torch.float16 if torch.cuda.is_available() else None, # you can change to torch.bfloat16. My GPU does not supports it. 
}
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
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

    
def main():
    args = parse_args()
    print(args)
    # model,tokenizer = load_model(device="auto",use_lora=args.use_lora)
    model,tokenizer = load_model(device="cuda")
    dataset = load_dataset("mrSoul7766/ECTSum")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,  
        padding="longest",
        label_pad_token_id=-100,
    )

    split_dataset = dataset["train"].train_test_split(test_size=0.1,seed=42)
    train_dataset = split_dataset["train"].take(100)
    val_dataset = split_dataset["test"].take(2)
    test_dataset = dataset["test"].take(2)
    print(f"train dataset size : {train_dataset.num_rows} \nval dataset size: {val_dataset.num_rows} \ntest dataset size : {test_dataset.num_rows}")
    

    # train set
    tokenized_train_dataset = train_dataset.map(lambda sample: preprocess_sample(sample, tokenizer, max_len=8192)).remove_columns(column_names=["text", "summary"])
    # print("Shape of first train example input_ids:", len(tokenized_train_dataset[0]["input_ids"]))
    # print("Shape of first train example labels:", len(tokenized_train_dataset[0]["labels"]))

    # val set
    tokenized_val_dataset = val_dataset.map(lambda sample: preprocess_sample(sample, tokenizer, max_len=8192, is_train=False))
    # print("Shape of first val example input_ids:", len(tokenized_val_dataset[0]["input_ids"]))
    # print("Shape of first val example labels:", len(tokenized_val_dataset[0]["labels"]))
    
    # test set
    tokenized_test_dataset = test_dataset.map(lambda sample: preprocess_sample(sample, tokenizer, max_len=8192, is_train=False))


    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = f"./logs/{timestamp}"
    
    # Define training arguments
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
        # logging_dir="./logs",
        # report_to="tensorboard",
        report_to="wandb",
        logging_dir=logging_dir,
        # run_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.output_dir.split('/')[-1]}",
        output_dir=args.output_dir
    )
    
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
    if args.save_every_epoch:
        callbacks.append(SavePeftModelCallback(args=args))
        

    # Initialize Trainer with early stopping
    trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            data_collator=data_collator,
            compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
            callbacks=callbacks
        )
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
   
    trainer.train()
    
    
    # Save the model
    trainer.save_model(args.output_dir ,"final_model")
    tokenizer.save_pretrained(args.output_dir,"final_model")
    # Save the tokenizer 
    
    # Test best model on Test dataset
    print("\n--- Evaluating on Test Dataset ---")
    test_results = trainer.predict(test_dataset=tokenized_test_dataset)
    
    print("\nTest Evaluation Metrics:")
    pprint.pprint(test_results.metrics)
    
    wandb.finish()
    
if __name__ == "__main__":
    main()
    # to enable wandb, you need to login in to wandb and add your api key.
    # customize the prompt based on models. I have done for two qwen and llama (preprocessing step)
    # select 7b or greater model, if possible
    # set device_map to "auto", in case any sync issue, run on single device
    # make sure sample size is correct line 89-90
    # set max_length = 8192 line 123
    # check lr for lora or full finetunning. 