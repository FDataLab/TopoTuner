import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig, pipeline
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import DataCollatorForLanguageModeling,DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from trl import SFTTrainer
from typing import Dict, Union, List, Any
from data_preprocessing import DEFAULT_SYSTEM_PROMPT1,DEFAULT_SYSTEM_PROMPT2,DEFAULT_SYSTEM_PROMPT3, preprocess_dataset, create_prompt_deepseek_qwen
import datetime
from model_saving import SavePeftModelCallback
from evaluate_scirpt import evaluate_model,evaluate_during_training
import os
from args import parse_args


# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="FinEntity-finetuning"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

def load_model(model_id="", device= None, use_lora:bool=False):
    
    device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            trust_remote_code=True,
                                            padding_side = "right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.config.pad_token_id = tokenizer.config.eos_token_id
    
    model_load_kwargs = {
    "trust_remote_code": True,
    "device_map": device,
    "torch_dtype":torch.bfloat16,
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
    data = load_dataset("yixuantt/FinEntity")
    split_dataset = data["train"].train_test_split(test_size=0.05, seed=43)
    train_split = split_dataset["train"].train_test_split(test_size=0.05, seed=43)
    train_dataset = train_split["train"].take(5)
    val_dataset = train_split["test"].take(2)
    test_dataset = split_dataset["test"].take(10)
    
    print(f"train dataset size : {train_dataset.num_rows} \nval dataset size: {val_dataset.num_rows} \ntest dataset size : {test_dataset.num_rows}")
    
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model,tokenizer = load_model(model_name,device="cuda",use_lora=args.use_lora)
    
    data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False
    )
    sample_batch = [preprocess_dataset(train_dataset[i], tokenizer, is_train=True, max_len=1024) for i in range(min(4, len(train_dataset)))]
    collated_batch = data_collator(sample_batch)
    print("Collated Input IDs Shape:", collated_batch["input_ids"].shape)
    print("Collated Labels Shape:", collated_batch["labels"].shape)


    #train set
    tokenized_train_dataset = train_dataset.map(lambda sample: preprocess_dataset(sample,tokenizer,max_len=2048))
    
    tokenized_val_dataset = val_dataset.map(lambda sample: preprocess_dataset(sample,tokenizer,max_len=2048,is_train=False))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = f"./FinEntity/logs/{timestamp}"

    training_args = TrainingArguments(
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        gradient_accumulation_steps=1,  
        per_device_train_batch_size=args.batch_size,
        auto_find_batch_size=False,
        per_device_eval_batch_size=args.batch_size,
        dataloader_pin_memory=True,
        fp16=False,
        bf16=False,  
        
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
        output_dir=args.output_dir
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
    
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir ,"final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir,"final_model"))
    
    # Evaluate the model
    # print("\n--- Evaluating the fine-tuned model on the validation set ---")
    # evaluate_model(os.path.join(args.output_dir,"final_model"), test_dataset, tokenizer, max_len=1024)
    

if __name__ == "__main__":

    main()
    
    # Note: 
    """
    # 1) The model evaluation is yet to be done. The evaluation will depend on how model generate output after fine-tuning. I have rough skeleten of the evaluation, however, we will wait for fine-tune model. 
    2) Also note that, the val evaluation during fine-tuning won't make sense because again we are not sure how model generate output yet. 
    
    """
