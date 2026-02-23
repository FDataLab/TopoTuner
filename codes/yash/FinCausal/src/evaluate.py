import os
import torch
import pandas as pd
import numpy as np
import re # For parsing tags in compute_metrics
from datasets import Dataset # For Dataset.from_list if needed by load_and_preprocess_csv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments, # Trainer still needs some TrainingArguments
    Trainer,
    DataCollatorForLanguageModeling, # Or your chosen collator
)
from model_loading_saving import load_model
from data_preprocessing import load_and_preprocess_csv,preprocess_function_for_finetuning
from metric import compute_metrics_for_causal_extraction


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SPECIAL_TOKENS= ["<cause>", "</cause>", "<effect>", "</effect>"]
def evaluate_model_on_test_set(
    raw_test_dataset_object: Dataset,
    model_path, # Path to the directory containing the fine-tuned model and tokenizer
    tokenizer_path, # Usually same as model_path
    max_test_samples=None,
    special_tokens_list=SPECIAL_TOKENS, # Pass SPECIAL_TOKENS here
    max_seq_len=512,          # Pass MAX_SEQ_LENGTH here
    eval_batch_size=4,         # Pass PER_DEVICE_EVAL_BATCH_SIZE
    device="cuda:3"):
    # 2. Load the fine-tuned model and tokenizer
    print(f"Loading fine-tuned tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    # tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list}) 

    print(f"Loading fine-tuned model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    # model.resize_token_embeddings(len(tokenizer)) 

    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model and tokenizer loaded. Model is on device: {device}")

    # 3. Preprocess and tokenize the test dataset for the model
    # This uses the same function as for training/validation to ensure consistency
    print("Tokenizing test dataset for evaluation...")
    tokenized_test_dataset = raw_test_dataset_object.map(
        preprocess_function_for_finetuning, # Your function for tokenizing and label masking
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_seq_len},
        batched=True,
        remove_columns=raw_test_dataset_object.column_names
    )
    if not tokenized_test_dataset or len(tokenized_test_dataset) == 0:
        print("Tokenized test dataset is empty. Aborting evaluation.")
        return None
    print(f"Test dataset tokenized. Number of examples: {len(tokenized_test_dataset)}")

    # 4. Set up minimal TrainingArguments and Trainer for prediction
    # output_dir is just for potential (though unlikely for predict-only) trainer outputs, not model saving
    temp_eval_output_dir = os.path.join(os.path.dirname(model_path), "temp_test_eval_outputs")
    os.makedirs(temp_eval_output_dir, exist_ok=True)

    eval_args = TrainingArguments(
        output_dir=temp_eval_output_dir,
        per_device_eval_batch_size=eval_batch_size,
        dataloader_drop_last=False, # Ensure all test samples are evaluated
        report_to="none", # No need to report to wandb/tensorboard for this
        # fp16=torch.cuda.is_available(), # Can enable if model supports it and was trained with it
    )

    # The compute_metrics function needs the tokenizer
    metrics_fn_for_test = lambda p: compute_metrics_for_causal_extraction(p, tokenizer_object=tokenizer)

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), # Use same collator
        compute_metrics=metrics_fn_for_test
    )

    # 5. Get predictions and metrics using trainer.predict()
    print("Running predictions on the test set...")
    prediction_output = trainer.predict(test_dataset=tokenized_test_dataset)

    print("\n--- Test Set Evaluation Results ---")
    final_metrics = {}
    if prediction_output.metrics:
        for key, value in prediction_output.metrics.items():
            # The trainer prefixes eval metrics with "eval_" by default
            metric_key_for_display = key.replace("eval_", "test_")
            print(f"{metric_key_for_display}: {value:.4f}")
            final_metrics[metric_key_for_display] = value
    else:
        print("Metrics were not automatically computed by trainer.predict().")
        print("This might happen if compute_metrics was not set correctly or an error occurred.")
        # You could manually call your metric function here if needed:
        # manual_metrics = compute_metrics_for_causal_extraction(
        #    (prediction_output.predictions, prediction_output.label_ids),
        #    tokenizer_object=tokenizer
        # )
        # print("Manually computed metrics:", manual_metrics)


    # Optional: Save raw predictions (decoded text) if you want to inspect them
    # This requires decoding the raw logits/token IDs from prediction_output.predictions
    # predicted_token_ids = np.argmax(prediction_output.predictions, axis=-1)
    # decoded_predictions_text = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    # with open(os.path.join(temp_eval_output_dir, "test_set_predictions.txt"), "w", encoding="utf-8") as f:
    #     for line in decoded_predictions_text:
    #         f.write(line + "\n")
    # print(f"Raw decoded predictions saved to {os.path.join(temp_eval_output_dir, 'test_set_predictions.txt')}")

    print("--- Evaluation on Test Set Complete ---")
    return final_metrics


def main():
    
    FINETUNED_MODEL_DIR = "./finetuned_model/"
    model,tokenizer = load_model(model_id=FINETUNED_MODEL_DIR,device="cuda")
    
    data_path = "/home/yash/Finetunning/FinCausal/data/train.csv"
    dataset = load_and_preprocess_csv(data_path)
    split_dataset = dataset["train"].train_test_split(test_size=0.1,seed=42)
    test_dataset = dataset["test"].take(2)
    
    test_metrics = evaluate_model_on_test_set(
                raw_test_dataset_object=test_dataset, # Pass the Dataset object
                model_path=FINETUNED_MODEL_DIR,
                tokenizer_path=FINETUNED_MODEL_DIR,
                special_tokens_list=SPECIAL_TOKENS, # Global or passed
                max_seq_len=1024,         # Global or passed
                eval_batch_size=3 # Global or passed
            )
    if test_metrics:
        print("\nFinal Test Metrics Summary (on your selected test sample):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v}")
    
    