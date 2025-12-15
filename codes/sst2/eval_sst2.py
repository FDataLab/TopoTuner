import json
import os
from typing import Dict, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm
import re
from .data_preprocessing_sst2 import build_prompt, infer_prompt_format_from_model_id

def extract_sentiment_from_response(response: str) -> str:
    """Mirror IMDB's robust extraction logic."""
    response = response.lower().strip()

        # Quick first-token heuristic (handles cases like "positive." or "negative\n")
    head = response[:20].strip().lower().strip(" .,!?:;\n\t\r")
    if head.startswith("positive") or head.startswith("pos"):
        return "positive"
    if head.startswith("negative") or head.startswith("neg"):
        return "negative"

    # Look for explicit sentiment indicators anywhere
    if "positive" in response:
        return "positive"
    if "negative" in response:
        return "negative"

    # Look for <final-answer> tags
    final_answer_match = re.search(r'<final-answer>(.*?)</final-answer>', response, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).strip().lower()
        if "positive" in answer:
            return "positive"
        if "negative" in answer:
            return "negative"

    # Look for "Answer:" pattern
    answer_match = re.search(r'answer:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip().lower()
        if "positive" in answer:
            return "positive"
        if "negative" in answer:
            return "negative"

    return "unknown"

def evaluate_sst2(
    model,
    tokenizer,
    split: str = "validation",
    limit: Optional[int] = None,
    max_new_tokens: int = 6,
    batch_size: int = 8,
    progress_bar: bool = True,
    debug_print: bool = False,
    save_jsonl: Optional[str] = None,
    save_tsv: Optional[str] = None,
    run_name: str = "",
    phase: str = "",
    epoch: int = 0,
    step: int = 0,
    output_dir: str = "",
) -> Dict[str, float]:

    """
    Evaluate model on SST2 sentiment analysis.
    Returns:
     Dict with 'accuracy', 'positive_acc', 'negative_acc', 'n' keys
    """

    model.eval()

    # Load SST2 dataset
    ds_dict = load_dataset("stanfordnlp/sst2")
    dataset = ds_dict[split]

    # Deterministic stratified sub-sampling (mirror IMDB style)
    if limit and limit > 0:
        from datasets import concatenate_datasets
        # Split by label on the split dataset
        ds_pos = dataset.filter(lambda ex: ex["label"] == 1)
        ds_neg = dataset.filter(lambda ex: ex["label"] == 0)
        # Deterministic shuffle
        ds_pos = ds_pos.shuffle(seed=42)
        ds_neg = ds_neg.shuffle(seed=42)
        # Target counts (roughly 50/50)
        k_pos = min(limit // 2, len(ds_pos))
        k_neg = min(limit - k_pos, len(ds_neg))
        # If one class is scarce, backfill from the other
        if k_pos + k_neg < limit:
            remaining = limit - (k_pos + k_neg)
            # Prefer backfilling from the larger pool
            extra_pos = min(remaining, len(ds_pos) - k_pos)
            k_pos += extra_pos
            remaining -= extra_pos
            if remaining > 0:
                extra_neg = min(remaining, len(ds_neg) - k_neg)
                k_neg += extra_neg
        # Select and combine, then shuffle to mix
        parts = []
        if k_pos > 0:
            parts.append(ds_pos.select(range(k_pos)))
        if k_neg > 0:
            parts.append(ds_neg.select(range(k_neg)))
        if parts:
            dataset = concatenate_datasets(parts).shuffle(seed=42)

    correct = 0
    positive_correct = 0
    negative_correct = 0
    positive_total = 0
    negative_total = 0
    total = len(dataset)

    results = []

    # Get prompt format (handle non-dict config objects like LlamaConfig)
    cfg = getattr(model, 'config', None)
    model_id = getattr(cfg, '_name_or_path', '') if cfg is not None else ''
    prompt_format = infer_prompt_format_from_model_id(model_id)

    iterator = tqdm(dataset, desc="Evaluating SST2") if progress_bar else dataset
    
    for i, example in enumerate(iterator):
        text = example["sentence"]
        true_label = example["label"]
        true_sentiment = "positive" if true_label == 1 else "negative"
        
        # Build prompt
        prompt = build_prompt(tokenizer, text, prompt_format=prompt_format, use_instruction=True)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(prompt):].strip()
        
        # Extract predicted sentiment
        predicted_sentiment = extract_sentiment_from_response(response)
        
        # Check if correct
        is_correct = predicted_sentiment == true_sentiment
        if is_correct:
            correct += 1
        
        # Track positive/negative accuracy
        if true_sentiment == "positive":
            positive_total += 1
            if is_correct:
                positive_correct += 1
        else:  # negative
            negative_total += 1
            if is_correct:
                negative_correct += 1

        # Store result
        result = {
            "index": i,
            "text": text[:100] + "..." if len(text) > 100 else text,
            "true_sentiment": true_sentiment,
            "predicted_sentiment": predicted_sentiment,
            "correct": is_correct,
            "response": response[:200] + "..." if len(response) > 200 else response,
        }
        results.append(result)

        if debug_print and i < 5:
            print(f"Example {i}:")
            print(f"  Text: {text[:100]}...")
            print(f"  True: {true_sentiment}")
            print(f"  Predicted: {predicted_sentiment}")
            print(f"  Correct: {is_correct}")
            print(f"  Response: {response[:100]}...")
            print()

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    positive_acc = positive_correct / positive_total if positive_total > 0 else 0.0
    negative_acc = negative_correct / negative_total if negative_total > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "positive_acc": positive_acc,
        "negative_acc": negative_acc,
        "n": total,
    }
    
    # Save results if requested
    if save_jsonl:
        os.makedirs(os.path.dirname(save_jsonl), exist_ok=True)
        with open(save_jsonl, "a") as f:
            record = {
                "epoch": epoch,
                "step": step,
                "phase": phase,
                "run_name": run_name,
                "dataset": "SST2",
                "accuracy": accuracy,
                "positive_acc": positive_acc,
                "negative_acc": negative_acc,
                "n": total,
            }
            f.write(json.dumps(record) + "\n")
    if save_tsv:
        import csv
        os.makedirs(os.path.dirname(save_tsv), exist_ok=True)
        new_file = not os.path.exists(save_tsv)
        with open(save_tsv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "step", "phase", "run_name", "dataset", "accuracy", "positive_acc", "negative_acc", "n"], delimiter="\t")
            if new_file:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch,
                "step": step,
                "phase": phase,
                "run_name": run_name,
                "dataset": "SST2",
                "accuracy": accuracy,
                "positive_acc": positive_acc,
                "negative_acc": negative_acc,
                "n": total,
            })
    return metrics
