import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from collections import Counter

from data_preprocessing import create_prompt_deepseek_qwen

# --- Evaluation Script ---

def preprocess_eval_dataset(data, tokenizer, max_len: 512):
    text = data["content"]
    prompt = create_prompt_deepseek_qwen(text)
    tokenized_input = tokenizer(prompt,
                                max_length=max_len,
                                truncation=True,
                                padding=False,
                                padding_side="right",
                                return_tensors='pt')
    return {
        "input_ids": tokenized_input["input_ids"].squeeze(0),
        "attention_mask": tokenized_input["attention_mask"].squeeze(0),
        "ground_truth": [{'entity': a["value"], 'label': a["label"]} for a in data["annotations"]]
    }

def extract_entities_from_prediction(text):
    entities = []
    if text.lower() != "none":
        for item in text.split(','):
            item = item.strip()
            if " (" in item and ")" == item[-1]:
                try:
                    entity, label_with_paren = item.split(" (")
                    label = label_with_paren[:-1].strip()
                    entities.append({"entity": entity.strip(), "label": label})
                except ValueError:
                    pass  # Handle cases where the format might be slightly off
    return entities

def calculate_metrics(predicted, ground_truth):
    tp = 0
    fp = 0
    fn = 0

    predicted_set = set([(p['entity'].lower(), p['label'].lower()) for p in predicted])
    ground_truth_set = set([(gt['entity'].lower(), gt['label'].lower()) for gt in ground_truth])

    tp = len(predicted_set.intersection(ground_truth_set))
    fp = len(predicted_set - ground_truth_set)
    fn = len(ground_truth_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_model(model_path, dataset, tokenizer, max_len,temperature, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.add_eos_token
        tokenizer.config.pad_token_id = tokenizer.config.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    if "lora" in model_path.lower():
        model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    predictions = []
    ground_truths_list = []

    processed_dataset = dataset.map(
        preprocess_eval_dataset,
        fn_kwargs={"tokenizer": tokenizer, "max_len": max_len},
        remove_columns=dataset.column_names,
    )

    for data in tqdm(processed_dataset):
        input_ids = torch.tensor(data["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(data["attention_mask"]).unsqueeze(0).to(model.device)
        ground_truth = data["ground_truth"]
        ground_truths_list.append(ground_truth)

        with torch.inference_mode():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512,temperature=temperature,do_sample=True)
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<｜Assistant｜>" in predicted_text:
            predicted_text = predicted_text.split("<｜Assistant｜>")[1]
        
        predictions.append(predicted_text)

    predicted_entities = [extract_entities_from_prediction(pred) for pred in predictions]

    all_metrics = []
    for pred_ents, gt_ents in zip(predicted_entities, ground_truths_list):
        metrics = calculate_metrics(pred_ents, gt_ents)
        all_metrics.append(metrics)

    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics) if all_metrics else 0

    print(f"Evaluation Results:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
    
def evaluate_during_training(model, dataset, tokenizer, max_len, device="cuda"):
    model.eval()
    predictions = []
    ground_truths_list = []

    processed_dataset = dataset.map(
        preprocess_eval_dataset,
        fn_kwargs={"tokenizer": tokenizer, "max_len": max_len},
        remove_columns=dataset.column_names,
    )

    for data in tqdm(processed_dataset, desc="Evaluating"):
        input_ids = torch.tensor(data["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(data["attention_mask"]).unsqueeze(0).to(model.device)
        ground_truth = data["ground_truth"]
        ground_truths_list.append(ground_truth)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512)
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(predicted_text)

    predicted_entities = [extract_entities_from_prediction(pred) for pred in predictions]

    all_metrics = []
    for pred_ents, gt_ents in zip(predicted_entities, ground_truths_list):
        metrics = calculate_metrics(pred_ents, gt_ents)
        all_metrics.append(metrics)

    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics) if all_metrics else 0

    metrics = {
        "eval_precision": avg_precision,
        "eval_recall": avg_recall,
        "eval_f1": avg_f1,
        "eval_loss": 1.0 # Placeholder - you might need to calculate the actual loss here
    }

    print(f"\nEvaluation Results (Epoch End):")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-Score: {avg_f1:.4f}")
    print(f"Eval Loss (Placeholder): {metrics['eval_loss']:.4f}") # Print the placeholder loss

    model.train() # Set back to train mode
    return metrics