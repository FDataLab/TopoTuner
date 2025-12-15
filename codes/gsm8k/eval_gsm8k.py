import os
import csv
import json
import time
import re
from typing import Optional, Dict

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

class QStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Stop when we see "Q:" (new question)
        self.stop_tokens = [
            tokenizer.encode("Q:", add_special_tokens=False)[0],
            tokenizer.encode("\nQ:", add_special_tokens=False)[0],
        ]
    
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Check if the last token is a stop token
        if len(input_ids) > 0:
            last_token = input_ids[0, -1].item()
            if last_token in self.stop_tokens:
                return True
        return False

from .data_preprocessing_gsm8k import (
    infer_prompt_format_from_model_id,
    eos_for_model,
    create_prompt_llama3,
    create_prompt_llama2,
    create_prompt_mistral,
    create_prompt_qwen,
    create_prompt_olmo,
    DEFAULT_SYSTEM_PROMPT,
)

def _get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_mem_alloc": None, "gpu_mem_reserved": None}
    gpu_id = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_id)
    return {
        "gpu": props.name,
        "gpu_id": gpu_id,
        "gpu_mem_alloc": torch.cuda.memory_allocated(gpu_id) // 1024**2,
        "gpu_mem_reserved": torch.cuda.memory_reserved(gpu_id) // 1024**2,
    }

def _append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _append_tsv(path: str, row: dict, field_order: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=field_order, delimiter="\t")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in field_order})

def normalize_number(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None

def extract_final_answer(text):
    """Extract final answer using EleutherAI standard method."""
    # Method 1: Look for "The answer is X" pattern (EleutherAI standard) - HIGHEST PRIORITY
    # Find ALL occurrences and take the LAST one (most recent)
    matches = re.findall(r"The answer is (\-?[0-9\.\,\$]+)", text)
    if matches:
        return normalize_number(matches[-1])  # Take the last occurrence
    
    # Method 2: Look for <final-answer> ... </final-answer>
    m = re.search(r"<final-answer>\s*([0-9.,]+)\s*</final-answer>", text)
    if m:
        return normalize_number(m.group(1))
    
    # Method 3: Look for #### format (GSM8K standard) - but handle comma-separated lists
    m = re.search(r"####\s*([0-9.,]+)", text)
    if m:
        answer_text = m.group(1)
        # If it's a comma-separated list, take the last number
        if ',' in answer_text:
            numbers = [n.strip() for n in answer_text.split(',')]
            # Take the last valid number
            for num in reversed(numbers):
                if num.isdigit():
                    return normalize_number(num)
        return normalize_number(answer_text)
    
    # Method 4: Flexible extract (last number) - LOWEST PRIORITY
    nums = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if nums:
        # Take the last valid number
        for num_tuple in reversed(nums):
            for num in num_tuple:
                if num and num.strip():
                    return normalize_number(num.strip())
    
    return None

# Few-shot examples from EleutherAI lm-evaluation-harness
FEWSHOT_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29."
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."
    }
]

def build_fewshot_prompt(question, num_examples=5):
    """Build few-shot prompt using EleutherAI examples."""
    prompt = ""
    
    # Add few-shot examples
    for i in range(min(num_examples, len(FEWSHOT_EXAMPLES))):
        example = FEWSHOT_EXAMPLES[i]
        prompt += f"Q: {example['question']}\n\nA: {example['answer']}\n\n"
    
    # Add the current question
    prompt += f"Q: {question}\n\nA:"
    
    return prompt

@torch.no_grad()
def evaluate_gsm8k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    split: str = "test",
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    batch_size: int = 32,
    *,
    progress_bar: bool = True,
    save_jsonl: Optional[str] = None,
    save_tsv: Optional[str] = None,
    run_name: str = "",
    phase: str = "adhoc",
    epoch: int = -1,
    step: int = -1,
    output_dir: str = "",
    debug_print: bool = False,
    use_fewshot: bool = True,
    num_fewshot_examples: int = 5,
) -> Dict[str, float]:
    """
    Evaluate GSM8K by extracting <final-answer> tags.
    """
    start_time = time.time()
    
    # Load GSM8K dataset
    ds = load_dataset("openai/gsm8k", "main")[split]
    
    # Apply limit if specified
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    
    n_total = len(ds)
    correct = 0

    model_id = getattr(model, "name_or_path", None) or getattr(model.config, "_name_or_path", "")
    prompt_format = infer_prompt_format_from_model_id(str(model_id))
    eos_id = eos_for_model(tokenizer, str(model_id))

    # Ensure pad token is set (some tokenizers lack it)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    model.eval()

    print(f"[Eval][GSM8K] split={split} n={n_total} batch_size={batch_size}", flush=True)

    # Process in batches
    num_batches = (n_total + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), disable=not progress_bar, desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_total)
        batch_ds = ds.select(range(start_idx, end_idx))
        
        # Prepare batch prompts
        batch_prompts = []
        batch_gold_answers = []
        
        for ex in batch_ds:
            question = ex["question"]
            answer_raw = ex["answer"]
            
            # Extract ground-truth final numeric answer (#### N)
            gt_match = re.search(r"####\s*([0-9.,]+)", answer_raw)
            gt_final = normalize_number(gt_match.group(1)) if gt_match else None
            
            if use_fewshot:
                prompt = build_fewshot_prompt(question, num_fewshot_examples)
            else:
                prompt = f"Question: {question}\nAnswer:"
            batch_prompts.append(prompt)
            batch_gold_answers.append(gt_final)

        if not batch_prompts:
            continue

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        # Generate responses using EleutherAI standard parameters with stopping criteria
        stopping_criteria = StoppingCriteriaList([QStoppingCriteria(tokenizer)])
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # EleutherAI uses deterministic generation
            temperature=0.0,  # EleutherAI uses temperature 0.0
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_id,
            stopping_criteria=stopping_criteria,
        )

        # Decode responses
        preds_full = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        # Process each example in the batch
        for i, (prompt, gt_final, pred_full) in enumerate(zip(batch_prompts, batch_gold_answers, preds_full)):
            # Extract prediction
            pred_extracted = extract_final_answer(pred_full)
            
            # Check if correct
            is_correct = pred_extracted is not None and gt_final is not None and pred_extracted == gt_final
            if is_correct:
                correct += 1
            
            # Debug print for first few examples
            if debug_print and batch_idx < 2 and i < 3:
                print("="*70)
                print(f"Q: {batch_ds[i]['question']}")
                print(f"GT raw: {batch_ds[i]['answer']}")
                print(f"GT (final): {gt_final}")
                print(f"Pred (full): {pred_full}")
                print(f"Pred (extracted): {pred_extracted}")
                print(f"Compare → pred={pred_extracted} vs gt={gt_final} → {'✓' if is_correct else '✗'}")

    # Calculate accuracy
    accuracy = correct / n_total * 100 if n_total > 0 else 0.0
    
    # Prepare results
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": n_total,
        "em": accuracy,  # For compatibility with existing code
        "n": n_total,    # For compatibility with existing code
    }
    
    # Logging
    gpu_info = _get_gpu_info()
    elapsed_time = time.time() - start_time
    
    log_record = {
        "dataset": "gsm8k",
        "split": split,
        "run_name": run_name,
        "phase": phase,
        "epoch": epoch,
        "step": step,
        "model_id": str(model_id),
        "accuracy": accuracy,
        "correct": correct,
        "total": n_total,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "elapsed_time": elapsed_time,
        "output_dir": output_dir,
        **gpu_info
    }
    
    print(f"[Eval][GSM8K] Accuracy: {accuracy:.2f}% ({correct}/{n_total}) "
          f"Time: {elapsed_time:.1f}s GPU: {gpu_info['gpu']}", flush=True)
    
    # Save logs if requested
    if save_jsonl:
        _append_jsonl(save_jsonl, log_record)
    
    if save_tsv:
        field_order = ["dataset", "split", "run_name", "phase", "epoch", "step", 
                      "model_id", "accuracy", "correct", "total", "batch_size", 
                      "max_new_tokens", "elapsed_time", "output_dir", 
                      "gpu", "gpu_id", "gpu_mem_alloc", "gpu_mem_reserved"]
        _append_tsv(save_tsv, log_record, field_order)
    
    # Restore model state
    model.config.use_cache = prev_cache
    
    return results