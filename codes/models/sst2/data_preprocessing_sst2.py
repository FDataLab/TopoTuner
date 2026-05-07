import random
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

SST2_SYSTEM_PROMPT = """You are a helpful assistant that analyzes single-sentence movie reviews and determines their sentiment.

Classify the sentiment as either positive or negative.

You must answer only in one of these exact formats:
<final-answer>positive</final-answer>
<final-answer>negative</final-answer>"""

def wrap_final_answer(text: str) -> str:
    return f"<final-answer>{text}</final-answer>"

def eos_for_model(tokenizer, model_id: str):
    """Get the appropriate EOS token for different models."""
    if "llama" in model_id.lower():
        return tokenizer.eos_token_id
    elif "mistral" in model_id.lower():
        return tokenizer.eos_token_id
    elif "qwen" in model_id.lower():
        return tokenizer.eos_token_id
    elif "olmo" in model_id.lower():
        return tokenizer.eos_token_id
    else:
        return tokenizer.eos_token_id

def _create_prompt_generic(text: str, use_instruction: bool, prompt_template: str) -> str:
    instruction = f"{prompt_template}\n\n" if use_instruction else ""
    return f"{instruction}Review:\n{text}\n\nAnswer:"

def create_prompt_llama3(tokenizer, text: str, use_instruction: bool, prompt_template: str) -> str:
    return _create_prompt_generic(text, use_instruction, prompt_template)

def create_prompt_llama2(text: str, use_instruction: bool, prompt_template: str) -> str:
    return _create_prompt_generic(text, use_instruction, prompt_template)

def create_prompt_mistral(text: str, use_instruction: bool, prompt_template: str) -> str:
    return _create_prompt_generic(text, use_instruction, prompt_template)

def create_prompt_olmo(text: str, use_instruction: bool, prompt_template: str) -> str:
    return _create_prompt_generic(text, use_instruction, prompt_template)

def create_prompt_qwen(tokenizer, text: str, use_instruction: bool, prompt_template: str) -> str:
    return _create_prompt_generic(text, use_instruction, prompt_template)

def build_prompt(tokenizer, text: str, prompt_format: str, use_instruction: bool = True) -> str:
    if prompt_format == "llama3":
        return create_prompt_llama3(tokenizer, text, use_instruction=use_instruction, prompt_template=SST2_SYSTEM_PROMPT)
    elif prompt_format == "llama2":
        return create_prompt_llama2(text, use_instruction=use_instruction, prompt_template=SST2_SYSTEM_PROMPT)
    elif prompt_format == "mistral":
        return create_prompt_mistral(text, use_instruction=use_instruction, prompt_template=SST2_SYSTEM_PROMPT)
    elif prompt_format == "olmo":
        return create_prompt_olmo(text, use_instruction=use_instruction, prompt_template=SST2_SYSTEM_PROMPT)
    else:  # "qwen"
        return create_prompt_qwen(tokenizer, text, use_instruction=use_instruction, prompt_template=SST2_SYSTEM_PROMPT)

def infer_prompt_format_from_model_id(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if "llama-3" in model_id_lower or "llama3" in model_id_lower:
        return "llama3"
    elif "llama-2" in model_id_lower or "llama2" in model_id_lower:
        return "llama2"
    elif "mistral" in model_id_lower:
        return "mistral"
    elif "olmo" in model_id_lower:
        return "olmo"
    elif "qwen" in model_id_lower:
        return "qwen"
    else:
        return "llama3"  # default

def preprocess_dataset(
    example,
    tokenizer,
    max_len: int = 256,
    use_instruction: bool = True,
    prompt_format: str = "llama3",
    is_train: bool = True,
):
    text = example["sentence"]
    label = example["label"]
    sentiment = "positive" if label == 1 else "negative"
    prompt = build_prompt(tokenizer, text, prompt_format=prompt_format, use_instruction=use_instruction)
    answer = wrap_final_answer(sentiment)
    # GSM8K-style: include EOS in the supervised tail (loss on answer + EOS)
    eos_tok = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + eos_tok, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def custom_data_collator(features, tokenizer):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }