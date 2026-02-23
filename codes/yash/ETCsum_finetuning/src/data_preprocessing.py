import torch
from typing import Dict, Union, List, Any
import numpy as np

# Improved system prompt with clearer instruction and output format
SYSTEM_PROMPT_FINANCIAL_SUMMARY = """You are a highly skilled financial analyst. Your task is to carefully analyze the provided financial document and generate a concise, structured summary. The summary should be in bullet points, with each point representing a key insight, trend, or significant financial detail. Ensure the summary is brief, accurate, and directly addresses the core financial information."""


DEFAULT_SYSTEM_PROMPT = """You are a financial expert. Read the financial document carefully and generate a concise bullet-point summary. Each bullet point should highlight a key insight or important detail. Keep the summary brief, clear, and to the point."""

# Instruction template for clarity
INSTRUCTION_TEMPLATE = "Analyze the following financial document and provide a bullet-point summary of the key findings:"


# very common
def create_prompt(text: str) -> str:
    """Create a prompt for instruction fine-tuning, incorporating a clearer structure."""
    return f"<instruction>{SYSTEM_PROMPT_FINANCIAL_SUMMARY}\n\n{INSTRUCTION_TEMPLATE}\n\n### Input: {text.strip()}\n\n### Response:\n- "
# Default system prompt for the financial summarization task

# model specific
def create_prompt_deepseek_qwen(text: str) -> str:
 
    return f"<｜begin of sentence｜><｜User｜>{DEFAULT_SYSTEM_PROMPT}\n\nSummarize the following financial document:\n{text.strip()}<｜Assistant｜>"




def preprocess_sample(data: Dict[str, Any], tokenizer, max_len=8192, is_train:bool=True ) -> Dict[str, Any]:
    """Preprocess a single data for model training."""
    try:
        # Extract text and summary based on data structure
        if isinstance(data.get("text"), dict):
            doc = data["text"].get("text", "")
            summary = data["text"].get("summary", "")
        else:
            # Try alternate format
            doc = data.get("text", "")
            summary = data.get("summary", "")
            
            # If still empty, try to use 'doc' directly
            if not doc:
                doc = data.get("doc", "")
        
        # Create the prompt format
        # you can customize the prompt based on models
        if is_train:
            prompt = create_prompt_deepseek_qwen(doc) + f"""\n### Response: {summary.strip()}"""
        else:
            prompt =create_prompt_deepseek_qwen(doc)
        
        # Tokenize the text with padding and truncation
        tokenized = tokenizer(
            prompt,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        
        # Remove the batch dimension that tokenizer adds when return_tensors="pt"
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        
        # Store original text
        tokenized["combined_text"] = prompt
        
        # Prepare labels for causal language modeling
        # if is_train:
        tokenized["labels"] = tokenized["input_ids"].clone()
        # else:
        #     # For validation, we don't need labels for the input part
        #     # We can pad with -100 (the ignore index in PyTorch)
        #     input_len = prompt.rfind("### Response:") + len("### Response:")
        #     input_tokens = tokenizer(prompt[:input_len], max_length=max_len, truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
        #     labels = [-100] * len(input_tokens)
        #     tokenized["labels"] = torch.tensor(labels)
        return tokenized
    
    except Exception as e:
        print(f"Error preprocessing example: {e}")
        # Return empty tensors with the right shape as fallback
        input_ids = torch.zeros(max_len, dtype=torch.long)
        attention_mask = torch.zeros(max_len, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "combined_text": ""
        }



    