import torch
import random
import re
from torch.nn.utils.rnn import pad_sequence

# ----------------- Special tokens -----------------
SPECIAL_TOKENS = ["<final-answer>", "</final-answer>"]

# Clear instruction to wrap the final numeric answer in <final-answer> tags.
DEFAULT_SYSTEM_PROMPT = """You are a highly skilled math teacher. Your task is to solve the given grade-school level word problem.
Carefully analyze the problem, perform all necessary intermediate steps, and provide a final numeric answer.

Rules:
- Show a brief, step-by-step solution.
- Always wrap the final numeric answer with <final-answer>…</final-answer>.
- End with a single final numeric line in this exact format:
  Answer: <final-answer>[NUMBER]</final-answer>

Example:
Question: Mary has 3 apples. She buys 2 more and then eats 1. How many apples does she have?
Solution:
Mary starts with 3 apples.
She buys 2 more: 3 + 2 = 5.
She eats 1: 5 - 1 = 4.
Answer: <final-answer>4</final-answer>
"""

# ----------------- Utils -----------------
def add_special_tokens_if_missing(tokenizer):
    """Add <final-answer> tokens if they are not present (idempotent)."""
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

def wrap_final_answer(answer: str) -> str:
    """
    Wrap the last numeric answer with <final-answer>...</final-answer>.
    If 'Answer:' exists, target the last number after it.
    """
    if "Answer:" in answer:
        prefix, final = answer.rsplit("Answer:", maxsplit=1)
    else:
        prefix, final = "", answer

    matches = list(re.finditer(r"\b\d+(?:\.\d+)?\b", final.strip()))
    if not matches:
        return f"{prefix}Answer: <final-answer>{final.strip()}</final-answer>"

    last = matches[-1]
    s, e = last.span()
    num = final[s:e]
    wrapped_final = final[:s] + f"<final-answer>{num}</final-answer>" + final[e:]
    return f"{prefix}Answer: {wrapped_final.strip()}"

# ----------------- Prompt builders -----------------

# LLaMA-3: use the tokenizer's chat template (must pass tokenizer)
def create_prompt_llama3(tokenizer, question: str, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Build a prompt for LLaMA-3. Prefer the tokenizer's chat_template (if available),
    otherwise fall back to a manual template.
    """
    messages = []
    if use_instruction:
        messages.append({"role": "system", "content": prompt_template})
    messages.append({"role": "user", "content": question})

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        # official instruct models (e.g., Llama-3.1-8B-Instruct) have this
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # fallback if chat_template is missing (base model case)
        sys_text = prompt_template if use_instruction else ""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{sys_text}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

# LLaMA-2 style ([INST] ... <<SYS>> ...)
def create_prompt_llama2(question: str, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{question} [/INST]"
    else:
        return f"<s>[INST] {question} [/INST]"

# Mistral-Instruct commonly supports the same [INST] format
def create_prompt_mistral(question: str, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{question} [/INST]"
    else:
        return f"<s>[INST] {question} [/INST]"

# Qwen: prefer its chat template via tokenizer (needs tokenizer)
def create_prompt_qwen(tokenizer, question: str, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    messages = []
    if use_instruction:
        messages.append({"role": "system", "content": prompt_template})
    messages.append({"role": "user", "content": question})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback minimal role-tag style
        sys_text = f"System: {prompt_template}\n" if use_instruction else ""
        return f"{sys_text}User: {question}\nAssistant:"

# OLMo (no strict chat tokens; plain instruction+question is fine)
def create_prompt_olmo(question: str, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    if use_instruction:
        return f"{prompt_template}\n\nQuestion: {question}\nAnswer:"
    else:
        return question

def infer_prompt_format_from_model_id(model_id: str) -> str:
    mid = model_id.lower()
    if "llama-3" in mid or "llama3" in mid or "llama-3.1" in mid or "llama-3.2" in mid:
        return "llama3"
    if "llama-2" in mid or "llama2" in mid:
        return "llama2"
    if "mistral" in mid:
        return "mistral"
    if "qwen" in mid:
        return "qwen"
    if "olmo" in mid:
        return "olmo"
    # fallback if unknown
    return "qwen"

def eos_for_model(tokenizer, model_id: str):
    """
    Choose the correct end-of-turn / EOS token for generate().
    """
    mid = (model_id or "").lower()
    # LLaMA-3 uses <|eot_id|>
    if "llama-3" in mid or "llama3" in mid:
        try:
            return tokenizer.convert_tokens_to_ids("<|eot_id|>")
        except Exception:
            return tokenizer.eos_token_id
    # Qwen chat often uses <|im_end|>
    if "qwen" in mid:
        try:
            return tokenizer.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            return tokenizer.eos_token_id
    # others (llama2, mistral, olmo) use the default EOS
    return tokenizer.eos_token_id

# ----------------- Preprocess (mask prompt with -100) -----------------
def preprocess_dataset(
    example,
    tokenizer,
    max_len: int = 1024,
    use_instruction: bool = True,
    prompt_format: str = "llama3",  # set by finetune.py via infer_prompt_format_from_model_id(...)
    is_train: bool = True,
):
    """
    Convert raw GSM8K example into causal LM inputs.
    - Prompt tokens are masked in labels with -100.
    - Answer is wrapped with <final-answer>...</final-answer>.
    """
    question = example["question"]
    answer = wrap_final_answer(example["answer"])

    # Build prompt per format
    if prompt_format == "llama3":
        prompt = create_prompt_llama3(tokenizer, question, use_instruction=use_instruction)
    elif prompt_format == "llama2":
        prompt = create_prompt_llama2(question, use_instruction=use_instruction)
    elif prompt_format == "mistral":
        prompt = create_prompt_mistral(question, use_instruction=use_instruction)
    elif prompt_format == "olmo":
        prompt = create_prompt_olmo(question, use_instruction=use_instruction)
    else:  # "qwen"
        prompt = create_prompt_qwen(tokenizer, question, use_instruction=use_instruction)

    # Tokenize separately and then concat to mask prompt tokens
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    # Truncate
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]

    # Optional light debug
    if len(input_ids) == 0 or len(labels) == 0:
        print(f"⚠️ Empty sequence for: {example}")
    if random.random() < 0.01:
        pass
        # print("INPUT IDS:", input_ids)
        # print("LABELS   :", labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ----------------- Collator (pad; ignore loss on -100) -----------------
def custom_data_collator(features, tokenizer):
    """
    Pad input_ids, attention_mask, and labels to the same length.
    Label padding uses -100 so it's ignored by the loss.
    """
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    max_len = max(
        max(len(x) for x in input_ids),
        max(len(x) for x in attention_mask),
        max(len(x) for x in labels),
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    padded_labels = []
    for l in labels:
        if len(l) < max_len:
            l = torch.cat([l, torch.full((max_len - len(l),), -100, dtype=torch.long)])
        else:
            l = l[:max_len]
        padded_labels.append(l)
    labels = torch.stack(padded_labels)

    return {
        "input_ids": input_ids[:, :max_len],
        "attention_mask": attention_mask[:, :max_len],
        "labels": labels[:, :max_len],
    }

if __name__ == "__main__":
    pass