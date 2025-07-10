import torch
from torch.nn.utils.rnn import pad_sequence

# Special tokens used for marking the final answer
SPECIAL_TOKENS = ["<final-answer>", "</final-answer>"]

# System prompt used in Qwen-style and LLaMA-style formats
DEFAULT_SYSTEM_PROMPT = """You are a highly skilled math teacher. Your task is to solve the given grade-school level word problem. 
Carefully analyze the problem, perform all necessary intermediate steps, and provide a final numeric answer.
Present your reasoning in a clear, step-by-step manner followed by the final answer on the last line.

Format:
- Start with step-by-step explanation.
- End with: "Answer: [final numeric answer]"

Example:
Question: Mary has 3 apples. She buys 2 more and then eats 1. How many apples does she have?
Solution:
Mary starts with 3 apples.
She buys 2 more: 3 + 2 = 5.
She eats 1: 5 - 1 = 4.
Answer: <final-answer>4</final-answer>
"""

# def print_existing_special_tokens(tokenizer):
#     print(f"\n🔍 Printing existing special tokens from tokenizer '{tokenizer.name_or_path}'\n")

#     special_tokens = tokenizer.special_tokens_map

#     # Print standard ones: bos/eos/pad
#     for key in ["bos_token", "eos_token", "pad_token"]:
#         token = special_tokens.get(key)
#         if token:
#             token_id = tokenizer.convert_tokens_to_ids(token)
#             print(f"{token_id}: AddedToken({repr(token)}, "
#                   f"rstrip=False, lstrip=False, single_word=False, "
#                   f"normalized=False, special=True)")

#     print("\n🔍 Printing additional special tokens:\n")

#     # Print additional_special_tokens
#     for token in special_tokens.get("additional_special_tokens", []):
#         token_id = tokenizer.convert_tokens_to_ids(token)
#         print(f"{token_id}: AddedToken({repr(token)}, "
#               f"rstrip=False, lstrip=False, single_word=False, "
#               f"normalized=False, special=True)")


def add_special_tokens_if_missing(tokenizer):
    """
    Add custom tokens like <final-answer> if not already present in tokenizer.
    """
    #import pdb; pdb.set_trace()
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)

import re

def wrap_final_answer(answer: str) -> str:
    """
    Wrap the last numeric answer with <final-answer>...</final-answer>.
    If 'Answer:' is present, it tries to locate the last number in that section.
    """
    if "Answer:" in answer:
        prefix, final = answer.rsplit("Answer:", maxsplit=1)
    else:
        prefix, final = "", answer

    # Find all numbers (can include decimals)
    matches = list(re.finditer(r"\b\d+(?:\.\d+)?\b", final.strip()))

    if not matches:
        # No numeric answer found; fallback to full wrapping
        return f"{prefix}Answer: <final-answer>{final.strip()}</final-answer>"

    # Get last number match
    last_match = matches[-1]
    start, end = last_match.span()
    numeric_part = final[start:end]

    # Reconstruct final with wrapped numeric answer
    wrapped_final = (
        final[:start]
        + f"<final-answer>{numeric_part}</final-answer>"
        + final[end:]
    )

    return f"{prefix}Answer: {wrapped_final.strip()}"

def create_prompt_qwen(tokenizer, question: str, use_instruction: bool = True, prompt_template: str = DEFAULT_SYSTEM_PROMPT) -> str:
    if use_instruction:
        return f"<｜begin_of_sentence｜><｜User｜>{prompt_template}\n{question}f{tokenizer.eos_token}"
    else:
        return question

def create_prompt_deepseek_llama(question: str, use_instruction: bool = True, prompt_template: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Format prompt in LLaMA style with [INST] and optional system instruction.
    """
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{question} [/INST]"
    else:
        return f"<s>[INST] {question} [/INST]"

def preprocess_dataset(example, tokenizer, max_len=1024, use_instruction=True, prompt_format="qwen", is_train=True):
    """
    Convert raw example into input/label for Causal LM training.
    Only the answer part is supervised — prompt tokens are masked with -100.
    """
    question = example["question"]
    answer = wrap_final_answer(example["answer"])

    # Create prompt based on the chosen format
    if prompt_format == "llama":
        prompt = create_prompt_deepseek_llama(question, use_instruction=use_instruction)
    else:
        prompt = create_prompt_qwen(tokenizer, question, use_instruction=use_instruction)

    # Tokenize prompt and answer separately
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    # Combine input_ids
    input_ids = prompt_ids + answer_ids

    # Mask prompt tokens with -100 in labels
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    # Truncate if needed
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]

    # Logging
    # print("🧾 Prompt:\n", prompt)
    # print("✅ Answer:\n", answer)
    # print("🧠 Tokens:\n", tokenizer.convert_ids_to_tokens(input_ids))
    # print("📌 Labels:\n", labels)

    detok_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    # print("🧾 Detokenized Input:\n", detok_text)

    # Debug shape
    if len(input_ids) == 0 or len(labels) == 0:
        print(f"⚠️ Empty sequence for: {example}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def custom_data_collator(features, tokenizer):
    """
    Pad input_ids, attention_mask, and labels to the same length.
    Label padding is done using -100 to ignore during loss computation.
    """
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    max_len = max(
        max(len(seq) for seq in input_ids),
        max(len(seq) for seq in attention_mask),
        max(len(seq) for seq in labels)
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    padded_labels = []
    for l in labels:
        if len(l) < max_len:
            l = torch.cat([l, torch.full((max_len - len(l),), -100)])
        else:
            l = l[:max_len]
        padded_labels.append(l)
    labels = torch.stack(padded_labels)

    return {
        "input_ids": input_ids[:, :max_len],
        "attention_mask": attention_mask[:, :max_len],
        "labels": labels[:, :max_len],
    }
