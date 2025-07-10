from typing import Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

# IFEval system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful writing assistant. Your job is to classify the constraints of a writing instruction.

You will be given:
- A text prompt
- A list of high-level instruction IDs
- A list of detailed constraint objects (called "keywords")

Your task is to output a list of keyword field names used in the constraints. These field names must come from a predefined set of 25 possible keys.
⚠️ Output must be a list of dictionaries corresponding to the keywords and their values. Do not include any free-text explanation.

Here are the 25 possible field names:
["num_highlights", "relation", "num_words", "num_placeholders", "prompt_to_repeat", "num_bullets", "section_spliter", "num_sections", "capital_relation", "capital_frequency", "keywords", "num_paragraphs", "language", "let_relation", "letter", "let_frequency", "end_phrase", "forbidden_words", "keyword", "frequency", "num_sentences", "postscript_marker", "first_word", "nth_paragraph"]
---

Example:

Prompt: Write a 300+ word summary of the Wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections in markdown format.
Instruction IDs: [
  "punctuation:no_comma",
  "detectable_format:number_highlighted_sections",
  "length_constraints:number_words"
]

Output (Keywords):
[
  {"num_highlights": 3},
  {"relation": "at least", "num_words": 300}
]
"""

def print_existing_special_tokens(tokenizer):
    print(f"\n🔍 Printing existing special tokens from tokenizer '{tokenizer.name_or_path}'\n")
    
    # Loop over vocab to find special-looking tokens
    for token, token_id in tokenizer.get_vocab().items():
        # Print only special-looking tokens (you can refine this if needed)
        if token.startswith("<|") and token.endswith("|>"):
            print(f"{token_id}: AddedToken({repr(token)}, "
                  f"rstrip=False, lstrip=False, single_word=False, "
                  f"normalized=False, special=True)")

def build_prompt(prompt: str, instruction_ids: list, prompt_format: str, use_instruction: bool = True) -> str:
    base_prompt = (
        f"Prompt:\n{prompt}\n"
        f"Instruction IDs:\n{instruction_ids}\n"
        f"Constraints (kwargs):\n"
    )
    if prompt_format == "llama":
        if use_instruction:
            return f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n{base_prompt} [/INST]"
        else:
            return f"<s>[INST] {base_prompt} [/INST]"
    else:  # Qwen-style
        if use_instruction:
            return f"<｜begin_of_sentence｜><｜User｜>{DEFAULT_SYSTEM_PROMPT}\n{base_prompt}<｜end_of_sentence｜><｜Assistant｜>"
        else:
            return base_prompt

def build_answer(kwargs: list) -> str:
    formatted_constraints = []
    for i, constraint in enumerate(kwargs):
        parts = [f"{k}: {v}" for k, v in constraint.items() if v is not None]
        formatted_constraints.append(f"  - Constraint {i+1}: {{ {', '.join(parts)} }}")
    return "\n".join(formatted_constraints)

def preprocess_ifeval(
    example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_len: int = 1024,
    prompt_format: str = "qwen",
    use_instruction: bool = True,
) -> Dict:
    prompt_text = build_prompt(
        example["prompt"],
        example["instruction_id_list"],
        prompt_format=prompt_format,
        use_instruction=use_instruction
    )
    answer_text = build_answer(example["kwargs"])

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def custom_data_collator(features, tokenizer):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    max_len = max(max(len(x) for x in input_ids),
                  max(len(x) for x in attention_mask),
                  max(len(x) for x in labels))

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    padded_labels = []
    for l in labels:
        pad_len = max_len - len(l)
        if pad_len > 0:
            l = torch.cat([l, torch.full((pad_len,), -100, dtype=torch.long)])
        padded_labels.append(l[:max_len])
    labels = torch.stack(padded_labels)

    return {
        "input_ids": input_ids[:, :max_len],
        "attention_mask": attention_mask[:, :max_len],
        "labels": labels[:, :max_len],
    }
