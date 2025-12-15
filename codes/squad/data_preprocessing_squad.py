import torch
import random
from torch.nn.utils.rnn import pad_sequence

# We will reuse prompt builders and utilities from HotpotQA but with a SQuAD-specific system prompt
from codes.hotpotqa.data_preprocessing_hotpotqa import (
    create_prompt_llama2,
    create_prompt_llama3,
    create_prompt_mistral,
    create_prompt_qwen,
    create_prompt_olmo,
    infer_prompt_format_from_model_id,
    eos_for_model,
)

SQUAD_SYSTEM_PROMPT = """You are a reading comprehension assistant. Use the provided context to answer the question.

Rules:
- Show brief, step-by-step reasoning when needed.
- End with a single final short answer.
- Output the final line in this exact format:
  Answer: [SHORT_ANSWER]
"""

def build_prompt(tokenizer, question: str, context: str | None, prompt_format: str, use_instruction: bool = True) -> str:
    if prompt_format == "llama3":
        return create_prompt_llama3(tokenizer, question, context=context, use_instruction=use_instruction, prompt_template=SQUAD_SYSTEM_PROMPT)
    if prompt_format == "llama2":
        return create_prompt_llama2(question, context=context, use_instruction=use_instruction, prompt_template=SQUAD_SYSTEM_PROMPT)
    if prompt_format == "mistral":
        return create_prompt_mistral(question, context=context, use_instruction=use_instruction, prompt_template=SQUAD_SYSTEM_PROMPT)
    if prompt_format == "olmo":
        return create_prompt_olmo(question, context=context, use_instruction=use_instruction, prompt_template=SQUAD_SYSTEM_PROMPT)
    return create_prompt_qwen(tokenizer, question, context=context, use_instruction=use_instruction, prompt_template=SQUAD_SYSTEM_PROMPT)

def preprocess_dataset(
    example,
    tokenizer,
    max_len: int = 1024,
    use_instruction: bool = True,
    prompt_format: str = "llama3",
    is_train: bool = True,
):
    """
    Convert raw SQuAD example into causal LM inputs.
    - Prompt tokens are masked in labels with -100.
    - Target ends with a plain short answer line: "Answer: ...".
    """
    question = example["question"]
    context = example.get("context")
    # SQuAD answers is a dict with list of texts; use the first
    answers = example.get("answers", {}).get("text", [])
    short_answer = (answers[0] if answers else "").strip()

    prompt = build_prompt(tokenizer, question, context=context, prompt_format=prompt_format, use_instruction=use_instruction)
    target = f"Answer: {short_answer}".strip()

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    # Truncate
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = attention_mask[:max_len]

    if len(input_ids) == 0 or len(labels) == 0:
        print(f"⚠️ Empty sequence for: {example}")
    if random.random() < 0.01:
        pass

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def custom_data_collator(features, tokenizer):
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

