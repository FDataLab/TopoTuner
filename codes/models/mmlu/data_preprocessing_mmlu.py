import re
import torch
from torch.nn.utils.rnn import pad_sequence

# --------------- Special markers ---------------
CHOICE_LETTERS = ["A", "B", "C", "D"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the multiple-choice question by reasoning briefly "
    "and then state the final choice. Respond with the final option letter wrapped in <final-answer> tags.\n"
    "Only output the final answer letter, nothing else."
    "Example: Answer: <final-answer>A</final-answer>"
)

def _format_mcq(question: str, choices: list[str]) -> str:
    lines = [question.strip(), ""]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)

# --------------- Prompt builders ---------------
def _format_few_shot_examples(examples: list[dict]) -> str:
    parts = []
    for shot in examples:
        shot_q = shot.get("question", "")
        shot_choices = shot.get("choices", [])
        shot_ans = shot.get("answer", "")
        parts.append(_format_mcq(shot_q, shot_choices))
        parts.append(f"Answer: <final-answer>{shot_ans}</final-answer>")
        parts.append("")
    return "\n".join(parts).strip()


def create_prompt_llama3(tokenizer, question: str, choices: list[str], use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT, few_shot_examples: list[dict] | None = None) -> str:
    messages = []
    if use_instruction:
        messages.append({"role": "system", "content": prompt_template})
    if few_shot_examples:
        for shot in few_shot_examples:
            messages.append({"role": "user", "content": _format_mcq(shot.get("question", ""), shot.get("choices", []))})
            messages.append({"role": "assistant", "content": f"Answer: <final-answer>{shot.get('answer', '')}</final-answer>"})
    messages.append({"role": "user", "content": _format_mcq(question, choices)})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = _format_few_shot_examples(few_shot_examples or [])
    few_shot_block = f"{prefix}\n\n" if prefix else ""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{prompt_template}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{few_shot_block}{_format_mcq(question, choices)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

def create_prompt_llama2(question: str, choices: list[str], use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT, few_shot_examples: list[dict] | None = None) -> str:
    prefix = _format_few_shot_examples(few_shot_examples or [])
    prefix_block = f"{prefix}\n\n" if prefix else ""
    body = f"{prefix_block}{_format_mcq(question, choices)}".strip()
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{body} [/INST]"
    else:
        return f"<s>[INST] {body} [/INST]"

def create_prompt_mistral(question: str, choices: list[str], use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT, few_shot_examples: list[dict] | None = None) -> str:
    prefix = _format_few_shot_examples(few_shot_examples or [])
    prefix_block = f"{prefix}\n\n" if prefix else ""
    body = f"{prefix_block}{_format_mcq(question, choices)}".strip()
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{body} [/INST]"
    else:
        return f"<s>[INST] {body} [/INST]"

def create_prompt_qwen(tokenizer, question: str, choices: list[str], use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT, few_shot_examples: list[dict] | None = None) -> str:
    messages = []
    if use_instruction:
        messages.append({"role": "system", "content": prompt_template})
    if few_shot_examples:
        for shot in few_shot_examples:
            messages.append({"role": "user", "content": _format_mcq(shot.get("question", ""), shot.get("choices", []))})
            messages.append({"role": "assistant", "content": f"Answer: <final-answer>{shot.get('answer', '')}</final-answer>"})
    messages.append({"role": "user", "content": _format_mcq(question, choices)})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sys_text = f"System: {prompt_template}\n" if use_instruction else ""
    prefix = _format_few_shot_examples(few_shot_examples or [])
    prefix_block = f"{prefix}\n\n" if prefix else ""
    return f"{sys_text}User: {prefix_block}{_format_mcq(question, choices)}\nAssistant:"

def create_prompt_olmo(question: str, choices: list[str], use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT, few_shot_examples: list[dict] | None = None) -> str:
    prefix = _format_few_shot_examples(few_shot_examples or [])
    prefix_block = f"{prefix}\n\n" if prefix else ""
    body = f"{prefix_block}{_format_mcq(question, choices)}".strip()
    if use_instruction:
        return f"{prompt_template}\n\n{body}\nAnswer:"
    else:
        return f"{body}\nAnswer:"

def infer_prompt_format_from_model_id(model_id: str) -> str:
    mid = (model_id or "").lower()
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
    return "qwen"

def eos_for_model(tokenizer, model_id: str):
    mid = (model_id or "").lower()
    if "llama-3" in mid or "llama3" in mid:
        try:
            return tokenizer.convert_tokens_to_ids("<|eot_id|>")
        except Exception:
            return tokenizer.eos_token_id
    if "qwen" in mid:
        try:
            return tokenizer.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            return tokenizer.eos_token_id
    return tokenizer.eos_token_id

LETTER_RE = re.compile(r"\b([A-D])\b")

def extract_choice_letter(text: str) -> str:
    # Prefer explicit tag
    m = re.search(r"<final-answer>\s*([A-D])\s*</final-answer>", text, flags=re.I)
    if m:
        return m.group(1).upper()
    # try last standalone letter occurrence
    letters = LETTER_RE.findall(text)
    return letters[-1].upper() if letters else ""

# ----------------- Preprocess for training -----------------
def preprocess_dataset(example, tokenizer, max_len: int = 1024, prompt_format: str = "llama3", is_train: bool = True):
    question = example["question"]
    # choices may be in dict with 'text' list or explicit A-D fields
    choices = [example.get("A"), example.get("B"), example.get("C"), example.get("D")]
    if not all(isinstance(x, str) and x for x in choices):
        # Handle MMLU format: choices is a list of strings
        if isinstance(example.get("choices"), list):
            choices = example["choices"]
        else:
            choices = example.get("choices", {}).get("text", []) if isinstance(example.get("choices"), dict) else []
    if prompt_format == "llama3":
        prompt = create_prompt_llama3(tokenizer, question, choices, use_instruction=True)
    elif prompt_format == "llama2":
        prompt = create_prompt_llama2(question, choices, use_instruction=True)
    elif prompt_format == "mistral":
        prompt = create_prompt_mistral(question, choices, use_instruction=True)
    elif prompt_format == "olmo":
        prompt = create_prompt_olmo(question, choices, use_instruction=True)
    else:
        prompt = create_prompt_qwen(tokenizer, question, choices, use_instruction=True)

    # For supervised fine-tuning, target is the correct letter, normalize as "Answer: <final-answer>X</final-answer>"
    answer = example.get("answer", "")
    
    # Convert numeric answer (0,1,2,3) to letter (A,B,C,D)
    if isinstance(answer, int):
        gold = chr(65 + answer)  # 0->A, 1->B, 2->C, 3->D
    else:
        gold = str(answer).strip().upper()
    
    target = f"Answer: <final-answer>{gold}</final-answer>"
    # GSM8K-style: include EOS in the supervised tail
    eos_tok = tokenizer.eos_token or ""

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(target + eos_tok, add_special_tokens=False)["input_ids"]

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

