import torch
import random
import re
from torch.nn.utils.rnn import pad_sequence

# Instruction tailored for HotpotQA to produce a concise final short answer.
# OLD FORMAT (for testing):
DEFAULT_SYSTEM_PROMPT_OLD = """You are a knowledgeable assistant. Your task is to answer multi-hop questions based on evidence.

Rules:
- Show brief, step-by-step reasoning.
- End with a single final short answer.
- Output the final line in this exact format:
  Answer: [SHORT_ANSWER]
"""

# SIMPLIFIED FORMAT (easier for model to learn):
DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant answering multi-hop questions.

Guidelines:
- Combine information from multiple pieces of evidence.
- Reason internally and concisely.
- Use only the provided evidence; do not rely on outside knowledge.
- Provide only the final answer in the required format.

Output format:
Answer: [SHORT_ANSWER]
"""

# ----------------- Prompt builders -----------------

def build_hotpot_context(example, evidence_mode: str = "supporting") -> str | None:
    """
    Returns context string.
    evidence_mode:
      - "full": use all context passages
      - "supporting": use only sentences referenced by supporting_facts
    HotpotQA HF format:
      example["context"] = [[title, [sent1, sent2, ...]], ...]
      example["supporting_facts"] = [[title, sent_idx], ...]
    """
    ctx_val = example.get("context")
    if not ctx_val:
        return None

    # Normalize context into map: title -> list_of_sentences
    ctx_map = {}
    if isinstance(ctx_val, list):
        for item in ctx_val:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                title, sents = item
            elif isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences", [])
            else:
                continue
            if title is None:
                title = ""
            if not isinstance(sents, (list, tuple)):
                sents = [str(sents)]
            ctx_map[str(title)] = [str(x) for x in sents]
    elif isinstance(ctx_val, dict):
        # Some formats store parallel lists
        titles = ctx_val.get("title", [])
        sentences = ctx_val.get("sentences", [])
        for title, sents in zip(titles, sentences):
            if not isinstance(sents, (list, tuple)):
                sents = [str(sents)]
            ctx_map[str(title)] = [str(x) for x in sents]

    if evidence_mode == "supporting":
        supp = example.get("supporting_facts", []) or []
        lines = []
        for item in supp:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title, sent_idx = item[0], item[1]
            title = str(title)
            sents = ctx_map.get(title, [])
            if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                lines.append(f"{title}: {sents[sent_idx]}")
        # Fallback to full if something goes wrong
        if lines:
            return "\n".join(lines)
        evidence_mode = "full"

    # evidence_mode == "full"
    parts = []
    for title, sents in ctx_map.items():
        joined = " ".join(sents)
        parts.append(f"{title}: {joined}".strip())
    return "\n".join(parts) if parts else None


# LLaMA-3: use the tokenizer's chat template (must pass tokenizer)
def create_prompt_llama3(tokenizer, question: str, context: str | None = None, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    messages = []
    if use_instruction:
        messages.append({"role": "system", "content": prompt_template})
    user_text = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        sys_text = prompt_template if use_instruction else ""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{sys_text}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

def create_prompt_llama2(question: str, context: str | None = None, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    body = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{body} [/INST]"
    else:
        return f"<s>[INST] {body} [/INST]"

def create_prompt_mistral(question: str, context: str | None = None, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    body = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    if use_instruction:
        return f"<s>[INST] <<SYS>>\n{prompt_template}\n<</SYS>>\n\n{body} [/INST]"
    else:
        return f"<s>[INST] {body} [/INST]"

def create_prompt_qwen(tokenizer, question: str, context: str | None = None, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    messages = []
    if use_instruction:
        messages.append({"role": "system", "content": prompt_template})
    user_text = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_text})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        sys_text = f"System: {prompt_template}\n" if use_instruction else ""
        return f"{sys_text}User: {user_text}\nAssistant:"

def create_prompt_olmo(question: str, context: str | None = None, use_instruction=True, prompt_template=DEFAULT_SYSTEM_PROMPT) -> str:
    body = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    if use_instruction:
        return f"{prompt_template}\n\n{body}\nAnswer:"
    else:
        return body

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

def preprocess_dataset(
    example,
    tokenizer,
    max_len: int = 2048,
    use_instruction: bool = True,
    prompt_format: str = "llama3",
    is_train: bool = True,
    evidence_mode: str = "supporting",   # NEW
):
    """
    Convert raw HotpotQA example into causal LM inputs.
    - Prompt tokens are masked in labels with -100.
    - Target text ends with: "Answer: ...".
    - evidence_mode:
        "full"       -> full context passages
        "supporting" -> only supporting_facts sentences
    """
    question = example["question"]
    short_answer = str(example.get("answer", "")).strip()

    # NEW: build context based on evidence_mode
    context = build_hotpot_context(example, evidence_mode=evidence_mode)

    answer = f"Answer: {short_answer}".strip()

    # Build prompt per format (unchanged)
    if prompt_format == "llama3":
        prompt = create_prompt_llama3(tokenizer, question, context=context, use_instruction=use_instruction)
    elif prompt_format == "llama2":
        prompt = create_prompt_llama2(question, context=context, use_instruction=use_instruction)
    elif prompt_format == "mistral":
        prompt = create_prompt_mistral(question, context=context, use_instruction=use_instruction)
    elif prompt_format == "olmo":
        prompt = create_prompt_olmo(question, context=context, use_instruction=use_instruction)
    else:  # "qwen"
        prompt = create_prompt_qwen(tokenizer, question, context=context, use_instruction=use_instruction)

    # Tokenize separately and then concat to mask prompt tokens
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_len:
        answer_len = len(answer_ids)
        keep_prompt = max_len - answer_len
        if keep_prompt <= 0:
            input_ids = (prompt_ids + answer_ids)[-max_len:]
            labels = ([-100] * len(prompt_ids) + answer_ids)[-max_len:]
            attention_mask = [1] * len(input_ids)
        else:
            trimmed_prompt = prompt_ids[-keep_prompt:] if keep_prompt < len(prompt_ids) else prompt_ids
            input_ids = trimmed_prompt + answer_ids
            labels = [-100] * len(trimmed_prompt) + answer_ids
            attention_mask = [1] * len(input_ids)

    if len(input_ids) == 0 or len(labels) == 0:
        print(f"⚠️ Empty sequence for: {example}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# ----------------- Collator (pad; ignore loss on -100) -----------------
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

