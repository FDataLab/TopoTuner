import os
import logging
from typing import Dict, Any, List, Tuple

import re
import string

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


LOG_DIR = "/home/kadir/topo/logs"
os.makedirs(LOG_DIR, exist_ok=True)
# Will be set based on prompt version
LOG_PATH = None


def setup_logging(log_path: str = None) -> None:
    global LOG_PATH
    if log_path:
        LOG_PATH = log_path
    elif LOG_PATH is None:
        LOG_PATH = os.path.join(LOG_DIR, "hotpotqa_exploration_multi.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def normalize_context(ex: Dict[str, Any]) -> str:
    ctx = ex["context"]
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(f"{title}: {' '.join(sents)}")
        return "\n".join(parts)
    else:
        parts = []
        for title, sents in ctx:
            parts.append(f"{title}: {' '.join(sents)}")
        return "\n".join(parts)


def _normalize_text(s: str) -> str:
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def answer_em_f1(pred: str, gold: str) -> Tuple[float, float]:
    p = _normalize_text(pred)
    g = _normalize_text(gold)
    em = 1.0 if p == g else 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    if not p_tokens and not g_tokens:
        return em, 1.0
    if not p_tokens or not g_tokens:
        return em, 0.0
    common = {}
    for t in p_tokens:
        if t in g_tokens:
            common[t] = min(common.get(t, 0) + 1, g_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return em, 0.0
    prec = num_same / len(p_tokens)
    rec = num_same / len(g_tokens)
    f1 = 2 * prec * rec / (prec + rec)
    return em, f1


def extract_answer(text: str) -> str:
    """
    Extract the final answer from model output.
    Uses the same robust extraction logic as eval_hotpotqa.py
    Always takes the FIRST clean answer and stops at repetitive patterns.
    """
    # Repetitive patterns that indicate we should stop processing
    repetitive_patterns = [
        r'#\+#',  # Pattern like "#+# # system"
        r'systematically\s+user',
        r'sistema\w*\s+user',
        r'preferredStyle',
        r'1st\s+century',
        r'\d+st\s+century',
        r'definitely\s+systematically',
        r'^\s*\d+st\s+',  # Lines starting with "1st ", "2nd ", etc.
    ]
    
    # Find FIRST "Answer:" occurrence
    first_answer_idx = text.find("Answer:")
    if first_answer_idx == -1:
        # No "Answer:" found - base models often don't follow format
        # Fallback: try to extract first meaningful line/sentence
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines:
            # Skip obvious junk patterns
            line_lower = line.lower()
            if any(skip in line_lower for skip in [
                "question:", "user:", "assistant:", "context:", "step",
                "reasoning", "identify", "determine", "find", "based on",
                "from the", "the text", "the context", "let me", "i need",
                "okay, let's", "first,", "next,", "then,", "finally,"
            ]):
                continue
            # Skip if it's too long (likely reasoning, not answer)
            if len(line.split()) > 20:
                continue
            # Found a potential answer line
            # Clean it up
            answer = line.strip()
            # Remove trailing period if present
            if len(answer) > 1 and answer.endswith('.'):
                if not re.match(r'^[A-Z]\.$', answer) and not re.match(r'^([A-Z]\.)+$', answer):
                    answer = answer.rstrip('.')
            return answer
        
        # If no good line found, try first sentence
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 2:
                continue
            sent_lower = sent.lower()
            if any(skip in sent_lower for skip in [
                "question:", "user:", "assistant:", "context:", "step",
                "reasoning", "identify", "determine", "find"
            ]):
                continue
            if len(sent.split()) <= 15:  # Reasonable answer length
                return sent.strip()
        
        # Last resort: return first non-empty line
        if lines:
            return lines[0].strip()
        
        return ""
    
    # Extract text starting from first "Answer:"
    text_from_answer = text[first_answer_idx + 7:].strip()  # +7 to skip "Answer:"
    
    # Find where to stop (first repetitive pattern or reasonable boundary)
    stop_idx = len(text_from_answer)
    for pattern in repetitive_patterns:
        match = re.search(pattern, text_from_answer, re.IGNORECASE | re.MULTILINE)
        if match:
            stop_idx = min(stop_idx, match.start())
    
    # Also stop at second "Answer:" if it appears (indicates repetition)
    second_answer = text_from_answer.find("Answer:", 1)
    if second_answer != -1:
        stop_idx = min(stop_idx, second_answer)
    
    # Extract the answer portion
    answer_text = text_from_answer[:stop_idx].strip()
    
    # Take only the first line (stop at newline)
    first_line = answer_text.split('\n')[0].strip()
    
    # Clean up the answer - detect and remove trailing junk patterns
    # Pattern 1: Remove ".1st", ".2nd", etc. and everything after
    first_line = re.sub(r'\.\d+st.*$', '', first_line, flags=re.IGNORECASE).strip()
    first_line = re.sub(r'\.\d+.*$', '', first_line).strip()  # Remove ".1", ".2", etc.
    first_line = re.sub(r'\.#.*$', '', first_line).strip()  # Remove ".#" patterns
    
    # Pattern 2: Remove patterns like "Famemy", "Famemyประโยค", "Famesystem" etc.
    # These are malformed concatenations where junk got attached to the answer
    # Look for lowercase letters or non-ASCII chars directly after a word
    first_line = re.sub(r'([A-Z][a-zA-Z\s]+?)(?:[a-z]{1,3}(?:system|my|ประโยค|system\d+)).*$', r'\1', first_line, flags=re.IGNORECASE).strip()
    
    # Pattern 3: Remove standalone "system", "my", unicode patterns
    first_line = re.sub(r'(?:my|system|ประโยค|system\d+).*$', '', first_line, flags=re.IGNORECASE).strip()
    
    # Pattern 4: Remove any remaining repetitive patterns
    for pattern in repetitive_patterns:
        first_line = re.sub(pattern, '', first_line, flags=re.IGNORECASE).strip()
    
    # Remove XML tags if present
    first_line = re.sub(r'<final-answer>.*?</final-answer>', '', first_line, flags=re.S).strip()
    first_line = re.sub(r'</final-answer>.*$', '', first_line).strip()
    first_line = re.sub(r'final-answer.*$', '', first_line).strip()
    
    # Remove brackets
    first_line = re.sub(r'\[.*?\]', '', first_line).strip()
    
    # Pattern 5: Handle malformed concatenations like "Famemy", "Wordsystem"
    # Strategy: Find junk and remove it, preserving the word it's attached to
    
    # Look for junk patterns: "my", "system", etc. (case insensitive)
    # Try to find them attached to words first
    junk_attached_pattern = r'([A-Za-z]+)(my|system|ประโยค|system\d+)'
    attached_match = re.search(junk_attached_pattern, first_line, re.IGNORECASE)
    if attached_match:
        # Found junk attached to a word (e.g., "Famemy")
        word_part = attached_match.group(1)  # "Fame" from "Famemy"
        junk_part = attached_match.group(2)   # "my" from "Famemy"
        
        # Replace "word+junk" with just "word" in the string
        # Find the full match and replace it
        full_match = attached_match.group(0)  # "Famemy"
        first_line = first_line.replace(full_match, word_part, 1)  # Replace first occurrence only
        
        # Now remove any remaining junk patterns
        first_line = re.sub(r'(?:my|system|ประโยค|system\d+).*$', '', first_line, flags=re.IGNORECASE).strip()
    else:
        # No attached junk, look for standalone junk patterns
        junk_pattern = r'(?:my|system|ประโยค|system\d+)'
        junk_match = re.search(junk_pattern, first_line, re.IGNORECASE)
        if junk_match:
            # Extract everything before the junk
            first_line = first_line[:junk_match.start()].strip()
    
    # Final cleanup: remove trailing period if it's standalone (not part of abbreviation)
    # But be careful - "U.S." should keep the period
    # Simple heuristic: if period is followed by nothing or whitespace, and answer is > 1 char, remove it
    if len(first_line) > 1 and first_line.endswith('.'):
        # Check if it's likely an abbreviation (short word with periods)
        if not re.match(r'^[A-Z]\.$', first_line) and not re.match(r'^([A-Z]\.)+$', first_line):
            first_line = first_line.rstrip('.')
    
    return first_line


# Examples for the prompt template
EXAMPLES = """Example 1:
Search Results:
Document 1: The Beatles were a British rock band formed in Liverpool in 1960.
Document 2: Liverpool is a city in England.

Question: What country were The Beatles from?
Answer: England

Example 2:
Search Results:
Document 1: Python is a programming language created by Guido van Rossum.
Document 2: Guido van Rossum is Dutch.

Question: What nationality is the creator of Python?
Answer: Dutch"""


def build_messages(context_text: str, question: str, include_irrelevant_note: bool = False) -> List[Dict[str, str]]:
    """
    Build chat-style messages using the new prompt template format.
    Format: "Write a high-quality answer... For example: {examples} {search_results} Question: {question} Answer:"
    
    Args:
        include_irrelevant_note: If True, adds "(some of which might be irrelevant)" to the prompt
    """
    # Format context as "Search Results" with document markers
    search_results = f"Search Results:\n{context_text}"
    
    # Build the base instruction
    if include_irrelevant_note:
        instruction = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
    else:
        instruction = "Write a high-quality answer for the given question using only the provided search results."
    
    # Build the full prompt
    prompt_text = (
        f"{instruction} "
        f"For example: {EXAMPLES}\n\n"
        f"{search_results}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    return [
        {"role": "user", "content": prompt_text},
    ]


def build_fallback_prompt(context_text: str, question: str, include_irrelevant_note: bool = False) -> str:
    """
    Fallback non-chat prompt for models without a chat_template.
    Uses the same template format but without chat structure.
    
    Args:
        include_irrelevant_note: If True, adds "(some of which might be irrelevant)" to the prompt
    """
    # Format context as "Search Results"
    search_results = f"Search Results:\n{context_text}"
    
    # Build the base instruction
    if include_irrelevant_note:
        instruction = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant)."
    else:
        instruction = "Write a high-quality answer for the given question using only the provided search results."
    
    # Build the full prompt
    prompt_text = (
        f"{instruction} "
        f"For example: {EXAMPLES}\n\n"
        f"{search_results}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    return prompt_text


MODEL_CONFIGS = [
    # Your 4 main Hotpot models (base)
    {
        "name": "Llama-3.1-8B",
        "hf_id": "meta-llama/Llama-3.1-8B",
    },
    {
        "name": "Llama-3.2-3B",
        "hf_id": "meta-llama/Llama-3.2-3B",
    },
    {
        "name": "Mistral-7B-v0.3",
        "hf_id": "mistralai/Mistral-7B-v0.3",
    },
    {
        "name": "Qwen-3-8B-Base",
        "hf_id": "Qwen/Qwen3-8B-Base",
    },
    # Two instruct-style models for comparison
    {
        "name": "Qwen-3-8B (Instruct)",
        "hf_id": "Qwen/Qwen3-8B",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
]


def run_probes(include_irrelevant_note: bool = False) -> None:
    """
    Run probes on models with specified prompt version.
    
    Args:
        include_irrelevant_note: If True, uses prompt with "(some of which might be irrelevant)"
    """
    logger = logging.getLogger(__name__)
    
    # Set log path based on prompt version
    if include_irrelevant_note:
        log_path = os.path.join(LOG_DIR, "hotpotqa_exploration_multi_v2_irrelevant.log")
    else:
        log_path = os.path.join(LOG_DIR, "hotpotqa_exploration_multi_v2_standard.log")
    
    # Update logging to use the new path
    setup_logging(log_path=log_path)

    logger.info("Loading HotpotQA distractor split...")
    logger.info("Prompt version: %s", "with '(some of which might be irrelevant)'" if include_irrelevant_note else "standard")
    ds = load_dataset("hotpot_qa", "distractor")
    val_ds = ds["validation"]
    logger.info("Validation size: %d", len(val_ds))

    N = 50
    val_subset = [val_ds[i] for i in range(N)]

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        model_id = cfg["hf_id"]
        logger.info("==============")
        logger.info("Model: %s (%s)", name, model_id)

        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=hf_token,
            )
            model.eval()
            model.config.use_cache = True
        except Exception as e:
            logger.exception("Failed to load model %s: %s", model_id, e)
            continue

        em_sum = 0.0
        f1_sum = 0.0

        with torch.inference_mode():
            for idx, ex in enumerate(val_subset):
                ctx_text = normalize_context(ex)
                q = ex["question"].strip()

                messages = build_messages(ctx_text, q, include_irrelevant_note=include_irrelevant_note)
                
                # Debug: log full prompt for first sample
                if idx == 0:
                    logger.info("--- DEBUG: First sample prompt ---")
                    if getattr(tok, "chat_template", None):
                        prompt_str = tok.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        )
                    else:
                        prompt_str = build_fallback_prompt(ctx_text, q, include_irrelevant_note=include_irrelevant_note)
                    logger.info("Prompt (first 500 chars): %s", prompt_str[:500])
                    logger.info("--- End debug ---")

                # Prefer chat templates if available (for Instruct / chat models)
                try:
                    if getattr(tok, "chat_template", None):
                        inputs = tok.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(model.device)
                        input_ids = inputs
                        attention_mask = None
                    else:
                        prompt = build_fallback_prompt(ctx_text, q, include_irrelevant_note=include_irrelevant_note)
                        enc = tok(prompt, return_tensors="pt").to(model.device)
                        input_ids = enc["input_ids"]
                        attention_mask = enc.get("attention_mask", None)
                except Exception:
                    # Fallback if chat_template path fails
                    prompt = build_fallback_prompt(ctx_text, q, include_irrelevant_note=include_irrelevant_note)
                    enc = tok(prompt, return_tensors="pt").to(model.device)
                    input_ids = enc["input_ids"]
                    attention_mask = enc.get("attention_mask", None)

                gen_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": 96,
                    "do_sample": False,  # Greedy decoding
                    "pad_token_id": tok.pad_token_id,
                    "eos_token_id": tok.eos_token_id,
                }
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask

                out = model.generate(**gen_kwargs)
                # Decode ONLY the newly generated tokens (not the prompt)
                input_len = input_ids.shape[1]
                generated_ids = out[0][input_len:]
                text = tok.decode(generated_ids, skip_special_tokens=True)
                
                # Debug: log full generated text for first sample
                if idx == 0:
                    logger.info("--- DEBUG: First sample generation ---")
                    logger.info("Full generated text (first 500 chars): %s", text[:500])
                    logger.info("--- End debug ---")
                
                pred = extract_answer(text)
                
                # Debug: log extraction result for first sample
                if idx == 0:
                    logger.info("--- DEBUG: Extraction result ---")
                    logger.info("Extracted answer: '%s'", pred)
                    logger.info("--- End debug ---")
                gold = str(ex["answer"]).strip()

                em, f1 = answer_em_f1(pred, gold)
                em_sum += em
                f1_sum += f1

                logger.info(
                    "[%s][%02d] Q=%s | Gold=%s | Pred=%s | EM=%.3f F1=%.3f",
                    name,
                    idx,
                    q,
                    gold,
                    pred,
                    em,
                    f1,
                )

        logger.info(
            "[%s] Aggregate over %d: EM=%.3f F1=%.3f (Prompt: %s)",
            name,
            N,
            em_sum / N,
            f1_sum / N,
            "with irrelevant note" if include_irrelevant_note else "standard",
        )

        # cleanup per-model
        del model
        torch.cuda.empty_cache()


def main() -> None:
    import sys
    
    # Check if we should run with irrelevant note variant
    include_irrelevant = "--irrelevant" in sys.argv or "-i" in sys.argv
    
    # Run both versions if no argument specified, otherwise run the specified one
    if len(sys.argv) == 1 or "--both" in sys.argv or "-b" in sys.argv:
        # Run standard version
        print("Running standard prompt version...")
        run_probes(include_irrelevant_note=False)
        
        # Run with irrelevant note version
        print("\nRunning prompt version with '(some of which might be irrelevant)'...")
        run_probes(include_irrelevant_note=True)
    else:
        # Run specified version
        run_probes(include_irrelevant_note=include_irrelevant)


if __name__ == "__main__":
    main()


