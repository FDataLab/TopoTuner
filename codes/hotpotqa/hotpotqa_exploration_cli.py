import os
import logging
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import string


LOG_DIR = "/home/kadir/topo/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "hotpotqa_exploration_llama32.log")


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers (if running multiple times)
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
    """Turn HotpotQA context into a multi-line string."""
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


def build_llama3_chat_prompt(system: str, user: str) -> str:
    """Construct a Llama-3 style chat prompt string (system + user, assistant turn)."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_answer_only_prompt(ex: Dict[str, Any], context_text: str) -> str:
    """Build a Llama-3 style chat prompt for answer-only."""
    q = ex["question"].strip()
    system = (
        "You are a knowledgeable assistant. Use the provided context to answer "
        "the user's question with a single short phrase starting with 'Answer:'."
    )
    user = (
        f"Context:\n{context_text}\n\n"
        f"Question: {q}\n\n"
        "Remember: respond as 'Answer: ...'"
    )
    return build_llama3_chat_prompt(system, user)


def extract_answer_from_text(text: str) -> str:
    """Extract the short answer from generated text."""
    if "Answer:" in text:
        tail = text.split("Answer:", 1)[1]
        lines = tail.splitlines()
        if not lines:
            return ""
        return lines[0].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def build_sup_ans_prompt(ex: Dict[str, Any], context_text: str) -> str:
    """Build a Llama-3 style chat prompt for Sup+Answer."""
    q = ex["question"].strip()
    system = (
        "You are a knowledgeable assistant. Use the provided context to answer "
        "the user's question and identify supporting facts."
    )
    user = (
        f"Context:\n{context_text}\n\n"
        f"Question: {q}\n\n"
        "First, list the supporting facts as lines in this format:\n"
        "- [Title] TITLE [Sent] INDEX: SENTENCE\n"
        "Then, on the last line, output the final short answer starting with:\n"
        "Answer:\n"
    )
    return build_llama3_chat_prompt(system, user)


def parse_supporting_facts(text: str) -> List[Tuple[str, int]]:
    """Parse '- [Title] {title} [Sent] {idx}: ...' lines."""
    sfs: List[Tuple[str, int]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- [Title]"):
            m = re.match(r"^- \[Title\] (.+?) \[Sent\] (\d+):", line)
            if m:
                title = m.group(1).strip()
                idx = int(m.group(2))
                sfs.append((title, idx))
    return sfs


def normalize_gold_supporting_facts(ex: Dict[str, Any]) -> List[Tuple[str, int]]:
    sf = ex.get("supporting_facts", {})
    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        return list(zip(titles, sent_ids))
    else:
        return list(sf)


def sup_em_f1(pred_sfs: List[Tuple[str, int]],
              gold_sfs: List[Tuple[str, int]]) -> Tuple[float, float]:
    from collections import Counter

    pred = Counter(pred_sfs)
    gold = Counter(gold_sfs)
    em = 1.0 if pred == gold else 0.0
    common = sum((pred & gold).values())
    if common == 0:
        return em, 0.0
    prec = common / max(1, sum(pred.values()))
    rec = common / max(1, sum(gold.values()))
    f1 = 2 * prec * rec / (prec + rec)
    return em, f1


def run_probes() -> None:
    logger = logging.getLogger(__name__)

    logger.info("Loading HotpotQA distractor split...")
    ds = load_dataset("hotpot_qa", "distractor")
    val_ds = ds["validation"]
    logger.info("Validation size: %d", len(val_ds))

    LLAMA32_ID = "meta-llama/Llama-3.2-3B"
    HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    logger.info("Loading Llama-3.2-3B: %s", LLAMA32_ID)
    tok = AutoTokenizer.from_pretrained(LLAMA32_ID, trust_remote_code=True, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA32_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    model.eval()
    model.config.use_cache = True

    N = 20
    val_subset = [val_ds[i] for i in range(N)]

    # ---------- Answer-only ----------
    logger.info("Running Answer-only probe on %d validation samples...", N)
    ans_em_sum = 0.0
    ans_f1_sum = 0.0

    with torch.inference_mode():
        for idx, ex in enumerate(val_subset):
            context_text = normalize_context(ex)
            prompt_str = build_answer_only_prompt(ex, context_text)
            enc = tok(prompt_str, return_tensors="pt").to(model.device)

            out = model.generate(
                input_ids=enc["input_ids"],
                max_new_tokens=96,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )

            full_text = tok.decode(out[0], skip_special_tokens=True)
            pred_ans = extract_answer_from_text(full_text)
            gold_ans = str(ex["answer"]).strip()

            em, f1 = answer_em_f1(pred_ans, gold_ans)
            ans_em_sum += em
            ans_f1_sum += f1

            logger.info(
                "[Answer-only][%02d] Q=%s | Gold=%s | Pred=%s | EM=%.3f F1=%.3f",
                idx, ex["question"], gold_ans, pred_ans, em, f1,
            )

    logger.info(
        "Answer-only aggregate over %d: EM=%.3f F1=%.3f",
        N, ans_em_sum / N, ans_f1_sum / N,
    )

    # ---------- Sup+Answer ----------
    logger.info("Running Sup+Answer probe on %d validation samples...", N)
    sup_em_sum = 0.0
    sup_f1_sum = 0.0
    ans2_em_sum = 0.0
    ans2_f1_sum = 0.0

    with torch.inference_mode():
        for idx, ex in enumerate(val_subset):
            context_text = normalize_context(ex)
            prompt_str = build_sup_ans_prompt(ex, context_text)
            enc = tok(prompt_str, return_tensors="pt").to(model.device)

            out = model.generate(
                input_ids=enc["input_ids"],
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )

            full_text = tok.decode(out[0], skip_special_tokens=True)

            pred_sfs = parse_supporting_facts(full_text)
            gold_sfs = normalize_gold_supporting_facts(ex)

            sup_em_val, sup_f1_val = sup_em_f1(pred_sfs, gold_sfs)
            sup_em_sum += sup_em_val
            sup_f1_sum += sup_f1_val

            pred_ans2 = extract_answer_from_text(full_text)
            gold_ans2 = str(ex["answer"]).strip()
            em2, f12 = answer_em_f1(pred_ans2, gold_ans2)
            ans2_em_sum += em2
            ans2_f1_sum += f12

            logger.info(
                "[Sup+Ans][%02d] Q=%s | GoldAns=%s | PredAns=%s | "
                "GoldSup=%s | PredSup=%s | AnsEM=%.3f AnsF1=%.3f SupEM=%.3f SupF1=%.3f",
                idx, ex["question"], gold_ans2, pred_ans2,
                gold_sfs, pred_sfs, em2, f12, sup_em_val, sup_f1_val,
            )

    logger.info(
        "Sup+Answer aggregate over %d: AnsEM=%.3f AnsF1=%.3f SupEM=%.3f SupF1=%.3f",
        N, ans2_em_sum / N, ans2_f1_sum / N, sup_em_sum / N, sup_f1_sum / N,
    )

    # cleanup
    del model
    torch.cuda.empty_cache()


def main() -> None:
    setup_logging()
    run_probes()


if __name__ == "__main__":
    main()


