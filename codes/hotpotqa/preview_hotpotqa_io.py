import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer

from .data_preprocessing_hotpotqa import (
    build_hotpot_context,
    create_prompt_llama3,
    create_prompt_llama2,
    create_prompt_mistral,
    create_prompt_qwen,
    create_prompt_olmo,
    infer_prompt_format_from_model_id,
    DEFAULT_SYSTEM_PROMPT,
)


def build_prompt_and_answer(example, tokenizer, model_name: str, evidence_mode: str = "full"):
    """
    Build the exact text input (prompt) and target output (answer) that we use for training.
    This mirrors the logic in preprocess_dataset, but returns text instead of token IDs.
    """
    question = example["question"]
    short_answer = str(example.get("answer", "")).strip()

    # Build context according to evidence_mode ("full" or "supporting")
    context = build_hotpot_context(example, evidence_mode=evidence_mode)

    answer_text = f"Answer: {short_answer}".strip()

    prompt_format = infer_prompt_format_from_model_id(model_name)

    if prompt_format == "llama3":
        prompt = create_prompt_llama3(
            tokenizer,
            question,
            context=context,
            use_instruction=True,
            prompt_template=DEFAULT_SYSTEM_PROMPT,
        )
    elif prompt_format == "llama2":
        prompt = create_prompt_llama2(
            question,
            context=context,
            use_instruction=True,
            prompt_template=DEFAULT_SYSTEM_PROMPT,
        )
    elif prompt_format == "mistral":
        prompt = create_prompt_mistral(
            question,
            context=context,
            use_instruction=True,
            prompt_template=DEFAULT_SYSTEM_PROMPT,
        )
    elif prompt_format == "olmo":
        prompt = create_prompt_olmo(
            question,
            context=context,
            use_instruction=True,
            prompt_template=DEFAULT_SYSTEM_PROMPT,
        )
    else:  # "qwen" or fallback
        prompt = create_prompt_qwen(
            tokenizer,
            question,
            context=context,
            use_instruction=True,
            prompt_template=DEFAULT_SYSTEM_PROMPT,
        )

    return prompt, answer_text, context


def main():
    parser = argparse.ArgumentParser(
        description="Preview HotpotQA training inputs (prompts) and outputs (answers)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Model name (only tokenizer is loaded, not the full model).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HotpotQA split to sample from (train/validation).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="distractor",
        help="HotpotQA configuration (usually 'distractor').",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to print.",
    )
    parser.add_argument(
        "--evidence-mode",
        type=str,
        default="full",
        choices=["full", "supporting"],
        help="Whether to use full context or only supporting facts.",
    )

    args = parser.parse_args()

    print(">>> Loading tokenizer:", args.model_name, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_name)

    print(f">>> Loading HotpotQA dataset: config={args.config}, split={args.split}", flush=True)
    ds = load_dataset("hotpot_qa", args.config)[args.split]

    n = min(args.num_examples, len(ds))
    print(f">>> Sampling {n} examples with evidence_mode='{args.evidence_mode}'", flush=True)
    print("=" * 80, flush=True)

    for idx in range(n):
        ex = ds[idx]
        prompt, answer_text, context = build_prompt_and_answer(
            ex, tok, args.model_name, evidence_mode=args.evidence_mode
        )

        question = ex.get("question", "")

        print(f"[{idx}] QUESTION:", question, flush=True)
        print("-" * 80, flush=True)
        print("CONTEXT (used for this example):", flush=True)
        if context:
            print(context, flush=True)
        else:
            print("(None)", flush=True)
        print("-" * 80, flush=True)
        print("PROMPT (input to the model):", flush=True)
        print(prompt, flush=True)
        print("-" * 80, flush=True)
        print("TARGET OUTPUT (training label):", flush=True)
        print(answer_text, flush=True)
        print("=" * 80, flush=True)


if __name__ == "__main__":
    main()

