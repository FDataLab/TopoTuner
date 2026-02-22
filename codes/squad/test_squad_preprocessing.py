#!/usr/bin/env python3
"""
Test script to verify SQuAD dataset preprocessing and model inference.
Loads samples, processes them, runs inference with the model,
and calculates EM and F1 scores.
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from codes.squad.data_preprocessing_squad import (
    preprocess_dataset,
    infer_prompt_format_from_model_id,
    build_prompt,
)
from codes.hotpotqa.data_preprocessing_hotpotqa import eos_for_model
from codes.squad.eval_squad import _normalize_text, _em_and_f1, _extract_final_answer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    NUM_SAMPLES = 100  # Test with 100 samples
    MAX_LEN = 1024
    MAX_NEW_TOKENS = 256
    OUTPUT_FILE = "/home/kadir/topo/logs/squad_preprocessing_test.jsonl"
    SPLIT = "validation"
    RUN_INFERENCE = True  # Set to False to skip model inference
    
    print("=" * 80)
    print("SQuAD Preprocessing Test")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Max length: {MAX_LEN}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading SQuAD dataset...")
    ds = load_dataset("squad")[SPLIT]
    ds_subset = ds.select(range(min(NUM_SAMPLES, len(ds))))
    print(f"Loaded {len(ds_subset)} samples from {SPLIT} split")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left" if RUN_INFERENCE else "right",  # Left for generation, right for training
        token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Get prompt format and EOS token
    prompt_format = infer_prompt_format_from_model_id(MODEL_NAME)
    eos_id = eos_for_model(tokenizer, MODEL_NAME)
    print(f"Prompt format: {prompt_format}")
    print(f"EOS token ID: {eos_id}")
    print()
    
    # Load model if running inference
    model = None
    if RUN_INFERENCE:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": 0},
            dtype=torch.bfloat16,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        model.eval()
        model.config.use_cache = True
        print("Model loaded and set to eval mode")
        print()
    
    # Process samples
    print("Processing samples...")
    print("-" * 80)
    
    results = []
    total_em = 0.0
    total_f1 = 0.0
    
    iterator = tqdm(ds_subset, desc="Processing") if RUN_INFERENCE else ds_subset
    
    for idx, example in enumerate(iterator):
        # Extract raw data
        question = example["question"]
        context = example.get("context", "")
        answers = example.get("answers", {}).get("text", [])
        gold_answer = (answers[0] if answers else "").strip()
        
        # Build prompt (same as in finetuning)
        prompt = build_prompt(
            tokenizer,
            question,
            context=context,
            prompt_format=prompt_format,
            use_instruction=True
        )
        
        # Build target (same as in finetuning)
        target = f"Answer: {gold_answer}".strip()
        
        # Preprocess using the same function as finetuning
        processed = preprocess_dataset(
            example,
            tokenizer,
            max_len=MAX_LEN,
            use_instruction=True,
            prompt_format=prompt_format,
            is_train=True,
        )
        
        # Decode to verify
        input_ids = processed["input_ids"]
        labels = processed["labels"]
        attention_mask = processed["attention_mask"]
        
        # Decode prompt part (where labels are -100)
        prompt_ids = [id for id, label in zip(input_ids, labels) if label == -100]
        answer_ids = [id for id, label in zip(input_ids, labels) if label != -100]
        
        decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=False)
        
        # Run inference if model is loaded
        generated_text = ""
        extracted_answer = ""
        em = 0.0
        f1 = 0.0
        
        if RUN_INFERENCE and model is not None:
            # Tokenize prompt for generation (use left padding)
            tokenizer.padding_side = "left"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode generated text (only the new tokens)
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            # Extract answer
            extracted_answer = _extract_final_answer(generated_text)
            
            # Calculate EM and F1
            em, f1 = _em_and_f1(extracted_answer, gold_answer)
            total_em += em
            total_f1 += f1
        
        # Store result
        result = {
            "sample_id": idx,
            "question": question,
            "context": context[:500] + "..." if len(context) > 500 else context,  # Truncate for readability
            "context_full_length": len(context),
            "gold_answer": gold_answer,
            "target": target,
            "prompt": prompt,
            "decoded_prompt": decoded_prompt,
            "decoded_answer": decoded_answer,
            "input_ids_length": len(input_ids),
            "labels_length": len(labels),
            "num_prompt_tokens": len(prompt_ids),
            "num_answer_tokens": len(answer_ids),
            "prompt_matches": prompt.strip() == decoded_prompt.strip(),
            "answer_matches": target.strip() == decoded_answer.strip(),
        }
        
        # Add inference results if available
        if RUN_INFERENCE and model is not None:
            result.update({
                "generated_text": generated_text,
                "extracted_answer": extracted_answer,
                "em": em,
                "f1": f1,
            })
        
        results.append(result)
        
        # Print first sample for immediate feedback
        if idx == 0:
            print("\n" + "=" * 80)
            print("SAMPLE 0 (First sample preview):")
            print("=" * 80)
            print(f"\nQuestion: {question}")
            print(f"\nContext (first 300 chars): {context[:300]}...")
            print(f"\nGold Answer: {gold_answer}")
            print(f"\nTarget: {target}")
            print(f"\n{'='*80}")
            print("PROMPT (what model sees):")
            print("="*80)
            print(prompt)
            print(f"\n{'='*80}")
            print("TOKENIZED INFO:")
            print("="*80)
            print(f"Total input_ids length: {len(input_ids)}")
            print(f"Prompt tokens: {len(prompt_ids)}")
            print(f"Answer tokens: {len(answer_ids)}")
            print(f"Labels (non -100): {len([l for l in labels if l != -100])}")
            print(f"\nDecoded prompt matches original: {result['prompt_matches']}")
            print(f"Decoded answer matches target: {result['answer_matches']}")
            print("\n" + "=" * 80)
            print()
    
    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(results)} samples to {OUTPUT_FILE}")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    print(f"Total samples processed: {len(results)}")
    print(f"Average input length: {sum(r['input_ids_length'] for r in results) / len(results):.1f} tokens")
    print(f"Average prompt tokens: {sum(r['num_prompt_tokens'] for r in results) / len(results):.1f} tokens")
    print(f"Average answer tokens: {sum(r['num_answer_tokens'] for r in results) / len(results):.1f} tokens")
    
    prompt_matches = sum(1 for r in results if r['prompt_matches'])
    answer_matches = sum(1 for r in results if r['answer_matches'])
    print(f"\nPrompt decoding matches: {prompt_matches}/{len(results)} ({prompt_matches/len(results)*100:.1f}%)")
    print(f"Answer decoding matches: {answer_matches}/{len(results)} ({answer_matches/len(results)*100:.1f}%)")
    
    # Evaluation metrics if inference was run
    if RUN_INFERENCE and model is not None:
        avg_em = (total_em / len(results)) * 100
        avg_f1 = (total_f1 / len(results)) * 100
        print("\n" + "=" * 80)
        print("Evaluation Metrics (on 100 samples):")
        print("=" * 80)
        print(f"Exact Match (EM): {avg_em:.2f}% ({total_em:.0f}/{len(results)} correct)")
        print(f"F1 Score: {avg_f1:.2f}%")
        print("=" * 80)
    
    # Check for issues
    print("\n" + "=" * 80)
    print("Validation Checks:")
    print("=" * 80)
    
    all_prompts_match = all(r['prompt_matches'] for r in results)
    all_answers_match = all(r['answer_matches'] for r in results)
    all_lengths_valid = all(r['input_ids_length'] > 0 for r in results)
    all_labels_valid = all(
        r['num_prompt_tokens'] + r['num_answer_tokens'] == r['input_ids_length']
        for r in results
    )
    
    print(f"✅ All prompts decode correctly: {all_prompts_match}")
    print(f"✅ All answers decode correctly: {all_answers_match}")
    print(f"✅ All sequences have valid length: {all_lengths_valid}")
    print(f"✅ Prompt + Answer tokens = Total tokens: {all_labels_valid}")
    
    if all_prompts_match and all_answers_match and all_lengths_valid and all_labels_valid:
        print("\n🎉 All validation checks passed! Preprocessing looks correct.")
    else:
        print("\n⚠️  Some validation checks failed. Please review the output file.")
    
    print("=" * 80)
    print(f"\nTo review all samples, check: {OUTPUT_FILE}")
    print("Each line is a JSON object with all preprocessing details.")
    if RUN_INFERENCE and model is not None:
        print("Each sample also includes: generated_text, extracted_answer, em, f1")
    print("=" * 80)

if __name__ == "__main__":
    main()
