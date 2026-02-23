from datasets import load_dataset, Dataset
import pandas as pd


def load_and_preprocess_csv(csv_path, max_samples=None):
    """
    Reads the CSV, preprocesses it into the instruction-input-output format,
    and returns a Hugging Face Dataset object.
    """
    print(f"Loading and preprocessing CSV from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='iso-8859-1')
        print(f"Successfully read {len(df)} rows from {csv_path}.")
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        raise
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise

    if max_samples:
        print(f"Using a subset of {min(max_samples, len(df))} samples.")
        df = df.head(min(max_samples, len(df)))

    instruction_template = (
        "Identify the cause and effect in the following sentence by adding "
        "<cause></cause> and <effect></effect> tags around the respective parts. "
        "The cause is the event or situation that leads to another. "
        "The effect is the outcome or result."
    )

    processed_examples = []
    # replace <e1> with <cause> and <e2> with <effect>
    for index, row in df.iterrows():
        
        text = str(row.get(' Text',"")).strip()
        sentence = str(row.get(' Sentence', '')).strip()
        cause_text = str(row.get(' Cause', '')).strip()
        effect_text = str(row.get(' Effect', '')).strip()

        if not sentence or not cause_text or not effect_text:
            warnings_count += 1
            skipped_rows += 1
            continue

        input_for_llm = text # Original sentence is the "input" part of the prompt
        output_for_llm = sentence.replace("e1","cause").replace("e2","effect") # Start with original sentence to insert tags

        processed_examples.append({
            "instruction": instruction_template,
            "input": input_for_llm,      # Original sentence
            "output": output_for_llm     # Sentence with <cause>/<effect> tags
        })
    
    # Convert list of dicts to Hugging Face Dataset
    formatted_dataset = Dataset.from_list(processed_examples)
    print(f"Converted CSV to Hugging Face Dataset with {len(formatted_dataset)} examples.")
    return formatted_dataset

def preprocess_function_for_finetuning(examples, tokenizer, max_length=2048):
    # When batched=True, `examples` is a dictionary where each key
    # (like "instruction", "input", "output") maps to a list of values (the batch).
    num_examples = len(examples["input"]) # Get the number of examples in the current batch
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    # Your snippet's logic is right here:
    for i in range(num_examples): # Loop through each example in the batch
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]    # This is the original sentence
        output_text = examples["output"][i]  # This is the sentence with <cause/effect> tags

        prompt_part = f"{instruction}\n\nInput: {input_text}\n\nOutput: " # LLM will complete after this
        full_text_for_model = f"{prompt_part}{output_text}{tokenizer.eos_token}"

        # And then the tokenization of this formatted string:
        tokenized_full = tokenizer(
            full_text_for_model,
            max_length=max_length,
            truncation=True,
            padding=False, # Data collator will handle padding
            add_special_tokens=True
        )
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = list(input_ids) 
        
        tokenized_prompt_section = tokenizer(prompt_part, max_length=max_length, truncation=True, add_special_tokens=True)
        prompt_length_in_full_sequence = len(tokenized_prompt_section["input_ids"])
        
        # If tokenized_prompt_section includes an EOS from its own tokenization AND it's the final token,
        # but it's NOT meant to be the end of the *prompt structure* before completion starts,
        # we might need to adjust prompt_length_in_full_sequence.
        # This is often tricky. A common case is prompt ending, then completion.
        # If BOS is added: BOS + prompt_tokens + completion_tokens + EOS
        # len(BOS + prompt_tokens) is the part to mask.
        # Example: tokenizer("Test prompt") -> [BOS, T, P, EOS]. Here prompt_length is 4.
        # tokenizer("Test prompt" + "Completion" + EOS) -> [BOS, T, P, C, EOS]
        # Labels: [-100, -100, -100, -100, ID(C), ID(EOS)]
        # The current `prompt_length_in_full_sequence` should be okay if tokenizer("prompt_part", add_special_tokens=True) correctly reflects the token count of the prefix.
        
        actual_prompt_mask_len = min(prompt_length_in_full_sequence, len(labels))
        for k in range(actual_prompt_mask_len):
            # we are masking the in the instruction + input text with -100 so that we are left with just output text token ids. These output labels are used for loss calculation. 
            if k < len(labels):
                 labels[k] = -100
        
        if all(l == -100 for l in labels) and len(labels) > 0 : # Check if all labels got masked
            # This can happen if max_length is too short and only the prompt fits (or part of it)
            # print(f"Warning: All labels masked for an example. Prompt might be too long or max_length too short. Index in batch: {i}")
            # To avoid issues with empty label sequences for loss calculation, ensure at least one non-masked label if possible,
            # or Trainer might complain. However, if input is truncated such that only prompt fits, this is expected.
            raise ValueError(f"Warning: All labels masked for an example. Prompt might be too long or max_length too short. Index in batch: {i}")
            

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
        
    return model_inputs


if __name__ == "__main__":
    data_path = "/home/yash/Finetunning/FinCausal/data/train.csv"
    data= load_and_preprocess_csv(data_path)
    #sample
    print(data[2])