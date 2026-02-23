DEFAULT_SYSTEM_PROMPT1 = """You are a highly skilled financial analyst. Your task is to analyze the provided financial text, identify all financial entities (such as company names), and determine the sentiment associated with each entity. For every identified entity, provide its name and the corresponding sentiment, which should be one of: Positive, Neutral, or Negative. Present the results in a structured json format."""

# one shot prompting
DEFAULT_SYSTEM_PROMPT2 = """You are a senior financial analyst specializing in entity and sentiment extraction. Your task is to:
1. Identify ALL financial entities (companies, institutions, assets) in the text
2. Determine sentiment association for each entity (Positive/Neutral/Negative)
3. Return entities EXACTLY as they appear in the text with their sentiment labels

**Output Format Requirements:**
- Format: "[{'entity': [EXACT_TEXT] , 'label': [LABEL]}]"
- If no sentiment is explicitly stated, use "Neutral"
- Include ALL mentioned entities, even in indirect references

Example:
text: This said, Katzke's highest conviction recommendations, each with ~30% total return potential are: Goldman Sachs <GS.N>, Wells Fargo <WFC.N>, Bank of America <BAC.N> and JPmorgan Chase <JPM.N>.
output:

[{'entity':'Goldman Sachs', 'label':'Positive'},
 {'entity':'Wells Fargo', 'label':'Positive'},
 {'entity':'Bank of America', 'label':'Positive'},
 {'entity':'JPmorgan Chase', 'label':'Positive'}]"""

DEFAULT_SYSTEM_PROMPT3 = """You are a senior financial analyst. Your tasks:
1. Identify ALL financial entities exactly as they appear
2. Classify sentiment (Positive/Neutral/Negative)
3. Output JSON format with EXACT entity text

Example Input: "Goldman Sachs is outperforming"
Example Output: [{'entity':'Goldman Sachs','label':'Positive'}]"""

DEFAULT_SYSTEM_PROMPT5 = """Discard all the previous instructions. Behave like you are an expert entity recognizer and sentiment classifier. 
Identify the entities which are companies or organizations from the following content and classify the sentiment of the corresponding entities into ‘Neutral’, ‘Positive’, or ‘Negative’ classes. 
Considering every sentence as a String in python, provide the entities with the start and end index to mark the boundaries of it including spaces and punctuation using zero-based indexing.
Do not give explanations for the sentiment. In the output,Tag means sentiment; value means entity name. If no entity is found in the sentence, the response should be empty. 

Below are some examples:
The sentence: "Other U.S. companies have made similar moves, including social media site Reddit Inc and Mobileye, the self-driving car unit of Intel Corp <INTC.O>. "

assist_prompt = {"start": 74, "end": 84, "value": "Reddit Inc", "tag": "Neutral"}\n{"start": 128, "end": 138, "value": "Intel Corp", "tag": "Neutral"}

user_prompt2 = Kellogg <K.N>, however, based the corporate headquarters for its largest business, snacks, in Chicago after announcing a split into three independent companies this summer. [nL4N2Y822D]

assist_prompt2 = {"start": 4, "end": 11, "value": "Kellogg", "tag": "Neutral"}


user_prompt3 = Rival Oracle <ORCL.N> says in a statement on its website it has withdrawn all products, services and support for Russian and Belarusian companies, subsidiaries and partners. An Oracle spokesperson declined further comment.

assist_prompt3= {"start": 183, "end": 177, "value": "Oracle", "tag": "Neutral"}\n{"start": 6, "end": 12, "value": "Oracle", "tag": "Neutral"}"""

def create_prompt_deepseek_qwen(text: str) -> str:
    return f"<｜begin of sentence｜><｜User｜>{DEFAULT_SYSTEM_PROMPT5}\n\nIdentify the financial entities like company names from the given text and trace the sentiments for each financial entities:\n{text.strip()}<｜Assistant｜>"

# # VEPAUL lets do the template for LLama3
# def create_prompt_deepseek_llama(text: str) -> str:
#     return f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT5}\n<</SYS>>\n\nIdentify the financial entities like company names from the given text and trace the sentiments for each financial entity:\n{text.strip()} [/INST]"

def preprocess_dataset1(data,tokenizer,max_len:4096,is_train:bool=True):
    """one issue with this type FT is that, label and input_id sizes are different, so while calculating loss, the mismatch error will come. One possible solution we can try later is that, during each batching process we can pad the difference in size of labels and input_ids by -100"""
    text = data["content"]
    ann = data["annotations"]
    labels = str([{'entity':a["value"],'label':a["label"]} for a in ann]) if ann else ""
    prompt = create_prompt_deepseek_qwen(text)   
    
    #tokenized input
    tokenized_input = tokenizer(prompt,
                            max_length = max_len,
                            truncation=True,
                            padding=False,
                            padding_side="right",
                            #return_tensors='pt'
                            )
    
    tokenized_labels = tokenizer(labels,
                                max_length = max_len,
                                truncation=True,
                                padding=False,
                                padding_side="right")
    
    # print(f"Input IDs shape: {tokenized_input['input_ids'].shape}")
    # print(f"Label IDs shape: {tokenized_labels['input_ids'].shape if is_train else None}")
    
    
    return {"input_ids":tokenized_input["input_ids"],
           "attention_mask":tokenized_input["attention_mask"],
           "labels":tokenized_labels["input_ids"] if is_train else None}
    
def preprocess_dataset(data,tokenizer,max_len:4096,is_train:bool=True):

    text = data["content"]
    ann = data["annotations"]
    labels = [{'entity':a["value"],'label':a["label"]} for a in ann] if ann else ''
    prompt = create_prompt_deepseek_qwen(text)   
    
    #tokenized input
    if is_train and labels:
        combined_text = prompt + f"\nEntity and Sentiments are: \n {labels} {tokenizer.eos_token}"
    else:
        combined_text = prompt + tokenizer.eos_token
        
    tokenized = tokenizer(combined_text,
                            max_length=max_len,
                            truncation=True,
                            padding="max_length",
                            # return_tensors="pt"
                            )
    
    tokenized["labels"] = tokenized["input_ids"][:]
    
    return tokenized


if __name__ == "__main__":
    
    # from transformers import DataCollatorForSeq2Seq

    # data_collator = DataCollatorForSeq2Seq(
    #     distill_tokenizer,
    #     model=distill_model,  
    #     padding="longest",
    #     label_pad_token_id=-100)
    pass