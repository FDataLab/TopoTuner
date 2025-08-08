DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def create_prompt_llama2(question: str,
                         use_instruction: bool = True,
                         prompt_template: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Llama 2 chat format:
    <s>[INST] <<SYS>>
    {system}
    <</SYS>>
    
    {user} [/INST]
    """
    if use_instruction:
        return (
            "<s>[INST] <<SYS>>\n"
            f"{prompt_template}\n"
            "<</SYS>>\n\n"
            f"{question} [/INST]"
        )
    else:
        return f"<s>[INST] {question} [/INST]"


def create_prompt_llama3(question: str,
                         use_instruction: bool = True,
                         prompt_template: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Llama 3 / 3.1 chat format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    (Model should generate the assistant content next.)
    """
    if use_instruction:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{prompt_template}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{question}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    else:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{question}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )


MODEL_IDS = {
    # Llama 2 7B
    "llama2_7b_base": "meta-llama/Llama-2-7b-hf",
    "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",

    # Llama 3 8B
    "llama3_8b_base": "meta-llama/Meta-Llama-3-8B",
    "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",

    # Llama 3.1 8B
    "llama3_1_8b_base": "meta-llama/Meta-Llama-3.1-8B",
    "llama3_1_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}