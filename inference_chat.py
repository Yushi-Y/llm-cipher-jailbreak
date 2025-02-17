from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from template_generation import instruction_prompts  
from cipher_utils import caesar_cipher, pig_latin_cipher  
import re

def load_model(model_name):
    """Loads the open-source model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model


def translate_with_llm(input_text, model_name, translation_type):
    """Sends text to an open-source model for translation using the appropriate chat template."""
    tokenizer, model = load_model(model_name)
    model_device = next(model.parameters()).device

    # Check if the model supports chat templates
    has_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    is_gemma = "gemma" in model_name.lower()  # Detect Gemma models

    if is_gemma:
        # Gemma does NOT support system messages, only user-assistant alternation
        messages = [{"role": "user", "content": instruction_prompts[translation_type]["user"].format(input_text=input_text)}]
    else:
        # LLaMA and other chat-based models support system messages
        messages = [
            {"role": "system", "content": instruction_prompts[translation_type]["system"]},
            {"role": "user", "content": instruction_prompts[translation_type]["user"].format(input_text=input_text)}
        ]

    # Use chat template if available
    if has_chat_template:
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Ensures proper turn alternation for Gemma
            return_tensors="pt",
            padding=True
        ).to(model_device)
    else:
        # Fallback for models without chat templates (standard prompt)
        formatted_prompt = instruction_prompts[translation_type]["user"].format(input_text=input_text)
        tokenized_chat = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )["input_ids"].to(model_device)

    # Generate response
    outputs = model.generate(
        tokenized_chat,
        max_new_tokens=300,
        do_sample=False
    )

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text




def evaluate_translation(original_text, translated_text, translation_type, shift=3):
    """Evaluates the correctness of the translation based on direct Caesar cipher or Pig Latin transformation.
    
    - After locating the last occurrence of "the translation is" (case insensitive) in translated_text,
      it extracts the portion after it and performs per-word matching against expected_output.
    
    - Computes word matching rate = (number of matched words) / (total words in expected_output).
    
    Returns:
        word_matching_rate (float): Fraction of words correctly matched.
        expected_output (str): The expected correct translation.
    """
    if "caesar" in translation_type:
        expected_output = caesar_cipher(original_text, shift, decrypt=(translation_type == "caesar_to_plain"))
    else:
        #### DECRYPTION FUNCTION IS CURRENTLY BROKEN ####
        expected_output = pig_latin_cipher(original_text, decrypt=(translation_type == "pig_latin_to_plain"))

    # Convert both expected and translated outputs to lowercase and split into words
    expected_words = expected_output.strip().lower().split()
    
    # Locate the last occurrence of "the translation is" (case insensitive)
    match = re.search(r"(?i)(.*)\bthe translation is\b", translated_text)

    if match:
        # Extract everything after the last "the translation is"
        extracted_text = translated_text[match.end():].strip().lower()
    else:
        # If "the translation is" is not found, compare entire translated_text
        extracted_text = translated_text.strip().lower()
    
    translated_words = extracted_text.split()
    
    # Compute word matching count
    matched_words = sum(1 for word in expected_words if word in translated_words)

    # Compute word matching rate
    word_matching_rate = matched_words / len(expected_words) if expected_words else 0

    return word_matching_rate, expected_output



# def evaluate_translation(original_text, translated_text, translation_type, shift=3):
#     """Evaluates the correctness of the translation based on direct Caesar cipher or Pig Latin transformation."""
#     if "caesar" in translation_type:
#         expected_output = caesar_cipher(original_text, shift, decrypt=(translation_type == "caesar_to_plain"))
#     else:
#         #### DECRYPTION FUNCTION IS CURRENTLY BROKEN ####
#         expected_output = pig_latin_cipher(original_text, decrypt=(translation_type == "pig_latin_to_plain"))

#     # Convert to lowercase to ignore case differences
#     is_correct = expected_output.strip().lower() in translated_text.strip().lower()
    
#     return is_correct, expected_output



if __name__ == "__main__":
    model_name = "google/gemma-2-9b-it" # "meta-llama/Llama-3.1-8B-Instruct"  # Use IT model 

    # **Example 1: Translate natural language to Caesar Cipher**
    original_text = "Meet me at the park at midnight."
    ciphered_text = translate_with_llm(original_text, model_name, "plain_to_caesar")
    print("Original Input:", original_text)
    print("Caesar Cipher Output:", ciphered_text)
    correct, expected = evaluate_translation(original_text, ciphered_text, "plain_to_caesar")
    print("Correct Translation:", correct)
    if not correct:
        print("Expected Output:", expected)

    # **Example 2: Translate Caesar Cipher back to natural language**
    caesar_text = "PHHW PH DW WKH SDUN DW PLGQLJKW."
    decoded_text = translate_with_llm(caesar_text, model_name, "caesar_to_plain")
    print("Ciphered Input:", caesar_text)
    print("Decoded Output:", decoded_text)
    correct, expected = evaluate_translation(caesar_text, decoded_text, "caesar_to_plain")
    print("Correct Translation:", correct)
    if not correct:
        print("Expected Output:", expected)

    # **Example 3: Translate natural language to Pig Latin**
    original_pig_text = "Hello world this is Pig Latin"
    pig_latin_text = translate_with_llm(original_pig_text, model_name, "plain_to_pig_latin")
    print("Original Input:", original_pig_text)
    print("Pig Latin Output:", pig_latin_text)
    correct, expected = evaluate_translation(original_pig_text, pig_latin_text, "plain_to_pig_latin")
    print("Correct Translation:", correct)
    if not correct:
        print("Expected Output:", expected)

    # **Example 4: Translate Pig Latin back to natural language**
    pig_latin_input = "Ellohay orldway isthay isay Igpay Atinlay"
    plain_text = translate_with_llm(pig_latin_input, model_name, "pig_latin_to_plain")
    print("Pig Latin Input:", pig_latin_input)
    print("Decoded Output:", plain_text)
    correct, expected = evaluate_translation(pig_latin_input, plain_text, "pig_latin_to_plain")
    print("Correct Translation:", correct)
    if not correct:
        print("Expected Output:", expected)
