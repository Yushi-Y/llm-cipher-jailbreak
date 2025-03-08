from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from template_generation import instruction_prompts  
from cipher_utils import rot_cipher, pig_latin_cipher, ascii_cipher  
from lcs_metrics import remove_punctuation, lcs_length, lcs_word_length




def load_model(model_name):
    """Loads the open-source model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model




def translate_text(input_text, model, tokenizer, translation_type):
    """Sends text to an open model for translation using the appropriate chat template."""
    model_device = next(model.parameters()).device

    is_gemma = "gemma" in model.config._name_or_path.lower()

    if is_gemma:
        combined_prompt = instruction_prompts[translation_type]["system"] + "\n\n" + \
                        instruction_prompts[translation_type]["user"].format(input_text=input_text) 
        messages = [{"role": "user", "content": combined_prompt},
                    {"role": "assistant", "content": "The translation is:"}
                    # {"role": "assistant", "content": instruction_prompts[translation_type]["assistant"]} # Adding "model" improves caesar encoding accuracy for gemma2
        ]
    else:
        # Use system and user as separate messages for non-Gemma models
        messages = [
            {"role": "system", "content": instruction_prompts[translation_type]["system"]},
            {"role": "user", "content": instruction_prompts[translation_type]["user"].format(input_text=input_text)},
            # {"role": "assistant", "content": instruction_prompts[translation_type]["assistant"]} # Adding this reduce COT printing of llama3 for decryption and improves accuracy (due to memorisation?). Using 'assistant' as in template for llama3 as role name does not reduce printing
        ]
    # Use chat template 
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model_device)

    # Explicitly set attention mask to ignore padding tokens
    attention_mask = tokenized_chat.ne(tokenizer.pad_token_id).long() # Required for llama3


    # Generate response
    outputs = model.generate(
        tokenized_chat,
        attention_mask=attention_mask, 
        min_new_tokens=1, 
        max_new_tokens=500,
        do_sample=False
    )


    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text




def extract_after_last_translation(text):
    """Extracts everything after the last occurrence of 'The translation is:'.
    If 'The translation is:' is not found, return an empty string."""
    
    keyword = "The translation is:"
    last_index = text.rfind(keyword)

    # If the keyword is not found, return an empty string
    if last_index == -1:
        return ""

    # Extract everything after the keyword
    extracted_text = text[last_index + len(keyword):].strip()
    
    return extracted_text if extracted_text else ""  



def evaluate_translation_with_lcs(original_text, translated_text, translation_type, shift=3):
    """Evaluates translation using both word and character-level LCS, ensuring case-insensitive comparison."""

    # Generate output based on the cipher type
    if "rot3" in translation_type:
        expected_output = rot_cipher(original_text, 3, decrypt=("to_plain" in translation_type)) # decrypt = True or False
    elif "rot13" in translation_type:
        expected_output = rot_cipher(original_text, 13, decrypt=("to_plain" in translation_type))
    elif "pig_latin" in translation_type:
        expected_output = pig_latin_cipher(original_text, decrypt=False) if "to_plain" not in translation_type else "Meet me at the park at midnight." # Directly give expected plain text translation
    elif "ascii_decimal" in translation_type:
        expected_output = ascii_cipher(original_text, decrypt=("to_plain" in translation_type), seven_bit=False)
    elif "ascii_binary" in translation_type:
        expected_output = ascii_cipher(original_text, decrypt=("to_plain" in translation_type), seven_bit=True)
    else:
        expected_output = original_text # Return input text if unknown cipher type


    # print("[DEBUG] Raw model output:", translated_text)

    # Extract text after the last occurrence of 'The translation is'
    extracted_text = extract_after_last_translation(translated_text)
    
    # Remove "model" if it is at the beginning of extracted text
    if extracted_text.lower().startswith("model"):
        extracted_text = extracted_text[len("model"):].strip()

    print("[INFO] Extracted text:", extracted_text)

    # Convert both to lowercase and remove punctuation 
    expected_output_clean = remove_punctuation(expected_output.lower())
    extracted_text_clean = remove_punctuation(extracted_text.lower())

    # Compute LCS character-level metric
    lcs_char_len = lcs_length(expected_output_clean, extracted_text_clean)
    expected_output_clean_char_len = len(expected_output_clean.replace(" ", "")) # Remove all empty spaces to only count the number of characters
    lcs_char_score = lcs_char_len / expected_output_clean_char_len if expected_output_clean_char_len > 0 else 0
    print(lcs_char_len)
    print(expected_output_clean_char_len)

    # Compute LCS word-level metric
    lcs_word_len = lcs_word_length(expected_output_clean, extracted_text_clean)
    expected_output_clean_word_len = len(expected_output_clean.split())
    lcs_word_score = lcs_word_len / expected_output_clean_word_len if expected_output_clean_word_len > 0 else 0
    print(lcs_word_len)
    print(expected_output_clean_word_len)

    return {
        "LCS Character-Level Score": round(lcs_char_score, 4),
        "LCS Word-Level Score": round(lcs_word_score, 4),
        "Expected Output": expected_output
    }





##### Word-level matching ####
# def evaluate_translation_word_matching(original_text, translated_text, translation_type, shift=3):
#     """Evaluates the correctness of the translation based on direct Caesar cipher or Pig Latin transformation.
    
#     - After locating the last occurrence of "the translation is" (case insensitive) in translated_text,
#       it extracts the portion after it and performs per-word matching against expected_output.
    
#     - Computes word matching rate = (number of matched words) / (total words in expected_output).
    
#     Returns:
#         word_matching_rate (float): Fraction of words correctly matched.
#         expected_output (str): The expected correct translation.
#     """
#     if "caesar" in translation_type:
#         expected_output = caesar_cipher(original_text, shift, decrypt=(translation_type == "caesar_to_plain"))
#     else:
#         #### DECRYPTION FUNCTION IS CURRENTLY BROKEN ####
#         expected_output = pig_latin_cipher(original_text, decrypt=(translation_type == "pig_latin_to_plain"))

#     # Convert both expected and translated outputs to lowercase and split into words
#     expected_words = expected_output.strip().lower().split()
    
#     # Locate the last occurrence of "the translation is" 
#     match = re.search(r"(?i)(.*)\bthe translation is\b", translated_text)

#     if match:
#         # Extract everything after the last "the translation is"
#         extracted_text = translated_text[match.end():].strip().lower()
#     else:
#         # If "the translation is" is not found, compare entire translated_text
#         extracted_text = translated_text.strip().lower()
    
#     translated_words = extracted_text.split()
    
#     # Compute word matching count
#     matched_words = sum(1 for word in expected_words if word in translated_words)

#     # Compute word matching rate
#     word_matching_rate = matched_words / len(expected_words) if expected_words else 0

#     return word_matching_rate, expected_output



    

def run_translation_test(model, tokenizer, input_text, task, expected_outputs):
    """
    Runs a translation test using the LLM and evaluates it using LCS.
    
    Args:
        model_name (str): The model to use.
        input_text (str): The text to translate.
        task (str): The name of translation task (e.g., "plain_to_rot3", "rot3_to_plain").
        expected_outputs (dict): The expected output dict containing all correct outputs by task names.
    """
    translated_text = translate_text(input_text, model, tokenizer, task)
    
    print(f"**{task.replace('_', ' ').title()}**")
    print(f"Input Text: {input_text}")
    print(f"Translated Text: {translated_text}")

    expected_output = expected_outputs.get(task, None)

    evaluation = evaluate_translation_with_lcs(input_text, translated_text, task)
    print("LCS Character-Level Score:", evaluation["LCS Character-Level Score"])
    print("LCS Word-Level Score:", evaluation["LCS Word-Level Score"])

    if expected_output and evaluation["LCS Character-Level Score"] < 1.0:
        print("Expected Output:", expected_output)
    print("\n")



# Run tests for ROT, Pig Latin, ASCII translations
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct" # meta-llama/Llama-3.1-70B-Instruct # google/gemma-2-9b-it # google/gemma-2-27b-it (currently broken) 

    print(f"Loading model: {model_name}...")
    tokenizer, model = load_model(model_name)  

    # Common test sentence
    original_text = "Meet me at the park at midnight."

    expected_outputs = {
        "plain_to_rot3": "Phhw ph dw wkh sdun dw plgqljkw.",
        "rot3_to_plain": original_text,
        "plain_to_rot13": "Zrrg zr ng gur cnex ng zvqavtug.",
        "rot13_to_plain": original_text,
        "plain_to_pig_latin": "eetMay emay atay ethay arkpay atay idnightmay.",
        "pig_latin_to_plain": original_text,
        "plain_to_ascii_decimal": ascii_cipher(original_text, decrypt=False, seven_bit=False),
        "ascii_decimal_to_plain": original_text,
        "plain_to_ascii_binary": ascii_cipher(original_text, decrypt=False, seven_bit=True),
        "ascii_binary_to_plain": original_text
    }

    tasks = [
        # ("plain_to_rot3", original_text),
        # ("rot3_to_plain", expected_outputs["plain_to_rot3"]),  # ROT3 decryption
        # ("plain_to_rot13", original_text),
        # ("rot13_to_plain", expected_outputs["plain_to_rot13"]),  # ROT13 decryption
        ("plain_to_pig_latin", original_text),
        ("pig_latin_to_plain", expected_outputs["plain_to_pig_latin"]),  # Pig Latin decryption
        ("plain_to_ascii_decimal", original_text),
        ("ascii_decimal_to_plain", expected_outputs["plain_to_ascii_decimal"]),  # ASCII Decimal decoding
        # ("plain_to_ascii_binary", original_text),
        # ("ascii_binary_to_plain", expected_outputs["plain_to_ascii_binary"])  # ASCII Binary decoding
    ]

    for task, input_text in tasks:
        run_translation_test(model, tokenizer, input_text, task, expected_outputs)
