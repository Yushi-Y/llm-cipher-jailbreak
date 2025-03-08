import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from template_generation import instruction_prompts  
from cipher_utils import rot_cipher, pig_latin_cipher, ascii_cipher  
from lcs_metrics import remove_punctuation, lcs_length, lcs_word_length

#### TO DO: BATCH PROCESSING ROWS ####


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





# **Load the CSV file**
script_dir = os.path.dirname(os.path.abspath(__file__))

input_csv = os.path.join(script_dir, "ciphered_harmful_prompts.csv")
output_csv = os.path.join(script_dir, "translated_harmful_prompts_gemma2.csv")

df = pd.read_csv(input_csv)

model_name = "google/gemma-2-9b-it" 
tokenizer, model = load_model(model_name)

plain_to_caesar_accuracies = []
plain_to_pig_latin_accuracies = []
caesar_to_plain_accuracies = []
pig_latin_to_plain_accuracies = []


def process_row(row, index):
    """Processes each row: translates, evaluates, and stores results."""
    original_text = str(row["harmless_prompt"])
    expected_caesar = str(row["harmless_prompt_caesar"])
    expected_pig_latin = str(row["harmless_prompt_pig_latin"])

    # Translate (plain → cipher)
    plain_to_caesar_translation = translate_with_llm(original_text, tokenizer, model, model_name, "plain_to_caesar")
    plain_to_pig_latin_translation = translate_with_llm(original_text, tokenizer, model, model_name, "plain_to_pig_latin")

    # Translate back (cipher → plain)
    caesar_to_plain_translation = translate_with_llm(expected_caesar, tokenizer, model, model_name, "caesar_to_plain")
    pig_latin_to_plain_translation = translate_with_llm(expected_pig_latin, tokenizer, model, model_name, "pig_latin_to_plain")

    # Evaluate accuracy (plain → cipher)
    plain_to_caesar_accuracy = evaluate_translation_with_lcs(expected_caesar, plain_to_caesar_translation)
    plain_to_pig_latin_accuracy = evaluate_translation_with_lcs(expected_pig_latin, plain_to_pig_latin_translation)

    # Evaluate accuracy (cipher → plain)
    caesar_to_plain_accuracy = evaluate_translation_with_lcs(original_text, caesar_to_plain_translation)
    pig_latin_to_plain_accuracy = evaluate_translation_with_lcs(original_text, pig_latin_to_plain_translation)

    # Store accuracy scores
    plain_to_caesar_accuracies.append(plain_to_caesar_accuracy)
    plain_to_pig_latin_accuracies.append(plain_to_pig_latin_accuracy)
    caesar_to_plain_accuracies.append(caesar_to_plain_accuracy)
    pig_latin_to_plain_accuracies.append(pig_latin_to_plain_accuracy)

    print(f"Processed row {index+1}/{len(df)}")

    return pd.Series([
        plain_to_caesar_translation, plain_to_pig_latin_translation, 
        caesar_to_plain_translation, pig_latin_to_plain_translation, 
        plain_to_caesar_accuracy, plain_to_pig_latin_accuracy, 
        caesar_to_plain_accuracy, pig_latin_to_plain_accuracy
    ])

df[[
    "plain_to_caesar_translation", "plain_to_pig_latin_translation", 
    "caesar_to_plain_translation", "pig_latin_to_plain_translation",
    "plain_to_caesar_accuracy", "plain_to_pig_latin_accuracy", 
    "caesar_to_plain_accuracy", "pig_latin_to_plain_accuracy"
]] = df.apply(lambda row: process_row(row, df.index.get_loc(row.name)), axis=1)


# **Compute overall accuracy**
average_plain_to_caesar_accuracy = sum(plain_to_caesar_accuracies) / len(plain_to_caesar_accuracies)
average_plain_to_pig_latin_accuracy = sum(plain_to_pig_latin_accuracies) / len(plain_to_pig_latin_accuracies)
average_caesar_to_plain_accuracy = sum(caesar_to_plain_accuracies) / len(caesar_to_plain_accuracies)
average_pig_latin_to_plain_accuracy = sum(pig_latin_to_plain_accuracies) / len(pig_latin_to_plain_accuracies)

# **Save the modified CSV**
df.to_csv(output_csv, index=False)
print(f"Translation and evaluation completed. Saved to {output_csv}")
print(f"Average Plain → Caesar Cipher Accuracy: {average_plain_to_caesar_accuracy:.4f}")
print(f"Average Plain → Pig Latin Accuracy: {average_plain_to_pig_latin_accuracy:.4f}")
print(f"Average Caesar → Plain Accuracy: {average_caesar_to_plain_accuracy:.4f}")
print(f"Average Pig Latin → Plain Accuracy: {average_pig_latin_to_plain_accuracy:.4f}")
