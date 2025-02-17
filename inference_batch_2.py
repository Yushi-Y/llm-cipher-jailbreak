import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import os
from template_generation import instruction_prompts  

#### TO DO: BATCH PROCESSING ROWS ####

def load_model(model_name):
    """Loads the model and tokenizer once."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model


def translate_with_llm(input_text, tokenizer, model, model_name, translation_type):
    """Translates text using an LLM without reloading the model each time."""
    model_device = next(model.parameters()).device

    if not input_text or pd.isna(input_text):  # Skip empty rows
        return ""

    is_gemma = "gemma" in model_name.lower()

    if is_gemma:
        # Combine system and user instructions into one user message
        combined_prompt = instruction_prompts[translation_type]["system"] + "\n\n" + \
                        instruction_prompts[translation_type]["user"].format(input_text=input_text)
        messages = [{"role": "user", "content": combined_prompt}]
    else:
        # Use system and user as separate messages for non-Gemma models
        messages = [
            {"role": "system", "content": instruction_prompts[translation_type]["system"]},
            {"role": "user", "content": instruction_prompts[translation_type]["user"].format(input_text=input_text)}
        ]

    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model_device)

    outputs = model.generate(tokenized_chat, max_new_tokens=300, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_translation(expected_output, translated_text):
    """Evaluates the correctness of the translation based on word-matching rate."""
    expected_words = expected_output.strip().lower().split()
    
    # Locate the last occurrence of "the translation is" (case insensitive)
    match = re.search(r"(?i)(.*)\bthe translation is\b", translated_text)

    if match:
        extracted_text = translated_text[match.end():].strip().lower()
    else:
        extracted_text = translated_text.strip().lower()

    translated_words = extracted_text.split()

    # Compute word-matching count
    matched_words = sum(1 for word in expected_words if word in translated_words)
    word_matching_rate = matched_words / len(expected_words) if expected_words else 0

    return word_matching_rate


# **Load the CSV file**
script_dir = os.path.dirname(os.path.abspath(__file__))

input_csv = os.path.join(script_dir, "ciphered_harmful_prompts.csv")
output_csv = os.path.join(script_dir, "translated_harmful_prompts_llama3.csv")

df = pd.read_csv(input_csv)

# **Load the model once**
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer, model = load_model(model_name)

# Accuracy lists
plain_to_caesar_accuracies = []
plain_to_pig_latin_accuracies = []
caesar_to_plain_accuracies = []
pig_latin_to_plain_accuracies = []


def process_row(row, index):
    """Processes each row: translates, evaluates, and stores results."""
    original_text = str(row["harmless_prompt"])
    expected_caesar = str(row["harmless_prompt_caesar"])
    expected_pig_latin = str(row["harmless_prompt_pig_latin"])

    # Translate using the preloaded model (plain → cipher)
    plain_to_caesar_translation = translate_with_llm(original_text, tokenizer, model, model_name, "plain_to_caesar")
    plain_to_pig_latin_translation = translate_with_llm(original_text, tokenizer, model, model_name, "plain_to_pig_latin")

    # Translate back using the preloaded model (cipher → plain)
    caesar_to_plain_translation = translate_with_llm(expected_caesar, tokenizer, model, model_name, "caesar_to_plain")
    pig_latin_to_plain_translation = translate_with_llm(expected_pig_latin, tokenizer, model, model_name, "pig_latin_to_plain")

    # Evaluate accuracy (plain → cipher)
    plain_to_caesar_accuracy = evaluate_translation(expected_caesar, plain_to_caesar_translation)
    plain_to_pig_latin_accuracy = evaluate_translation(expected_pig_latin, plain_to_pig_latin_translation)

    # Evaluate accuracy (cipher → plain)
    caesar_to_plain_accuracy = evaluate_translation(original_text, caesar_to_plain_translation)
    pig_latin_to_plain_accuracy = evaluate_translation(original_text, pig_latin_to_plain_translation)

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
