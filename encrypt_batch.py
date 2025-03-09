import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from encrypt_decrypt_prompts import encrypt_decrypt_prompts
from lcs_metrics import remove_punctuation, lcs_length, lcs_word_length


def load_model(model_name):
    """Loads the open-source model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model


def batch_translate_text(input_texts, model, tokenizer, prompt_type, is_gemma):
    """Batch process text for translation."""
    model_device = next(model.parameters()).device

    if prompt_type not in encrypt_decrypt_prompts:
        print(f"[ERROR] Invalid prompt type: {prompt_type}")
        return [""] * len(input_texts)

    # Prepare batch messages
    messages = []
    for text in input_texts:
        if is_gemma:  # Use combined prompts for Gemma models
            combined_prompt = encrypt_decrypt_prompts[prompt_type]["system"] + "\n\n" + \
                            encrypt_decrypt_prompts[prompt_type]["user"].format(input_text=text)
            messages.append([
                {"role": "user", "content": combined_prompt},
                {"role": "assistant", "content": ""}  # Ensure empty assistant message for valid response
            ])
        else:
            messages.append([
                {"role": "system", "content": encrypt_decrypt_prompts[prompt_type]["system"]},
                {"role": "user", "content": encrypt_decrypt_prompts[prompt_type]["user"].format(input_text=text)},
                {"role": "assistant", "content": ""}  
            ])


    # Tokenize batch
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to(model_device)

    # Explicitly set attention mask to ignore padding tokens
    attention_mask = tokenized_chat.ne(tokenizer.pad_token_id).long() # Required for llama3

    # Generate translations in batch
    outputs = model.generate(
        tokenized_chat,
        attention_mask=attention_mask, 
        # min_new_tokens=1,
        max_new_tokens=500,
        do_sample=False
    )

    translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract only the text after the LAST occurrence of "The translation is:"
    extracted_texts = []
    for translated_text in translated_texts:
        last_index = translated_text.rfind("The translation is:")
        extracted_text = translated_text[last_index + len("The translation is:"):].strip() if last_index != -1 else translated_text.strip() # If not found return all translated text

        # Remove "model" if it appears at the beginning
        if extracted_text.lower().startswith("model"):
            extracted_text = extracted_text[len("model"):].strip()

        extracted_texts.append(extracted_text)

    return extracted_texts


def evaluate_lcs_batch(ground_truths, generated_texts):
    """Computes LCS character-level and word-level scores for a batch."""
    results = []
    for ground_truth, generated_text in zip(ground_truths, generated_texts):
        ground_truth_clean = remove_punctuation(ground_truth.lower())
        generated_text_clean = remove_punctuation(generated_text.lower())

        # Compute LCS character-level metric
        lcs_char_len = lcs_length(ground_truth_clean, generated_text_clean)
        ground_truth_char_len = len(ground_truth_clean.replace(" ", ""))
        lcs_char_score = lcs_char_len / ground_truth_char_len if ground_truth_char_len > 0 else 0

        # Compute LCS word-level metric
        lcs_word_len = lcs_word_length(ground_truth_clean, generated_text_clean)
        ground_truth_word_len = len(ground_truth_clean.split())
        lcs_word_score = lcs_word_len / ground_truth_word_len if ground_truth_word_len > 0 else 0

        results.append((round(lcs_char_score, 4), round(lcs_word_score, 4)))

    return results



#### Process CSV in Batches ####
input_file = "/data/kebl6672/llm_jailbreak/harmful_prompts/ciphered_harmful_prompts.csv"
output_file = "/data/kebl6672/llm_jailbreak/translated_results/encrypt_cipher_gemma2_new.csv"

model_name = "google/gemma-2-9b-it"
print(f"Loading model: {model_name}...")
tokenizer, model = load_model(model_name)

is_gemma = "gemma" in model_name.lower()

# Cipher types
cipher_list = ["rot3", "rot13", "pig_latin", "ascii_decimal", "ascii_7bit"]
batch_size = 50  # Process batches of rows

# Read input csv in chunks 
df_chunks = pd.read_csv(input_file, chunksize=batch_size)

# If output file exists, delete to prevent appending to old results
# if os.path.exists(output_file):
#     os.remove(output_file)

for chunk_idx, df_chunk in enumerate(df_chunks):
    print(f"Processing batch {chunk_idx + 1}...")

    # Process harmful and harmless prompts separately
    for column_prefix in ["harmful_prompt", "harmless_prompt"]:
        for cipher in cipher_list:
            ground_truth_col = f"{column_prefix}_{cipher}"
            generated_col = f"{column_prefix}_{cipher}_generated"
            lcs_char_col = f"{column_prefix}_{cipher}_lcs_char"
            lcs_word_col = f"{column_prefix}_{cipher}_lcs_word"

            # Match keys for encryption
            prompt_type = f"plain_to_{cipher}"

            # Translate text in batch
            input_texts = df_chunk[column_prefix].astype(str).tolist()
            df_chunk[generated_col] = batch_translate_text(input_texts, model, tokenizer, prompt_type, is_gemma)

            # Compute LCS evaluation in batch
            lcs_results = evaluate_lcs_batch(df_chunk[ground_truth_col].astype(str).tolist(), df_chunk[generated_col].tolist())
            df_chunk[[lcs_char_col, lcs_word_col]] = pd.DataFrame(lcs_results, index=df_chunk.index)

    # Write to CSV incrementally, only include header for the first batch
    df_chunk.to_csv(output_file, mode='a', index=False, header=(chunk_idx == 0)) # Append mode
    
    # Free memory by deleting processed chunk
    del df_chunk

print(f"Evaluated CSV saved as {output_file}")