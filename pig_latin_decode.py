from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from instruction_harmful_prompts import cipher_harmful_prompts
import pandas as pd
import os



def load_model(model_name):
    """Loads the open-source model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model




def batch_generate_text(input_texts, model, tokenizer, prompt_type, is_gemma):
    """Batch process text for generation."""
    model_device = next(model.parameters()).device

    if prompt_type not in cipher_harmful_prompts:
        print(f"[ERROR] Invalid prompt type: {prompt_type}")
        return [""] * len(input_texts)

    # Prepare batch messages
    messages = []
    for text in input_texts:
        if is_gemma:  # Use combined prompts for Gemma models
            combined_prompt = cipher_harmful_prompts[prompt_type]["system"] + "\n\n" + \
                            cipher_harmful_prompts[prompt_type]["user"].format(input_text=text)
            messages.append([
                {"role": "user", "content": combined_prompt},
                {"role": "assistant", "content": ""}  
            ])
        else:
            messages.append([
                {"role": "system", "content": cipher_harmful_prompts[prompt_type]["system"]},
                {"role": "user", "content": cipher_harmful_prompts[prompt_type]["user"].format(input_text=text)},
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
        max_new_tokens=500,
        do_sample=False
    )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract only the text after the LAST occurrence of "The answer is:"
    extracted_texts = []
    for generated_text in generated_texts:
        last_index = generated_text.rfind("The answer is:")
        extracted_text = generated_text[last_index + len("The answer is:"):].strip() if last_index != -1 else generated_text.strip() # If not found return all translated text

        # Remove "model" or "assistant" if it appears at the beginning
        if extracted_text.lower().startswith("model"):
            extracted_text = extracted_text[len("model"):].strip()
        elif extracted_text.lower().startswith("assistant"):
            extracted_text = extracted_text[len("assistant"):].strip()

        extracted_texts.append(extracted_text)

    return extracted_texts




if __name__ == "__main__":
    model_name = "google/gemma-2-9b-it" 
    
    print(f"Loading model: {model_name}...")
    tokenizer, model = load_model(model_name)  

    is_gemma = "gemma" in model_name.lower()

    input_file = "/data/kebl6672/llm_jailbreak/generated_responses/pig_latin_translation_gemma2.csv"
    output_file = "/data/kebl6672/llm_jailbreak/generated_responses/pig_latin_full_response_gemma2.csv"

    batch_size = 25  # Process batches of rows

    # Read input csv in chunks 
    df_chunks = pd.read_csv(input_file, chunksize=batch_size)

    # If output file exists, delete it to avoid appending old results
    if os.path.exists(output_file):
        os.remove(output_file)

    for chunk_idx, df_chunk in enumerate(df_chunks):
        print(f"Processing batch {chunk_idx + 1}...")

        # Process harmful prompts only
        column_prefix = "harmful_prompt" # "harmful_prompt_pig_latin_generated"
        generated_col = f"{column_prefix}_original_answer"

        prompt_type = 'original_harmful_input'

        # Generate text in batch
        input_texts = df_chunk[column_prefix].astype(str).tolist()
        df_chunk[generated_col] = batch_generate_text(input_texts, model, tokenizer, prompt_type, is_gemma)

        # Write to CSV incrementally, only include header for the first batch
        df_chunk.to_csv(output_file, mode='a', index=False, header=(chunk_idx == 0))  # Append mode
        
        # Free memory 
        del df_chunk

    print(f"Evaluated CSV saved as {output_file}")