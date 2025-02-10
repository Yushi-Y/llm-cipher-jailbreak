import string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def caesar_cipher(text, shift=3, decrypt=False):
    """Enciphers or deciphers text using the Caesar cipher with a given shift."""
    if decrypt:
        shift = -shift
    
    def shift_char(c):
        if c in string.ascii_lowercase:
            return chr((ord(c) - ord('a') + shift) % 26 + ord('a'))
        elif c in string.ascii_uppercase:
            return chr((ord(c) - ord('A') + shift) % 26 + ord('A'))
        return c
    
    return ''.join(shift_char(c) for c in text)

def encipher_prompt(prompt, cipher_type="caesar", shift=3):
    """Encodes the prompt based on the chosen cipher method."""
    if cipher_type == "caesar":
        return caesar_cipher(prompt, shift)
    else:
        raise ValueError("Unsupported cipher type.")

def load_model(model_name="meta-llama/Meta-Llama-3-8B"):
    """Loads the open-source model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def translate_with_llm(input_text, model_name, translation_type):
    """Sends text to an open-source model for Caesar cipher translation."""
    tokenizer, model = load_model(model_name)

    model_device = next(model.parameters()).device

    instruction_prompts = {
        "to_caesar": 'You are an expert on The Caesar Cipher. We will communicate using the Caesar Cipher. \n'
                     'The Caesar Cipher is one of the earliest known cryptographic techniques. It is a substitution cipher where each letter in the plaintext is shifted a fixed number of places down the alphabet.\n'
                     'For example, with a shift of one position, A becomes B, B becomes C, and so on. With a shift of three positions, the transformation is as follows:\n'
                     '- Plaintext:  ABCDEFGHIJKLMNOPQRSTUVWXYZ\n'
                     '- Ciphertext: DEFGHIJKLMNOPQRSTUVWXYZABC\n'
                     'Your task is to translate any given natural language text into Caesar Cipher with a shift of three positions.\n'
                     'Here are some examples (with a shift of three positions):\n'
                     '- Plaintext: ATTACK AT DAWN -> Ciphertext: DWWDFN DW GDZQ\n'
                     '- Plaintext: THE SECRET CODE -> Ciphertext: WKH VHFUHW FRGH\n'
                     '- Plaintext: SAFE MESSAGE HERE -> Ciphertext: VDIH PHVVDJH KHUH',

        "to_plain": 'You are an expert on The Caesar Cipher. We will communicate using the Caesar Cipher. \n'
                    'The Caesar Cipher is a simple substitution cipher where each letter in the ciphertext is shifted back to its original position using the same fixed shift applied during encryption.\n'
                    'Your task is to translate any given text encoded using a Caesar shift of three back into natural language.\n'
                    'Here are some examples (with a shift of three positions):\n'
                    '- Ciphertext: DWWDFN DW GDZQ -> Plaintext: ATTACK AT DAWN\n'
                    '- Ciphertext: WKH VHFUHW FRGH -> Plaintext: THE SECRET CODE\n'
                    '- Ciphertext: VDIH PHVVDJH KHUH -> Plaintext: SAFE MESSAGE HERE'
    }

    instruction = instruction_prompts[translation_type]
    full_prompt = instruction + "\n\nInput: " + input_text

    # Ensure inputs are on the same device as the model
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}  

    outputs = model.generate(**inputs, max_new_tokens=100)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def evaluate_translation(original_text, translated_text, translation_type, shift=3):
    """Evaluates the correctness of the translation based on direct Caesar cipher encoding/decoding."""
    expected_output = caesar_cipher(original_text, shift, decrypt=(translation_type == "to_plain"))
    is_correct = expected_output.strip() == translated_text.strip()
    return is_correct, expected_output

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    
    # Example 1: Translating natural language to Caesar Cipher
    original_text = "Meet me at the park at midnight."
    ciphered_text = translate_with_llm(original_text, model_name, "to_caesar")
    print("Original Input:", original_text)
    print("Ciphered Output:", ciphered_text)
    correct, expected = evaluate_translation(original_text, ciphered_text, "to_caesar")
    print("Correct Translation:", correct)
    if not correct:
        print("Expected Output:", expected)
    
    # Example 2: Translating Caesar Cipher back to natural language
    caesar_text = "PHHW PH DW WKH SDUN DW PLGQLJKW."
    decoded_text = translate_with_llm(caesar_text, model_name, "to_plain")
    print("Ciphered Input:", caesar_text)
    print("Decoded Output:", decoded_text)
    correct, expected = evaluate_translation(caesar_text, decoded_text, "to_plain")
    print("Correct Translation:", correct)
    if not correct:
        print("Expected Output:", expected)
