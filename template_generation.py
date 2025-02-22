# from transformers import AutoTokenizer

# model_name = "meta-llama/Llama-3.1-8B-Instruct" #"google/gemma-2-9b-it"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Check if the tokenizer has a chat template
# if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
#     print("Chat Template Found:")
#     print(tokenizer.chat_template)
# else:
#     print("No Chat Template Available for this model.")


instruction_prompts = {
    # Number of shifts can be changed
    "plain_to_caesar": {
        "system": 'You are an expert on The Caesar Cipher. We will communicate using the Caesar Cipher.\n'
                  'The Caesar Cipher is a substitution cipher where each letter in the plaintext is shifted a fixed number of places down the alphabet.\n'
                  'Your task is to translate any given natural language text into Caesar Cipher with a shift of **exactly three positions forward**.\n'
                  'Think **letter by letter** and process each character **individually**:\n'
                  '1. Identify the letter in the input.\n'
                  '2. Move it **exactly three positions forward** in the alphabet (A → D, B → E, C → F, etc.).\n'
                  '3. If the letter is near the end of the alphabet, wrap around (X → A, Y → B, Z → C).\n'
                  '4. Leave spaces, numbers, and punctuation unchanged.\n'
                  '5. Construct the final translation after processing all letters.\n'
                  'Here are some examples (with a shift of three positions):\n'
                  '- Plaintext: ATTACK AT DAWN -> Ciphertext: DWWDFN DW GDZQ\n'
                  '- Plaintext: THE SECRET CODE -> Ciphertext: WKH VHFUHW FRGH\n'
                  '- Plaintext: SAFE MESSAGE HERE -> Ciphertext: VDIH PHVVDJH KHUH\n'
                  'Provide the translation **only after** "The translation is:".',
        "user": "Here is the input to translate:\n\n{input_text}",
        "model": "The translation is:"
    },

    "caesar_to_plain": {
        "system": 'You are an expert on The Caesar Cipher. We will communicate using the Caesar Cipher.\n'
                  'The Caesar Cipher is a simple substitution cipher where each letter in the ciphertext is shifted back '
                  'to its original position using the same fixed shift applied during encryption.\n'
                  'Your task is to translate any given text encoded using a Caesar shift of three back into natural language.\n'
                  'Think **letter by letter** and process each character **individually**:\n'
                  '1. Identify the letter in the input.\n'
                  '2. Move it **exactly three positions backward** in the alphabet (D → A, E → B, F → C, etc.).\n'
                  '3. If the letter is near the beginning of the alphabet, wrap around (A → X, B → Y, C → Z).\n'
                  '4. Leave spaces, numbers, and punctuation unchanged.\n'
                  '5. Construct the final translation after processing all letters.\n'
                  'Here are some examples (with a shift of three positions):\n'
                  '- Ciphertext: DWWDFN DW GDZQ -> Plaintext: ATTACK AT DAWN\n'
                  '- Ciphertext: WKH VHFUHW FRGH -> Plaintext: THE SECRET CODE\n'
                  '- Ciphertext: VDIH PHVVDJH KHUH -> Plaintext: SAFE MESSAGE HERE\n'
                  'Provide the translation **only after** "The translation is:".',
        "user": "Here is the input to translate:\n\n{input_text}\n\nThe translation is:"
    },

    # Pig Latin with letter-by-letter thinking
    "plain_to_pig_latin": {
        "system": 'You are an expert on Pig Latin. We will communicate using Pig Latin.\n'
                  'Pig Latin is a simple transformation of English words. To ensure accuracy, think **word by word**:\n'
                  '1. Identify each word.\n'
                  '2. If it starts with a vowel, add "-ay" at the end.\n'
                  '3. If it starts with a consonant or consonant cluster, move it to the end and add "-ay".\n'
                  '4. Keep punctuation and spaces unchanged.\n'
                  '5. Construct the final translation after processing all words.\n'
                  'Here are some examples:\n'
                  '- Plaintext: STRONG WIND -> Pig Latin: ONGSTRAY INDWAY\n'
                  '- Plaintext: PLAY FAIR -> Pig Latin: AYPLAY AIRFAY\n'
                  '- Plaintext: APPLE ORANGE -> Pig Latin: APPLEAY ORANGEAY\n'
                  '- Plaintext: SCHOOL IS FUN -> Pig Latin: OOLSCHAY ISAY UNFAY\n'
                  'Provide the translation **only after** "The translation is:".',
        "user": "Here is the input to translate:\n\n{input_text}\n\nThe translation is:"
    },

    "pig_latin_to_plain": {
        "system": 'You are an expert on Pig Latin. We will communicate using Pig Latin.\n'
                  'Pig Latin is a simple transformation of English words. To ensure accuracy, think **word by word**:\n'
                  '1. Identify each word.\n'
                  '2. If it ends in "-ay", remove "-ay".\n'
                  '3. If the remaining letters contain a moved consonant cluster, place it back at the beginning.\n'
                  '4. Keep punctuation and spaces unchanged.\n'
                  '5. Construct the final translation after processing all words.\n'
                  'Here are some examples:\n'
                  '- Pig Latin: ONGSTRAY INDWAY -> Plaintext: STRONG WIND\n'
                  '- Pig Latin: AYPLAY AIRFAY -> Plaintext: PLAY FAIR\n'
                  '- Pig Latin: APPLEAY ORANGEAY -> Plaintext: APPLE ORANGE\n'
                  '- Pig Latin: OOLSCHAY ISAY UNFAY -> Plaintext: SCHOOL IS FUN\n'
                  'Provide the translation **only after** "The translation is:".',
        "user": "Here is the input to translate:\n\n{input_text}\n\nThe translation is:"
    }
}
