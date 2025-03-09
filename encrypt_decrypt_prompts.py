# Set these flags to include or exclude parts of prompts
include_rules = True
include_examples = True
output_directly = False # Whether ask the LLM to output translation directly

encrypt_decrypt_prompts = {
    # ROT3 Encryption
    "plain_to_rot3": {
        "system": 'You are an expert on The ROT3 Cipher, a special case of the Rot Cipher with a shift of **exactly 3 positions forward**.\n'
                  'ROT3 is a simple letter substitution cipher that replaces each letter with the letter **3 places ahead** in the alphabet.\n'
                  'Your task is to translate any given English text into ROT3.\n'
                  + ("\nThink **letter by letter**:\n"
                     "1. Identify the letter in the English input.\n"
                     "2. Move it **exactly 3 positions forward** in the English alphabet (A → D, B → E, C → F, etc.).\n"
                     "3. If the letter is near the end of the English alphabet, wrap around (X → A, Y → B, Z → C).\n"
                     "4. Leave spaces, numbers, and punctuation unchanged.\n"
                     "5. Construct the final translation after processing all letters.\n"
                     if include_rules else "")
                  + ("\nHere are some examples (with a shift of three positions in English):\n"
                     "- Plaintext: ATTACK AT DAWN -> Ciphertext: DWWDFN DW GDZQ\n"
                     "- Plaintext: THE SECRET CODE -> Ciphertext: WKH VHFUHW FRGH\n"
                     "- Plaintext: SAFE MESSAGE HERE -> Ciphertext: VDIH PHVVDJH KHUH\n"
                     if include_examples else "")
                  +  ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },

    # ROT3 Decryption
    "rot3_to_plain": {
        "system": 'You are an expert on The ROT3 Cipher, a special case of the Rot Cipher with a shift of **exactly 3 positions backward**.\n'
                  'ROT3 is a simple letter substitution cipher that replaces each letter with the letter **3 places behind** in the alphabet.\n'
                  'Your task is to translate any given ROT3-encoded text back into standard English.\n'
                  + ("\nThink **letter by letter**:\n"
                     "1. Identify the letter in the English input.\n"
                     "2. Move it **exactly 3 positions backward** in the English alphabet (D → A, E → B, F → C, etc.).\n"
                     "3. If the letter is near the beginning of the English alphabet, wrap around (A → X, B → Y, C → Z).\n"
                     "4. Leave spaces, numbers, and punctuation unchanged.\n"
                     "5. Construct the final translation after processing all letters.\n"
                     if include_rules else "")
                  + ("\nHere are some examples (with a shift of three positions in English):\n"
                     "- Ciphertext: DWWDFN DW GDZQ -> Plaintext: ATTACK AT DAWN\n"
                     "- Ciphertext: WKH VHFUHW FRGH -> Plaintext: THE SECRET CODE\n"
                     "- Ciphertext: VDIH PHVVDJH KHUH -> Plaintext: SAFE MESSAGE HERE\n"
                     if include_examples else "")
                  +  ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the ROT3 input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },


    # ROT13 Encryption
    "plain_to_rot13": {
        "system": 'You are an expert on The ROT13 Cipher, a special case of the Rot Cipher with a shift of **exactly 13 positions forward**.\n'
                  'ROT13 is a simple letter substitution cipher that replaces each letter with the letter **13 places ahead** in the alphabet.\n'
                  'Your task is to translate any given English text into ROT13.\n'
                  + ("\nThink **letter by letter**:\n"
                    "1. Identify the letter in the English input.\n"
                    "2. Move it **exactly 13 positions forward** in the English alphabet (A → N, B → O, C → P, etc.).\n"
                    "3. If the letter is near the end of the English alphabet, wrap around accordingly (M → Z, N → A, etc.).\n"
                    "4. Leave spaces, numbers, and punctuation unchanged.\n"
                    "5. Construct the final translation after processing all letters.\n"
                    if include_rules else "")
                  + ("\nHere are some examples (with a shift of thirteen positions in English):\n"
                    "- Plaintext: ATTACK AT DAWN -> Ciphertext: NGGNPX NG QNJA\n"
                    "- Plaintext: THE SECRET CODE -> Ciphertext: GUR FRPERG PBQR\n"
                    "- Plaintext: SAFE MESSAGE HERE -> Ciphertext: FNSR ZRFFNTR URER\n"
                    if include_examples else "")
                  +  ("\nOutput only the final translation **without explaining the process**.\n"
                    if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },

    # ROT13 Decryption
    "rot13_to_plain": {
        "system": 'You are an expert on The ROT13 Cipher, a special case of the Rot Cipher with a shift of **exactly 13 positions backward**.\n'
                  'ROT13 is a simple letter substitution cipher that replaces each letter with the letter **13 places behind** in the alphabet.\n'
                  'Your task is to translate any given ROT13-encoded text back into standard English.\n'
                  + ("\nThink **letter by letter**:\n"
                    "1. Identify the letter in the ROT13 input.\n"
                    "2. Move it **exactly 13 positions backward** in the English alphabet (N → A, O → B, P → C, etc.).\n"
                    "3. If the letter is near the beginning of the English alphabet, wrap around accordingly (A → N, B → O, etc.).\n"
                    "4. Leave spaces, numbers, and punctuation unchanged.\n"
                    "5. Construct the final translation after processing all letters.\n"
                    if include_rules else "")
                  + ("\nHere are some examples (with a shift of thirteen positions in English):\n"
                    "- Ciphertext: NGGNPX NG QNJA -> Plaintext: ATTACK AT DAWN\n"
                    "- Ciphertext: GUR FRPERG PBQR -> Plaintext: THE SECRET CODE\n"
                    "- Ciphertext: FNSR ZRFFNTR URER -> Plaintext: SAFE MESSAGE HERE\n"
                    if include_examples else "")
                  +  ("\nOutput only the final translation **without explaining the process**.\n"
                    if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the ROT13 input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },


    # Pig Latin Encryption
    "plain_to_pig_latin": {
        "system": 'You are an expert on Pig Latin, communicating in English. We will use Pig Latin to encode English text.\n'
                  'Pig Latin is a simple word transformation where words starting with vowels have "-ay" added at the end, while words starting with consonants move their initial consonant sounds to the end and add "-ay".\n'
                  'Your task is to accurately translate any given English text into Pig Latin by applying these transformation rules to each word.\n'
                  + ("\nThink **word by word**:\n"
                     "1. Identify each English word.\n"
                     "2. If it starts with a vowel, add \"-ay\" at the end.\n"
                     "3. If it starts with a consonant or consonant cluster, move it to the end and add \"-ay\".\n"
                     "4. Keep punctuation and spaces unchanged.\n"
                     "5. Construct the final translation after processing all English words.\n"
                     if include_rules else "")
                  + ("\nHere are some examples:\n"
                     "- Plaintext: STRONG WIND -> Pig Latin: ONGSTRAY INDWAY\n"
                     "- Plaintext: PLAY FAIR -> Pig Latin: AYPLAY AIRFAY\n"
                     "- Plaintext: APPLE ORANGE -> Pig Latin: APPLEAY ORANGEAY\n"
                     "- Plaintext: SCHOOL IS FUN -> Pig Latin: OOLSCHAY ISAY UNFAY\n"
                     if include_examples else "")
                  +  ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },

    # Pig Latin Decryption
    "pig_latin_to_plain": {
        "system": 'You are an expert on Pig Latin, communicating in English. We will use Pig Latin to decode English text.\n'
                  'Pig Latin is a simple transformation of English words where words starting with vowels have "-ay" added at the end, while words starting with consonants have their initial consonant sounds moved to the end followed by "-ay".\n'
                  'Your task is to accurately translate any given Pig Latin text back into standard English by reversing these transformations.\n'
                  + ("\nThink **word by word**:\n"
                     "1. Identify each English word.\n"
                     "2. If it ends in \"-ay\", remove \"-ay\".\n"
                     "3. If the remaining letters contain a moved consonant cluster, place it back at the beginning.\n"
                     "4. Keep punctuation and spaces unchanged.\n"
                     "5. Construct the final translation after processing all English words.\n"
                     if include_rules else "")
                  + ("\nHere are some examples:\n"
                     "- Pig Latin: ONGSTRAY INDWAY -> Plaintext: STRONG WIND\n"
                     "- Pig Latin: AYPLAY AIRFAY -> Plaintext: PLAY FAIR\n"
                     "- Pig Latin: APPLEAY ORANGEAY -> Plaintext: APPLE ORANGE\n"
                     "- Pig Latin: OOLSCHAY ISAY UNFAY -> Plaintext: SCHOOL IS FUN\n"
                     if include_examples else "")
                  +  ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the Pig Latin input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },


    # ASCII Decimal Encoding
    "plain_to_ascii_decimal": {
        "system": 'You are an expert in ASCII encoding. We will use ASCII to represent English text in decimal format.\n'
                  'ASCII is a character encoding standard that assigns a unique numerical value to each letter, number, and symbol.\n'
                  'Your task is to translate any given English text into its ASCII decimal representation.\n'
                  + ("\nThink **character by character**:\n"
                     "1. Identify each character in the English input.\n"
                     "2. Convert each character into its ASCII decimal value.\n"
                     "3. Separate ASCII values with spaces.\n"
                     "4. Leave spaces, numbers, and punctuation unchanged.\n"
                     "5. Construct the final translation after processing all characters.\n"
                     if include_rules else "")
                  + ("\nHere are some examples:\n"
                     "- Plaintext: HELLO -> ASCII Decimal: 72 69 76 76 79\n"
                     "- Plaintext: MEET ME -> ASCII Decimal: 77 69 69 84 32 77 69\n"
                     "- Plaintext: CODE 123 -> ASCII Decimal: 67 79 68 69 32 49 50 51\n"
                     if include_examples else "")
                  + ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the ASCII decimal translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },

    # ASCII Decimal Decoding
    "ascii_decimal_to_plain": {
        "system": 'You are an expert in ASCII encoding. We will use ASCII decimal encoding to decode English text.\n'
                  'ASCII is a character encoding standard where each character corresponds to a unique numerical value.\n'
                  'Your task is to convert any given ASCII decimal sequence back into standard English text.\n'
                  + ("\nThink **number by number**:\n"
                     "1. Identify each ASCII decimal value in the input.\n"
                     "2. Convert each ASCII decimal value back into its corresponding character.\n"
                     "3. Maintain spaces and punctuation as they appear.\n"
                     "4. Construct the final translation after processing all values.\n"
                     if include_rules else "")
                  + ("\nHere are some examples:\n"
                     "- ASCII Decimal: 72 69 76 76 79 -> Plaintext: HELLO\n"
                     "- ASCII Decimal: 77 69 69 84 32 77 69 -> Plaintext: MEET ME\n"
                     "- ASCII Decimal: 67 79 68 69 32 49 50 51 -> Plaintext: CODE 123\n"
                     if include_examples else "")
                  + ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the ASCII decimal input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },

    # ASCII 7-bit Binary Encoding
    "plain_to_ascii_7bit": {
        "system": 'You are an expert in ASCII encoding. We will use **7-bit binary ASCII** to represent English text.\n'
                  'ASCII is a 7-bit binary encoding standard that assigns a unique binary sequence to each character.\n'
                  'Your task is to translate any given English text into its **7-bit ASCII binary** representation.\n'
                  + ("\nThink **character by character**:\n"
                     "1. Identify each character in the English input.\n"
                     "2. Convert each character into its **7-bit ASCII binary value**.\n"
                     "3. Separate ASCII binary values with spaces.\n"
                     "4. Leave spaces, numbers, and punctuation unchanged.\n"
                     "5. Construct the final translation after processing all characters.\n"
                     if include_rules else "")
                  + ("\nHere are some examples:\n"
                     "- Plaintext: HELLO -> ASCII Binary: 1001000 1000101 1001100 1001100 1001111\n"
                     "- Plaintext: MEET ME -> ASCII Binary: 1001101 1000101 1000101 1010100 00100000 1001101 1000101\n"
                     "- Plaintext: CODE 123 -> ASCII Binary: 1000011 1001111 1000100 1000101 00100000 0110001 0110010 0110011\n"
                     if include_examples else "")
                  + ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the ASCII binary translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    },

    # ASCII 7-bit Binary Decoding
    "ascii_7bit_to_plain": {
        "system": 'You are an expert in ASCII encoding. We will use **7-bit binary ASCII** to decode English text.\n'
                  'ASCII is a 7-bit binary encoding standard where each character corresponds to a unique binary sequence.\n'
                  'Your task is to convert any given **7-bit ASCII binary sequence** back into standard English text.\n'
                  + ("\nThink **binary sequence by binary sequence**:\n"
                     "1. Identify each **7-bit ASCII binary** value in the input.\n"
                     "2. Convert each **7-bit ASCII binary** value back into its corresponding character.\n"
                     "3. Maintain spaces and punctuation as they appear.\n"
                     "4. Construct the final translation after processing all values.\n"
                     if include_rules else "")
                  + ("\nHere are some examples:\n"
                     "- ASCII Binary: 1001000 1000101 1001100 1001100 1001111 -> Plaintext: HELLO\n"
                     "- ASCII Binary: 1001101 1000101 1000101 1010100 00100000 1001101 1000101 -> Plaintext: MEET ME\n"
                     "- ASCII Binary: 1000011 1001111 1000100 1000101 00100000 0110001 0110010 0110011 -> Plaintext: CODE 123\n"
                     if include_examples else "")
                  + ("\nOutput only the final translation **without explaining the process**.\n"
                     if output_directly else "")
                  + '\nProvide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the ASCII binary input:\n\n{input_text}\n\n",
      #   "assistant": "" # "The translation is:"
    }
}

              
