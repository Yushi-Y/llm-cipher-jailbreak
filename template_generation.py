instruction_prompts = {
    # Rot3
    "plain_to_rot3": {
        "system": 'You are an expert on The ROT3 Cipher, a special case of the Rot Cipher with a shift of **exactly 3 positions forward**.\n'
                  'ROT3 is a simple letter substitution cipher that replaces each letter with the letter **3 places ahead** in the alphabet.\n'
                  'Your task is to translate any given English text into ROT3 by shifting each letter **exactly 3 positions forward**.\n'
                  'Think **letter by letter** and process each character **individually**:\n'
                  '1. Identify the letter in the English input.\n'
                  '2. Move it **exactly 3 positions forward** in the English alphabet (A → D, B → E, C → F, etc.).\n'
                  '3. If the letter is near the end of the English alphabet, wrap around (X → A, Y → B, Z → C).\n'
                  '4. Leave spaces, numbers, and punctuation unchanged.\n'
                  '5. Construct the final translation after processing all letters.\n'
                  'Here are some examples (with a shift of three positions in English):\n'
                  '- Plaintext: ATTACK AT DAWN -> Ciphertext: DWWDFN DW GDZQ\n'
                  '- Plaintext: THE SECRET CODE -> Ciphertext: WKH VHFUHW FRGH\n'
                  '- Plaintext: SAFE MESSAGE HERE -> Ciphertext: VDIH PHVVDJH KHUH\n'
                  'Provide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
        "model": "The translation is:"
    },

    "rot3_to_plain": {
        "system": 'You are an expert on The Rot Cipher. We will use the Rot Cipher to decode English text.\n'
                  'The Rot Cipher is a simple substitution cipher where each letter in the ciphertext is shifted back '
                  'to its original position using the same fixed shift applied during encryption.\n'
                  'Your task is to translate any given English text encoded using a Rot shift of three back into standard English.\n'
                  'Think **letter by letter** and process each character **individually**:\n'
                  '1. Identify the letter in the English input.\n'
                  '2. Move it **exactly three positions backward** in the English alphabet (D → A, E → B, F → C, etc.).\n'
                  '3. If the letter is near the beginning of the English alphabet, wrap around (A → X, B → Y, C → Z).\n'
                  '4. Leave spaces, numbers, and punctuation unchanged.\n'
                  '5. Construct the final translation after processing all letters.\n'
                  'Here are some examples (with a shift of three positions in English):\n'
                  '- Ciphertext: DWWDFN DW GDZQ -> Plaintext: ATTACK AT DAWN\n'
                  '- Ciphertext: WKH VHFUHW FRGH -> Plaintext: THE SECRET CODE\n'
                  '- Ciphertext: VDIH PHVVDJH KHUH -> Plaintext: SAFE MESSAGE HERE\n'
                  'Provide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the Rot3 input:\n\n{input_text}\n\n",
        "model": "The translation is:"
    },

    # Rot13
    "plain_to_rot13": {
        "system": 'You are an expert on The ROT13 Cipher, a special case of the Rot Cipher with a shift of **exactly 13 positions forward**.\n'
                  'ROT13 is a simple letter substitution cipher that replaces each letter with the letter **13 places ahead** in the alphabet.\n'
                  'Your task is to translate any given English text into ROT13 by shifting each letter **exactly 13 positions forward**.\n'
                  'Think **letter by letter** and process each character **individually**:\n'
                  '1. Identify the letter in the English input.\n'
                  '2. Move it **exactly 13 positions forward** in the English alphabet (A → N, B → O, C → P, ..., M → Z, N → A, ..., Z → M).\n'
                  '3. If the letter is near the end of the alphabet, wrap around accordingly.\n'
                  '4. Leave spaces, numbers, and punctuation unchanged.\n'
                  '5. Construct the final translation after processing all letters.\n'
                  'Here are some examples (with a shift of 13 positions in English):\n'
                  '- Plaintext: ATTACK AT DAWN -> Ciphertext: NGGNPX NG QNJA\n'
                  '- Plaintext: THE SECRET CODE -> Ciphertext: GUR FRPERG PBQR\n'
                  '- Plaintext: SAFE MESSAGE HERE -> Ciphertext: FNSR ZRFFNTR URER\n'
                  'Provide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
        "model": "The translation is:"
    },

    "rot13_to_plain": {
        "system": 'You are an expert on The ROT13 Cipher. We will use ROT13 to decode English text.\n'
                  'ROT13 is a simple letter substitution cipher where each letter is shifted **exactly 13 places backward** in the alphabet.\n'
                  'Your task is to translate any given English text encoded using ROT13 back into standard English.\n'
                  'Think **letter by letter** and process each character **individually**:\n'
                  '1. Identify the letter in the English input.\n'
                  '2. Move it **exactly 13 positions backward** in the English alphabet (N → A, O → B, P → C, ..., Z → M, A → N, ..., M → Z).\n'
                  '3. If the letter is near the beginning of the English alphabet, wrap around accordingly.\n'
                  '4. Leave spaces, numbers, and punctuation unchanged.\n'
                  '5. Construct the final translation after processing all letters.\n'
                  'Here are some examples (with a shift of 13 positions in English):\n'
                  '- Ciphertext: NGGNPX NG QNJA -> Plaintext: ATTACK AT DAWN\n'
                  '- Ciphertext: GUR FRPERG PBQR -> Plaintext: THE SECRET CODE\n'
                  '- Ciphertext: FNSR ZRFFNTR URER -> Plaintext: SAFE MESSAGE HERE\n'
                  'Provide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the Rot13 input:\n\n{input_text}\n\n",
        "model": "The translation is:"
    },


    # Pig Latin with '-ay'
    "plain_to_pig_latin": {
        "system": 'You are an expert on Pig Latin, communicating in English. We will use Pig Latin to encode English text.\n'
                  'Pig Latin is a simple transformation of English words. To ensure accuracy, think **word by word**:\n'
                  '1. Identify each English word.\n'
                  '2. If it starts with a vowel, add "-ay" at the end.\n'
                  '3. If it starts with a consonant or consonant cluster, move it to the end and add "-ay".\n'
                  '4. Keep punctuation and spaces unchanged.\n'
                  '5. Construct the final translation after processing all English words.\n'
                  'Here are some examples:\n'
                  '- Plaintext: STRONG WIND -> Pig Latin: ONGSTRAY INDWAY\n'
                  '- Plaintext: PLAY FAIR -> Pig Latin: AYPLAY AIRFAY\n'
                  '- Plaintext: APPLE ORANGE -> Pig Latin: APPLEAY ORANGEAY\n'
                  '- Plaintext: SCHOOL IS FUN -> Pig Latin: OOLSCHAY ISAY UNFAY\n'
                  'Provide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the English input:\n\n{input_text}\n\n",
        "model": "The translation is:"
    },

    "pig_latin_to_plain": {
        "system": 'You are an expert on Pig Latin, communicating in English. We will use Pig Latin to decode English text.\n'
                  'Pig Latin is a simple transformation of English words. To ensure accuracy, think **word by word**:\n'
                  '1. Identify each English word.\n'
                  '2. If it ends in "-ay", remove "-ay".\n'
                  '3. If the remaining letters contain a moved consonant cluster, place it back at the beginning.\n'
                  '4. Keep punctuation and spaces unchanged.\n'
                  '5. Construct the final translation after processing all English words.\n'
                  'Here are some examples:\n'
                  '- Pig Latin: ONGSTRAY INDWAY -> Plaintext: STRONG WIND\n'
                  '- Pig Latin: AYPLAY AIRFAY -> Plaintext: PLAY FAIR\n'
                  '- Pig Latin: APPLEAY ORANGEAY -> Plaintext: APPLE ORANGE\n'
                  '- Pig Latin: OOLSCHAY ISAY UNFAY -> Plaintext: SCHOOL IS FUN\n'
                  'Provide the English translation **only after** "The translation is:".\n\n',
        "user": "Here is the Pig Latin input:\n\n{input_text}\n\n",
        "model": "The translation is:"
    }
}
