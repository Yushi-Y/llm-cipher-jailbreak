instruction_prompts = {
    # Caesar Cipher: Encode to Caesar (shift 3)
    "plain_to_caesar": {
        "system": 'You are an expert in The Caesar Cipher, a simple letter-substitution cipher used in English.\n'
                  'Your task is to translate English text into **Caesar Cipher** with a shift of **exactly three positions forward**.\n'
                  'Follow these rules strictly:\n'
                  '- Shift each letter forward by **three places** (A → D, B → E, C → F, etc.).\n'
                  '- Wrap around at the end of the alphabet (X → A, Y → B, Z → C).\n'
                  '- **Do not change** spaces, numbers, or punctuation.\n'
                  'Examples of correct translations (shift = 3):\n'
                  '- **ATTACK AT DAWN** → **DWWDFN DW GDZQ**\n'
                  '- **THE SECRET CODE** → **WKH VHFUHW FRGH**\n'
                  '- **SAFE MESSAGE HERE** → **VDIH PHVVDJH KHUH**\n'
                  '**Translate the following English text** and provide only the ciphertext, starting after:\n'
                  '"The translation is:".',
        "user": "Translate this English text into Caesar Cipher (shift 3):\n\n{input_text}\n\nThe translation is:"
    },

    # Caesar Cipher: Decode to English
    "caesar_to_plain": {
        "system": 'You are an expert in The Caesar Cipher, a simple letter-substitution cipher used in English.\n'
                  'Your task is to **decode** text encrypted using the Caesar Cipher (shift 3) **back into English**.\n'
                  'Follow these rules strictly:\n'
                  '- Shift each letter **backward** by **three places** (D → A, E → B, F → C, etc.).\n'
                  '- Wrap around at the beginning of the alphabet (A → X, B → Y, C → Z).\n'
                  '- **Do not change** spaces, numbers, or punctuation.\n'
                  'Examples of correct translations (shift = 3):\n'
                  '- **DWWDFN DW GDZQ** → **ATTACK AT DAWN**\n'
                  '- **WKH VHFUHW FRGH** → **THE SECRET CODE**\n'
                  '- **VDIH PHVVDJH KHUH** → **SAFE MESSAGE HERE**\n'
                  '**Translate the following ciphered text back into English**, and provide only the plaintext, starting after:\n'
                  '"The translation is:".',
        "user": "Translate this Caesar Cipher text (shift 3) back to English:\n\n{input_text}\n\nThe translation is:"
    },

    # Pig Latin: Encode to Pig Latin
    "plain_to_pig_latin": {
        "system": 'You are an expert in Pig Latin, a simple word transformation game used in English.\n'
                  'Your task is to **translate English text into Pig Latin**, following these strict rules:\n'
                  '- If a word **begins with a vowel**, add "-ay" at the end.\n'
                  '- If a word **begins with a consonant (or cluster)**, move it to the end and add "-ay".\n'
                  '- Keep punctuation and spaces **unchanged**.\n'
                  'Examples of correct translations:\n'
                  '- **STRONG WIND** → **ONGSTRAY INDWAY**\n'
                  '- **PLAY FAIR** → **AYPLAY AIRFAY**\n'
                  '- **APPLE ORANGE** → **APPLEAY ORANGEAY**\n'
                  '- **SCHOOL IS FUN** → **OOLSCHAY ISAY UNFAY**\n'
                  '**Translate the following English text** into Pig Latin and provide only the translation, starting after:\n'
                  '"The translation is:".',
        "user": "Translate this English text into Pig Latin:\n\n{input_text}\n\nThe translation is:"
    },

    # Pig Latin: Decode to English
    "pig_latin_to_plain": {
        "system": 'You are an expert in Pig Latin, a simple word transformation game used in English.\n'
                  'Your task is to **decode** Pig Latin text **back into English**, following these strict rules:\n'
                  '- If a word **ends in "-ay"**, remove "-ay".\n'
                  '- If the remaining letters contain a moved consonant cluster, place it **back at the beginning**.\n'
                  '- Keep punctuation and spaces **unchanged**.\n'
                  'Examples of correct translations:\n'
                  '- **ONGSTRAY INDWAY** → **STRONG WIND**\n'
                  '- **AYPLAY AIRFAY** → **PLAY FAIR**\n'
                  '- **APPLEAY ORANGEAY** → **APPLE ORANGE**\n'
                  '- **OOLSCHAY ISAY UNFAY** → **SCHOOL IS FUN**\n'
                  '**Translate the following Pig Latin text back into English**, and provide only the translation, starting after:\n'
                  '"The translation is:".',
        "user": "Translate this Pig Latin text back into English:\n\n{input_text}\n\nThe translation is:"
    }
}
