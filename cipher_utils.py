import string

def caesar_cipher(text, shift=3, decrypt=False):
    """Enciphers or deciphers text using the Caesar cipher with a given number of positions to shift."""
    if decrypt:
        shift = -shift

    def shift_char(c):
        if c in string.ascii_lowercase:
            return chr((ord(c) - ord('a') + shift) % 26 + ord('a'))
        elif c in string.ascii_uppercase:
            return chr((ord(c) - ord('A') + shift) % 26 + ord('A'))
        return c

    return ''.join(shift_char(c) for c in text)




def pig_latin_cipher(text, decrypt=False):
    """Enciphers or deciphers text using Pig Latin.
    
    - Encryption: Moves the first consonant cluster to the end and adds "-ay".
    - Decryption: Restores the original word from Pig Latin.
    
    Args:
        text (str): The input text to encrypt or decrypt.
        decrypt (bool): If True, decrypts from Pig Latin to English.
    
    Returns:
        str: The transformed text.
    """
    
    def pig_latin_word_encrypt(word):
        """Converts a single word to Pig Latin."""
        vowels = "AEIOUaeiou"
        if word[0] in vowels:  # Words starting with a vowel
            return word + "ay"
        
        # Find first vowel position
        for i, char in enumerate(word):
            if char in vowels:
                return word[i:] + word[:i] + "ay"
        
        return word + "ay"  # If no vowels, return unchanged with "ay"


### THIS IS INCORRECT ###
#     def pig_latin_word_decrypt(word):
#         """Converts a Pig Latin word back to English."""
#         if not word.endswith("ay"):
#             return word  # If not Pig Latin, return unchanged
        
#         word = word[:-2]  # Remove "ay" at the end

#         # Handle words that originally started with vowels
#         if word[-1] in "AEIOUaeiou":
#             return word  

#         # Find consonant cluster moved to end and restore it to front
#         consonant_cluster = ""
#         for i in range(len(word) - 1, -1, -1):
#             if word[i] in "AEIOUaeiou":
#                 break
#             consonant_cluster = word[i] + consonant_cluster  # Build consonant cluster
        
#         return consonant_cluster + word[:-len(consonant_cluster)]  # Move consonants back

    words = text.split()
    transformed_words = []

    for word in words:
        # Preserve punctuation
        prefix = ''.join(c for c in word if c in string.punctuation)
        suffix = ''.join(c for c in reversed(word) if c in string.punctuation)
        stripped_word = word.strip(string.punctuation)

        transformed_word = pig_latin_word_encrypt(stripped_word)
        
        # if stripped_word.isdigit():  # Leave numbers unchanged
        #     transformed_word = stripped_word
        # else:
        #     transformed_word = (
        #         pig_latin_word_decrypt(stripped_word) if decrypt else pig_latin_word_encrypt(stripped_word)
        #     )

        transformed_words.append(prefix + transformed_word + suffix)

    return " ".join(transformed_words)
