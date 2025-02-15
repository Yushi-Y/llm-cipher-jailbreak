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
    - Decryption: Moves the last consonant cluster back to the beginning and removes "-ay".
    
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

    def pig_latin_word_decrypt(word):
        """Converts a Pig Latin word back to English."""
        if not word.endswith("ay"):
            return word  # If not Pig Latin, return unchanged
        
        word = word[:-2]  # Remove "ay" at the end
        
        # Find last consonant cluster before a vowel
        for i in range(len(word) - 1, -1, -1):
            if word[i] in "AEIOUaeiou":
                return word[i:] + word[:i]  # Move last consonant cluster back

        return word  # If no vowels, return as-is

    words = text.split()
    transformed_words = [
        pig_latin_word_decrypt(word) if decrypt else pig_latin_word_encrypt(word)
        for word in words
    ]

    return " ".join(transformed_words)
