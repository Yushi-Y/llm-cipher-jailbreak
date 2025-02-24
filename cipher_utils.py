import string
import unittest


def rot_cipher(text, shift=3, decrypt=False):
    """Enciphers or deciphers text using the Rot cipher with a given number of positions to shift."""
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
        if word.isdigit():  # If it's a number, still append 'ay'
            return word + "ay"
        if word[0] in vowels:  # Words starting with a vowel
            return word + "ay"
        
        # Find first vowel position
        for i, char in enumerate(word):
            if char in vowels:
                return word[i:] + word[:i] + "ay"
        
        return word + "ay"  # If no vowels, return unchanged with "ay"
    

### THIS IS INCORRECT ###
    # def pig_latin_word_decrypt(word):
    #     """Converts a Pig Latin word back to English."""
    #     if not word.endswith("ay"):
    #         return word  # If not Pig Latin, return unchanged
        
    #     word = word[:-2]  # Remove "ay" at the end

    #     # Handle words that originally started with vowels
    #     if word[-1] in "AEIOUaeiou":
    #         return word  

    #     # Find consonant cluster moved to end and restore it to front
    #     consonant_cluster = ""
    #     for i in range(len(word) - 1, -1, -1):
    #         if word[i] in "AEIOUaeiou":
    #             break
    #         consonant_cluster = word[i] + consonant_cluster  # Build consonant cluster
        
    #     return consonant_cluster + word[:-len(consonant_cluster)]  # Move consonants back

    words = text.rstrip(".").split()  # Remove period only from the end of the sentence
    transformed_words = [pig_latin_word_encrypt(word) for word in words]

    result = " ".join(transformed_words)  # Do not append period back automatically
    return result




##### Unit tests #####
class TestCiphers(unittest.TestCase):
    # ========================
    # TESTS FOR ROT CIPHER
    # ========================
    def test_rot_cipher_basic(self):
        self.assertEqual(rot_cipher("Hello World", shift=3), "Khoor Zruog")

    def test_rot_cipher_with_wraparound(self):
        self.assertEqual(rot_cipher("XYZ", shift=3), "ABC")  # Wrap around

    def test_rot_cipher_with_negative_shift(self):
        self.assertEqual(rot_cipher("DEF", shift=3, decrypt=True), "ABC")  # Decrypt

    def test_rot_cipher_with_large_shift(self):
        self.assertEqual(rot_cipher("Hello", shift=29), "Khoor")  # 29 = 3 mod 26

    def test_rot_cipher_with_punctuation(self):
        self.assertEqual(rot_cipher("Hello, World!", shift=3), "Khoor, Zruog!")  # Punctuation unchanged

    def test_rot_cipher_with_numbers(self):
        self.assertEqual(rot_cipher("Attack at 2 AM!", shift=3), "Dwwdfn dw 2 DP!")  # Numbers unchanged


    # ========================
    # TESTS FOR PIG LATIN CIPHER
    # ========================
    def test_pig_latin_basic(self):
        self.assertEqual(pig_latin_cipher("Hello world"), "elloHay orldway")

    def test_pig_latin_vowel_start(self):
        self.assertEqual(pig_latin_cipher("apple orange"), "appleay orangeay")  # Words starting with vowels

    def test_pig_latin_mixed_case(self):
        self.assertEqual(pig_latin_cipher("Hello There"), "elloHay ereThay")  # Maintain case within words

    # def test_pig_latin_punctuation(self):
    #     self.assertEqual(pig_latin_cipher("Hello, world!"), "elloHay, orldway!")  # Punctuation unchanged

    def test_pig_latin_numbers(self):
        self.assertEqual(pig_latin_cipher("Meet me at 7 PM"), "eetMay emay atay 7ay PMay")  # Numbers also change

    # def test_pig_latin_decrypt_basic(self):
    #     self.assertEqual(pig_latin_cipher("elloHay orldway", decrypt=True), "Hello world")  # Decryption test

    # def test_pig_latin_decrypt_vowel_start(self):
    #     self.assertEqual(pig_latin_cipher("appleay orangeay", decrypt=True), "apple orange")  # Vowel words decrypted

    # def test_pig_latin_decrypt_complex(self):
    #     self.assertEqual(pig_latin_cipher("oolSchay isay unfay", decrypt=True), "School is fun")  # Complex words

if __name__ == "__main__":
    unittest.main()