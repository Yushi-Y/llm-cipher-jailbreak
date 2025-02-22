import unittest
from cipher_utils import caesar_cipher, pig_latin_cipher  
class TestCiphers(unittest.TestCase):

    # ========================
    # TESTS FOR CAESAR CIPHER
    # ========================
    def test_caesar_cipher_basic(self):
        self.assertEqual(caesar_cipher("Hello World", shift=3), "Khoor Zruog")

    def test_caesar_cipher_with_wraparound(self):
        self.assertEqual(caesar_cipher("XYZ", shift=3), "ABC")  # Wrap around

    def test_caesar_cipher_with_negative_shift(self):
        self.assertEqual(caesar_cipher("DEF", shift=3, decrypt=True), "ABC")  # Decrypt

    def test_caesar_cipher_with_large_shift(self):
        self.assertEqual(caesar_cipher("Hello", shift=29), "Khoor")  # 29 = 3 mod 26

    # def test_caesar_cipher_with_punctuation(self):
    #     self.assertEqual(caesar_cipher("Hello, World!", shift=3), "Khoor, Zruog!")  # Punctuation unchanged

    def test_caesar_cipher_with_numbers(self):
        self.assertEqual(caesar_cipher("Attack at 2 AM!", shift=3), "Dwwdfn dw 2 DP!")  # Numbers unchanged


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
        self.assertEqual(pig_latin_cipher("Meet me at 7 PM"), "eetMay emay atay 7 PMay")  # Numbers unchanged

    def test_pig_latin_decrypt_basic(self):
        self.assertEqual(pig_latin_cipher("elloHay orldway", decrypt=True), "Hello world")  # Decryption test

    def test_pig_latin_decrypt_vowel_start(self):
        self.assertEqual(pig_latin_cipher("appleay orangeay", decrypt=True), "apple orange")  # Vowel words decrypted

    def test_pig_latin_decrypt_complex(self):
        self.assertEqual(pig_latin_cipher("oolSchay isay unfay", decrypt=True), "School is fun")  # Complex words

if __name__ == "__main__":
    unittest.main()
