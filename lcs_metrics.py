import string

def remove_punctuation(s):
    """Removes all punctuation from a string."""
    return s.translate(str.maketrans('', '', string.punctuation))


def lcs_length(s1, s2):
    """Computes the length of the Longest Common Subsequence (LCS) at the character level. Spaces are not considered character match"""
    s1, s2 = remove_punctuation(s1), remove_punctuation(s2)
    s1, s2 = s1.replace(" ", ""), s2.replace(" ", "") # Remove all empty spaces
    m, n = len(s1), len(s2)
    
    # Initialize LCS table of dimensions (m+1) x (n+1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]  # Match found, increment LCS length
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # Take max from previous computations
    
    return dp[m][n]


def lcs_word_length(s1, s2):
    """Computes the length of the Longest Common Subsequence (LCS) at the word level."""
    s1, s2 = remove_punctuation(s1), remove_punctuation(s2)
    words1 = s1.split()
    words2 = s2.split()
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1] 
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]





# Example test cases for removing punctuations
# text1 = "Hello, world! How's it going?"
# text2 = "Python (3.9) is awesome!!!"

# print(remove_punctuation(text1))  # Output: "Hello world Hows it going"
# print(remove_punctuation(text2))  # Output: "Python 39 is awesome"


# Unit tests for LCS functions
if __name__ == "__main__":
    s1 = "Meet me at the park at midnight."
    print(lcs_length(s1, "Meet me at the park at midnight"))
    print(len(s1.replace(" ", "")))
    print(lcs_length(s1, "hello world"))

    # Character-level LCS tests
    assert lcs_length(s1, "Meet me at the park at midnight") == len(s1.replace(" ", "")) - 1
    assert lcs_length(s1, "Meet at the park at midnight.") == len("Meet at the park at midnight.".replace(" ", "")) - 1 
    assert lcs_length(s1, "Meet me at the park.") == len("Meet me at the park.".replace(" ", "")) - 1
    assert lcs_length(s1, "Meet at park midnight.") < len(s1.replace(" ", ""))  
    assert lcs_length(s1, "hello world") == 4  # Very little commonality

    # Word-level LCS tests
    assert lcs_word_length(s1, "Meet me at the park at midnight.") == 7  
    assert lcs_word_length(s1, "Meet at the park at midnight.") == 6  
    assert lcs_word_length(s1, "at the park at midnight.") == 5  
    assert lcs_word_length(s1, "Meet me park midnight.") == 4  
    assert lcs_word_length(s1, "midnight park the at") < 4  # Words shuffled
    assert lcs_word_length(s1, "hello world") == 0  

    print("All tests passed successfully!")