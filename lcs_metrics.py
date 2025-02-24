import re

def lcs_length(s1, s2):
    """Computes the length of the Longest Common Subsequence (LCS) at the character level."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def lcs_word_length(s1, s2):
    """Computes the length of the Longest Common Subsequence (LCS) at the word level."""
    words1 = s1.split()
    words2 = s2.split()
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]



# Unit tests for LCS functions
if __name__ == "__main__":
    s1 = "Meet me at the park at midnight."

    # Character-level LCS tests
    assert lcs_length(s1, "Meet me at the park at midnight.") == len(s1)  # Identical strings
    assert lcs_length(s1, "Meet at the park at midnight.") == len("Meet at the park at midnight.")  
    assert lcs_length(s1, "Meet me at the park.") == len("Meet me at the park.") 
    assert lcs_length(s1, "Meet at park midnight.") < len(s1)  # Partial substring match
    assert lcs_length(s1, "hello world") == 5  # Very little commonality

    # Word-level LCS tests
    assert lcs_word_length(s1, "Meet me at the park at midnight.") == 7  # Identical sentence 
    assert lcs_word_length(s1, "Meet at the park at midnight.") == 6  
    assert lcs_word_length(s1, "at the park at midnight.") == 5  
    assert lcs_word_length(s1, "Meet me park midnight.") == 4  
    assert lcs_word_length(s1, "midnight park the at") < 4  # Words shuffled
    assert lcs_word_length(s1, "hello world") == 0  # No common words

    print("All tests passed successfully!")