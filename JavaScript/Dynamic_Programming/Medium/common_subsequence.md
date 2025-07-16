# Longest Common Subsequence

## Problem Statement
Given two strings `text1` and `text2`, return the length of their longest common subsequence. A subsequence is a sequence that can be derived by deleting some or no elements without changing the order.

**Example**:
- Input: `text1 = "abcde", text2 = "ace"`
- Output: `3`
- Explanation: The longest common subsequence is "ace".

**Constraints**:
- `1 <= text1.length, text2.length <= 1000`
- `text1` and `text2` consist of only lowercase English characters.

## Solution

### JavaScript
```javascript
function longestCommonSubsequence(text1, text2) {
    const m = text1.length, n = text2.length;
    const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (text1[i - 1] === text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[m][n];
}
```

## Reasoning
- **Approach**: Use a 2D DP array where `dp[i][j]` represents the length of the LCS for prefixes `text1[0:i]` and `text2[0:j]`. If characters match, `dp[i][j] = dp[i-1][j-1] + 1`; otherwise, `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`.
- **Why DP?**: Efficiently computes LCS by breaking it into smaller subproblems.
- **Edge Cases**:
  - Empty string: Return 0.
  - No common characters: Return 0.
- **Optimizations**: Use 2D array; can optimize to O(min(m,n)) space with a 1D array and swapping.

## Complexity Analysis
- **Time Complexity**: O(m * n), filling the DP array.
- **Space Complexity**: O(m * n) for the 2D DP array. Can be optimized to O(min(m,n)) with a 1D array.

## Best Practices
- Use clear variable names (e.g., `dp`, `m`, `n`).
- For Python, use type hints and list comprehension.
- For JavaScript, use array methods for initialization.
- For Java, follow Google Java Style Guide.
- Use 0-based indexing with offset for clarity.

## Alternative Approaches
- **Optimized Space DP**: Use a 1D array (O(m * n) time, O(min(m,n)) space).
- **Recursion with Memoization**: Cache results (O(m * n) time, O(m * n) space). Less efficient than iterative.