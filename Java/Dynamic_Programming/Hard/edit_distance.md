# Edit Distance

## Problem Statement
Given two strings `word1` and `word2`, return the minimum number of operations (insert, delete, replace) required to convert `word1` to `word2`.

**Example**:
- Input: `word1 = "horse", word2 = "ros"`
- Output: `3`
- Explanation: horse -> rorse (replace 'h' with 'r') -> rose (remove 'r') -> ros (remove 'e').

**Constraints**:
- `0 <= word1.length, word2.length <= 500`
- `word1` and `word2` consist of lowercase English letters.

## Solution

### Java
```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                }
            }
        }
        return dp[m][n];
    }
}
```

## Reasoning
- **Approach**: Use a 2D DP array where `dp[i][j]` represents the minimum operations to convert `word1[0:i]` to `word2[0:j]`. If characters match, `dp[i][j] = dp[i-1][j-1]`; otherwise, take the minimum of insert (`dp[i][j-1]`), delete (`dp[i-1][j]`), or replace (`dp[i-1][j-1]`) plus 1. Initialize first row and column for empty string conversions.
- **Why DP?**: Breaks down the problem into smaller subproblems, avoiding recomputation.
- **Edge Cases**:
  - Empty strings: Return length of the other string.
  - Single character: Compare and compute operations.
- **Optimizations**: Use 2D array; can optimize to O(min(m,n)) space with a 1D array.

## Complexity Analysis
- **Time Complexity**: O(m * n), filling the DP array.
- **Space Complexity**: O(m * n) for the 2D DP array. Can be optimized to O(min(m,n)) with a 1D array.

## Best Practices
- Use clear variable names (e.g., `dp`, `m`, `n`).
- For Python, use type hints and list comprehension.
- For JavaScript, use array methods for initialization.
- For Java, follow Google Java Style Guide.
- Initialize boundaries for empty string cases.

## Alternative Approaches
- **Optimized Space DP**: Use a 1D array (O(m * n) time, O(min(m,n)) space).
- **Recursion with Memoization**: Cache results (O(m * n) time, O(m * n) space). Less efficient than iterative.