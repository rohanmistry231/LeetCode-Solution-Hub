# Regular Expression Matching

## Problem Statement
Given an input string `s` and a pattern `p`, implement regular expression matching with support for '.' (matches any single character) and '*' (matches zero or more of the preceding character). The matching should cover the entire input string.

**Example**:
- Input: `s = "aa", p = "a*"`
- Output: `true`
- Explanation: 'a*' matches any sequence of 'a', including "aa".

**Constraints**:
- `0 <= s.length <= 20`
- `0 <= p.length <= 20`
- `s` contains only lowercase letters.
- `p` contains lowercase letters, '.', and '*'.
- '*' is always preceded by a valid character.

## Solution

### JavaScript
```javascript
function isMatch(s, p) {
    const m = s.length, n = p.length;
    const dp = Array(m + 1).fill().map(() => Array(n + 1).fill(false));
    dp[0][0] = true;
    for (let j = 2; j <= n; j++) {
        if (p[j - 1] === '*') dp[0][j] = dp[0][j - 2];
    }
    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            if (p[j - 1] === '*') {
                dp[i][j] = dp[i][j - 2] || (dp[i - 1][j] && (s[i - 1] === p[j - 2] || p[j - 2] === '.'));
            } else {
                dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] === p[j - 1] || p[j - 1] === '.');
            }
        }
    }
    return dp[m][n];
}
```

## Reasoning
- **Approach**: Use a 2D DP array where `dp[i][j]` indicates if `s[0:i]` matches `p[0:j]`. If `p[j-1]` is '*', check zero occurrences (`dp[i][j-2]`) or match with `s[i-1]` if it equals `p[j-2]` or `p[j-2]` is '.'. Otherwise, check if characters match or `p[j-1]` is '.'. Initialize `dp[0][0] = true` and handle empty string with '*' patterns.
- **Why DP?**: Avoids exponential backtracking by caching results for subproblems.
- **Edge Cases**:
  - Empty string or pattern: Handle via initialization.
  - Pattern with only '*': Check valid preceding characters.
- **Optimizations**: Use boolean array to save space; initialize empty pattern cases.

## Complexity Analysis
- **Time Complexity**: O(m * n), filling the DP array.
- **Space Complexity**: O(m * n) for the 2D DP array. Can be optimized to O(min(m,n)) with a 1D array.

## Best Practices
- Use clear variable names (e.g., `dp`, `m`, `n`).
- For Python, use type hints and boolean array.
- For JavaScript, use array methods for initialization.
- For Java, follow Google Java Style Guide.
- Handle '*' cases efficiently with clear logic.

## Alternative Approaches
- **Backtracking**: Recursive exploration (O(2^(min(m,n/2))) time). Slower for large inputs.
- **Optimized Space DP**: Use a 1D array (O(m * n) time, O(min(m,n)) space).