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

### Java
```java
class Solution {
    public boolean isMatch(String s, String p) {
        return backtrack(s, p, 0, 0);
    }
    
    private boolean backtrack(String s, String p, int i, int j) {
        if (j == p.length()) return i == s.length();
        boolean firstMatch = i < s.length() && (p.charAt(j) == s.charAt(i) || p.charAt(j) == '.');
        if (j + 1 < p.length() && p.charAt(j + 1) == '*') {
            return backtrack(s, p, i, j + 2) || (firstMatch && backtrack(s, p, i + 1, j));
        }
        return firstMatch && backtrack(s, p, i + 1, j + 1);
    }
}
```

## Reasoning
- **Approach**: Use backtracking to match the string and pattern. For each position, check if the current characters match (or pattern has '.'). If the next character is '*', try skipping the '*' (zero occurrences) or matching the current character and staying at the same pattern position. Continue until the pattern or string is exhausted.
- **Why Backtracking?**: It handles the recursive nature of '*' by exploring both options (use or skip).
- **Edge Cases**:
  - Empty string or pattern: Handle base case.
  - Pattern with only '*': Check valid preceding character.
- **Optimizations**: Avoid extra space by using recursive calls; handle '*' cases efficiently.

## Complexity Analysis
- **Time Complexity**: O(2^(min(s.length, p.length/2))), as '*' can lead to exponential branching in worst cases.
- **Space Complexity**: O(min(s.length, p.length)) for the recursion stack.

## Best Practices
- Use clear variable names (e.g., `i`, `j` for string and pattern indices).
- For Python, use type hints and concise conditions.
- For JavaScript, use logical operators for clarity.
- For Java, follow Google Java Style Guide and use char access.
- Handle '*' cases separately for clarity.

## Alternative Approaches
- **Dynamic Programming**: Use a DP table to memoize results (O(s.length * p.length) time, O(s.length * p.length) space). More efficient for larger inputs.
- **Greedy**: Not feasible due to backtracking nature of '*'.