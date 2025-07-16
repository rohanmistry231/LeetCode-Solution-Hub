# Longest Valid Parentheses

## Problem Statement
Given a string containing only '(' and ')', return the length of the longest valid (well-formed) parentheses substring.

**Example**:
- Input: `s = "(()"`
- Output: `2`
- Explanation: The longest valid substring is "()".

**Constraints**:
- `0 <= s.length <= 3 * 10^4`
- `s[i]` is '(' or ')'.

## Solution

### Java
```java
class Solution {
    public int longestValidParentheses(String s) {
        int n = s.length();
        int[] dp = new int[n + 1];
        int maxLength = 0;
        for (int i = 2; i <= n; i++) {
            if (s.charAt(i - 1) == ')') {
                if (s.charAt(i - 2) == '(') {
                    dp[i] = dp[i - 2] + 2;
                } else if (i - dp[i - 1] - 2 >= 0 && s.charAt(i - dp[i - 1] - 2) == '(') {
                    dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2;
                }
                maxLength = Math.max(maxLength, dp[i]);
            }
        }
        return maxLength;
    }
}
```

## Reasoning
- **Approach**: Use a 1D DP array where `dp[i]` represents the length of the longest valid parentheses substring ending at index `i-1`. For a ')', check if the previous character is '(' (forming "()" pair) or if there's a valid pair before the previous valid substring. Update `dp[i]` accordingly and track the maximum length.
- **Why DP?**: Efficiently computes valid substring lengths by leveraging previous results.
- **Edge Cases**:
  - Empty string: Return 0.
  - No valid pairs: Return 0.
- **Optimizations**: Use 1D array; avoid stack-based solutions for simplicity.

## Complexity Analysis
- **Time Complexity**: O(n), single pass through the string.
- **Space Complexity**: O(n) for the DP array.

## Best Practices
- Use clear variable names (e.g., `dp`, `max_length`).
- For Python, use type hints and array initialization.
- For JavaScript, use array methods for clarity.
- For Java, follow Google Java Style Guide.
- Handle closing parentheses cases carefully.

## Alternative Approaches
- **Stack**: Use a stack to track indices of '(' and compute lengths (O(n) time, O(n) space). More complex.
- **Two-Pass Counting**: Count open/close parentheses in both directions (O(n) time, O(1) space). Less intuitive.