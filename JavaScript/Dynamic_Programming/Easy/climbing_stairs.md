# Climbing Stairs

## Problem Statement
You are climbing a staircase with `n` steps. Each time, you can climb 1 or 2 steps. Return the number of distinct ways to climb to the top.

**Example**:
- Input: `n = 2`
- Output: `2`
- Explanation: There are two ways: (1,1) and (2).

**Constraints**:
- `1 <= n <= 45`

## Solution

### JavaScript
```javascript
function climbStairs(n) {
    if (n <= 2) return n;
    const dp = new Array(n + 1).fill(0);
    dp[1] = 1;
    dp[2] = 2;
    for (let i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

## Reasoning
- **Approach**: Use dynamic programming to compute the number of ways to reach step `n`. Each step can be reached from step `n-1` (1 step) or `n-2` (2 steps). Thus, `dp[i] = dp[i-1] + dp[i-2]`. Initialize base cases: `dp[1] = 1`, `dp[2] = 2`.
- **Why DP?**: Avoids redundant recursive calls by storing intermediate results, similar to Fibonacci.
- **Edge Cases**:
  - `n = 1`: One way (1 step).
  - `n = 2`: Two ways (1+1, 2).
- **Optimizations**: Use array-based DP for clarity; can optimize to O(1) space with two variables.

## Complexity Analysis
- **Time Complexity**: O(n), single loop from 3 to n.
- **Space Complexity**: O(n) for the DP array. Can be optimized to O(1) using two variables.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints for clarity.
- For JavaScript, use array initialization with `fill`.
- For Java, follow Google Java Style Guide.
- Handle base cases early to avoid unnecessary computation.

## Alternative Approaches
- **Optimized Space DP**: Use two variables instead of an array (O(n) time, O(1) space).
- **Recursion with Memoization**: Cache results to avoid recomputation (O(n) time, O(n) space). Less efficient than iterative.
- **Math (Fibonacci)**: Direct formula for nth Fibonacci (O(log n) time with matrix exponentiation). Overkill for this problem.