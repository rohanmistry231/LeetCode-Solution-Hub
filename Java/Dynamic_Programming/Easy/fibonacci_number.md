# Fibonacci Number

## Problem Statement
The Fibonacci numbers are defined as: `F(0) = 0`, `F(1) = 1`, and `F(n) = F(n-1) + F(n-2)` for `n > 1`. Given `n`, return the value of `F(n)`.

**Example**:
- Input: `n = 2`
- Output: `1`
- Explanation: `F(2) = F(1) + F(0) = 1 + 0 = 1`.

**Constraints**:
- `0 <= n <= 30`

## Solution

### Java
```java
class Solution {
    public int fib(int n) {
        if (n <= 1) return n;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

## Reasoning
- **Approach**: Use dynamic programming to compute the nth Fibonacci number. Each number is the sum of the previous two: `dp[i] = dp[i-1] + dp[i-2]`. Initialize `dp[0] = 0`, `dp[1] = 1`.
- **Why DP?**: Avoids exponential recursive calls by storing intermediate results.
- **Edge Cases**:
  - `n = 0`: Return 0.
  - `n = 1`: Return 1.
- **Optimizations**: Use array-based DP; can optimize to O(1) space with two variables.

## Complexity Analysis
- **Time Complexity**: O(n), single loop from 2 to n.
- **Space Complexity**: O(n) for the DP array. Can be optimized to O(1) using two variables.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints for clarity.
- For JavaScript, use array initialization with `fill`.
- For Java, follow Google Java Style Guide.
- Handle base cases early to avoid unnecessary computation.

## Alternative Approaches
- **Optimized Space DP**: Use two variables (O(n) time, O(1) space).
- **Recursion with Memoization**: Cache results (O(n) time, O(n) space). Less efficient than iterative.
- **Matrix Exponentiation**: Compute Fibonacci using matrix (O(log n) time). Overkill for small n.