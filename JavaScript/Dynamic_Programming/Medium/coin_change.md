# Coin Change

## Problem Statement
You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money. Return the fewest number of coins needed to make up that amount. If the amount cannot be made, return -1.

**Example**:
- Input: `coins = [1,2,5], amount = 11`
- Output: `3`
- Explanation: `11 = 5 + 5 + 1`.

**Constraints**:
- `1 <= coins.length <= 12`
- `1 <= coins[i] <= 2^31 - 1`
- `0 <= amount <= 10^4`

## Solution

### JavaScript
```javascript
function coinChange(coins, amount) {
    const dp = new Array(amount + 1).fill(Infinity);
    dp[0] = 0;
    for (let i = 1; i <= amount; i++) {
        for (const coin of coins) {
            if (coin <= i) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[amount] === Infinity ? -1 : dp[amount];
}
```

## Reasoning
- **Approach**: Use dynamic programming where `dp[i]` represents the minimum number of coins to make amount `i`. For each amount, try each coin and update `dp[i] = min(dp[i], dp[i - coin] + 1)`. Initialize `dp[0] = 0` and others to infinity. Return -1 if the final amount is unreachable.
- **Why DP?**: Solves the unbounded knapsack problem efficiently by storing minimum coins for each amount.
- **Edge Cases**:
  - `amount = 0`: Return 0.
  - No solution: Return -1.
- **Optimizations**: Use a 1D DP array; check coin validity to avoid unnecessary computations.

## Complexity Analysis
- **Time Complexity**: O(amount * coins.length), iterating through each amount and coin.
- **Space Complexity**: O(amount) for the DP array.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints and `float('inf')`.
- For JavaScript, use `Infinity` for initialization.
- For Java, use `Integer.MAX_VALUE` and follow Google Java Style Guide.
- Handle unreachable amounts with -1.

## Alternative Approaches
- **BFS**: Treat amounts as nodes in a graph (O(amount * coins.length) time, O(amount) space). More complex.
- **Recursion with Memoization**: Cache results (O(amount * coins.length) time, O(amount) space). Less efficient than iterative.