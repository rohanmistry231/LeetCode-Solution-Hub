# Min Cost Climbing Stairs

## Problem Statement
You are given an array `cost` where `cost[i]` is the cost of climbing the ith step. You can climb 1 or 2 steps at a time. You can start from step 0 or 1. Return the minimum cost to reach the top (beyond the last step).

**Example**:
- Input: `cost = [10,15,20]`
- Output: `15`
- Explanation: Start at step 1 (15), then reach the top (0 cost).

**Constraints**:
- `2 <= cost.length <= 1000`
- `0 <= cost[i] <= 999`

## Solution

### Java
```java
class Solution {
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n + 1];
        for (int i = 2; i <= n; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
        }
        return dp[n];
    }
}
```

## Reasoning
- **Approach**: Use dynamic programming to compute the minimum cost to reach each step. For step `i`, take the minimum of coming from `i-1` or `i-2`: `dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])`. The top is beyond the last step, so return `dp[n]`.
- **Why DP?**: Avoids recomputing costs by storing minimum costs for each step.
- **Edge Cases**:
  - `n = 2`: Return minimum of `cost[0]` or `cost[1]`.
- **Optimizations**: Use array-based DP; can optimize to O(1) space with two variables.

## Complexity Analysis
- **Time Complexity**: O(n), single loop from 2 to n.
- **Space Complexity**: O(n) for the DP array. Can be optimized to O(1) using two variables.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints for clarity.
- For JavaScript, use array initialization with `fill`.
- For Java, follow Google Java Style Guide.
- Initialize `dp[0]` and `dp[1]` as 0 since starting costs are free.

## Alternative Approaches
- **Optimized Space DP**: Use two variables (O(n) time, O(1) space).
- **Recursion with Memoization**: Cache results (O(n) time, O(n) space). Less efficient than iterative.
- **Brute Force**: Try all paths (O(2^n) time). Too slow.