# House Robber

## Problem Statement
You are a robber planning to rob houses along a street. Each house has a certain amount of money, represented by `nums`. You cannot rob adjacent houses. Return the maximum amount you can rob without alerting the police.

**Example**:
- Input: `nums = [1,2,3,1]`
- Output: `4`
- Explanation: Rob house 1 (1) and house 3 (3) for a total of 4.

**Constraints**:
- `0 <= nums.length <= 100`
- `0 <= nums[i] <= 400`

## Solution

### JavaScript
```javascript
function rob(nums) {
    if (!nums.length) return 0;
    if (nums.length <= 2) return Math.max(...nums);
    const dp = new Array(nums.length).fill(0);
    dp[0] = nums[0];
    dp[1] = Math.max(nums[0], nums[1]);
    for (let i = 2; i < nums.length; i++) {
        dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    return dp[dp.length - 1];
}
```

## Reasoning
- **Approach**: Use dynamic programming to track the maximum loot up to each house. For house `i`, either skip it (`dp[i-1]`) or rob it and skip the previous house (`dp[i-2] + nums[i]`). Thus, `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`.
- **Why DP?**: Avoids recomputing subproblems by storing maximum loot for each prefix.
- **Edge Cases**:
  - Empty array: Return 0.
  - One or two houses: Return maximum value.
- **Optimizations**: Use array-based DP; can optimize to O(1) space with two variables.

## Complexity Analysis
- **Time Complexity**: O(n), single loop through the array.
- **Space Complexity**: O(n) for the DP array. Can be optimized to O(1) using two variables.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints and handle edge cases.
- For JavaScript, use spread operator for max in small arrays.
- For Java, follow Google Java Style Guide.
- Handle edge cases early for efficiency.

## Alternative Approaches
- **Optimized Space DP**: Use two variables (O(n) time, O(1) space).
- **Recursion with Memoization**: Cache results (O(n) time, O(n) space). Less efficient than iterative.
- **Brute Force**: Try all valid combinations (O(2^(n/2)) time). Too slow.