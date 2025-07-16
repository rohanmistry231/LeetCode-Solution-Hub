# House Robber II

## Problem Statement
You are a robber planning to rob houses in a circular street, represented by `nums`. You cannot rob adjacent houses, and the first and last houses are adjacent. Return the maximum amount you can rob without alerting the police.

**Example**:
- Input: `nums = [2,3,2]`
- Output: `3`
- Explanation: Rob house 2 (3), as houses 1 and 3 are adjacent.

**Constraints**:
- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 1000`

## Solution

### JavaScript
```javascript
function rob(nums) {
    if (nums.length === 1) return nums[0];
    if (nums.length === 2) return Math.max(nums[0], nums[1]);
    
    function robLinear(arr) {
        const dp = new Array(arr.length).fill(0);
        dp[0] = arr[0];
        dp[1] = Math.max(arr[0], arr[1]);
        for (let i = 2; i < arr.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + arr[i]);
        }
        return dp[arr.length - 1];
    }
    
    return Math.max(robLinear(nums.slice(0, -1)), robLinear(nums.slice(1)));
}
```

## Reasoning
- **Approach**: Since the houses form a circle, solve two linear House Robber problems: one excluding the last house (`nums[0:n-1]`) and one excluding the first house (`nums[1:n]`). Return the maximum of the two results. Each linear problem uses DP where `dp[i] = max(dp[i-1], dp[i-2] + arr[i])`.
- **Why DP?**: Breaks the circular constraint into two linear subproblems, reusing the House Robber solution.
- **Edge Cases**:
  - Single house: Return its value.
  - Two houses: Return maximum value.
- **Optimizations**: Reuse linear House Robber logic; can optimize space in `robLinear` to O(1).

## Complexity Analysis
- **Time Complexity**: O(n), two linear passes of length n-1.
- **Space Complexity**: O(n) for the DP array in each linear problem. Can be optimized to O(1) with two variables.

## Best Practices
- Use clear variable names (e.g., `dp`, `robLinear`).
- For Python, use type hints and helper function.
- For JavaScript, use array slicing and helper function.
- For Java, use `Arrays.copyOfRange` and follow Google Java Style Guide.
- Handle edge cases early for clarity.

## Alternative Approaches
- **Optimized Space DP**: Use two variables in `robLinear` (O(n) time, O(1) space).
- **Recursion with Memoization**: Cache results for linear problems (O(n) time, O(n) space). Less efficient than iterative.