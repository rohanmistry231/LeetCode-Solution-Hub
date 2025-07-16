# Burst Balloons

## Problem Statement
Given an array `nums` of n integers representing balloons, where bursting the ith balloon gives `nums[i-1] * nums[i] * nums[i+1]` coins (with `nums[-1]` and `nums[n]` as 1). After bursting, the balloon is removed, and adjacent balloons become adjacent. Return the maximum coins you can collect by bursting all balloons.

**Example**:
- Input: `nums = [3,1,5,8]`
- Output: `167`
- Explanation: Burst in order: [3,1,5,8] -> [3,5,8] -> [3,8] -> [8] -> [], coins = 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 167.

**Constraints**:
- `1 <= nums.length <= 300`
- `0 <= nums[i] <= 100`

## Solution

### Java
```java
class Solution {
    public int maxCoins(int[] nums) {
        int n = nums.length;
        int[] newNums = new int[n + 2];
        newNums[0] = 1;
        newNums[n + 1] = 1;
        for (int i = 0; i < n; i++) newNums[i + 1] = nums[i];
        int[][] dp = new int[n + 2][n + 2];
        for (int length = 2; length < n + 2; length++) {
            for (int left = 0; left < n + 2 - length; left++) {
                int right = left + length;
                for (int i = left + 1; i < right; i++) {
                    dp[left][right] = Math.max(dp[left][right], 
                                              dp[left][i] + dp[i][right] + newNums[left] * newNums[i] * newNums[right]);
                }
            }
        }
        return dp[0][n + 1];
    }
}
```

## Reasoning
- **Approach**: Add boundary balloons (value 1) to simplify calculations. Use a 2D DP array where `dp[left][right]` represents the maximum coins for the subarray `nums[left:right]`. For each subarray, try each balloon `i` as the last to burst, computing coins as `nums[left] * nums[i] * nums[right] + dp[left][i] + dp[i][right]`. Take the maximum across all `i`.
- **Why DP?**: Solves the problem by dividing it into subarrays, avoiding recomputation of overlapping subproblems.
- **Edge Cases**:
  - Single balloon: Return its value times 1*1.
  - Empty array: Return 0 (not applicable due to constraints).
- **Optimizations**: Use interval DP; precompute boundaries to simplify logic.

## Complexity Analysis
- **Time Complexity**: O(n^3), where n is the length of `nums`. The DP table has O(n^2) states, and each state considers O(n) choices for the last balloon.
- **Space Complexity**: O(n^2) for the 2D DP array.

## Best Practices
- Use clear variable names (e.g., `dp`, `left`, `right`).
- For Python, use type hints and list comprehension.
- For JavaScript, use spread operator for array extension.
- For Java, follow Google Java Style Guide and use array copying.
- Add boundary balloons to simplify calculations.

## Alternative Approaches
- **Recursion with Memoization**: Cache interval results (O(n^3) time, O(n^2) space). Less efficient than iterative.
- **Divide and Conquer**: Without memoization, leads to exponential time. Infeasible.