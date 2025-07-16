# Longest Increasing Subsequence

## Problem Statement
Given an integer array `nums`, return the length of the longest strictly increasing subsequence.

**Example**:
- Input: `nums = [10,9,2,5,3,7,101,18]`
- Output: `4`
- Explanation: The longest increasing subsequence is `[2,3,7,101]`, with length 4.

**Constraints**:
- `1 <= nums.length <= 2500`
- `-10^4 <= nums[i] <= 10^4`

## Solution

### Java
```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        int max = 1;
        for (int val : dp) {
            max = Math.max(max, val);
        }
        return max;
    }
}
```

## Reasoning
- **Approach**: Use dynamic programming where `dp[i]` represents the length of the longest increasing subsequence ending at index `i`. For each `i`, check all previous indices `j` where `nums[i] > nums[j]` and update `dp[i] = max(dp[i], dp[j] + 1)`. Return the maximum value in `dp`.
- **Why DP?**: Avoids recomputing subsequences by storing lengths for each ending index.
- **Edge Cases**:
  - Empty array: Return 0.
  - Single element: Return 1.
- **Optimizations**: Use a simple array; can optimize to O(n log n) with binary search (not shown here for simplicity).

## Complexity Analysis
- **Time Complexity**: O(n^2), where n is the length of `nums`, due to nested loops.
- **Space Complexity**: O(n) for the DP array.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints for clarity.
- For JavaScript, use array methods for readability.
- For Java, use `Arrays.fill` and follow Google Java Style Guide.
- Initialize `dp` with 1s since each element is a subsequence of length 1.

## Alternative Approaches
- **Binary Search**: Maintain an active list of subsequence ends (O(n log n) time, O(n) space). More complex.
- **Recursion with Memoization**: Cache results (O(n^2) time, O(n^2) space). Less efficient than iterative.