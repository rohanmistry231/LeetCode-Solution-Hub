# Unique Paths

## Problem Statement
A robot starts at the top-left corner of an `m x n` grid and can only move right or down. Return the number of unique paths to reach the bottom-right corner.

**Example**:
- Input: `m = 3, n = 2`
- Output: `3`
- Explanation: Three paths: (right, down), (down, right), (down, down, right).

**Constraints**:
- `1 <= m, n <= 100`
- Answer will fit in a 32-bit signed integer.

## Solution

### Java
```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j] = 1;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }
}
```

## Reasoning
- **Approach**: Use a 2D DP array where `dp[i][j]` represents the number of unique paths to reach `(i,j)`. Each cell can be reached from above (`dp[i-1][j]`) or left (`dp[i][j-1]`), so `dp[i][j] = dp[i-1][j] + dp[i][j-1]`. Initialize first row and column to 1 (only one way to reach those cells).
- **Why DP?**: Avoids recomputing paths by storing results for each cell.
- **Edge Cases**:
  - `m = 1` or `n = 1`: Only one path (all right or all down).
- **Optimizations**: Use 2D array; can optimize to O(min(m,n)) space using a 1D array.

## Complexity Analysis
- **Time Complexity**: O(m * n), filling the DP array.
- **Space Complexity**: O(m * n) for the 2D DP array. Can be optimized to O(min(m,n)) with a 1D array.

## Best Practices
- Use clear variable names (e.g., `dp` for dynamic programming array).
- For Python, use type hints and list comprehension.
- For JavaScript, use array methods for initialization.
- For Java, follow Google Java Style Guide.
- Initialize boundaries to 1 for clarity.

## Alternative Approaches
- **Optimized Space DP**: Use a 1D array (O(m * n) time, O(min(m,n)) space).
- **Combinatorial**: Compute `(m+n-2) choose (m-1)` (O(min(m,n)) time). Requires careful handling of large numbers.
- **Recursion with Memoization**: Cache results (O(m * n) time, O(m * n) space). Less efficient than iterative.