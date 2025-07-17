# Binary Tree Maximum Path Sum

## Problem Statement
Given a binary tree, find the maximum path sum. A path is a sequence of nodes connected by edges, and the path sum is the sum of node values. The path does not need to pass through the root.

**Example**:
- Input: `root = [1,2,3]`
- Output: `6`
- Explanation: The maximum path is [2,1,3] with sum 6.

**Constraints**:
- The number of nodes is in the range `[1, 3 * 10^4]`.
- `-1000 <= Node.val <= 1000`

## Solution

### Java
```java
class Solution {
    private int maxSum = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
    
    private int maxGain(TreeNode node) {
        if (node == null) return 0;
        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);
        int currentPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, currentPath);
        return node.val + Math.max(leftGain, rightGain);
    }
}
```

## Reasoning
- **Approach**: Use recursive DFS to compute the maximum gain from each node (max path sum through one child). Track the global maximum path sum, including paths that combine left and right subtrees through the current node. Ignore negative gains.
- **Why Recursive?**: Allows tracking both single-path gains (for parent) and full-path sums (including both children) in one pass.
- **Edge Cases**:
  - Single node: Return node value.
  - Negative values: Ignore negative subtree sums.
  - Skewed tree: Path may be a single branch.
- **Optimizations**: Use max to ignore negative contributions; single pass updates global max.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the tree height, due to the recursion stack (O(n) in worst case).

## Best Practices
- Use clear variable names (e.g., `maxSum`, `maxGain`).
- For Python, use type hints and nonlocal variable.
- For JavaScript, use `Math.max` for clarity.
- For Java, use class variable and follow Google Java Style Guide.
- Handle negative values efficiently.

## Alternative Approaches
- **Brute Force**: Check all possible paths (O(n^2) time). Too slow.
- **Two-Pass DFS**: Compute max sums separately (O(n) time, O(n) space). More complex.