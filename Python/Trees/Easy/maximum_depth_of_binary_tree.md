# Maximum Depth of Binary Tree

## Problem Statement
Given the root of a binary tree, return its maximum depth. The maximum depth is the number of nodes along the longest path from the root to a leaf.

**Example**:
- Input: `root = [3,9,20,null,null,15,7]`
- Output: `3`

**Constraints**:
- The number of nodes is in the range `[0, 10^4]`.
- `-100 <= Node.val <= 100`

## Solution

### Python
```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1
```

## Reasoning
- **Approach**: Use recursive depth calculation. For each node, compute the maximum depth of left and right subtrees and return the maximum plus 1 (for the current node).
- **Why Recursive?**: Depth calculation naturally decomposes into subproblems, making recursion simple and intuitive.
- **Edge Cases**:
  - Empty tree: Return 0.
  - Single node: Return 1.
  - Skewed tree: Depth equals length of path.
- **Optimizations**: Recursive solution is concise; iterative BFS/DFS possible but not simpler.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the height of the tree, due to the recursion stack (O(n) in worst case for skewed tree).

## Best Practices
- Use clear variable names (e.g., `leftDepth`, `rightDepth`).
- For Python, use type hints.
- For JavaScript, use `Math.max` for clarity.
- For Java, follow Google Java Style Guide.
- Keep recursion straightforward.

## Alternative Approaches
- **Iterative BFS**: Use a queue to process levels (O(n) time, O(w) space, where w is max width). Suitable for wide trees.
- **Iterative DFS**: Use a stack with depth tracking (O(n) time, O(h) space). Similar complexity but more complex code.