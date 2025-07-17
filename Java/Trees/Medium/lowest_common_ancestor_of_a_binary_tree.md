# Lowest Common Ancestor of a Binary Tree

## Problem Statement
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes `p` and `q`. The LCA is the lowest node that has both `p` and `q` as descendants (a node can be a descendant of itself).

**Example**:
- Input: `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1`
- Output: `3`

**Constraints**:
- The number of nodes in the tree is in the range `[2, 10^5]`.
- `-10^9 <= Node.val <= 10^9`
- All `Node.val` are unique.
- `p != q`
- `p` and `q` exist in the tree.

## Solution

### Java
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) return root;
        return left != null ? left : right;
    }
}
```

## Reasoning
- **Approach**: Use recursive postorder traversal. If the current node is null or equals `p` or `q`, return it. Recurse on left and right subtrees. If both return non-null, the current node is the LCA. Otherwise, return the non-null result.
- **Why Recursive?**: Postorder traversal ensures we process children before parents, identifying the LCA when both nodes are found in different subtrees.
- **Edge Cases**:
  - `p` or `q` is root: Return root.
  - Nodes in same subtree: LCA is higher up.
  - Skewed tree: Works like a linked list.
- **Optimizations**: Simple recursion with no extra data structures; early return on node match.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the height of the tree, due to the recursion stack (O(n) in worst case for skewed tree).

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints.
- For JavaScript, use strict equality checks.
- For Java, follow Google Java Style Guide.
- Minimize checks for efficiency.

## Alternative Approaches
- **Path Tracking**: Find paths to `p` and `q`, then find last common node (O(n) time, O(n) space). More memory-intensive.
- **Iterative**: Use a stack with parent pointers (O(n) time, O(n) space). More complex.