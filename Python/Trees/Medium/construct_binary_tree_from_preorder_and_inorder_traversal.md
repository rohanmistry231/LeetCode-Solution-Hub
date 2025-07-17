# Construct Binary Tree from Preorder and Inorder Traversal

## Problem Statement
Given two integer arrays `preorder` and `inorder` representing the preorder and inorder traversals of a binary tree, construct and return the binary tree.

**Example**:
- Input: `preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]`
- Output: `[3,9,20,null,null,15,7]`

**Constraints**:
- `1 <= preorder.length <= 3000`
- `inorder.length == preorder.length`
- `-3000 <= preorder[i], inorder[i] <= 3000`
- `preorder` and `inorder` consist of unique values.
- Each value of `inorder` appears in `preorder`.
- `preorder` and `inorder` represent a valid tree.

## Solution

### Python
```python
class Solution:
    def buildTree(self, preorder: list[int], inorder: list[int]) -> TreeNode:
        inorder_map = {val: i for i, val in enumerate(inorder)}
        
        def build(pre_start, pre_end, in_start, in_end):
            if pre_start > pre_end:
                return None
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            root_idx = inorder_map[root_val]
            left_size = root_idx - in_start
            
            root.left = build(pre_start + 1, pre_start + left_size, in_start, root_idx - 1)
            root.right = build(pre_start + left_size + 1, pre_end, root_idx + 1, in_end)
            return root
        
        return build(0, len(preorder) - 1, 0, len(inorder) - 1)
```

## Reasoning
- **Approach**: Use recursion with a hash map to locate root indices. The first element of `preorder` is the root. Find its index in `inorder` to split left and right subtrees. Recursively build subtrees using adjusted indices.
- **Why Recursive?**: Preorder and inorder traversals allow splitting the tree into root, left, and right subtrees, making recursion natural.
- **Edge Cases**:
  - Single node: Return node with no children.
  - Empty arrays: Handled by constraints (length >= 1).
  - Skewed tree: Works with proper index management.
- **Optimizations**: Use hash map for O(1) inorder index lookup; recursive index tracking.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is processed once and hash map lookups are O(1).
- **Space Complexity**: O(n) for the hash map and O(h) for the recursion stack, where h is the tree height (O(n) in worst case).

## Best Practices
- Use clear variable names (e.g., `inorderMap`, `rootIdx`).
- For Python, use type hints and dictionary comprehension.
- For JavaScript, use `Map` for key-value pairs.
- For Java, use `HashMap` and follow Google Java Style Guide.
- Optimize index lookups with hash map.

## Alternative Approaches
- **Iterative**: Use a stack to mimic recursion (O(n) time, O(h) space). More complex.
- **Linear Search**: Find root index without hash map (O(n^2) time). Too slow.