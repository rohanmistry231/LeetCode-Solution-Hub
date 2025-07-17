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

### Java
```java
import java.util.*;

class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        return build(preorder, 0, preorder.length - 1, inorderMap, 0, inorder.length - 1);
    }
    
    private TreeNode build(int[] preorder, int preStart, int preEnd, 
                          Map<Integer, Integer> inorderMap, int inStart, int inEnd) {
        if (preStart > preEnd) return null;
        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);
        int rootIdx = inorderMap.get(rootVal);
        int leftSize = rootIdx - inStart;
        
        root.left = build(preorder, preStart + 1, preStart + leftSize, inorderMap, inStart, rootIdx - 1);
        root.right = build(preorder, preStart + leftSize + 1, preEnd, inorderMap, rootIdx + 1, inEnd);
        return root;
    }
}
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