# Binary Tree Inorder Traversal

## Problem Statement
Given the root of a binary tree, return the inorder traversal of its nodes' values (left, root, right).

**Example**:
- Input: `root = [1,null,2,3]`
- Output: `[1,3,2]`

**Constraints**:
- The number of nodes is in the range `[0, 100]`.
- `-100 <= Node.val <= 100`

## Solution

### Java
```java
import java.util.*;

class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        inorder(root, result);
        return result;
    }
    
    private void inorder(TreeNode node, List<Integer> result) {
        if (node == null) return;
        inorder(node.left, result);
        result.add(node.val);
        inorder(node.right, result);
    }
}
```

## Reasoning
- **Approach**: Use recursive inorder traversal. Visit the left subtree, append the current node’s value, then visit the right subtree. Store results in a list.
- **Why Recursive?**: Inorder traversal naturally follows a recursive pattern, ensuring correct order (left, root, right) with minimal code.
- **Edge Cases**:
  - Empty tree: Return empty list.
  - Single node: Return [node.val].
  - Skewed tree: Works like a linked list.
- **Optimizations**: Recursive solution is clean and leverages call stack; iterative solution possible but more complex.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the height of the tree, due to the recursion stack (O(n) in worst case for skewed tree).

## Best Practices
- Use clear variable names (e.g., `result`, `node`).
- For Python, use type hints and helper function.
- For JavaScript, use nested function for recursion.
- For Java, use `List` and follow Google Java Style Guide.
- Keep recursion simple for readability.

## Alternative Approaches
- **Iterative**: Use a stack to mimic recursion (O(n) time, O(h) space). More complex but avoids recursion overhead.
- **Morris Traversal**: Threaded binary tree approach (O(n) time, O(1) space). Advanced and less readable.