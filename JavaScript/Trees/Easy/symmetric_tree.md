# Symmetric Tree

## Problem Statement
Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

**Example**:
- Input: `root = [1,2,2,3,4,4,3]`
- Output: `true`

**Constraints**:
- The number of nodes is in the range `[1, 1000]`.
- `-100 <= Node.val <= 100`

## Solution

### JavaScript
```javascript
class Solution {
    isSymmetric(root) {
        function isMirror(left, right) {
            if (!left && !right) return true;
            if (!left || !right) return false;
            return (left.val === right.val && 
                    isMirror(left.left, right.right) && 
                    isMirror(left.right, right.left));
        }
        
        return isMirror(root, root);
    }
}
```

## Reasoning
- **Approach**: Use recursive mirror checking. Compare the left and right subtrees of the root, ensuring left’s left matches right’s right and left’s right matches right’s left. Check node values and recurse symmetrically.
- **Why Recursive?**: Symmetry checking naturally compares mirrored subtrees, making recursion intuitive and concise.
- **Edge Cases**:
  - Single node: Symmetric (true).
  - Empty tree: Handled by constraints (non-empty).
  - Unbalanced subtrees: Return false.
- **Optimizations**: Recursive solution is clean; iterative possible but more complex.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the height of the tree, due to the recursion stack (O(n) in worst case for skewed tree).

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints and helper function.
- For JavaScript, use strict equality (`===`).
- For Java, follow Google Java Style Guide.
- Ensure symmetric recursion for clarity.

## Alternative Approaches
- **Iterative**: Use a queue to compare nodes level-by-level (O(n) time, O(w) space, where w is max width). More complex.
- **BFS with Mirroring**: Compare level arrays (O(n) time, O(n) space). Less efficient.