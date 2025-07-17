# Recover Binary Search Tree

## Problem Statement
Given a binary search tree (BST) with two nodes swapped, recover the tree by swapping them back without changing the structure.

**Example**:
- Input: `root = [1,3,null,null,2]`
- Output: `[3,1,null,null,2]`
- Explanation: Swap 1 and 3 to restore the BST.

**Constraints**:
- The number of nodes is in the range `[2, 1000]`.
- `-2^31 <= Node.val <= 2^31 - 1`

## Solution

### JavaScript
```javascript
class Solution {
    constructor() {
        this.first = this.second = this.prev = null;
    }
    
    recoverTree(root) {
        function inorder(node) {
            if (!node) return;
            inorder(node.left);
            if (this.prev && this.prev.val > node.val) {
                if (!this.first) this.first = this.prev;
                this.second = node;
            }
            this.prev = node;
            inorder(node.right);
        }
        
        inorder.call(this, root);
        [this.first.val, this.second.val] = [this.second.val, this.first.val];
    }
}
```

## Reasoning
- **Approach**: Use inorder traversal to detect the two swapped nodes in a BST (where inorder should be sorted). Track the previous node and identify violations (prev.val > current.val). The first violation gives the first wrong node, and the last gives the second. Swap their values.
- **Why Inorder?**: BST inorder traversal produces a sorted sequence, making swapped nodes detectable as violations.
- **Edge Cases**:
  - Two nodes swapped: Detect and swap.
  - Adjacent nodes swapped: Single violation.
  - Skewed tree: Works like a linked list.
- **Optimizations**: Single pass with O(1) extra space (excluding recursion stack); swap values directly.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the tree height, due to the recursion stack (O(n) in worst case).

## Best Practices
- Use clear variable names (e.g., `first`, `second`, `prev`).
- For Python, use type hints and class variables.
- For JavaScript, bind context with `call`.
- For Java, use class variables and follow Google Java Style Guide.
- Minimize extra space with pointers.

## Alternative Approaches
- **Morris Traversal**: Inorder without recursion stack (O(n) time, O(1) space). More complex.
- **Explicit Inorder Array**: Store inorder traversal and sort (O(n) time, O(n) space). Less efficient.