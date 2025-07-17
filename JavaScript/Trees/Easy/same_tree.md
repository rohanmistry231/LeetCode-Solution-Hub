# Same Tree

## Problem Statement
Given the roots of two binary trees `p` and `q`, check if they are the same or not. Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

**Example**:
- Input: `p = [1,2,3], q = [1,2,3]`
- Output: `true`

**Constraints**:
- The number of nodes in both trees is in the range `[0, 100]`.
- `-10^4 <= Node.val <= 10^4`

## Solution

### JavaScript
```javascript
class Solution {
    isSameTree(p, q) {
        if (!p && !q) return true;
        if (!p || !q) return false;
        return (p.val === q.val && 
                this.isSameTree(p.left, q.left) && 
                this.isSameTree(p.right, q.right));
    }
}
```

## Reasoning
- **Approach**: Use recursive comparison. Check if both nodes are null (same), one is null (different), or values differ. Recurse on left and right subtrees.
- **Why Recursive?**: Tree comparison decomposes into checking corresponding nodes and subtrees, making recursion natural and concise.
- **Edge Cases**:
  - Both empty: Return true.
  - One empty: Return false.
  - Single node: Compare values.
- **Optimizations**: Recursive solution is minimal; iterative possible but not simpler.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the minimum number of nodes in `p` or `q`, as each node is visited once.
- **Space Complexity**: O(h), where h is the height of the smaller tree, due to the recursion stack (O(n) in worst case for skewed tree).

## Best Practices
- Use clear variable names (e.g., `p`, `q`).
- For Python, use type hints.
- For JavaScript, use strict equality (`===`).
- For Java, follow Google Java Style Guide.
- Keep comparison logic concise.

## Alternative Approaches
- **Iterative BFS**: Use a queue to compare nodes level-by-level (O(n) time, O(w) space, where w is max width). More complex.
- **Iterative DFS**: Use a stack to compare nodes (O(n) time, O(h) space). Similar complexity but less intuitive.