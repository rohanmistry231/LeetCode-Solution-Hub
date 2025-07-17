# Binary Tree Zigzag Level Order Traversal

## Problem Statement
Given the root of a binary tree, return the zigzag level order traversal of its nodes' values (i.e., from left to right for odd levels, right to left for even levels, starting with level 1 as left to right).

**Example**:
- Input: `root = [3,9,20,null,null,15,7]`
- Output: `[[3],[20,9],[15,7]]`

**Constraints**:
- The number of nodes in the tree is in the range `[0, 2000]`.
- `-100 <= Node.val <= 100`

## Solution

### Python
```python
from collections import deque

class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> list[list[int]]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        left_to_right = True
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            if not left_to_right:
                current_level.reverse()
            result.append(current_level)
            left_to_right = not left_to_right
        
        return result
```

## Reasoning
- **Approach**: Use BFS with a queue to process nodes level by level, similar to level-order traversal. Track direction with a boolean (`leftToRight`). Reverse the level’s values for even levels (right-to-left).
- **Why BFS?**: Level-order traversal requires processing nodes level by level, and BFS with a queue is ideal. Reversing levels handles zigzag pattern.
- **Edge Cases**:
  - Empty tree: Return empty list.
  - Single node: Return [[node.val]].
  - Skewed tree: Each level has one node, no reversal needed.
- **Optimizations**: Reverse only when needed; use queue size for level separation.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is processed once, with O(w) for reversing each level (w is level width).
- **Space Complexity**: O(w), where w is the maximum width of the tree (queue size, O(n) in worst case for a complete binary tree).

## Best Practices
- Use clear variable names (e.g., `queue`, `leftToRight`).
- For Python, use `deque` and type hints.
- For JavaScript, use array as queue with `shift`/`push`.
- For Java, use `LinkedList` as `Queue` and follow Google Java Style Guide.
- Toggle direction efficiently with boolean.

## Alternative Approaches
- **Recursive DFS**: Track levels and reverse post-process (O(n) time, O(h) space). More complex for zigzag.
- **Two Stacks**: Use stacks to alternate directions (O(n) time, O(w) space). Less intuitive.