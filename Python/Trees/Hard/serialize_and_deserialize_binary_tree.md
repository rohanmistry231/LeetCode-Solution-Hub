# Serialize and Deserialize Binary Tree

## Problem Statement
Design an algorithm to serialize and deserialize a binary tree. Serialization converts the tree into a string, and deserialization reconstructs the tree from the string.

**Example**:
- Input: `root = [1,2,3,null,null,4,5]`
- Output: `[1,2,3,null,null,4,5]`

**Constraints**:
- The number of nodes is in the range `[0, 10^4]`.
- `-1000 <= Node.val <= 1000`

## Solution

### Python
```python
class Codec:
    def serialize(self, root: TreeNode) -> str:
        if not root:
            return "null"
        return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"
    
    def deserialize(self, data: str) -> TreeNode:
        def dfs():
            val = next(vals)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        
        vals = iter(data.split(","))
        return dfs()
```

## Reasoning
- **Approach**: For serialization, use preorder traversal to convert the tree into a comma-separated string, with "null" for empty nodes. For deserialization, split the string and use a recursive DFS or queue to rebuild the tree in the same preorder.
- **Why Preorder?**: Preorder (root, left, right) ensures the root is processed first, making it easier to reconstruct the tree during deserialization.
- **Edge Cases**:
  - Empty tree: Return "null" or empty tree.
  - Single node: Serialize to "val,null,null".
  - Skewed tree: Works like a linked list.
- **Optimizations**: Use comma-separated string for simplicity; iterator/queue for efficient deserialization.

## Complexity Analysis
- **Time Complexity**: O(n) for both serialize and deserialize, where n is the number of nodes, as each node is processed once.
- **Space Complexity**: O(n) for the string/output and O(h) for the recursion stack/queue, where h is the tree height (O(n) in worst case).

## Best Practices
- Use clear variable names (e.g., `vals`, `node`).
- For Python, use type hints and iterator.
- For JavaScript, use array splitting and index tracking.
- For Java, use `Queue` and follow Google Java Style Guide.
- Ensure robust string handling.

## Alternative Approaches
- **Level-Order Serialization**: Use BFS (O(n) time, O(w) space, where w is max width). More complex deserialization.
- **Custom Encoding**: Use different delimiters or formats (O(n) time, O(n) space). Less standard.