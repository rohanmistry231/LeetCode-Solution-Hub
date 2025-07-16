# Clone Graph

## Problem Statement
Given a reference to a node in a connected undirected graph, return a deep copy (clone) of the graph. Each node contains an integer value and a list of its neighbors.

**Example**:
- Input: `adjList = [[2,4],[1,3],[2,4],[1,3]]`
- Output: `[[2,4],[1,3],[2,4],[1,3]]`
- Explanation: Clone the graph with the same structure and values.

**Constraints**:
- `1 <= number of nodes <= 100`
- `1 <= node.val <= 100`
- Each node value is unique.
- The graph is connected and undirected.

## Solution

### JavaScript
```javascript
function Node(val, neighbors) {
    this.val = val;
    this.neighbors = neighbors || [];
}

function cloneGraph(node) {
    if (!node) return null;
    const clones = new Map();
    
    function dfs(original) {
        if (clones.has(original)) return clones.get(original);
        const clone = new Node(original.val);
        clones.set(original, clone);
        for (const neighbor of original.neighbors) {
            clone.neighbors.push(dfs(neighbor));
        }
        return clone;
    }
    
    return dfs(node);
}
```

## Reasoning
- **Approach**: Use DFS to traverse the graph, creating a clone for each node and mapping original nodes to their clones. For each node, recursively clone its neighbors and add them to the clone’s neighbor list. Use a map to avoid duplicating nodes.
- **Why DFS?**: Naturally handles the recursive structure of the graph, ensuring all nodes are cloned exactly once.
- **Edge Cases**:
  - Null node: Return null.
  - Single node: Clone with no neighbors.
- **Optimizations**: Use a map to cache clones, preventing cycles and redundant cloning.

## Complexity Analysis
- **Time Complexity**: O(V + E), where V is the number of nodes and E is the number of edges, as each node and edge is visited once.
- **Space Complexity**: O(V) for the map and recursion stack.

## Best Practices
- Use clear variable names (e.g., `clones`, `original`).
- For Python, use type hints and dictionary for clones.
- For JavaScript, use Map for object keys.
- For Java, use `HashMap` and follow Google Java Style Guide.
- Handle cycles using a map to cache clones.

## Alternative Approaches
- **BFS**: Use a queue to clone nodes (O(V + E) time, O(V) space). Similar performance but iterative.
- **Iterative DFS**: Use a stack (O(V + E) time, O(V) space). More complex.