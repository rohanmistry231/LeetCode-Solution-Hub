# Find if Path Exists in Graph

## Problem Statement
Given an undirected graph with `n` nodes labeled from 0 to `n-1` and a list of undirected edges, determine if there is a valid path from `source` to `destination`.

**Example**:
- Input: `n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2`
- Output: `true`
- Explanation: There are paths like 0->1->2.

**Constraints**:
- `1 <= n <= 2 * 10^5`
- `0 <= edges.length <= 2 * 10^5`
- `edges[i].length == 2`
- `0 <= source, destination <= n-1`

## Solution

### JavaScript
```javascript
function validPath(n, edges, source, destination) {
    const graph = Array(n).fill().map(() => []);
    for (const [u, v] of edges) {
        graph[u].push(v);
        graph[v].push(u);
    }
    
    const visited = new Array(n).fill(false);
    function dfs(node) {
        if (node === destination) return true;
        visited[node] = true;
        for (const neighbor of graph[node]) {
            if (!visited[neighbor] && dfs(neighbor)) return true;
        }
        return false;
    }
    
    return dfs(source);
}
```

## Reasoning
- **Approach**: Build an adjacency list for the undirected graph. Use DFS to explore from `source` to find `destination`. Mark visited nodes to avoid cycles. Return true if `destination` is reached, false otherwise.
- **Why DFS?**: Efficiently checks for a path in an undirected graph, stopping early upon finding the destination.
- **Edge Cases**:
  - `source == destination`: Return true.
  - No edges: Return true if `source == destination`, else false.
- **Optimizations**: Use adjacency list for sparse graphs; early return upon finding destination.

## Complexity Analysis
- **Time Complexity**: O(V + E), where V is the number of nodes (n) and E is the number of edges, as DFS visits each node and edge at most once.
- **Space Complexity**: O(V) for the recursion stack and visited array, plus O(E) for the adjacency list.

## Best Practices
- Use clear variable names (e.g., `graph`, `visited`).
- For Python, use type hints and list comprehension.
- For JavaScript, use array methods for initialization.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Build undirected graph efficiently with adjacency list.

## Alternative Approaches
- **BFS**: Use a queue to explore nodes (O(V + E) time, O(V) space). Similar performance but iterative.
- **Union-Find**: Check if source and destination are in the same component (O(E * α(V)) time). More complex for this problem.