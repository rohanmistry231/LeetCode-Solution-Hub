# Critical Connections in a Network

## Problem Statement
Given an undirected graph with `n` nodes (labeled 0 to `n-1`) and a list of undirected `connections`, return a list of critical connections. A critical connection is an edge that, if removed, increases the number of connected components.

**Example**:
- Input: `n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]`
- Output: `[[1,3]]`
- Explanation: Removing [1,3] disconnects node 3 from the rest.

**Constraints**:
- `2 <= n <= 10^5`
- `n-1 <= connections.length <= 10^5`
- `0 <= connections[i][0], connections[i][1] < n`
- All connections are unique.

## Solution

### Python
```python
def criticalConnections(n: int, connections: list[list[int]]) -> list[list[int]]:
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    discovery = [float('inf')] * n
    low = [float('inf')] * n
    time = [0]
    result = []
    
    def dfs(u: int, parent: int) -> None:
        discovery[u] = low[u] = time[0]
        time[0] += 1
        for v in graph[u]:
            if v == parent:
                continue
            if discovery[v] == float('inf'):
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > discovery[u]:
                    result.append([u, v])
            else:
                low[u] = min(low[u], discovery[v])
    
    dfs(0, -1)
    return result
```

## Reasoning
- **Approach**: Use Tarjan’s algorithm to find bridges in an undirected graph. Assign discovery times and lowest reachable times (low-link values) to nodes. An edge (u, v) is critical if `low[v] > discovery[u]`, meaning v cannot reach back to u’s ancestors without that edge.
- **Why Tarjan’s?**: Efficiently identifies bridges in a single DFS pass, suitable for undirected graphs.
- **Edge Cases**:
  - Single edge: Check if it’s critical.
  - Disconnected graph: Not applicable due to constraints (graph is connected).
- **Optimizations**: Use adjacency list; single DFS pass with low-link values.

## Complexity Analysis
- **Time Complexity**: O(V + E), where V is the number of nodes and E is the number of edges, as DFS visits each node and edge once.
- **Space Complexity**: O(V + E) for the adjacency list, O(V) for discovery/low arrays and recursion stack.

## Best Practices
- Use clear variable names (e.g., `discovery`, `low`).
- For Python, use type hints and list comprehension.
- For JavaScript, use array methods for initialization.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Handle parent node explicitly to avoid backtracking errors.

## Alternative Approaches
- **DFS with Edge Removal**: Remove each edge and check connectivity (O(E * (V + E)) time). Too slow.
- **Biconnected Components**: Identify articulation points and bridges (O(V + E) time). Similar but more complex.