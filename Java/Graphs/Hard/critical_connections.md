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

### Java
```java
import java.util.*;

class Solution {
    public List<List<Integer>> criticalConnections(int n, int[][] connections) {
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) graph[i] = new ArrayList<>();
        for (int[] conn : connections) {
            graph[conn[0]].add(conn[1]);
            graph[conn[1]].add(conn[0]);
        }
        
        int[] discovery = new int[n];
        int[] low = new int[n];
        Arrays.fill(discovery, Integer.MAX_VALUE);
        Arrays.fill(low, Integer.MAX_VALUE);
        int[] time = {0};
        List<List<Integer>> result = new ArrayList<>();
        
        dfs(0, -1, graph, discovery, low, time, result);
        return result;
    }
    
    private void dfs(int u, int parent, List<Integer>[] graph, int[] discovery, int[] low, int[] time, List<List<Integer>> result) {
        discovery[u] = low[u] = time[0]++;
        for (int v : graph[u]) {
            if (v == parent) continue;
            if (discovery[v] == Integer.MAX_VALUE) {
                dfs(v, u, graph, discovery, low, time, result);
                low[u] = Math.min(low[u], low[v]);
                if (low[v] > discovery[u]) {
                    result.add(Arrays.asList(u, v));
                }
            } else {
                low[u] = Math.min(low[u], discovery[v]);
            }
        }
    }
}
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