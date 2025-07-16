# Minimum Cost to Connect All Points

## Problem Statement
Given an array `points` where `points[i] = [xi, yi]` represents a point on the XY-plane, return the minimum cost to connect all points. The cost of connecting two points is the Manhattan distance: `|xi - xj| + |yi - yj|`. All points must be connected directly or indirectly.

**Example**:
- Input: `points = [[0,0],[2,2],[3,10],[5,2],[7,0]]`
- Output: `20`
- Explanation: Connect points to form a minimum spanning tree with total cost 20.

**Constraints**:
- `1 <= points.length <= 1000`
- `-10^6 <= xi, yi <= 10^6`
- All points are unique.

## Solution

### Python
```python
def minCostConnectPoints(points: list[list[int]]) -> int:
    n = len(points)
    if n <= 1:
        return 0
    
    # Build adjacency list of costs
    adj = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            adj.append((cost, i, j))
    adj.sort()  # Sort by cost
    
    # Union-Find
    parent = list(range(n))
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int) -> bool:
        if find(x) == find(y):
            return False
        parent[find(x)] = find(y)
        return True
    
    # Kruskal's algorithm
    total_cost = 0
    edges_used = 0
    for cost, u, v in adj:
        if union(u, v):
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return total_cost
```

## Reasoning
- **Approach**: Use Kruskal’s algorithm to find the Minimum Spanning Tree (MST). Compute Manhattan distances for all pairs of points, sort edges by cost, and use Union-Find to select edges that connect all points without forming cycles. Stop when `n-1` edges are used.
- **Why Kruskal’s?**: Efficiently finds the MST for a complete graph, minimizing total connection cost.
- **Edge Cases**:
  - Single point: Return 0.
  - Two points: Return their Manhattan distance.
- **Optimizations**: Use path compression in Union-Find; sort edges to prioritize low-cost connections.

## Complexity Analysis
- **Time Complexity**: O(N^2 log N), where N is the number of points. Building the edge list takes O(N^2), sorting takes O(N^2 log N), and Union-Find operations are nearly O(1) with path compression.
- **Space Complexity**: O(N^2) for the edge list, O(N) for the parent array.

## Best Practices
- Use clear variable names (e.g., `adj`, `parent`).
- For Python, use type hints and list for edges.
- For JavaScript, use array for edges and Map for Union-Find if needed.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Break early when `n-1` edges are used.

## Alternative Approaches
- **Prim’s Algorithm**: Use a priority queue to build MST (O(N^2 log N) time). Similar complexity.
- **Brute Force**: Try all possible spanning trees (O(N!)). Infeasible.