# Number of Islands II

## Problem Statement
Given an `m x n` grid initially filled with water, you are given a list of positions to turn into land. After each position is turned into land, return the number of islands formed. An island is a group of 1s (land) connected 4-directionally.

**Example**:
- Input: `m = 3, n = 3, positions = [[0,0],[0,1],[1,2],[2,1]]`
- Output: `[1,1,2,3]`
- Explanation: After each position, the number of islands is computed.

**Constraints**:
- `1 <= m, n <= 10^4`
- `1 <= positions.length <= 10^4`
- `positions[i].length == 2`
- `0 <= positions[i][0] < m`
- `0 <= positions[i][1] < n`

## Solution

### Java
```java
import java.util.*;

class Solution {
    private Map<String, String> parent = new HashMap<>();
    
    private String find(String x) {
        if (!parent.containsKey(x) || !parent.get(x).equals(x)) {
            parent.put(x, find(parent.get(x)));
        }
        return parent.get(x);
    }
    
    private void union(String x, String y) {
        parent.put(find(x), find(y));
    }
    
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> result = new ArrayList<>();
        int islands = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        
        for (int[] pos : positions) {
            int r = pos[0], c = pos[1];
            String curr = r + "," + c;
            if (!parent.containsKey(curr)) {
                parent.put(curr, curr);
                islands++;
                for (int[] dir : directions) {
                    int nr = r + dir[0], nc = c + dir[1];
                    String neighbor = nr + "," + nc;
                    if (parent.containsKey(neighbor) && !find(curr).equals(find(neighbor))) {
                        union(curr, neighbor);
                        islands--;
                    }
                }
            }
            result.add(islands);
        }
        
        return result;
    }
}
```

## Reasoning
- **Approach**: Use a Union-Find data structure to track connected land components. For each new land position, add it to the parent map and increment the island count. Check 4-directional neighbors; if a neighbor is land and in a different component, merge them and decrement the island count.
- **Why Union-Find?**: Efficiently tracks connected components as new land is added, handling merges dynamically.
- **Edge Cases**:
  - Single position: One island.
  - Duplicate positions: Ignore (handled by parent check).
- **Optimizations**: Use path compression in `find` for near-constant time operations.

## Complexity Analysis
- **Time Complexity**: O(k * α(m*n)), where k is the number of positions and α is the inverse Ackermann function (nearly constant). Each position involves constant-time checks and unions.
- **Space Complexity**: O(m * n) for the parent map in the worst case.

## Best Practices
- Use clear variable names (e.g., `parent`, `islands`).
- For Python, use type hints and tuple keys.
- For JavaScript, use string keys for Map.
- For Java, use `HashMap` and follow Google Java Style Guide.
- Implement path compression in Union-Find.

## Alternative Approaches
- **DFS/BFS**: Recompute islands for each position (O(k * m * n) time). Too slow.
- **Grid-Based**: Maintain grid and update islands (O(k * m * n) time). Inefficient for large grids.