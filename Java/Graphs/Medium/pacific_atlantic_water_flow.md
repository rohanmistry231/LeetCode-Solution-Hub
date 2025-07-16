# Pacific Atlantic Water Flow

## Problem Statement
Given an `m x n` matrix `heights` where `heights[i][j]` is the height of the cell, water can flow from a cell to another if the height of the destination cell is less than or equal to the current cell. Water can flow to the Pacific Ocean (top/left edges) and Atlantic Ocean (bottom/right edges). Return a list of coordinates where water can flow to both oceans.

**Example**:
- Input: `heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]`
- Output: `[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]`

**Constraints**:
- `m == heights.length`
- `n == heights[i].length`
- `1 <= m, n <= 200`
- `0 <= heights[i][j] <= 10^5`

## Solution

### Java
```java
import java.util.*;

class Solution {
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        int rows = heights.length, cols = heights[0].length;
        Set<String> pacific = new HashSet<>();
        Set<String> atlantic = new HashSet<>();
        
        for (int c = 0; c < cols; c++) {
            dfs(0, c, pacific, Integer.MIN_VALUE, heights, rows, cols); // Pacific: top row
            dfs(rows - 1, c, atlantic, Integer.MIN_VALUE, heights, rows, cols); // Atlantic: bottom row
        }
        for (int r = 0; r < rows; r++) {
            dfs(r, 0, pacific, Integer.MIN_VALUE, heights, rows, cols); // Pacific: left column
            dfs(r, cols - 1, atlantic, Integer.MIN_VALUE, heights, rows, cols); // Atlantic: right column
        }
        
        List<List<Integer>> result = new ArrayList<>();
        for (String coord : pacific) {
            if (atlantic.contains(coord)) {
                String[] parts = coord.split(",");
                result.add(Arrays.asList(Integer.parseInt(parts[0]), Integer.parseInt(parts[1])));
            }
        }
        return result;
    }
    
    private void dfs(int r, int c, Set<String> ocean, int prevHeight, int[][] heights, int rows, int cols) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || ocean.contains(r + "," + c) || heights[r][c] < prevHeight) return;
        ocean.add(r + "," + c);
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] dir : directions) {
            dfs(r + dir[0], c + dir[1], ocean, heights[r][c], heights, rows, cols);
        }
    }
}
```

## Reasoning
- **Approach**: Use two DFS searches: one from Pacific edges (top row, left column) and one from Atlantic edges (bottom row, right column). Track reachable cells in two sets. Return the intersection of cells reachable by both oceans.
- **Why DFS?**: Efficiently explores all cells that water can flow to, considering reverse flow (from ocean to cell) for simplicity.
- **Edge Cases**:
  - Single cell: Check if it reaches both oceans.
  - No intersection: Return empty list.
- **Optimizations**: Use sets for O(1) lookup; start DFS from edges to reduce complexity.

## Complexity Analysis
- **Time Complexity**: O(m * n), as each cell is visited at most twice (once per ocean).
- **Space Complexity**: O(m * n) for the sets and recursion stack.

## Best Practices
- Use clear variable names (e.g., `pacific`, `atlantic`).
- For Python, use type hints and set operations.
- For JavaScript, use string keys for Set.
- For Java, use `HashSet` and follow Google Java Style Guide.
- Use reverse flow to simplify height checks.

## Alternative Approaches
- **BFS**: Use queues for exploration (O(m * n) time, O(m * n) space). Similar performance.
- **Forward Flow**: Check flow from each cell to both oceans (O(m * n * m * n) time). Inefficient.