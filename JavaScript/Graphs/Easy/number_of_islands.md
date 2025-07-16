# Number of Islands

## Problem Statement
Given an `m x n` 2D binary grid `grid` where '1' represents land and '0' represents water, return the number of islands. An island is surrounded by water and formed by connecting adjacent lands (4-directionally).

**Example**:
- Input: `grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]`
- Output: `1`

**Constraints**:
- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 300`
- `grid[i][j]` is '0' or '1'.

## Solution

### JavaScript
```javascript
function numIslands(grid) {
    if (!grid.length) return 0;
    const rows = grid.length, cols = grid[0].length;
    let islands = 0;
    
    function dfs(r, c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] !== '1') return;
        grid[r][c] = '0';
        for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
            dfs(r + dr, c + dc);
        }
    }
    
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (grid[r][c] === '1') {
                islands++;
                dfs(r, c);
            }
        }
    }
    return islands;
}
```

## Reasoning
- **Approach**: Iterate through the grid. For each '1' (land), increment the island count and use DFS to mark all connected land cells as visited ('0'). This ensures each island is counted exactly once.
- **Why DFS?**: Efficiently explores connected components (islands) in the grid, marking all parts of an island in one pass.
- **Edge Cases**:
  - Empty grid: Return 0.
  - No land: Return 0.
- **Optimizations**: Modify grid in-place to avoid extra space; use DFS for connected component exploration.

## Complexity Analysis
- **Time Complexity**: O(m * n), where m and n are grid dimensions, as each cell is visited at most once.
- **Space Complexity**: O(m * n) for the recursion stack in the worst case (grid is all land).

## Best Practices
- Use clear variable names (e.g., `islands`, `grid`).
- For Python, use type hints and direction tuples.
- For JavaScript, use array destructuring for directions.
- For Java, use 2D array for directions and follow Google Java Style Guide.
- Modify grid in-place to save space.

## Alternative Approaches
- **BFS**: Use a queue to explore islands (O(m * n) time, O(min(m,n)) space). Similar performance but iterative.
- **Union-Find**: Connect adjacent land cells (O(m * n * α(m*n)) time). Overkill for this problem.