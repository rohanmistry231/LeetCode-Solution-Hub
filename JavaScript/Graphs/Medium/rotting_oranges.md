# Rotting Oranges

## Problem Statement
Given a `m x n` grid where each cell can be 0 (empty), 1 (fresh orange), or 2 (rotten orange), every minute, any fresh orange adjacent to a rotten orange (4-directionally) becomes rotten. Return the minimum number of minutes until no fresh oranges remain, or -1 if impossible.

**Example**:
- Input: `grid = [[2,1,1],[1,1,0],[0,1,1]]`
- Output: `4`

**Constraints**:
- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 10`
- `grid[i][j]` is 0, 1, or 2.

## Solution

### JavaScript
```javascript
function orangesRotting(grid) {
    const rows = grid.length, cols = grid[0].length;
    const queue = [];
    let fresh = 0;
    
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (grid[r][c] === 2) queue.push([r, c]);
            else if (grid[r][c] === 1) fresh++;
        }
    }
    
    let minutes = 0;
    const directions = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    
    while (queue.length && fresh > 0) {
        const size = queue.length;
        for (let i = 0; i < size; i++) {
            const [r, c] = queue.shift();
            for (const [dr, dc] of directions) {
                const nr = r + dr, nc = c + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === 1) {
                    grid[nr][nc] = 2;
                    queue.push([nr, nc]);
                    fresh--;
                }
            }
        }
        minutes++;
    }
    
    return fresh === 0 ? minutes : -1;
}
```

## Reasoning
- **Approach**: Use BFS to simulate the rotting process. Start with all rotten oranges in a queue and count fresh oranges. For each minute, process all current rotten oranges, spreading rot to adjacent fresh oranges. Increment minutes per level. Return -1 if fresh oranges remain.
- **Why BFS?**: Ensures minimum time by processing oranges level by level (minute by minute).
- **Edge Cases**:
  - No fresh oranges: Return 0.
  - Unreachable fresh oranges: Return -1.
- **Optimizations**: Track fresh oranges to avoid checking the entire grid; modify grid in-place.

## Complexity Analysis
- **Time Complexity**: O(m * n), as each cell is enqueued/dequeued at most once.
- **Space Complexity**: O(m * n) for the queue in the worst case (all cells are rotten).

## Best Practices
- Use clear variable names (e.g., `fresh`, `queue`).
- For Python, use `deque` and type hints.
- For JavaScript, use array for queue operations.
- For Java, use `LinkedList` and follow Google Java Style Guide.
- Track fresh oranges to optimize termination.

## Alternative Approaches
- **DFS**: Not suitable, as it doesn’t guarantee minimum time.
- **Simulation with Grid Copy**: Use a new grid to track changes (O(m * n) time, O(m * n) space). Less space-efficient.