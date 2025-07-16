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

### Python
```python
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r: int, c: int, ocean: set, prev_height: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in ocean or heights[r][c] < prev_height:
            return
        ocean.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc, ocean, heights[r][c])
    
    for c in range(cols):
        dfs(0, c, pacific, float('-inf'))  # Pacific: top row
        dfs(rows - 1, c, atlantic, float('-inf'))  # Atlantic: bottom row
    for r in range(rows):
        dfs(r, 0, pacific, float('-inf'))  # Pacific: left column
        dfs(r, cols - 1, atlantic, float('-inf'))  # Atlantic: right column
    
    return [[r, c] for (r, c) in pacific & atlantic]
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