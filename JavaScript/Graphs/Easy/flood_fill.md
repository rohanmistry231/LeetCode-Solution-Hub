# Flood Fill

## Problem Statement
An image is represented by an `m x n` integer grid `image` where `image[i][j]` represents the pixel value. Given a starting pixel `(sr, sc)`, a new color `newColor`, and the original color, replace the color of all connected pixels (4-directionally) with the same original color as `(sr, sc)` with `newColor`.

**Example**:
- Input: `image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2`
- Output: `[[2,2,2],[2,2,0],[2,0,1]]`
- Explanation: The connected pixels with color 1 are changed to 2.

**Constraints**:
- `m == image.length`
- `n == image[i].length`
- `1 <= m, n <= 50`
- `0 <= image[i][j], newColor <= 65535`
- `0 <= sr < m`
- `0 <= sc < n`

## Solution

### JavaScript
```javascript
function floodFill(image, sr, sc, newColor) {
    if (image[sr][sc] === newColor) return image;
    const oldColor = image[sr][sc];
    const rows = image.length, cols = image[0].length;
    
    function dfs(r, c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || image[r][c] !== oldColor) return;
        image[r][c] = newColor;
        for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
            dfs(r + dr, c + dc);
        }
    }
    
    dfs(sr, sc);
    return image;
}
```

## Reasoning
- **Approach**: Use depth-first search (DFS) to explore all 4-directionally connected pixels with the same `oldColor`. Replace each with `newColor`. If the starting pixel is already `newColor`, return the image unchanged. Check boundaries and color match to avoid infinite recursion.
- **Why DFS?**: Naturally explores connected components in a grid, ensuring all relevant pixels are updated.
- **Edge Cases**:
  - Starting pixel is `newColor`: Return unchanged image.
  - Single pixel: Change only that pixel if needed.
- **Optimizations**: Modify image in-place; early return if `newColor` matches `oldColor`.

## Complexity Analysis
- **Time Complexity**: O(m * n), where m and n are the dimensions of the image, as each pixel is visited at most once.
- **Space Complexity**: O(m * n) for the recursion stack in the worst case (entire grid is one color).

## Best Practices
- Use clear variable names (e.g., `oldColor`, `newColor`).
- For Python, use type hints and direction tuples.
- For JavaScript, use array destructuring for directions.
- For Java, use 2D array for directions and follow Google Java Style Guide.
- Check for same-color early to avoid unnecessary DFS.

## Alternative Approaches
- **BFS**: Use a queue to explore pixels (O(m * n) time, O(m * n) space). Similar performance but iterative.
- **Iterative with Stack**: Simulate DFS with a stack (O(m * n) time, O(m * n) space). More complex.