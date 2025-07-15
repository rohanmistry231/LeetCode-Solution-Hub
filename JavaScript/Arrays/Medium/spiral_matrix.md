# Spiral Matrix

## Problem Statement
Given an `m x n` matrix, return all elements in spiral order (clockwise starting from top-left).

**Example**:
- Input: `matrix = [[1,2,3],[4,5,6],[7,8,9]]`
- Output: `[1,2,3,6,9,8,7,4,5]`

**Constraints**:
- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 10`
- `-100 <= matrix[i][j] <= 100`

## Solution

### JavaScript
```javascript
function spiralOrder(matrix) {
    if (!matrix.length || !matrix[0].length) return [];
    const result = [];
    let top = 0, bottom = matrix.length - 1;
    let left = 0, right = matrix[0].length - 1;
    while (top <= bottom && left <= right) {
        for (let j = left; j <= right; j++) {
            result.push(matrix[top][j]);
        }
        top++;
        for (let i = top; i <= bottom; i++) {
            result.push(matrix[i][right]);
        }
        right--;
        if (top <= bottom) {
            for (let j = right; j >= left; j--) {
                result.push(matrix[bottom][j]);
            }
            bottom--;
        }
        if (left <= right) {
            for (let i = bottom; i >= top; i--) {
                result.push(matrix[i][left]);
            }
            left++;
        }
    }
    return result;
}
```

## Reasoning
- **Approach**: Use four pointers (top, bottom, left, right) to track boundaries. Traverse in spiral order (right, down, left, up), shrinking boundaries after each direction. Check conditions to avoid duplicate traversals.
- **Why Pointers?**: Boundaries ensure we cover all elements in the correct order without extra space.
- **Edge Cases**:
  - Empty matrix or single row/column: Handle explicitly.
  - Odd-sized matrix: Center element handled in final traversal.
- **Optimizations**: Single pass through all elements; conditional checks prevent over-traversal.

## Complexity Analysis
- **Time Complexity**: O(m * n), where m and n are matrix dimensions.
- **Space Complexity**: O(1), excluding the output array.

## Best Practices
- Use clear variable names (e.g., `top`, `bottom`).
- For Python, use type hints and empty checks.
- For JavaScript, use early return for empty cases.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Check boundaries to avoid duplicate traversals.

## Alternative Approaches
- **Layer-by-Layer**: Similar but less structured (O(m * n) time, O(1) space).
- **DFS/Recursion**: Complex and less efficient for this problem.