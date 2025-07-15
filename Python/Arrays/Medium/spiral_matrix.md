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

### Python
```python
def spiralOrder(matrix: list[list[int]]) -> list[int]:
    if not matrix or not matrix[0]:
        return []
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        if top <= bottom:
            # Traverse left
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        if left <= right:
            # Traverse up
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    return result
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