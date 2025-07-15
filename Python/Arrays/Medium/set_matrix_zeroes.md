# Set Matrix Zeroes

## Problem Statement
Given an `m x n` integer matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

**Example**:
- Input: `matrix = [[1,1,1],[1,0,1],[1,1,1]]`
- Output: `[[1,0,1],[0,0,0],[1,0,1]]`

**Constraints**:
- `m == matrix.length`
- `n == matrix[0].length`
- `1 <= m, n <= 200`
- `-2^31 <= matrix[i][j] <= 2^31 - 1`

## Solution

### Python
```python
def setZeroes(matrix: list[list[int]]) -> None:
    m, n = len(matrix), len(matrix[0])
    first_row_zero = False
    first_col_zero = False
    # Check if first row or column needs to be zeroed
    for j in range(n):
        if matrix[0][j] == 0:
            first_row_zero = True
    for i in range(m):
        if matrix[i][0] == 0:
            first_col_zero = True
    # Use first row and column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    # Set rows to zero based on first column
    for i in range(1, m):
        if matrix[i][0] == 0:
            for j in range(1, n):
                matrix[i][j] = 0
    # Set columns to zero based on first row
    for j in range(1, n):
        if matrix[0][j] == 0:
            for i in range(1, m):
                matrix[i][j] = 0
    # Set first row and column if needed
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
```

## Reasoning
- **Approach**: Use the first row and column as markers to indicate which rows/columns need to be zeroed. Store flags for first row/column zeros, then mark zeros in the rest of the matrix. Finally, set rows and columns to zero based on markers.
- **Why Markers?**: Avoids extra space by using the matrix itself for storage.
- **Edge Cases**:
  - Single row/column: Handled by flags.
  - All zeros: Correctly sets entire matrix to zero.
- **Optimizations**: O(1) space by reusing matrix; separate checks for first row/column to avoid conflicts.

## Complexity Analysis
- **Time Complexity**: O(m * n), where m and n are matrix dimensions.
- **Space Complexity**: O(1), using only constant extra space.

## Best Practices
- Use clear variable names (e.g., `first_row_zero`).
- For Python, use type hints and clear loop structures.
- For JavaScript, use concise variable declarations.
- For Java, follow Google Java Style Guide.
- Handle first row/column separately to avoid overwriting markers.

## Alternative Approaches
- **Extra Space**: Use separate arrays for row/column flags (O(m + n) space).
- **Brute Force**: Copy matrix and modify (O(m * n) space). Not in-place.