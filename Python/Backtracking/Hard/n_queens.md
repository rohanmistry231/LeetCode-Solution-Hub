# N-Queens

## Problem Statement
The n-queens puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens threaten each other. Return all distinct solutions to the n-queens puzzle. Each solution contains a distinct board configuration, where 'Q' indicates a queen and '.' indicates an empty space.

**Example**:
- Input: `n = 4`
- Output: `[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]`

**Constraints**:
- `1 <= n <= 9`

## Solution

### Python
```python
def solveNQueens(n: int) -> list[list[str]]:
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def is_safe(row: int, col: int, queens: set) -> bool:
        for r, c in queens:
            if row == r or col == c or abs(row - r) == abs(col - c):
                return False
        return True
    
    def backtrack(row: int, queens: set) -> None:
        if row == n:
            result.append([''.join(row) for row in board])
            return
        for col in range(n):
            if is_safe(row, col, queens):
                board[row][col] = 'Q'
                queens.add((row, col))
                backtrack(row + 1, queens)
                board[row][col] = '.'
                queens.remove((row, col))
    
    backtrack(0, set())
    return result
```

## Reasoning
- **Approach**: Use backtracking to place queens row by row. Check if a position is safe (no conflicts in row, column, or diagonals). If safe, place a queen, recurse to the next row, and backtrack by removing the queen. Collect valid board configurations when all queens are placed.
- **Why Backtracking?**: It systematically explores all possible placements while pruning invalid configurations.
- **Edge Cases**:
  - `n = 1`: Returns `[["Q"]]`.
  - `n = 2` or `3`: No solutions possible.
- **Optimizations**: Use a set to track queen positions; check safety efficiently by avoiding redundant calculations.

## Complexity Analysis
- **Time Complexity**: O(n!), as it explores all possible queen placements, reduced by pruning.
- **Space Complexity**: O(n) for the recursion stack and queen set, plus O(n^2) for the board and O(n! * n^2) for the output.

## Best Practices
- Use clear variable names (e.g., `queens`, `board`).
- For Python, use type hints and set for queen positions.
- For JavaScript, use Set and array methods for clarity.
- For Java, use `HashSet` and follow Google Java Style Guide.
- Check safety efficiently to reduce redundant work.

## Alternative Approaches
- **Bit Manipulation**: Use bitsets for rows, columns, and diagonals (O(n!) time, O(n) space). More complex.
- **Iterative**: Generate solutions iteratively (same complexity). Less intuitive.