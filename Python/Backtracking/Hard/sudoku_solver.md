# Sudoku Solver

## Problem Statement
Write a program to solve a Sudoku puzzle by filling the empty cells. A Sudoku solution must satisfy all standard rules: each row, column, and 3x3 sub-box must contain digits 1-9 without repetition. Empty cells are indicated by '.'.

**Example**:
- Input: `board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]`
- Output: `[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]`

**Constraints**:
- `board.length == 9`
- `board[i].length == 9`
- `board[i][j]` is a digit 1-9 or '.'.

## Solution

### Python
```python
def solveSudoku(board: list[list[str]]) -> None:
    def is_valid(row: int, col: int, num: str) -> bool:
        for i in range(9):
            if board[row][i] == num or board[i][col] == num or board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
                return False
        return True
    
    def backtrack() -> bool:
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = '.'
                    return False
        return True
    
    backtrack()
```

## Reasoning
- **Approach**: Use backtracking to try placing digits 1-9 in each empty cell. Check if the digit is valid in the row, column, and 3x3 sub-box. If valid, place the digit, recurse, and backtrack if no solution is found. Stop when the board is filled.
- **Why Backtracking?**: It explores all possible digit placements while pruning invalid ones.
- **Edge Cases**:
  - Single empty cell: Try all digits.
  - Invalid board: Backtracking ensures a valid solution or failure.
- **Optimizations**: Check validity before recursion; modify board in-place to avoid extra space.

## Complexity Analysis
- **Time Complexity**: O(9^(m)), where m is the number of empty cells, as each empty cell can try up to 9 digits.
- **Space Complexity**: O(1) for the recursion stack (since board is modified in-place), excluding input/output.

## Best Practices
- Use clear variable names (e.g., `row`, `col`, `num`).
- For Python, use type hints and string iteration.
- For JavaScript, use modern loops and math for sub-box indices.
- For Java, follow Google Java Style Guide and use char for digits.
- Validate digits efficiently to prune branches.

## Alternative Approaches
- **Constraint Propagation**: Use dancing links or constraint satisfaction (complex, same worst-case time). More complex.
- **Bit Manipulation**: Track valid digits with bitsets (O(9^(m)) time, O(1) space). Harder to implement.