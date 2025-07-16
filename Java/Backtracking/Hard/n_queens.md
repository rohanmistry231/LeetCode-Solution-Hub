# N-Queens

## Problem Statement
The n-queens puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens threaten each other. Return all distinct solutions to the n-queens puzzle. Each solution contains a distinct board configuration, where 'Q' indicates a queen and '.' indicates an empty space.

**Example**:
- Input: `n = 4`
- Output: `[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]`

**Constraints**:
- `1 <= n <= 9`

## Solution

### Java
```java
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        backtrack(0, n, new HashSet<>(), board, result);
        return result;
    }
    
    private void backtrack(int row, int n, Set<int[]> queens, char[][] board, List<List<String>> result) {
        if (row == n) {
            List<String> solution = new ArrayList<>();
            for (char[] r : board) {
                solution.add(new String(r));
            }
            result.add(solution);
            return;
        }
        for (int col = 0; col < n; col++) {
            if (isSafe(row, col, queens)) {
                board[row][col] = 'Q';
                queens.add(new int[]{row, col});
                backtrack(row + 1, n, queens, board, result);
                board[row][col] = '.';
                queens.remove(new int[]{row, col});
            }
        }
    }
    
    private boolean isSafe(int row, int col, Set<int[]> queens) {
        for (int[] queen : queens) {
            int r = queen[0], c = queen[1];
            if (row == r || col == c || Math.abs(row - r) == Math.abs(col - c)) {
                return false;
            }
        }
        return true;
    }
}
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