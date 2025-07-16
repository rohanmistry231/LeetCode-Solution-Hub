# Word Search

## Problem Statement
Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid. The word must be constructed from letters of sequentially adjacent cells (horizontally or vertically neighboring). The same letter cell may not be used more than once.

**Example**:
- Input: `board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"`
- Output: `true`

**Constraints**:
- `m == board.length`
- `n = board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- `board` and `word` consist of only uppercase and lowercase English letters.

## Solution

### Java
```java
class Solution {
    public boolean exist(char[][] board, String word) {
        int rows = board.length, cols = board[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == word.charAt(0) && backtrack(board, word, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }
    
    private boolean backtrack(char[][] board, String word, int i, int j, int k) {
        if (k == word.length()) return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(k)) {
            return false;
        }
        char temp = board[i][j];
        board[i][j] = '#';
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] dir : directions) {
            if (backtrack(board, word, i + dir[0], j + dir[1], k + 1)) {
                return true;
            }
        }
        board[i][j] = temp;
        return false;
    }
}
```

## Reasoning
- **Approach**: Use backtracking to explore all possible paths from each cell. Mark visited cells to avoid reuse. Check all four directions for the next character in the word. Restore the board after each exploration.
- **Why Backtracking?**: It explores all possible paths while pruning invalid ones (e.g., out-of-bounds or mismatched characters).
- **Edge Cases**:
  - Single cell: Check if it matches the word.
  - Word not found: Return false.
  - Empty board or word: Handle appropriately.
- **Optimizations**: Mark cells with a temporary character (`#`) to avoid extra space; start from cells matching the first character.

## Complexity Analysis
- **Time Complexity**: O(m * n * 4^L), where m and n are board dimensions, and L is the word length. Each cell can branch into 4 directions.
- **Space Complexity**: O(L) for the recursion stack.

## Best Practices
- Use clear variable names (e.g., `i`, `j`, `k` for coordinates and word index).
- For Python, use type hints and direction tuples.
- For JavaScript, use array destructuring for directions.
- For Java, use 2D array for directions and follow Google Java Style Guide.
- Mark and restore cells to avoid extra space.

## Alternative Approaches
- **DFS with Extra Space**: Use a visited set (O(m * n * L) time, O(m * n) space). Uses more space.
- **Brute Force**: Check all paths without pruning (exponential time). Inefficient.