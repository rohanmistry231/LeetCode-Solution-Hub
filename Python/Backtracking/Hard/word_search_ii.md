# Word Search II

## Problem Statement
Given an `m x n` board of characters and a list of strings `words`, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells (horizontally or vertically neighboring). The same letter cell may not be used more than once in a word.

**Example**:
- Input: `board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]`
- Output: `["eat","oath"]`

**Constraints**:
- `m == board.length`
- `n == board[i].length`
- `1 <= m, n <= 12`
- `board[i][j]` is a lowercase letter.
- `1 <= words.length <= 3 * 10^4`
- `1 <= words[i].length <= 10`
- `words[i]` consists of lowercase letters.
- All words are unique.

## Solution

### Python
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word
    
    result = []
    rows, cols = len(board), len(board[0])
    
    def backtrack(i: int, j: int, node: TrieNode) -> None:
        if i < 0 or i >= rows or j < 0 or j >= cols or board[i][j] not in node.children:
            return
        char = board[i][j]
        next_node = node.children[char]
        if next_node.word:
            result.append(next_node.word)
            next_node.word = None  # Avoid duplicates
        board[i][j] = '#'
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            backtrack(i + di, j + dj, next_node)
        board[i][j] = char
    
    for i in range(rows):
        for j in range(cols):
            backtrack(i, j, root)
    return result
```

## Reasoning
- **Approach**: Build a trie from the word list to efficiently check prefixes. Use backtracking to explore paths from each cell, following the trie structure. Mark visited cells to avoid reuse, and collect words when a trie node’s word is found. Clear word after adding to avoid duplicates.
- **Why Trie + Backtracking?**: Trie reduces search time for valid words, and backtracking explores all valid paths efficiently.
- **Edge Cases**:
  - Empty board or words: Return empty list.
  - Single cell: Check if it matches any word.
- **Optimizations**: Use trie to prune invalid paths; mark visited cells in-place; clear word after adding to avoid duplicates.

## Complexity Analysis
- **Time Complexity**: O(m * n * 4^L), where m and n are board dimensions, and L is the maximum word length. Each cell can start a path of length up to L.
- **Space Complexity**: O(W * L) for the trie, where W is the number of words, plus O(L) for the recursion stack, and O(W) for the output.

## Best Practices
- Use clear variable names (e.g., `node`, `children`).
- For Python, use type hints and a TrieNode class.
- For JavaScript, use class for TrieNode and modern loops.
- For Java, use array-based trie and follow Google Java Style Guide.
- Clear word in trie to avoid duplicates.

## Alternative Approaches
- **Backtracking without Trie**: Check each word individually (O(m * n * W * L) time). Slower for large word lists.
- **DFS with Set**: Use a set for words (O(m * n * 4^L) time, O(W) space). Less efficient for prefix matching.