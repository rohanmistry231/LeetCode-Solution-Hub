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

### JavaScript
```javascript
class TrieNode {
    constructor() {
        this.children = {};
        this.word = null;
    }
}

function findWords(board, words) {
    const root = new TrieNode();
    for (const word of words) {
        let node = root;
        for (const char of word) {
            if (!node.children[char]) {
                node.children[char] = new TrieNode();
            }
            node = node.children[char];
        }
        node.word = word;
    }
    
    const result = [], rows = board.length, cols = board[0].length;
    
    function backtrack(i, j, node) {
        if (i < 0 || i >= rows || j < 0 || j >= cols || !node.children[board[i][j]]) return;
        const char = board[i][j];
        const nextNode = node.children[char];
        if (nextNode.word) {
            result.push(nextNode.word);
            nextNode.word = null;
        }
        board[i][j] = '#';
        for (const [di, dj] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
            backtrack(i + di, j + dj, nextNode);
        }
        board[i][j] = char;
    }
    
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            backtrack(i, j, root);
        }
    }
    return result;
}
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