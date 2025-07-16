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

### Java
```java
import java.util.ArrayList;
import java.util.List;

class TrieNode {
    TrieNode[] children = new TrieNode[26];
    String word;
}

class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new TrieNode();
                }
                node = node.children[index];
            }
            node.word = word;
        }
        
        List<String> result = new ArrayList<>();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                backtrack(board, i, j, root, result);
            }
        }
        return result;
    }
    
    private void backtrack(char[][] board, int i, int j, TrieNode node, List<String> result) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] == '#') return;
        char c = board[i][j];
        TrieNode nextNode = node.children[c - 'a'];
        if (nextNode == null) return;
        if (nextNode.word != null) {
            result.add(nextNode.word);
            nextNode.word = null;
        }
        board[i][j] = '#';
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] dir : directions) {
            backtrack(board, i + dir[0], j + dir[1], nextNode, result);
        }
        board[i][j] = c;
    }
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