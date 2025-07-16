# Word Ladder II

## Problem Statement
Given two words `beginWord` and `endWord`, and a dictionary `wordList`, return all shortest transformation sequences from `beginWord` to `endWord`, where each transformation changes exactly one letter, and the new word exists in `wordList`.

**Example**:
- Input: `beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]`
- Output: `[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]`

**Constraints**:
- `1 <= beginWord.length <= 10`
- `endWord.length == beginWord.length`
- `1 <= wordList.length <= 1000`
- `wordList[i].length == beginWord.length`
- All words consist of lowercase English letters.
- All words in `wordList` are unique.

## Solution

### JavaScript
```javascript
function findLadders(beginWord, endWord, wordList) {
    if (!wordList.includes(endWord)) return [];
    
    const wordSet = new Set(wordList);
    const graph = new Map();
    const level = new Map([[beginWord, [[beginWord]]]]);
    const visited = new Set([beginWord]);
    const queue = [beginWord];
    
    while (queue.length && !level.has(endWord)) {
        const nextLevel = new Map();
        const size = queue.length;
        for (let i = 0; i < size; i++) {
            const word = queue.shift();
            for (let j = 0; j < word.length; j++) {
                const pattern = word.slice(0, j) + '*' + word.slice(j + 1);
                for (let k = 0; k < 26; k++) {
                    const newWord = word.slice(0, j) + String.fromCharCode(97 + k) + word.slice(j + 1);
                    if (wordSet.has(newWord)) {
                        if (!visited.has(newWord)) {
                            visited.add(newWord);
                            queue.push(newWord);
                            nextLevel.set(newWord, []);
                        }
                        const paths = level.get(word);
                        for (const path of paths) {
                            nextLevel.get(newWord).push([...path, newWord]);
                        }
                    }
                }
            }
        }
        level.clear();
        for (const [word, paths] of nextLevel) {
            level.set(word, paths);
        }
    }
    
    return level.get(endWord) || [];
}
```

## Reasoning
- **Approach**: Use BFS to find all shortest paths from `beginWord` to `endWord`. Build a graph by generating all possible one-letter transformations. Track paths level by level to ensure shortest sequences. Use a visited set to avoid cycles and a level map to store all paths to each word.
- **Why BFS?**: Guarantees shortest paths by exploring level by level, collecting all valid sequences.
- **Edge Cases**:
  - `endWord` not in `wordList`: Return empty list.
  - No path exists: Return empty list.
  - Single word: Check if it matches `endWord`.
- **Optimizations**: Use a set for O(1) word lookup; generate patterns efficiently.

## Complexity Analysis
- **Time Complexity**: O(N * 26 * L * P), where N is the length of `wordList`, L is the length of each word, and P is the number of paths. Each word generates 26*L transformations.
- **Space Complexity**: O(N * P) for storing paths in the level map and queue.

## Best Practices
- Use clear variable names (e.g., `level`, `visited`).
- For Python, use `defaultdict`, `deque`, and type hints.
- For JavaScript, use `Map` for path storage.
- For Java, use `HashMap` and `ArrayList`, follow Google Java Style Guide.
- Use set for fast word lookup.

## Alternative Approaches
- **DFS with Backtracking**: Find paths but may not guarantee shortest (O(N!) time). Infeasible.
- **Bidirectional BFS**: Search from both ends (potentially faster but complex). O(N * 26 * L) time.