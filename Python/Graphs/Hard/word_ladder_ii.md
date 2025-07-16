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

### Python
```python
from collections import defaultdict, deque

def findLadders(beginWord: str, endWord: str, wordList: list[str]) -> list[list[str]]:
    if endWord not in wordList:
        return []
    
    # Build graph
    wordList = set(wordList)
    graph = defaultdict(list)
    level = {beginWord: [[beginWord]]}
    visited = {beginWord}
    queue = deque([beginWord])
    
    while queue and endWord not in level:
        next_level = defaultdict(list)
        for _ in range(len(queue)):
            word = queue.popleft()
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                for j in range(26):
                    new_word = word[:i] + chr(97 + j) + word[i+1:]
                    if new_word in wordList and new_word not in visited:
                        visited.add(new_word)
                        queue.append(new_word)
                        next_level[new_word].extend(path + [new_word] for path in level[word])
                    elif new_word in wordList:
                        next_level[new_word].extend(path + [new_word] for path in level[word])
        level = next_level
    
    return level[endWord] if endWord in level else []
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