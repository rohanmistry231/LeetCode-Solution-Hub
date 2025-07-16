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

### Java
```java
import java.util.*;

class Solution {
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) return new ArrayList<>();
        
        Set<String> wordSet = new HashSet<>(wordList);
        Map<String, List<List<String>>> level = new HashMap<>();
        level.put(beginWord, new ArrayList<>(List.of(new ArrayList<>(List.of(beginWord)))));
        Set<String> visited = new HashSet<>(Set.of(beginWord));
        Queue<String> queue = new LinkedList<>(List.of(beginWord));
        
        while (!queue.isEmpty() && !level.containsKey(endWord)) {
            Map<String, List<List<String>>> nextLevel = new HashMap<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String word = queue.poll();
                for (int j = 0; j < word.length(); j++) {
                    for (char c = 'a'; c <= 'z'; c++) {
                        String newWord = word.substring(0, j) + c + word.substring(j + 1);
                        if (wordSet.contains(newWord)) {
                            if (!visited.contains(newWord)) {
                                visited.add(newWord);
                                queue.offer(newWord);
                                nextLevel.putIfAbsent(newWord, new ArrayList<>());
                            }
                            List<List<String>> paths = level.get(word);
                            for (List<String> path : paths) {
                                List<String> newPath = new ArrayList<>(path);
                                newPath.add(newWord);
                                nextLevel.computeIfAbsent(newWord, k -> new ArrayList<>()).add(newPath);
                            }
                        }
                    }
                }
            }
            level = nextLevel;
        }
        
        return level.getOrDefault(endWord, new ArrayList<>());
    }
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