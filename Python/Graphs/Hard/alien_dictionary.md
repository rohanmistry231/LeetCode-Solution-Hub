# Alien Dictionary

## Problem Statement
Given a list of strings `words` from an alien language, where words are sorted lexicographically by the rules of this language, derive the order of letters in this language. If the order is invalid, return an empty string. If multiple valid orders exist, return any one.

**Example**:
- Input: `words = ["wrt","wrf","er","ett","rftt"]`
- Output: `"wertf"`
- Explanation: The order of letters is w < e < r < t < f.

**Constraints**:
- `1 <= words.length <= 100`
- `1 <= words[i].length <= 20`
- `words[i]` consists of lowercase English letters.
- All strings in `words` are unique.

## Solution

### Python
```python
from collections import defaultdict, deque

def alienOrder(words: list[str]) -> str:
    # Build graph and in-degree
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}
    for w1, w2 in zip(words, words[1:]):
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break
        else:
            if len(w1) > len(w2):
                return ""
    
    # Topological sort with BFS
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycle or incomplete order
    return "".join(result) if len(result) == len(in_degree) else ""
```

## Reasoning
- **Approach**: Build a directed graph of character dependencies by comparing adjacent words. Use topological sort (BFS) to determine the order. Track in-degrees to find characters with no dependencies. Check for invalid cases (e.g., longer word before shorter word with same prefix) and cycles.
- **Why Topological Sort?**: Determines the order of characters based on their dependencies, handling partial orders efficiently.
- **Edge Cases**:
  - Single word: Return its characters.
  - Invalid order (e.g., "abc" before "ab"): Return empty string.
  - Cycle in graph: Return empty string.
- **Optimizations**: Use path compression in graph building; early validation of word length.

## Complexity Analysis
- **Time Complexity**: O(C + N), where C is the total number of characters across all words, and N is the number of edges in the graph (from comparing words).
- **Space Complexity**: O(K), where K is the number of unique characters (up to 26), for the graph and in-degree map.

## Best Practices
- Use clear variable names (e.g., `graph`, `inDegree`).
- For Python, use `defaultdict` and `deque`.
- For JavaScript, use `Map` and `Set` for graph representation.
- For Java, use `HashMap` and `HashSet`, follow Google Java Style Guide.
- Validate word order early to avoid unnecessary processing.

## Alternative Approaches
- **DFS for Topological Sort**: Detect cycles and build order (O(C + N) time, O(K) space). Similar complexity but recursive.
- **Brute Force**: Try all permutations (O(K!) time). Infeasible for large character sets.