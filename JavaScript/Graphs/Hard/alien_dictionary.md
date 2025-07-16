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

### JavaScript
```javascript
function alienOrder(words) {
    const graph = new Map();
    const inDegree = new Map();
    
    // Initialize in-degree for all characters
    for (const word of words) {
        for (const c of word) {
            if (!inDegree.has(c)) inDegree.set(c, 0);
        }
    }
    
    // Build graph
    for (let i = 0; i < words.length - 1; i++) {
        const w1 = words[i], w2 = words[i + 1];
        for (let j = 0; j < Math.min(w1.length, w2.length); j++) {
            if (w1[j] !== w2[j]) {
                if (!graph.has(w1[j])) graph.set(w1[j], new Set());
                if (!graph.get(w1[j]).has(w2[j])) {
                    graph.get(w1[j]).add(w2[j]);
                    inDegree.set(w2[j], (inDegree.get(w2[j]) || 0) + 1);
                }
                break;
            }
            if (j === w2.length - 1 && w1.length > w2.length) return "";
        }
    }
    
    // Topological sort with BFS
    const queue = [];
    for (const [c, deg] of inDegree) {
        if (deg === 0) queue.push(c);
    }
    
    const result = [];
    while (queue.length) {
        const char = queue.shift();
        result.push(char);
        if (graph.has(char)) {
            for (const neighbor of graph.get(char)) {
                inDegree.set(neighbor, inDegree.get(neighbor) - 1);
                if (inDegree.get(neighbor) === 0) queue.push(neighbor);
            }
        }
    }
    
    return result.length === inDegree.size ? result.join("") : "";
}
```

### Java
```java
import java.util.*;

class Solution {
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> graph = new HashMap<>();
        Map<Character, Integer> inDegree = new HashMap<>();
        
        // Initialize in-degree
        for (String word : words) {
            for (char c : word.toCharArray()) {
                inDegree.putIfAbsent(c, 0);
            }
        }
        
        // Build graph
        for (int i = 0; i < words.length - 1; i++) {
            String w1 = words[i], w2 = words[i + 1];
            for (int j = 0; j < Math.min(w1.length(), w2.length()); j++) {
                if (w1.charAt(j) != w2.charAt(j)) {
                    graph.computeIfAbsent(w1.charAt(j), k -> new HashSet<>()).add(w2.charAt(j));
                    inDegree.merge(w2.charAt(j), 1, Integer::sum);
                    break;
                }
                if (j == w2.length() - 1 && w1.length() > w2.length()) return "";
            }
        }
        
        // Topological sort with BFS
        Queue<Character> queue = new LinkedList<>();
        for (Map.Entry<Character, Integer> entry : inDegree.entrySet()) {
            if (entry.getValue() == 0) queue.offer(entry.getKey());
        }
        
        StringBuilder result = new StringBuilder();
        while (!queue.isEmpty()) {
            char c = queue.poll();
            result.append(c);
            if (graph.containsKey(c)) {
                for (char neighbor : graph.get(c)) {
                    inDegree.merge(neighbor, -1, Integer::sum);
                    if (inDegree.get(neighbor) == 0) queue.offer(neighbor);
                }
            }
        }
        
        return result.length() == inDegree.size() ? result.toString() : "";
    }
}
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