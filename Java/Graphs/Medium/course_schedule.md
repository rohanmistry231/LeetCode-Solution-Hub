# Course Schedule

## Problem Statement
There are `numCourses` courses labeled from 0 to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [ai, bi]` indicates that you must take course `bi` first before taking course `ai`. Return `true` if you can finish all courses, otherwise `false`.

**Example**:
- Input: `numCourses = 2, prerequisites = [[1,0]]`
- Output: `true`
- Explanation: Take course 0, then course 1.

**Constraints**:
- `1 <= numCourses <= 2000`
- `0 <= prerequisites.length <= 5000`
- `prerequisites[i].length == 2`
- `0 <= ai, bi < numCourses`
- All pairs are unique.

## Solution

### Java
```java
import java.util.*;

class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<Integer>[] graph = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) graph[i] = new ArrayList<>();
        for (int[] prereq : prerequisites) {
            graph[prereq[0]].add(prereq[1]);
        }
        
        int[] visited = new int[numCourses]; // 0: unvisited, 1: visiting, 2: visited
        for (int course = 0; course < numCourses; course++) {
            if (visited[course] == 0 && !dfs(course, graph, visited)) {
                return false;
            }
        }
        return true;
    }
    
    private boolean dfs(int course, List<Integer>[] graph, int[] visited) {
        if (visited[course] == 1) return false;
        if (visited[course] == 2) return true;
        visited[course] = 1;
        for (int prereq : graph[course]) {
            if (!dfs(prereq, graph, visited)) return false;
        }
        visited[course] = 2;
        return true;
    }
}
```

## Reasoning
- **Approach**: Build a directed graph using an adjacency list where each course points to its prerequisites. Use DFS to detect cycles, marking nodes as unvisited (0), visiting (1), or visited (2). A cycle indicates that courses cannot be completed.
- **Why DFS?**: Detects cycles in a directed graph, ensuring all courses can be taken without circular dependencies.
- **Edge Cases**:
  - No prerequisites: Return true.
  - Single course: Return true.
- **Optimizations**: Use a single visited array to track states; explore only unvisited nodes.

## Complexity Analysis
- **Time Complexity**: O(V + E), where V is `numCourses` and E is the number of prerequisites, as DFS visits each node and edge once.
- **Space Complexity**: O(V + E) for the adjacency list and O(V) for the visited array and recursion stack.

## Best Practices
- Use clear variable names (e.g., `graph`, `visited`).
- For Python, use type hints and list comprehension.
- For JavaScript, use array methods for initialization.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Use state-based visited array for cycle detection.

## Alternative Approaches
- **Topological Sort (Kahn’s Algorithm)**: Use BFS with in-degree counting (O(V + E) time, O(V + E) space). Similar performance.
- **Union-Find**: Not suitable due to directed graph nature.