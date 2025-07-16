# Keys and Rooms

## Problem Statement
There are `n` rooms labeled from 0 to `n-1`, and all rooms except 0 are locked. Each room `i` contains a list of keys `rooms[i]` to access other rooms. You start in room 0 and can move to any room for which you have a key. Return true if you can visit all rooms, false otherwise.

**Example**:
- Input: `rooms = [[1],[2],[3],[]]`
- Output: `true`
- Explanation: Start in room 0, use key to room 1, then 2, then 3.

**Constraints**:
- `n == rooms.length`
- `2 <= n <= 1000`
- `0 <= rooms[i].length <= 1000`
- `0 <= rooms[i][j] < n`
- All rooms[0] are distinct and rooms[i][j] != i.

## Solution

### Java
```java
import java.util.*;

class Solution {
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        boolean[] visited = new boolean[n];
        visited[0] = true;
        
        dfs(0, rooms, visited);
        for (boolean v : visited) {
            if (!v) return false;
        }
        return true;
    }
    
    private void dfs(int room, List<List<Integer>> rooms, boolean[] visited) {
        for (int key : rooms.get(room)) {
            if (!visited[key]) {
                visited[key] = true;
                dfs(key, rooms, visited);
            }
        }
    }
}
```

## Reasoning
- **Approach**: Model the problem as a directed graph where rooms are nodes and keys are edges. Start from room 0 and use DFS to mark all reachable rooms as visited. Check if all rooms are visited.
- **Why DFS?**: Explores all reachable rooms efficiently, handling the directed graph structure.
- **Edge Cases**:
  - Single room: Return true (room 0 is accessible).
  - No keys in room 0: Check if n == 1.
- **Optimizations**: Use a boolean array for visited rooms; modify in-place to save space.

## Complexity Analysis
- **Time Complexity**: O(n + k), where n is the number of rooms and k is the total number of keys, as each room and key is visited at most once.
- **Space Complexity**: O(n) for the visited array and recursion stack.

## Best Practices
- Use clear variable names (e.g., `visited`, `room`).
- For Python, use type hints and `all` for checking visited.
- For JavaScript, use `every` for array checking.
- For Java, use `List` interface and follow Google Java Style Guide.
- Mark room 0 as visited initially.

## Alternative Approaches
- **BFS**: Use a queue to explore rooms (O(n + k) time, O(n) space). Similar performance but iterative.
- **Union-Find**: Connect rooms with keys (O(k * α(n)) time). Overkill for this problem.