# Trapping Rain Water

## Problem Statement
Given an array of non-negative integers `height` representing an elevation map where each bar has a width of 1, compute how much water it can trap after raining.

**Example**:
- Input: `height = [0,1,0,2,1,0,1,3,2,1,2,1]`
- Output: `6`
- Explanation: The elevation map traps 6 units of water.

**Constraints**:
- `1 <= height.length <= 2 * 10^4`
- `0 <= height[i] <= 10^5`

## Solution

### Java
```java
import java.util.*;

class Solution {
    public int trap(int[] height) {
        Deque<Integer> stack = new ArrayDeque<>();
        int water = 0;
        
        for (int i = 0; i < height.length; i++) {
            while (!stack.isEmpty() && height[i] > height[stack.peek()]) {
                int bottom = stack.pop();
                if (stack.isEmpty()) break;
                int left = stack.peek();
                int h = Math.min(height[left], height[i]) - height[bottom];
                int w = i - left - 1;
                water += h * w;
            }
            stack.push(i);
        }
        
        return water;
    }
}
```

## Reasoning
- **Approach**: Use a monotonic stack to maintain indices of decreasing heights. For each bar, pop bars when a taller bar is found, calculating trapped water using the height difference and width between the current bar and the previous stack bar. The stack tracks left boundaries.
- **Why Monotonic Stack?**: Efficiently identifies trapped water regions by maintaining a decreasing height order, processing each bar in one pass.
- **Edge Cases**:
  - Single bar: No water (0).
  - Monotonic array: No water trapped.
  - Empty array: Handled by constraints (length >= 1).
- **Optimizations**: Single pass; use stack to track boundaries dynamically.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `height`, as each index is pushed and popped at most once.
- **Space Complexity**: O(n), for the stack.

## Best Practices
- Use clear variable names (e.g., `stack`, `water`).
- For Python, use type hints and list as stack.
- For JavaScript, use array as stack with `push`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Calculate water efficiently with stack.

## Alternative Approaches
- **Two Pointers**: Use left/right pointers to track boundaries (O(n) time, O(1) space). Equally efficient but different approach.
- **Precompute Maxes**: Store left/right max heights (O(n) time, O(n) space). More memory-intensive.