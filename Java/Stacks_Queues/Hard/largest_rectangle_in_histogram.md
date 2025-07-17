# Largest Rectangle in Histogram

## Problem Statement
Given an array of integers `heights` representing the histogram's bar heights where each bar has a width of 1, find the area of the largest rectangle in the histogram.

**Example**:
- Input: `heights = [2,1,5,6,2,3]`
- Output: `10`
- Explanation: The largest rectangle is formed by heights [5,6] with area 5 * 2 = 10.

**Constraints**:
- `1 <= heights.length <= 10^5`
- `0 <= heights[i] <= 10^4`

## Solution

### Java
```java
import java.util.*;

class Solution {
    public int largestRectangleArea(int[] heights) {
        Deque<Integer> stack = new ArrayDeque<>();
        int maxArea = 0;
        int[] extended = new int[heights.length + 1];
        System.arraycopy(heights, 0, extended, 0, heights.length);
        
        for (int i = 0; i < extended.length; i++) {
            while (!stack.isEmpty() && extended[i] < extended[stack.peek()]) {
                int h = extended[stack.pop()];
                int w = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, h * w);
            }
            stack.push(i);
        }
        
        return maxArea;
    }
}
```

## Reasoning
- **Approach**: Use a monotonic stack to maintain indices of increasing heights. For each bar, pop bars from the stack when a smaller height is encountered, calculating the rectangle area with the popped height and width (distance to current index or previous stack index). A sentinel value (0) ensures all bars are processed.
- **Why Monotonic Stack?**: Efficiently computes the largest rectangle by tracking the boundaries of each bar’s potential rectangle in one pass.
- **Edge Cases**:
  - Single bar: Area is height * 1.
  - All same height: Area is height * length.
  - Empty heights: Handled by constraints (length >= 1).
- **Optimizations**: Use sentinel to simplify logic; single pass through heights.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `heights`, as each index is pushed and popped at most once.
- **Space Complexity**: O(n), for the stack.

## Best Practices
- Use clear variable names (e.g., `stack`, `maxArea`).
- For Python, use type hints and list as stack.
- For JavaScript, use array as stack with `push`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Use sentinel value for cleaner code.

## Alternative Approaches
- **Brute Force**: For each bar, find left/right boundaries (O(n^2) time). Too slow.
- **Divide and Conquer**: Split at minimum height (O(n log n) average). More complex.