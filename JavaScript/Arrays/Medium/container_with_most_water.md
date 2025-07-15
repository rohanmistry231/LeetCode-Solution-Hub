# Container With Most Water

## Problem Statement
Given an integer array `height` of length `n`, find two lines that, together with the x-axis, form a container with the most water. Return the maximum area.

**Example**:
- Input: `height = [1,8,6,2,5,4,8,3,7]`
- Output: `49`
- Explanation: The container formed by indices 1 and 8 (heights 8 and 7) has area `7 * (8-1) = 49`.

**Constraints**:
- `n == height.length`
- `2 <= n <= 10^5`
- `0 <= height[i] <= 10^4`

## Solution

### JavaScript
```javascript
function maxArea(height) {
    let left = 0, right = height.length - 1;
    let maxArea = 0;
    while (left < right) {
        const width = right - left;
        maxArea = Math.max(maxArea, Math.min(height[left], height[right]) * width);
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    return maxArea;
}
```

## Reasoning
- **Approach**: Use two pointers (left, right) starting at array ends. Compute area as `min(height[left], height[right]) * (right - left)`. Move the pointer with the smaller height inward to maximize potential area.
- **Why Two Pointers?**: Moving the smaller height maximizes area potential, as width decreases but height might increase.
- **Edge Cases**:
  - Two elements: Compute single area.
  - All equal heights: Handled by moving either pointer.
- **Optimizations**: Single pass with two pointers; no extra space needed.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `height`. Single pass with two pointers.
- **Space Complexity**: O(1), using only constant space.

## Best Practices
- Use clear variable names (e.g., `left`, `right`, `max_area`).
- For Python, use type hints for clarity.
- For JavaScript, use `Math.min/max` for readability.
- For Java, follow Google Java Style Guide.
- Move smaller height to optimize area calculation.

## Alternative Approaches
- **Brute Force**: Check all pairs of lines (O(n²) time). Inefficient for large inputs.
- **Greedy with Sorting**: Sort heights and try combinations (O(n log n) time). Less efficient.