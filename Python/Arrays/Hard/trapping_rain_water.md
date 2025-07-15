# Trapping Rain Water

## Problem Statement
Given an array `height` of non-negative integers representing an elevation map, compute how much water it can trap after raining.

**Example**:
- Input: `height = [0,1,0,2,1,0,1,3,2,1,2,1]`
- Output: `6`
- Explanation: The elevation map traps 6 units of water.

**Constraints**:
- `n == height.length`
- `1 <= n <= 2 * 10^4`
- `0 <= height[i] <= 10^5`

## Solution

### Python
```python
def trap(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    return water
```

## Reasoning
- **Approach**: Use two pointers to track maximum heights from left and right. Water trapped at each position is the minimum of `left_max` and `right_max` minus the current height. Move the pointer from the smaller height side to ensure correct water calculation.
- **Why Two Pointers?**: Avoids computing max heights for each position separately, achieving O(n) time with O(1) space.
- **Edge Cases**:
  - Single element: No water trapped.
  - Monotonic array: No water trapped.
- **Optimizations**: Single pass with two pointers; no extra arrays needed.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `height`. Single pass with two pointers.
- **Space Complexity**: O(1), using only constant space.

## Best Practices
- Use clear variable names (e.g., `left_max`, `right_max`).
- For Python, use type hints for clarity.
- For JavaScript, use `Math.max` for readability.
- For Java, follow Google Java Style Guide.
- Process smaller height side to optimize water calculation.

## Alternative Approaches
- **Two Arrays**: Store left and right max heights (O(n) time, O(n) space). Uses more space.
- **Brute Force**: Compute min(left_max, right_max) for each index (O(n²) time). Inefficient.