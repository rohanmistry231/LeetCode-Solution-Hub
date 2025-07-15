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

### Java
```java
class Solution {
    public int trap(int[] height) {
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0, water = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                leftMax = Math.max(leftMax, height[left]);
                water += leftMax - height[left];
                left++;
            } else {
                rightMax = Math.max(rightMax, height[right]);
                water += rightMax - height[right];
                right--;
            }
        }
        return water;
    }
}
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