# Find First and Last Position of Element in Sorted Array

## Problem Statement
Given an array of integers `nums` sorted in ascending order, find the starting and ending position of a given `target` value. If `target` is not found, return `[-1, -1]`. The algorithm must run in O(log n) time.

**Example**:
- Input: `nums = [5,7,7,8,8,10], target = 8`
- Output: `[3,4]`

**Constraints**:
- `0 <= nums.length <= 10^5`
- `-10^9 <= nums[i], target <= 10^9`
- `nums` is sorted in ascending order.

## Solution

### Python
```python
def searchRange(nums: list[int], target: int) -> list[int]:
    def binarySearch(left_bias: bool) -> int:
        left, right = 0, len(nums) - 1
        i = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                i = mid
                if left_bias:
                    right = mid - 1
                else:
                    left = mid + 1
        return i
    
    return [binarySearch(True), binarySearch(False)]
```

## Reasoning
- **Approach**: Use binary search twice: once to find the leftmost occurrence (bias left) and once for the rightmost (bias right). When `target` is found, continue searching left or right to find the boundary. Return `[-1, -1]` if not found.
- **Why Binary Search?**: Achieves O(log n) time by halving the search space, suitable for finding boundaries in a sorted array.
- **Edge Cases**:
  - Empty array: Return `[-1, -1]`.
  - Target not found: Return `[-1, -1]`.
  - Single occurrence: Return same index for both.
- **Optimizations**: Reuse binary search with a bias parameter; avoid overflow with midpoint calculation.

## Complexity Analysis
- **Time Complexity**: O(log n), as each binary search takes O(log n) time, and we perform two searches.
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `leftBias`, `i`).
- For Python, use type hints and helper function.
- For JavaScript, use boolean parameter for bias.
- For Java, follow Google Java Style Guide and use array for result.
- Modularize binary search for reuse.

## Alternative Approaches
- **Linear Scan**: Find first and last linearly (O(n) time). Too slow for O(log n) requirement.
- **Single Binary Search with Expansion**: Find any occurrence, then expand (O(n) worst case). Inefficient.