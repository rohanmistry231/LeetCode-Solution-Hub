# Search in Rotated Sorted Array

## Problem Statement
Given an integer array `nums` sorted in ascending order, which is rotated at some pivot, and a target value, return the index of `target` if it exists, or `-1` if it does not. You must write an algorithm with O(log n) runtime complexity.

**Example**:
- Input: `nums = [4,5,6,7,0,1,2], target = 0`
- Output: `4`

**Constraints**:
- `1 <= nums.length <= 5000`
- `-10^4 <= nums[i], target <= 10^4`
- All values in `nums` are unique.
- `nums` is sorted and rotated at some pivot.

## Solution

### Java
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            
            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
}
```

## Reasoning
- **Approach**: Use binary search, adjusted for rotation. Check if the left or right half is sorted by comparing `nums[left]` with `nums[mid]`. If left half is sorted, check if `target` lies within it; otherwise, search the right half. Repeat until `target` is found or the search space is exhausted.
- **Why Binary Search?**: Maintains O(log n) time by halving the search space, handling rotation with conditional checks.
- **Edge Cases**:
  - Single element: Check directly.
  - Target not present: Return -1.
  - Fully sorted or fully rotated: Binary search still works.
- **Optimizations**: Use `left + (right - left) / 2` to avoid overflow; single pass with conditional logic.

## Complexity Analysis
- **Time Complexity**: O(log n), where n is the length of `nums`, as the search space halves each iteration.
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints for clarity.
- For JavaScript, use `Math.floor` for integer division.
- For Java, use overflow-safe midpoint calculation and follow Google Java Style Guide.
- Handle rotation with clear conditional checks.

## Alternative Approaches
- **Linear Search**: Check each element (O(n) time). Too slow for O(log n) requirement.
- **Find Pivot First**: Find rotation point, then binary search (O(log n) time). Two passes, less efficient.