# Find Minimum in Rotated Sorted Array II

## Problem Statement
Given a sorted array `nums` rotated at some pivot, with possible duplicates, find the minimum element. The algorithm must run in O(log n) time in the average case.

**Example**:
- Input: `nums = [2,2,2,0,1]`
- Output: `0`

**Constraints**:
- `1 <= nums.length <= 5000`
- `-5000 <= nums[i] <= 5000`
- `nums` is sorted and rotated, with possible duplicates.

## Solution

### Java
```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                right--;
            }
        }
        
        return nums[left];
    }
}
```

## Reasoning
- **Approach**: Use binary search, comparing `nums[mid]` with `nums[right]`. If greater, minimum is in right half; if less, minimum is in left half including mid; if equal, exclude rightmost element due to duplicates. Converge to the minimum.
- **Why Binary Search?**: Handles rotation and duplicates, aiming for O(log n) average time, though worst case is O(n) with many duplicates.
- **Edge Cases**:
  - Single element: Return it.
  - All duplicates: Linear scan in worst case.
  - No rotation: Minimum at index 0.
- **Optimizations**: Use `right--` to handle duplicates; avoid overflow with midpoint calculation.

## Complexity Analysis
- **Time Complexity**: O(log n) average, O(n) worst case (all duplicates).
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints.
- For JavaScript, use `Math.floor` for integer division.
- For Java, follow Google Java Style Guide and use overflow-safe midpoint.
- Handle duplicates by excluding rightmost element.

## Alternative Approaches
- **Linear Scan**: Find minimum by iteration (O(n) time). Too slow.
- **Find Pivot**: Locate rotation point, then take next element (O(log n) average). Similar complexity.