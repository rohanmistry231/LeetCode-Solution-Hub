# Sort Colors

## Problem Statement
Given an array `nums` with n objects colored 0, 1, or 2, sort them in-place so that objects of the same color are adjacent, with the colors in the order 0, 1, and 2.

**Example**:
- Input: `nums = [2,0,2,1,1,0]`
- Output: `[0,0,1,1,2,2]`

**Constraints**:
- `1 <= nums.length <= 300`
- `nums[i]` is 0, 1, or 2.

## Solution

### Java
```java
class Solution {
    public void sortColors(int[] nums) {
        int low = 0, mid = 0, high = nums.length - 1;
        
        while (mid <= high) {
            if (nums[mid] == 0) {
                int temp = nums[low];
                nums[low] = nums[mid];
                nums[mid] = temp;
                low++;
                mid++;
            } else if (nums[mid] == 1) {
                mid++;
            } else {
                int temp = nums[mid];
                nums[mid] = nums[high];
                nums[high] = temp;
                high--;
            }
        }
    }
}
```

## Reasoning
- **Approach**: Use the Dutch National Flag algorithm with three pointers: `low` for 0s, `mid` for 1s, and `high` for 2s. Swap elements to place 0s before `low`, 1s between `low` and `high`, and 2s after `high`. Move pointers based on `nums[mid]`.
- **Why Dutch National Flag?**: Sorts three distinct values in one pass with O(1) space.
- **Edge Cases**:
  - Single element: Already sorted.
  - All same value: Algorithm handles efficiently.
  - Small array: Works correctly.
- **Optimizations**: In-place sorting; single pass with three pointers.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`, as we traverse the array once.
- **Space Complexity**: O(1), as we sort in-place using only pointers.

## Best Practices
- Use clear variable names (e.g., `low`, `mid`, `high`).
- For Python, use type hints and tuple unpacking for swaps.
- For JavaScript, use array destructuring for swaps.
- For Java, follow Google Java Style Guide and use explicit swaps.
- Optimize with single-pass algorithm.

## Alternative Approaches
- **Counting Sort**: Count 0s, 1s, 2s, then rewrite array (O(n) time, O(1) space). Two passes, less elegant.
- **Standard Sort**: Use built-in sort (O(n log n) time). Overkill for three values.