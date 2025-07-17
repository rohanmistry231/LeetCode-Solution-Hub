# Binary Search

## Problem Statement
Given an array of integers `nums` sorted in ascending order and an integer `target`, return the index of `target` if it exists, or `-1` if it does not.

**Example**:
- Input: `nums = [-1,0,3,5,9,12], target = 9`
- Output: `4`
- Explanation: 9 exists at index 4.

**Constraints**:
- `1 <= nums.length <= 10^4`
- `-10^4 <= nums[i], target <= 10^4`
- `nums` is sorted in ascending order.
- All integers in `nums` are unique.

## Solution

### JavaScript
```javascript
function search(nums, target) {
    let left = 0, right = nums.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] === target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    
    return -1;
}
```

## Reasoning
- **Approach**: Use binary search to exploit the sorted array. Initialize two pointers (`left`, `right`) at the array's ends. Compute the middle index, compare `nums[mid]` with `target`, and adjust the search range based on the comparison. Return the index if found, or -1 if not.
- **Why Binary Search?**: Reduces the search space by half each iteration, achieving O(log n) time complexity.
- **Edge Cases**:
  - Empty array: Handled by constraints (`nums.length >= 1`).
  - Target not in array: Return -1.
  - Single element: Check directly.
- **Optimizations**: Use `left + (right - left) / 2` to avoid integer overflow; single pass with logarithmic time.

## Complexity Analysis
- **Time Complexity**: O(log n), where n is the length of `nums`, as the search space halves each iteration.
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints for clarity.
- For JavaScript, use `Math.floor` for integer division.
- For Java, use overflow-safe midpoint calculation and follow Google Java Style Guide.
- Handle edge cases concisely.

## Alternative Approaches
- **Linear Search**: Check each element (O(n) time). Inefficient for sorted arrays.
- **Recursive Binary Search**: Same logic recursively (O(log n) time, O(log n) space). Less space-efficient.