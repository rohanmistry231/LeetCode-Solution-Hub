# Search in Rotated Sorted Array II

## Problem Statement
Given a sorted array `nums` rotated at some pivot, with possible duplicates, and a target value, return `true` if `target` exists in `nums`, or `false` otherwise. The algorithm should aim for O(log n) time in the average case.

**Example**:
- Input: `nums = [2,5,6,0,0,1,2], target = 0`
- Output: `true`

**Constraints**:
- `1 <= nums.length <= 5000`
- `-10^4 <= nums[i], target <= 10^4`
- `nums` is sorted and rotated, with possible duplicates.

## Solution

### Python
```python
def search(nums: list[int], target: int) -> bool:
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return True
        
        if nums[left] < nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        elif nums[left] > nums[mid]:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
        else:
            left += 1
    
    return False
```

## Reasoning
- **Approach**: Use binary search, adjusted for rotation and duplicates. Compare `nums[mid]` with `nums[left]` to determine sorted half. If sorted, check if `target` lies within; otherwise, search the other half. If equal (duplicates), skip leftmost element.
- **Why Binary Search?**: Aims for O(log n) average time, though duplicates may cause O(n) worst case.
- **Edge Cases**:
  - Single element: Check directly.
  - All duplicates: Linear scan in worst case.
  - Target not present: Return false.
- **Optimizations**: Handle duplicates by incrementing `left`; use overflow-safe midpoint.

## Complexity Analysis
- **Time Complexity**: O(log n) average, O(n) worst case (all duplicates).
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints.
- For JavaScript, use `Math.floor` for integer division.
- For Java, follow Google Java Style Guide and use overflow-safe midpoint.
- Handle duplicates efficiently.

## Alternative Approaches
- **Linear Search**: Check each element (O(n) time). Too slow.
- **Find Pivot First**: Locate rotation point, then search (O(log n) average). Similar complexity.