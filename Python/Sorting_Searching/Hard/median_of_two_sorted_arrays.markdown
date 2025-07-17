# Median of Two Sorted Arrays

## Problem Statement
Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return the median of the two sorted arrays. The overall run time complexity should be O(log (m + n)).

**Example**:
- Input: `nums1 = [1,3], nums2 = [2]`
- Output: `2.0`
- Explanation: Merged array = [1,2,3], median = 2.

**Constraints**:
- `nums1.length == m`
- `nums2.length == n`
- `0 <= m, n <= 1000`
- `1 <= m + n <= 2000`
- `-10^6 <= nums1[i], nums2[i] <= 10^6`

## Solution

### Python
```python
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition_x = (left + right) // 2
        partition_y = (m + n + 1) // 2 - partition_x
        
        max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
        min_right_x = float('inf') if partition_x == m else nums1[partition_x]
        max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
        min_right_y = float('inf') if partition_y == n else nums2[partition_y]
        
        if max_left_x <= min_right_y and max_left_y <= min_right_x:
            if (m + n) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            else:
                return max(max_left_x, max_left_y)
        elif max_left_x > min_right_y:
            right = partition_x - 1
        else:
            left = partition_x + 1
    
    raise ValueError("Input arrays are not sorted")
```

## Reasoning
- **Approach**: Use binary search on the smaller array to find the correct partition such that the left halves of both arrays are less than or equal to the right halves. Compute the median based on whether the total length is odd or even.
- **Why Binary Search?**: Achieves O(log (m + n)) by searching on the smaller array, reducing the problem to finding the correct partition.
- **Edge Cases**:
  - One array empty: Find median of the other.
  - Single element arrays: Compare and compute median.
  - Odd/even total length: Adjust median calculation.
- **Optimizations**: Use smaller array for binary search; handle edge cases with infinities.

## Complexity Analysis
- **Time Complexity**: O(log min(m, n)), as binary search is performed on the smaller array.
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `partitionX`, `maxLeftX`).
- For Python, use type hints and handle edge cases.
- For JavaScript, use `Infinity` for boundaries.
- For Java, follow Google Java Style Guide and use explicit type casting.
- Swap arrays to optimize for smaller size.

## Alternative Approaches
- **Merge Arrays**: Merge and find median (O(m + n) time). Too slow.
- **Two Pointers**: Merge until median position (O(m + n) time). Inefficient.