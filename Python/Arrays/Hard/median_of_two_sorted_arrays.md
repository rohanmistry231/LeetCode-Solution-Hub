# Median of Two Sorted Arrays

## Problem Statement
Given two sorted arrays `nums1` and `nums2` of size `m` and `n`, return the median of the two sorted arrays. The overall run time complexity should be O(log(min(m,n))).

**Example**:
- Input: `nums1 = [1,3], nums2 = [2]`
- Output: `2.0`
- Explanation: Merged array = `[1,2,3]`, median is `2`.

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
    x, y = len(nums1), len(nums2)
    left, right = 0, x
    while left <= right:
        partition_x = (left + right) // 2
        partition_y = (x + y + 1) // 2 - partition_x
        max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
        min_right_x = float('inf') if partition_x == x else nums1[partition_x]
        max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
        min_right_y = float('inf') if partition_y == y else nums2[partition_y]
        if max_left_x <= min_right_y and max_left_y <= min_right_x:
            if (x + y) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            return max(max_left_x, max_left_y)
        elif max_left_x > min_right_y:
            right = partition_x - 1
        else:
            left = partition_x + 1
    raise ValueError("Input arrays are not sorted")
```

## Reasoning
- **Approach**: Use binary search on the smaller array to find a partition such that the left and right halves of the merged array are balanced. Ensure `max_left_x <= min_right_y` and `max_left_y <= min_right_x` to find the correct partition. Compute the median based on whether the total length is odd or even.
- **Why Binary Search?**: Achieves O(log(min(m,n))) time by searching partitions in the smaller array.
- **Edge Cases**:
  - Empty array: Handled by swapping arrays.
  - Odd/even total length: Handled in median calculation.
  - Single element arrays: Correctly partitioned.
- **Optimizations**: Process smaller array to reduce search space; use infinity for boundary conditions.

## Complexity Analysis
- **Time Complexity**: O(log(min(m,n))), where m and n are array lengths, due to binary search.
- **Space Complexity**: O(1), using only constant space.

## Best Practices
- Use clear variable names (e.g., `partition_x`, `max_left_x`).
- For Python, use type hints and handle edge cases with infinity.
- For JavaScript, use `Infinity` for boundary conditions.
- For Java, use `Integer.MIN_VALUE/MAX_VALUE` and follow Google Java Style Guide.
- Swap arrays to optimize for smaller input.

## Alternative Approaches
- **Merge Arrays**: Merge arrays and find median (O(m + n) time, O(m + n) space). Too slow.
- **Brute Force**: Try all partitions (O(min(m,n) * (m+n)) time). Inefficient.