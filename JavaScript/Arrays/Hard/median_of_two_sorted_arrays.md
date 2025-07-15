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

### JavaScript
```javascript
function findMedianSortedArrays(nums1, nums2) {
    if (nums1.length > nums2.length) [nums1, nums2] = [nums2, nums1];
    const x = nums1.length, y = nums2.length;
    let left = 0, right = x;
    while (left <= right) {
        const partitionX = Math.floor((left + right) / 2);
        const partitionY = Math.floor((x + y + 1) / 2) - partitionX;
        const maxLeftX = partitionX === 0 ? -Infinity : nums1[partitionX - 1];
        const minRightX = partitionX === x ? Infinity : nums1[partitionX];
        const maxLeftY = partitionY === 0 ? -Infinity : nums2[partitionY - 1];
        const minRightY = partitionY === y ? Infinity : nums2[partitionY];
        if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
            if ((x + y) % 2 === 0) {
                return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2;
            }
            return Math.max(maxLeftX, maxLeftY);
        } else if (maxLeftX > minRightY) {
            right = partitionX - 1;
        } else {
            left = partitionX + 1;
        }
    }
    throw new Error("Input arrays are not sorted");
}
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