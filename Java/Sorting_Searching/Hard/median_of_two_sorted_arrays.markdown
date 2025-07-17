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

### Java
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            int[] temp = nums1;
            nums1 = nums2;
            nums2 = temp;
        }
        
        int m = nums1.length, n = nums2.length;
        int left = 0, right = m;
        
        while (left <= right) {
            int partitionX = (left + right) / 2;
            int partitionY = (m + n + 1) / 2 - partitionX;
            
            int maxLeftX = partitionX == 0 ? Integer.MIN_VALUE : nums1[partitionX - 1];
            int minRightX = partitionX == m ? Integer.MAX_VALUE : nums1[partitionX];
            int maxLeftY = partitionY == 0 ? Integer.MIN_VALUE : nums2[partitionY - 1];
            int minRightY = partitionY == n ? Integer.MAX_VALUE : nums2[partitionY];
            
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                if ((m + n) % 2 == 0) {
                    return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2.0;
                } else {
                    return Math.max(maxLeftX, maxLeftY);
                }
            } else if (maxLeftX > minRightY) {
                right = partitionX - 1;
            } else {
                left = partitionX + 1;
            }
        }
        
        throw new IllegalArgumentException("Input arrays are not sorted");
    }
}
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