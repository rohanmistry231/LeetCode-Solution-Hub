# Merge Sorted Array

## Problem Statement
Given two sorted integer arrays `nums1` and `nums2`, merge `nums2` into `nums1` as one sorted array. `nums1` has enough space to hold additional elements from `nums2`. The number of elements initialized in `nums1` and `nums2` are `m` and `n` respectively.

**Example**:
- Input: `nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3`
- Output: `[1,2,2,3,5,6]`

**Constraints**:
- `nums1.length == m + n`
- `nums2.length == n`
- `0 <= m, n <= 200`
- `-10^9 <= nums1[i], nums2[i] <= 10^9`

## Solution

### JavaScript
```javascript
function merge(nums1, m, nums2, n) {
    let p1 = m - 1, p2 = n - 1, p = m + n - 1;
    
    while (p2 >= 0) {
        if (p1 >= 0 && nums1[p1] > nums2[p2]) {
            nums1[p] = nums1[p1];
            p1--;
        } else {
            nums1[p] = nums2[p2];
            p2--;
        }
        p--;
    }
}
```

## Reasoning
- **Approach**: Merge from the end to avoid overwriting `nums1`. Use three pointers: `p1` for `nums1`, `p2` for `nums2`, and `p` for the merged array’s end. Compare `nums1[p1]` and `nums2[p2]`, placing the larger value at `p` and moving the corresponding pointer.
- **Why Merge from End?**: Avoids shifting elements in `nums1`, as extra space is at the end.
- **Edge Cases**:
  - `n=0`: `nums1` is already sorted.
  - `m=0`: Copy `nums2` into `nums1`.
  - Single element arrays: Compare and place directly.
- **Optimizations**: In-place merging; single pass from end to start.

## Complexity Analysis
- **Time Complexity**: O(m + n), as we process each element from both arrays once.
- **Space Complexity**: O(1), as we merge in-place using only pointers.

## Best Practices
- Use clear variable names (e.g., `p1`, `p2`).
- For Python, use type hints for clarity.
- For JavaScript, use concise conditionals.
- For Java, follow Google Java Style Guide.
- Merge from end to avoid extra space.

## Alternative Approaches
- **Copy and Sort**: Copy `nums2` into `nums1` and sort (O((m+n) log (m+n)) time). Inefficient.
- **Merge to New Array**: Merge into a new array, then copy back (O(m+n) time, O(m+n) space). Less space-efficient.