# Move Zeroes

## Problem Statement
Given an integer array `nums`, move all `0`’s to the end of the array while maintaining the relative order of non-zero elements. The operation must be done in-place.

**Example**:
- Input: `nums = [0,1,0,3,12]`
- Output: `[1,3,12,0,0]`

**Constraints**:
- `1 <= nums.length <= 10^4`
- `-2^31 <= nums[i] <= 2^31 - 1`

## Solution

### Java
```java
class Solution {
    public void moveZeroes(int[] nums) {
        int nonZeroPos = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                int temp = nums[nonZeroPos];
                nums[nonZeroPos] = nums[i];
                nums[i] = temp;
                nonZeroPos++;
            }
        }
    }
}
```

## Reasoning
- **Approach**: Use two pointers. Move all non-zero elements to the front by swapping with the `non_zero_pos` index, incrementing it each time. Zeros naturally shift to the end as non-zero elements are placed earlier.
- **Why Two Pointers?**: This ensures in-place operation and maintains relative order without extra space.
- **Edge Cases**:
  - All zeros or no zeros: Works correctly without special handling.
  - Single element: No change needed.
- **Optimizations**: Single pass to move non-zeros; no need to explicitly set zeros at the end since swaps handle it.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Single pass through the array.
- **Space Complexity**: O(1), as operations are in-place.

## Best Practices
- Use descriptive variable names (e.g., `non_zero_pos`).
- For Python, use type hints and ensure in-place modification.
- For JavaScript, use array destructuring for clean swaps.
- For Java, use explicit temporary variables for swaps and follow Google Java Style Guide.
- Avoid unnecessary passes or extra arrays.

## Alternative Approaches
- **Two Passes**: Move non-zeros to the front, then fill the rest with zeros (O(n) time, O(1) space). Less efficient due to extra pass.
- **Extra Array**: Copy non-zeros to a new array, then zeros (O(n) time, O(n) space). Not in-place.