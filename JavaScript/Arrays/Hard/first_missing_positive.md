# First Missing Positive

## Problem Statement
Given an unsorted integer array `nums`, return the smallest missing positive integer. You must implement an algorithm that runs in O(n) time and uses O(1) space.

**Example**:
- Input: `nums = [1,2,0]`
- Output: `3`

**Constraints**:
- `1 <= nums.length <= 5 * 10^5`
- `-2^31 <= nums[i] <= 2^31 - 1`

## Solution

### JavaScript
```javascript
function firstMissingPositive(nums) {
    const n = nums.length;
    for (let i = 0; i < n; i++) {
        if (nums[i] <= 0 || nums[i] > n) {
            nums[i] = n + 1;
        }
    }
    for (let i = 0; i < n; i++) {
        const num = Math.abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -Math.abs(nums[num - 1]);
        }
    }
    for (let i = 0; i < n; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }
    return n + 1;
}
```

## Reasoning
- **Approach**: Use the array itself as a hash table. Ignore non-positive and numbers > n. Place each number `x` at index `x-1` by marking the index negative. The first positive index indicates the missing number.
- **Why In-Place Marking?**: Achieves O(1) space by using the array to track presence of numbers 1 to n.
- **Edge Cases**:
  - Empty array: Return 1.
  - All numbers present: Return n + 1.
  - Negative numbers: Ignored via preprocessing.
- **Optimizations**: Three linear passes; no extra space.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Three passes through the array.
- **Space Complexity**: O(1), using the array itself.

## Best Practices
- Use clear variable names (e.g., `num`, `n`).
- For Python, use type hints and handle edge cases.
- For JavaScript, use `Math.abs` for clarity.
- For Java, follow Google Java Style Guide.
- Use array indices to avoid extra space.

## Alternative Approaches
- **Hash Set**: Store numbers and check for missing (O(n) time, O(n) space). Violates space constraint.
- **Sorting**: Sort and find missing number (O(n log n) time). Too slow.