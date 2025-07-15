# Three Sum

## Problem Statement
Given an integer array `nums`, return all unique triplets `[nums[i], nums[j], nums[k]]` such that `i != j != k` and `nums[i] + nums[j] + nums[k] == 0`. The solution set must not contain duplicate triplets.

**Example**:
- Input: `nums = [-1,0,1,2,-1,-4]`
- Output: `[[-1,-1,2],[-1,0,1]]`

**Constraints**:
- `0 <= nums.length <= 3000`
- `-10^5 <= nums[i] <= 10^5`

## Solution

### JavaScript
```javascript
function threeSum(nums) {
    nums.sort((a, b) => a - b);
    const result = [];
    for (let i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] === nums[i - 1]) continue;
        let left = i + 1, right = nums.length - 1;
        while (left < right) {
            const total = nums[i] + nums[left] + nums[right];
            if (total === 0) {
                result.push([nums[i], nums[left], nums[right]]);
                left++;
                right--;
                while (left < right && nums[left] === nums[left - 1]) left++;
                while (left < right && nums[right] === nums[right + 1]) right--;
            } else if (total < 0) {
                left++;
            } else {
                right--;
            }
        }
    }
    return result;
}
```

## Reasoning
- **Approach**: Sort the array and use a two-pointer technique. Fix one element and use two pointers to find a pair summing to the negation of the fixed element. Skip duplicates to ensure unique triplets.
- **Why Sort?**: Sorting allows two-pointer technique and easy duplicate skipping.
- **Edge Cases**:
  - Less than 3 elements: Return empty list.
  - No valid triplets: Return empty list.
  - Duplicates: Handled by skipping identical elements.
- **Optimizations**: Sorting reduces complexity; duplicate checks prevent redundant triplets.

## Complexity Analysis
- **Time Complexity**: O(n²), where n is the length of `nums`. Sorting is O(n log n), and two-pointer search is O(n) per fixed element.
- **Space Complexity**: O(1) or O(n) depending on sorting implementation, excluding output.

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints and list comprehension.
- For JavaScript, use arrow functions for sorting.
- For Java, use `Arrays.asList` and follow Google Java Style Guide.
- Skip duplicates to ensure unique results.

## Alternative Approaches
- **Hash Map**: Use a hash map for two-sum within a loop (O(n²) time, O(n) space). Less efficient due to hash map overhead.
- **Brute Force**: Check all triplets (O(n³) time). Inefficient.