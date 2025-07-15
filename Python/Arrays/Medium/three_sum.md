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

### Python
```python
def threeSum(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
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