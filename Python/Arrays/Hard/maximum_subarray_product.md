# Maximum Subarray Product

## Problem Statement
Given an integer array `nums`, find a contiguous non-empty subarray with the largest product, and return that product.

**Example**:
- Input: `nums = [2,3,-2,4]`
- Output: `6`
- Explanation: Subarray `[2,3]` has the largest product `6`.

**Constraints**:
- `1 <= nums.length <= 2 * 10^4`
- `-10 <= nums[i] <= 10`
- The product of any subarray is guaranteed to fit in a 32-bit integer.

## Solution

### Python
```python
def maxProduct(nums: list[int]) -> int:
    max_so_far = min_so_far = result = nums[0]
    for num in nums[1:]:
        temp_max = max(num, max_so_far * num, min_so_far * num)
        min_so_far = min(num, max_so_far * num, min_so_far * num)
        max_so_far = temp_max
        result = max(result, max_so_far)
    return result
```

## Reasoning
- **Approach**: Track maximum and minimum products ending at each index, as a negative number can turn a minimum product into a maximum with another negative number. Update `max_so_far`, `min_so_far`, and global `result` at each step.
- **Why Track Min?**: Negative numbers can flip maximum to minimum products, so we need both.
- **Edge Cases**:
  - Single element: Return that element.
  - Zeros: Reset product, consider single numbers.
  - All negatives: Maximum might be a single number or product of two negatives.
- **Optimizations**: Single pass; constant space.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Single pass through the array.
- **Space Complexity**: O(1), using only constant space.

## Best Practices
- Use clear variable names (e.g., `max_so_far`, `min_so_far`).
- For Python, use type hints for clarity.
- For JavaScript, use `Math.max/min` for readability.
- For Java, follow Google Java Style Guide.
- Track both max and min products to handle negative numbers.

## Alternative Approaches
- **Brute Force**: Check all subarrays (O(n²) time). Inefficient.
- **Divide and Conquer**: Split array and compute max product (O(n log n) time). Too complex.