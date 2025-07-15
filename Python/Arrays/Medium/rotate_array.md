# Rotate Array

## Problem Statement
Given an integer array `nums`, rotate the array to the right by `k` steps, where `k` is non-negative. The operation must be done in-place.

**Example**:
- Input: `nums = [1,2,3,4,5,6,7], k = 3`
- Output: `[5,6,7,1,2,3,4]`

**Constraints**:
- `1 <= nums.length <= 10^5`
- `-2^31 <= nums[i] <= 2^31 - 1`
- `0 <= k <= 10^5`

## Solution

### Python
```python
def rotate(nums: list[int], k: int) -> None:
    n = len(nums)
    k = k % n
    def reverse(start: int, end: int) -> None:
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

## Reasoning
- **Approach**: Use the reverse array method: reverse the entire array, then reverse the first `k` elements, then reverse the rest. This effectively rotates the array right by `k` steps in-place.
- **Why Reverse?**: It’s an elegant way to achieve rotation in-place with O(n) time and O(1) space.
- **Edge Cases**:
  - `k > n`: Use `k % n` to handle large `k`.
  - `k = 0` or `n = 1`: No rotation needed.
- **Optimizations**: Modulo `k` to handle large rotations; in-place swaps to minimize space.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Three reverse passes.
- **Space Complexity**: O(1), as operations are in-place.

## Best Practices
- Use clear function names (e.g., `reverse`).
- For Python, use type hints and modular helper functions.
- For JavaScript, use array destructuring for swaps.
- For Java, use private helper methods and follow Google Java Style Guide.
- Handle large `k` with modulo.

## Alternative Approaches
- **Cyclic Replacements**: Move each element to its new position (O(n) time, O(1) space). Complex to implement.
- **Extra Array**: Copy elements to new positions (O(n) time, O(n) space). Not in-place.