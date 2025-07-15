# Product of Array Except Self

## Problem Statement
Given an integer array `nums`, return an array `answer` such that `answer[i]` is the product of all elements except `nums[i]`. The solution must run in O(n) time and cannot use division.

**Example**:
- Input: `nums = [1,2,3,4]`
- Output: `[24,12,8,6]`

**Constraints**:
- `2 <= nums.length <= 10^5`
- `-30 <= nums[i] <= 30`
- The product of any prefix or suffix is guaranteed to fit in a 32-bit integer.

## Solution

### JavaScript
```javascript
function productExceptSelf(nums) {
    const n = nums.length;
    const answer = new Array(n).fill(1);
    let leftProduct = 1;
    for (let i = 0; i < n; i++) {
        answer[i] = leftProduct;
        leftProduct *= nums[i];
    }
    let rightProduct = 1;
    for (let i = n - 1; i >= 0; i--) {
        answer[i] *= rightProduct;
        rightProduct *= nums[i];
    }
    return answer;
}
```

## Reasoning
- **Approach**: Compute products of all elements to the left and right of each index without division. Use two passes: one to fill the answer with left products, another to multiply by right products.
- **Why No Division?**: Division is avoided to handle potential zero values and meet the O(n) requirement without extra space.
- **Edge Cases**:
  - Array of length 2: Correctly computes products.
  - Zeros in array: Handled implicitly by multiplying left and right products.
- **Optimizations**: Single output array; no extra space beyond O(1) variables.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Two passes through the array.
- **Space Complexity**: O(1), excluding the output array.

## Best Practices
- Use clear variable names (e.g., `left_product`, `right_product`).
- For Python, use type hints and initialize arrays efficiently.
- For JavaScript, use modern array methods and `fill`.
- For Java, use `Arrays.fill` and follow Google Java Style Guide.
- Avoid division to handle edge cases robustly.

## Alternative Approaches
- **Division**: Compute total product and divide by each element (O(n) time, O(1) space, but fails with zeros).
- **Extra Arrays**: Use two arrays for left and right products (O(n) time, O(n) space). Violates space constraint.