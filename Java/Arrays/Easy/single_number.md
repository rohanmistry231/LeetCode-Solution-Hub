# Single Number

## Problem Statement
Given a non-empty array of integers `nums`, every element appears twice except for one. Find that single one. You must implement a solution with a linear runtime complexity and use only constant extra space.

**Example**:
- Input: `nums = [2,2,1]`
- Output: `1`

**Constraints**:
- `1 <= nums.length <= 3 * 10^4`
- `-3 * 10^4 <= nums[i] <= 3 * 10^4`
- Each element appears twice except for one element which appears once.

## Solution

### Java
```java
class Solution {
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }
}
```

## Reasoning
- **Approach**: Use XOR bitwise operation. XOR of a number with itself is 0, and XOR of a number with 0 is the number itself. Since all numbers except one appear twice, XORing all elements cancels out pairs, leaving the single number.
- **Why XOR?**: It meets the requirement for O(n) time and O(1) space, as it processes each element once without extra storage.
- **Edge Cases**:
  - Single element: Returns that element.
  - All pairs except one: XOR correctly isolates the single number.
- **Optimizations**: Single pass with a single variable; no additional data structures needed.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Single pass through the array.
- **Space Complexity**: O(1), as only one variable is used.

## Best Practices
- Use clear variable names (e.g., `result` for XOR accumulator).
- For Python, include type hints for clarity.
- For JavaScript, use `for...of` for readable iteration.
- For Java, follow Google Java Style Guide.
- Leverage XOR’s properties for elegant, efficient solutions.

## Alternative Approaches
- **Hash Set**: Store elements in a set, removing duplicates (O(n) time, O(n) space). Violates constant space requirement.
- **Hash Map**: Count frequencies and find the element with count 1 (O(n) time, O(n) space). Also violates space constraint.
- **Sorting**: Sort and find the unpaired element (O(n log n) time, O(1) space). Slower than XOR.