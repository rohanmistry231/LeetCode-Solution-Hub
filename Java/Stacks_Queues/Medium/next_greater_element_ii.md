# Next Greater Element II

## Problem Statement
Given a circular integer array `nums` (i.e., the next element of `nums[nums.length - 1]` is `nums[0]`), return the next greater element for each element in `nums`. The next greater element is the first element to its right (circularly) that is greater. If it does not exist, return -1.

**Example**:
- Input: `nums = [1,2,1]`
- Output: `[2,-1,2]`

**Constraints**:
- `1 <= nums.length <= 10^4`
- `-10^9 <= nums[i] <= 10^9`

## Solution

### Java
```java
import java.util.*;

class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        Arrays.fill(result, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        
        for (int i = 0; i < 2 * n; i++) {
            int curr = nums[i % n];
            while (!stack.isEmpty() && nums[stack.peek()] < curr) {
                result[stack.pop()] = curr;
            }
            if (i < n) {
                stack.push(i);
            }
        }
        
        return result;
    }
}
```

## Reasoning
- **Approach**: Use a monotonic stack to track indices in decreasing order. Iterate through the array twice (to handle circularity), popping indices when a larger element is found and setting the result. Only push indices in the first iteration to avoid duplicates.
- **Why Monotonic Stack?**: Efficiently finds next greater elements in one pass, handling circularity with a double iteration.
- **Edge Cases**:
  - Single element: Return [-1].
  - All same values: Return all -1.
  - Maximum element: Set to -1.
- **Optimizations**: Use modulo for circular access; single stack pass with double iteration.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`, as we process each element twice and stack operations are amortized O(1).
- **Space Complexity**: O(n), for the stack and output array.

## Best Practices
- Use clear variable names (e.g., `stack`, `curr`).
- For Python, use type hints and list as stack.
- For JavaScript, use array as stack with `push`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Handle circularity with modulo and single stack.

## Alternative Approaches
- **Brute Force**: For each element, scan circularly for next greater (O(n^2) time). Too slow.
- **Array Duplication**: Concatenate array to itself (O(n) time, O(n) extra space). Less memory-efficient.