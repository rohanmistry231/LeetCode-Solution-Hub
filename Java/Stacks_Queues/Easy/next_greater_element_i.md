# Next Greater Element I

## Problem Statement
Given two distinct integer arrays `nums1` and `nums2`, where `nums1` is a subset of `nums2`, for each element in `nums1`, find the next greater element in `nums2`. The next greater element is the first element to its right in `nums2` that is greater. If it does not exist, return -1.

**Example**:
- Input: `nums1 = [4,1,2], nums2 = [1,3,4,2]`
- Output: `[-1,3,-1]`

**Constraints**:
- `1 <= nums1.length <= nums2.length <= 1000`
- `0 <= nums1[i], nums2[i] <= 10^4`
- All integers in `nums1` and `nums2` are unique.
- All elements of `nums1` are in `nums2`.

## Solution

### Java
```java
import java.util.*;

class Solution {
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Deque<Integer> stack = new ArrayDeque<>();
        Map<Integer, Integer> nextGreater = new HashMap<>();
        
        for (int num : nums2) {
            while (!stack.isEmpty() && stack.peek() < num) {
                nextGreater.put(stack.pop(), num);
            }
            stack.push(num);
        }
        
        while (!stack.isEmpty()) {
            nextGreater.put(stack.pop(), -1);
        }
        
        int[] result = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            result[i] = nextGreater.get(nums1[i]);
        }
        return result;
    }
}
```

## Reasoning
- **Approach**: Use a monotonic stack to find the next greater element for each element in `nums2`. Push elements onto the stack; when a larger element is found, pop smaller elements and map them to the larger element. Remaining elements map to -1. Use a hash map to store results and map `nums1` elements.
- **Why Monotonic Stack?**: Efficiently finds next greater elements in one pass by maintaining a decreasing stack.
- **Edge Cases**:
  - Single element in `nums2`: Map to -1.
  - `nums1` empty: Return empty array (handled by constraints).
  - No greater element: Return -1.
- **Optimizations**: Single pass through `nums2`; use hash map for O(1) lookups.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums2`, as we process each element once and map `nums1` elements in O(m).
- **Space Complexity**: O(n), for the stack and hash map.

## Best Practices
- Use clear variable names (e.g., `stack`, `nextGreater`).
- For Python, use type hints and dictionary.
- For JavaScript, use `Map` for key-value pairs.
- For Java, use `Deque` and `HashMap`, follow Google Java Style Guide.
- Process `nums2` efficiently with stack.

## Alternative Approaches
- **Brute Force**: For each `nums1` element, scan `nums2` (O(m * n) time). Too slow.
- **Array Scan**: Store indices and scan right (O(n) time, O(n) space). Less elegant.