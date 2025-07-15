# Contains Duplicate

## Problem Statement
Given an integer array `nums`, return `true` if any value appears at least twice in the array, and `false` if every element is distinct.

**Example**:
- Input: `nums = [1,2,3,1]`
- Output: `true`
- Explanation: The element `1` appears twice.

**Constraints**:
- `1 <= nums.length <= 10^5`
- `-10^9 <= nums[i] <= 10^9`

## Solution

### Java
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (!set.add(num)) {
                return true;
            }
        }
        return false;
    }
}
```

## Reasoning
- **Approach**: Use a set to check for duplicates. In Python and JavaScript, convert the array to a set and compare sizes. In Java, add elements to a set and check if any addition fails (indicating a duplicate).
- **Why Set?**: Sets store unique elements, making duplicate detection efficient. Comparing set size to array length is a concise way to check for duplicates.
- **Edge Cases**:
  - Empty array: Returns `false` (no duplicates).
  - Single element: Returns `false` (no duplicates possible).
  - Large arrays: Set operations remain efficient due to O(1) average-time lookups.
- **Optimizations**: The Python/JavaScript solutions are concise and leverage built-in set conversion. The Java solution stops early upon finding a duplicate.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Set conversion or iteration takes linear time.
- **Space Complexity**: O(n) to store the set of unique elements.

## Best Practices
- Use clear variable names (e.g., `set` for the data structure).
- For Python, include type hints for clarity and pylint compliance.
- For JavaScript, use `Set` for efficient unique value storage.
- For Java, use `HashSet` and follow Google Java Style Guide.
- Prefer concise solutions when they maintain readability and performance.

## Alternative Approaches
- **Sorting**: Sort the array and check adjacent elements for duplicates (O(n log n) time, O(1) space if in-place).
- **Brute Force**: Compare each element with every other element (O(n²) time, O(1) space). Inefficient for large arrays.