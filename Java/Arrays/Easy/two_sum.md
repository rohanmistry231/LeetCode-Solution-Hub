# Two Sum

## Problem Statement
Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`. You may assume that each input has exactly one solution, and you may not use the same element twice. Return the answer in any order.

**Example**:
- Input: `nums = [2,7,11,15], target = 9`
- Output: `[0,1]`
- Explanation: Because `nums[0] + nums[1] == 9`, we return `[0, 1]`.

**Constraints**:
- `2 <= nums.length <= 10^4`
- `-10^9 <= nums[i] <= 10^9`
- `-10^9 <= target <= 10^9`
- Only one valid answer exists.

## Solution

### Java
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> numMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numMap.containsKey(complement)) {
                return new int[] {numMap.get(complement), i};
            }
            numMap.put(nums[i], i);
        }
        return new int[0];
    }
}
```

## Reasoning
- **Approach**: Use a hash map to store numbers and their indices. For each number, compute its complement (`target - num`). If the complement exists in the hash map, return its index and the current index. Otherwise, add the number and its index to the hash map.
- **Why Hash Map?**: A brute-force approach (checking all pairs) takes O(n²) time. Using a hash map reduces time to O(n) by trading space for speed, as lookups are O(1) on average.
- **Edge Cases**:
  - Empty array or no solution: Not applicable due to problem constraints (exactly one solution exists).
  - Duplicate numbers: Handled by storing the most recent index in the hash map.
- **Optimizations**: Single pass through the array; no need to store all numbers before checking complements.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. We iterate through the array once, with O(1) hash map operations.
- **Space Complexity**: O(n) to store up to n numbers in the hash map.

## Best Practices
- Use meaningful variable names (e.g., `num_map`, `complement`) for clarity.
- For Python, adhere to PEP 8 and ensure pylint 10/10 score with type hints.
- For JavaScript, use `Map` over objects for hash maps to avoid key coercion issues.
- For Java, use `HashMap` and follow Google Java Style Guide for consistent formatting.
- Include comments for complex logic (e.g., hash map purpose).

## Alternative Approaches
- **Brute Force**: Check all pairs of numbers (O(n²) time, O(1) space). Not recommended due to inefficiency.
- **Sorting + Two Pointers**: Sort the array and use two pointers to find the pair (O(n log n) time, O(n) space for storing indices). Less efficient than hash map for this problem.