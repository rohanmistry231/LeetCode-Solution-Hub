# Subsets

## Problem Statement
Given an integer array `nums` of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets. Return the answer in any order.

**Example**:
- Input: `nums = [1,2,3]`
- Output: `[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]`

**Constraints**:
- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- All elements in `nums` are unique.

## Solution

### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(0, nums, new ArrayList<>(), result);
        return result;
    }
    
    private void backtrack(int index, int[] nums, List<Integer> current, List<List<Integer>> result) {
        result.add(new ArrayList<>(current));
        for (int i = index; i < nums.length; i++) {
            current.add(nums[i]);
            backtrack(i + 1, nums, current, result);
            current.remove(current.size() - 1);
        }
    }
}
```

## Reasoning
- **Approach**: Use backtracking to generate all subsets. At each index, include or exclude the current number, adding the current subset to the result at each step. Use an index to avoid duplicates.
- **Why Backtracking?**: Efficiently explores all possible subsets by making choices at each step.
- **Edge Cases**:
  - Empty array: Return `[[]]`.
  - Single element: Return `[[], [element]]`.
- **Optimizations**: Add subset at each step; copy current list to avoid reference issues.

## Complexity Analysis
- **Time Complexity**: O(2^n), where n is the length of `nums`, as there are 2^n possible subsets.
- **Space Complexity**: O(n) for the recursion stack, plus O(2^n) for the output.

## Best Practices
- Use clear variable names (e.g., `index`, `current`).
- For Python, use type hints and list copying.
- For JavaScript, use spread operator for copying.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Include empty subset in results.

## Alternative Approaches
- **Bit Manipulation**: Use binary numbers to represent subsets (O(2^n) time). Less intuitive.
- **Iterative**: Build subsets by adding each element to existing subsets (same complexity). More complex.