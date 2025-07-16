# Permutations

## Problem Statement
Given an array `nums` of distinct integers, return all possible permutations. You can return the answer in any order.

**Example**:
- Input: `nums = [1,2,3]`
- Output: `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`

**Constraints**:
- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- All integers in `nums` are unique.

## Solution

### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(new ArrayList<>(), nums, result);
        return result;
    }
    
    private void backtrack(List<Integer> current, int[] nums, List<List<Integer>> result) {
        if (current.size() == nums.length) {
            result.add(new ArrayList<>(current));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (!current.contains(nums[i])) {
                current.add(nums[i]);
                backtrack(current, nums, result);
                current.remove(current.size() - 1);
            }
        }
    }
}
```

## Reasoning
- **Approach**: Use backtracking to generate all permutations by selecting each number from the remaining set and recursing. Track used numbers to avoid duplicates. Add the current permutation to the result when all numbers are used.
- **Why Backtracking?**: It systematically explores all possible arrangements, ensuring all permutations are generated.
- **Edge Cases**:
  - Single element: Returns one permutation.
  - Empty array: Returns empty list of lists.
- **Optimizations**: In Java, use a contains check to avoid extra array copying; in Python/JavaScript, slice arrays to create new remaining sets.

## Complexity Analysis
- **Time Complexity**: O(n!), where n is the length of `nums`, as there are n! permutations.
- **Space Complexity**: O(n) for the recursion stack and current permutation, plus O(n!) for the output.

## Best Practices
- Use clear variable names (e.g., `current`, `remaining`).
- For Python, use type hints and list slicing.
- For JavaScript, use spread operator for array copying.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Avoid duplicate permutations by tracking used elements.

## Alternative Approaches
- **Iterative**: Generate permutations using factorial number system (O(n!) time). More complex.
- **Heap’s Algorithm**: Non-recursive permutation generation (O(n!) time). Less intuitive.