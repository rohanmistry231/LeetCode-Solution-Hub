# Permutations II

## Problem Statement
Given a collection of numbers, `nums`, that might contain duplicates, return all possible unique permutations in any order.

**Example**:
- Input: `nums = [1,1,2]`
- Output: `[[1,1,2],[1,2,1],[2,1,1]]`

**Constraints**:
- `1 <= nums.length <= 8`
- `-10 <= nums[i] <= 10`

## Solution

### Python
```python
def permuteUnique(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []
    used = [False] * len(nums)
    
    def backtrack(current: list[int]) -> None:
        if len(current) == len(nums):
            result.append(current[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False
    
    backtrack([])
    return result
```

## Reasoning
- **Approach**: Sort the array to group duplicates, then use backtracking with a used array to track selected numbers. Skip duplicates at the same level to avoid duplicate permutations.
- **Why Sort?**: Ensures duplicates are adjacent, allowing easy skipping.
- **Edge Cases**:
  - All duplicates: Generate unique permutations (e.g., `[1,1]` → `[[1,1]]`).
  - Single number: Return one permutation.
- **Optimizations**: Use a boolean array for tracking; skip duplicates to reduce redundant work.

## Complexity Analysis
- **Time Complexity**: O(n!), where n is the length of `nums`, as there are up to n! permutations, reduced by duplicates.
- **Space Complexity**: O(n) for the recursion stack and used array, plus O(n!) for the output.

## Best Practices
- Use clear variable names (e.g., `used`, `current`).
- For Python, use type hints and sorting.
- For JavaScript, use array sorting and boolean array.
- For Java, use `Arrays.sort` and follow Google Java Style Guide.
- Skip duplicates to ensure unique permutations.

## Alternative Approaches
- **Iterative**: Generate permutations iteratively (O(n!) time). More complex.
- **Swap-Based**: Use swapping with duplicate handling (O(n!) time). Less intuitive.