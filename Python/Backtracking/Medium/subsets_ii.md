# Subsets II

## Problem Statement
Given an integer array `nums` that may contain duplicates, return all possible unique subsets (the power set). The solution set must not contain duplicate subsets. Return the answer in any order.

**Example**:
- Input: `nums = [1,2,2]`
- Output: `[[],[1],[1,2],[1,2,2],[2],[2,2]]`

**Constraints**:
- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`

## Solution

### Python
```python
def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    nums.sort()
    result = []
    
    def backtrack(index: int, current: list[int]) -> None:
        result.append(current[:])
        for i in range(index, len(nums)):
            if i > index and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result
```

## Reasoning
- **Approach**: Sort the array to group duplicates, then use backtracking to generate subsets. Skip duplicate elements at the same level to avoid duplicate subsets. Add each subset to the result.
- **Why Sort?**: Sorting ensures duplicates are adjacent, making it easier to skip them.
- **Edge Cases**:
  - Empty array: Return `[[]]`.
  - All duplicates: Generate unique subsets (e.g., `[1,1]` → `[[],[1],[1,1]]`).
- **Optimizations**: Skip duplicates at the same recursion level; copy current list to avoid reference issues.

## Complexity Analysis
- **Time Complexity**: O(2^n), where n is the length of `nums`, as there are up to 2^n subsets. Sorting takes O(n log n).
- **Space Complexity**: O(n) for the recursion stack, plus O(2^n) for the output.

## Best Practices
- Use clear variable names (e.g., `index`, `current`).
- For Python, use type hints and sorting.
- For JavaScript, use spread operator and sorting.
- For Java, use `Arrays.sort` and follow Google Java Style Guide.
- Skip duplicates to ensure unique subsets.

## Alternative Approaches
- **Bit Manipulation**: Use binary numbers with duplicate handling (O(2^n) time). More complex.
- **Iterative**: Build subsets iteratively (same complexity). Less intuitive.