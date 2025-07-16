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

### JavaScript
```javascript
function permuteUnique(nums) {
    nums.sort((a, b) => a - b);
    const result = [], used = new Array(nums.length).fill(false);
    
    function backtrack(current) {
        if (current.length === nums.length) {
            result.push([...current]);
            return;
        }
        for (let i = 0; i < nums.length; i++) {
            if (used[i] || (i > 0 && nums[i] === nums[i - 1] && !used[i - 1])) continue;
            used[i] = true;
            current.push(nums[i]);
            backtrack(current);
            current.pop();
            used[i] = false;
        }
    }
    
    backtrack([]);
    return result;
}
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