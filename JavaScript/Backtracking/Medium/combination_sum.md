# Combination Sum

## Problem Statement
Given an array of distinct integers `candidates` and a target integer `target`, return a list of all unique combinations of `candidates` where the chosen numbers sum to `target`. You may use the same number an unlimited number of times. Return the answer in any order.

**Example**:
- Input: `candidates = [2,3,6,7], target = 7`
- Output: `[[2,2,3],[7]]`

**Constraints**:
- `1 <= candidates.length <= 30`
- `2 <= candidates[i] <= 40`
- All elements of `candidates` are distinct.
- `1 <= target <= 40`

## Solution

### JavaScript
```javascript
function combinationSum(candidates, target) {
    const result = [];
    
    function backtrack(index, current, total) {
        if (total === target) {
            result.push([...current]);
            return;
        }
        if (total > target || index >= candidates.length) return;
        current.push(candidates[index]);
        backtrack(index, current, total + candidates[index]);
        current.pop();
        backtrack(index + 1, current, total);
    }
    
    backtrack(0, [], 0);
    return result;
}
```

## Reasoning
- **Approach**: Use backtracking to try including or excluding each candidate. Track the current sum and add to the result when it equals the target. Allow reusing the same number by keeping the index unchanged.
- **Why Backtracking?**: It explores all possible combinations while pruning branches where the sum exceeds the target.
- **Edge Cases**:
  - No solution: Return empty list.
  - Single candidate: Check if it divides target evenly.
- **Optimizations**: Prune when sum exceeds target; copy current list to avoid reference issues.

## Complexity Analysis
- **Time Complexity**: O(N^(T/M)), where N is the length of `candidates`, T is the target, and M is the minimum candidate value. Each number can be used multiple times, leading to exponential growth.
- **Space Complexity**: O(T/M) for the recursion stack, plus O(N^(T/M)) for the output.

## Best Practices
- Use clear variable names (e.g., `total`, `current`).
- For Python, use type hints and pruning.
- For JavaScript, use spread operator for copying.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Prune branches when sum exceeds target.

## Alternative Approaches
- **Dynamic Programming**: Use a DP table to store combinations (O(N*T) time). More complex for output format.
- **Iterative**: Build combinations iteratively (same complexity). Less intuitive.