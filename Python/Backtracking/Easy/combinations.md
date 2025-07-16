# Combinations

## Problem Statement
Given two integers `n` and `k`, return all possible combinations of `k` numbers chosen from the range `[1, n]`. You may return the answer in any order.

**Example**:
- Input: `n = 4, k = 2`
- Output: `[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]`

**Constraints**:
- `1 <= n <= 20`
- `1 <= k <= n`

## Solution

### Python
```python
def combine(n: int, k: int) -> list[list[int]]:
    result = []
    
    def backtrack(start: int, current: list[int]) -> None:
        if len(current) == k:
            result.append(current[:])
            return
        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(1, [])
    return result
```

## Reasoning
- **Approach**: Use backtracking to build combinations. Start from 1 and add numbers up to `n`, ensuring each combination has `k` numbers. Use a start index to avoid duplicates and maintain order.
- **Why Backtracking?**: Systematically generates all valid combinations while pruning invalid branches.
- **Edge Cases**:
  - `k = 0`: Return empty list of lists.
  - `k = n`: Return single combination of all numbers.
- **Optimizations**: Copy current list to result to avoid reference issues; use start index to ensure ascending order.

## Complexity Analysis
- **Time Complexity**: O(n choose k) = O(n! / (k!(n-k)!)), the number of k-combinations.
- **Space Complexity**: O(k) for the recursion stack, plus O(n choose k) for the output.

## Best Practices
- Use clear variable names (e.g., `start`, `current`).
- For Python, use type hints and list copying.
- For JavaScript, use spread operator for copying.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Use start index to avoid duplicate combinations.

## Alternative Approaches
- **Iterative**: Use a loop to generate combinations (same complexity). More complex to implement.
- **Math-Based**: Use combinatorial formulas (complex to code).