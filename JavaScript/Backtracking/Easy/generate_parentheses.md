# Generate Parentheses

## Problem Statement
Given an integer `n`, generate all valid combinations of `n` pairs of parentheses. A combination is valid if it is well-formed (every open parenthesis has a matching close parenthesis).

**Example**:
- Input: `n = 3`
- Output: `["((()))","(()())","(())()","()(())","()()()"]`

**Constraints**:
- `1 <= n <= 8`

## Solution

### JavaScript
```javascript
function generateParenthesis(n) {
    const result = [];
    
    function backtrack(openCount, closeCount, current) {
        if (current.length === 2 * n) {
            result.push(current);
            return;
        }
        if (openCount < n) {
            backtrack(openCount + 1, closeCount, current + "(");
        }
        if (closeCount < openCount) {
            backtrack(openCount, closeCount + 1, current + ")");
        }
    }
    
    backtrack(0, 0, "");
    return result;
}
```

## Reasoning
- **Approach**: Use backtracking to build valid parentheses combinations. Track open and close parenthesis counts. Add an open parenthesis if fewer than `n` are used, and a close parenthesis if there are more open than close. Add to result when length is `2n`.
- **Why Backtracking?**: Ensures only valid combinations by enforcing parentheses rules during recursion.
- **Edge Cases**:
  - `n = 1`: Returns `["()"]`.
  - Invalid sequences: Prevented by count checks.
- **Optimizations**: Prune invalid branches early by checking counts.

## Complexity Analysis
- **Time Complexity**: O(4^n / √n), the nth Catalan number, representing the number of valid parentheses combinations.
- **Space Complexity**: O(n) for the recursion stack, plus O(4^n / √n) for the output.

## Best Practices
- Use clear variable names (e.g., `open_count`, `close_count`).
- For Python, use type hints for clarity.
- For JavaScript, use modern function syntax.
- For Java, use `ArrayList` and follow Google Java Style Guide.
- Validate parentheses counts to avoid invalid combinations.

## Alternative Approaches
- **Dynamic Programming**: Build solutions iteratively (O(4^n / √n) time, O(n) space). More complex.
- **Closure Number**: Use mathematical properties of Catalan numbers (same complexity). Less intuitive.