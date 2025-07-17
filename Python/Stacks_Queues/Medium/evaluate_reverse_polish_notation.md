# Evaluate Reverse Polish Notation

## Problem Statement
Evaluate the value of an arithmetic expression in Reverse Polish Notation (RPN). Valid operators are `+`, `-`, `*`, `/`. Each operand is an integer, and division truncates toward zero.

**Example**:
- Input: `tokens = ["2","1","+","3","*"]`
- Output: `9`
- Explanation: ((2 + 1) * 3) = 9

**Constraints**:
- `1 <= tokens.length <= 10^4`
- `tokens[i]` is either an operator (`+`, `-`, `*`, `/`) or an integer in `[-200, 200]`.
- The expression is valid and division by zero does not occur.

## Solution

### Python
```python
def evalRPN(tokens: list[str]) -> int:
    stack = []
    operators = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: int(x / y)
    }
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            stack.append(operators[token](a, b))
        else:
            stack.append(int(token))
    
    return stack[0]
```

## Reasoning
- **Approach**: Use a stack to process tokens. For numbers, push to the stack. For operators, pop two operands, apply the operation, and push the result. The final result is the stack’s top element.
- **Why Stack?**: RPN evaluates operands in order, and a stack naturally handles the last-in-first-out order for operations.
- **Edge Cases**:
  - Single token: Return the number.
  - Minimum tokens: Handled by valid expression constraint.
  - Division by zero: Not applicable per constraints.
- **Optimizations**: Use a hash map for operator functions; single pass through tokens.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `tokens`, as each token is processed once.
- **Space Complexity**: O(n), for the stack in the worst case.

## Best Practices
- Use clear variable names (e.g., `stack`, `operators`).
- For Python, use type hints and lambda functions.
- For JavaScript, use `Math.trunc` for division.
- For Java, use `Deque` and `BiFunction`, follow Google Java Style Guide.
- Define operators in a map for clarity.

## Alternative Approaches
- **Recursive**: Parse expression recursively (O(n) time, O(n) space). More complex.
- **Array-Based**: Store operands in array (O(n) time, O(n) space). Less intuitive.