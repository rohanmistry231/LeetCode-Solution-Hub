# Basic Calculator II

## Problem Statement
Given a string `s` representing an expression, evaluate it and return its value. The expression contains non-negative integers, `+`, `-`, `*`, `/` operators, and spaces. Division truncates toward zero.

**Example**:
- Input: `s = "3+2*2"`
- Output: `7`
- Explanation: 3 + (2 * 2) = 7

**Constraints**:
- `1 <= s.length <= 3 * 10^5`
- `s` consists of digits, `+`, `-`, `*`, `/`, and spaces.
- `s` represents a valid expression.
- No parentheses; all numbers are non-negative integers.
- Division by zero does not occur.

## Solution

### Java
```java
import java.util.*;

class Solution {
    public int calculate(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        int num = 0;
        char op = '+';
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (Character.isDigit(c)) {
                num = num * 10 + (c - '0');
            }
            if ("+-*/".indexOf(c) >= 0 || i == s.length() - 1) {
                if (op == '+') stack.push(num);
                else if (op == '-') stack.push(-num);
                else if (op == '*') stack.push(stack.pop() * num);
                else if (op == '/') stack.push(stack.pop() / num);
                op = c;
                num = 0;
            }
        }
        
        int result = 0;
        while (!stack.isEmpty()) {
            result += stack.pop();
        }
        return result;
    }
}
```

## Reasoning
- **Approach**: Use a stack to handle operator precedence. Parse the string, building numbers digit by digit. For each operator or end of string, apply the previous operator (`+`, `-`, `*`, `/`) to the current number and update the stack. Sum the stack for the final result.
- **Why Stack?**: Handles high-precedence operators (`*`, `/`) immediately while deferring low-precedence (`+`, `-`) to the end, avoiding parentheses.
- **Edge Cases**:
  - Single number: Return it.
  - Spaces: Skip them.
  - Large numbers: Build digit by digit.
- **Optimizations**: Process high-precedence operators in-place; single pass through string.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `s`, as we process each character once.
- **Space Complexity**: O(n), for the stack in the worst case.

## Best Practices
- Use clear variable names (e.g., `stack`, `op`).
- For Python, use type hints and string iteration.
- For JavaScript, use regex for digit check.
- For Java, use `Deque` and follow Google Java Style Guide.
- Handle multi-digit numbers correctly.

## Alternative Approaches
- **Two Stacks**: Use separate stacks for numbers and operators (O(n) time, O(n) space). More complex.
- **Recursive Descent**: Parse expression recursively (O(n) time, O(n) space). Overkill for no parentheses.