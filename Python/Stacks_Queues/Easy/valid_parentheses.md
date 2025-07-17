# Valid Parentheses

## Problem Statement
Given a string `s` containing only the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid. A string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

**Example**:
- Input: `s = "()[]{}"`
- Output: `true`

**Constraints**:
- `1 <= s.length <= 10^4`
- `s` consists of parentheses only '()[]{}'.

## Solution

### Python
```python
def isValid(s: str) -> bool:
    stack = []
    brackets = {')': '(', '}': '{', ']': '['}
    
    for c in s:
        if c in brackets.values():
            stack.append(c)
        elif c in brackets:
            if not stack or stack.pop() != brackets[c]:
                return False
    
    return len(stack) == 0
```

## Reasoning
- **Approach**: Use a stack to track opening brackets. For each character, if it’s an opening bracket, push it onto the stack. If it’s a closing bracket, check if the stack’s top matches the corresponding opening bracket. If not, or stack is empty, return false. Finally, check if the stack is empty.
- **Why Stack?**: Ensures correct order of bracket matching by tracking opening brackets until their closing counterparts are found.
- **Edge Cases**:
  - Single character: Return false.
  - Unmatched brackets: Return false.
  - Empty string: Return true.
- **Optimizations**: Use a hash map for bracket pairs; single pass through the string.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `s`, as we process each character once.
- **Space Complexity**: O(n), for the stack in the worst case (all opening brackets).

## Best Practices
- Use clear variable names (e.g., `stack`, `brackets`).
- For Python, use type hints and dictionary for mapping.
- For JavaScript, use object for bracket pairs and concise checks.
- For Java, use `Deque` and follow Google Java Style Guide.
- Check stack emptiness for validity.

## Alternative Approaches
- **Recursive**: Parse string recursively (O(n) time, O(n) space). More complex.
- **Counter-Based**: Count bracket pairs (O(n) time, O(1) space). Fails for order validation.