# Min Stack

## Problem Statement
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time. Implement the `MinStack` class with methods:
- `push(val)`: Pushes element `val` onto stack.
- `pop()`: Removes the top element.
- `top()`: Gets the top element.
- `getMin()`: Retrieves the minimum element.

**Example**:
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // returns -3
minStack.pop();
minStack.top();    // returns 0
minStack.getMin(); // returns -2
```

**Constraints**:
- `-2^31 <= val <= 2^31 - 1`
- Methods `pop`, `top`, `getMin` are called on non-empty stack.
- At most 3 * 10^4 calls will be made.

## Solution

### Python
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]
```

## Reasoning
- **Approach**: Use two stacks: `stack` for all elements and `minStack` for tracking minimums. For `push`, add to `stack` and to `minStack` if `val` is less than or equal to the current minimum. For `pop`, remove from `stack` and from `minStack` if the popped element is the current minimum. `top` and `getMin` access the tops of respective stacks.
- **Why Two Stacks?**: `minStack` maintains minimums in O(1) time by only storing elements that are potential minimums.
- **Edge Cases**:
  - Empty stack: Handled by constraints (valid calls).
  - Duplicate minimums: Push to `minStack` to maintain correct minimum after pops.
  - Single element: Both stacks handle correctly.
- **Optimizations**: Only push to `minStack` when necessary; use `<=` to handle duplicates.

## Complexity Analysis
- **Time Complexity**: O(1) for all operations (`push`, `pop`, `top`, `getMin`).
- **Space Complexity**: O(n), where n is the number of elements, for the two stacks.

## Best Practices
- Use clear variable names (e.g., `stack`, `minStack`).
- For Python, use type hints and list as stack.
- For JavaScript, use array as stack with `push`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Handle duplicates in `minStack` for correctness.

## Alternative Approaches
- **Single Stack with Pairs**: Store (value, currentMin) pairs (O(1) time, O(n) space). Uses more memory.
- **Brute Force Min**: Scan stack for minimum on `getMin` (O(n) time). Too slow.