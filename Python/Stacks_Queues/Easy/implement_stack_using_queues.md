# Implement Stack using Queues

## Problem Statement
Implement a stack using two queues. The implemented stack should support all the functions of a stack (`push`, `pop`, `top`, `empty`).

**Example**:
```
MyStack stack = new MyStack();
stack.push(1);
stack.push(2);
stack.top();   // returns 2
stack.pop();   // returns 2
stack.empty(); // returns false
```

**Constraints**:
- `1 <= x <= 9`
- At most 100 calls will be made to `push`, `pop`, `top`, and `empty`.
- All calls to `pop` and `top` are valid.

## Solution

### Python
```python
from collections import deque

class MyStack:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, x: int) -> None:
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> int:
        return self.q1.popleft()
    
    def top(self) -> int:
        return self.q1[0]
    
    def empty(self) -> bool:
        return len(self.q1) == 0
```

## Reasoning
- **Approach**: Use two queues, `q1` and `q2`. For `push`, add the element to `q2`, move all elements from `q1` to `q2`, then swap `q1` and `q2`. This ensures the newest element is at the front of `q1`. `pop` and `top` access the front of `q1`, and `empty` checks `q1`’s size.
- **Why Two Queues?**: Simulates LIFO by reordering elements during push to keep the newest at the front.
- **Edge Cases**:
  - Empty stack: Handle `pop`/`top` with constraints (always valid).
  - Single element: Push and pop work directly.
- **Optimizations**: Swap queues to avoid unnecessary copying; use `q1` for main storage.

## Complexity Analysis
- **Time Complexity**:
  - `push`: O(n), as we move n elements.
  - `pop`, `top`, `empty`: O(1).
- **Space Complexity**: O(n), for storing n elements across two queues.

## Best Practices
- Use clear variable names (e.g., `q1`, `q2`).
- For Python, use `deque` and type hints.
- For JavaScript, use array as queue with `shift`/`push`.
- For Java, use `LinkedList` as `Queue` and follow Google Java Style Guide.
- Minimize queue operations where possible.

## Alternative Approaches
- **Single Queue**: Rotate queue after each push (O(n) push, O(1) others). Similar complexity.
- **Deque as Stack**: Use deque directly (O(1) all operations). Not allowed by problem constraints.