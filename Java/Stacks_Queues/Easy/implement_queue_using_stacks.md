# Implement Queue using Stacks

## Problem Statement
Implement a queue using two stacks. The implemented queue should support all the functions of a queue (`push`, `pop`, `peek`, `empty`).

**Example**:
```
MyQueue queue = new MyQueue();
queue.push(1);
queue.push(2);
queue.peek();  // returns 1
queue.pop();   // returns 1
queue.empty(); // returns false
```

**Constraints**:
- `1 <= x <= 9`
- At most 100 calls will be made to `push`, `pop`, `peek`, and `empty`.
- All calls to `pop` and `peek` are valid.

## Solution

### Java
```java
import java.util.*;

class MyQueue {
    private Deque<Integer> s1;
    private Deque<Integer> s2;
    
    public MyQueue() {
        s1 = new ArrayDeque<>();
        s2 = new ArrayDeque<>();
    }
    
    public void push(int x) {
        s1.push(x);
    }
    
    public int pop() {
        if (s2.isEmpty()) {
            while (!s1.isEmpty()) {
                s2.push(s1.pop());
            }
        }
        return s2.pop();
    }
    
    public int peek() {
        if (s2.isEmpty()) {
            while (!s1.isEmpty()) {
                s2.push(s1.pop());
            }
        }
        return s2.peek();
    }
    
    public boolean empty() {
        return s1.isEmpty() && s2.isEmpty();
    }
}
```

## Reasoning
- **Approach**: Use two stacks, `s1` for pushing and `s2` for popping/peeking. For `push`, add to `s1`. For `pop` or `peek`, if `s2` is empty, transfer all elements from `s1` to `s2` to reverse order (FIFO). `pop` removes from `s2`, `peek` checks top of `s2`, `empty` checks both stacks.
- **Why Two Stacks?**: Simulates FIFO by reversing order during transfer from `s1` to `s2`.
- **Edge Cases**:
  - Empty queue: Handled by constraints (valid calls).
  - Single element: Push and pop work directly.
- **Optimizations**: Transfer only when `s2` is empty to amortize cost; use `s1` for pushes.

## Complexity Analysis
- **Time Complexity**:
  - `push`: O(1).
  - `pop`, `peek`: O(n) worst case, O(1) amortized if multiple pops/peeks.
  - `empty`: O(1).
- **Space Complexity**: O(n), for storing n elements across two stacks.

## Best Practices
- Use clear variable names (e.g., `s1`, `s2`).
- For Python, use type hints and list as stack.
- For JavaScript, use array as stack with `push`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Minimize transfers for efficiency.

## Alternative Approaches
- **Single Stack**: Reverse stack for each pop/peek (O(n) time). Inefficient.
- **Deque as Queue**: Use deque directly (O(1) all operations). Not allowed by constraints.