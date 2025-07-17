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

### Java
```java
import java.util.*;

class MyStack {
    private Queue<Integer> q1;
    private Queue<Integer> q2;
    
    public MyStack() {
        q1 = new LinkedList<>();
        q2 = new LinkedList<>();
    }
    
    public void push(int x) {
        q2.offer(x);
        while (!q1.isEmpty()) {
            q2.offer(q1.poll());
        }
        Queue<Integer> temp = q1;
        q1 = q2;
        q2 = temp;
    }
    
    public int pop() {
        return q1.poll();
    }
    
    public int top() {
        return q1.peek();
    }
    
    public boolean empty() {
        return q1.isEmpty();
    }
}
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