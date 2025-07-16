# Swap Nodes in Pairs

## Problem Statement
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the nodes (only nodes themselves may be changed).

**Example**:
- Input: `head = [1,2,3,4]`
- Output: `[2,1,4,3]`

**Constraints**:
- The number of nodes in the list is in the range `[0, 100]`.
- `0 <= Node.val <= 100`

## Solution

### Python
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def swapPairs(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    
    next_node = head.next
    head.next = swapPairs(next_node.next)
    next_node.next = head
    return next_node
```

## Reasoning
- **Approach**: Use recursion to swap pairs. For each pair, swap the current node with the next, recursively swap the rest of the list, and connect the swapped pair. Base case: return if list is empty or has one node.
- **Why Recursive?**: Simplifies swapping logic by handling pairs recursively, ensuring clean pointer updates.
- **Edge Cases**:
  - Empty list or single node: Return as is.
  - Odd number of nodes: Last node remains unswapped.
- **Optimizations**: Recursive approach is concise; no extra space except recursion stack.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as we process each node once.
- **Space Complexity**: O(n) due to the recursion stack for n/2 recursive calls.

## Best Practices
- Use clear variable names (e.g., `nextNode`).
- For Python, use type hints for clarity.
- For JavaScript, use concise null checks.
- For Java, follow Google Java Style Guide.
- Handle base cases early.

## Alternative Approaches
- **Iterative**: Use pointers to swap pairs in-place (O(n) time, O(1) space). More complex but space-efficient.
- **Dummy Node Iterative**: Use a dummy node to simplify iterative swapping (O(n) time, O(1) space). Slightly clearer.