# Reverse Linked List

## Problem Statement
Given the head of a singly linked list, reverse the list and return its head.

**Example**:
- Input: `head = [1,2,3,4,5]`
- Output: `[5,4,3,2,1]`

**Constraints**:
- The number of nodes in the list is in the range `[0, 5000]`.
- `-5000 <= Node.val <= 5000`

## Solution

### Python
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head: ListNode) -> ListNode:
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

## Reasoning
- **Approach**: Iteratively reverse the list by adjusting pointers. For each node, save the next node, point the current node’s next to the previous node, and move forward. The `prev` node becomes the new head.
- **Why Iterative?**: Avoids recursion stack space, making it more efficient for large lists.
- **Edge Cases**:
  - Empty list: Return null.
  - Single node: Return the node.
- **Optimizations**: In-place reversal to minimize space; single pass through the list.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as we traverse the list once.
- **Space Complexity**: O(1), as we only use a few pointers.

## Best Practices
- Use clear variable names (e.g., `prev`, `current`).
- For Python, use type hints for clarity.
- For JavaScript, use descriptive variable names.
- For Java, follow Google Java Style Guide.
- Handle null checks explicitly.

## Alternative Approaches
- **Recursive**: Recursively reverse by adjusting pointers (O(n) time, O(n) space due to recursion stack).
- **Stack-Based**: Push nodes onto a stack and pop to reverse (O(n) time, O(n) space). Less efficient.