# Reverse Nodes in k-Group

## Problem Statement
Given the head of a linked list, reverse every k nodes and return the head. If the number of nodes is not a multiple of k, leave the remaining nodes as is.

**Example**:
- Input: `head = [1,2,3,4,5], k = 2`
- Output: `[2,1,4,3,5]`

**Constraints**:
- The number of nodes in the list is `n`.
- `1 <= k <= n <= 5000`
- `0 <= Node.val <= 1000`

## Solution

### Python
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseKGroup(head: ListNode, k: int) -> ListNode:
    if not head or k == 1:
        return head
    
    def getLength(node: ListNode) -> int:
        length = 0
        while node:
            length += 1
            node = node.next
        return length
    
    def reverse(start: ListNode, end: ListNode) -> ListNode:
        prev, curr = None, start
        while curr != end:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev
    
    dummy = ListNode(0, head)
    prev_group = dummy
    length = getLength(head)
    
    while length >= k:
        curr = prev_group.next
        next_group = curr.next
        for _ in range(k - 1):
            next_group = next_group.next
        prev_group.next = reverse(curr, next_group)
        curr.next = next_group
        prev_group = curr
        length -= k
    
    return dummy.next
```

## Reasoning
- **Approach**: Divide the list into groups of k nodes. For each group, reverse the nodes using an iterative reverse function and reconnect to the next group. Track the remaining length to handle partial groups. Use a dummy node to simplify head handling.
- **Why Iterative Reverse?**: In-place reversal minimizes space usage while maintaining clarity.
- **Edge Cases**:
  - k=1: Return list as is.
  - Fewer than k nodes: Return list as is.
  - Empty list: Return null.
- **Optimizations**: Compute length once; use dummy node for head; reconnect groups efficiently.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as we traverse the list to compute length and reverse each group.
- **Space Complexity**: O(1), as we only use a few pointers (excluding recursion in some languages).

## Best Practices
- Use clear variable names (e.g., `prevGroup`, `nextGroup`).
- For Python, use type hints and helper functions.
- For JavaScript, use descriptive variable names.
- For Java, follow Google Java Style Guide.
- Handle edge cases with early returns.

## Alternative Approaches
- **Recursive**: Recursively reverse k nodes (O(n) time, O(n/k) space). Less space-efficient.
- **Stack-Based**: Use a stack to reverse k nodes (O(n) time, O(k) space). Less efficient.