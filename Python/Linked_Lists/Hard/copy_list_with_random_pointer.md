# Copy List with Random Pointer

## Problem Statement
A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null. Return a deep copy of the list.

**Example**:
- Input: `head = [[7,null],[13,0],[11,4],[10,2],[1,0]]`
- Output: `[[7,null],[13,0],[11,4],[10,2],[1,0]]`

**Constraints**:
- `0 <= n <= 1000`
- `-10^4 <= Node.val <= 10^4`
- `Node.random` is null or points to a node in the list.

## Solution

### Python
```python
class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random

def copyRandomList(head: 'Node') -> 'Node':
    if not head:
        return None
    
    # Step 1: Interleave copied nodes
    curr = head
    while curr:
        copied = Node(curr.val)
        copied.next = curr.next
        curr.next = copied
        curr = copied.next
    
    # Step 2: Set random pointers
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    
    # Step 3: Separate lists
    dummy = Node(0)
    copied_curr = dummy
    curr = head
    while curr:
        copied_curr.next = curr.next
        copied_curr = copied_curr.next
        curr.next = copied_curr.next
        curr = curr.next
    
    return dummy.next
```

## Reasoning
- **Approach**: Use a three-step process: (1) Interleave original and copied nodes (A -> A' -> B -> B'). (2) Set random pointers for copied nodes (A'.random = A.random.next). (3) Separate the copied list from the original.
- **Why Interleaving?**: Allows setting random pointers without extra space, as copied nodes are adjacent to originals.
- **Edge Cases**:
  - Empty list: Return null.
  - Single node: Copy with random pointer.
  - Random pointer to null: Set copied random to null.
- **Optimizations**: In-place interleaving; single pass for each step.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each step traverses the list once.
- **Space Complexity**: O(1), as we only use a few pointers (excluding output list).

## Best Practices
- Use clear variable names (e.g., `copiedCurr`, `curr`).
- For Python, use type hints for clarity.
- For JavaScript, use concise null checks.
- For Java, follow Google Java Style Guide.
- Break into clear steps for readability.

## Alternative Approaches
- **Hash Map**: Map original nodes to copied nodes (O(n) time, O(n) space). Simpler but uses extra space.
- **Recursive**: Recursively copy nodes and random pointers (O(n) time, O(n) space). Less efficient.