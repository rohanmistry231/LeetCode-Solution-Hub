# Odd Even Linked List

## Problem Statement
Given the head of a singly linked list, group all nodes with odd indices together followed by the nodes with even indices, and return the reordered list. The first node is considered odd, the second node even, and so on.

**Example**:
- Input: `head = [1,2,3,4,5]`
- Output: `[1,3,5,2,4]`

**Constraints**:
- The number of nodes in the list is in the range `[0, 10^4]`.
- `-10^6 <= Node.val <= 10^6`

## Solution

### Java
```java
class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) return head;
        
        ListNode odd = head;
        ListNode even = head.next;
        ListNode evenHead = even;
        
        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        
        odd.next = evenHead;
        return head;
    }
}
```

## Reasoning
- **Approach**: Use two pointers (odd and even) to separate nodes. Odd pointer links to odd-indexed nodes, even pointer to even-indexed nodes. Maintain the even list’s head to connect it after odd nodes. Adjust pointers in-place to reorder the list.
- **Why Iterative?**: In-place pointer manipulation is space-efficient and straightforward for reordering.
- **Edge Cases**:
  - Empty or single node: Return as is.
  - Odd number of nodes: Even list ends early.
- **Optimizations**: In-place reordering; save even list’s head for final connection.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as we traverse the list once.
- **Space Complexity**: O(1), as we only use pointers.

## Best Practices
- Use clear variable names (e.g., `odd`, `evenHead`).
- For Python, use type hints for clarity.
- For JavaScript, use descriptive variable names.
- For Java, follow Google Java Style Guide.
- Save even list’s head for efficient connection.

## Alternative Approaches
- **Two Lists**: Create separate odd and even lists, then merge (O(n) time, O(1) space). More complex.
- **Array-Based**: Store nodes in array and reorder (O(n) time, O(n) space). Less space-efficient.