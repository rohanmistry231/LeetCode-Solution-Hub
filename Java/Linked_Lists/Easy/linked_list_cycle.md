# Linked List Cycle

## Problem Statement
Given the head of a linked list, determine if the list has a cycle. A cycle exists if a node can be reached again by following the next pointers.

**Example**:
- Input: `head = [3,2,0,-4], pos = 1` (node at index 1 is connected to by the last node)
- Output: `true`

**Constraints**:
- The number of nodes in the list is in the range `[0, 10^4]`.
- `-10^5 <= Node.val <= 10^5`
- `pos` is -1 or a valid index in the linked list.

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
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;
        ListNode slow = head;
        ListNode fast = head.next;
        
        while (slow != fast) {
            if (fast == null || fast.next == null) return false;
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return true;
    }
}
```

## Reasoning
- **Approach**: Use Floyd’s Cycle-Finding Algorithm (two-pointer technique). Move a slow pointer one step and a fast pointer two steps. If they meet, a cycle exists. If fast reaches the end, no cycle exists.
- **Why Floyd’s Algorithm?**: Detects cycles in O(n) time with O(1) space, leveraging the fact that fast will catch up to slow in a cycle.
- **Edge Cases**:
  - Empty list or single node: No cycle.
  - Cycle at head: Detected by meeting pointers.
- **Optimizations**: Start fast at `head.next` to avoid initial overlap; check null pointers.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as fast pointer catches up in linear time or reaches the end.
- **Space Complexity**: O(1), as only two pointers are used.

## Best Practices
- Use clear variable names (e.g., `slow`, `fast`).
- For Python, use type hints for clarity.
- For JavaScript, use early null checks.
- For Java, follow Google Java Style Guide.
- Optimize by starting fast at `head.next`.

## Alternative Approaches
- **Hash Set**: Store visited nodes in a set (O(n) time, O(n) space). Less space-efficient.
- **Marking Nodes**: Modify node values to mark visits (O(n) time, O(1) space, but destructive).