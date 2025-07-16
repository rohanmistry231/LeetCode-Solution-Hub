# Middle of the Linked List

## Problem Statement
Given the head of a singly linked list, return the middle node. If there are two middle nodes, return the second one.

**Example**:
- Input: `head = [1,2,3,4,5]`
- Output: `[3,4,5]`

**Constraints**:
- The number of nodes in the list is in the range `[1, 100]`.
- `1 <= Node.val <= 100`

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
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
```

## Reasoning
- **Approach**: Use the two-pointer technique (slow and fast pointers). Slow moves one step, fast moves two steps. When fast reaches the end, slow is at the middle. For even-length lists, slow lands on the second middle node.
- **Why Two Pointers?**: Finds the middle in one pass without counting nodes, using O(1) space.
- **Edge Cases**:
  - Single node: Return the head.
  - Even number of nodes: Return the second middle node.
- **Optimizations**: Single pass with two pointers; no extra space needed.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as we traverse the list once.
- **Space Complexity**: O(1), as only two pointers are used.

## Best Practices
- Use clear variable names (e.g., `slow`, `fast`).
- For Python, use type hints for clarity.
- For JavaScript, use concise pointer updates.
- For Java, follow Google Java Style Guide.
- Handle null checks in the loop condition.

## Alternative Approaches
- **Count and Find**: Count nodes, then traverse to the middle (O(n) time, O(1) space). Two passes, less efficient.
- **Array-Based**: Store nodes in an array and return middle (O(n) time, O(n) space). Less space-efficient.