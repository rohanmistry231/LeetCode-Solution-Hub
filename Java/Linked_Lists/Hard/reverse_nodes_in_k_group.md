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

### Java
```java
class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Solution {
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k == 1) return head;
        
        int getLength(ListNode node) {
            int length = 0;
            while (node != null) {
                length++;
                node = node.next;
            }
            return length;
        }
        
        ListNode reverse(ListNode start, ListNode end) {
            ListNode prev = null, curr = start;
            while (curr != end) {
                ListNode nextNode = curr.next;
                curr.next = prev;
                prev = curr;
                curr = nextNode;
            }
            return prev;
        }
        
        ListNode dummy = new ListNode(0, head);
        ListNode prevGroup = dummy;
        int length = getLength(head);
        
        while (length >= k) {
            ListNode curr = prevGroup.next;
            ListNode nextGroup = curr.next;
            for (int i = 0; i < k - 1; i++) {
                nextGroup = nextGroup.next;
            }
            prevGroup.next = reverse(curr, nextGroup);
            curr.next = nextGroup;
            prevGroup = curr;
            length -= k;
        }
        
        return dummy.next;
    }
}
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