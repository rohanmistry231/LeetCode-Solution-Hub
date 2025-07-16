# Remove Nth Node From End of List

## Problem Statement
Given the head of a linked list, remove the nth node from the end and return the head.

**Example**:
- Input: `head = [1,2,3,4,5], n = 2`
- Output: `[1,2,3,5]`

**Constraints**:
- The number of nodes in the list is `sz`.
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`

## Solution

### JavaScript
```javascript
function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val);
    this.next = (next === undefined ? null : next);
}

function removeNthFromEnd(head, n) {
    const dummy = new ListNode(0, head);
    let slow = dummy, fast = dummy;
    
    for (let i = 0; i <= n; i++) {
        fast = fast.next;
    }
    
    while (fast) {
        slow = slow.next;
        fast = fast.next;
    }
    
    slow.next = slow.next.next;
    return dummy.next;
}
```

## Reasoning
- **Approach**: Use two pointers (slow and fast). Move fast n+1 steps ahead, then move both pointers until fast reaches the end. Slow will be just before the node to remove. Adjust the next pointer to skip the nth node.
- **Why Two Pointers?**: Allows finding the nth node from the end in one pass without counting the list length.
- **Edge Cases**:
  - Remove head: Use dummy node to simplify.
  - Single node: Return null if n=1.
- **Optimizations**: Dummy node handles head removal; single pass for efficiency.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as we traverse the list once.
- **Space Complexity**: O(1), as we only use two pointers and a dummy node.

## Best Practices
- Use clear variable names (e.g., `slow`, `fast`).
- For Python, use type hints for clarity.
- For JavaScript, use concise loop constructs.
- For Java, follow Google Java Style Guide.
- Use dummy node to handle edge cases.

## Alternative Approaches
- **Two Passes**: Count list length, then traverse to remove node (O(n) time, O(1) space). Less efficient.
- **Stack-Based**: Store nodes in a stack, pop n times (O(n) time, O(n) space). Less space-efficient.