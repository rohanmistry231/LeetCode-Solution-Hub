# Add Two Numbers

## Problem Statement
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each node contains a single digit. Add the two numbers and return the sum as a linked list.

**Example**:
- Input: `l1 = [2,4,3], l2 = [5,6,4]`
- Output: `[7,0,8]`
- Explanation: 342 + 465 = 807.

**Constraints**:
- The number of nodes in each linked list is in the range `[1, 100]`.
- `0 <= Node.val <= 9`
- It is guaranteed that the list represents a number that does not have leading zeros.

## Solution

### JavaScript
```javascript
function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val);
    this.next = (next === undefined ? null : next);
}

function addTwoNumbers(l1, l2) {
    const dummy = new ListNode(0);
    let current = dummy;
    let carry = 0;
    
    while (l1 || l2 || carry) {
        const x = l1 ? l1.val : 0;
        const y = l2 ? l2.val : 0;
        const total = x + y + carry;
        carry = Math.floor(total / 10);
        current.next = new ListNode(total % 10);
        current = current.next;
        l1 = l1 ? l1.next : null;
        l2 = l2 ? l2.next : null;
    }
    
    return dummy.next;
}
```

## Reasoning
- **Approach**: Use a dummy node to build the result list. Traverse both lists, summing digits and tracking carry. Create a new node for each digit of the sum (total % 10) and update carry (total // 10). Continue until both lists and carry are exhausted.
- **Why Iterative?**: Simplifies handling of carry and node creation, avoiding recursion stack space.
- **Edge Cases**:
  - Lists of different lengths: Use 0 for missing digits.
  - Carry at the end: Add an extra node if carry remains.
- **Optimizations**: Use dummy node for clean head handling; process carry in one pass.

## Complexity Analysis
- **Time Complexity**: O(max(n, m)), where n and m are the lengths of `l1` and `l2`, as we traverse both lists once.
- **Space Complexity**: O(1) excluding the output list, as we only use a few pointers and variables.

## Best Practices
- Use clear variable names (e.g., `dummy`, `carry`).
- For Python, use type hints for clarity.
- For JavaScript, use ternary operators for concise checks.
- For Java, follow Google Java Style Guide.
- Handle null checks and carry efficiently.

## Alternative Approaches
- **Recursive**: Recurse on next nodes with carry (O(max(n, m)) time, O(max(n, m)) space). Less space-efficient.
- **Convert to Numbers**: Convert lists to integers, add, then convert back (O(max(n, m)) time, but may overflow for large numbers).