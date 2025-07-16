# Intersection of Two Linked Lists

## Problem Statement
Given the heads of two singly linked lists `headA` and `headB`, return the node at which the two lists intersect. If the two lists have no intersection, return null. The lists may share a common tail.

**Example**:
- Input: `headA = [4,1,8,4,5], headB = [5,6,1,8,4,5], intersect at node with value 8`
- Output: `Reference to node with value 8`

**Constraints**:
- The number of nodes in both lists is in the range `[1, 10^5]`.
- `0 <= Node.val <= 10^6`
- There are no cycles in the lists.

## Solution

### JavaScript
```javascript
function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val);
    this.next = (next === undefined ? null : next);
}

function getIntersectionNode(headA, headB) {
    if (!headA || !headB) return null;
    
    let a = headA, b = headB;
    while (a !== b) {
        a = a ? a.next : headB;
        b = b ? b.next : headA;
    }
    
    return a;
}
```

## Reasoning
- **Approach**: Use two pointers to traverse both lists. When a pointer reaches the end of its list, redirect it to the other list’s head. If the lists intersect, the pointers will meet at the intersection node after equalizing the path lengths.
- **Why Two Pointers?**: Equalizes the length difference between lists, ensuring pointers meet at the intersection or null in one pass.
- **Edge Cases**:
  - No intersection: Pointers meet at null.
  - Lists of different lengths: Redirecting handles length disparity.
  - Empty lists: Return null.
- **Optimizations**: Single pass with O(1) space; no need to compute lengths.

## Complexity Analysis
- **Time Complexity**: O(n + m), where n and m are the lengths of `headA` and `headB`, as pointers traverse both lists once.
- **Space Complexity**: O(1), as only two pointers are used.

## Best Practices
- Use clear variable names (e.g., `a`, `b`).
- For Python, use type hints for clarity.
- For JavaScript, use ternary operators for concise checks.
- For Java, follow Google Java Style Guide.
- Handle null checks early.

## Alternative Approaches
- **Length Difference**: Compute lengths, align pointers, then traverse (O(n + m) time, O(1) space). Two passes, less elegant.
- **Hash Set**: Store nodes of one list, check for intersection (O(n + m) time, O(n) space). Less space-efficient.