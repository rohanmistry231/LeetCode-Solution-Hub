# Merge Two Sorted Lists

## Problem Statement
You are given the heads of two sorted linked lists `list1` and `list2`. Merge the two lists into one sorted linked list and return its head.

**Example**:
- Input: `list1 = [1,2,4], list2 = [1,3,4]`
- Output: `[1,1,2,3,4,4]`

**Constraints**:
- The number of nodes in both lists is in the range `[0, 50]`.
- `-100 <= Node.val <= 100`
- Both `list1` and `list2` are sorted in non-decreasing order.

## Solution

### JavaScript
```javascript
function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val);
    this.next = (next === undefined ? null : next);
}

function mergeTwoLists(list1, list2) {
    const dummy = new ListNode(0);
    let current = dummy;
    
    while (list1 && list2) {
        if (list1.val <= list2.val) {
            current.next = list1;
            list1 = list1.next;
        } else {
            current.next = list2;
            list2 = list2.next;
        }
        current = current.next;
    }
    
    current.next = list1 || list2;
    return dummy.next;
}
```

## Reasoning
- **Approach**: Use a dummy node to simplify the merging process. Compare nodes from `list1` and `list2`, attaching the smaller node to the result list. Move the pointer of the chosen list forward. Finally, attach the remaining nodes from either list.
- **Why Iterative?**: Merging is straightforward and doesn’t require recursion, keeping space complexity minimal.
- **Edge Cases**:
  - One or both lists empty: Return the other list or null.
  - Lists of different lengths: Append the remaining nodes.
- **Optimizations**: Use a dummy node to avoid edge case handling for the head; modify pointers in-place.

## Complexity Analysis
- **Time Complexity**: O(n + m), where n and m are the lengths of `list1` and `list2`, as we traverse both lists once.
- **Space Complexity**: O(1), excluding the output list, as we only use a dummy node and pointers.

## Best Practices
- Use clear variable names (e.g., `dummy`, `current`).
- For Python, use type hints for clarity.
- For JavaScript, use concise logical OR for remaining nodes.
- For Java, follow Google Java Style Guide.
- Use dummy node to simplify head handling.

## Alternative Approaches
- **Recursive**: Merge recursively by selecting the smaller head and recursing on the rest (O(n + m) time, O(n + m) space due to recursion stack).
- **In-Place with Modification**: Modify one list to include the other (complex and error-prone).