# Merge k Sorted Lists

## Problem Statement
You are given an array of `k` linked lists, each linked list is sorted in ascending order. Merge all the linked lists into one sorted linked list and return its head.

**Example**:
- Input: `lists = [[1,4,5],[1,3,4],[2,6]]`
- Output: `[1,1,2,3,4,4,5,6]`

**Constraints**:
- `k == lists.length`
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- `lists[i]` is sorted in ascending order.

## Solution

### JavaScript
```javascript
function ListNode(val, next) {
    this.val = (val === undefined ? 0 : val);
    this.next = (next === undefined ? null : next);
}

function mergeKLists(lists) {
    const dummy = new ListNode(0);
    let current = dummy;
    const heap = new MinPriorityQueue({ priority: x => x.val });
    
    lists.forEach((head, i) => {
        if (head) heap.enqueue([head.val, i, head]);
    });
    
    while (!heap.isEmpty()) {
        const [val, i, node] = heap.dequeue().element;
        current.next = node;
        current = current.next;
        if (node.next) heap.enqueue([node.next.val, i, node.next]);
    }
    
    return dummy.next;
}
```

## Reasoning
- **Approach**: Use a min-heap to store the head nodes of all lists. Pop the smallest node, add it to the result list, and push its next node (if any) into the heap. Repeat until the heap is empty. Use a dummy node to simplify list construction.
- **Why Min-Heap?**: Ensures the smallest value is always selected, maintaining the sorted order efficiently across k lists.
- **Edge Cases**:
  - Empty lists array: Return null.
  - Some lists empty: Skip null lists.
  - Single list: Return the list as is.
- **Optimizations**: Use heap for O(log k) access to the smallest node; process nodes in-place.

## Complexity Analysis
- **Time Complexity**: O(N log k), where N is the total number of nodes across all lists, and k is the number of lists. Each heap operation is O(log k), and we process N nodes.
- **Space Complexity**: O(k) for the heap, O(1) excluding the output list.

## Best Practices
- Use clear variable names (e.g., `dummy`, `heap`).
- For Python, use `heapq` and type hints.
- For JavaScript, use a priority queue library or implement a min-heap.
- For Java, use `PriorityQueue` and follow Google Java Style Guide.
- Handle null lists efficiently.

## Alternative Approaches
- **Merge Two Lists Repeatedly**: Merge lists pairwise (O(Nk) time). Less efficient.
- **Brute Force**: Collect all nodes, sort, and rebuild (O(N log N) time, O(N) space). Less efficient for large lists.