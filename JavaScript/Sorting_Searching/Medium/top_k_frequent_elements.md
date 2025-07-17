# Top K Frequent Elements

## Problem Statement
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. You may return the answer in any order.

**Example**:
- Input: `nums = [1,1,1,2,2,3], k = 2`
- Output: `[1,2]`

**Constraints**:
- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`
- `k` is in the range `[1, the number of unique elements in the array]`.

## Solution

### JavaScript
```javascript
function topKFrequent(nums, k) {
    const count = new Map();
    for (const num of nums) {
        count.set(num, (count.get(num) || 0) + 1);
    }
    
    const heap = new MinPriorityQueue({ priority: x => x[0] });
    for (const [num, freq] of count) {
        heap.enqueue([freq, num]);
        if (heap.size() > k) heap.dequeue();
    }
    
    return heap.toArray().map(x => x.element[1]);
}
```

## Reasoning
- **Approach**: Count frequencies using a hash map. Use a min-heap of size k to store elements by frequency. For each unique element, add to the heap and remove the smallest frequency if size exceeds k. Return the remaining elements.
- **Why Min-Heap?**: Efficiently maintains k most frequent elements with O(log k) operations.
- **Edge Cases**:
  - k=1: Return most frequent element.
  - Single unique element: Return it if k=1.
  - k equals unique elements: Return all.
- **Optimizations**: Use min-heap to limit size to k; avoid sorting frequencies.

## Complexity Analysis
- **Time Complexity**: O(n log k), where n is the length of `nums`. Counting is O(n), heap operations are O(log k) per unique element.
- **Space Complexity**: O(n) for the hash map, O(k) for the heap.

## Best Practices
- Use clear variable names (e.g., `count`, `heap`).
- For Python, use `Counter` and `heapq`.
- For JavaScript, use `Map` and priority queue library.
- For Java, use `HashMap` and `PriorityQueue`, follow Google Java Style Guide.
- Process frequencies efficiently with heap.

## Alternative Approaches
- **Sorting**: Sort by frequency (O(n log n) time). Less efficient.
- **Bucket Sort**: Use frequency buckets (O(n) time, O(n) space). Faster but more complex.