# Kth Largest Element in an Array

## Problem Statement
Given an integer array `nums` and an integer `k`, return the kth largest element in the array. The kth largest element is the kth element in the sorted array in descending order.

**Example**:
- Input: `nums = [3,2,1,5,6,4], k = 2`
- Output: `5`

**Constraints**:
- `1 <= k <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`

## Solution

### Python
```python
import heapq

def findKthLargest(nums: list[int], k: int) -> int:
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]
```

## Reasoning
- **Approach**: Use a min-heap of size k to track the k largest elements. For each element, add to the heap and remove the smallest if size exceeds k. The heap’s root is the kth largest element.
- **Why Min-Heap?**: Maintains k largest elements efficiently, with O(log k) insertion and removal, achieving O(n log k) overall.
- **Edge Cases**:
  - k=1: Return maximum.
  - k=nums.length: Return minimum.
  - Single element: Return it if k=1.
- **Optimizations**: Use min-heap to keep only k elements; avoid sorting entire array.

## Complexity Analysis
- **Time Complexity**: O(n log k), where n is the length of `nums`, as each insertion/removal is O(log k).
- **Space Complexity**: O(k), for the min-heap.

## Best Practices
- Use clear variable names (e.g., `heap`).
- For Python, use `heapq` with type hints.
- For JavaScript, use a min-priority queue library.
- For Java, use `PriorityQueue` and follow Google Java Style Guide.
- Limit heap size to k for efficiency.

## Alternative Approaches
- **Sorting**: Sort array and pick kth element (O(n log n) time). Less efficient.
- **QuickSelect**: Use partitioning to find kth largest (O(n) average time, O(1) space). More complex.