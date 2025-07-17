# Sliding Window Maximum

## Problem Statement
Given an array `nums` and an integer `k`, return the maximum element in each sliding window of size `k` as it slides from left to right.

**Example**:
- Input: `nums = [1,3,-1,-3,5,3,6,7], k = 3`
- Output: `[3,3,5,5,6,7]`

**Constraints**:
- `1 <= nums.length <= 10^5`
- `-10^4 <= nums[i] <= 10^4`
- `1 <= k <= nums.length`

## Solution

### Python
```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    result = []
    dq = deque()
    
    for i in range(len(nums)):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

## Reasoning
- **Approach**: Use a deque to maintain indices of potential maximums in a decreasing order. For each index, remove out-of-window indices from the front and smaller elements from the back. Add the current index and store the front element as the maximum when the window is full.
- **Why Deque?**: Allows O(1) access to both ends, maintaining a monotonic queue of indices for efficient maximum tracking.
- **Edge Cases**:
  - k=1: Return original array.
  - Single window: Return maximum of first k elements.
  - All same values: Return array of that value.
- **Optimizations**: Use deque for O(1) operations; single pass through array.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`, as each index is pushed and popped at most once.
- **Space Complexity**: O(k), for the deque, plus O(n-k+1) for the output array.

## Best Practices
- Use clear variable names (e.g., `dq`, `result`).
- For Python, use `deque` and type hints.
- For JavaScript, use array as deque with `shift`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Maintain monotonic property for efficiency.

## Alternative Approaches
- **Brute Force**: Find max for each window (O(n*k) time). Too slow.
- **Priority Queue**: Use max heap for windows (O(n log k) time). Less efficient.