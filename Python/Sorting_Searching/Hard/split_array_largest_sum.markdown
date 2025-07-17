# Split Array Largest Sum

## Problem Statement
Given an array `nums` and an integer `m`, split the array into `m` non-empty subarrays such that the largest sum of any subarray is minimized. Return the minimized largest sum.

**Example**:
- Input: `nums = [7,2,5,10,8], m = 2`
- Output: `18`
- Explanation: Split into [7,2,5] and [10,8], largest sum = 18.

**Constraints**:
- `1 <= nums.length <= 1000`
- `0 <= nums[i] <= 10^6`
- `1 <= m <= min(50, nums.length)`

## Solution

### Python
```python
def splitArray(nums: list[int], m: int) -> int:
    def canSplit(max_sum: int) -> bool:
        count, curr_sum = 1, 0
        for num in nums:
            curr_sum += num
            if curr_sum > max_sum:
                count += 1
                curr_sum = num
                if count > m:
                    return False
        return True
    
    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if canSplit(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

## Reasoning
- **Approach**: Use binary search to find the minimum largest sum. Set `left` as the maximum element (minimum possible sum) and `right` as the total sum (maximum possible sum). For each `mid`, check if the array can be split into `m` subarrays with sums <= `mid`. Adjust the range accordingly.
- **Why Binary Search?**: Efficiently finds the minimum valid sum in O(log(sum(nums))) time.
- **Edge Cases**:
  - m=1: Return sum of array.
  - m=nums.length: Return maximum element.
  - Large numbers: Handle with long integers if needed.
- **Optimizations**: Use binary search to minimize checks; validate splits in O(n) per iteration.

## Complexity Analysis
- **Time Complexity**: O(n log S), where n is the length of `nums` and S is the sum of `nums`. Binary search takes O(log S), and each check is O(n).
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `left`, `maxSum`).
- For Python, use type hints and helper function.
- For JavaScript, use spread operator for max.
- For Java, follow Google Java Style Guide and modularize check logic.
- Optimize range for binary search.

## Alternative Approaches
- **Dynamic Programming**: Use DP to compute minimum largest sum (O(n^2 * m) time). Too slow.
- **Greedy**: Try splitting greedily (O(n log n) time). Complex and less reliable.