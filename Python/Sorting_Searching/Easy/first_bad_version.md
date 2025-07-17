# First Bad Version

## Problem Statement
You are a product manager and currently leading a team to develop a new product. Since each version is developed based on the previous version, all versions after a bad version are also bad. Given `n` versions `[1, 2, ..., n]`, find the first bad version using a function `isBadVersion(version)` that returns whether a version is bad.

**Example**:
- Input: `n = 5`, bad version = 4
- Output: `4`
- Explanation: `isBadVersion(3) -> false`, `isBadVersion(4) -> true`, so 4 is the first bad version.

**Constraints**:
- `1 <= n <= 2^31 - 1`

## Solution

### Python
```python
# The isBadVersion API is defined as: def isBadVersion(version: int) -> bool
def firstBadVersion(n: int) -> int:
    left, right = 1, n
    
    while left < right:
        mid = left + (right - left) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    
    return left
```

## Reasoning
- **Approach**: Use binary search to find the first bad version. Treat versions as a sorted array where false (good) precedes true (bad). Find the leftmost true by adjusting the search range: if `isBadVersion(mid)` is true, search left half (including mid); else, search right half.
- **Why Binary Search?**: Efficiently finds the boundary in O(log n) time, minimizing API calls.
- **Edge Cases**:
  - n=1: Check directly.
  - First version bad: Return 1.
  - Last version bad: Return n.
- **Optimizations**: Use `left + (right - left) / 2` to avoid overflow; return `left` as it converges to the first bad version.

## Complexity Analysis
- **Time Complexity**: O(log n), where n is the number of versions, as the search space halves each iteration.
- **Space Complexity**: O(1), as only a few variables are used.

## Best Practices
- Use clear variable names (e.g., `left`, `right`).
- For Python, use type hints for clarity.
- For JavaScript, wrap solution in a closure for API context.
- For Java, follow Google Java Style Guide and extend parent class.
- Minimize API calls with binary search.

## Alternative Approaches
- **Linear Search**: Check versions sequentially (O(n) time). Too slow for large n.
- **Recursive Binary Search**: Same logic recursively (O(log n) time, O(log n) space). Less space-efficient.