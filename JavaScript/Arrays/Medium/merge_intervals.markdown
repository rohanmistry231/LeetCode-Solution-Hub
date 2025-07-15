# Merge Intervals

## Problem Statement
Given an array of intervals where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals and return an array of the non-overlapping intervals that cover all the intervals in the input.

**Example**:
- Input: `intervals = [[1,3],[2,6],[8,10],[15,18]]`
- Output: `[[1,6],[8,10],[15,18]]`
- Explanation: Since intervals `[1,3]` and `[2,6]` overlap, merge them into `[1,6]`.

**Constraints**:
- `1 <= intervals.length <= 10^4`
- `intervals[i].length == 2`
- `0 <= start_i <= end_i <= 10^4`

## Solution

### JavaScript
```javascript
function merge(intervals) {
    intervals.sort((a, b) => a[0] - b[0]);
    const result = [];
    for (const interval of intervals) {
        if (!result.length || result[result.length - 1][1] < interval[0]) {
            result.push(interval);
        } else {
            result[result.length - 1][1] = Math.max(result[result.length - 1][1], interval[1]);
        }
    }
    return result;
}
```

## Reasoning
- **Approach**: Sort intervals by start time. Iterate through sorted intervals, merging overlapping ones by updating the end time of the last interval in the result if it overlaps with the current interval. Add non-overlapping intervals directly to the result.
- **Why Sort?**: Sorting ensures we process intervals in order, making it easier to identify overlaps (if current start ≤ previous end).
- **Edge Cases**:
  - Single interval: Return as is.
  - No overlaps: Return sorted intervals.
  - All overlapping: Merge into one interval.
- **Optimizations**: Single pass after sorting; in-place merging by updating the last interval’s end.

## Complexity Analysis
- **Time Complexity**: O(n log n), where n is the number of intervals, due to sorting. The merging pass is O(n).
- **Space Complexity**: O(n) for the output array. Sorting may use O(log n) or O(n) depending on the language’s sort implementation.

## Best Practices
- Use clear variable names (e.g., `result`, `interval`).
- For Python, use type hints and `lambda` for sorting.
- For JavaScript, use arrow functions for concise sorting.
- For Java, use `ArrayList` for dynamic results and follow Google Java Style Guide.
- Sort first to simplify overlap checks.

## Alternative Approaches
- **Brute Force**: Check each interval against all others for overlaps (O(n²) time, O(n) space). Inefficient for large inputs.
- **Interval Tree**: Use a tree structure for overlaps (O(n log n) time, O(n) space). Overkill for this problem.