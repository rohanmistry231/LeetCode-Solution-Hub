# Daily Temperatures

## Problem Statement
Given an array of integers `temperatures` representing daily temperatures, return an array `answer` such that `answer[i]` is the number of days until a warmer day (i.e., the first day with a higher temperature). If no such day exists, set `answer[i] = 0`.

**Example**:
- Input: `temperatures = [73,74,75,71,69,72,76,73]`
- Output: `[1,1,4,2,1,1,0,0]`

**Constraints**:
- `1 <= temperatures.length <= 10^5`
- `30 <= temperatures[i] <= 100`

## Solution

### JavaScript
```javascript
function dailyTemperatures(temperatures) {
    const n = temperatures.length;
    const answer = new Array(n).fill(0);
    const stack = [];
    
    for (let i = 0; i < n; i++) {
        while (stack.length && temperatures[i] > temperatures[stack[stack.length - 1]]) {
            const prevIndex = stack.pop();
            answer[prevIndex] = i - prevIndex;
        }
        stack.push(i);
    }
    
    return answer;
}
```

## Reasoning
- **Approach**: Use a monotonic stack to track indices of temperatures in decreasing order. For each temperature, pop indices from the stack where the temperature is lower, calculating the day difference. Push the current index onto the stack.
- **Why Monotonic Stack?**: Efficiently finds the next warmer day in one pass by maintaining a stack of indices with decreasing temperatures.
- **Edge Cases**:
  - Single day: Return [0].
  - No warmer day: Answer remains 0 (default).
  - All same temperature: All answers are 0.
- **Optimizations**: Single pass; use stack to avoid redundant comparisons.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `temperatures`, as each index is pushed and popped at most once.
- **Space Complexity**: O(n), for the stack and output array.

## Best Practices
- Use clear variable names (e.g., `stack`, `prevIndex`).
- For Python, use type hints and list as stack.
- For JavaScript, use array as stack with `push`/`pop`.
- For Java, use `Deque` and follow Google Java Style Guide.
- Initialize output array with zeros for simplicity.

## Alternative Approaches
- **Brute Force**: For each day, scan forward for a warmer day (O(n^2) time). Too slow.
- **Array Scan**: Store indices and scan right (O(n^2) worst case). Inefficient.