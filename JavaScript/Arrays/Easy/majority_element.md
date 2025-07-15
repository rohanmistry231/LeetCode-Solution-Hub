# Majority Element

## Problem Statement
Given an array `nums` of size `n`, return the majority element. The majority element is the element that appears more than `⌊n/2⌋` times. You may assume the majority element always exists.

**Example**:
- Input: `nums = [2,2,1,1,1,2,2]`
- Output: `2`
- Explanation: `2` appears 4 times, which is more than `n/2 = 3.5`.

**Constraints**:
- `1 <= nums.length <= 5 * 10^4`
- `-10^9 <= nums[i] <= 10^9`

## Solution

### JavaScript
```javascript
function majorityElement(nums) {
    let count = 0;
    let candidate = 0;
    for (const num of nums) {
        if (count === 0) {
            candidate = num;
        }
        count += (num === candidate ? 1 : -1);
    }
    return candidate;
}
```

## Reasoning
- **Approach**: Use Boyer-Moore Voting Algorithm. Since the majority element appears more than `n/2` times, maintain a candidate and count. Increment count when seeing the candidate, decrement otherwise. When count reaches 0, pick a new candidate. The final candidate is the majority element.
- **Why Boyer-Moore?**: It guarantees the majority element in a single pass without extra space, leveraging the fact that the majority element outweighs others.
- **Edge Cases**:
  - Single element: Returns that element.
  - Majority element exists: Guaranteed by problem constraints.
- **Optimizations**: Single pass with minimal variables; no need for validation due to problem assumptions.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `nums`. Single pass through the array.
- **Space Complexity**: O(1), as only two variables are used.

## Best Practices
- Use clear variable names (e.g., `candidate`, `count`).
- For Python, include type hints for clarity.
- For JavaScript, use `for...of` for readability.
- For Java, follow Google Java Style Guide.
- Keep logic simple to leverage the algorithm’s elegance.

## Alternative Approaches
- **Hash Map**: Count frequencies and return the element with count > `n/2` (O(n) time, O(n) space).
- **Sorting**: Sort the array and return the middle element (O(n log n) time, O(1) space). Less efficient than Boyer-Moore.