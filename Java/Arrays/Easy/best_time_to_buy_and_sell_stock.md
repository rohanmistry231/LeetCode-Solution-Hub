# Best Time to Buy and Sell Stock

## Problem Statement
Given an array `prices` where `prices[i]` is the price of a stock on day `i`, return the maximum profit you can achieve from a single buy and sell transaction. You must buy before selling. If no profit is possible, return 0.

**Example**:
- Input: `prices = [7,1,5,3,6,4]`
- Output: `5`
- Explanation: Buy on day 2 (`price = 1`) and sell on day 5 (`price = 6`), profit = `6 - 1 = 5`.

**Constraints**:
- `1 <= prices.length <= 10^5`
- `0 <= prices[i] <= 10^4`

## Solution

### Java
```java
class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int price : prices) {
            minPrice = Math.min(minPrice, price);
            maxProfit = Math.max(maxProfit, price - minPrice);
        }
        return maxProfit;
    }
}
```

## Reasoning
- **Approach**: Track the minimum price seen so far and the maximum profit possible by selling at the current price. Iterate through the array once, updating the minimum price and checking if the current price yields a higher profit.
- **Why One Pass?**: By maintaining the minimum price, we can compute potential profits in a single pass, avoiding the need to check all pairs (O(n²)).
- **Edge Cases**:
  - Single price: Returns 0 (no transaction possible).
  - Decreasing prices: Returns 0 (no profit possible).
  - Large arrays: Single pass ensures efficiency.
- **Optimizations**: Avoid unnecessary variables; update `min_price` and `max_profit` in one loop.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `prices`. Single pass through the array.
- **Space Complexity**: O(1), as only two variables are used.

## Best Practices
- Use descriptive variable names (e.g., `min_price`, `max_profit`).
- For Python, use `float('inf')` for initial max/min values and include type hints.
- For JavaScript, use `Infinity` and modern `for...of` loops for readability.
- For Java, use `Integer.MAX_VALUE` and follow Google Java Style Guide.
- Keep logic simple and avoid nested loops for efficiency.

## Alternative Approaches
- **Brute Force**: Check all possible buy-sell pairs (O(n²) time, O(1) space). Inefficient for large inputs.
- **Kadane’s Algorithm Variant**: Treat the problem as finding the maximum subarray sum of price differences (similar complexity but less intuitive here).