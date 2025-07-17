# Path Sum III

## Problem Statement
Given the root of a binary tree and an integer `targetSum`, return the number of paths that sum to `targetSum`. A path can start and end at any node and must go downward.

**Example**:
- Input: `root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8`
- Output: `3`
- Explanation: Paths [5,3], [5,2,1], [-3,11] sum to 8.

**Constraints**:
- The number of nodes is in the range `[1, 10^4]`.
- `-10^9 <= Node.val <= 10^9`
- `-10^9 <= targetSum <= 10^9`

## Solution

### Python
```python
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        prefix_sums = {0: 1}
        
        def dfs(node, curr_sum):
            if not node:
                return 0
            curr_sum += node.val
            count = prefix_sums.get(curr_sum - targetSum, 0)
            
            prefix_sums[curr_sum] = prefix_sums.get(curr_sum, 0) + 1
            count += dfs(node.left, curr_sum) + dfs(node.right, curr_sum)
            prefix_sums[curr_sum] -= 1
            if prefix_sums[curr_sum] == 0:
                del prefix_sums[curr_sum]
                
            return count
        
        return dfs(root, 0)
```

## Reasoning
- **Approach**: Use a prefix sum technique with a hash map to count paths. Track the cumulative sum along each path. For each node, check if `currSum - targetSum` exists in the hash map to count valid paths ending at the current node. Update the hash map and recurse, cleaning up after to avoid double-counting.
- **Why Prefix Sums?**: Allows efficient counting of all downward paths summing to `targetSum` by tracking cumulative sums.
- **Edge Cases**:
  - Single node: Check if node.val equals targetSum.
  - Negative values: Handled by prefix sums.
  - Empty tree: Handled by constraints (non-empty).
- **Optimizations**: Use hash map for O(1) lookups; clean up prefix sums to save space.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the number of nodes, as each node is visited once.
- **Space Complexity**: O(h), where h is the tree height, for the recursion stack and hash map (O(n) in worst case for skewed tree).

## Best Practices
- Use clear variable names (e.g., `prefixSums`, `currSum`).
- For Python, use type hints and dictionary.
- For JavaScript, use `Map` for key-value pairs.
- For Java, use `HashMap` with `long` for sums and follow Google Java Style Guide.
- Clean up hash map to minimize space.

## Alternative Approaches
- **Brute Force**: Check all possible paths from each node (O(n^2) time). Too slow.
- **Two-Pass DFS**: Compute paths separately (O(n^2) time). Inefficient.