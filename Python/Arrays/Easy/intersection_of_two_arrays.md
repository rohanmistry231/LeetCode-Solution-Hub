# Intersection of Two Arrays

## Problem Statement
Given two integer arrays `nums1` and `nums2`, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays, and you may return the result in any order.

**Example**:
- Input: `nums1 = [1,2,2,1], nums2 = [2,2]`
- Output: `[2,2]`

**Constraints**:
- `1 <= nums1.length, nums2.length <= 10^4`
- `0 <= nums1[i], nums2[i] <= 10^9`

## Solution

### Python
```python
def intersect(nums1: list[int], nums2: list[int]) -> list[int]:
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    count = {}
    for num in nums1:
        count[num] = count.get(num, 0) + 1
    result = []
    for num in nums2:
        if num in count and count[num] > 0:
            result.append(num)
            count[num] -= 1
    return result
```

## Reasoning
- **Approach**: Use a hash map to count frequencies in the smaller array, then iterate through the larger array to find common elements, decrementing counts. Swap arrays if `nums1` is larger to optimize space.
- **Why Hash Map?**: Efficiently tracks frequencies and handles duplicates correctly. Swapping ensures minimal space usage.
- **Edge Cases**:
  - Empty arrays: Return empty array.
  - No intersection: Return empty array.
  - Duplicate elements: Count-based approach handles multiplicity.
- **Optimizations**: Process the smaller array first to reduce hash map size; single pass for each array.

## Complexity Analysis
- **Time Complexity**: O(n + m), where n and m are lengths of `nums1` and `nums2`. Two passes through the arrays.
- **Space Complexity**: O(min(n, m)) for the hash map, plus O(min(n, m)) for the result array.

## Best Practices
- Use clear variable names (e.g., `count`, `result`).
- For Python, use `get` method for safe dictionary access and type hints.
- For JavaScript, use `Map` for frequency counting and modern loops.
- For Java, use `HashMap` and `ArrayList` for dynamic results, following Google Java Style Guide.
- Optimize space by processing the smaller array first.

## Alternative Approaches
- **Sorting + Two Pointers**: Sort both arrays and use two pointers to find common elements (O(n log n + m log m) time, O(1) space excluding output).
- **Brute Force**: Check each element of `nums1` against `nums2` (O(n * m) time, O(1) space). Inefficient for large arrays.