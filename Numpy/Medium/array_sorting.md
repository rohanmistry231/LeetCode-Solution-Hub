# Array Sorting

## Problem Statement
Write a NumPy program to sort a 2D array by a specific column.

**Input**:
- `arr`: 2D array of shape `(m, n)` (e.g., `[[3, 1], [1, 2], [2, 3]]`)
- `col`: Column index to sort by (e.g., `1`)

**Output**:
- Sorted 2D array (e.g., `[[3, 1], [1, 2], [2, 3]]`)

**Constraints**:
- `1 <= m, n <= 1000`
- `0 <= col < n`
- `-10^5 <= arr[i][j] <= 10^5`

## Solution
```python
import numpy as np

def array_sorting(arr: list, col: int) -> np.ndarray:
    # Convert to numpy array and sort by column
    return np.array(arr)[np.argsort(np.array(arr)[:, col])]
```

## Reasoning
- **Approach**: Convert input list to NumPy array. Use `np.argsort` on the specified column to get sorted indices. Index the array with these indices to sort rows. Return the result.
- **Why np.argsort?**: Efficiently sorts indices, preserving row integrity.
- **Edge Cases**:
  - Single row: Returns same array.
  - Invalid column: Ensured by `col < n`.
  - Ties in column: `np.argsort` uses stable sorting.
- **Optimizations**: `np.argsort` is optimized for performance.

## Performance Analysis
- **Time Complexity**: O(m log m), where m is the number of rows, for sorting.
- **Space Complexity**: O(m) for the index array and output.
- **NumPy Efficiency**: `np.argsort` uses optimized C-based sorting.

## Best Practices
- Use `np.argsort` for sorting by column.
- Validate `col` to avoid out-of-bounds errors.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Sorting**: Sort manually (O(m log m), slower).
  ```python
  def array_sorting(arr: list, col: int) -> list:
      return sorted(arr, key=lambda x: x[col])
  ```
- **np.sort**: Use structured sorting (O(m log m), similar performance).
  ```python
  import numpy as np
  def array_sorting(arr: list, col: int) -> np.ndarray:
      arr = np.array(arr)
      return arr[arr[:, col].argsort()]
  ```