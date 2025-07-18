# Array Reshape

## Problem Statement
Write a NumPy program to reshape a 1D array into a 2D array with specified rows and columns.

**Input**:
- `arr`: 1D array of numbers (e.g., `[1, 2, 3, 4, 5, 6]`)
- `rows`: Number of rows (e.g., `2`)
- `cols`: Number of columns (e.g., `3`)

**Output**:
- 2D array of shape `(rows, cols)` (e.g., `[[1, 2, 3], [4, 5, 6]]`)

**Constraints**:
- `1 <= len(arr) <= 10^5`
- `rows * cols = len(arr)`
- `1 <= rows, cols <= 10^3`
- `-10^5 <= arr[i] <= 10^5`

## Solution
```python
import numpy as np

def array_reshape(arr: list, rows: int, cols: int) -> np.ndarray:
    # Convert to numpy array and reshape
    return np.array(arr).reshape(rows, cols)
```

## Reasoning
- **Approach**: Convert input list to NumPy array. Use `reshape` to transform into a 2D array with `rows` and `cols`. Return the result.
- **Why np.reshape?**: Efficiently reorganizes array without copying data.
- **Edge Cases**:
  - Invalid shape: Ensured by `rows * cols = len(arr)`.
  - Single element: Reshapes to `(1, 1)` if valid.
  - Large arrays: NumPy handles efficiently.
- **Optimizations**: `np.reshape` is a view operation, minimizing memory usage.

## Performance Analysis
- **Time Complexity**: O(1), as reshaping is a metadata operation.
- **Space Complexity**: O(n), where n is the array length, for the input array.
- **NumPy Efficiency**: `reshape` is highly optimized, avoiding data copying.

## Best Practices
- Use `np.reshape` for efficient array reshaping.
- Validate `rows * cols = len(arr)` before reshaping.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Reshape**: Use loops (O(n), inefficient).
  ```python
  def array_reshape(arr: list, rows: int, cols: int) -> list:
      result = []
      for i in range(rows):
          result.append(arr[i * cols:(i + 1) * cols])
      return result
  ```
- **np.array_split**: Split and stack (O(n), less direct).
  ```python
  import numpy as np
  def array_reshape(arr: list, rows: int, cols: int) -> np.ndarray:
      arr = np.array(arr)
      return np.array([arr[i * cols:(i + 1) * cols] for i in range(rows)])
  ```