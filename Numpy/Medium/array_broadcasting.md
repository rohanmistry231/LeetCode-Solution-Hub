# Array Broadcasting

## Problem Statement
Write a NumPy program to perform element-wise addition of a 1D array to each row of a 2D array using broadcasting.

**Input**:
- `A`: 2D array of shape `(m, n)` (e.g., `[[1, 2, 3], [4, 5, 6]]`)
- `B`: 1D array of shape `(n,)` (e.g., `[10, 20, 30]`)

**Output**:
- 2D array of shape `(m, n)` (e.g., `[[11, 22, 33], [14, 25, 36]]`)

**Constraints**:
- `1 <= m, n <= 1000`
- `B.shape[0] = A.shape[1]`
- `-10^5 <= A[i][j], B[i] <= 10^5`

## Solution
```python
import numpy as np

def array_broadcasting(A: list, B: list) -> np.ndarray:
    # Convert to numpy arrays and broadcast
    A = np.array(A)
    B = np.array(B)
    return A + B
```

## Reasoning
- **Approach**: Convert input lists to NumPy arrays. Use NumPy’s broadcasting to add `B` to each row of `A`. Return the result.
- **Why Broadcasting?**: Automatically aligns arrays of compatible shapes, avoiding explicit loops.
- **Edge Cases**:
  - Shape mismatch: Ensured by `B.shape[0] = A.shape[1]`.
  - Single row: Broadcasting works as expected.
  - Large values: NumPy handles with float64 precision.
- **Optimizations**: Broadcasting is a zero-copy operation, highly efficient.

## Performance Analysis
- **Time Complexity**: O(m * n), where m, n are dimensions of A.
- **Space Complexity**: O(m * n) for the output array.
- **NumPy Efficiency**: Broadcasting uses C-based operations for speed.

## Best Practices
- Use broadcasting for element-wise operations.
- Ensure compatible shapes for broadcasting.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Loop**: Add manually (O(m * n), slower).
  ```python
  def array_broadcasting(A: list, B: list) -> list:
      result = [[A[i][j] + B[j] for j in range(len(B))] for i in range(len(A))]
      return result
  ```
- **np.add**: Explicit function call (O(m * n), same performance).
  ```python
  import numpy as np
  def array_broadcasting(A: list, B: list) -> np.ndarray:
      return np.add(np.array(A), np.array(B))
  ```