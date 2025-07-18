# Matrix Multiplication

## Problem Statement
Write a NumPy program to perform matrix multiplication of two 2D arrays.

**Input**:
- `A`: 2D array of shape `(m, n)` (e.g., `[[1, 2], [3, 4]]`)
- `B`: 2D array of shape `(n, p)` (e.g., `[[5, 6], [7, 8]]`)

**Output**:
- 2D array of shape `(m, p)` (e.g., `[[19, 22], [43, 50]]`)

**Constraints**:
- `1 <= m, n, p <= 1000`
- `A.shape[1] = B.shape[0]`
- `-10^5 <= A[i][j], B[i][j] <= 10^5`

## Solution
```python
import numpy as np

def matrix_multiplication(A: list, B: list) -> np.ndarray:
    # Convert to numpy arrays and perform matrix multiplication
    return np.matmul(np.array(A), np.array(B))
```

## Reasoning
- **Approach**: Convert input lists to NumPy arrays. Use `np.matmul` for matrix multiplication. Return the result.
- **Why np.matmul?**: Optimized for matrix operations, leveraging BLAS for high performance.
- **Edge Cases**:
  - Invalid shapes: Ensured by `A.shape[1] = B.shape[0]`.
  - Single row/column: Handled as valid matrices.
  - Large values: NumPy uses float64, avoiding overflow.
- **Optimizations**: `np.matmul` is highly optimized, avoiding manual loops.

## Performance Analysis
- **Time Complexity**: O(m * n * p), where m, n, p are dimensions of the matrices.
- **Space Complexity**: O(m * p) for the output matrix.
- **NumPy Efficiency**: `np.matmul` uses BLAS for optimized matrix operations.

## Best Practices
- Use `np.matmul` for matrix multiplication.
- Validate matrix shapes before operation.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **np.dot**: Alternative for matrix multiplication (O(m * n * p), same performance).
  ```python
  import numpy as np
  def matrix_multiplication(A: list, B: list) -> np.ndarray:
      return np.dot(np.array(A), np.array(B))
  ```
- **Manual Loop**: Compute manually (O(m * n * p), inefficient).
  ```python
  def matrix_multiplication(A: list, B: list) -> list:
      m, n = len(A), len(A[0])
      n, p = len(B), len(B[0])
      result = [[0] * p for _ in range(m)]
      for i in range(m):
          for j in range(p):
              for k in range(n):
                  result[i][j] += A[i][k] * B[k][j]
      return result
  ```