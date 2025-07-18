# Sparse Matrix Multiplication

## Problem Statement
Write a NumPy program to multiply two sparse matrices efficiently.

**Input**:
- `A`: 2D sparse matrix as list of `(row, col, value)` tuples (e.g., `[(0, 1, 2), (1, 0, 3)]`)
- `B`: 2D sparse matrix as list of `(row, col, value)` tuples (e.g., `[(0, 0, 5), (1, 1, 6)]`)
- `m`, `n`, `p`: Dimensions (A: `m x n`, B: `n x p`)

**Output**:
- Sparse matrix as list of `(row, col, value)` tuples

**Constraints**:
- `1 <= m, n, p <= 1000`
- `A.shape[1] = B.shape[0]`
- `-10^5 <= value <= 10^5`
- Number of non-zero elements <= 10^5

## Solution
```python
import numpy as np
from scipy.sparse import csr_matrix

def sparse_matrix_multiplication(A: list, B: list, m: int, n: int, p: int) -> list:
    # Convert tuples to sparse matrices
    rows_A, cols_A, vals_A = zip(*A) if A else ([], [], [])
    rows_B, cols_B, vals_B = zip(*B) if B else ([], [], [])
    
    A_sparse = csr_matrix((vals_A, (rows_A, cols_A)), shape=(m, n))
    B_sparse = csr_matrix((vals_B, (rows_B, cols_B)), shape=(n, p))
    
    # Multiply sparse matrices
    result = A_sparse @ B_sparse
    
    # Convert back to tuple list
    rows, cols = result.nonzero()
    vals = result.data
    return [(r, c, v) for r, c, v in zip(rows, cols, vals)]
```

## Reasoning
- **Approach**: Convert tuple lists to sparse matrices using `csr_matrix`. Perform multiplication with `@`. Extract non-zero elements as tuples. Return the result.
- **Why csr_matrix?**: Compressed Sparse Row format is efficient for matrix operations, reducing memory and computation for sparse data.
- **Edge Cases**:
  - Empty matrices: Returns empty list if no non-zero results.
  - Single non-zero element: Handled correctly.
  - Shape mismatch: Ensured by constraints.
- **Optimizations**: `csr_matrix` minimizes memory; sparse multiplication skips zero entries.

## Performance Analysis
- **Time Complexity**: O(nnz_A * nnz_B / n), where nnz_A, nnz_B are non-zero elements, for sparse multiplication.
- **Space Complexity**: O(nnz_result) for output, where nnz_result is non-zero elements in result.
- **NumPy/SciPy Efficiency**: `csr_matrix` and `@` use optimized sparse algorithms.

## Best Practices
- Use `csr_matrix` for sparse matrix operations.
- Validate input shapes and indices.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Dense Multiplication**: Convert to dense and multiply (O(m * n * p), inefficient).
  ```python
  import numpy as np
  def sparse_matrix_multiplication(A: list, B: list, m: int, n: int, p: int) -> list:
      A_dense = np.zeros((m, n))
      B_dense = np.zeros((n, p))
      for r, c, v in A:
          A_dense[r, c] = v
      for r, c, v in B:
          B_dense[r, c] = v
      result = np.matmul(A_dense, B_dense)
      rows, cols = np.nonzero(result)
      return [(r, c, result[r, c]) for r, c in zip(rows, cols)]
  ```
- **Manual Sparse**: Compute manually (O(nnz_A * nnz_B), complex).
  ```python
  import numpy as np
  def sparse_matrix_multiplication(A: list, B: list, m: int, n: int, p: int) -> list:
      result = {}
      for r1, c1, v1 in A:
          for r2, c2, v2 in B:
              if c1 == r2:
                  r, c = r1, c2
                  result[(r, c)] = result.get((r, c), 0) + v1 * v2
      return [(r, c, v) for (r, c), v in result.items() if v != 0]
  ```