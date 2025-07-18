# Eigenvalue Computation

## Problem Statement
Write a NumPy program to compute the eigenvalues of a square matrix.

**Input**:
- `A`: 2D array of shape `(n, n)` (e.g., `[[1, 2], [3, 4]]`)

**Output**:
- 1D array of eigenvalues (e.g., `[-0.37228132, 5.37228132]`)

**Constraints**:
- `1 <= n <= 1000`
- `-10^5 <= A[i][j] <= 10^5`
- `A` is a square matrix

## Solution
```python
import numpy as np

def eigenvalue_computation(A: list) -> np.ndarray:
    # Convert to numpy array and compute eigenvalues
    return np.linalg.eigvals(np.array(A))
```

## Reasoning
- **Approach**: Convert input list to NumPy array. Use `np.linalg.eigvals` to compute eigenvalues. Return the result.
- **Why np.linalg.eigvals?**: Optimized for eigenvalue computation using LAPACK, suitable for square matrices.
- **Edge Cases**:
  - 1x1 matrix: Returns single eigenvalue.
  - Non-square matrix: Ensured by constraints.
  - Singular matrix: Returns valid eigenvalues (may include zeros).
- **Optimizations**: `np.linalg.eigvals` uses efficient linear algebra routines.

## Performance Analysis
- **Time Complexity**: O(n^3), where n is the matrix dimension, for eigenvalue computation.
- **Space Complexity**: O(n) for the output array.
- **NumPy Efficiency**: Uses LAPACK for optimized linear algebra.

## Best Practices
- Use `np.linalg.eigvals` for eigenvalue computation.
- Ensure input is a square matrix.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Eigenvalues**: Solve characteristic polynomial (O(n^3), less robust).
  ```python
  import numpy as np
  def eigenvalue_computation(A: list) -> np.ndarray:
      A = np.array(A)
      n = A.shape[0]
      # Compute characteristic polynomial coefficients
      coeffs = np.poly(A)
      # Find roots (eigenvalues)
      return np.roots(coeffs)
  ```
- **Full Eigen Decomposition**: Use `np.linalg.eig` (O(n^3), includes eigenvectors).
  ```python
  import numpy as np
  def eigenvalue_computation(A: list) -> np.ndarray:
      return np.linalg.eig(np.array(A))[0]
  ```