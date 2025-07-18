# Element-Wise Multiplication

## Problem Statement
Write a NumPy program to perform element-wise multiplication of two 1D arrays.

**Input**:
- `arr1`: 1D array of numbers (e.g., `[1, 2, 3]`)
- `arr2`: 1D array of numbers (e.g., `[4, 5, 6]`)

**Output**:
- 1D array of element-wise products (e.g., `[4, 10, 18]`)

**Constraints**:
- `1 <= len(arr1) = len(arr2) <= 10^5`
- `-10^5 <= arr1[i], arr2[i] <= 10^5`

## Solution
```python
import numpy as np

def element_wise_multiplication(arr1: list, arr2: list) -> np.ndarray:
    # Convert to numpy arrays and multiply
    return np.array(arr1) * np.array(arr2)
```

## Reasoning
- **Approach**: Convert input lists to NumPy arrays. Use NumPy’s element-wise multiplication (`*`) operator. Return the result.
- **Why NumPy multiplication?**: Vectorized operation is faster than Python loops.
- **Edge Cases**:
  - Same length: Ensured by constraints.
  - Zero elements: Returns element-wise product (e.g., `[0, 0]`).
  - Large values: NumPy handles with float64 precision.
- **Optimizations**: `np.array` multiplication is C-based and vectorized.

## Performance Analysis
- **Time Complexity**: O(n), where n is the array length, for element-wise multiplication.
- **Space Complexity**: O(n) for the output array.
- **NumPy Efficiency**: Uses optimized C operations for fast computation.

## Best Practices
- Use NumPy’s `*` for element-wise operations.
- Ensure input arrays have the same shape.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Python List Comprehension**: Manual multiplication (O(n), slower).
  ```python
  def element_wise_multiplication(arr1: list, arr2: list) -> list:
      return [a * b for a, b in zip(arr1, arr2)]
  ```
- **np.multiply**: Explicit function call (O(n), same performance).
  ```python
  import numpy as np
  def element_wise_multiplication(arr1: list, arr2: list) -> np.ndarray:
      return np.multiply(np.array(arr1), np.array(arr2))
  ```