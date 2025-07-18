# Array Indexing

## Problem Statement
Write a NumPy program to extract elements from a 1D array using a list of indices.

**Input**:
- `arr`: 1D array of numbers (e.g., `[10, 20, 30, 40, 50]`)
- `indices`: 1D array of indices (e.g., `[0, 2, 4]`)

**Output**:
- 1D array of elements at specified indices (e.g., `[10, 30, 50]`)

**Constraints**:
- `1 <= len(arr) <= 10^5`
- `1 <= len(indices) <= len(arr)`
- `0 <= indices[i] < len(arr)`
- `-10^5 <= arr[i] <= 10^5`

## Solution
```python
import numpy as np

def array_indexing(arr: list, indices: list) -> np.ndarray:
    # Convert to numpy arrays and index
    return np.array(arr)[indices]
```

## Reasoning
- **Approach**: Convert input list to NumPy array. Use NumPy’s advanced indexing with `indices` to extract elements. Return the result.
- **Why NumPy indexing?**: Fast and concise compared to Python list indexing.
- **Edge Cases**:
  - Empty indices: Returns empty array.
  - Single index: Returns single-element array.
  - Invalid indices: Ensured by constraints.
- **Optimizations**: NumPy’s indexing is optimized for performance.

## Performance Analysis
- **Time Complexity**: O(k), where k is the length of indices, for indexing operation.
- **Space Complexity**: O(k) for the output array.
- **NumPy Efficiency**: Advanced indexing is implemented in C for speed.

## Best Practices
- Use NumPy’s advanced indexing for efficiency.
- Validate indices to avoid out-of-bounds errors.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Python List Indexing**: Manual indexing (O(k), slower).
  ```python
  def array_indexing(arr: list, indices: list) -> list:
      return [arr[i] for i in indices]
  ```
- **np.take**: Explicit indexing function (O(k), same performance).
  ```python
  import numpy as np
  def array_indexing(arr: list, indices: list) -> np.ndarray:
      return np.take(np.array(arr), indices)
  ```