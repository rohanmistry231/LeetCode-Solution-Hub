# Array Sum

## Problem Statement
Write a NumPy program to compute the sum of all elements in a 1D array.

**Input**:
- `arr`: 1D array of numbers (e.g., `[1, 2, 3, 4]`)

**Output**:
- Sum of all elements (e.g., `10`)

**Constraints**:
- `1 <= len(arr) <= 10^5`
- `-10^5 <= arr[i] <= 10^5`

## Solution
```python
import numpy as np

def array_sum(arr: list) -> float:
    # Convert to numpy array and compute sum
    return np.sum(arr)
```

## Reasoning
- **Approach**: Convert input list to NumPy array implicitly via `np.sum`. Use NumPy’s `sum` function to compute the total. Return the result.
- **Why np.sum?**: Optimized for fast array summation using vectorized operations.
- **Edge Cases**:
  - Empty array: `np.sum` returns 0 (handled by constraints).
  - Single element: Returns the element itself.
  - Large values: NumPy handles overflow with float64.
- **Optimizations**: `np.sum` is highly optimized, avoiding Python loops.

## Performance Analysis
- **Time Complexity**: O(n), where n is the array length, for summing elements.
- **Space Complexity**: O(1), as no additional storage is needed beyond input.
- **NumPy Efficiency**: Uses C-based vectorized operations for fast summation.

## Best Practices
- Use `np.sum` for array summation.
- Avoid Python loops for better performance.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Python Sum**: Use built-in `sum` (O(n), slower).
  ```python
  def array_sum(arr: list) -> float:
      return sum(arr)
  ```
- **Manual Loop**: Sum elements manually (O(n), least efficient).
  ```python
  def array_sum(arr: list) -> float:
      total = 0
      for x in arr:
          total += x
      return total
  ```