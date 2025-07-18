# Statistical Computations

## Problem Statement
Write a NumPy program to compute the mean, median, and standard deviation of a 1D array.

**Input**:
- `arr`: 1D array of numbers (e.g., `[1, 2, 3, 4, 5]`)

**Output**:
- Tuple of `(mean, median, std)` (e.g., `(3.0, 3.0, 1.4142135623730951)`)

**Constraints**:
- `1 <= len(arr) <= 10^5`
- `-10^5 <= arr[i] <= 10^5`

## Solution
```python
import numpy as np

def statistical_computations(arr: list) -> tuple:
    # Convert to numpy array
    arr = np.array(arr)
    # Compute mean, median, and standard deviation
    return np.mean(arr), np.median(arr), np.std(arr)
```

## Reasoning
- **Approach**: Convert input list to NumPy array. Use `np.mean`, `np.median`, and `np.std` to compute statistics. Return as a tuple.
- **Why NumPy functions?**: Optimized for statistical computations, handling edge cases robustly.
- **Edge Cases**:
  - Single element: Mean, median equal element; std is 0.
  - Large values: NumPy uses float64 for precision.
  - Empty array: Handled by constraints.
- **Optimizations**: NumPy’s statistical functions are vectorized and efficient.

## Performance Analysis
- **Time Complexity**: O(n) for mean and std; O(n log n) for median due to sorting.
- **Space Complexity**: O(1) for computations (excluding input).
- **NumPy Efficiency**: Uses optimized C-based operations.

## Best Practices
- Use `np.mean`, `np.median`, `np.std` for statistics.
- Avoid manual loops for performance.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Computation**: Compute manually (O(n) for mean/std, O(n log n) for median, slower).
  ```python
  def statistical_computations(arr: list) -> tuple:
      n = len(arr)
      mean = sum(arr) / n
      sorted_arr = sorted(arr)
      median = sorted_arr[n // 2] if n % 2 else (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2
      std = (sum((x - mean) ** 2 for x in arr) / n) ** 0.5
      return mean, median, std
  ```
- **SciPy Stats**: Use SciPy for median (O(n log n), similar performance).
  ```python
  from scipy import stats
  import numpy as np
  def statistical_computations(arr: list) -> tuple:
      arr = np.array(arr)
      return np.mean(arr), stats.median(arr), np.std(arr)
  ```