# Train-Test Split

## Problem Statement
Write a Scikit-Learn program to split a dataset into training and testing sets.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [3, 4], [5, 6], [7, 8]]`)
- `y`: 1D array of labels (e.g., `[0, 1, 0, 1]`)
- Test size ratio: `test_size` (e.g., `0.25`)

**Output**:
- Training and testing sets: `X_train`, `X_test`, `y_train`, `y_test`

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] < 10`
- `0.1 <= test_size <= 0.5`

## Solution
```python
from sklearn.model_selection import train_test_split
import numpy as np

def train_test_split_data(X: list, y: list, test_size: float) -> tuple:
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays. Use `train_test_split` to split data into training and testing sets based on `test_size`. Set `random_state=42` for reproducibility. Return the four resulting arrays.
- **Why train_test_split?**: Provides a simple, robust way to split data while maintaining label distribution.
- **Edge Cases**:
  - Small dataset: `test_size` constraints ensure sufficient training data.
  - Unbalanced labels: `train_test_split` preserves class distribution by default.
  - Single sample: Fails (handled by constraints).
- **Optimizations**: Use `random_state` for reproducibility; default shuffling ensures randomization.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of samples, for shuffling and splitting.
- **Space Complexity**: O(n * f) for input and output arrays, where f is the number of features (2).
- **Scikit-Learn Efficiency**: `train_test_split` uses NumPy for fast array operations.

## Best Practices
- Use `train_test_split` for data splitting.
- Set `random_state` for reproducibility.
- Ensure `X` is 2D and `y` is 1D.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Split**: Split data manually (O(n), less robust).
  ```python
  import numpy as np
  def train_test_split_data(X: list, y: list, test_size: float) -> tuple:
      X = np.array(X)
      y = np.array(y)
      n = len(X)
      n_test = int(n * test_size)
      indices = np.random.permutation(n)
      test_idx, train_idx = indices[:n_test], indices[n_test:]
      return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
  ```
- **Stratified Split**: Use stratified split for balanced classes (O(n), better for imbalanced data).
  ```python
  from sklearn.model_selection import train_test_split
  import numpy as np
  def train_test_split_data(X: list, y: list, test_size: float) -> tuple:
      X = np.array(X)
      y = np.array(y)
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
      return X_train, X_test, y_train, y_test
  ```