# Data Preprocessing

## Problem Statement
Write a PyTorch program to preprocess a dataset by normalizing features (min-max scaling) and converting labels to one-hot encoding.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [3, 4], [5, 6]]`)
- `y`: 1D array of categorical labels (e.g., `[0, 1, 2]`)

**Output**:
- Normalized features (values in [0, 1])
- One-hot encoded labels

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] < 10`

## Solution
```python
import torch

def data_preprocessing(X: list, y: list) -> tuple:
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Min-max scaling
    X_min = torch.min(X, dim=0).values
    X_max = torch.max(X, dim=0).values
    X_normalized = (X - X_min) / (X_max - X_min + 1e-10)
    
    # One-hot encoding
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=int(torch.max(y).item()) + 1)
    
    return X_normalized, y_one_hot
```

## Reasoning
- **Approach**: Convert inputs to tensors. Normalize features using min-max scaling: `(X - min) / (max - min)`. Convert labels to one-hot encoding using `torch.nn.functional.one_hot`. Return normalized features and one-hot labels.
- **Why Min-Max and One-Hot?**: Min-max scaling ensures features are in [0, 1], suitable for neural networks. One-hot encoding converts categorical labels for multi-class classification.
- **Edge Cases**:
  - Single feature value: Scaling handles constant features (adds small epsilon to avoid division by zero).
  - Single label: One-hot encoding works for single class.
  - Small dataset: Operations remain valid.
- **Optimizations**: Use `torch.min/max` for vectorized scaling; add epsilon (`1e-10`) for numerical stability.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of samples, for vectorized min-max and one-hot operations.
- **Space Complexity**: O(n * f + n * c), where f is the number of features (2) and c is the number of unique labels.
- **PyTorch Efficiency**: `torch.min/max` and `one_hot` are optimized for parallel execution.

## Best Practices
- Use `torch.tensor` for input conversion.
- Add small epsilon to avoid division by zero.
- Specify `dtype` (`float32` for features, `long` for labels).
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Scikit-learn**: Use `MinMaxScaler` and `OneHotEncoder` (O(n), not PyTorch).
  ```python
  from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
  import numpy as np
  def data_preprocessing(X: list, y: list) -> tuple:
      X = np.array(X)
      y = np.array(y).reshape(-1, 1)
      X_normalized = MinMaxScaler().fit_transform(X)
      y_one_hot = OneHotEncoder(sparse_output=False).fit_transform(y)
      return torch.tensor(X_normalized, dtype=torch.float32), torch.tensor(y_one_hot, dtype=torch.float32)
  ```
- **Manual Scaling**: Compute min/max manually (O(n), less efficient).
  ```python
  import torch
  def data_preprocessing(X: list, y: list) -> tuple:
      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.long)
      X_min = torch.min(X, dim=0).values
      X_max = torch.max(X, dim=0).values
      X_normalized = torch.where(X_max == X_min, X, (X - X_min) / (X_max - X_min))
      y_one_hot = torch.nn.functional.one_hot(y, num_classes=int(torch.max(y).item()) + 1)
      return X_normalized, y_one_hot
  ```