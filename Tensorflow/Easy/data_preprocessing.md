# Data Preprocessing

## Problem Statement
Write a TensorFlow program to preprocess a dataset by normalizing features (min-max scaling) and converting labels to one-hot encoding.

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
import tensorflow as tf

def data_preprocessing(X: list, y: list) -> tuple:
    # Convert to tensors
    X = tf.constant(X, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.int32)
    
    # Min-max scaling
    X_min = tf.reduce_min(X, axis=0)
    X_max = tf.reduce_max(X, axis=0)
    X_normalized = (X - X_min) / (X_max - X_min + 1e-10)
    
    # One-hot encoding
    y_one_hot = tf.one_hot(y, depth=tf.reduce_max(y) + 1)
    
    return X_normalized, y_one_hot
```

## Reasoning
- **Approach**: Convert inputs to tensors. Normalize features using min-max scaling: `(X - min) / (max - min)`. Convert labels to one-hot encoding using `tf.one_hot`. Return normalized features and one-hot labels.
- **Why Min-Max and One-Hot?**: Min-max scaling ensures features are in [0, 1], suitable for neural networks. One-hot encoding converts categorical labels for multi-class classification.
- **Edge Cases**:
  - Single feature value: Scaling handles constant features (adds small epsilon to avoid division by zero).
  - Single label: One-hot encoding works for single class.
  - Small dataset: Operations remain valid.
- **Optimizations**: Use `tf.reduce_min/max` for vectorized scaling; add epsilon (`1e-10`) for numerical stability.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of samples, for vectorized min-max and one-hot operations.
- **Space Complexity**: O(n * f + n * c), where f is the number of features (2) and c is the number of unique labels.
- **TensorFlow Efficiency**: Vectorized operations (`reduce_min/max`, `one_hot`) are optimized.

## Best Practices
- Use `tf.constant` for input data.
- Add small epsilon to avoid division by zero.
- Specify `dtype` for tensors (`float32`, `int32`).
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Scikit-learn**: Use `MinMaxScaler` and `OneHotEncoder` (O(n), not TensorFlow).
  ```python
  from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
  import numpy as np
  def data_preprocessing(X: list, y: list) -> tuple:
      X = np.array(X)
      y = np.array(y).reshape(-1, 1)
      X_normalized = MinMaxScaler().fit_transform(X)
      y_one_hot = OneHotEncoder(sparse=False).fit_transform(y)
      return X_normalized, y_one_hot
  ```
- **Manual Scaling**: Compute min/max manually (O(n), less efficient).
  ```python
  import tensorflow as tf
  def data_preprocessing(X: list, y: list) -> tuple:
      X = tf.constant(X, dtype=tf.float32)
      y = tf.constant(y, dtype=tf.int32)
      X_min = tf.reduce_min(X, axis=0)
      X_max = tf.reduce_max(X, axis=0)
      X_normalized = tf.where(X_max == X_min, X, (X - X_min) / (X_max - X_min))
      y_one_hot = tf.one_hot(y, depth=tf.reduce_max(y) + 1)
      return X_normalized, y_one_hot
  ```