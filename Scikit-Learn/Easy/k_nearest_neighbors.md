# K-Nearest Neighbors

## Problem Statement
Write a Scikit-Learn program to train a K-Nearest Neighbors (KNN) classifier for binary classification.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of binary labels (e.g., `[0, 0, 1, 1]`)
- Number of neighbors: `k` (e.g., `3`)

**Output**:
- Trained KNN classifier

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] <= 1`
- `1 <= k <= len(X)`

## Solution
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def k_nearest_neighbors(X: list, y: list, k: int) -> KNeighborsClassifier:
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Train KNN classifier
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    
    return model
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays. Train a `KNeighborsClassifier` with `k` neighbors using `fit`. Return the trained model.
- **Why KNN?**: Simple, non-parametric algorithm suitable for binary classification; effective for small datasets with clear separation.
- **Edge Cases**:
  - Single sample: Model fits but prediction is trivial.
  - `k = len(X)`: Uses all points, may overfit.
  - Unbalanced classes: KNN can be biased; weights can be adjusted.
- **Optimizations**: Use default Euclidean distance; Scikit-Learn’s KNN uses efficient nearest neighbor search.

## Performance Analysis
- **Time Complexity**: O(n * log n) for training (building KD-tree), O(k * n) for prediction per sample.
- **Space Complexity**: O(n * f), where f is the number of features (2).
- **Scikit-Learn Efficiency**: Uses optimized KD-tree or Ball-tree for fast neighbor search.

## Best Practices
- Ensure `X` is 2D and `y` is 1D for Scikit-Learn.
- Validate `k` to avoid overfitting or underfitting.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Weighted KNN**: Use distance-weighted voting (similar complexity, potentially better accuracy).
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  import numpy as np
  def k_nearest_neighbors(X: list, y: list, k: int) -> KNeighborsClassifier:
      X = np.array(X)
      y = np.array(y)
      model = KNeighborsClassifier(n_neighbors=k, weights='distance')
      model.fit(X, y)
      return model
  ```
- **Manual KNN**: Compute distances manually (O(n^2), inefficient).
  ```python
  import numpy as np
  from collections import Counter
  def k_nearest_neighbors(X: list, y: list, k: int):
      class ManualKNN:
          def __init__(self, k):
              self.k = k
          def fit(self, X, y):
              self.X = np.array(X)
              self.y = np.array(y)
          def predict(self, X_test):
              distances = np.sqrt(((self.X - X_test[:, np.newaxis]) ** 2).sum(axis=2))
              nearest_indices = np.argsort(distances, axis=1)[:, :self.k]
              nearest_labels = self.y[nearest_indices]
              return [Counter(labels).most_common(1)[0][0] for labels in nearest_labels]
      model = ManualKNN(k)
      model.fit(X, y)
      return model
  ```