# Feature Selection

## Problem Statement
Write a Scikit-Learn program to select the top k features using SelectKBest with the chi-squared test for classification.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2, 3], [2, 1, 4], [3, 3, 5], [4, 4, 6]]`)
- `y`: 1D array of class labels (e.g., `[0, 0, 1, 1]`)
- Number of features to select: `k` (e.g., `2`)

**Output**:
- Selected features (shape `(n, k)`)

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `2 <= X[i].length <= 100`
- `0 <= y[i] < 10`
- `1 <= k <= X[i].length`

## Solution
```python
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

def feature_selection(X: list, y: list, k: int) -> np.ndarray:
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Ensure non-negative values for chi2
    X = np.maximum(X, 0)
    
    # Select top k features
    selector = SelectKBest(score_func=chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays. Ensure non-negative values for chi-squared test. Use `SelectKBest` with `chi2` to select the top `k` features based on statistical significance. Return the transformed feature matrix.
- **Why SelectKBest with chi2?**: Chi-squared test is suitable for categorical labels, identifying features with strong class associations.
- **Edge Cases**:
  - Negative features: Handled by `np.maximum` to ensure chi2 compatibility.
  - `k = X[i].length`: Selects all features.
  - Single sample: Selection works but may be unreliable.
- **Optimizations**: Use `chi2` for efficiency; `np.maximum` ensures compatibility.

## Performance Analysis
- **Time Complexity**: O(n * f), where n is the number of samples and f is the number of features, for computing chi-squared scores.
- **Space Complexity**: O(n * k) for the output matrix.
- **Scikit-Learn Efficiency**: `SelectKBest` uses optimized NumPy operations for feature scoring.

## Best Practices
- Use `chi2` for non-negative classification features.
- Ensure `X` is 2D and non-negative.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **ANOVA F-test**: Use f_classif for feature selection (O(n * f), continuous features).
  ```python
  from sklearn.feature_selection import SelectKBest, f_classif
  import numpy as np
  def feature_selection(X: list, y: list, k: int) -> np.ndarray:
      X = np.array(X)
      y = np.array(y)
      selector = SelectKBest(score_func=f_classif, k=k)
      X_selected = selector.fit_transform(X, y)
      return X_selected
  ```
- **Manual Chi2**: Compute chi-squared scores manually (O(n * f), more control).
  ```python
  from scipy.stats import chi2_contingency
  import numpy as np
  def feature_selection(X: list, y: list, k: int) -> np.ndarray:
      X = np.array(X)
      y = np.array(y)
      X = np.maximum(X, 0)
      scores = []
      for i in range(X.shape[1]):
          contingency = np.histogram2d(X[:, i], y, bins=(10, len(np.unique(y))))[0]
          score = chi2_contingency(contingency)[0]
          scores.append(score)
      top_k = np.argsort(scores)[-k:]
      return X[:, top_k]
  ```