# Cross-Validation with Nested Grid Search

## Problem Statement
Write a Scikit-Learn program to perform nested cross-validation for hyperparameter tuning of an SVM classifier.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of class labels (e.g., `[0, 0, 1, 1]`)
- Parameter grid: `param_grid` (e.g., `{'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}`)
- Outer CV folds: `outer_cv` (e.g., `5`)

**Output**:
- Trained SVM model with best hyperparameters

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] < 10`
- `2 <= outer_cv <= 10`
- `param_grid` contains valid SVM parameters

## Solution
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

def cross_validation_nested_grid_search(X: list, y: list, param_grid: dict, outer_cv: int) -> SVC:
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Inner grid search
    inner_grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    
    # Perform nested cross-validation
    scores = cross_val_score(inner_grid, X, y, cv=outer_cv, n_jobs=-1)
    
    # Train final model with best parameters
    inner_grid.fit(X, y)
    
    return inner_grid.best_estimator_
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays. Use `GridSearchCV` for inner loop hyperparameter tuning with 5-fold CV. Evaluate performance with outer loop `cross_val_score` using `outer_cv` folds. Train final model on full data with best parameters. Return the best estimator.
- **Why Nested CV?**: Provides unbiased performance estimation by separating hyperparameter tuning (inner loop) from model evaluation (outer loop).
- **Edge Cases**:
  - Small dataset: Cross-validation ensures robust evaluation.
  - Unbalanced classes: Cross-validation mitigates bias.
  - Large `param_grid`: Increases computation, mitigated by `n_jobs=-1`.
- **Optimizations**: Use `n_jobs=-1` for parallel processing in both loops.

## Performance Analysis
- **Time Complexity**: O(n * k_outer * k_inner * p), where n is the number of samples, k_outer is outer folds, k_inner is inner folds (5), and p is parameter combinations.
- **Space Complexity**: O(n * f), where f is the number of features (2).
- **Scikit-Learn Efficiency**: `GridSearchCV` and `cross_val_score` leverage parallel processing and optimized libsvm.

## Best Practices
- Use nested CV for unbiased evaluation.
- Set `n_jobs=-1` for parallel computation.
- Ensure `X` is 2D and `y` is 1D.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Randomized Search**: Use RandomizedSearchCV in inner loop (O(n * k_outer * i), where i is iterations).
  ```python
  from sklearn.svm import SVC
  from sklearn.model_selection import RandomizedSearchCV, cross_val_score
  import numpy as np
  def cross_validation_nested_grid_search(X: list, y: list, param_grid: dict, outer_cv: int) -> SVC:
      X = np.array(X)
      y = np.array(y)
      inner_grid = RandomizedSearchCV(SVC(), param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)
      scores = cross_val_score(inner_grid, X, y, cv=outer_cv, n_jobs=-1)
      inner_grid.fit(X, y)
      return inner_grid.best_estimator_
  ```
- **Manual Nested CV**: Implement loops manually (O(n * k_outer * k_inner * p), more control).
  ```python
  from sklearn.svm import SVC
  from sklearn.model_selection import KFold
  import numpy as np
  def cross_validation_nested_grid_search(X: list, y: list, param_grid: dict, outer_cv: int) -> SVC:
      X = np.array(X)
      y = np.array(y)
      outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
      best_model = None
      best_score = -np.inf
      for train_idx, test_idx in outer_kf.split(X):
          X_train, X_test = X[train_idx], X[test_idx]
          y_train, y_test = y[train_idx], y[test_idx]
          inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
          for C in param_grid['C']:
              for kernel in param_grid['kernel']:
                  model = SVC(C=C, kernel=kernel)
                  scores = []
                  for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                      model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                      scores.append(model.score(X_train[inner_val_idx], y_train[inner_val_idx]))
                  if np.mean(scores) > best_score:
                      best_score = np.mean(scores)
                      best_model = SVC(C=C, kernel=kernel)
      best_model.fit(X, y)
      return best_model
  ```