# Grid Search Tuning

## Problem Statement
Write a Scikit-Learn program to perform hyperparameter tuning for a Support Vector Machine (SVM) classifier using GridSearchCV.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of class labels (e.g., `[0, 0, 1, 1]`)
- Parameter grid: `param_grid` (e.g., `{'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}`)

**Output**:
- Trained SVM model with best hyperparameters

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] < 10`
- `param_grid` contains valid SVM parameters

## Solution
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

def grid_search_tuning(X: list, y: list, param_grid: dict) -> SVC:
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Define SVM model
    model = SVC()
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    
    # Return best model
    return grid_search.best_estimator_
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays. Initialize an SVM classifier. Use `GridSearchCV` to search `param_grid` with 5-fold cross-validation. Return the model with the best hyperparameters.
- **Why GridSearchCV?**: Automates hyperparameter tuning, optimizing model performance via cross-validation.
- **Edge Cases**:
  - Small dataset: Cross-validation ensures robust evaluation.
  - Unbalanced classes: Cross-validation mitigates bias.
  - Large `param_grid`: Increases computation, mitigated by `n_jobs=-1` for parallel processing.
- **Optimizations**: Use `n_jobs=-1` for parallel grid search; default scoring (accuracy) for classification.

## Performance Analysis
- **Time Complexity**: O(n * k * p), where n is the number of samples, k is the number of folds (5), and p is the number of parameter combinations.
- **Space Complexity**: O(n * f), where f is the number of features (2).
- **Scikit-Learn Efficiency**: `GridSearchCV` leverages parallel processing; SVM uses optimized libsvm.

## Best Practices
- Use `GridSearchCV` for hyperparameter tuning.
- Set `n_jobs=-1` for parallel computation.
- Ensure `X` is 2D and `y` is 1D.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Randomized Search**: Use RandomizedSearchCV (O(n * k * i), where i is iterations).
  ```python
  from sklearn.svm import SVC
  from sklearn.model_selection import RandomizedSearchCV
  import numpy as np
  def grid_search_tuning(X: list, y: list, param_grid: dict) -> SVC:
      X = np.array(X)
      y = np.array(y)
      model = SVC()
      random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)
      random_search.fit(X, y)
      return random_search.best_estimator_
  ```
- **Manual Tuning**: Loop over parameters manually (O(n * p), less robust).
  ```python
  from sklearn.svm import SVC
  from sklearn.model_selection import cross_val_score
  import numpy as np
  def grid_search_tuning(X: list, y: list, param_grid: dict) -> SVC:
      X = np.array(X)
      y = np.array(y)
      best_score = -np.inf
      best_model = None
      for C in param_grid['C']:
          for kernel in param_grid['kernel']:
              model = SVC(C=C, kernel=kernel)
              scores = cross_val_score(model, X, y, cv=5)
              if scores.mean() > best_score:
                  best_score = scores.mean()
                  best_model = model
      best_model.fit(X, y)
      return best_model
  ```