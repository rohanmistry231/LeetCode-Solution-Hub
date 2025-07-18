# Basic Linear Regression

## Problem Statement
Write a Scikit-Learn program to train a linear regression model to predict `y` from `x` using the equation `y = w * x + b`.

**Input**:
- `X`: 1D array of features (e.g., `[1, 2, 3, 4]`)
- `y`: 1D array of target values (e.g., `[2, 4, 6, 8]`)

**Output**:
- Trained model's weights `w` and bias `b`

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `-10^5 <= X[i], y[i] <= 10^5`

## Solution
```python
from sklearn.linear_model import LinearRegression
import numpy as np

def basic_linear_regression(X: list, y: list) -> tuple:
    # Convert inputs to numpy arrays
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Extract weights and bias
    w = model.coef_[0]
    b = model.intercept_
    
    return w, b
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays and reshape `X` for Scikit-Learn. Train a `LinearRegression` model using `fit`. Extract and return the slope (`coef_`) and intercept (`intercept_`).
- **Why LinearRegression?**: Scikit-Learn’s `LinearRegression` solves linear regression analytically, minimizing mean squared error efficiently.
- **Edge Cases**:
  - Single data point: Model fits but may be trivial.
  - Noisy data: Least squares method minimizes average error.
  - Constant `X`: Model fits intercept-only model.
- **Optimizations**: Reshape `X` to 2D array as required by Scikit-Learn; use analytical solution for speed.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of samples, for solving the linear regression analytically (normal equation).
- **Space Complexity**: O(n) for input arrays and model parameters.
- **Scikit-Learn Efficiency**: `LinearRegression` uses optimized linear algebra libraries (e.g., LAPACK).

## Best Practices
- Reshape `X` to 2D array (`[-1, 1]`) for Scikit-Learn compatibility.
- Use `numpy.array` for input conversion.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Gradient Descent**: Use SGDRegressor for iterative solution (O(n * iterations)).
  ```python
  from sklearn.linear_model import SGDRegressor
  import numpy as np
  def basic_linear_regression(X: list, y: list) -> tuple:
      X = np.array(X).reshape(-1, 1)
      y = np.array(y)
      model = SGDRegressor(max_iter=1000, tol=1e-3)
      model.fit(X, y)
      return model.coef_[0], model.intercept_[0]
  ```
- **Manual Solution**: Compute w, b analytically (O(n), more control).
  ```python
  import numpy as np
  def basic_linear_regression(X: list, y: list) -> tuple:
      X = np.array(X)
      y = np.array(y)
      w = np.cov(X, y)[0, 1] / np.var(X)
      b = np.mean(y) - w * np.mean(X)
      return w, b
  ```