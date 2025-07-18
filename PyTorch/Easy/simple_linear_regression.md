# Simple Linear Regression

## Problem Statement
Write a PyTorch program to train a simple linear regression model to predict `y` from `x` using the equation `y = w * x + b`.

**Input**:
- `x`: 1D array of features (e.g., `[1, 2, 3, 4]`)
- `y`: 1D array of labels (e.g., `[2, 4, 6, 8]`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained model parameters `w` and `b`

**Constraints**:
- `1 <= len(x), len(y) <= 10^4`
- `-10^5 <= x[i], y[i] <= 10^5`
- `1 <= epochs <= 1000`

## Solution
```python
import torch
import torch.nn as nn

def simple_linear_regression(x: list, y: list, epochs: int) -> tuple:
    # Convert inputs to tensors
    x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Define model
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(1, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = LinearRegression()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    
    # Extract weights and bias
    w = model.linear.weight.item()
    b = model.linear.bias.item()
    
    return w, b
```

## Reasoning
- **Approach**: Convert inputs to tensors and reshape for model input. Define a linear regression model using `nn.Linear`. Use MSE loss and SGD optimizer. Train for `epochs`, then extract and return `w` and `b`.
- **Why nn.Module?**: Provides a structured way to define models and handle parameters in PyTorch.
- **Edge Cases**:
  - Single data point: Model may overfit but trains.
  - Noisy data: MSE minimizes average error.
  - Large epochs: Risk of overfitting, but constraints limit to 1000.
- **Optimizations**: Use SGD with small learning rate; reshape tensors for proper input dimensions.

## Performance Analysis
- **Time Complexity**: O(n * epochs), where n is the length of input arrays, for forward/backward passes per epoch.
- **Space Complexity**: O(n) for input tensors and model parameters.
- **PyTorch Efficiency**: `nn.Linear` and `MSELoss` are optimized; SGD leverages vectorized operations.

## Best Practices
- Use `nn.Module` for model definition.
- Specify `dtype=torch.float32` for numerical stability.
- Use `optimizer.zero_grad()` to clear gradients.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Computation**: Compute gradients manually (more control, more complex).
  ```python
  import torch
  def simple_linear_regression(x: list, y: list, epochs: int) -> tuple:
      x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
      y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
      w = torch.tensor(0.0, requires_grad=True)
      b = torch.tensor(0.0, requires_grad=True)
      for _ in range(epochs):
          y_pred = w * x + b
          loss = torch.mean((y_pred - y) ** 2)
          loss.backward()
          with torch.no_grad():
              w -= 0.01 * w.grad
              b -= 0.01 * b.grad
              w.grad.zero_()
              b.grad.zero_()
      return w.item(), b.item()
  ```
- **NumPy**: Solve analytically (O(n), not iterative).
  ```python
  import numpy as np
  def simple_linear_regression(x: list, y: list, epochs: int) -> tuple:
      x = np.array(x)
      y = np.array(y)
      w = np.cov(x, y)[0, 1] / np.var(x)
      b = np.mean(y) - w * np.mean(x)
      return w, b
  ```