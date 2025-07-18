# Custom Loss Function

## Problem Statement
Write a PyTorch program to train a regression model with a custom loss function: mean squared error with an additional penalty for predictions below a threshold (e.g., 0).

**Input**:
- `X`: 1D array of features (e.g., `[1, 2, 3, 4]`)
- `y`: 1D array of target values (e.g., `[2, 4, 6, 8]`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained model

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `-10^5 <= X[i], y[i] <= 10^5`
- `1 <= epochs <= 1000`

## Solution
```python
import torch
import torch.nn as nn

def custom_loss_function(X: list, y: list, epochs: int) -> nn.Module:
    # Convert inputs to tensors
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Define custom loss: MSE + penalty for predictions < 0
    def custom_loss(y_pred, y_true):
        mse = torch.mean((y_pred - y_true) ** 2)
        penalty = torch.mean(torch.where(y_pred < 0, y_pred ** 2, torch.tensor(0.0)))
        return mse + 0.1 * penalty
    
    # Define model
    class RegressionModel(nn.Module):
        def __init__(self):
            super(RegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    model = RegressionModel()
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = custom_loss(y_pred, y)
        loss.backward()
        optimizer.step()
    
    return model
```

## Reasoning
- **Approach**: Convert inputs to tensors and reshape for model input. Define a custom loss function combining MSE with a penalty for negative predictions (`y_pred < 0`). Build a single-layer regression model. Use SGD optimizer. Train for `epochs` and return the model.
- **Why Custom Loss?**: Penalizes undesirable predictions (e.g., negative values) to guide training.
- **Edge Cases**:
  - All positive predictions: Penalty is 0, reduces to MSE.
  - Small dataset: Risk of overfitting, mitigated by simple model.
  - Zero targets: Penalty applies if predictions are negative.
- **Optimizations**: Use `torch.where` for vectorized penalty; small penalty weight (0.1) balances loss terms.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of samples and m is the number of parameters, for forward/backward passes.
- **Space Complexity**: O(n + m), where m is the number of model parameters (small).
- **PyTorch Efficiency**: `torch.where` is optimized for conditionals; `nn.Linear` is efficient.

## Best Practices
- Define custom loss as a function for reusability.
- Use `torch.where` for vectorized conditional logic.
- Reshape inputs for model compatibility.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Gradient**: Compute gradients manually (more control, more complex).
  ```python
  import torch
  def custom_loss_function(X: list, y: list, epochs: int) -> nn.Module:
      X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
      y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
      class RegressionModel(nn.Module):
          def __init__(self):
              super(RegressionModel, self).__init__()
              self.linear = nn.Linear(1, 1)
          def forward(self, x):
              return self.linear(x)
      model = RegressionModel()
      for _ in range(epochs):
          y_pred = model(X)
          mse = torch.mean((y_pred - y) ** 2)
          penalty = torch.mean(torch.where(y_pred < 0, y_pred ** 2, torch.tensor(0.0)))
          loss = mse + 0.1 * penalty
          loss.backward()
          with torch.no_grad():
              for param in model.parameters():
                  param -= 0.01 * param.grad
                  param.grad.zero_()
      return model
  ```
- **Custom Loss Module**: Define loss as a module (more reusable, similar performance).
  ```python
  import torch
  import torch.nn as nn
  class CustomLoss(nn.Module):
      def __init__(self):
          super(CustomLoss, self).__init__()
      def forward(self, y_pred, y_true):
          mse = torch.mean((y_pred - y_true) ** 2)
          penalty = torch.mean(torch.where(y_pred < 0, y_pred ** 2, torch.tensor(0.0)))
          return mse + 0.1 * penalty
  def custom_loss_function(X: list, y: list, epochs: int) -> nn.Module:
      X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
      y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
      class RegressionModel(nn.Module):
          def __init__(self):
              super(RegressionModel, self).__init__()
              self.linear = nn.Linear(1, 1)
          def forward(self, x):
              return self.linear(x)
      model = RegressionModel()
      criterion = CustomLoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
      for _ in range(epochs):
          optimizer.zero_grad()
          y_pred = model(X)
          loss = criterion(y_pred, y)
          loss.backward()
          optimizer.step()
      return model
  ```