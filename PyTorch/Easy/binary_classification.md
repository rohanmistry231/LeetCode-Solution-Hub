# Binary Classification

## Problem Statement
Write a PyTorch program to train a binary classifier using a single-layer neural network to predict labels (0 or 1) from features.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of binary labels (e.g., `[0, 0, 1, 1]`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained model predicting probabilities

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] <= 1`
- `1 <= epochs <= 1000`

## Solution
```python
import torch
import torch.nn as nn

def binary_classification(X: list, y: list, epochs: int) -> nn.Module:
    # Convert inputs to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # Define model
    class BinaryClassifier(nn.Module):
        def __init__(self):
            super(BinaryClassifier, self).__init__()
            self.linear = nn.Linear(2, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            return self.sigmoid(self.linear(x))
    
    model = BinaryClassifier()
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    
    return model
```

## Reasoning
- **Approach**: Convert inputs to tensors. Define a single-layer neural network with a sigmoid activation for binary classification. Use BCE loss and SGD optimizer. Train for `epochs` and return the model.
- **Why nn.Module?**: Simplifies model definition and parameter management in PyTorch.
- **Edge Cases**:
  - Small dataset: Risk of overfitting, mitigated by simple model.
  - Unbalanced labels: BCE handles imbalance reasonably.
  - Single sample: Model trains but may overfit.
- **Optimizations**: Use sigmoid for binary output; small learning rate for stability.

## Performance Analysis
- **Time Complexity**: O(n * epochs * f), where n is the number of samples and f is the number of features (2), for forward/backward passes.
- **Space Complexity**: O(n + m), where m is the number of model parameters (small).
- **PyTorch Efficiency**: `nn.Linear` and `BCELoss` are optimized; SGD leverages vectorized operations.

## Best Practices
- Use `nn.Module` for model definition.
- Reshape labels for BCE loss (`[-1, 1]`).
- Use `optimizer.zero_grad()` to clear gradients.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Computation**: Compute gradients manually (more control, more complex).
  ```python
  import torch
  def binary_classification(X: list, y: list, epochs: int) -> nn.Module:
      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
      class BinaryClassifier(nn.Module):
          def __init__(self):
              super(BinaryClassifier, self).__init__()
              self.linear = nn.Linear(2, 1)
              self.sigmoid = nn.Sigmoid()
          def forward(self, x):
              return self.sigmoid(self.linear(x))
      model = BinaryClassifier()
      for _ in range(epochs):
          y_pred = model(X)
          loss = -torch.mean(y * torch.log(y_pred + 1e-10) + (1 - y) * torch.log(1 - y_pred + 1e-10))
          loss.backward()
          with torch.no_grad():
              for param in model.parameters():
                  param -= 0.01 * param.grad
                  param.grad.zero_()
      return model
  ```
- **Scikit-learn**: Use logistic regression (O(n), not PyTorch).
  ```python
  from sklearn.linear_model import LogisticRegression
  def binary_classification(X: list, y: list, epochs: int) -> LogisticRegression:
      model = LogisticRegression(max_iter=epochs)
      model.fit(X, y)
      return model
  ```