# Custom Model Architecture

## Problem Statement
Write a PyTorch program to build a custom neural network with skip connections for multi-class classification.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of class labels (e.g., `[0, 0, 1, 2]`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained custom model with skip connections

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] < 10`
- `1 <= epochs <= 1000`

## Solution
```python
import torch
import torch.nn as nn

def custom_model_architecture(X: list, y: list, epochs: int) -> nn.Module:
    # Convert inputs to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Define custom model with skip connections
    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.fc1 = nn.Linear(2, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(24, 8)  # 16 (from fc1) + 8 (from fc2)
            self.fc4 = nn.Linear(8, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x1 = self.relu(self.fc1(x))
            x2 = self.relu(self.fc2(x1))
            x3 = torch.cat([x1, x2], dim=1)  # Skip connection
            x4 = self.relu(self.fc3(x3))
            return self.fc4(x4)
    
    model = CustomModel()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
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
- **Approach**: Convert inputs to tensors. Define a custom model with a skip connection (concatenating the output of the first layer with the second). Use ReLU for hidden layers and a linear output layer for multi-class classification. Compile with CrossEntropyLoss and Adam. Train and return the model.
- **Why Skip Connections?**: Enhance feature reuse and gradient flow, improving training for deeper networks.
- **Edge Cases**:
  - Small dataset: Risk of overfitting, mitigated by simple architecture.
  - Unbalanced classes: CrossEntropyLoss handles imbalance.
  - Single sample: Model trains but may overfit.
- **Optimizations**: Use `torch.cat` for skip connection; Adam for fast convergence.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of samples and m is the number of parameters, for forward/backward passes.
- **Space Complexity**: O(n + m) for input tensors and model parameters.
- **PyTorch Efficiency**: `torch.cat` and Adam are optimized; skip connections improve training stability.

## Best Practices
- Define custom models with `nn.Module`.
- Use `torch.cat` for skip connections.
- Use `CrossEntropyLoss` for multi-class classification.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Additive Skip Connection**: Add instead of concatenate (requires same dimensions).
  ```python
  import torch
  import torch.nn as nn
  def custom_model_architecture(X: list, y: list, epochs: int) -> nn.Module:
      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.long)
      class CustomModel(nn.Module):
          def __init__(self):
              super(CustomModel, self).__init__()
              self.fc1 = nn.Linear(2, 16)
              self.fc2 = nn.Linear(16, 16)
              self.fc3 = nn.Linear(16, 8)
              self.fc4 = nn.Linear(8, 10)
              self.relu = nn.ReLU()
          def forward(self, x):
              x1 = self.relu(self.fc1(x))
              x2 = self.relu(self.fc2(x1))
              x3 = self.relu(x1 + x2)  # Skip connection
              x4 = self.relu(self.fc3(x3))
              return self.fc4(x4)
      model = CustomModel()
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      for _ in range(epochs):
          optimizer.zero_grad()
          y_pred = model(X)
          loss = criterion(y_pred, y)
          loss.backward()
          optimizer.step()
      return model
  ```
- **Sequential without Skip**: Simpler model without skip connections (faster, less robust).
  ```python
  import torch
  import torch.nn as nn
  def custom_model_architecture(X: list, y: list, epochs: int) -> nn.Module:
      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.long)
      class SimpleModel(nn.Module):
          def __init__(self):
              super(SimpleModel, self).__init__()
              self.layers = nn.Sequential(
                  nn.Linear(2, 16),
                  nn.ReLU(),
                  nn.Linear(16, 8),
                  nn.ReLU(),
                  nn.Linear(8, 10)
              )
          def forward(self, x):
              return self.layers(x)
      model = SimpleModel()
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      for _ in range(epochs):
          optimizer.zero_grad()
          y_pred = model(X)
          loss = criterion(y_pred, y)
          loss.backward()
          optimizer.step()
      return model
  ```