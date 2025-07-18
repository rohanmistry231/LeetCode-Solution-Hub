# Multi-Layer Perceptron

## Problem Statement
Write a PyTorch program to train a multi-layer perceptron (MLP) for multi-class classification.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of class labels (e.g., `[0, 0, 1, 2]`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained MLP model predicting class probabilities

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] < 10`
- `1 <= epochs <= 1000`

## Solution
```python
import torch
import torch.nn as nn

def multi_layer_perceptron(X: list, y: list, epochs: int) -> nn.Module:
    # Convert inputs to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Define MLP model
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 10)  # 10 classes
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = MLP()
    
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
- **Approach**: Convert inputs to tensors. Define an MLP with two hidden layers (16 and 8 units, ReLU activation) and an output layer (10 units for 10 classes). Use CrossEntropyLoss (includes softmax) and Adam optimizer. Train for `epochs` and return the model.
- **Why nn.Sequential?**: Simplifies MLP layer definition; CrossEntropyLoss is suitable for multi-class classification.
- **Edge Cases**:
  - Small dataset: Risk of overfitting, mitigated by simple architecture.
  - Unbalanced classes: CrossEntropyLoss handles imbalance reasonably.
  - Single sample: Model trains but may overfit.
- **Optimizations**: Use Adam for faster convergence; ReLU for non-linearity.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of samples and m is the number of model parameters, for forward/backward passes.
- **Space Complexity**: O(n + m), where m is the number of parameters (weights and biases).
- **PyTorch Efficiency**: `nn.Sequential` and Adam optimizer are optimized; vectorized operations avoid loops.

## Best Practices
- Use `nn.Sequential` for simple layered models.
- Specify `dtype=torch.long` for classification labels.
- Use `optimizer.zero_grad()` to clear gradients.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual Layers**: Define layers without `nn.Sequential` (more control, similar performance).
  ```python
  import torch
  import torch.nn as nn
  def multi_layer_perceptron(X: list, y: list, epochs: int) -> nn.Module:
      X = torch.tensor(X, dtype=torch.float32)
      y = torch.tensor(y, dtype=torch.long)
      class MLP(nn.Module):
          def __init__(self):
              super(MLP, self).__init__()
              self.fc1 = nn.Linear(2, 16)
              self.fc2 = nn.Linear(16, 8)
              self.fc3 = nn.Linear(8, 10)
              self.relu = nn.ReLU()
          def forward(self, x):
              x = self.relu(self.fc1(x))
              x = self.relu(self.fc2(x))
              x = self.fc3(x)
              return x
      model = MLP()
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
- **Scikit-learn MLP**: Use `MLPClassifier` (O(n * epochs), not PyTorch).
  ```python
  from sklearn.neural_network import MLPClassifier
  def multi_layer_perceptron(X: list, y: list, epochs: int) -> MLPClassifier:
      model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=epochs, learning_rate_init=0.01)
      model.fit(X, y)
      return model
  ```