# Time Series Prediction

## Problem Statement
Write a PyTorch program to predict the next value in a time series using a simple Recurrent Neural Network (RNN).

**Input**:
- `series`: 1D array of time series values (e.g., `[1, 2, 3, 4, 5]`)
- `time_steps`: Number of time steps for input sequences (e.g., `3`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained RNN model

**Constraints**:
- `time_steps + 1 <= len(series) <= 10^4`
- `-10^5 <= series[i] <= 10^5`
- `1 <= time_steps <= 10`
- `1 <= epochs <= 1000`

## Solution
```python
import torch
import torch.nn as nn
import numpy as np

def time_series_prediction(series: list, time_steps: int, epochs: int) -> nn.Module:
    # Convert series to tensor
    series = torch.tensor(series, dtype=torch.float32)
    
    # Create sequences
    def create_sequences(series, time_steps):
        X, y = [], []
        for i in range(len(series) - time_steps):
            X.append(series[i:i + time_steps])
            y.append(series[i + time_steps])
        return torch.stack(X), torch.stack(y)
    
    X, y = create_sequences(series, time_steps)
    
    # Define RNN model
    class RNNModel(nn.Module):
        def __init__(self):
            super(RNNModel, self).__init__()
            self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.linear(out[:, -1, :])  # Last time step
            return out
    
    model = RNNModel()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Reshape X for RNN [samples, time_steps, features]
    X = X.reshape(-1, time_steps, 1)
    
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
- **Approach**: Convert series to tensor. Create input-output sequences where each input is `time_steps` values and the output is the next value. Define an RNN model with 10 hidden units and a linear output layer. Use MSE loss and Adam optimizer. Train and return the model.
- **Why RNN?**: Captures sequential dependencies for time series prediction.
- **Edge Cases**:
  - Short series: Ensured by `time_steps + 1 <= len(series)`.
  - Single sequence: Model trains but may overfit.
  - Noisy data: MSE minimizes average error.
- **Optimizations**: Use `batch_first=True` for RNN; reshape data for RNN input.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of sequences and m is the number of model parameters, for RNN training.
- **Space Complexity**: O(n * time_steps + m) for sequences and model parameters.
- **PyTorch Efficiency**: `nn.RNN` and Adam are optimized; sequence creation is efficient with `torch.stack`.

## Best Practices
- Use `batch_first=True` for intuitive RNN input shape.
- Reshape inputs correctly (`[samples, time_steps, features]`).
- Use `mse` for regression tasks.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **LSTM**: Use LSTM instead of RNN (more robust, higher complexity).
  ```python
  import torch
  import torch.nn as nn
  import numpy as np
  def time_series_prediction(series: list, time_steps: int, epochs: int) -> nn.Module:
      series = torch.tensor(series, dtype=torch.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return torch.stack(X), torch.stack(y)
      X, y = create_sequences(series, time_steps)
      class RNNModel(nn.Module):
          def __init__(self):
              super(RNNModel, self).__init__()
              self.lstm = nn.LSTM(input_size=1, hidden_size=10, batch_first=True)
              self.linear = nn.Linear(10, 1)
          def forward(self, x):
              out, _ = self.lstm(x)
              out = self.linear(out[:, -1, :])
              return out
      model = RNNModel()
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      X = X.reshape(-1, time_steps, 1)
      for _ in range(epochs):
          optimizer.zero_grad()
          y_pred = model(X)
          loss = criterion(y_pred, y)
          loss.backward()
          optimizer.step()
      return model
  ```
- **Manual Sequence Prediction**: Predict without RNN (O(n), not sequential).
  ```python
  import torch
  import torch.nn as nn
  import numpy as np
  def time_series_prediction(series: list, time_steps: int, epochs: int) -> nn.Module:
      series = torch.tensor(series, dtype=torch.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return torch.stack(X), torch.stack(y)
      X, y = create_sequences(series, time_steps)
      class LinearModel(nn.Module):
          def __init__(self):
              super(LinearModel, self).__init__()
              self.linear = nn.Linear(time_steps, 1)
          def forward(self, x):
              return self.linear(x)
      model = LinearModel()
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      for _ in range(epochs):
          optimizer.zero_grad()
          y_pred = model(X)
          loss = criterion(y_pred, y)
          loss.backward()
          optimizer.step()
      return model
  ```