# Complex RNN Prediction

## Problem Statement
Write a PyTorch program to predict the next value in a time series using a stacked RNN model with LSTM layers.

**Input**:
- `series`: 1D array of time series values (e.g., `[1, 2, 3, 4, 5, 6, 7]`)
- `time_steps`: Number of time steps for input sequences (e.g., `3`)
- Number of epochs: `epochs` (e.g., `100`)

**Output**:
- Trained stacked LSTM model

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

def complex_rnn_prediction(series: list, time_steps: int, epochs: int) -> nn.Module:
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
    
    # Define stacked LSTM model
    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size=1, hidden_size=32, batch_first=True, return_sequences=True)
            self.lstm2 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)
            self.linear = nn.Linear(16, 1)
        
        def forward(self, x):
            out, _ = self.lstm1(x)
            out, _ = self.lstm2(out)
            out = self.linear(out[:, -1, :])  # Last time step
            return out
    
    model = LSTMModel()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Reshape X for LSTM [samples, time_steps, features]
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
- **Approach**: Convert series to tensor. Create input-output sequences with `time_steps` values as input and the next value as output. Define a stacked LSTM model with two layers (32 and 16 units) and a linear output layer. Use MSE loss and Adam optimizer. Train and return the model.
- **Why Stacked LSTM?**: Multiple LSTM layers capture complex temporal dependencies; suitable for challenging time series tasks.
- **Edge Cases**:
  - Short series: Ensured by `time_steps + 1 <= len(series)`.
  - Single sequence: Model trains but may overfit.
  - Noisy data: MSE minimizes average error; LSTMs handle noise better than simple RNNs.
- **Optimizations**: Use `batch_first=True` for intuitive input shape; Adam for faster convergence.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of sequences and m is the number of model parameters, for LSTM training.
- **Space Complexity**: O(n * time_steps + m) for sequences and model parameters.
- **PyTorch Efficiency**: `nn.LSTM` and Adam are optimized for sequential data; vectorized operations enhance performance.

## Best Practices
- Use `return_sequences=True` for stacked LSTMs.
- Reshape inputs correctly (`[samples, time_steps, features]`).
- Use `mse` for regression tasks.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **GRU**: Use GRU instead of LSTM (similar performance, fewer parameters).
  ```python
  import torch
  import torch.nn as nn
  import numpy as np
  def complex_rnn_prediction(series: list, time_steps: int, epochs: int) -> nn.Module:
      series = torch.tensor(series, dtype=torch.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return torch.stack(X), torch.stack(y)
      X, y = create_sequences(series, time_steps)
      class GRUModel(nn.Module):
          def __init__(self):
              super(GRUModel, self).__init__()
              self.gru1 = nn.GRU(input_size=1, hidden_size=32, batch_first=True, return_sequences=True)
              self.gru2 = nn.GRU(input_size=32, hidden_size=16, batch_first=True)
              self.linear = nn.Linear(16, 1)
          def forward(self, x):
              out, _ = self.gru1(x)
              out, _ = self.gru2(out)
              out = self.linear(out[:, -1, :])
              return out
      model = GRUModel()
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
- **Simple RNN**: Use simpler RNN layers (faster, less robust).
  ```python
  import torch
  import torch.nn as nn
  import numpy as np
  def complex_rnn_prediction(series: list, time_steps: int, epochs: int) -> nn.Module:
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
              self.rnn1 = nn.RNN(input_size=1, hidden_size=32, batch_first=True, return_sequences=True)
              self.rnn2 = nn.RNN(input_size=32, hidden_size=16, batch_first=True)
              self.linear = nn.Linear(16, 1)
          def forward(self, x):
              out, _ = self.rnn1(x)
              out, _ = self.rnn2(out)
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