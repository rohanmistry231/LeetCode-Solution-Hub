# Complex RNN Prediction

## Problem Statement
Write a TensorFlow program to predict the next value in a time series using a stacked RNN model with LSTM layers.

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
import tensorflow as tf
import numpy as np

def complex_rnn_prediction(series: list, time_steps: int, epochs: int) -> tf.keras.Model:
    # Convert series to tensor
    series = tf.constant(series, dtype=tf.float32)
    
    # Create sequences
    def create_sequences(series, time_steps):
        X, y = [], []
        for i in range(len(series) - time_steps):
            X.append(series[i:i + time_steps])
            y.append(series[i + time_steps])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(series, time_steps)
    
    # Define stacked LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(time_steps, 1)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 loss='mse')
    
    # Reshape X for LSTM [samples, time_steps, features]
    X = X.reshape(-1, time_steps, 1)
    
    # Train model
    model.fit(X, y, epochs=epochs, verbose=0)
    
    return model
```

## Reasoning
- **Approach**: Convert series to tensor. Create input-output sequences with `time_steps` values as input and the next value as output. Build a stacked LSTM model with two layers (32 and 16 units) and a dense output layer. Compile with MSE loss and Adam optimizer. Train and return the model.
- **Why Stacked LSTM?**: Multiple LSTM layers capture complex temporal dependencies; suitable for hard time series tasks.
- **Edge Cases**:
  - Short series: Ensured by `time_steps + 1 <= len(series)`.
  - Single sequence: Model trains but may overfit.
  - Noisy data: MSE minimizes average error; LSTMs handle noise better than simple RNNs.
- **Optimizations**: Use Adam for faster convergence; reshape data correctly for LSTM input.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of sequences and m is the number of model parameters, for LSTM training.
- **Space Complexity**: O(n * time_steps + m) for sequences and model parameters.
- **TensorFlow Efficiency**: LSTMs are optimized for sequential data; Adam optimizer enhances convergence.

## Best Practices
- Use `return_sequences=True` for stacked LSTMs.
- Reshape inputs correctly (`[samples, time_steps, features]`).
- Use `mse` for regression tasks.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **GRU**: Use GRU instead of LSTM (similar performance, fewer parameters).
  ```python
  import tensorflow as tf
  import numpy as np
  def complex_rnn_prediction(series: list, time_steps: int, epochs: int) -> tf.keras.Model:
      series = tf.constant(series, dtype=tf.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return np.array(X), np.array(y)
      X, y = create_sequences(series, time_steps)
      model = tf.keras.Sequential([
          tf.keras.layers.GRU(32, return_sequences=True, input_shape=(time_steps, 1)),
          tf.keras.layers.GRU(16),
          tf.keras.layers.Dense(1)
      ])
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
      X = X.reshape(-1, time_steps, 1)
      model.fit(X, y, epochs=epochs, verbose=0)
      return model
  ```
- **Simple RNN**: Use simpler RNN layers (faster, less robust).
  ```python
  import tensorflow as tf
  import numpy as np
  def complex_rnn_prediction(series: list, time_steps: int, epochs: int) -> tf.keras.Model:
      series = tf.constant(series, dtype=tf.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return np.array(X), np.array(y)
      X, y = create_sequences(series, time_steps)
      model = tf.keras.Sequential([
          tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=(time_steps, 1)),
          tf.keras.layers.SimpleRNN(16),
          tf.keras.layers.Dense(1)
      ])
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
      X = X.reshape(-1, time_steps, 1)
      model.fit(X, y, epochs=epochs, verbose=0)
      return model
  ```