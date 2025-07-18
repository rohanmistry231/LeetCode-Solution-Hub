# Time Series Prediction

## Problem Statement
Write a TensorFlow program to predict the next value in a time series using a simple Recurrent Neural Network (RNN).

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
import tensorflow as tf
import numpy as np

def time_series_prediction(series: list, time_steps: int, epochs: int) -> tf.keras.Model:
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
    
    # Define RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(10, activation='tanh', input_shape=(time_steps, 1)),
        tf.keras.layers.Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 loss='mse')
    
    # Reshape X for RNN [samples, time_steps, features]
    X = X.reshape(-1, time_steps, 1)
    
    # Train model
    model.fit(X, y, epochs=epochs, verbose=0)
    
    return model
```

## Reasoning
- **Approach**: Convert series to tensor. Create input-output sequences where each input is `time_steps` values and the output is the next value. Build a simple RNN with 10 units (tanh activation) and a dense output layer. Compile with MSE loss and Adam optimizer. Train and return the model.
- **Why SimpleRNN?**: Suitable for basic time series prediction; captures sequential dependencies.
- **Edge Cases**:
  - Short series: Ensured by `time_steps + 1 <= len(series)`.
  - Single sequence: Model trains but may overfit.
  - Noisy data: MSE minimizes average error.
- **Optimizations**: Use `tanh` for stable gradients; reshape data for RNN input.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of sequences and m is the number of model parameters, for RNN training.
- **Space Complexity**: O(n * time_steps + m) for sequences and model parameters.
- **TensorFlow Efficiency**: `SimpleRNN` and Adam are optimized; sequence creation is efficient with NumPy.

## Best Practices
- Reshape inputs correctly for RNN (`[samples, time_steps, features]`).
- Use `tanh` activation for RNN stability.
- Use `mse` for regression tasks.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **LSTM**: Use LSTM instead of SimpleRNN (more robust, higher complexity).
  ```python
  import tensorflow as tf
  import numpy as np
  def time_series_prediction(series: list, time_steps: int, epochs: int) -> tf.keras.Model:
      series = tf.constant(series, dtype=tf.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return np.array(X), np.array(y)
      X, y = create_sequences(series, time_steps)
      model = tf.keras.Sequential([
          tf.keras.layers.LSTM(10, activation='tanh', input_shape=(time_steps, 1)),
          tf.keras.layers.Dense(1)
      ])
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
      X = X.reshape(-1, time_steps, 1)
      model.fit(X, y, epochs=epochs, verbose=0)
      return model
  ```
- **Manual Sequence Prediction**: Predict without RNN (O(n), not sequential).
  ```python
  import tensorflow as tf
  def time_series_prediction(series: list, time_steps: int, epochs: int) -> tf.keras.Model:
      series = tf.constant(series, dtype=tf.float32)
      def create_sequences(series, time_steps):
          X, y = [], []
          for i in range(len(series) - time_steps):
              X.append(series[i:i + time_steps])
              y.append(series[i + time_steps])
          return np.array(X), np.array(y)
      X, y = create_sequences(series, time_steps)
      model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(time_steps,))])
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
      model.fit(X, y, epochs=epochs, verbose=0)
      return model
  ```