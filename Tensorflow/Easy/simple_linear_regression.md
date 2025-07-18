# Simple Linear Regression

## Problem Statement
Write a TensorFlow program to train a simple linear regression model to predict `y` from `x` using the equation `y = w * x + b`.

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
import tensorflow as tf

def simple_linear_regression(x: list, y: list, epochs: int) -> tuple:
    # Convert inputs to tensors
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    
    # Initialize weights and bias
    w = tf.Variable(0.0, dtype=tf.float32)
    b = tf.Variable(0.0, dtype=tf.float32)
    
    # Define model
    def model(x):
        return w * x + b
    
    # Define loss function (MSE)
    def loss_fn(y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    # Training loop
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        gradients = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(gradients, [w, b]))
    
    return w.numpy(), b.numpy()
```

## Reasoning
- **Approach**: Convert inputs to tensors. Initialize trainable variables `w` and `b`. Define a linear model (`w * x + b`) and mean squared error (MSE) loss. Use gradient descent (SGD) to optimize parameters over `epochs`. Return trained `w` and `b`.
- **Why GradientTape?**: Enables automatic differentiation for gradient computation in TensorFlow 2.x.
- **Edge Cases**:
  - Single data point: Model may overfit but still trains.
  - Noisy data: MSE minimizes average error.
  - Large epochs: Risk of overfitting, but constraints limit to 1000.
- **Optimizations**: Use `SGD` with small learning rate; vectorized operations for efficiency.

## Performance Analysis
- **Time Complexity**: O(n * epochs), where n is the length of input arrays, for forward/backward passes per epoch.
- **Space Complexity**: O(n) for input tensors and model variables.
- **TensorFlow Efficiency**: `GradientTape` and `SGD` are optimized for small datasets; vectorized operations avoid loops.

## Best Practices
- Use `tf.Variable` for trainable parameters.
- Specify `dtype=tf.float32` for numerical stability.
- Use Keras `SGD` optimizer for simplicity.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Keras Model**: Use Keras Sequential API (more structured, similar performance).
  ```python
  import tensorflow as tf
  def simple_linear_regression(x: list, y: list, epochs: int) -> tuple:
      x = tf.constant(x, dtype=tf.float32)
      y = tf.constant(y, dtype=tf.float32)
      model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
      model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')
      model.fit(x, y, epochs=epochs, verbose=0)
      w, b = model.get_weights()
      return w[0][0], b[0]
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