# Custom Loss Function

## Problem Statement
Write a TensorFlow program to train a regression model with a custom loss function: mean squared error with an additional penalty for predictions below a threshold (e.g., 0).

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
import tensorflow as tf

def custom_loss_function(X: list, y: list, epochs: int) -> tf.keras.Model:
    # Convert inputs to tensors
    X = tf.constant(X, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    
    # Define custom loss: MSE + penalty for predictions < 0
    def custom_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        penalty = tf.reduce_mean(tf.where(y_pred < 0, tf.square(y_pred), 0.0))
        return mse + 0.1 * penalty
    
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=(1,))
    ])
    
    # Compile model with custom loss
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                 loss=custom_loss)
    
    # Train model
    model.fit(X, y, epochs=epochs, verbose=0)
    
    return model
```

## Reasoning
- **Approach**: Convert inputs to tensors. Define a custom loss function combining mean squared error (MSE) with a penalty for negative predictions (`y_pred < 0`). Build a single-layer regression model. Compile with the custom loss and SGD optimizer. Train for `epochs` and return the model.
- **Why Custom Loss?**: Allows penalizing undesirable predictions (e.g., negative values) to guide training.
- **Edge Cases**:
  - All positive predictions: Penalty is 0, reduces to MSE.
  - Small dataset: Risk of overfitting, mitigated by simple model.
  - Zero targets: Penalty still applies if predictions are negative.
- **Optimizations**: Use `tf.where` for vectorized penalty; small penalty weight (0.1) balances loss terms.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of samples and m is the number of parameters, for forward/backward passes.
- **Space Complexity**: O(n + m), where m is the number of model parameters (small).
- **TensorFlow Efficiency**: Custom loss is vectorized; `tf.where` is optimized for conditionals.

## Best Practices
- Define custom loss as a function for reusability.
- Use `tf.where` for vectorized conditional logic.
- Specify `input_shape` in first layer.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **GradientTape**: Manual training with custom loss (more control, more complex).
  ```python
  import tensorflow as tf
  def custom_loss_function(X: list, y: list, epochs: int) -> tf.keras.Model:
      X = tf.constant(X, dtype=tf.float32)
      y = tf.constant(y, dtype=tf.float32)
      model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(1,))])
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      for _ in range(epochs):
          with tf.GradientTape() as tape:
              y_pred = model(X)
              mse = tf.reduce_mean(tf.square(y - y_pred))
              penalty = tf.reduce_mean(tf.where(y_pred < 0, tf.square(y_pred), 0.0))
              loss = mse + 0.1 * penalty
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return model
  ```
- **Keras Custom Loss Class**: Define loss as a class (more reusable, similar performance).
  ```python
  import tensorflow as tf
  class CustomLoss(tf.keras.losses.Loss):
      def call(self, y_true, y_pred):
          mse = tf.reduce_mean(tf.square(y_true - y_pred))
          penalty = tf.reduce_mean(tf.where(y_pred < 0, tf.square(y_pred), 0.0))
          return mse + 0.1 * penalty
  def custom_loss_function(X: list, y: list, epochs: int) -> tf.keras.Model:
      X = tf.constant(X, dtype=tf.float32)
      y = tf.constant(y, dtype=tf.float32)
      model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(1,))])
      model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=CustomLoss())
      model.fit(X, y, epochs=epochs, verbose=0)
      return model
  ```