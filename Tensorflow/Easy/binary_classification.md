# Binary Classification

## Problem Statement
Write a TensorFlow program to train a binary classifier using a single-layer neural network to predict labels (0 or 1) from features.

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
import tensorflow as tf

def binary_classification(X: list, y: list, epochs: int) -> tf.keras.Model:
    # Convert inputs to tensors
    X = tf.constant(X, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    
    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    model.fit(X, y, epochs=epochs, verbose=0)
    
    return model
```

## Reasoning
- **Approach**: Convert inputs to tensors. Build a single-layer neural network with a sigmoid activation for binary classification. Compile with binary cross-entropy loss and SGD optimizer. Train for specified `epochs` and return the trained model.
- **Why Keras Sequential?**: Simplifies model definition and training for a basic neural network.
- **Edge Cases**:
  - Small dataset: Risk of overfitting, mitigated by simple model.
  - Unbalanced labels: Cross-entropy handles imbalance reasonably.
  - Single feature pair: Model still trains but may overfit.
- **Optimizations**: Use `sigmoid` for binary output; small learning rate for stability.

## Performance Analysis
- **Time Complexity**: O(n * epochs * f), where n is the number of samples and f is the number of features (2), for forward/backward passes.
- **Space Complexity**: O(n + m), where m is the number of model parameters (small for single layer).
- **TensorFlow Efficiency**: Keras API leverages optimized backend; eager execution simplifies debugging.

## Best Practices
- Use Keras `Sequential` for simple models.
- Specify `input_shape` in first layer.
- Use `binary_crossentropy` for binary classification.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **GradientTape**: Manual training loop (more control, more complex).
  ```python
  import tensorflow as tf
  def binary_classification(X: list, y: list, epochs: int) -> tf.keras.Model:
      X = tf.constant(X, dtype=tf.float32)
      y = tf.constant(y, dtype=tf.float32)
      model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, activation='sigmoid', input_shape=(2,))])
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      for _ in range(epochs):
          with tf.GradientTape() as tape:
              y_pred = model(X)
              loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_pred))
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return model
  ```
- **Scikit-learn**: Use logistic regression (O(n), not TensorFlow).
  ```python
  from sklearn.linear_model import LogisticRegression
  def binary_classification(X: list, y: list, epochs: int) -> LogisticRegression:
      model = LogisticRegression(max_iter=epochs)
      model.fit(X, y)
      return model
  ```