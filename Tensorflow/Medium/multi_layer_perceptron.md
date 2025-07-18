# Multi-Layer Perceptron

## Problem Statement
Write a TensorFlow program to train a multi-layer perceptron (MLP) for multi-class classification.

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
import tensorflow as tf

def multi_layer_perceptron(X: list, y: list, epochs: int) -> tf.keras.Model:
    # Convert inputs to tensors
    X = tf.constant(X, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.int32)
    
    # Define MLP model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    model.fit(X, y, epochs=epochs, verbose=0)
    
    return model
```

## Reasoning
- **Approach**: Convert inputs to tensors. Build an MLP with two hidden layers (16 and 8 units, ReLU activation) and an output layer (10 units, softmax for multi-class). Compile with sparse categorical cross-entropy loss and Adam optimizer. Train for specified `epochs` and return the model.
- **Why MLP with Keras?**: Keras `Sequential` API simplifies building and training neural networks; multiple layers improve feature learning for classification.
- **Edge Cases**:
  - Small dataset: Risk of overfitting, mitigated by simple architecture.
  - Unbalanced classes: Cross-entropy handles imbalance reasonably.
  - Single sample: Model trains but may overfit.
- **Optimizations**: Use Adam for faster convergence; ReLU for non-linearity; softmax for probability outputs.

## Performance Analysis
- **Time Complexity**: O(n * epochs * m), where n is the number of samples and m is the number of model parameters, for forward/backward passes.
- **Space Complexity**: O(n + m), where m is the number of parameters (weights and biases).
- **TensorFlow Efficiency**: Keras leverages optimized backend; Adam optimizer is efficient for gradient descent.

## Best Practices
- Use Keras `Sequential` for layered models.
- Specify `input_shape` in first layer.
- Use `sparse_categorical_crossentropy` for integer labels.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **GradientTape**: Manual training loop (more control, more complex).
  ```python
  import tensorflow as tf
  def multi_layer_perceptron(X: list, y: list, epochs: int) -> tf.keras.Model:
      X = tf.constant(X, dtype=tf.float32)
      y = tf.constant(y, dtype=tf.int32)
      model = tf.keras.Sequential([
          tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
          tf.keras.layers.Dense(8, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
      ])
      optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
      for _ in range(epochs):
          with tf.GradientTape() as tape:
              y_pred = model(X)
              loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, y_pred))
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      return model
  ```
- **Scikit-learn MLP**: Use `MLPClassifier` (O(n * epochs), not TensorFlow).
  ```python
  from sklearn.neural_network import MLPClassifier
  def multi_layer_perceptron(X: list, y: list, epochs: int) -> MLPClassifier:
      model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=epochs, learning_rate_init=0.01)
      model.fit(X, y)
      return model
  ```