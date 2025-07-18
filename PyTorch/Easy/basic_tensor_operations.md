# Basic Tensor Operations

## Problem Statement
Write a PyTorch program to perform basic tensor operations: add two tensors and multiply the result by a scalar.

**Input**:
- Two 1D tensors `A` and `B` of equal length (e.g., `[1, 2, 3]` and `[4, 5, 6]`)
- A scalar value `s` (e.g., `2`)

**Output**:
- A tensor representing `(A + B) * s` (e.g., `[10, 14, 18]`)

**Constraints**:
- `1 <= len(A), len(B) <= 10^4`
- `-10^5 <= A[i], B[i], s <= 10^5`

## Solution
```python
import torch

def basic_tensor_operations(A: list, B: list, s: int) -> torch.Tensor:
    # Convert lists to tensors
    tensor_a = torch.tensor(A, dtype=torch.float32)
    tensor_b = torch.tensor(B, dtype=torch.float32)
    # Perform addition and scalar multiplication
    result = (tensor_a + tensor_b) * s
    return result
```

## Reasoning
- **Approach**: Convert input lists to PyTorch tensors using `torch.tensor`. Perform vectorized addition (`+`) and scalar multiplication (`*`). Return the resulting tensor.
- **Why PyTorch Tensors?**: Tensors enable efficient, vectorized operations, leveraging PyTorch’s optimized backend for CPU/GPU.
- **Edge Cases**:
  - Single-element tensors: Works as expected (e.g., `[1] + [2] * 3 = [9]`).
  - Zero scalar: Returns zero tensor.
  - Large tensors: PyTorch handles efficiently due to vectorization.
- **Optimizations**: Use `float32` for compatibility; avoid loops with vectorized operations.

## Performance Analysis
- **Time Complexity**: O(n), where n is the length of the input tensors, for vectorized addition and multiplication.
- **Space Complexity**: O(n) for storing input and output tensors.
- **PyTorch Efficiency**: Operations are optimized by PyTorch’s backend, leveraging hardware acceleration if available.

## Best Practices
- Use `torch.tensor` for input conversion.
- Specify `dtype=torch.float32` for consistency.
- Use vectorized operations to avoid loops.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **NumPy Conversion**: Convert tensors to NumPy arrays for operations (less efficient, breaks PyTorch graph).
  ```python
  import torch
  import numpy as np
  def basic_tensor_operations(A: list, B: list, s: int) -> torch.Tensor:
      tensor_a = torch.tensor(A, dtype=torch.float32)
      tensor_b = torch.tensor(B, dtype=torch.float32)
      result = torch.tensor((tensor_a.numpy() + tensor_b.numpy()) * s, dtype=torch.float32)
      return result
  ```
- **Manual Loop**: Iterate over elements (O(n), highly inefficient).
  ```python
  import torch
  def basic_tensor_operations(A: list, B: list, s: int) -> torch.Tensor:
      result = [(a + b) * s for a, b in zip(A, B)]
      return torch.tensor(result, dtype=torch.float32)
  ```