# Fourier Transform

## Problem Statement
Write a NumPy program to apply Fast Fourier Transform (FFT) to a 1D signal.

**Input**:
- `signal`: 1D array of numbers (e.g., `[1, 2, 3, 4]`)

**Output**:
- 1D array of complex FFT coefficients

**Constraints**:
- `1 <= len(signal) <= 10^5`
- `-10^5 <= signal[i] <= 10^5`

## Solution
```python
import numpy as np

def fourier_transform(signal: list) -> np.ndarray:
    # Convert to numpy array and compute FFT
    return np.fft.fft(np.array(signal))
```

## Reasoning
- **Approach**: Convert input list to NumPy array. Use `np.fft.fft` to compute the Fast Fourier Transform. Return the complex coefficients.
- **Why np.fft.fft?**: Implements efficient FFT algorithm (Cooley-Tukey), optimal for signal processing.
- **Edge Cases**:
  - Single element: Returns same element.
  - Small arrays: FFT works efficiently.
  - Large values: Handled with complex128 precision.
- **Optimizations**: `np.fft.fft` uses optimized FFT implementation.

## Performance Analysis
- **Time Complexity**: O(n log n), where n is the signal length, for FFT computation.
- **Space Complexity**: O(n) for the output array.
- **NumPy Efficiency**: Uses optimized FFTW library for fast computation.

## Best Practices
- Use `np.fft.fft` for Fourier transforms.
- Ensure input is a 1D array.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Manual DFT**: Compute Discrete Fourier Transform (O(n^2), slow).
  ```python
  import numpy as np
  def fourier_transform(signal: list) -> np.ndarray:
      signal = np.array(signal)
      n = len(signal)
      result = np.zeros(n, dtype=np.complex128)
      for k in range(n):
          for t in range(n):
              result[k] += signal[t] * np.exp(-2j * np.pi * t * k / n)
      return result
  ```
- **SciPy FFT**: Use SciPy’s FFT (O(n log n), similar performance).
  ```python
  from scipy.fft import fft
  import numpy as np
  def fourier_transform(signal: list) -> np.ndarray:
      return fft(np.array(signal))
  ```