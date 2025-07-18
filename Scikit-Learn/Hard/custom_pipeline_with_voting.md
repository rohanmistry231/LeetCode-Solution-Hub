# Custom Pipeline with Voting

## Problem Statement
Write a Scikit-Learn program to build a pipeline with preprocessing and a voting classifier combining Logistic Regression, Random Forest, and SVM for binary classification.

**Input**:
- `X`: 2D array of features (e.g., `[[1, 2], [2, 1], [3, 3], [4, 4]]`)
- `y`: 1D array of binary labels (e.g., `[0, 0, 1, 1]`)

**Output**:
- Trained voting classifier pipeline

**Constraints**:
- `1 <= len(X), len(y) <= 10^4`
- `X[i]` has 2 features
- `0 <= y[i] <= 1`

## Solution
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

def custom_pipeline_with_voting(X: list, y: list) -> Pipeline:
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Define classifiers
    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    
    # Define voting classifier
    voting_clf = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft')
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('voting', voting_clf)
    ])
    
    # Train pipeline
    pipeline.fit(X, y)
    
    return pipeline
```

## Reasoning
- **Approach**: Convert inputs to NumPy arrays. Create a pipeline with `StandardScaler` for preprocessing and a `VotingClassifier` combining Logistic Regression, Random Forest, and SVM with soft voting. Train and return the pipeline.
- **Why Voting Classifier?**: Combines strengths of diverse models, improving robustness and accuracy for binary classification.
- **Edge Cases**:
  - Small dataset: Risk of overfitting, mitigated by ensemble.
  - Unbalanced classes: Soft voting (probability-based) handles imbalance.
  - Single sample: Pipeline trains but prediction is trivial.
- **Optimizations**: Use `probability=True` for SVM soft voting; `random_state` for reproducibility.

## Performance Analysis
- **Time Complexity**: O(n * (t_rf + t_lr + t_svc)), where n is the number of samples, t_rf is O(n_trees * n * log n), t_lr is O(n), and t_svc is O(n^2).
- **Space Complexity**: O(n * f + m), where f is the number of features (2) and m is model parameters.
- **Scikit-Learn Efficiency**: `VotingClassifier` optimizes ensemble predictions; `Pipeline` ensures consistent preprocessing.

## Best Practices
- Use `Pipeline` to prevent data leakage.
- Set `voting='soft'` for probability-based ensemble.
- Ensure `X` is 2D and `y` is 1D.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Hard Voting**: Use majority voting (O(n * (t_rf + t_lr + t_svc)), simpler).
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import VotingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  import numpy as np
  def custom_pipeline_with_voting(X: list, y: list) -> Pipeline:
      X = np.array(X)
      y = np.array(y)
      clf1 = LogisticRegression(random_state=42)
      clf2 = RandomForestClassifier(random_state=42)
      clf3 = SVC(random_state=42)
      voting_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='hard')
      pipeline = Pipeline([('scaler', StandardScaler()), ('voting', voting_clf)])
      pipeline.fit(X, y)
      return pipeline
  ```
- **Single Classifier**: Use single classifier in pipeline (O(n), less robust).
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression
  import numpy as np
  def custom_pipeline_with_voting(X: list, y: list) -> Pipeline:
      X = np.array(X)
      y = np.array(y)
      pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42))])
      pipeline.fit(X, y)
      return pipeline
  ```