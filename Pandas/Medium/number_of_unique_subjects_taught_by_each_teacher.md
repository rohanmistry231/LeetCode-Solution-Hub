# Number of Unique Subjects Taught by Each Teacher

## Problem Statement
Write a Pandas query to find the number of unique subjects taught by each teacher in the `Teacher` table.

**Table: Teacher**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| teacher_id  | int     |
| subject_id  | int     |
| dept_id     | int     |
+-------------+---------+
(subject_id, dept_id) is the primary key.
```

**Example**:
- Input: `[[1, 2, 3], [1, 2, 4], [1, 3, 3], [2, 1, 1], [2, 2, 1], [2, 3, 1], [2, 4, 1]]`
- Output: `[[1, 2], [2, 4]]`

**Constraints**:
- `1 <= Teacher.teacher_id, Teacher.subject_id, Teacher.dept_id <= 10^5`

## Solution
```python
import pandas as pd

def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
    return teacher.groupby('teacher_id')['subject_id'].nunique().reset_index(name='cnt')
```

## Reasoning
- **Approach**: Group the `Teacher` table by `teacher_id` and use `nunique` to count unique `subject_id` values for each teacher. Reset the index to make `teacher_id` a column and rename the count column to `cnt`.
- **Why GroupBy and nunique?**: `groupby` aggregates by teacher, and `nunique` efficiently counts distinct subjects without manual deduplication.
- **Edge Cases**:
  - Single teacher: Returns one row with their unique subject count.
  - No subjects: Handled by constraints (non-empty).
  - Duplicate subjects: `nunique` ignores duplicates.
- **Optimizations**: Use `nunique` for efficiency; `reset_index` ensures proper output format.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of rows in `teacher`, for grouping and counting unique values.
- **Space Complexity**: O(k), where k is the number of unique teachers in the output.
- **Pandas Efficiency**: `groupby` and `nunique` are optimized for aggregation.

## Best Practices
- Use `nunique` for counting distinct values.
- Rename output columns clearly (`cnt`).
- Use `reset_index` to convert groupby result to DataFrame.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Drop Duplicates**: Remove duplicates before grouping (O(n), more verbose).
  ```python
  import pandas as pd
  def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
      return teacher[['teacher_id', 'subject_id']].drop_duplicates().groupby('teacher_id').size().reset_index(name='cnt')
  ```
- **Set and Count**: Use `apply` with set (O(n), less efficient).
  ```python
  import pandas as pd
  def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
      return teacher.groupby('teacher_id')['subject_id'].apply(lambda x: len(set(x))).reset_index(name='cnt')
  ```