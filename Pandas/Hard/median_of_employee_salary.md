# Median of Employee Salary

## Problem Statement
Write a Pandas query to find the median salary for each department in the `Employee` table.

**Table: Employee**
```
+--------------+---------+
| Column Name  | Type    |
+--------------+---------+
| id           | int     |
| department   | varchar |
| salary       | int     |
+--------------+---------+
id is the primary key.
```

**Example**:
- Input: `[[1, "A", 2341], [2, "A", 341], [3, "A", 15], [4, "A", 15314], [5, "B", 15], [6, "B", 10400]]`
- Output: `[[1, "A", 2341], [4, "A", 15314], [5, "B", 15], [6, "B", 10400]]`

**Constraints**:
- `1 <= Employee.id <= 10^5`
- `1 <= Employee.salary <= 10^5`
- `department` is a string of lowercase letters.

## Solution
```python
import pandas as pd

def median_salary(employee: pd.DataFrame) -> pd.DataFrame:
    # Add row number and count per department
    employee['row_num'] = employee.groupby('department')['salary'].rank(method='first')
    employee['total_count'] = employee.groupby('department')['salary'].transform('count')
    # Filter rows where row_num is near median
    median_rows = employee[employee['row_num'].isin([
        (employee['total_count'] + 1) // 2,
        (employee['total_count'] + 2) // 2
    ])]
    return median_rows[['id', 'department', 'salary']].sort_values(['department', 'salary'])
```

## Reasoning
- **Approach**: Use `rank` to assign row numbers to salaries within each department, sorted by salary. Compute total count per department with `transform`. Select rows where `row_num` is near the median (floor and ceiling of `(count + 1)/2` for odd/even counts). Return `id`, `department`, and `salary`, sorted for consistency.
- **Why Rank and Transform?**: `rank` provides ordered positions; `transform` computes group sizes efficiently, enabling median selection.
- **Edge Cases**:
  - Single salary: Returns that salary.
  - Even number of salaries: Returns two middle salaries.
  - Odd number: Returns middle salary.
- **Optimizations**: Use `rank` and `transform` for efficiency; filter with `isin` for median rows.

## Performance Analysis
- **Time Complexity**: O(n log n), where n is the number of rows in `employee`, due to sorting for `rank`.
- **Space Complexity**: O(n) for temporary columns and output DataFrame.
- **Pandas Efficiency**: `rank` and `transform` are optimized for group operations.

## Best Practices
- Use `rank` for ordered positions within groups.
- Use `transform` for group-level counts.
- Select only required columns for output.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Quantile**: Use `quantile` for median (O(n log n), less control).
  ```python
  import pandas as pd
  def median_salary(employee: pd.DataFrame) -> pd.DataFrame:
      medians = employee.groupby('department')['salary'].median().reset_index()
      return employee.merge(medians, on=['department', 'salary'])[['id', 'department', 'salary']].sort_values(['department', 'salary'])
  ```
- **Apply**: Compute medians with `apply` (O(n log n), more verbose).
  ```python
  import pandas as pd
  def median_salary(employee: pd.DataFrame) -> pd.DataFrame:
      def get_median(group):
          n = len(group)
          sorted_group = group.sort_values('salary')
          if n % 2 == 0:
              return sorted_group.iloc[n//2-1:n//2+1]
          return sorted_group.iloc[[n//2]]
      return employee.groupby('department').apply(get_median)[['id', 'department', 'salary']].reset_index(drop=True).sort_values(['department', 'salary'])
  ```