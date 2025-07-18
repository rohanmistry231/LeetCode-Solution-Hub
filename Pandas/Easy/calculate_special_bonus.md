# Calculate Special Bonus

## Problem Statement
Write a Pandas query to calculate a special bonus for employees. The bonus is 100% of salary if the employee ID is odd and the name does not start with 'M'; otherwise, the bonus is 0.

**Table: Employees**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| employee_id | int     |
| name        | varchar |
| salary      | int     |
+-------------+---------+
employee_id is the primary key.
```

**Example**:
- Input: `[[1, "Meir", 3000], [2, "Michael", 3800], [3, "Susan", 2400]]`
- Output: `[[1, 3000], [2, 0], [3, 2400]]`

**Constraints**:
- `1 <= Employees.employee_id <= 10^5`
- `0 <= Employees.salary <= 10^5`
- `1 <= Employees.name.length <= 1000`

## Solution
```python
import pandas as pd

def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    employees['bonus'] = employees.apply(
        lambda x: x['salary'] if x['employee_id'] % 2 == 1 and not x['name'].startswith('M') else 0, 
        axis=1
    )
    return employees[['employee_id', 'bonus']]
```

## Reasoning
- **Approach**: Use `apply` to compute the bonus for each row. Check if `employee_id` is odd (`% 2 == 1`) and `name` does not start with 'M' (`startswith('M')`). If both conditions are met, bonus is `salary`; otherwise, 0. Return `employee_id` and `bonus` columns.
- **Why Apply?**: Allows row-wise logic combining multiple conditions, as vectorized operations are less straightforward for this conditional check.
- **Edge Cases**:
  - No qualifying employees: Returns 0 for all bonuses.
  - Zero salary: Bonus is 0 even if conditions are met.
  - Single employee: Returns one row with appropriate bonus.
- **Optimizations**: Select only required columns; use `apply` for clarity despite slight performance cost.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of rows in `employees`, due to row-wise `apply`.
- **Space Complexity**: O(n) for the new `bonus` column and output DataFrame.
- **Pandas Efficiency**: `apply` is less efficient than vectorized operations but suitable for complex row-wise logic.

## Best Practices
- Use clear column names (`bonus`).
- Select only required columns (`employee_id`, `bonus`).
- Use `lambda` in `apply` for concise logic.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Vectorized**: Use `numpy.where` for better performance.
  ```python
  import pandas as pd
  import numpy as np
  def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
      employees['bonus'] = np.where(
          (employees['employee_id'] % 2 == 1) & (~employees['name'].str.startswith('M')),
          employees['salary'], 0
      )
      return employees[['employee_id', 'bonus']]
  ```
- **Query Method**: Less applicable due to complex conditions.