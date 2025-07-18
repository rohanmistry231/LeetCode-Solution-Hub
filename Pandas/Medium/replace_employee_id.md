# Replace Employee ID

## Problem Statement
Write a Pandas query to replace employee IDs with their corresponding unique identifiers from the `Employees` and `EmployeeUNI` tables.

**Table: Employees**
```
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| name          | varchar |
+---------------+---------+
id is the primary key.
```

**Table: EmployeeUNI**
```
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| unique_id     | int     |
+---------------+---------+
(id, unique_id) is the primary key.
```

**Example**:
- Input:
  - Employees: `[[1, "Alice"], [7, "Bob"], [11, "Meir"], [90, "Winston"], [3, "Jonathan"]]`
  - EmployeeUNI: `[[3, 1], [11, 2], [1, 3]]`
- Output: `[[3, 1, "Alice"], [2, 11, "Meir"], [null, 7, "Bob"], [null, 90, "Winston"]]`

**Constraints**:
- `1 <= Employees.id, EmployeeUNI.id <= 10^5`
- `1 <= Employees.name.length <= 100`
- `1 <= EmployeeUNI.unique_id <= 10^5`

## Solution
```python
import pandas as pd

def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    return employees.merge(employee_uni[['id', 'unique_id']], on='id', how='left')[['unique_id', 'id', 'name']]
```

## Reasoning
- **Approach**: Use `merge` with a left join to combine `Employees` and `EmployeeUNI` on `id`. Select `unique_id`, `id`, and `name` columns. The left join ensures all employees are included, with `null` for `unique_id` if no match exists in `EmployeeUNI`.
- **Why Left Join?**: Preserves all rows from `Employees`, aligning with the requirement to include all employees even without a unique ID.
- **Edge Cases**:
  - No matching IDs in `EmployeeUNI`: Returns `null` for `unique_id`.
  - Single employee: Returns one row with or without `unique_id`.
  - Empty `EmployeeUNI`: All `unique_id` values are `null`.
- **Optimizations**: Use `merge` for efficient joining; select only required columns to reduce memory usage.

## Performance Analysis
- **Time Complexity**: O(n + m), where n is the number of rows in `Employees` and m is the number of rows in `EmployeeUNI`, for hash-based merge.
- **Space Complexity**: O(k), where k is the number of rows in the output DataFrame.
- **Pandas Efficiency**: `merge` is optimized for joins; `how='left'` ensures all employees are included.

## Best Practices
- Use `merge` with explicit `on` and `how` parameters.
- Select only required columns (`unique_id`, `id`, `name`).
- Avoid loops or manual matching for performance.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Join Method**: Use `employees.join(employee_uni.set_index('id'), on='id', how='left')[['unique_id', 'id', 'name']]` (similar performance, less intuitive).
- **Map with Dictionary**: Create a dictionary from `EmployeeUNI` and map `id` to `unique_id` (O(n + m), more verbose).
  ```python
  import pandas as pd
  def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
      id_map = employee_uni.set_index('id')['unique_id'].to_dict()
      employees['unique_id'] = employees['id'].map(id_map)
      return employees[['unique_id', 'id', 'name']]
  ```