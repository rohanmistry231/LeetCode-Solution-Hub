# Second Highest Salary

## Problem Statement
Write a SQL query to report the second highest salary from the `Employee` table. If there is no second highest salary, return `null`.

**Table: Employee**
```
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| salary      | int  |
+-------------+------+
id is the primary key.
```

**Example**:
- Input: `[[1, 100], [2, 200], [3, 300]]`
- Output: `200`
- Input: `[[1, 100]]`
- Output: `null`

**Constraints**:
- `1 <= Employee.id <= 10^5`
- `0 <= Employee.salary <= 10^5`

## Solution
```sql
SELECT MAX(salary) AS SecondHighestSalary
FROM Employee
WHERE salary < (SELECT MAX(salary) FROM Employee);
```

## Reasoning
- **Approach**: Find the maximum salary excluding the highest salary using a subquery. The outer query selects the maximum salary less than the highest salary. If no such salary exists, it returns `null`.
- **Why Subquery?**: Allows filtering out the highest salary to find the next highest in a single query.
- **Edge Cases**:
  - Single salary: Returns `null` (no second highest).
  - Duplicate salaries: Returns the second distinct highest salary.
  - Empty table: Handled by constraints (at least one row).
- **Optimizations**: Simple subquery avoids sorting; leverages `MAX` for efficiency.

## Performance Analysis
- **Time Complexity**: O(n) for each `MAX` operation, assuming a table scan (no index on `salary`).
- **Space Complexity**: O(1), excluding output storage.
- **Index Usage**: An index on `salary` could optimize `MAX` operations, but not strictly necessary for small datasets.

## Best Practices
- Use clear column alias (`SecondHighestSalary`).
- Avoid unnecessary sorting (e.g., `ORDER BY` with `LIMIT`).
- Write concise subqueries for readability.
- Format SQL consistently.

## Alternative Approaches
- **LIMIT/OFFSET**: `SELECT DISTINCT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET 1` (O(n log n) due to sorting).
- **Self-Join**: Join table with itself to find second highest (O(n^2), less efficient).
- **DENSE_RANK**: Use window function (O(n log n), overkill for simple case).