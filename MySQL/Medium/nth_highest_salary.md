# Nth Highest Salary

## Problem Statement
Write a SQL query to get the nth highest salary from the `Employee` table. If there is no nth highest salary, return `null`.

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
- Input: 
  - Employee: `[[1, 100], [2, 200], [3, 300], [4, 200]]`
  - N = 2
- Output: `200`

**Constraints**:
- `1 <= Employee.id <= 10^5`
- `0 <= Employee.salary <= 10^5`
- `1 <= N <= 10^9`

## Solution
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
    SELECT DISTINCT salary
    FROM Employee
    ORDER BY salary DESC
    LIMIT 1 OFFSET N-1
  );
END
```

## Reasoning
- **Approach**: Create a function to return the nth highest salary. Use `DISTINCT` to handle duplicate salaries, sort by `salary` in descending order, and use `LIMIT 1 OFFSET N-1` to get the nth highest salary. If no such salary exists, the query returns `null`.
- **Why DISTINCT and LIMIT?**: `DISTINCT` ensures unique salaries are considered, and `LIMIT` with `OFFSET` efficiently skips to the nth record.
- **Edge Cases**:
  - N > number of unique salaries: Returns `null`.
  - Duplicate salaries: `DISTINCT` ensures correct nth rank.
  - Single salary: Returns `null` if N > 1.
- **Optimizations**: Use `DISTINCT` to avoid duplicate issues; `LIMIT` ensures minimal row fetching.

## Performance Analysis
- **Time Complexity**: O(n log n) due to sorting by `salary`.
- **Space Complexity**: O(1), excluding output storage.
- **Index Usage**: An index on `salary` can optimize sorting and `DISTINCT` operations.

## Best Practices
- Use clear function name (`getNthHighestSalary`).
- Apply `DISTINCT` to handle duplicates correctly.
- Use `LIMIT` and `OFFSET` for efficient row selection.
- Format SQL consistently with proper indentation.

## Alternative Approaches
- **Subquery**: Use a subquery to count higher salaries (O(n^2), less efficient).
  ```sql
  SELECT salary
  FROM Employee e1
  WHERE N-1 = (SELECT COUNT(DISTINCT salary) FROM Employee e2 WHERE e2.salary > e1.salary);
  ```
- **Dense Rank**: Use `DENSE_RANK()` (O(n log n), requires window function support).
  ```sql
  SELECT salary
  FROM (SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
        FROM Employee) t
  WHERE rnk = N;
  ```