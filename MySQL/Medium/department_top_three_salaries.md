# Department Top Three Salaries

## Problem Statement
Write a SQL query to find the top three unique salaries for each department in the `Employee` and `Department` tables.

**Table: Employee**
```
+--------------+---------+
| Column Name  | Type    |
+--------------+---------+
| id           | int     |
| name         | varchar |
| salary       | int     |
| departmentId | int     |
+--------------+---------+
id is the primary key.
```

**Table: Department**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| name        | varchar |
+-------------+---------+
id is the primary key.
```

**Example**:
- Input:
  - Employee: `[[1, "Joe", 85000, 1], [2, "Henry", 80000, 2], [3, "Sam", 60000, 2], [4, "Max", 90000, 1], [5, "Janet", 69000, 1], [6, "Randy", 85000, 1]]`
  - Department: `[[1, "IT"], [2, "Sales"]]`
- Output: `[[1, "IT", "Max", 90000], [1, "IT", "Joe", 85000], [1, "IT", "Randy", 85000], [2, "Sales", "Henry", 80000], [2, "Sales", "Sam", 60000]]`

**Constraints**:
- `1 <= Employee.id, Department.id <= 10^5`
- `0 <= Employee.salary <= 10^5`
- `1 <= Employee.departmentId <= 10^5`

## Solution
```sql
SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary
FROM Employee e
JOIN Department d ON e.departmentId = d.id
WHERE (
    SELECT COUNT(DISTINCT e2.salary)
    FROM Employee e2
    WHERE e2.departmentId = e.departmentId AND e2.salary > e.salary
) < 3
ORDER BY d.name, e.salary DESC;
```

## Reasoning
- **Approach**: Join `Employee` and `Department` tables to get department names. Use a correlated subquery to count how many distinct salaries in the same department are higher than the current employee’s salary. If fewer than 3, the salary is in the top three. Order by department and salary for clarity.
- **Why Correlated Subquery?**: Allows ranking salaries within each department without window functions, ensuring top three unique salaries.
- **Edge Cases**:
  - Fewer than three salaries in a department: Return all salaries.
  - Duplicate salaries: Handled by `DISTINCT` in subquery.
  - No employees in a department: Handled by `JOIN` (no output).
- **Optimizations**: Subquery counts distinct salaries for accuracy; join on primary keys ensures efficiency.

## Performance Analysis
- **Time Complexity**: O(n * m) in worst case, where n is the number of employees and m is the number of employees in the same department, due to correlated subquery.
- **Space Complexity**: O(1), excluding output storage.
- **Index Usage**: Indexes on `Employee.departmentId`, `Employee.salary`, and `Department.id` optimize joins and subquery performance.

## Best Practices
- Use clear aliases (e.g., `e`, `d`) for readability.
- Specify join conditions with `ON`.
- Use `DISTINCT` in subquery to handle duplicate salaries.
- Format SQL consistently with proper indentation.

## Alternative Approaches
- **Window Function**: Use `DENSE_RANK()` for ranking (O(n log n), cleaner).
  ```sql
  SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary
  FROM (
      SELECT name, salary, departmentId, 
             DENSE_RANK() OVER (PARTITION BY departmentId ORDER BY salary DESC) AS rnk
      FROM Employee
  ) e
  JOIN Department d ON e.departmentId = d.id
  WHERE e.rnk <= 3
  ORDER BY d.name, e.salary DESC;
  ```
- **Self-Join**: Compare salaries within departments (O(n^2), less efficient).
  ```sql
  SELECT d.name AS Department, e1.name AS Employee, e1.salary AS Salary
  FROM Employee e1
  JOIN Department d ON e1.departmentId = d.id
  WHERE (
      SELECT COUNT(DISTINCT e2.salary)
      FROM Employee e2
      WHERE e2.departmentId = e1.departmentId AND e2.salary > e1.salary
  ) < 3
  ORDER BY d.name, e1.salary DESC;
  ```