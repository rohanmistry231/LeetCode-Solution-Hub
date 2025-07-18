# Median Employee Salary

## Problem Statement
Write a SQL query to find the median salary for each company in the `Employee` table.

**Table: Employee**
```
+--------------+---------+
| Column Name  | Type    |
+--------------+---------+
| id           | int     |
| company      | varchar |
| salary       | int     |
+--------------+---------+
id is the primary key.
```

**Example**:
- Input: `[[1, "A", 2341], [2, "A", 341], [3, "A", 15], [4, "A", 15314], [5, "B", 15], [6, "B", 10400]]`
- Output: `[[1, "A", 2341], [4, "A", 15314], [5, "B", 15], [6, "B", 10400]]`
- Explanation: For company A, median salaries are 2341 and 15314 (sorted: 15, 341, 2341, 15314). For company B, median is 15 and 10400 (sorted: 15, 10400).

**Constraints**:
- `1 <= Employee.id <= 10^5`
- `1 <= Employee.salary <= 10^5`
- `company` is a string of lowercase letters.

## Solution
```sql
SELECT id, company, salary
FROM (
    SELECT id, company, salary,
           ROW_NUMBER() OVER (PARTITION BY company ORDER BY salary) AS row_num,
           COUNT(*) OVER (PARTITION BY company) AS total_count
    FROM Employee
) t
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2))
ORDER BY company, salary;
```

## Reasoning
- **Approach**: Use window functions to assign row numbers to salaries within each company (`ROW_NUMBER`) and count total rows per company (`COUNT`). The median is the salary where the row number is approximately half the total count (for odd count, middle row; for even, two middle rows). Use `FLOOR` and `CEIL` to handle both cases.
- **Why Window Functions?**: `ROW_NUMBER` and `COUNT` allow efficient ranking and counting within groups, avoiding complex joins or subqueries.
- **Edge Cases**:
  - Single salary per company: Return that salary.
  - Even number of salaries: Return two middle salaries.
  - Odd number of salaries: Return middle salary.
- **Optimizations**: Window functions reduce need for multiple table scans; ordering ensures consistent output.

## Performance Analysis
- **Time Complexity**: O(n log n) due to sorting for `ROW_NUMBER` and `COUNT` within partitions.
- **Space Complexity**: O(n) for storing window function results.
- **Index Usage**: Indexes on `company` and `salary` optimize partitioning and sorting.

## Best Practices
- Use clear column aliases (e.g., `row_num`, `total_count`).
- Include `ORDER BY` in output for clarity.
- Use `FLOOR` and `CEIL` to handle median for even/odd counts.
- Format SQL consistently.

## Alternative Approaches
- **Self-Join**: Count salaries above and below to find median (O(n^2), inefficient).
  ```sql
  SELECT e1.id, e1.company, e1.salary
  FROM Employee e1
  WHERE (
      SELECT COUNT(*) FROM Employee e2 WHERE e2.company = e1.company AND e2.salary > e1.salary
  ) = (
      SELECT COUNT(*) FROM Employee e2 WHERE e2.company = e1.company AND e2.salary < e1.salary
  )
  OR (
      SELECT COUNT(*) FROM Employee e2 WHERE e2.company = e1.company AND e2.salary > e1.salary
  ) + 1 = (
      SELECT COUNT(*) FROM Employee e2 WHERE e2.company = e1.company AND e2.salary < e1.salary
  )
  ORDER BY e1.company, e1.salary;
  ```
- **Temporary Table**: Store sorted salaries and select middle rows (O(n log n), more complex).