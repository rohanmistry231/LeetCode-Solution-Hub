# Employees Earning More Than Their Managers

## Problem Statement
Write a SQL query to find employees who earn more than their managers.

**Table: Employee**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| name        | varchar |
| salary      | int     |
| managerId   | int     |
+-------------+---------+
id is the primary key.
```

**Example**:
- Input: `[[1, "Joe", 70000, 3], [2, "Henry", 80000, 4], [3, "Sam", 60000, null], [4, "Max", 90000, null]]`
- Output: `["Joe"]`
- Explanation: Joe earns 70000, more than his manager Sam’s 60000.

**Constraints**:
- `1 <= Employee.id <= 10^5`
- `0 <= Employee.salary <= 10^5`
- `managerId` is either `null` or exists in `id`.

## Solution
```sql
SELECT e1.name AS Employee
FROM Employee e1
JOIN Employee e2 ON e1.managerId = e2.id
WHERE e1.salary > e2.salary;
```

## Reasoning
- **Approach**: Use a self-join to compare each employee (`e1`) with their manager (`e2`) by joining on `managerId = id`. Filter where the employee’s salary exceeds the manager’s salary. Select the employee’s name.
- **Why Self-Join?**: Allows comparison of rows within the same table by linking employees to their managers.
- **Edge Cases**:
  - No manager (`managerId` is `null`): Excluded by `JOIN`.
  - Same salary: Excluded by strict comparison (`>`).
  - Single employee: No output if no manager or condition unmet.
- **Optimizations**: Use `JOIN` instead of `LEFT JOIN` since `managerId` must exist; simple condition for filtering.

## Performance Analysis
- **Time Complexity**: O(n) for a hash join, where n is the number of rows in `Employee`.
- **Space Complexity**: O(1), excluding output storage.
- **Index Usage**: Indexes on `id` and `managerId` optimize join performance.

## Best Practices
- Use clear aliases (e.g., `e1`, `e2`) for self-joins.
- Specify join conditions explicitly with `ON`.
- Use column alias (`Employee`) for clarity.
- Format SQL consistently.

## Alternative Approaches
- **Subquery**: Use subquery to fetch manager’s salary (O(n), less readable).
- **LEFT JOIN**: Includes employees without managers (incorrect for this problem).
- **Window Function**: Overkill and unnecessary for simple comparison.