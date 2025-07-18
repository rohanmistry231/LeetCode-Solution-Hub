# Consecutive Numbers

## Problem Statement
Write a SQL query to find all numbers that appear at least three times consecutively in the `Logs` table.

**Table: Logs**
```
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| num         | int  |
+-------------+------+
id is the primary key, sorted in ascending order.
```

**Example**:
- Input: `[[1, 1], [2, 1], [3, 1], [4, 2], [5, 1], [6, 2], [7, 2]]`
- Output: `[1]`
- Explanation: 1 appears consecutively at ids 1, 2, 3.

**Constraints**:
- `1 <= Logs.id <= 10^5`
- `1 <= Logs.num <= 100`

## Solution
```sql
SELECT DISTINCT l1.num AS ConsecutiveNums
FROM Logs l1
JOIN Logs l2 ON l1.id = l2.id - 1
JOIN Logs l3 ON l2.id = l3.id - 1
WHERE l1.num = l2.num AND l2.num = l3.num;
```

## Reasoning
- **Approach**: Use self-joins to compare three consecutive rows by joining on `id = id - 1`. Check if `num` values are equal across the three rows. Use `DISTINCT` to avoid duplicate numbers in the output.
- **Why Self-Join?**: Allows checking consecutive rows by leveraging the ordered `id` column.
- **Edge Cases**:
  - Less than three rows: Returns empty result.
  - Multiple consecutive sequences: Returns each distinct number.
  - No consecutive numbers: Returns empty result.
- **Optimizations**: Use `DISTINCT` to avoid duplicates; join on `id` differences ensures consecutive check.

## Performance Analysis
- **Time Complexity**: O(n) for hash joins, where n is the number of rows in `Logs`.
- **Space Complexity**: O(1), excluding output storage.
- **Index Usage**: Primary key index on `id` optimizes joins.

## Best Practices
- Use clear aliases (e.g., `l1`, `l2`, `l3`) for self-joins.
- Specify join conditions explicitly with `ON`.
- Use `DISTINCT` to ensure unique output.
- Format SQL consistently.

## Alternative Approaches
- **Window Functions**: Use `LAG` to compare previous values (O(n), more complex).
  ```sql
  SELECT DISTINCT num AS ConsecutiveNums
  FROM (
      SELECT num, 
             LAG(num, 1) OVER (ORDER BY id) AS prev1,
             LAG(num, 2) OVER (ORDER BY id) AS prev2
      FROM Logs
  ) t
  WHERE num = prev1 AND num = prev2;
  ```
- **Subquery**: Check consecutive ids (O(n^2), less efficient).
  ```sql
  SELECT DISTINCT num AS ConsecutiveNums
  FROM Logs l1
  WHERE l1.num = (SELECT num FROM Logs l2 WHERE l2.id = l1.id + 1)
    AND l1.num = (SELECT num FROM Logs l3 WHERE l3.id = l1.id + 2);
  ```