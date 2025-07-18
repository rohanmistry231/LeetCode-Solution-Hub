# Human Traffic of Stadium

## Problem Statement
Write a SQL query to find three consecutive days in the `Stadium` table where the number of people is at least 100 for each day.

**Table: Stadium**
```
+---------------+---------+
| Column Name   | Type    |
+---------------+---------+
| id            | int     |
| visit_date    | date    |
| people        | int     |
+---------------+---------+
id is the primary key, sorted in ascending order.
```

**Example**:
- Input: `[[1, "2017-01-01", 10], [2, "2017-01-02", 109], [3, "2017-01-03", 150], [4, "2017-01-04", 99], [5, "2017-01-05", 145], [6, "2017-01-06", 145], [7, "2017-01-07", 199], [8, "2017-01-09", 188]]`
- Output: `[[5, "2017-01-05", 145], [6, "2017-01-06", 145], [7, "2017-01-07", 199]]`
- Explanation: Days 5, 6, 7 have >= 100 people consecutively.

**Constraints**:
- `1 <= Stadium.id <= 10^5`
- `0 <= Stadium.people <= 10^9`

## Solution
```sql
SELECT DISTINCT s1.id, s1.visit_date, s1.people
FROM Stadium s1
JOIN Stadium s2 ON s1.id = s2.id - 1
JOIN Stadium s3 ON s2.id = s3.id - 1
WHERE s1.people >= 100 AND s2.people >= 100 AND s3.people >= 100
UNION
SELECT DISTINCT s2.id, s2.visit_date, s2.people
FROM Stadium s1
JOIN Stadium s2 ON s1.id = s2.id - 1
JOIN Stadium s3 ON s2.id = s3.id - 1
WHERE s1.people >= 100 AND s2.people >= 100 AND s3.people >= 100
UNION
SELECT DISTINCT s3.id, s3.visit_date, s3.people
FROM Stadium s1
JOIN Stadium s2 ON s1.id = s2.id - 1
JOIN Stadium s3 ON s2.id = s3.id - 1
WHERE s1.people >= 100 AND s2.people >= 100 AND s3.people >= 100
ORDER BY visit_date;
```

## Reasoning
- **Approach**: Use self-joins to identify three consecutive days by joining on `id = id - 1`. Check if `people >= 100` for all three days. Use `UNION` to include all rows from the triplet (first, middle, last). `DISTINCT` avoids duplicates if a day appears in multiple triplets. Order by `visit_date`.
- **Why Self-Join and UNION?**: Self-join checks consecutive days via `id` differences; `UNION` ensures all days in a valid triplet are included.
- **Edge Cases**:
  - Fewer than three days: Returns empty result.
  - Multiple triplets: `UNION` and `DISTINCT` handle overlaps.
  - Gaps in `id`: Handled by join condition.
- **Optimizations**: Use `DISTINCT` to avoid duplicates; join on primary key ensures efficiency.

## Performance Analysis
- **Time Complexity**: O(n) for hash joins, where n is the number of rows in `Stadium`.
- **Space Complexity**: O(k), where k is the number of result rows, for `UNION` and output.
- **Index Usage**: Primary key index on `id` optimizes joins; index on `people` may help filtering.

## Best Practices
- Use clear aliases (e.g., `s1`, `s2`, `s3`) for self-joins.
- Use `UNION` to combine results cleanly.
- Include `ORDER BY` for consistent output.
- Format SQL consistently.

## Alternative Approaches
- **Window Functions**: Use `LAG` and `LEAD` to check adjacent rows (O(n), more complex).
  ```sql
  SELECT id, visit_date, people
  FROM (
      SELECT id, visit_date, people,
             people >= 100 AND 
             LAG(people, 1) OVER (ORDER BY id) >= 100 AND 
             LAG(people, 2) OVER (ORDER BY id) >= 100 AS valid_start,
             people >= 100 AND 
             LAG(people, 1) OVER (ORDER BY id) >= 100 AND 
             LEAD(people, 1) OVER (ORDER BY id) >= 100 AS valid_mid,
             people >= 100 AND 
             LEAD(people, 1) OVER (ORDER BY id) >= 100 AND 
             LEAD(people, 2) OVER (ORDER BY id) >= 100 AS valid_end
      FROM Stadium
  ) t
  WHERE valid_start OR valid_mid OR valid_end
  ORDER BY visit_date;
  ```
- **Subquery**: Check consecutive ids (O(n^2), less efficient).