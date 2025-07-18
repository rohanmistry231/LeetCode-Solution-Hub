# Friend Requests II: Who Has the Most Friends

## Problem Statement
Write a SQL query to find the person with the most friend requests (sent or received) in the `RequestAccepted` table.

**Table: RequestAccepted**
```
+----------------+---------+
| Column Name    | Type    |
+----------------+---------+
| requester_id   | int     |
| accepter_id    | int     |
| accept_date    | date    |
+----------------+---------+
(requester_id, accepter_id) is the primary key.
```

**Example**:
- Input: `[[1, 2, "2016-06-03"], [1, 3, "2016-06-08"], [2, 3, "2016-06-08"], [3, 4, "2016-06-09"]]`
- Output: `[[3, 3]]`
- Explanation: Person 3 has 3 friend requests (sent to 4, received from 1 and 2).

**Constraints**:
- `1 <= RequestAccepted.requester_id, accepter_id <= 1000`

## Solution
```sql
SELECT id, COUNT(*) AS num
FROM (
    SELECT requester_id AS id FROM RequestAccepted
    UNION ALL
    SELECT accepter_id AS id FROM RequestAccepted
) t
GROUP BY id
ORDER BY num DESC
LIMIT 1;
```

## Reasoning
- **Approach**: Use `UNION ALL` to combine `requester_id` and `accepter_id` into a single column of user IDs. Group by `id` and count occurrences to get the total friend requests per person. Order by count descending and limit to 1 to get the person with the most requests.
- **Why UNION ALL?**: Combines sent and received requests efficiently without deduplication (faster than `UNION`).
- **Edge Cases**:
  - Single request: Returns that person.
  - Tie for max requests: Returns one (arbitrary, as per problem).
  - No requests: Handled by constraints (non-empty).
- **Optimizations**: Use `UNION ALL` for performance; `LIMIT 1` ensures single output.

## Performance Analysis
- **Time Complexity**: O(n log n) for grouping and sorting, where n is the number of rows in `RequestAccepted`.
- **Space Complexity**: O(n) for storing combined IDs and group results.
- **Index Usage**: Indexes on `requester_id` and `accepter_id` optimize `UNION ALL` and grouping.

## Best Practices
- Use `UNION ALL` instead of `UNION` for performance when duplicates are acceptable.
- Use clear column alias (`num`) for count.
- Include `ORDER BY` and `LIMIT` for precise output.
- Format SQL consistently.

## Alternative Approaches
- **Subquery with JOIN**: Count separately and join (O(n log n), more complex).
  ```sql
  SELECT t.id, (COALESCE(r.cnt, 0) + COALESCE(a.cnt, 0)) AS num
  FROM (
      SELECT requester_id AS id, COUNT(*) AS cnt
      FROM RequestAccepted
      GROUP BY requester_id
  ) r
  FULL JOIN (
      SELECT accepter_id AS id, COUNT(*) AS cnt
      FROM RequestAccepted
      GROUP BY accepter_id
  ) a ON r.id = a.id
  ORDER BY num DESC
  LIMIT 1;
  ```
- **Subquery with COALESCE**: Similar but less readable.