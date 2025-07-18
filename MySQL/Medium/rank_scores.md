# Rank Scores

## Problem Statement
Write a SQL query to rank scores in the `Scores` table. The ranking should be consecutive (no gaps) and assign the same rank to equal scores.

**Table: Scores**
```
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| score       | decimal |
+-------------+------+
id is the primary key.
```

**Example**:
- Input: `[[1, 3.50], [2, 3.65], [3, 4.00], [4, 3.85], [5, 4.00], [6, 3.65]]`
- Output: `[[4.00, 1], [3.85, 2], [3.65, 3], [3.50, 4]]`

**Constraints**:
- `1 <= Scores.id <= 10^5`
- `0 <= Scores.score <= 10^2`

## Solution
```sql
SELECT score, DENSE_RANK() OVER (ORDER BY score DESC) AS rank
FROM Scores
ORDER BY score DESC;
```

## Reasoning
- **Approach**: Use the `DENSE_RANK()` window function to assign ranks to scores, ordered by `score` in descending order. `DENSE_RANK()` ensures consecutive ranks for equal scores without gaps.
- **Why DENSE_RANK?**: Provides consecutive ranking (e.g., 1, 2, 2, 3) as required, unlike `RANK()` (which creates gaps).
- **Edge Cases**:
  - Single score: Rank is 1.
  - All equal scores: All get rank 1.
  - Empty table: Handled by constraints (non-empty).
- **Optimizations**: Window function is efficient for ranking; sorting ensures correct output order.

## Performance Analysis
- **Time Complexity**: O(n log n) due to sorting for `DENSE_RANK()` and `ORDER BY`.
- **Space Complexity**: O(n) for storing the window function results.
- **Index Usage**: An index on `score` optimizes sorting and ranking.

## Best Practices
- Use `DENSE_RANK()` for consecutive rankings.
- Include `ORDER BY` in output for clarity.
- Keep query concise and readable.
- Format SQL consistently.

## Alternative Approaches
- **Self-Join**: Count higher distinct scores (O(n^2), inefficient).
  ```sql
  SELECT s1.score, (SELECT COUNT(DISTINCT s2.score) FROM Scores s2 WHERE s2.score > s1.score) + 1 AS rank
  FROM Scores s1
  ORDER BY score DESC;
  ```
- **Subquery with GROUP BY**: Group scores and assign ranks (O(n log n), more complex).
  ```sql
  SELECT score, (SELECT COUNT(DISTINCT score) FROM Scores s2 WHERE s2.score > s1.score) + 1 AS rank
  FROM Scores s1
  ORDER BY score DESC;
  ```