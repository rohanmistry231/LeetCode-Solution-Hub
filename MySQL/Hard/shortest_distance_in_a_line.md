# Shortest Distance in a Line

## Problem Statement
Write a SQL query to find the shortest distance between any two points in the `Point` table.

**Table: Point**
```
+-------------+------+
| Column Name | Type |
+-------------+------+
| x           | int  |
+-------------+------+
x is the primary key.
```

**Example**:
- Input: `[[-1], [0], [2]]`
- Output: `1`
- Explanation: The shortest distance is between -1 and 0 (or 0 and 2), which is 1.

**Constraints**:
- `2 <= Point.x <= 10^4`
- `Point.x` contains unique values.

## Solution
```sql
SELECT MIN(ABS(p1.x - p2.x)) AS shortest
FROM Point p1
JOIN Point p2 ON p1.x != p2.x;
```

## Reasoning
- **Approach**: Use a self-join to pair all distinct points (`p1.x != p2.x`). Calculate the absolute difference between their `x` values using `ABS(p1.x - p2.x)`. Use `MIN` to find the smallest distance.
- **Why Self-Join?**: Allows comparison of all pairs of points to compute distances.
- **Edge Cases**:
  - Two points: Returns their distance.
  - Multiple points: Finds minimum distance.
  - No duplicate x values: Ensured by constraints.
- **Optimizations**: Use `MIN` to avoid sorting entire result; `!=` ensures no self-pairing.

## Performance Analysis
- **Time Complexity**: O(n^2) for self-join, where n is the number of rows in `Point`. `MIN` is computed during join.
- **Space Complexity**: O(1), excluding output storage.
- **Index Usage**: Primary key index on `x` optimizes join and comparison.

## Best Practices
- Use clear aliases (e.g., `p1`, `p2`) for self-joins.
- Use `ABS` for distance calculation.
- Avoid unnecessary columns in output.
- Format SQL consistently.

## Alternative Approaches
- **Window Function**: Use `LEAD` to compare adjacent points (O(n log n), assumes sorted).
  ```sql
  SELECT MIN(ABS(x - LEAD(x) OVER (ORDER BY x))) AS shortest
  FROM Point;
  ```
- **Subquery**: Compute differences with subquery (O(n^2), less readable).
  ```sql
  SELECT MIN(ABS(p1.x - p2.x)) AS shortest
  FROM Point p1, Point p2
  WHERE p1.x < p2.x;
  ```