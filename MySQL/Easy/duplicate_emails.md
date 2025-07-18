# Duplicate Emails

## Problem Statement
Write a SQL query to report all duplicate email addresses in the `Person` table.

**Table: Person**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| id          | int     |
| email       | varchar |
+-------------+---------+
id is the primary key.
```

**Example**:
- Input: `[[1, "a@b.com"], [2, "c@d.com"], [3, "a@b.com"]]`
- Output: `["a@b.com"]`

**Constraints**:
- `1 <= Person.id <= 1000`
- `email` contains lowercase letters only.

## Solution
```sql
SELECT email
FROM Person
GROUP BY email
HAVING COUNT(*) > 1;
```

## Reasoning
- **Approach**: Use `GROUP BY` to group rows by `email` and `HAVING` to filter groups with more than one occurrence. Select the `email` column for duplicates.
- **Why GROUP BY?**: Aggregates rows by email, allowing `COUNT` to identify duplicates efficiently.
- **Edge Cases**:
  - No duplicates: Returns empty result.
  - Single email: Returns no rows.
  - All duplicates: Returns the repeated email.
- **Optimizations**: Simple `GROUP BY` and `HAVING` avoid complex joins or subqueries.

## Performance Analysis
- **Time Complexity**: O(n log n) for grouping and counting, depending on database’s hash/sort implementation.
- **Space Complexity**: O(k), where k is the number of unique emails, for storing groups.
- **Index Usage**: An index on `email` improves grouping performance.

## Best Practices
- Use `GROUP BY` and `HAVING` for aggregation-based filtering.
- Keep query minimal and readable.
- Avoid unnecessary columns in output.
- Format SQL consistently.

## Alternative Approaches
- **Self-Join**: Join table with itself on matching emails (O(n^2), less efficient).
- **Subquery**: Use subquery to count emails (O(n), more complex).
- **Window Function**: Use `COUNT` over partition (O(n log n), overkill).