# Combine Two Tables

## Problem Statement
Write a SQL query to report all people’s first name, last name, city, and state from the `Person` and `Address` tables. If a person has no address, report null for city and state.

**Table: Person**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| personId    | int     |
| lastName    | varchar |
| firstName   | varchar |
+-------------+---------+
personId is the primary key.
```

**Table: Address**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| addressId   | int     |
| personId    | int     |
| city        | varchar |
| state       | varchar |
+-------------+---------+
addressId is the primary key.
```

**Example**:
- Input: 
  - Person: `[[1, "Wang", "Allen"], [2, "Alice", "Bob"]]`
  - Address: `[[1, 2, "New York City", "New York"]]`
- Output: `[[2, "Alice", "Bob", "New York City", "New York"], [1, "Wang", "Allen", null, null]]`

**Constraints**:
- `1 <= Person.personId <= 1000`
- `1 <= Address.addressId <= 1000`
- `personId` in `Address` exists in `Person`.

## Solution
```sql
SELECT p.firstName, p.lastName, a.city, a.state
FROM Person p
LEFT JOIN Address a ON p.personId = a.personId;
```

## Reasoning
- **Approach**: Use a `LEFT JOIN` to combine the `Person` and `Address` tables, ensuring all persons are included even if they lack an address. Match rows on `personId`. Select the required columns: `firstName`, `lastName`, `city`, and `state`.
- **Why LEFT JOIN?**: Guarantees all records from `Person` are returned, with `NULL` for `city` and `state` when no matching address exists.
- **Edge Cases**:
  - No addresses: All `city` and `state` values are `NULL`.
  - Single person: Returns one row with or without address.
  - No matching addresses: Returns `NULL` for address fields.
- **Optimizations**: Use `LEFT JOIN` for inclusivity; simple join on primary key ensures efficiency.

## Performance Analysis
- **Time Complexity**: O(n + m), where n is the number of rows in `Person` and m is the number of rows in `Address`, assuming a hash join (typical for primary key joins).
- **Space Complexity**: O(1) for the query execution, excluding output storage.
- **Index Usage**: Primary keys (`personId`, `addressId`) ensure efficient joins. No additional indexes needed.

## Best Practices
- Use clear table aliases (e.g., `p`, `a`) for readability.
- Explicitly specify join conditions with `ON`.
- Use `LEFT JOIN` to handle missing data as required.
- Format SQL with consistent casing and indentation.

## Alternative Approaches
- **INNER JOIN**: Excludes persons without addresses (incorrect for this problem).
- **RIGHT JOIN**: Not suitable as `Person` is the primary table.
- **Subquery**: Less efficient and unnecessary (e.g., `SELECT city FROM Address WHERE personId = p.personId`).