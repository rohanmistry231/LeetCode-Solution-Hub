# Big Countries

## Problem Statement
Write a Pandas query to find countries that have an area of at least 3 million or a population of at least 25 million in the `World` table.

**Table: World**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| name        | varchar |
| continent   | varchar |
| area        | int     |
| population  | int     |
| gdp         | bigint  |
+-------------+---------+
name is the primary key.
```

**Example**:
- Input: `[[Afghanistan, Asia, 652230, 25510000, 20343000000], [Albania, Europe, 28748, 2831741, 12960000000], [Algeria, Africa, 2381741, 37100000, 188681000000]]`
- Output: `[[Algeria, Africa, 2381741, 37100000, 188681000000]]`

**Constraints**:
- `1 <= World.name.length <= 100`
- `0 <= World.area, World.population <= 10^9`
- `0 <= World.gdp <= 10^15`

## Solution
```python
import pandas as pd

def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    return world[(world['area'] >= 3000000) | (world['population'] >= 25000000)][['name', 'population', 'area']]
```

## Reasoning
- **Approach**: Use boolean indexing to filter rows where `area >= 3000000` or `population >= 25000000`. Select only the required columns (`name`, `population`, `area`) for the output.
- **Why Boolean Indexing?**: Pandas vectorized operations are efficient for filtering large datasets without loops.
- **Edge Cases**:
  - No qualifying countries: Returns empty DataFrame.
  - Single country: Returns one row if it meets criteria.
  - Missing values: Handled by constraints (no nulls).
- **Optimizations**: Use `|` for logical OR; select only required columns to reduce memory usage.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of rows in `world`, for vectorized filtering and column selection.
- **Space Complexity**: O(k), where k is the number of rows in the output DataFrame.
- **Pandas Efficiency**: Vectorized operations ensure fast execution; no need for explicit loops.

## Best Practices
- Use clear column names in filtering conditions.
- Select only required columns to minimize memory usage.
- Use vectorized operations (`|`) instead of loops.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Query Method**: Use `world.query('area >= 3000000 or population >= 25000000')[['name', 'population', 'area']]` (similar performance, less readable).
- **Loc Indexing**: Use `world.loc[(world['area'] >= 3000000) | (world['population'] >= 25000000), ['name', 'population', 'area']]` (equivalent performance, slightly more verbose).