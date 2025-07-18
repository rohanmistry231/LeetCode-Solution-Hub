# Recyclable and Low Fat Products

## Problem Statement
Write a Pandas query to find the IDs of products that are both recyclable and low fat in the `Products` table.

**Table: Products**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| product_id  | int     |
| low_fats    | varchar |
| recyclable  | varchar |
+-------------+---------+
product_id is the primary key.
```

**Example**:
- Input: `[[0, "Y", "N"], [1, "Y", "Y"], [2, "N", "Y"], [3, "Y", "Y"], [4, "N", "N"]]`
- Output: `[[1], [3]]`

**Constraints**:
- `1 <= Products.product_id <= 10^5`

## Solution
```python
import pandas as pd

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    return products[(products['low_fats'] == 'Y') & (products['recyclable'] == 'Y')][['product_id']]
```

## Reasoning
- **Approach**: Use boolean indexing to filter rows where `low_fats == 'Y'` and `recyclable == 'Y'`. Return only the `product_id` column.
- **Why Boolean Indexing?**: Pandas vectorized operations efficiently filter rows based on multiple conditions.
- **Edge Cases**:
  - No qualifying products: Returns empty DataFrame.
  - Single product: Returns one row if it meets criteria.
  - Missing values: Handled by constraints (no nulls).
- **Optimizations**: Use `&` for logical AND; select only `product_id` for minimal output.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of rows in `products`, for vectorized filtering.
- **Space Complexity**: O(k), where k is the number of rows in the output DataFrame.
- **Pandas Efficiency**: Vectorized operations (`==`, `&`) are highly optimized.

## Best Practices
- Use vectorized operations for filtering.
- Select only required columns (`product_id`).
- Use clear conditionals (`== 'Y'`).
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Query Method**: Use `products.query("low_fats == 'Y' and recyclable == 'Y'")[['product_id']]` (similar performance, less explicit).
- **Loc Indexing**: Use `products.loc[(products['low_fats'] == 'Y') & (products['recyclable'] == 'Y'), ['product_id']]` (equivalent performance, more verbose).