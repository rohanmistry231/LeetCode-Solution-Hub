# Group Sold Products By The Date

## Problem Statement
Write a Pandas query to group sold products by sale date in the `Activities` table, returning the date, number of unique products, and a comma-separated list of products.

**Table: Activities**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| sell_date   | date    |
| product     | varchar |
+-------------+---------+
No primary key; multiple rows for the same product may exist.
```

**Example**:
- Input: `[[2020-05-30, Basketball], [2020-05-30, Bible], [2020-06-01, T-Shirt], [2020-06-02, Basketball]]`
- Output: `[[2020-05-30, 2, Basketball,Bible], [2020-06-01, 1, T-Shirt], [2020-06-02, 1, Basketball]]`

**Constraints**:
- `1 <= Activities.product.length <= 1000`
- `Activities.sell_date` is a valid date.

## Solution
```python
import pandas as pd

def group_sold_products_by_date(activities: pd.DataFrame) -> pd.DataFrame:
    return activities.groupby('sell_date')['product'].agg([
        ('num_sold', 'nunique'),
        ('products', lambda x: ','.join(sorted(x.drop_duplicates())))
    ]).reset_index()
```

## Reasoning
- **Approach**: Group `Activities` by `sell_date`. Use `agg` to compute two metrics: `nunique` for the count of unique products (`num_sold`) and a lambda function to join sorted unique products with commas (`products`). Reset the index to make `sell_date` a column.
- **Why GroupBy and agg?**: `groupby` organizes by date, and `agg` allows multiple aggregations (count and string join) in one step.
- **Edge Cases**:
  - Single sale date: Returns one row with its product stats.
  - Duplicate products: `drop_duplicates` ensures unique products in the join.
  - No sales: Handled by constraints (non-empty).
- **Optimizations**: Use `nunique` for counting; `drop_duplicates` and `sorted` ensure correct product list.

## Performance Analysis
- **Time Complexity**: O(n log n), where n is the number of rows in `activities`, due to sorting products in the lambda function.
- **Space Complexity**: O(k), where k is the number of unique sale dates in the output.
- **Pandas Efficiency**: `groupby` and `nunique` are optimized; string joining is minimal overhead.

## Best Practices
- Use `agg` for multiple aggregations.
- Ensure unique products with `drop_duplicates`.
- Sort products for consistent output.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Separate GroupBy**: Compute `nunique` and join separately (O(n log n), more verbose).
  ```python
  import pandas as pd
  def group_sold_products_by_date(activities: pd.DataFrame) -> pd.DataFrame:
      counts = activities.groupby('sell_date')['product'].nunique().reset_index(name='num_sold')
      products = activities[['sell_date', 'product']].drop_duplicates().groupby('sell_date')['product'].apply(lambda x: ','.join(sorted(x))).reset_index(name='products')
      return counts.merge(products, on='sell_date')
  ```
- **Apply**: Use `apply` for both operations (O(n log n), less efficient).
  ```python
  import pandas as pd
  def group_sold_products_by_date(activities: pd.DataFrame) -> pd.DataFrame:
      return activities.groupby('sell_date')['product'].apply(lambda x: pd.Series({
          'num_sold': x.nunique(),
          'products': ','.join(sorted(x.drop_duplicates()))
      })).reset_index()
  ```