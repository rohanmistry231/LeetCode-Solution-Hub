# Product Sales Analysis

## Problem Statement
Write a Pandas query to report the total sales amount for each product in 2019, including only products with sales, from the `Sales` and `Product` tables.

**Table: Sales**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| sale_id     | int     |
| product_id  | int     |
| year        | int     |
| quantity    | int     |
| price       | int     |
+-------------+---------+
(sale_id, year) is the primary key.
```

**Table: Product**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| product_id  | int     |
| product_name| varchar |
+-------------+---------+
product_id is the primary key.
```

**Example**:
- Input:
  - Sales: `[[1, 100, 2019, 10, 5000], [2, 100, 2020, 12, 5000], [3, 200, 2019, 15, 7000]]`
  - Product: `[[100, "Nokia"], [200, "Apple"], [300, "Samsung"]]`
- Output: `[[Nokia, 50000], [Apple, 105000]]`

**Constraints**:
- `1 <= Sales.sale_id, Sales.product_id, Product.product_id <= 10^5`
- `2000 <= Sales.year <= 2025`
- `0 <= Sales.quantity, Sales.price <= 10^5`

## Solution
```python
import pandas as pd

def product_sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    # Filter sales for 2019
    sales_2019 = sales[sales['year'] == 2019]
    # Calculate total sales amount
    sales_2019['total_sales'] = sales_2019['quantity'] * sales_2019['price']
    # Group by product_id and sum total sales
    sales_agg = sales_2019.groupby('product_id')['total_sales'].sum().reset_index()
    # Merge with Product to get product_name
    result = sales_agg.merge(product, on='product_id')[['product_name', 'total_sales']]
    return result.rename(columns={'total_sales': 'total'})
```

## Reasoning
- **Approach**: Filter `Sales` for 2019. Compute total sales (`quantity * price`) per row. Group by `product_id` and sum total sales. Merge with `Product` to get `product_name`. Rename `total_sales` to `total` and return `product_name` and `total`.
- **Why Filter and GroupBy?**: Filtering reduces rows to process; `groupby` aggregates sales efficiently.
- **Edge Cases**:
  - No 2019 sales: Returns empty DataFrame.
  - Single product: Returns one row with its total.
  - Products with no sales: Excluded due to merge.
- **Optimizations**: Filter early; use vectorized multiplication and efficient merge.

## Performance Analysis
- **Time Complexity**: O(n + m), where n is the number of rows in `Sales` (for filtering and grouping) and m is the number of rows in `Product` (for merge).
- **Space Complexity**: O(k), where k is the number of products with 2019 sales.
- **Pandas Efficiency**: Vectorized operations and `groupby` are optimized; merge uses hash join.

## Best Practices
- Filter data early (`year == 2019`) to reduce computation.
- Use vectorized operations (`quantity * price`).
- Rename columns for clarity (`total`).
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Join First**: Merge tables before filtering (O(n + m), less efficient).
  ```python
  import pandas as pd
  def product_sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
      merged = sales.merge(product, on='product_id')
      merged['total_sales'] = merged['quantity'] * merged['price']
      result = merged[merged['year'] == 2019].groupby('product_name')['total_sales'].sum().reset_index()
      return result.rename(columns={'total_sales': 'total'})
  ```
- **Apply**: Use `apply` for grouping (O(n), less efficient).
  ```python
  import pandas as pd
  def product_sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
      sales_2019 = sales[sales['year'] == 2019]
      sales_2019['total_sales'] = sales_2019['quantity'] * sales_2019['price']
      result = sales_2019.groupby('product_id').apply(lambda x: x['total_sales'].sum()).reset_index(name='total')
      return result.merge(product, on='product_id')[['product_name', 'total']]
  ```