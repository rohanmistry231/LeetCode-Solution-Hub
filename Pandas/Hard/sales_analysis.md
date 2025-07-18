# Sales Analysis

## Problem Statement
Write a Pandas query to find the top product by sales for each year in the `Sales` and `Product` tables.

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
  - Sales: `[[1, 100, 2008, 10, 5000], [2, 100, 2009, 12, 5000], [3, 200, 2009, 15, 7000]]`
  - Product: `[[100, "Nokia"], [200, "Apple"]]`
- Output: `[[2008, "Nokia"], [2009, "Apple"]]`

**Constraints**:
- `1 <= Sales.sale_id, Sales.product_id, Product.product_id <= 10^5`
- `2000 <= Sales.year <= 2025`
- `0 <= Sales.quantity, Sales.price <= 10^5`

## Solution
```python
import pandas as pd

def sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    # Calculate total sales (quantity * price) per product and year
    sales['total_sales'] = sales['quantity'] * sales['price']
    # Group by year and product_id, sum total sales
    sales_agg = sales.groupby(['year', 'product_id'])['total_sales'].sum().reset_index()
    # Find the product with max sales per year
    max_sales = sales_agg.loc[sales_agg.groupby('year')['total_sales'].idxmax()]
    # Merge with Product to get product_name
    result = max_sales.merge(product, on='product_id')[['year', 'product_name']]
    return result.sort_values('year')
```

## Reasoning
- **Approach**: Compute total sales (`quantity * price`) for each row. Group by `year` and `product_id`, summing total sales. Identify the product with the maximum sales per year using `idxmax`. Merge with `Product` to get `product_name` and return `year` and `product_name`, sorted by year.
- **Why GroupBy and idxmax?**: `groupby` aggregates sales by year and product; `idxmax` efficiently selects the top product per year.
- **Edge Cases**:
  - Single sale per year: Returns that product.
  - Tie in sales: `idxmax` selects one (arbitrary, as per problem).
  - No sales: Handled by constraints (non-empty).
- **Optimizations**: Use vectorized multiplication; `idxmax` avoids sorting entire group.

## Performance Analysis
- **Time Complexity**: O(n + m), where n is the number of rows in `Sales` (for grouping and computation) and m is the number of rows in `Product` (for merge). Sorting by year is O(k log k), where k is the number of unique years (small).
- **Space Complexity**: O(k), where k is the number of unique (year, product_id) pairs in the output.
- **Pandas Efficiency**: Vectorized operations and `idxmax` are optimized; merge is efficient with hash join.

## Best Practices
- Use vectorized operations (`quantity * price`).
- Select only required columns (`year`, `product_name`).
- Sort output for consistency (`sort_values`).
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Rank with GroupBy**: Use `rank` to identify top products (O(n log n), more complex).
  ```python
  import pandas as pd
  def sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
      sales['total_sales'] = sales['quantity'] * sales['price']
      sales_agg = sales.groupby(['year', 'product_id'])['total_sales'].sum().reset_index()
      sales_agg['rank'] = sales_agg.groupby('year')['total_sales'].rank(method='first', ascending=False)
      result = sales_agg[sales_agg['rank'] == 1].merge(product, on='product_id')[['year', 'product_name']]
      return result.sort_values('year')
  ```
- **Apply**: Use `apply` to find max per group (O(n), less efficient).
  ```python
  import pandas as pd
  def sales_analysis(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
      sales['total_sales'] = sales['quantity'] * sales['price']
      sales_agg = sales.groupby(['year', 'product_id'])['total_sales'].sum().reset_index()
      max_sales = sales_agg.groupby('year').apply(lambda x: x.nlargest(1, 'total_sales')).reset_index(drop=True)
      return max_sales.merge(product, on='product_id')[['year', 'product_name']].sort_values('year')
  ```