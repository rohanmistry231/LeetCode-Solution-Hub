# Market Analysis I

## Problem Statement
Write a Pandas query to report the number of orders in 2019 for each buyer and their join date from the `Users` and `Orders` tables.

**Table: Users**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| user_id     | int     |
| join_date   | date    |
| favorite_brand | varchar |
+-------------+---------+
user_id is the primary key.
```

**Table: Orders**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| order_id    | int     |
| order_date  | date    |
| item_id     | int     |
| buyer_id    | int     |
| seller_id   | int     |
+-------------+---------+
order_id is the primary key.
```

**Example**:
- Input:
  - Users: `[[1, 2018-01-01, "Lenovo"], [2, 2019-02-09, "Samsung"]]`
  - Orders: `[[1, 2019-08-01, 4, 1, 2], [2, 2018-08-02, 2, 1, 3], [3, 2019-08-03, 3, 2, 3]]`
- Output: `[[1, 2018-01-01, 1], [2, 2019-02-09, 1]]`

**Constraints**:
- `1 <= Users.user_id, Orders.order_id, Orders.buyer_id <= 10^5`
- `2018-01-01 <= Users.join_date, Orders.order_date <= 2025-12-31`

## Solution
```python
import pandas as pd

def market_analysis(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # Filter orders for 2019
    orders_2019 = orders[orders['order_date'].dt.year == 2019]
    # Count orders per buyer
    order_counts = orders_2019.groupby('buyer_id').size().reset_index(name='orders_in_2019')
    # Left join with users to include all users
    result = users.merge(order_counts, left_on='user_id', right_on='buyer_id', how='left')[['user_id', 'join_date', 'orders_in_2019']]
    # Fill NaN with 0 for users with no orders
    result['orders_in_2019'] = result['orders_in_2019'].fillna(0).astype(int)
    return result
```

## Reasoning
- **Approach**: Filter `Orders` for 2019 using `dt.year`. Group by `buyer_id` and count orders. Left join with `Users` on `user_id` to include all users, even those without orders. Fill `NaN` in `orders_in_2019` with 0 and convert to integer. Return `user_id`, `join_date`, and `orders_in_2019`.
- **Why Left Join?**: Ensures all users are included, with 0 orders for those without 2019 orders.
- **Edge Cases**:
  - No 2019 orders: Returns 0 for all users.
  - No orders for a user: Returns 0 for their count.
  - Single user: Returns one row with their count.
- **Optimizations**: Filter early to reduce rows; use `size` for efficient counting.

## Performance Analysis
- **Time Complexity**: O(n + m), where n is the number of rows in `Orders` (for filtering and grouping) and m is the number of rows in `Users` (for merge).
- **Space Complexity**: O(k), where k is the number of users in the output.
- **Pandas Efficiency**: `dt.year` and `groupby` are optimized; `merge` uses hash join.

## Best Practices
- Filter data early (`dt.year`) to reduce computation.
- Use `left` join to include all users.
- Handle `NaN` with `fillna` for correct output.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Value Counts**: Use `value_counts` for counting (O(n), similar).
  ```python
  import pandas as pd
  def market_analysis(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
      orders_2019 = orders[orders['order_date'].dt.year == 2019]
      order_counts = orders_2019['buyer_id'].value_counts().reset_index(name='orders_in_2019')
      result = users.merge(order_counts, left_on='user_id', right_on='buyer_id', how='left')[['user_id', 'join_date', 'orders_in_2019']]
      return result.fillna({'orders_in_2019': 0}).astype({'orders_in_2019': int})
  ```
- **Apply**: Group and count with `apply` (O(n), less efficient).
  ```python
  import pandas as pd
  def market_analysis(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
      orders_2019 = orders[orders['order_date'].dt.year == 2019]
      order_counts = orders_2019.groupby('buyer_id').apply(len).reset_index(name='orders_in_2019')
      result = users.merge(order_counts, left_on='user_id', right_on='buyer_id', how='left')[['user_id', 'join_date', 'orders_in_2019']]
      return result.fillna({'orders_in_2019': 0}).astype({'orders_in_2019': int})
  ```