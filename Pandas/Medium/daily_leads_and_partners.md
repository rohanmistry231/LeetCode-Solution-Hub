# Daily Leads and Partners

## Problem Statement
Write a Pandas query to find the number of unique leads and partners for each date and make in the `DailySales` table.

**Table: DailySales**
```
+-------------+---------+
| Column Name | Type    |
+-------------+---------+
| date_id     | date    |
| make_name   | varchar |
| lead_id     | int     |
| partner_id  | int     |
+-------------+---------+
No primary key; multiple rows for the same lead_id or partner_id may exist.
```

**Example**:
- Input: `[[2020-12-8, Toyota, 0, 1], [2020-12-8, Toyota, 1, 0], [2020-12-7, Toyota, 0, 2], [2020-12-7, Honda, 0, 1]]`
- Output: `[[2020-12-7, Honda, 1, 1], [2020-12-7, Toyota, 1, 2], [2020-12-8, Toyota, 2, 2]]`

**Constraints**:
- `1 <= DailySales.lead_id, DailySales.partner_id <= 10^5`
- `DailySales.date_id` is a valid date.
- `1 <= DailySales.make_name.length <= 1000`

## Solution
```python
import pandas as pd

def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    return daily_sales.groupby(['date_id', 'make_name'])[['lead_id', 'partner_id']].nunique().reset_index().rename(columns={'lead_id': 'unique_leads', 'partner_id': 'unique_partners'})
```

## Reasoning
- **Approach**: Group `DailySales` by `date_id` and `make_name`. Use `nunique` to count unique `lead_id` and `partner_id` values for each group. Reset the index to make `date_id` and `make_name` columns, and rename the count columns to `unique_leads` and `unique_partners`.
- **Why GroupBy and nunique?**: `groupby` organizes by date and make, and `nunique` efficiently counts distinct IDs without manual deduplication.
- **Edge Cases**:
  - Single date/make: Returns one row with counts.
  - Duplicate IDs: `nunique` handles duplicates correctly.
  - No sales: Handled by constraints (non-empty).
- **Optimizations**: Use `nunique` for efficient counting; rename columns for clarity.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of rows in `daily_sales`, for grouping and counting unique values.
- **Space Complexity**: O(k), where k is the number of unique (date_id, make_name) pairs in the output.
- **Pandas Efficiency**: `groupby` and `nunique` are optimized for aggregation.

## Best Practices
- Use `nunique` for counting distinct values.
- Rename columns clearly (`unique_leads`, `unique_partners`).
- Use multi-column `groupby` for hierarchical aggregation.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Drop Duplicates**: Remove duplicates before grouping (O(n), more verbose).
  ```python
  import pandas as pd
  def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
      leads = daily_sales[['date_id', 'make_name', 'lead_id']].drop_duplicates().groupby(['date_id', 'make_name']).size().reset_index(name='unique_leads')
      partners = daily_sales[['date_id', 'make_name', 'partner_id']].drop_duplicates().groupby(['date_id', 'make_name']).size().reset_index(name='unique_partners')
      return leads.merge(partners, on=['date_id', 'make_name'])
  ```
- **Apply**: Use `apply` with set (O(n), less efficient).
  ```python
  import pandas as pd
  def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
      return daily_sales.groupby(['date_id', 'make_name']).agg({
          'lead_id': lambda x: len(set(x)),
          'partner_id': lambda x: len(set(x))
      }).reset_index().rename(columns={'lead_id': 'unique_leads', 'partner_id': 'unique_partners'})
  ```