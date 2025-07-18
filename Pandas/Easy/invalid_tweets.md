# Invalid Tweets

## Problem Statement
Write a Pandas query to find the IDs of tweets whose content length exceeds 140 characters in the `Tweets` table.

**Table: Tweets**
```
+----------------+---------+
| Column Name    | Type    |
+----------------+---------+
| tweet_id       | int     |
| content        | varchar |
+----------------+---------+
tweet_id is the primary key.
```

**Example**:
- Input: `[[1, "Vote for Biden"], [2, "Let us make America great again!"]]`
- Output: `[[2]]`

**Constraints**:
- `1 <= Tweets.tweet_id <= 10^5`
- `0 <= Tweets.content.length <= 10^4`

## Solution
```python
import pandas as pd

def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    return tweets[tweets['content'].str.len() > 140][['tweet_id']]
```

## Reasoning
- **Approach**: Use `str.len()` to compute the length of each tweet’s `content`. Filter rows where length > 140 using boolean indexing. Return only the `tweet_id` column.
- **Why str.len()?**: Pandas’ string method is vectorized, efficiently computing lengths for all rows.
- **Edge Cases**:
  - No invalid tweets: Returns empty DataFrame.
  - Empty content: Length is 0, so not selected.
  - Single tweet: Returns one row if invalid.
- **Optimizations**: Vectorized `str.len()` avoids loops; select only `tweet_id` for minimal output.

## Performance Analysis
- **Time Complexity**: O(n), where n is the number of rows in `tweets`, for vectorized string length calculation and filtering.
- **Space Complexity**: O(k), where k is the number of rows in the output DataFrame.
- **Pandas Efficiency**: `str.len()` is optimized for string operations; boolean indexing is fast.

## Best Practices
- Use vectorized string methods (`str.len()`).
- Select only required columns (`tweet_id`).
- Avoid loops for filtering.
- Follow PEP 8 for Python code style.

## Alternative Approaches
- **Apply**: Use `tweets[tweets['content'].apply(len) > 140][['tweet_id']]` (O(n), less efficient due to non-vectorized `apply`).
- **Query Method**: Use `tweets.query('content.str.len() > 140')[['tweet_id']]` (similar performance, less explicit).