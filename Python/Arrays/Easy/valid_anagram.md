# Valid Anagram

## Problem Statement
Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise. An anagram is a word formed by rearranging the letters of another.

**Example**:
- Input: `s = "anagram", t = "nagaram"`
- Output: `true`

**Constraints**:
- `1 <= s.length, t.length <= 5 * 10^4`
- `s` and `t` consist of lowercase English letters.

## Solution

### Python
```python
def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    char_count = [0] * 26
    for c1, c2 in zip(s, t):
        char_count[ord(c1) - ord('a')] += 1
        char_count[ord(c2) - ord('a')] -= 1
    return all(count == 0 for count in char_count)
```

## Reasoning
- **Approach**: Use a frequency array to count occurrences of each character in both strings. Since the strings contain only lowercase letters, a 26-element array suffices. Increment counts for `s` and decrement for `t`. If all counts are zero, the strings are anagrams.
- **Why Frequency Array?**: It’s efficient for fixed-size alphabets (26 letters) and avoids sorting or hash maps, reducing space and time complexity.
- **Edge Cases**:
  - Different lengths: Return `false` immediately.
  - Empty strings: Return `true` (both empty).
  - Same characters, different counts: Frequency array detects mismatches.
- **Optimizations**: Single pass through both strings; early length check avoids unnecessary computation.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of the strings. Single pass for counting, constant-time array operations.
- **Space Complexity**: O(1), as the frequency array size is fixed at 26.

## Best Practices
- Use clear variable names (e.g., `char_count`).
- For Python, use type hints and `zip` for concise iteration.
- For JavaScript, use `charCodeAt` and `every` for readability.
- For Java, use `charAt` and follow Google Java Style Guide.
- Check length first to avoid unnecessary processing.

## Alternative Approaches
- **Sorting**: Sort both strings and compare (O(n log n) time, O(1) or O(n) space depending on implementation).
- **Hash Map**: Count character frequencies with a hash map (O(n) time, O(n) space). Less efficient for small alphabets.