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

### Java
```java
class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        
        int[] count = new int[26];
        for (int i = 0; i < s.length(); i++) {
            count[s.charAt(i) - 'a']++;
            count[t.charAt(i) - 'a']--;
        }
        
        for (int x : count) {
            if (x != 0) return false;
        }
        return true;
    }
}
```

## Reasoning
- **Approach**: Use a frequency array to count character occurrences in `s` and `t`. Increment for `s` characters, decrement for `t` characters. If all counts are zero, the strings are anagrams. Check lengths first to avoid unnecessary processing.
- **Why Frequency Array?**: Efficiently tracks character counts in O(n) time with constant space for lowercase letters.
- **Edge Cases**:
  - Different lengths: Return false.
  - Empty strings: Return true (if both empty).
  - Same characters, different counts: Return false.
- **Optimizations**: Single pass over strings; use fixed-size array for 26 letters.

## Complexity Analysis
- **Time Complexity**: O(n), where n is the length of `s` or `t`, as we process each character once.
- **Space Complexity**: O(1), as the frequency array is fixed at 26 elements.

## Best Practices
- Use clear variable names (e.g., `count`).
- For Python, use type hints and `zip` for iteration.
- For JavaScript, use `charCodeAt` for character indexing.
- For Java, follow Google Java Style Guide.
- Check length early to optimize.

## Alternative Approaches
- **Sorting**: Sort both strings and compare (O(n log n) time). Less efficient.
- **Hash Map**: Use a map to count characters (O(n) time, O(n) space). Less space-efficient for fixed alphabet.