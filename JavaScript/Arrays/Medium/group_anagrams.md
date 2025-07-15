# Group Anagrams

## Problem Statement
Given an array of strings `strs`, group the anagrams together. You can return the answer in any order. An anagram is a word formed by rearranging the letters of another.

**Example**:
- Input: `strs = ["eat","tea","tan","ate","nat","bat"]`
- Output: `[["bat"],["nat","tan"],["ate","eat","tea"]]`

**Constraints**:
- `1 <= strs.length <= 10^4`
- `0 <= strs[i].length <= 100`
- `strs[i]` consists of lowercase English letters.

## Solution

### JavaScript
```javascript
function groupAnagrams(strs) {
    const anagramMap = new Map();
    for (const s of strs) {
        const key = s.split('').sort().join('');
        if (!anagramMap.has(key)) {
            anagramMap.set(key, []);
        }
        anagramMap.get(key).push(s);
    }
    return Array.from(anagramMap.values());
}
```

## Reasoning
- **Approach**: Use a hash map to group strings by their sorted characters (key). Each key maps to a list of anagrams. Sorting characters ensures anagrams have the same key.
- **Why Sorted Key?**: Sorting characters is a reliable way to identify anagrams, as all anagrams share the same sorted string.
- **Edge Cases**:
  - Empty array: Return empty list.
  - Single string: Return a list with one group.
  - Empty strings: Treated as anagrams of each other.
- **Optimizations**: Use `defaultdict` in Python for concise code; `computeIfAbsent` in Java for efficient map updates.

## Complexity Analysis
- **Time Complexity**: O(n * k * log k), where n is the number of strings and k is the maximum string length (due to sorting each string).
- **Space Complexity**: O(n * k) for the hash map and output.

## Best Practices
- Use clear variable names (e.g., `anagram_map`, `key`).
- For Python, use `defaultdict` to simplify map operations.
- For JavaScript, use `Map` for key-value pairs.
- For Java, use `computeIfAbsent` and follow Google Java Style Guide.
- Sort characters to create a consistent anagram key.

## Alternative Approaches
- **Character Count Key**: Use a frequency array or string (e.g., `"a1b2"`) as the key (O(n * k) time, O(n * k) space). Faster for small alphabets.
- **Brute Force**: Compare each string with others (O(n² * k) time). Inefficient.