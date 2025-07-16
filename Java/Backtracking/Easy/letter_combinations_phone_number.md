# Letter Combinations of a Phone Number

## Problem Statement
Given a string `digits` containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order. A mapping of digits to letters (like on a phone keypad) is given below.

- 2: "abc"
- 3: "def"
- 4: "ghi"
- 5: "jkl"
- 6: "mno"
- 7: "pqrs"
- 8: "tuv"
- 9: "wxyz"

**Example**:
- Input: `digits = "23"`
- Output: `["ad","ae","af","bd","be","bf","cd","ce","cf"]`

**Constraints**:
- `0 <= digits.length <= 4`
- `digits[i]` is a digit in the range `['2', '9']`.

## Solution

### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        if (digits == null || digits.isEmpty()) return result;
        String[] digitMap = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        
        backtrack(0, digits, "", result, digitMap);
        return result;
    }
    
    private void backtrack(int index, String digits, String current, List<String> result, String[] digitMap) {
        if (index == digits.length()) {
            result.add(current);
            return;
        }
        String letters = digitMap[digits.charAt(index) - '0'];
        for (char c : letters.toCharArray()) {
            backtrack(index + 1, digits, current + c, result, digitMap);
        }
    }
}
```

## Reasoning
- **Approach**: Use backtracking to explore all possible combinations. For each digit, iterate through its corresponding letters, build the current combination, and recurse to the next digit. When the combination length equals the input length, add it to the result.
- **Why Backtracking?**: It systematically explores all possible combinations, ensuring all valid letter sequences are generated.
- **Edge Cases**:
  - Empty string: Return empty list.
  - Single digit: Generate all letters for that digit.
- **Optimizations**: Use a static digit-to-letter mapping; avoid string concatenation in loops by passing current string.

## Complexity Analysis
- **Time Complexity**: O(3^N * 4^M), where N is the number of digits with 3 letters (2,3,4,5,6,8) and M is the number of digits with 4 letters (7,9). Each digit branches into 3 or 4 choices.
- **Space Complexity**: O(N) for the recursion stack, where N is the length of `digits`, plus O(3^N * 4^M) for the output.

## Best Practices
- Use clear variable names (e.g., `digit_map`, `current`).
- For Python, use type hints and dictionary for mapping.
- For JavaScript, use object for mapping and modern loops.
- For Java, use array for mapping and follow Google Java Style Guide.
- Handle empty input early to avoid unnecessary computation.

## Alternative Approaches
- **Iterative**: Use a queue to build combinations (O(3^N * 4^M) time, O(3^N * 4^M) space). Less intuitive.
- **Cartesian Product**: Generate combinations iteratively (same complexity). More complex to implement.