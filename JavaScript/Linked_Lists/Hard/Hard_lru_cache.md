# LRU Cache

## Problem Statement
Design a Least Recently Used (LRU) cache that supports `get` and `put` operations. `get(key)` returns the value if the key exists, otherwise -1. `put(key, value)` inserts or updates the key-value pair. If capacity is exceeded, remove the least recently used item.

**Example**:
```
LRUCache cache = new LRUCache(2);
cache.put(1, 1); // cache is {1=1}
cache.put(2, 2); // cache is {1=1, 2=2}
cache.get(1);    // returns 1
cache.put(3, 3); // evicts key 2, cache is {1=1, 3=3}
cache.get(2);    // returns -1 (not found)
```

**Constraints**:
- `1 <= capacity <= 3000`
- `0 <= key <= 10^4`
- `0 <= value <= 10^5`
- At most `2 * 10^5` calls will be made to `get` and `put`.

## Solution

### JavaScript
```javascript
class Node {
    constructor(key = 0, value = 0, prev = null, next = null) {
        this.key = key;
        this.value = value;
        this.prev = prev;
        this.next = next;
    }
}

class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
        this.head = new Node();
        this.tail = new Node();
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }
    
    _remove(node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    _add(node) {
        node.prev = this.head;
        node.next = this.head.next;
        this.head.next.prev = node;
        this.head.next = node;
    }
    
    get(key) {
        if (this.cache.has(key)) {
            const node = this.cache.get(key);
            this._remove(node);
            this._add(node);
            return node.value;
        }
        return -1;
    }
    
    put(key, value) {
        if (this.cache.has(key)) {
            this._remove(this.cache.get(key));
        }
        const node = new Node(key, value);
        this._add(node);
        this.cache.set(key, node);
        if (this.cache.size > this.capacity) {
            const lru = this.tail.prev;
            this._remove(lru);
            this.cache.delete(lru.key);
        }
    }
}
```

## Reasoning
- **Approach**: Use a doubly linked list and hash map for O(1) get and put. The list maintains the order of use (head: most recent, tail: least recent). The map stores key-node pairs for fast lookup. Move accessed/added nodes to the head and remove from the tail if capacity is exceeded.
- **Why Doubly Linked List?**: Allows O(1) removal and addition of nodes, critical for LRU operations.
- **Edge Cases**:
  - Empty cache: Handle get/put correctly.
  - Capacity 1: Evict immediately on new put.
  - Key not found: Return -1 for get.
- **Optimizations**: Use hash map for O(1) lookup; doubly linked list for O(1) node manipulation.

## Complexity Analysis
- **Time Complexity**: O(1) for both `get` and `put`, as map lookups and list operations are constant time.
- **Space Complexity**: O(capacity) for the hash map and doubly linked list.

## Best Practices
- Use clear variable names (e.g., `head`, `tail`).
- For Python, use type hints and helper methods.
- For JavaScript, use `Map` for key-node storage.
- For Java, follow Google Java Style Guide and use `HashMap`.
- Encapsulate list operations in helper methods.

## Alternative Approaches
- **Array-Based**: Use an array for order (O(n) time for operations). Inefficient.
- **Single Linked List**: Slower removal (O(n) time). Not practical.