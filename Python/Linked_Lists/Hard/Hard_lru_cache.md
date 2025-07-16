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

### Python
```python
class Node:
    def __init__(self, key=0, value=0, prev=None, next=None):
        self.key = key
        self.value = value
        self.prev = prev
        self.next = next

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add(self, node: Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

### Java
```java
class Node {
    int key, value;
    Node prev, next;
    Node(int key, int value) { this.key = key; this.value = value; }
}

class LRUCache {
    private int capacity;
    private Map<Integer, Node> cache;
    private Node head, tail;
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.cache = new HashMap<>();
        this.head = new Node(0, 0);
        this.tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
    }
    
    private void remove(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }
    
    private void add(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }
    
    public int get(int key) {
        if (cache.containsKey(key)) {
            Node node = cache.get(key);
            remove(node);
            add(node);
            return node.value;
        }
        return -1;
    }
    
    public void put(int key, int value) {
        if (cache.containsKey(key)) {
            remove(cache.get(key));
        }
        Node node = new Node(key, value);
        add(node);
        cache.put(key, node);
        if (cache.size() > capacity) {
            Node lru = tail.prev;
            remove(lru);
            cache.remove(lru.key);
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