# Python Data Structures

## Table of Contents
- [Lists](#lists)
- [Tuples](#tuples)
- [Dictionaries](#dictionaries)
- [Sets](#sets)
- [Collections Module](#collections-module)
- [Comprehensions](#comprehensions)

---

## Lists

Lists are **ordered**, **mutable** sequences that can hold items of any type.

### Creating Lists

```python
empty   = []
numbers = [1, 2, 3, 4, 5]
mixed   = [1, "two", 3.0, True, None]
nested  = [[1, 2], [3, 4], [5, 6]]

# From other iterables
from_range  = list(range(1, 11))   # [1, 2, ..., 10]
from_string = list("hello")        # ['h', 'e', 'l', 'l', 'o']
```

### Accessing Elements

```python
nums = [10, 20, 30, 40, 50]

print(nums[0])    # 10  — first element
print(nums[-1])   # 50  — last element
print(nums[-2])   # 40  — second from end

# Slicing: list[start:stop:step]
print(nums[1:4])  # [20, 30, 40] — indices 1,2,3 (stop is exclusive)
print(nums[:3])   # [10, 20, 30]
print(nums[2:])   # [30, 40, 50]
print(nums[::2])  # [10, 30, 50] — every other
print(nums[::-1]) # [50, 40, 30, 20, 10] — reversed copy
```

### Modifying Lists

```python
fruits = ["apple", "banana", "cherry"]

# Change element
fruits[1] = "blueberry"

# Slice assignment
fruits[0:2] = ["avocado", "blackberry"]

# append — add to end
fruits.append("date")

# extend — add multiple items
fruits.extend(["elderberry", "fig"])

# insert — at specific index
fruits.insert(1, "apricot")

# remove — first occurrence by value
fruits.remove("fig")

# pop — remove and return by index (default: last)
last = fruits.pop()
second = fruits.pop(1)

# del — remove by index/slice
del fruits[0]
del fruits[1:3]

# clear — remove all elements
copy = fruits.copy()
copy.clear()

print(fruits)
```

### List Methods

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

# Searching
print(numbers.index(5))    # 4 — first index of 5
print(numbers.count(1))    # 2 — occurrences of 1
print(5 in numbers)        # True

# Sorting
numbers.sort()                          # in-place ascending
numbers.sort(reverse=True)             # in-place descending
numbers.sort(key=lambda x: -x)        # custom key

sorted_copy = sorted(numbers)          # returns new list
sorted_copy = sorted(numbers, key=abs) # sort by absolute value

# Reversing
numbers.reverse()              # in-place
rev_copy = list(reversed(numbers))  # returns iterator

# Length
print(len(numbers))   # 10

# Min, max, sum
print(min(numbers), max(numbers), sum(numbers))
```

### List as Stack and Queue

```python
# Stack (LIFO) — use append/pop
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
print(stack.pop())  # 3

# Queue (FIFO) — use collections.deque for efficiency
from collections import deque
queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
print(queue.popleft())  # 1
```

### Copying Lists

```python
original = [1, [2, 3], 4]

# Shallow copy — nested objects still shared
shallow1 = original.copy()
shallow2 = original[:]
shallow3 = list(original)

# Deep copy — fully independent
import copy
deep = copy.deepcopy(original)

# Demonstration
shallow1[1].append(99)
print(original)   # [1, [2, 3, 99], 4] — affected!
print(deep)       # [1, [2, 3], 4]    — unaffected
```

---

## Tuples

Tuples are **ordered**, **immutable** sequences. They're faster than lists and can be used as dictionary keys.

```python
# Creating tuples
empty     = ()
single    = (42,)      # trailing comma required for single element!
point     = (3, 4)
rgb       = (255, 128, 0)
mixed     = (1, "two", 3.0)

# Packing/unpacking
coords = 10, 20          # tuple packing
x, y   = coords          # tuple unpacking
a, *b, c = (1, 2, 3, 4, 5)  # extended unpacking: a=1, b=[2,3,4], c=5

# Named tuples — self-documenting
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)          # 3 4
print(p[0], p[1])        # 3 4
print(p._asdict())       # OrderedDict([('x', 3), ('y', 4)])

# dataclass alternative (Python 3.7+)
from dataclasses import dataclass

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

p3 = Point3D(1.0, 2.0, 3.0)
```

### Tuple vs List

| Feature     | Tuple       | List        |
|-------------|-------------|-------------|
| Mutable     | No          | Yes         |
| Syntax      | `(1, 2)`    | `[1, 2]`    |
| Performance | Faster      | Slower      |
| Hashable    | Yes (if elements are) | No |
| Use case    | Fixed data  | Dynamic data |

---

## Dictionaries

Dictionaries are **ordered** (Python 3.7+), **mutable** mappings of key-value pairs.

### Creating Dictionaries

```python
empty  = {}
person = {"name": "Alice", "age": 30, "city": "NYC"}

# From sequences
keys   = ["a", "b", "c"]
values = [1, 2, 3]
d = dict(zip(keys, values))      # {'a': 1, 'b': 2, 'c': 3}
d = dict(a=1, b=2, c=3)          # keyword arguments
d = {k: v for k, v in zip(keys, values)}  # dict comprehension

# fromkeys
defaults = dict.fromkeys(["x", "y", "z"], 0)  # {'x': 0, 'y': 0, 'z': 0}
```

### Accessing and Modifying

```python
person = {"name": "Alice", "age": 30}

# Access
print(person["name"])             # "Alice" — KeyError if missing
print(person.get("name"))         # "Alice" — None if missing
print(person.get("email", "N/A")) # "N/A" — default if missing

# Add / update
person["email"] = "alice@example.com"
person.update({"age": 31, "city": "LA"})
person.update(age=31)

# Delete
del person["city"]
removed = person.pop("email")         # removes and returns value
removed_item = person.popitem()       # removes and returns last (key, value)

# Set default — only sets if key missing
person.setdefault("score", 0)

print(person)
```

### Iterating Dictionaries

```python
d = {"a": 1, "b": 2, "c": 3}

# Keys
for key in d:                   # default iteration is over keys
    print(key)
for key in d.keys():
    print(key)

# Values
for value in d.values():
    print(value)

# Key-value pairs
for key, value in d.items():
    print(f"{key}: {value}")

# Checking membership
print("a" in d)          # True — checks keys
print(1 in d.values())   # True — checks values
```

### Dictionary Methods

```python
d = {"a": 1, "b": 2, "c": 3}

print(len(d))            # 3
print(list(d.keys()))    # ['a', 'b', 'c']
print(list(d.values()))  # [1, 2, 3]
print(list(d.items()))   # [('a', 1), ('b', 2), ('c', 3)]

# Merging (Python 3.9+)
d1 = {"a": 1}
d2 = {"b": 2}
merged = d1 | d2          # {'a': 1, 'b': 2}
d1 |= d2                  # in-place merge

# Merging (older)
merged = {**d1, **d2}
merged = dict(d1, **d2)
```

### Nested Dictionaries

```python
users = {
    "alice": {"age": 30, "email": "alice@example.com"},
    "bob":   {"age": 25, "email": "bob@example.com"},
}

print(users["alice"]["email"])  # alice@example.com

# Safe nested access
email = users.get("charlie", {}).get("email", "N/A")

# Updating nested
users["alice"]["age"] = 31
users.setdefault("charlie", {})["age"] = 28
```

---

## Sets

Sets are **unordered** collections of **unique**, **hashable** elements.

### Creating Sets

```python
empty    = set()          # NOT {} — that's an empty dict!
numbers  = {1, 2, 3, 4, 5}
from_list = set([1, 1, 2, 2, 3])  # {1, 2, 3} — duplicates removed

# Frozenset — immutable set (can be used as dict key)
frozen = frozenset([1, 2, 3])
```

### Set Operations

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# Union — elements in a OR b
print(a | b)             # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))

# Intersection — elements in BOTH
print(a & b)             # {4, 5}
print(a.intersection(b))

# Difference — in a but NOT b
print(a - b)             # {1, 2, 3}
print(a.difference(b))

# Symmetric difference — in a OR b, but NOT both
print(a ^ b)             # {1, 2, 3, 6, 7, 8}
print(a.symmetric_difference(b))

# Subset / Superset
print({1, 2}.issubset(a))    # True
print(a.issuperset({1, 2}))  # True
print(a.isdisjoint({6, 7}))  # True — no common elements
```

### Modifying Sets

```python
s = {1, 2, 3}

s.add(4)           # add single element
s.update([5, 6])   # add multiple elements

s.remove(6)        # removes — KeyError if not found
s.discard(99)      # removes — no error if not found
popped = s.pop()   # removes and returns arbitrary element

s.clear()          # empties the set
```

### Common Use Cases

```python
# Remove duplicates from list (order NOT preserved)
nums = [1, 2, 2, 3, 3, 3]
unique = list(set(nums))

# Fast membership testing — O(1) vs O(n) for lists
valid_ids = {101, 102, 103, 104}
if 102 in valid_ids:   # O(1)
    print("Valid")

# Finding common/unique elements
list_a = ["apple", "banana", "cherry"]
list_b = ["banana", "date", "cherry"]
common  = set(list_a) & set(list_b)  # {'banana', 'cherry'}
only_a  = set(list_a) - set(list_b)  # {'apple'}
```

---

## Collections Module

The `collections` module provides specialized container data types.

### `defaultdict`

```python
from collections import defaultdict

# Never raises KeyError — uses a default factory
word_count = defaultdict(int)
for word in "the cat sat on the mat".split():
    word_count[word] += 1

print(dict(word_count))  # {'the': 2, 'cat': 1, 'sat': 1, ...}

# Grouping
groups = defaultdict(list)
for name, dept in [("Alice", "Eng"), ("Bob", "HR"), ("Carol", "Eng")]:
    groups[dept].append(name)
print(dict(groups))  # {'Eng': ['Alice', 'Carol'], 'HR': ['Bob']}
```

### `Counter`

```python
from collections import Counter

# Count elements
counter = Counter("mississippi")
print(counter)  # Counter({'i': 4, 's': 4, 'p': 2, 'm': 1})

counter = Counter(["a", "b", "a", "c", "b", "a"])
print(counter.most_common(2))  # [('a', 3), ('b', 2)]
print(counter.total())         # 6

# Arithmetic
c1 = Counter(a=3, b=2)
c2 = Counter(a=1, b=4, c=1)
print(c1 + c2)  # Counter({'b': 6, 'a': 4, 'c': 1})
print(c1 - c2)  # Counter({'a': 2}) — negatives dropped
print(c1 & c2)  # Counter({'a': 1, 'b': 2}) — intersection (min)
print(c1 | c2)  # Counter({'b': 4, 'a': 3, 'c': 1}) — union (max)
```

### `OrderedDict`

```python
from collections import OrderedDict

# In Python 3.7+ regular dicts maintain insertion order.
# OrderedDict has extra features:
od = OrderedDict()
od["a"] = 1
od["b"] = 2
od["c"] = 3

od.move_to_end("a")        # move to end
od.move_to_end("c", last=False)  # move to front
print(list(od.keys()))     # ['c', 'b', 'a']

# Useful: equal comparison considers order
d1 = OrderedDict([("a", 1), ("b", 2)])
d2 = OrderedDict([("b", 2), ("a", 1)])
print(d1 == d2)  # False — order matters
```

### `deque`

```python
from collections import deque

# Double-ended queue — O(1) append/pop from both ends
d = deque([1, 2, 3], maxlen=5)  # maxlen: auto-drops from opposite end

d.append(4)       # add to right
d.appendleft(0)   # add to left
d.extend([5, 6])  # extend right
d.extendleft([-1, -2])  # extend left (note: order reverses)

right = d.pop()     # remove from right
left  = d.popleft() # remove from left

d.rotate(2)         # rotate right by 2
d.rotate(-2)        # rotate left by 2

print(list(d))
```

### `ChainMap`

```python
from collections import ChainMap

defaults = {"color": "red", "size": "medium"}
user_prefs = {"color": "blue"}
env_overrides = {"size": "large"}

# Priority: env_overrides > user_prefs > defaults
config = ChainMap(env_overrides, user_prefs, defaults)
print(config["color"])  # blue
print(config["size"])   # large

# Only affects the first map
config["color"] = "green"
print(user_prefs)   # unaffected
```

---

## Comprehensions

### List Comprehension

```python
# [expression for item in iterable if condition]

squares = [x**2 for x in range(1, 11)]
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

evens = [x for x in range(20) if x % 2 == 0]
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Nested — flatten a 2D list
matrix = [[1,2,3],[4,5,6],[7,8,9]]
flat = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Conditional expression (ternary)
labels = ["even" if x % 2 == 0 else "odd" for x in range(5)]
# ['even', 'odd', 'even', 'odd', 'even']
```

### Dictionary Comprehension

```python
# {key_expr: val_expr for item in iterable if condition}

squares = {x: x**2 for x in range(1, 6)}
# {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Invert a dict
orig    = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in orig.items()}
# {1: 'a', 2: 'b', 3: 'c'}

# Filter
scores = {"Alice": 85, "Bob": 42, "Carol": 91}
passing = {name: score for name, score in scores.items() if score >= 60}
```

### Set Comprehension

```python
unique_squares = {x**2 for x in range(-5, 6)}
# {0, 1, 4, 9, 16, 25}
```

### Generator Expression

```python
# Like list comprehension but lazy — generates values on demand
gen = (x**2 for x in range(10))
print(type(gen))   # <class 'generator'>
print(next(gen))   # 0
print(next(gen))   # 1

# Memory efficient — ideal for large datasets
total = sum(x**2 for x in range(1_000_000))  # no intermediate list!

# Can be iterated only once
gen = (x for x in range(3))
print(list(gen))  # [0, 1, 2]
print(list(gen))  # [] — exhausted!
```

---

## Comparison: When to Use What

| Data Structure | Ordered | Mutable | Duplicates | Key-Value | Use When |
|---|---|---|---|---|---|
| `list` | ✅ | ✅ | ✅ | ❌ | Ordered sequence, frequent mutation |
| `tuple` | ✅ | ❌ | ✅ | ❌ | Fixed data, dict keys, unpacking |
| `dict` | ✅ (3.7+) | ✅ | keys: ❌ | ✅ | Key lookup, mapping, JSON-like data |
| `set` | ❌ | ✅ | ❌ | ❌ | Uniqueness, membership tests, set ops |
| `frozenset` | ❌ | ❌ | ❌ | ❌ | Immutable set, dict key |
| `deque` | ✅ | ✅ | ✅ | ❌ | Fast both-end append/pop, queues |
| `Counter` | N/A | ✅ | N/A | ✅ | Counting / frequency |
