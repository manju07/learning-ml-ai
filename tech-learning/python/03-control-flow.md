# Python Control Flow

## Table of Contents
- [Conditional Statements](#conditional-statements)
- [Loops](#loops)
- [Loop Control](#loop-control)
- [Pattern Matching (match/case)](#pattern-matching)
- [Exception Handling](#exception-handling)

---

## Conditional Statements

### `if / elif / else`

```python
score = 78

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")   # Grade: C
```

### Ternary (Conditional Expression)

```python
# value_if_true if condition else value_if_false
age = 20
status = "adult" if age >= 18 else "minor"

# Can be chained (use sparingly — readability)
x = 15
category = "high" if x > 20 else "medium" if x > 10 else "low"
# category = "medium"
```

### Truthiness

```python
# All these are falsy
falsy = [0, 0.0, "", [], {}, set(), None, False]

# Pythonic checks
items = []
if not items:           # idiomatic
    print("Empty list")

name = None
display = name or "Anonymous"   # "Anonymous"

config = None
value = config and config.get("key")  # None — short-circuit
```

---

## Loops

### `for` Loop

The `for` loop iterates over any **iterable** (list, tuple, string, range, dict, file, generator…).

```python
# Iterating a list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Iterating a string
for char in "Python":
    print(char, end=" ")   # P y t h o n

# Iterating a dict
person = {"name": "Alice", "age": 30}
for key in person:          # iterates keys
    print(key)
for key, value in person.items():   # key-value pairs
    print(f"{key}: {value}")

# range(start, stop, step)
for i in range(5):           # 0 1 2 3 4
    print(i, end=" ")
for i in range(1, 10, 2):   # 1 3 5 7 9
    print(i, end=" ")
for i in range(10, 0, -1):  # 10 9 8 ... 1
    print(i, end=" ")
```

### Enumerate

```python
fruits = ["apple", "banana", "cherry"]

# Without enumerate
for i in range(len(fruits)):
    print(i, fruits[i])

# With enumerate — Pythonic!
for i, fruit in enumerate(fruits):
    print(i, fruit)

# Custom start index
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
# 1. apple
# 2. banana
# 3. cherry
```

### Zip

```python
names  = ["Alice", "Bob", "Carol"]
scores = [95, 87, 92]
grades = ["A", "B", "A"]

# Iterate multiple iterables in parallel
for name, score, grade in zip(names, scores, grades):
    print(f"{name}: {score} ({grade})")

# zip stops at the shortest iterable
a = [1, 2, 3]
b = [10, 20]
print(list(zip(a, b)))  # [(1, 10), (2, 20)]

# zip_longest to fill with a default
from itertools import zip_longest
print(list(zip_longest(a, b, fillvalue=0)))  # [(1, 10), (2, 20), (3, 0)]

# Unzipping
pairs = [(1, "a"), (2, "b"), (3, "c")]
numbers, letters = zip(*pairs)
print(numbers)  # (1, 2, 3)
print(letters)  # ('a', 'b', 'c')
```

### `while` Loop

```python
# Basic while
count = 0
while count < 5:
    print(count)
    count += 1

# Infinite loop with break
while True:
    user_input = input("Type 'quit' to exit: ")
    if user_input == "quit":
        break
    print(f"You typed: {user_input}")

# while with condition
n = 100
while n > 1:
    n //= 2
print(n)  # 0 (or 1)
```

### `for` / `while` with `else`

The `else` block runs if the loop completes **without** hitting a `break`.

```python
# Search with for-else
def find_prime(numbers):
    for n in numbers:
        for divisor in range(2, n):
            if n % divisor == 0:
                break   # not prime
        else:
            return n    # no divisor found → prime
    return None

print(find_prime([4, 6, 7, 9]))  # 7

# while-else
attempts = 3
while attempts > 0:
    password = input("Password: ")
    if password == "secret":
        print("Access granted")
        break
    attempts -= 1
else:
    print("Account locked")
```

---

## Loop Control

### `break`

Exits the loop immediately.

```python
for n in range(100):
    if n * n > 50:
        print(f"First n where n² > 50: {n}")
        break
```

### `continue`

Skips the rest of the current iteration and moves to the next.

```python
for n in range(10):
    if n % 2 == 0:
        continue    # skip even numbers
    print(n, end=" ")   # 1 3 5 7 9
```

### `pass`

A no-op placeholder — syntactically required but nothing to do.

```python
for _ in range(5):
    pass   # do nothing

class EmptyClass:
    pass

def stub_function():
    pass
```

---

## Pattern Matching

Python 3.10+ introduces `match / case` — structural pattern matching.

### Basic Patterns

```python
command = "quit"

match command:
    case "quit":
        print("Quitting...")
    case "help":
        print("Help text...")
    case _:     # wildcard — matches anything
        print(f"Unknown command: {command}")
```

### Literal Patterns

```python
def http_status(status: int) -> str:
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case 500 | 503:      # OR pattern
            return "Server Error"
        case _:
            return "Unknown"
```

### Sequence Patterns

```python
point = (0, 5)

match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"On Y-axis at {y}")
    case (x, 0):
        print(f"On X-axis at {x}")
    case (x, y):
        print(f"Point at ({x}, {y})")
```

### Mapping Patterns

```python
def process_event(event: dict):
    match event:
        case {"type": "click", "button": btn}:
            print(f"Mouse click: {btn}")
        case {"type": "keypress", "key": key}:
            print(f"Key pressed: {key}")
        case {"type": t}:
            print(f"Unknown event type: {t}")
        case _:
            print("Invalid event")

process_event({"type": "keypress", "key": "Enter"})
```

### Class Patterns

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    radius: float

shape = Circle(center=Point(0, 0), radius=5)

match shape:
    case Circle(center=Point(x=0, y=0), radius=r):
        print(f"Circle at origin with radius {r}")
    case Circle(center=Point(x, y), radius=r):
        print(f"Circle at ({x},{y}) with radius {r}")
```

### Guard Clauses

```python
values = [1, -2, 3, -4, 5]

for v in values:
    match v:
        case n if n > 0:
            print(f"+{n}")
        case n if n < 0:
            print(f"{n}")
        case _:
            print("zero")
```

---

## Exception Handling

### `try / except / else / finally`

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero!")
        return None
    except TypeError as e:
        print(f"Type error: {e}")
        return None
    else:
        # runs if NO exception was raised
        print(f"Result: {result}")
        return result
    finally:
        # ALWAYS runs — cleanup
        print("Division attempt complete")

divide(10, 2)    # Result: 10.0 / Division attempt complete
divide(10, 0)    # Cannot divide by zero! / Division attempt complete
```

### Catching Multiple Exceptions

```python
try:
    value = int(input("Enter a number: "))
    result = 100 / value
except (ValueError, ZeroDivisionError) as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    raise   # re-raise the exception
```

### Raising Exceptions

```python
def set_age(age: int):
    if not isinstance(age, int):
        raise TypeError(f"Age must be int, got {type(age).__name__}")
    if age < 0 or age > 150:
        raise ValueError(f"Age must be between 0 and 150, got {age}")
    return age

# Raise with context
try:
    int("abc")
except ValueError as original:
    raise RuntimeError("Failed to parse config") from original
```

### Custom Exceptions

```python
class AppError(Exception):
    """Base class for application errors."""

class ValidationError(AppError):
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error on '{field}': {message}")

class DatabaseError(AppError):
    pass

# Usage
try:
    raise ValidationError("email", "invalid format")
except ValidationError as e:
    print(f"Field: {e.field}")    # email
    print(f"Message: {e.message}")  # invalid format
    print(e)                      # Validation error on 'email': invalid format
```

### Exception Hierarchy

```python
# BaseException
#   SystemExit
#   KeyboardInterrupt
#   Exception
#     TypeError
#     ValueError
#     RuntimeError
#     OSError
#       FileNotFoundError
#       PermissionError
#     LookupError
#       IndexError
#       KeyError
#     ArithmeticError
#       ZeroDivisionError
#       OverflowError
#     StopIteration

# Catch parent to handle all subclasses
try:
    d = {}
    print(d["missing"])
except LookupError:
    print("Handles both KeyError and IndexError")
```

### Context Managers and Exceptions

```python
# with statements suppress cleanup boilerplate
with open("file.txt", "w") as f:
    f.write("Hello")
# file auto-closed even if exception occurs

# contextlib.suppress — silence specific exceptions
from contextlib import suppress

with suppress(FileNotFoundError):
    import os
    os.remove("nonexistent.txt")   # silently ignored
```

---

## Itertools for Advanced Iteration

```python
import itertools

# chain — combine iterables
for x in itertools.chain([1, 2], [3, 4], [5]):
    print(x, end=" ")   # 1 2 3 4 5

# cycle — infinite repetition
colors = itertools.cycle(["red", "green", "blue"])
for _, color in zip(range(6), colors):
    print(color, end=" ")   # red green blue red green blue

# repeat
for x in itertools.repeat("hello", 3):
    print(x)    # hello hello hello

# product — cartesian product
for a, b in itertools.product([1, 2], ["x", "y"]):
    print(a, b)   # 1 x / 1 y / 2 x / 2 y

# permutations, combinations
from itertools import permutations, combinations
print(list(permutations("ABC", 2)))    # [('A','B'),('A','C'),('B','A'),...]
print(list(combinations("ABC", 2)))    # [('A','B'),('A','C'),('B','C')]

# islice — slice any iterable
import itertools
first_5 = list(itertools.islice(range(1_000_000), 5))  # [0,1,2,3,4]

# takewhile / dropwhile
print(list(itertools.takewhile(lambda x: x < 5, [1,3,5,7,2])))  # [1, 3]
print(list(itertools.dropwhile(lambda x: x < 5, [1,3,5,7,2])))  # [5, 7, 2]

# groupby — group consecutive items
data = [("Alice","Eng"),("Bob","Eng"),("Carol","HR"),("Dave","HR")]
for dept, members in itertools.groupby(data, key=lambda x: x[1]):
    print(dept, [m[0] for m in members])
```
