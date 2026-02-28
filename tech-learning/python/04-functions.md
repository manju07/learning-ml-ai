# Python Functions

## Table of Contents
- [Defining Functions](#defining-functions)
- [Arguments and Parameters](#arguments-and-parameters)
- [Return Values](#return-values)
- [Lambda Functions](#lambda-functions)
- [Higher-Order Functions](#higher-order-functions)
- [Closures](#closures)
- [Decorators](#decorators)
- [Generators](#generators)
- [Recursion](#recursion)
- [Type Hints](#type-hints)

---

## Defining Functions

```python
def greet(name):
    """Return a personalized greeting."""
    return f"Hello, {name}!"

# Calling
message = greet("Alice")
print(message)   # Hello, Alice!

# Functions are first-class objects
func = greet
print(func("Bob"))  # Hello, Bob!

# Store in data structures
functions = [len, str.upper, str.strip]
text = "  python  "
for f in functions:
    print(f(text))
```

---

## Arguments and Parameters

### Positional Arguments

```python
def add(a, b, c):
    return a + b + c

print(add(1, 2, 3))    # 6 — positional
print(add(1, c=3, b=2)) # 6 — keyword (any order)
```

### Default Parameters

```python
def greet(name, greeting="Hello", punctuation="!"):
    return f"{greeting}, {name}{punctuation}"

print(greet("Alice"))                     # Hello, Alice!
print(greet("Bob", "Hi"))                # Hi, Bob!
print(greet("Carol", punctuation="...")) # Hello, Carol...

# IMPORTANT: mutable defaults are shared across calls!
def bad_append(item, lst=[]):   # BUG!
    lst.append(item)
    return lst

print(bad_append(1))   # [1]
print(bad_append(2))   # [1, 2] — same list object!

# FIX: use None as default
def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### `*args` — Variable Positional Arguments

```python
def sum_all(*args):
    """Accept any number of positional arguments."""
    return sum(args)

print(sum_all(1, 2, 3))         # 6
print(sum_all(1, 2, 3, 4, 5))   # 15

# args is a tuple
def show(*args):
    print(type(args), args)

show(1, "two", 3.0)  # <class 'tuple'> (1, 'two', 3.0)

# Unpacking with *
numbers = [1, 2, 3]
print(sum_all(*numbers))   # 6 — unpacks list into args
```

### `**kwargs` — Variable Keyword Arguments

```python
def describe(**kwargs):
    """Accept any number of keyword arguments."""
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

describe(name="Alice", age=30, city="NYC")
# name: Alice
# age: 30
# city: NYC

# kwargs is a dict
def show(**kwargs):
    print(type(kwargs))   # <class 'dict'>

# Unpacking with **
options = {"color": "blue", "size": "large"}
describe(**options)
```

### Combined Parameters

```python
# Order: positional, *args, keyword-only, **kwargs
def full_function(a, b, *args, keyword_only=True, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"keyword_only={keyword_only}")
    print(f"kwargs={kwargs}")

full_function(1, 2, 3, 4, keyword_only=False, x=10, y=20)
# a=1, b=2
# args=(3, 4)
# keyword_only=False
# kwargs={'x': 10, 'y': 20}
```

### Keyword-Only Parameters

```python
# Parameters after * must be passed as keyword arguments
def draw_point(x, y, *, color="black", size=1):
    print(f"({x},{y}) color={color} size={size}")

draw_point(3, 4)                      # OK
draw_point(3, 4, color="red")         # OK
draw_point(3, 4, "red")              # TypeError! color must be keyword
```

### Positional-Only Parameters (Python 3.8+)

```python
# Parameters before / must be passed positionally
def add(a, b, /, c=0):
    return a + b + c

add(1, 2)         # OK
add(1, 2, 3)      # OK
add(1, b=2)       # TypeError! b must be positional
```

---

## Return Values

```python
# Return multiple values (returns a tuple)
def min_max(numbers):
    return min(numbers), max(numbers)

lo, hi = min_max([3, 1, 4, 1, 5, 9])
print(lo, hi)   # 1 9

# Returning None explicitly
def log(message, level="INFO"):
    print(f"[{level}] {message}")
    # implicitly returns None

result = log("test")
print(result)   # None

# Early return for guard clauses
def process(data):
    if data is None:
        return None
    if not data:
        return []
    return [item * 2 for item in data]
```

---

## Lambda Functions

Anonymous, single-expression functions.

```python
# Syntax: lambda parameters: expression
square = lambda x: x ** 2
add    = lambda x, y: x + y

print(square(5))     # 25
print(add(3, 4))     # 7

# Common uses with higher-order functions
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_nums = sorted(numbers, key=lambda x: -x)  # descending

words = ["banana", "apple", "cherry", "date"]
sorted_words = sorted(words, key=lambda w: len(w))  # by length
# ['date', 'apple', 'banana', 'cherry']

# Sorting complex objects
people = [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}]
by_age = sorted(people, key=lambda p: p["age"])

# Lambda with conditional
clamp = lambda x, lo, hi: max(lo, min(x, hi))
print(clamp(15, 0, 10))  # 10
```

---

## Higher-Order Functions

Functions that accept or return other functions.

### `map`

```python
numbers = [1, 2, 3, 4, 5]

# Apply function to every element — returns iterator
squared = list(map(lambda x: x**2, numbers))
# [1, 4, 9, 16, 25]

# With named function
def celsius_to_fahrenheit(c):
    return c * 9/5 + 32

temps_c = [0, 20, 37, 100]
temps_f = list(map(celsius_to_fahrenheit, temps_c))

# Multiple iterables
a = [1, 2, 3]
b = [10, 20, 30]
sums = list(map(lambda x, y: x + y, a, b))  # [11, 22, 33]

# Modern alternative: list comprehension (usually preferred)
squared = [x**2 for x in numbers]
```

### `filter`

```python
numbers = range(-5, 6)

# Keep elements where function returns True
positives = list(filter(lambda x: x > 0, numbers))
# [1, 2, 3, 4, 5]

# None as function — filter falsy values
values = [0, 1, "", "hello", None, [], [1, 2]]
truthy = list(filter(None, values))
# [1, 'hello', [1, 2]]

# List comprehension alternative
positives = [x for x in numbers if x > 0]
```

### `reduce`

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# Cumulatively apply function to elements
product = reduce(lambda acc, x: acc * x, numbers)   # 120
total   = reduce(lambda acc, x: acc + x, numbers)   # 15
total   = reduce(lambda acc, x: acc + x, numbers, 0) # initial value

# Find max without built-in
maximum = reduce(lambda a, b: a if a > b else b, numbers)  # 5
```

### `sorted` with `key`

```python
from operator import attrgetter, itemgetter

class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

students = [
    Student("Alice", 90),
    Student("Bob", 85),
    Student("Carol", 92),
]

# Sort by attribute
by_grade = sorted(students, key=attrgetter("grade"), reverse=True)

# Sort dict list by key
people = [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}]
by_name = sorted(people, key=itemgetter("name"))

# Multi-key sort
data = [(1, "b"), (2, "a"), (1, "a")]
sorted_data = sorted(data, key=lambda x: (x[0], x[1]))
# [(1, 'a'), (1, 'b'), (2, 'a')]
```

---

## Closures

A closure is a function that **remembers** variables from its enclosing scope.

```python
def make_multiplier(factor):
    def multiply(n):
        return n * factor   # 'factor' is captured from enclosing scope
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# Inspect closure
print(double.__closure__[0].cell_contents)  # 2
```

### `nonlocal`

```python
def make_counter(start=0):
    count = start

    def increment(by=1):
        nonlocal count    # tells Python to use enclosing scope variable
        count += by
        return count

    def reset():
        nonlocal count
        count = start

    return increment, reset

inc, rst = make_counter(10)
print(inc())    # 11
print(inc(5))   # 16
rst()
print(inc())    # 11 — reset worked
```

---

## Decorators

Decorators wrap a function to extend its behavior without modifying it.

### Basic Decorator

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Before {func.__name__}")
        result = func(*args, **kwargs)
        print(f"After {func.__name__}")
        return result
    return wrapper

@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("Alice")
# Before say_hello
# Hello, Alice!
# After say_hello

# Equivalent to:
# say_hello = my_decorator(say_hello)
```

### Preserving Metadata with `functools.wraps`

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)      # copies __name__, __doc__, etc.
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def my_function():
    """My function docstring."""
    pass

print(my_function.__name__)   # my_function (not wrapper!)
print(my_function.__doc__)    # My function docstring.
```

### Decorator with Arguments

```python
from functools import wraps

def repeat(times):
    """Decorator that repeats a function call `times` times."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say(message):
    print(message)

say("Hello!")
# Hello!
# Hello!
# Hello!
```

### Practical Decorator Examples

```python
import time
from functools import wraps

# Timing decorator
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

# Caching decorator (memoization)
def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

# Or use built-in:
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Retry decorator
def retry(max_attempts=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying...")
        return wrapper
    return decorator

@retry(max_attempts=3, exceptions=(ConnectionError,))
def fetch_data(url):
    # ...
    pass
```

### Class-Based Decorators

```python
class singleton:
    """Ensure only one instance of a class exists."""
    def __init__(self, cls):
        self._cls = cls
        self._instance = None

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = self._cls(*args, **kwargs)
        return self._instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Connecting to database...")

db1 = DatabaseConnection()  # Connecting to database...
db2 = DatabaseConnection()  # (no output — same instance)
print(db1 is db2)           # True
```

### Stacking Decorators

```python
@decorator_a
@decorator_b
@decorator_c
def my_function():
    pass

# Equivalent to:
# my_function = decorator_a(decorator_b(decorator_c(my_function)))
# Applied bottom-up, executed top-down
```

---

## Generators

Generators produce values **lazily**, one at a time, conserving memory.

```python
# Generator function — uses yield
def count_up(start, end):
    current = start
    while current <= end:
        yield current
        current += 1

gen = count_up(1, 5)
print(next(gen))   # 1
print(next(gen))   # 2
for n in gen:      # continues from where left off
    print(n)       # 3, 4, 5

# StopIteration raised when exhausted
gen2 = count_up(1, 2)
print(next(gen2))  # 1
print(next(gen2))  # 2
# next(gen2)       # StopIteration!
```

### `yield from`

```python
def chain(*iterables):
    for it in iterables:
        yield from it   # delegates to sub-iterable

result = list(chain([1, 2], "abc", (3, 4)))
# [1, 2, 'a', 'b', 'c', 3, 4]

# Recursive generator
def flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

data = [1, [2, [3, 4], 5], 6]
print(list(flatten(data)))  # [1, 2, 3, 4, 5, 6]
```

### Generator Pipelines

```python
def read_lines(filename):
    with open(filename) as f:
        yield from f

def filter_blanks(lines):
    for line in lines:
        if line.strip():
            yield line

def strip_lines(lines):
    for line in lines:
        yield line.strip()

# Build a lazy pipeline — no file fully loaded into memory
pipeline = strip_lines(filter_blanks(read_lines("data.txt")))
for line in pipeline:
    print(line)
```

### `send()` and Two-Way Communication

```python
def accumulator():
    total = 0
    while True:
        value = yield total   # yield sends current total, receives new value
        if value is None:
            break
        total += value

gen = accumulator()
next(gen)           # prime the generator (advance to first yield)
print(gen.send(10)) # 10
print(gen.send(20)) # 30
print(gen.send(5))  # 35
```

---

## Recursion

```python
def factorial(n):
    if n == 0:       # base case
        return 1
    return n * factorial(n - 1)   # recursive case

print(factorial(10))  # 3628800

# Python's default recursion limit
import sys
print(sys.getrecursionlimit())  # 1000
sys.setrecursionlimit(5000)     # increase if needed

# Tail recursion with accumulator (more efficient)
def factorial_tail(n, acc=1):
    if n == 0:
        return acc
    return factorial_tail(n - 1, acc * n)

# Tree traversal
def tree_sum(node):
    if node is None:
        return 0
    return node.val + tree_sum(node.left) + tree_sum(node.right)

# Flatten nested structure
def deep_sum(nested):
    total = 0
    for item in nested:
        if isinstance(item, (list, tuple)):
            total += deep_sum(item)
        else:
            total += item
    return total

print(deep_sum([1, [2, [3, 4]], [5, 6]]))  # 21
```

---

## Type Hints

Type hints improve readability and enable static analysis with tools like `mypy`.

```python
from typing import Optional, Union, List, Dict, Tuple, Callable, Any, TypeVar

# Basic annotations
def greet(name: str) -> str:
    return f"Hello, {name}"

def add(a: int, b: int) -> int:
    return a + b

# Optional (value or None)
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Union (one of multiple types)
def process(value: Union[int, str]) -> str:
    return str(value)

# Python 3.10+ — use | instead of Union
def process(value: int | str) -> str:
    return str(value)

# Collections
def average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)

def build_index(items: list[str]) -> dict[str, int]:
    return {item: i for i, item in enumerate(items)}

# Callable
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# TypeVar — generic functions
T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

# Literal types (Python 3.8+)
from typing import Literal

def set_direction(direction: Literal["north", "south", "east", "west"]) -> None:
    ...

# TypedDict (Python 3.8+)
from typing import TypedDict

class Movie(TypedDict):
    title: str
    year: int
    rating: float

movie: Movie = {"title": "Inception", "year": 2010, "rating": 8.8}
```

---

## `functools` Utilities

```python
from functools import partial, reduce, lru_cache, cached_property

# partial — fix some arguments
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube   = partial(power, exponent=3)

print(square(5))   # 25
print(cube(3))     # 27

# cached_property — computed once, then stored
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @cached_property
    def area(self):
        import math
        print("Computing area...")
        return math.pi * self.radius ** 2

c = Circle(5)
print(c.area)  # Computing area... 78.53...
print(c.area)  # 78.53... (cached — no recomputation)
```
