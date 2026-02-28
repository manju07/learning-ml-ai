# Python Advanced Concepts

## Table of Contents
- [Iterators and the Iterator Protocol](#iterators-and-the-iterator-protocol)
- [Context Managers](#context-managers)
- [Descriptors](#descriptors)
- [Type System Advanced](#type-system-advanced)
- [Memory Management](#memory-management)
- [Performance Optimization](#performance-optimization)
- [Introspection and Reflection](#introspection-and-reflection)
- [Functional Programming](#functional-programming)

---

## Iterators and the Iterator Protocol

An **iterable** implements `__iter__()`.  
An **iterator** implements both `__iter__()` and `__next__()`.

```python
class CountUp:
    """An iterator that counts from start to end."""

    def __init__(self, start, end):
        self.current = start
        self.end     = end

    def __iter__(self):
        return self    # iterator IS the iterable

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


counter = CountUp(1, 5)
for n in counter:
    print(n, end=" ")   # 1 2 3 4 5

# Manual iteration
it = iter([1, 2, 3])       # calls list.__iter__()
print(next(it))            # 1
print(next(it))            # 2
print(next(it, "default")) # 3 — next with default (no StopIteration)
print(next(it, "default")) # 'default' (exhausted)
```

### Infinite Iterator

```python
class InfiniteCounter:
    def __init__(self, start=0, step=1):
        self.current = start
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        value = self.current
        self.current += self.step
        return value

import itertools

# Use islice to take finite number from infinite iterator
counter = InfiniteCounter()
first_10 = list(itertools.islice(counter, 10))
print(first_10)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# itertools provides many useful infinite iterators
for i, val in zip(range(5), itertools.cycle([1, 2, 3])):
    print(val, end=" ")   # 1 2 3 1 2
```

### Custom Iterable vs Iterator

```python
class NumberRange:
    """Iterable (not iterator) — supports multiple passes."""

    def __init__(self, start, end):
        self.start = start
        self.end   = end

    def __iter__(self):
        # Returns a fresh iterator each time
        return NumberRangeIterator(self.start, self.end)


class NumberRangeIterator:
    """The actual iterator — one-pass."""

    def __init__(self, start, end):
        self.current = start
        self.end     = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        val = self.current
        self.current += 1
        return val


r = NumberRange(1, 4)
print(list(r))  # [1, 2, 3]
print(list(r))  # [1, 2, 3] — works again! (new iterator each time)
```

---

## Context Managers

### Class-Based Context Manager

```python
class Timer:
    """Measure elapsed time for a block of code."""
    import time as _time

    def __enter__(self):
        self.start = self._time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = self._time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False   # don't suppress exceptions

with Timer() as t:
    sum(range(1_000_000))
# Elapsed: 0.0234s
```

### `contextlib.contextmanager`

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label=""):
    start = time.perf_counter()
    try:
        yield    # code inside 'with' block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

with timer("Loop"):
    total = sum(range(1_000_000))

@contextmanager
def managed_resource(name):
    print(f"Acquiring {name}")
    resource = {"name": name, "active": True}
    try:
        yield resource
    except Exception as e:
        print(f"Error with {name}: {e}")
        raise
    finally:
        resource["active"] = False
        print(f"Released {name}")

with managed_resource("DB Connection") as conn:
    print(f"Using {conn['name']}")
```

### `contextlib` Utilities

```python
from contextlib import (
    suppress,
    redirect_stdout,
    redirect_stderr,
    ExitStack,
    asynccontextmanager,
)
import io

# suppress — silence specific exceptions
with suppress(FileNotFoundError, PermissionError):
    import os
    os.remove("nonexistent.txt")

# redirect_stdout — capture output
buffer = io.StringIO()
with redirect_stdout(buffer):
    print("This goes to buffer, not console")
output = buffer.getvalue()
print(f"Captured: {output!r}")

# ExitStack — dynamic number of context managers
files = ["a.txt", "b.txt", "c.txt"]
with ExitStack() as stack:
    handles = [stack.enter_context(open(f, "w")) for f in files]
    for h, name in zip(handles, files):
        h.write(f"Content of {name}")
# All files closed automatically
```

---

## Descriptors

Descriptors customize attribute access (get, set, delete) on a class.

```python
class Validator:
    """A descriptor that validates attribute values."""

    def __init__(self, min_val=None, max_val=None, type_=None):
        self.min_val = min_val
        self.max_val = max_val
        self.type_   = type_
        self.name    = None    # set by __set_name__

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self   # accessed from class, return descriptor itself
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        if self.type_ and not isinstance(value, self.type_):
            raise TypeError(f"{self.name} must be {self.type_.__name__}, got {type(value).__name__}")
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"{self.name} must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"{self.name} must be <= {self.max_val}")
        setattr(obj, self.private_name, value)

    def __delete__(self, obj):
        delattr(obj, self.private_name)


class Person:
    name = Validator(type_=str)
    age  = Validator(min_val=0, max_val=150, type_=int)

    def __init__(self, name, age):
        self.name = name   # triggers Validator.__set__
        self.age  = age

p = Person("Alice", 30)
print(p.name)   # Alice
print(p.age)    # 30

# p.age = -5    # ValueError!
# p.name = 123  # TypeError!

# Non-data descriptor (only __get__, no __set__)
class LazyProperty:
    """Compute property once and cache on instance."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.name, value)    # cache on instance (shadows descriptor)
        return value


class DataAnalysis:
    def __init__(self, data):
        self.data = data

    @LazyProperty
    def mean(self):
        print("Computing mean...")
        return sum(self.data) / len(self.data)

analysis = DataAnalysis([1, 2, 3, 4, 5])
print(analysis.mean)   # Computing mean... 3.0
print(analysis.mean)   # 3.0 (cached — no recomputation)
```

---

## Type System Advanced

### Generic Types

```python
from typing import TypeVar, Generic, Iterator

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

class Stack(Generic[T]):
    """A type-safe stack."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(reversed(self._items))


stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
print(stack.pop())   # 2
```

### `ParamSpec` and `Concatenate`

```python
from typing import ParamSpec, Callable, TypeVar
from functools import wraps
import time

P = ParamSpec("P")
T = TypeVar("T")

def timed(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that preserves full signature typing."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter() - start:.4f}s")
        return result
    return wrapper

@timed
def compute(n: int, factor: float = 1.0) -> float:
    return sum(range(n)) * factor

result: float = compute(1_000_000, factor=2.0)  # type-safe!
```

### `overload`

```python
from typing import overload, Union

@overload
def process(value: int) -> str: ...
@overload
def process(value: str) -> int: ...
@overload
def process(value: list[int]) -> float: ...

def process(value):
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, str):
        return int(value)
    elif isinstance(value, list):
        return sum(value) / len(value)
    raise TypeError(f"Unsupported type: {type(value)}")

# Type checker knows the exact return type based on argument type
s: str   = process(42)
n: int   = process("42")
f: float = process([1, 2, 3])
```

---

## Memory Management

### Reference Counting and Garbage Collection

```python
import sys
import gc

x = [1, 2, 3]
print(sys.getrefcount(x))  # 2 (x + getrefcount arg)

y = x
print(sys.getrefcount(x))  # 3

del y
print(sys.getrefcount(x))  # 2

# Circular references — handled by gc, not ref counting
import gc

class Node:
    def __init__(self):
        self.sibling = None

a = Node()
b = Node()
a.sibling = b
b.sibling = a   # circular!

del a, b        # ref count doesn't reach 0
gc.collect()    # explicitly run garbage collector

# Disable/enable gc
gc.disable()
gc.enable()
print(gc.isenabled())
```

### `weakref` — Weak References

```python
import weakref

class Cache:
    def __init__(self):
        self._cache: dict[int, weakref.ref] = {}

    def get(self, key):
        ref = self._cache.get(key)
        return ref() if ref is not None else None   # ref() returns obj or None

    def set(self, key, value):
        self._cache[key] = weakref.ref(value)   # weak reference — doesn't prevent GC


class HeavyObject:
    def __init__(self, data):
        self.data = data
    def __del__(self):
        print(f"HeavyObject deleted")


cache = Cache()
obj = HeavyObject("important data")
cache.set(1, obj)

print(cache.get(1))   # HeavyObject object
del obj               # HeavyObject deleted (weakref doesn't prevent GC)
print(cache.get(1))   # None
```

### Memory-Efficient Patterns

```python
import sys

# Generators vs lists
list_result = [x**2 for x in range(10_000)]
gen_result  = (x**2 for x in range(10_000))

print(sys.getsizeof(list_result))   # ~87,616 bytes
print(sys.getsizeof(gen_result))    # ~104 bytes (just the generator object)

# __slots__ saves memory
class WithDict:
    def __init__(self, x, y):
        self.x, self.y = x, y

class WithSlots:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y

import tracemalloc
tracemalloc.start()

objs_dict  = [WithDict(i, i) for i in range(10000)]
snapshot   = tracemalloc.take_snapshot()
top = snapshot.statistics("lineno")[0]
print(f"WithDict: {top.size:,} bytes")
```

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats
import io

def slow_function():
    return sum(i*i for i in range(1_000_000))

# Profile to string
pr = cProfile.Profile()
pr.enable()
slow_function()
pr.disable()

s = io.StringIO()
stats = pstats.Stats(pr, stream=s).sort_stats("cumulative")
stats.print_stats(10)  # top 10
print(s.getvalue())

# Profile with command line
# python -m cProfile -s cumulative my_script.py

# Line profiler (pip install line-profiler)
# @profile decorator — use with kernprof -l -v my_script.py

# Memory profiler (pip install memory-profiler)
# @profile decorator — use with python -m memory_profiler my_script.py
```

### Optimizations

```python
# 1. Use appropriate data structures
# O(n) list lookup vs O(1) set/dict lookup
big_list = list(range(1_000_000))
big_set  = set(range(1_000_000))

import timeit
print(timeit.timeit("999999 in big_list", globals=globals(), number=100))   # ~3s
print(timeit.timeit("999999 in big_set",  globals=globals(), number=100))   # ~0.00001s

# 2. Avoid repeated attribute lookups
import math
# Slow
for _ in range(1000):
    y = math.sqrt(2)

# Fast — cache method reference
sqrt = math.sqrt
for _ in range(1000):
    y = sqrt(2)

# 3. String concatenation
# Slow
result = ""
for i in range(10000):
    result += str(i)   # creates new string each time!

# Fast
result = "".join(str(i) for i in range(10000))

# 4. List comprehension vs loop
# Fast
squares = [x**2 for x in range(10000)]

# Slower
squares = []
for x in range(10000):
    squares.append(x**2)

# 5. NumPy for numerical operations (see NumPy docs)
import numpy as np
arr = np.arange(1_000_000)
result = np.sum(arr**2)   # ~100x faster than Python loop
```

### `functools.lru_cache`

```python
from functools import lru_cache, cache

@lru_cache(maxsize=128)   # LRU cache with max 128 entries
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

@cache  # Python 3.9+ — unbounded cache (equivalent to lru_cache(maxsize=None))
def expensive_function(x, y):
    # complex computation
    return x ** y

print(fibonacci(50))         # Fast! (cached)
print(fibonacci.cache_info())  # CacheInfo(hits=48, misses=51, maxsize=128, currsize=51)
fibonacci.cache_clear()       # clear cache
```

---

## Introspection and Reflection

```python
class MyClass:
    class_var = "hello"

    def __init__(self, x):
        self.x = x

    def method(self):
        return self.x

    @classmethod
    def class_method(cls):
        return cls.class_var

    @staticmethod
    def static_method():
        return "static"


obj = MyClass(42)

# Type info
print(type(obj))                # <class '__main__.MyClass'>
print(type(obj).__name__)       # MyClass
print(obj.__class__)            # <class '__main__.MyClass'>
print(obj.__class__.__name__)   # MyClass

# Attributes
print(dir(obj))                 # all attributes and methods
print(vars(obj))                # instance __dict__
print(vars(MyClass))            # class __dict__

# Inspect attributes
print(hasattr(obj, "x"))        # True
print(hasattr(obj, "missing"))  # False
print(getattr(obj, "x"))        # 42
print(getattr(obj, "missing", "default"))  # "default"
setattr(obj, "y", 100)
delattr(obj, "y")

# Method inspection
print(callable(obj.method))         # True
print(callable(obj.x))             # False

# MRO
print(MyClass.__mro__)

# Inspect module
import inspect

print(inspect.isclass(MyClass))        # True
print(inspect.ismethod(obj.method))   # True
print(inspect.isfunction(MyClass.method))  # True

# Source code
print(inspect.getsource(MyClass.method))

# Signature
sig = inspect.signature(MyClass.method)
print(sig.parameters)

# All methods of a class
methods = inspect.getmembers(MyClass, predicate=inspect.isfunction)
print([name for name, _ in methods])
```

---

## Functional Programming

### Immutability and Pure Functions

```python
# Pure function — same input always gives same output, no side effects
def pure_add(a, b):
    return a + b   # no mutation, no I/O, deterministic

# Impure — has side effects
count = 0
def impure_increment():
    global count
    count += 1   # mutates global state

# Prefer immutable data
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def translate(self, dx, dy):
        return Point(self.x + dx, self.y + dy)  # returns new, doesn't mutate

p1 = Point(1, 2)
p2 = p1.translate(3, 4)   # p1 unchanged
```

### `functools` for Functional Programming

```python
from functools import reduce, partial, compose

# reduce — fold a sequence
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda acc, x: acc * x, numbers, 1)  # 120
maximum = reduce(lambda a, b: a if a > b else b, numbers)  # 5

# partial — partially apply a function
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube   = partial(power, exponent=3)
print([square(i) for i in range(5)])   # [0, 1, 4, 9, 16]

# Compose functions (no built-in, easy to implement)
def compose(*funcs):
    def composed(x):
        for f in reversed(funcs):
            x = f(x)
        return x
    return composed

double    = lambda x: x * 2
add_one   = lambda x: x + 1
to_string = lambda x: str(x)

pipeline = compose(to_string, double, add_one)
print(pipeline(5))   # "12"  (5+1=6, 6*2=12, str(12)="12")
```

### `operator` Module

```python
from operator import add, sub, mul, truediv, neg, abs
from operator import eq, ne, lt, le, gt, ge
from operator import getitem, itemgetter, attrgetter, methodcaller
from functools import reduce

# Use operators as functions (faster than lambda)
numbers = [1, 2, 3, 4, 5]
total = reduce(add, numbers)    # 15 — faster than reduce(lambda a,b: a+b, ...)

# itemgetter — access by key/index (faster than lambda)
people = [{"name": "Bob", "age": 25}, {"name": "Alice", "age": 30}]
by_name = sorted(people, key=itemgetter("name"))

# attrgetter — access attribute
class Student:
    def __init__(self, name, gpa):
        self.name = name
        self.gpa  = gpa

students = [Student("Alice", 3.9), Student("Bob", 3.5)]
by_gpa = sorted(students, key=attrgetter("gpa"), reverse=True)

# methodcaller — call method
words = ["hello", "WORLD", "Python"]
upper_words = list(map(methodcaller("upper"), words))
# ['HELLO', 'WORLD', 'PYTHON']
```
