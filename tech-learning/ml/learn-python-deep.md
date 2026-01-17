# Deep Python Learning: Advanced Concepts and Best Practices

## Table of Contents
1. [Python Fundamentals Review](#python-fundamentals-review)
2. [Advanced Data Structures](#advanced-data-structures)
3. [Object-Oriented Programming](#object-oriented-programming)
4. [Functional Programming](#functional-programming)
5. [Decorators and Metaclasses](#decorators-and-metaclasses)
6. [Generators and Iterators](#generators-and-iterators)
7. [Context Managers](#context-managers)
8. [Concurrency and Parallelism](#concurrency-and-parallelism)
9. [Memory Management](#memory-management)
10. [Performance Optimization](#performance-optimization)
11. [Design Patterns](#design-patterns)
12. [Testing and Debugging](#testing-and-debugging)
13. [Best Practices](#best-practices)

---

## Python Fundamentals Review

### Pythonic Code

```python
# List Comprehensions
# Instead of:
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x ** 2)

# Use:
result = [x ** 2 for x in range(10) if x % 2 == 0]

# Dictionary Comprehensions
squares = {x: x ** 2 for x in range(10)}
filtered = {k: v for k, v in squares.items() if v > 10}

# Set Comprehensions
unique_squares = {x ** 2 for x in range(10)}

# Generator Expressions
gen = (x ** 2 for x in range(10) if x % 2 == 0)
```

### Advanced Slicing

```python
# Basic slicing
lst = list(range(10))
print(lst[2:7])      # [2, 3, 4, 5, 6]
print(lst[::2])      # [0, 2, 4, 6, 8] - every 2nd element
print(lst[::-1])     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] - reverse

# Slice objects
s = slice(2, 7, 2)
print(lst[s])        # [2, 4, 6]

# Multi-dimensional slicing
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[1:, :2])   # [[4, 5], [7, 8]]
```

### Unpacking

```python
# Tuple unpacking
a, b, c = (1, 2, 3)
a, *rest = (1, 2, 3, 4, 5)  # a=1, rest=[2, 3, 4, 5]
first, *middle, last = range(10)

# Dictionary unpacking
d1 = {'a': 1, 'b': 2}
d2 = {'c': 3, 'd': 4}
merged = {**d1, **d2}  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# Function arguments
def func(a, b, c, *args, **kwargs):
    print(a, b, c)
    print(args)   # tuple
    print(kwargs) # dict

func(1, 2, 3, 4, 5, x=10, y=20)
```

---

## Advanced Data Structures

### Collections Module

```python
from collections import defaultdict, Counter, deque, namedtuple, OrderedDict

# defaultdict - no KeyError
dd = defaultdict(list)
dd['key'].append('value')

# Counter - count occurrences
counter = Counter(['a', 'b', 'a', 'c', 'b', 'a'])
print(counter)  # Counter({'a': 3, 'b': 2, 'c': 1})
print(counter.most_common(2))  # [('a', 3), ('b', 2)]

# deque - efficient append/pop from both ends
dq = deque([1, 2, 3])
dq.appendleft(0)  # [0, 1, 2, 3]
dq.pop()           # [0, 1, 2]

# namedtuple - tuple with named fields
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)  # 1 2

# OrderedDict - remembers insertion order (Python 3.7+ dicts do this)
od = OrderedDict()
od['first'] = 1
od['second'] = 2
```

### Custom Data Structures

```python
class TreeNode:
    """Binary tree node"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class LinkedList:
    """Linked list implementation"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"LinkedList({self.val})"
```

---

## Object-Oriented Programming

### Advanced Classes

```python
class Person:
    """Example class with properties and methods"""
    
    # Class variable
    species = "Homo sapiens"
    
    def __init__(self, name, age):
        # Instance variables
        self._name = name  # Protected
        self.__age = age   # Private (name mangling)
    
    @property
    def name(self):
        """Getter for name"""
        return self._name
    
    @name.setter
    def name(self, value):
        """Setter for name"""
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._name = value
    
    @property
    def age(self):
        return self.__age
    
    def __str__(self):
        return f"Person(name={self._name}, age={self.__age})"
    
    def __repr__(self):
        return f"Person('{self._name}', {self.__age})"
    
    def __eq__(self, other):
        if not isinstance(other, Person):
            return False
        return self._name == other._name and self.__age == other.__age
    
    def __hash__(self):
        return hash((self._name, self.__age))
```

### Inheritance and Polymorphism

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement")
    
    def __str__(self):
        return f"{self.name} says {self.speak()}"

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal)  # Calls speak() method
```

### Multiple Inheritance and MRO

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

# Method Resolution Order (MRO)
print(D.__mro__)  # D -> B -> C -> A -> object
d = D()
print(d.method())  # "B" (first in MRO)
```

### Abstract Base Classes

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# Can't instantiate Shape directly
# shape = Shape()  # TypeError
rect = Rectangle(5, 3)  # OK
```

### Magic Methods

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __len__(self):
        return int((self.x ** 2 + self.y ** 2) ** 0.5)
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")
    
    def __contains__(self, value):
        return value in (self.x, self.y)
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def __call__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # Uses __add__
print(v3())   # Uses __call__
```

---

## Functional Programming

### Higher-Order Functions

```python
# Map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))

# Filter
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Reduce
from functools import reduce
product = reduce(lambda x, y: x * y, numbers)

# Sorted with key
students = [('Alice', 25), ('Bob', 20), ('Charlie', 22)]
sorted_by_age = sorted(students, key=lambda x: x[1])
```

### Lambda Functions

```python
# Simple lambda
add = lambda x, y: x + y
print(add(2, 3))  # 5

# Lambda with map
result = list(map(lambda x: x * 2, [1, 2, 3]))

# Lambda with filter
evens = list(filter(lambda x: x % 2 == 0, range(10)))

# Lambda with sorted
points = [(1, 2), (3, 1), (2, 3)]
sorted_points = sorted(points, key=lambda p: p[1])
```

### Partial Functions

```python
from functools import partial

def multiply(x, y):
    return x * y

# Create partial function
double = partial(multiply, 2)
print(double(5))  # 10

# With keyword arguments
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
print(square(5))  # 25
```

### Closures

```python
def outer_function(x):
    """Outer function that returns inner function"""
    def inner_function(y):
        return x + y
    return inner_function

add_five = outer_function(5)
print(add_five(3))  # 8

# Counter using closure
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c1 = make_counter()
print(c1())  # 1
print(c1())  # 2
```

---

## Decorators and Metaclasses

### Function Decorators

```python
def timer(func):
    """Decorator to measure function execution time"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Using decorator
slow_function()

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
```

### Class Decorators

```python
def singleton(cls):
    """Singleton decorator"""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Database connection created")

db1 = Database()  # Creates connection
db2 = Database()  # Returns same instance
```

### Property Decorators

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.fahrenheit)  # 77.0
temp.fahrenheit = 100
print(temp.celsius)  # 37.78
```

### Metaclasses

```python
class Meta(type):
    """Metaclass example"""
    def __new__(cls, name, bases, dct):
        # Add method to all classes using this metaclass
        dct['get_class_name'] = lambda self: self.__class__.__name__
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=Meta):
    pass

obj = MyClass()
print(obj.get_class_name())  # "MyClass"
```

---

## Generators and Iterators

### Generators

```python
def fibonacci(n):
    """Generator for Fibonacci sequence"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Use generator
for num in fibonacci(10):
    print(num)

# Generator expression
squares = (x ** 2 for x in range(10))
print(list(squares))

# Generator with send
def counter():
    count = 0
    while True:
        value = yield count
        if value is not None:
            count = value
        else:
            count += 1

c = counter()
next(c)  # Initialize
print(c.send(None))  # 1
print(c.send(10))    # 10
print(c.send(None))  # 11
```

### Custom Iterators

```python
class CountDown:
    """Custom iterator"""
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Use iterator
for num in CountDown(5):
    print(num)  # 5, 4, 3, 2, 1
```

---

## Context Managers

### Using Context Managers

```python
# File handling
with open('file.txt', 'r') as f:
    content = f.read()
# File automatically closed

# Multiple context managers
with open('input.txt', 'r') as infile, open('output.txt', 'w') as outfile:
    data = infile.read()
    outfile.write(data.upper())
```

### Custom Context Managers

```python
# Class-based
class Timer:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        print(f"{self.name} took {elapsed:.4f} seconds")
        return False  # Don't suppress exceptions

with Timer("Operation"):
    time.sleep(1)

# Function-based with contextlib
from contextlib import contextmanager

@contextmanager
def timer(name):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.4f} seconds")

with timer("Operation"):
    time.sleep(1)
```

---

## Concurrency and Parallelism

### Threading

```python
import threading
import time

def worker(name, delay):
    print(f"Worker {name} starting")
    time.sleep(delay)
    print(f"Worker {name} finished")

# Create threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i, 1))
    threads.append(t)
    t.start()

# Wait for all threads
for t in threads:
    t.join()

# Thread-safe operations
import queue
q = queue.Queue()
q.put(1)
q.put(2)
print(q.get())  # 1
```

### Multiprocessing

```python
from multiprocessing import Process, Pool
import os

def worker(name):
    print(f"Worker {name} (PID: {os.getpid()})")

# Process-based
processes = []
for i in range(3):
    p = Process(target=worker, args=(i,))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# Pool-based
def square(x):
    return x ** 2

with Pool(processes=4) as pool:
    results = pool.map(square, range(10))
print(results)
```

### Async/Await

```python
import asyncio

async def fetch_data(url):
    """Async function"""
    await asyncio.sleep(1)  # Simulate I/O
    return f"Data from {url}"

async def main():
    # Run concurrently
    results = await asyncio.gather(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3")
    )
    return results

# Run async function
results = asyncio.run(main())
```

---

## Memory Management

### Memory Profiling

```python
import sys
from memory_profiler import profile

# Check object size
x = [1, 2, 3, 4, 5]
print(sys.getsizeof(x))

# Memory profiler
@profile
def my_function():
    large_list = list(range(1000000))
    return sum(large_list)

my_function()
```

### Garbage Collection

```python
import gc

# Force garbage collection
gc.collect()

# Get garbage collection stats
print(gc.get_stats())

# Disable/enable GC
gc.disable()
# ... do something ...
gc.enable()
```

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

# Profile function
profiler = cProfile.Profile()
profiler.enable()
slow_function()
profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# First call computes
print(fibonacci(30))
# Second call uses cache
print(fibonacci(30))
```

### Cython and Numba

```python
# Numba JIT compilation
from numba import jit

@jit(nopython=True)
def fast_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Much faster for large arrays
import numpy as np
arr = np.arange(1000000)
result = fast_sum(arr)
```

---

## Design Patterns

### Singleton Pattern

```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

### Factory Pattern

```python
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

animal = AnimalFactory.create_animal("dog")
```

### Observer Pattern

```python
class Observer:
    def update(self, message):
        raise NotImplementedError

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def notify(self, message):
        for observer in self._observers:
            observer.update(message)
```

---

## Testing and Debugging

### Unit Testing

```python
import unittest

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(2 + 2, 4)
    
    def test_multiply(self):
        self.assertEqual(3 * 4, 12)
    
    def setUp(self):
        """Run before each test"""
        self.data = [1, 2, 3]
    
    def tearDown(self):
        """Run after each test"""
        pass

if __name__ == '__main__':
    unittest.main()
```

### pytest

```python
# test_example.py
import pytest

def test_addition():
    assert 2 + 2 == 4

@pytest.fixture
def sample_data():
    return [1, 2, 3]

def test_with_fixture(sample_data):
    assert len(sample_data) == 3

# Run: pytest test_example.py
```

### Debugging

```python
import pdb

def buggy_function(x, y):
    result = x + y
    pdb.set_trace()  # Breakpoint
    return result * 2

# Or use breakpoint() in Python 3.7+
def buggy_function_v2(x, y):
    result = x + y
    breakpoint()  # Enters debugger
    return result * 2
```

---

## Best Practices

### Code Style

```python
# PEP 8 compliance
# Use meaningful names
def calculate_total_price(items):
    return sum(item.price for item in items)

# Type hints
from typing import List, Dict, Optional

def process_data(data: List[Dict[str, int]]) -> Optional[Dict]:
    if not data:
        return None
    return {"count": len(data)}

# Docstrings
def complex_function(param1: int, param2: str) -> bool:
    """
    Brief description.
    
    Longer description explaining what the function does,
    its parameters, and return value.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### Error Handling

```python
# Specific exceptions
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Division by zero: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    print("Cleanup code")

# Custom exceptions
class CustomError(Exception):
    """Custom exception class"""
    pass

raise CustomError("Something went wrong")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

---

## Resources

- **Books**: 
  - "Fluent Python" by Luciano Ramalho
  - "Effective Python" by Brett Slatkin
  - "Python Tricks" by Dan Bader
- **Documentation**: 
  - Python.org official docs
  - Real Python tutorials
- **Practice**: 
  - LeetCode
  - HackerRank
  - Codewars

---

## Conclusion

Deep Python knowledge enables writing efficient, maintainable, and Pythonic code. Key takeaways:

1. **Master Fundamentals**: Data structures, OOP, functions
2. **Use Advanced Features**: Decorators, generators, context managers
3. **Optimize Performance**: Profiling, caching, parallel processing
4. **Follow Best Practices**: PEP 8, testing, documentation
5. **Keep Learning**: Python ecosystem evolves constantly

Remember: Pythonic code is readable, maintainable, and efficient!

