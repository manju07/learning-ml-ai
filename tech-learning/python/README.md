# Python Learning Guide

A comprehensive documentation reference for Python and its most popular libraries and frameworks. Each document includes detailed explanations and practical examples.

---

## Core Python

| Document | Topics Covered |
|----------|---------------|
| [01 — Basics](./01-basics.md) | Variables, data types, operators, strings, numbers, booleans, type conversion, I/O, comments |
| [02 — Data Structures](./02-data-structures.md) | Lists, tuples, dictionaries, sets, `collections` module, comprehensions |
| [03 — Control Flow](./03-control-flow.md) | if/elif/else, for/while loops, break/continue/pass, `match/case` (Python 3.10+), `itertools` |
| [04 — Functions](./04-functions.md) | Defining functions, *args/**kwargs, lambdas, higher-order functions, closures, decorators, generators, type hints |
| [05 — OOP](./05-oop.md) | Classes, inheritance, multiple inheritance (MRO), dunder methods, properties, abstract classes, dataclasses, metaclasses, protocols |
| [06 — Modules & Packages](./06-modules-packages.md) | Imports, packages, `__init__.py`, virtual environments, pip, pyproject.toml, standard library |
| [07 — File I/O](./07-file-io.md) | Reading/writing files, CSV, JSON, pickle, pathlib, temp files, filesystem ops |
| [08 — Concurrency](./08-concurrency.md) | Threading, multiprocessing, asyncio (async/await), `concurrent.futures`, locks, queues |
| [09 — Advanced](./09-advanced.md) | Iterator protocol, context managers, descriptors, generic types, memory management, profiling, introspection, functional programming |

---

## Libraries

| Document | Library | Use Case |
|----------|---------|----------|
| [NumPy](./libs/numpy.md) | `numpy` | Numerical arrays, math, linear algebra |
| [Pandas](./libs/pandas.md) | `pandas` | Data analysis, DataFrames, time series |
| [Matplotlib & Seaborn](./libs/matplotlib-seaborn.md) | `matplotlib`, `seaborn` | Data visualization, plots, charts |
| [Requests](./libs/requests.md) | `requests`, `httpx` | HTTP client, REST API calls |
| [SQLAlchemy](./libs/sqlalchemy.md) | `sqlalchemy` | SQL ORM and Core, database access |
| [Pydantic](./libs/pydantic.md) | `pydantic` | Data validation, settings management |

---

## Frameworks

| Document | Framework | Use Case |
|----------|-----------|----------|
| [FastAPI](./frameworks/fastapi.md) | `fastapi` | Modern async REST APIs, auto-docs |
| [Flask](./frameworks/flask.md) | `flask` | Lightweight web apps and APIs |
| [Pytest](./frameworks/pytest.md) | `pytest` | Testing, fixtures, mocking, coverage |

---

## Quick Reference

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Install common libraries
pip install numpy pandas matplotlib seaborn
pip install requests httpx
pip install sqlalchemy pydantic
pip install fastapi uvicorn[standard]
pip install flask flask-sqlalchemy
pip install pytest pytest-cov pytest-mock
```

### Key Concepts by Topic

#### Variables & Types
```python
x = 42           # int
y = 3.14         # float
z = "hello"      # str
b = True         # bool
n = None         # NoneType

type(x)          # <class 'int'>
isinstance(x, int)  # True
```

#### Collections
```python
lst  = [1, 2, 3]         # list — ordered, mutable
tup  = (1, 2, 3)         # tuple — ordered, immutable
dct  = {"a": 1}          # dict — key-value
s    = {1, 2, 3}         # set — unique, unordered
```

#### Functions
```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Lambda
square = lambda x: x**2

# Decorator
def log(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
```

#### Classes
```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name}: Woof!"
```

#### Async
```python
import asyncio

async def fetch(url):
    await asyncio.sleep(1)
    return f"Data from {url}"

async def main():
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
    )

asyncio.run(main())
```

---

## Learning Path

### Beginner
1. [01 — Basics](./01-basics.md)
2. [02 — Data Structures](./02-data-structures.md)
3. [03 — Control Flow](./03-control-flow.md)
4. [04 — Functions](./04-functions.md)

### Intermediate
5. [05 — OOP](./05-oop.md)
6. [06 — Modules & Packages](./06-modules-packages.md)
7. [07 — File I/O](./07-file-io.md)
8. [libs/NumPy](./libs/numpy.md)
9. [libs/Pandas](./libs/pandas.md)

### Advanced
10. [08 — Concurrency](./08-concurrency.md)
11. [09 — Advanced](./09-advanced.md)
12. [libs/SQLAlchemy](./libs/sqlalchemy.md)
13. [libs/Pydantic](./libs/pydantic.md)
14. [frameworks/FastAPI](./frameworks/fastapi.md)

### Data Science Track
1. [01 — Basics](./01-basics.md)
2. [02 — Data Structures](./02-data-structures.md)
3. [libs/NumPy](./libs/numpy.md)
4. [libs/Pandas](./libs/pandas.md)
5. [libs/Matplotlib & Seaborn](./libs/matplotlib-seaborn.md)

### Web Development Track
1. Core Python (01–05)
2. [libs/Pydantic](./libs/pydantic.md)
3. [libs/SQLAlchemy](./libs/sqlalchemy.md)
4. [frameworks/FastAPI](./frameworks/fastapi.md) or [frameworks/Flask](./frameworks/flask.md)
5. [libs/Requests](./libs/requests.md)
6. [frameworks/Pytest](./frameworks/pytest.md)

---

## Python Version Reference

| Feature | Min Version |
|---------|-------------|
| f-strings | Python 3.6+ |
| `dataclasses` | Python 3.7+ |
| `asyncio` stable | Python 3.7+ |
| Walrus operator `:=` | Python 3.8+ |
| Positional-only params `/` | Python 3.8+ |
| `match / case` | Python 3.10+ |
| Union type `X \| Y` | Python 3.10+ |
| `tomllib` (built-in) | Python 3.11+ |
| Lazy f-string evaluation | Python 3.12+ |

---

## Useful Resources

- [Official Python Docs](https://docs.python.org/3/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Cookbook (O'Reilly)](https://www.oreilly.com/library/view/python-cookbook-3rd/9781449357337/)
- [Fluent Python (O'Reilly)](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
