# Python Basics

## Table of Contents
- [Introduction](#introduction)
- [Variables and Data Types](#variables-and-data-types)
- [Operators](#operators)
- [Strings](#strings)
- [Numbers](#numbers)
- [Booleans](#booleans)
- [Type Conversion](#type-conversion)
- [Input and Output](#input-and-output)
- [Comments](#comments)

---

## Introduction

Python is a high-level, interpreted, dynamically typed programming language known for its clean syntax and readability. It follows the principle of *"readability counts"* — code should be easy to read and write.

```python
print("Hello, World!")
```

---

## Variables and Data Types

Python is **dynamically typed** — you don't declare the type; Python infers it at runtime.

### Basic Types

| Type      | Example           | Description                 |
|-----------|-------------------|-----------------------------|
| `int`     | `42`              | Integer (unlimited size)    |
| `float`   | `3.14`            | Floating-point number       |
| `complex` | `2 + 3j`          | Complex number              |
| `str`     | `"hello"`         | String (immutable sequence) |
| `bool`    | `True` / `False`  | Boolean                     |
| `NoneType`| `None`            | Absence of value            |

### Variable Assignment

```python
# Single assignment
name = "Alice"
age = 30
height = 5.7
is_active = True
nothing = None

# Multiple assignment
x = y = z = 0

# Tuple unpacking
a, b, c = 1, 2, 3
first, *rest = [1, 2, 3, 4, 5]  # first=1, rest=[2,3,4,5]

# Swap without temp variable
a, b = b, a
```

### Checking Types

```python
x = 42
print(type(x))          # <class 'int'>
print(isinstance(x, int))  # True
print(isinstance(x, (int, float)))  # True — checks multiple types
```

---

## Operators

### Arithmetic Operators

```python
a, b = 10, 3

print(a + b)   # 13  — addition
print(a - b)   # 7   — subtraction
print(a * b)   # 30  — multiplication
print(a / b)   # 3.333...  — true division (always float)
print(a // b)  # 3   — floor division (integer result)
print(a % b)   # 1   — modulo (remainder)
print(a ** b)  # 1000 — exponentiation
```

### Comparison Operators

```python
print(5 == 5)   # True   — equal
print(5 != 4)   # True   — not equal
print(5 > 3)    # True   — greater than
print(5 < 3)    # False  — less than
print(5 >= 5)   # True   — greater than or equal
print(5 <= 4)   # False  — less than or equal

# Identity operators
x = [1, 2]
y = x
z = [1, 2]
print(x is y)   # True  — same object
print(x is z)   # False — different objects, same value
print(x == z)   # True  — same value

# Membership operators
print(2 in [1, 2, 3])    # True
print(4 not in [1, 2, 3]) # True
```

### Logical Operators

```python
print(True and False)   # False
print(True or False)    # True
print(not True)         # False

# Short-circuit evaluation
x = 0
result = x or "default"  # "default" — x is falsy, so evaluates right
result = x and "value"   # 0 — x is falsy, short-circuits
```

### Bitwise Operators

```python
a, b = 0b1010, 0b1100   # 10, 12 in binary

print(a & b)   # 8  — AND
print(a | b)   # 14 — OR
print(a ^ b)   # 6  — XOR
print(~a)      # -11 — NOT (bitwise complement)
print(a << 1)  # 20  — left shift
print(a >> 1)  # 5   — right shift
```

### Assignment Operators

```python
x = 10
x += 5   # x = x + 5  → 15
x -= 3   # x = x - 3  → 12
x *= 2   # x = x * 2  → 24
x //= 5  # x = x // 5 → 4
x **= 2  # x = x ** 2 → 16
x %= 5   # x = x % 5  → 1
```

### Walrus Operator `:=` (Python 3.8+)

```python
# Assign and test in a single expression
import re

if m := re.search(r"\d+", "answer is 42"):
    print(m.group())  # 42

# In while loops
data = [1, 2, 3, 4, 5]
while chunk := data[:2]:
    print(chunk)
    data = data[2:]
```

---

## Strings

Strings are **immutable** sequences of Unicode characters.

### Creating Strings

```python
single  = 'hello'
double  = "world"
multi   = """This is
a multi-line string"""
raw     = r"C:\Users\name"    # raw — backslashes not treated as escapes
byte    = b"bytes"             # bytes literal
f_str   = f"Hello, {single}"  # f-string (Python 3.6+)
```

### String Operations

```python
s = "Python Programming"

# Indexing and slicing
print(s[0])       # 'P'
print(s[-1])      # 'g'
print(s[0:6])     # 'Python'
print(s[::2])     # every second char
print(s[::-1])    # reversed: 'gnimmargorP nohtyP'

# Length
print(len(s))     # 18

# Concatenation and repetition
print("Hello" + " " + "World")
print("ha" * 3)   # 'hahaha'
```

### Common String Methods

```python
s = "  Hello, World!  "

print(s.strip())          # 'Hello, World!' — removes leading/trailing whitespace
print(s.lstrip())         # 'Hello, World!  '
print(s.rstrip())         # '  Hello, World!'
print(s.lower())          # '  hello, world!  '
print(s.upper())          # '  HELLO, WORLD!  '
print(s.title())          # '  Hello, World!  '
print(s.replace(",", ";")) # '  Hello; World!  '
print(s.find("World"))    # 9 — index of first occurrence, -1 if not found
print(s.count("l"))       # 3
print(s.startswith("  H")) # True
print(s.endswith("!  "))  # True
print(s.split(","))       # ['  Hello', ' World!  ']
print(",".join(["a","b","c"]))  # 'a,b,c'
print(s.center(30, "*"))  # pads with * to width 30
print(s.isdigit())        # False
print("123".isdigit())    # True
print(s.isalpha())        # False
```

### F-Strings (Formatted String Literals)

```python
name = "Alice"
age = 30
pi = 3.14159

# Basic
print(f"Name: {name}, Age: {age}")

# Expressions
print(f"Next year: {age + 1}")

# Format specifiers
print(f"Pi: {pi:.2f}")             # Pi: 3.14
print(f"Large: {1000000:,}")       # Large: 1,000,000
print(f"Hex: {255:#x}")            # Hex: 0xff
print(f"Percent: {0.875:.1%}")     # Percent: 87.5%
print(f"{'center':^20}")           # centered in 20 chars

# Debug (Python 3.8+)
x = 42
print(f"{x=}")  # x=42
```

### String Formatting (old styles)

```python
# % formatting
print("Hello, %s! You are %d years old." % ("Alice", 30))

# str.format()
print("Hello, {}! You are {} years old.".format("Alice", 30))
print("Hello, {name}!".format(name="Alice"))
```

---

## Numbers

### Integer

```python
# Python integers have unlimited precision
big = 10 ** 100   # googol — no overflow!
print(big)

# Different bases
binary  = 0b1010   # 10
octal   = 0o17     # 15
hex_num = 0xFF     # 255

# Underscores for readability
million = 1_000_000
```

### Float

```python
x = 3.14
y = 1.5e-3    # scientific notation: 0.0015
z = float('inf')  # positive infinity
n = float('nan')  # not a number

# Floating-point precision gotcha
print(0.1 + 0.2)        # 0.30000000000000004
print(0.1 + 0.2 == 0.3) # False!

# Use decimal for precise arithmetic
from decimal import Decimal
print(Decimal('0.1') + Decimal('0.2'))  # 0.3
```

### Complex

```python
c = 2 + 3j
print(c.real)   # 2.0
print(c.imag)   # 3.0
print(abs(c))   # 3.605... — magnitude
print(c.conjugate())  # (2-3j)
```

### Math Module

```python
import math

print(math.pi)          # 3.141592653589793
print(math.e)           # 2.718281828459045
print(math.sqrt(16))    # 4.0
print(math.ceil(4.2))   # 5
print(math.floor(4.9))  # 4
print(math.factorial(5)) # 120
print(math.log(100, 10)) # 2.0
print(math.sin(math.pi / 2))  # 1.0
print(math.gcd(12, 18))  # 6
```

---

## Booleans

```python
# Boolean values
t = True
f = False

# Falsy values — evaluate to False in boolean context
falsy_values = [False, 0, 0.0, 0j, "", [], {}, set(), tuple(), None]

for val in falsy_values:
    print(f"{val!r} is falsy: {not bool(val)}")

# Truthy — everything else
print(bool(42))    # True
print(bool("hi"))  # True
print(bool([0]))   # True — non-empty list
```

---

## Type Conversion

```python
# Implicit conversion
result = 10 + 3.5   # int + float → float (3.5 is widened)

# Explicit conversion (casting)
print(int(3.9))     # 3   — truncates, does NOT round
print(int("42"))    # 42
print(float(5))     # 5.0
print(str(100))     # '100'
print(bool(0))      # False
print(list("abc"))  # ['a', 'b', 'c']
print(tuple([1,2])) # (1, 2)
print(set([1,1,2])) # {1, 2}

# Safe conversion with error handling
try:
    n = int("not a number")
except ValueError as e:
    print(f"Conversion error: {e}")
```

---

## Input and Output

```python
# Output
print("Hello")
print("a", "b", "c", sep="-")   # a-b-c
print("line1", end="")           # no newline
print("line2")                   # line1line2

# Input — always returns a string
name = input("Enter your name: ")
age  = int(input("Enter your age: "))  # cast manually

# Print to stderr
import sys
print("Error message", file=sys.stderr)
```

---

## Comments

```python
# This is a single-line comment

"""
This is a docstring — triple-quoted string used as
a multi-line comment or documentation string.
"""

def add(a, b):
    """
    Return the sum of a and b.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.

    Example:
        >>> add(2, 3)
        5
    """
    return a + b

# Access docstring
print(add.__doc__)
```

---

## Best Practices

- Use meaningful variable names: `user_age` over `ua`
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use `snake_case` for variables and functions, `PascalCase` for classes, `UPPER_CASE` for constants
- Prefer f-strings over `%` or `.format()` for string formatting
- Use `is` / `is not` for `None` comparisons, not `==`

```python
# Good
if value is None:
    ...

# Avoid
if value == None:
    ...
```
