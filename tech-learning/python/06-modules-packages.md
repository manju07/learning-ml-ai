# Python Modules and Packages

## Table of Contents
- [Modules](#modules)
- [Import Styles](#import-styles)
- [Packages](#packages)
- [The `__init__.py` File](#the-__init__py-file)
- [Relative Imports](#relative-imports)
- [The Module Search Path](#the-module-search-path)
- [Virtual Environments](#virtual-environments)
- [Package Management with pip](#package-management-with-pip)
- [Important Standard Library Modules](#important-standard-library-modules)

---

## Modules

A **module** is any `.py` file. It can contain functions, classes, and variables.

```
my_math.py
```

```python
# my_math.py

PI = 3.14159

def circle_area(radius):
    return PI * radius ** 2

def circle_perimeter(radius):
    return 2 * PI * radius

class Vector2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def magnitude(self):
        return (self.x**2 + self.y**2) ** 0.5
```

### `__name__` and the Main Guard

```python
# my_math.py

def main():
    print(circle_area(5))

if __name__ == "__main__":
    # Only runs when script is executed directly
    # NOT when imported as a module
    main()
```

```bash
python my_math.py       # __name__ == "__main__" → main() runs
python -c "import my_math"  # __name__ == "my_math" → main() does NOT run
```

---

## Import Styles

```python
# 1. Import the whole module
import my_math
print(my_math.PI)
print(my_math.circle_area(5))
v = my_math.Vector2D(3, 4)

# 2. Import specific names
from my_math import PI, circle_area, Vector2D
print(PI)
print(circle_area(5))

# 3. Import with alias
import my_math as mm
from my_math import circle_area as area

# 4. Import all (use sparingly — pollutes namespace)
from my_math import *   # imports names not starting with _

# Check what was imported
print(dir(my_math))           # all attributes
print(my_math.__file__)       # path to module file
print(my_math.__doc__)        # module docstring
```

### Import Best Practices

```python
# PEP 8 order:
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Optional

# 2. Third-party
import numpy as np
import pandas as pd
import requests

# 3. Local application
from my_package import my_module
from .utils import helper_function

# Avoid circular imports — use local imports if needed
def some_function():
    from my_other_module import something  # import inside function
    return something()
```

---

## Packages

A **package** is a directory containing an `__init__.py` file (and possibly subpackages).

```
my_project/
├── main.py
└── geometry/
    ├── __init__.py
    ├── shapes/
    │   ├── __init__.py
    │   ├── circle.py
    │   └── rectangle.py
    └── utils/
        ├── __init__.py
        └── math_helpers.py
```

```python
# geometry/shapes/circle.py
import math

class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius
```

```python
# main.py
from geometry.shapes.circle import Circle
from geometry.shapes import rectangle
from geometry import utils

c = Circle(5)
print(c.area())
```

---

## The `__init__.py` File

`__init__.py` controls what's available when the package is imported.

```python
# geometry/__init__.py

# Re-export commonly used names for convenience
from geometry.shapes.circle import Circle
from geometry.shapes.rectangle import Rectangle
from geometry.utils.math_helpers import distance

__all__ = ["Circle", "Rectangle", "distance"]  # controls `from geometry import *`

# Package metadata
__version__ = "1.0.0"
__author__ = "Alice"
```

```python
# Now users can do:
from geometry import Circle   # instead of geometry.shapes.circle
import geometry
print(geometry.__version__)   # 1.0.0
```

---

## Relative Imports

Within a package, use relative imports (`.`) instead of absolute.

```python
# geometry/shapes/circle.py
from ..utils.math_helpers import distance   # .. = parent package (geometry)
from . import rectangle                      # . = current package (shapes)
from .rectangle import Rectangle            # specific name from sibling
```

```
. = current package
.. = parent package
... = grandparent package
```

---

## The Module Search Path

When you `import foo`, Python searches in this order:

1. `sys.modules` cache (already imported modules)
2. Built-in modules (`sys.builtin_module_names`)
3. Directories in `sys.path`

```python
import sys

print(sys.path)
# ['', '/usr/lib/python312', '/usr/lib/python312/lib-dynload', ...]

# '' = current directory
# PYTHONPATH environment variable is also added

# Manipulate sys.path at runtime
sys.path.insert(0, "/path/to/my/modules")
import my_custom_module   # now findable

# Or set PYTHONPATH env variable:
# export PYTHONPATH="/path/to/my/modules:$PYTHONPATH"
```

---

## Virtual Environments

Virtual environments isolate project dependencies.

```bash
# Create
python -m venv venv           # creates ./venv/

# Activate
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Deactivate
deactivate

# Verify
which python                   # should point to venv
python --version

# Modern alternative: uv (extremely fast)
pip install uv
uv venv
source .venv/bin/activate
```

---

## Package Management with pip

```bash
# Install packages
pip install requests
pip install "requests>=2.28,<3"        # version constraint
pip install "django[rest]"              # with extras
pip install -e .                        # editable install (development)

# Install from requirements.txt
pip install -r requirements.txt

# Create requirements.txt
pip freeze > requirements.txt          # exact pinned versions
pip-compile pyproject.toml             # resolve from high-level deps

# Uninstall
pip uninstall requests

# Show info
pip show requests
pip list
pip list --outdated

# Search
pip search requests    # deprecated; use PyPI website

# Upgrade
pip install --upgrade requests
pip install --upgrade pip
```

### `pyproject.toml` (Modern Standard)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "my-package"
version = "1.0.0"
description = "A sample Python project"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "requests>=2.28",
    "pandas>=2.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7",
    "black",
    "mypy",
    "ruff",
]

[project.scripts]
my-tool = "my_package.cli:main"
```

---

## Important Standard Library Modules

### `os` — Operating System Interface

```python
import os

# Working directory
print(os.getcwd())             # current directory
os.chdir("/tmp")               # change directory

# Environment variables
home = os.environ["HOME"]
api_key = os.environ.get("API_KEY", "default")
os.environ["MY_VAR"] = "value"

# File/directory operations
os.makedirs("a/b/c", exist_ok=True)  # create nested dirs
os.rename("old.txt", "new.txt")
os.remove("file.txt")
os.rmdir("empty_dir")

# Path info
print(os.path.exists("/etc/hosts"))        # True/False
print(os.path.isfile("/etc/hosts"))        # True
print(os.path.isdir("/etc"))               # True
print(os.path.join("home", "user", "file")) # home/user/file
print(os.path.basename("/home/user/file")) # file
print(os.path.dirname("/home/user/file"))  # /home/user
print(os.path.splitext("image.png"))       # ('image', '.png')
print(os.path.abspath("./relative"))       # absolute path

# Walk directory tree
for root, dirs, files in os.walk("/etc"):
    for file in files:
        print(os.path.join(root, file))
```

### `pathlib` — Modern Path Handling (Recommended)

```python
from pathlib import Path

p = Path("/home/user/documents/report.txt")

print(p.name)        # report.txt
print(p.stem)        # report
print(p.suffix)      # .txt
print(p.parent)      # /home/user/documents
print(p.parts)       # ('/', 'home', 'user', 'documents', 'report.txt')

# Building paths with /
config = Path.home() / ".config" / "myapp" / "settings.json"

# Check and create
config.parent.mkdir(parents=True, exist_ok=True)

# Read/write
config.write_text('{"theme": "dark"}')
content = config.read_text()
data    = config.read_bytes()

# List directory
src = Path("src")
for py_file in src.rglob("*.py"):   # recursive glob
    print(py_file)

# Iterate
for item in Path(".").iterdir():
    if item.is_file():
        print(item)

# Stat
stat = p.stat()
print(stat.st_size, stat.st_mtime)
```

### `sys` — System-Specific Parameters

```python
import sys

print(sys.version)         # Python version string
print(sys.platform)        # 'linux', 'darwin', 'win32'
print(sys.argv)            # command-line arguments
print(sys.path)            # module search paths
print(sys.executable)      # path to Python interpreter

sys.exit(0)                # exit with code 0 (success)
sys.exit("Error message")  # exit with message (code 1)

# stdin / stdout / stderr
sys.stdout.write("Hello\n")
sys.stderr.write("Error\n")
data = sys.stdin.read()

# Recursion limit
sys.setrecursionlimit(2000)
```

### `json` — JSON Encoding/Decoding

```python
import json

# Dict → JSON string
data = {"name": "Alice", "age": 30, "scores": [95, 87, 92]}
json_str = json.dumps(data)                     # compact
json_str = json.dumps(data, indent=2)           # pretty
json_str = json.dumps(data, sort_keys=True)     # sorted keys

# JSON string → dict
loaded = json.loads(json_str)
print(loaded["name"])   # Alice

# File I/O
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

with open("data.json") as f:
    loaded = json.load(f)

# Custom encoder
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

data = {"created_at": datetime.now()}
print(json.dumps(data, cls=DateTimeEncoder))
```

### `datetime` — Date and Time

```python
from datetime import datetime, date, time, timedelta, timezone

# Current time
now = datetime.now()                    # local time
now_utc = datetime.now(timezone.utc)   # UTC
today = date.today()

# Create specific datetime
dt = datetime(2024, 3, 15, 10, 30, 0)

# Format and parse
formatted = dt.strftime("%Y-%m-%d %H:%M:%S")   # "2024-03-15 10:30:00"
parsed    = datetime.strptime("15/03/2024", "%d/%m/%Y")

# Arithmetic
tomorrow  = today + timedelta(days=1)
next_week = now + timedelta(weeks=1)
diff      = datetime(2025, 1, 1) - now
print(diff.days, diff.seconds)

# Timestamp
ts = now.timestamp()           # Unix timestamp (float)
dt = datetime.fromtimestamp(ts)

# ISO format (recommended)
iso = now.isoformat()
dt  = datetime.fromisoformat("2024-03-15T10:30:00")

# Timezone-aware
import pytz   # pip install pytz
ny = pytz.timezone("America/New_York")
dt_ny = datetime.now(ny)
```

### `re` — Regular Expressions

```python
import re

text = "Contact us at support@example.com or sales@company.org"

# Search — first match
m = re.search(r"\b\w+@\w+\.\w+\b", text)
if m:
    print(m.group())    # support@example.com
    print(m.start())    # start index
    print(m.end())      # end index

# Find all matches
emails = re.findall(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
# ['support@example.com', 'sales@company.org']

# Find all with groups
pattern = r"(\w+)@(\w+)\.(\w+)"
for m in re.finditer(pattern, text):
    print(m.group(0), m.group(1), m.group(2), m.group(3))

# Substitute
cleaned = re.sub(r"\s+", " ", "too   many    spaces")
# 'too many spaces'

# Split
parts = re.split(r"[,;]\s*", "one, two; three,four")
# ['one', 'two', 'three', 'four']

# Compile for reuse
email_re = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
is_email = bool(email_re.match("user@domain.com"))

# Groups
m = re.match(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})", "2024-03-15")
print(m.group("year"))   # 2024
print(m.groupdict())     # {'year': '2024', 'month': '03', 'day': '15'}
```

### `logging` — Logging

```python
import logging

# Basic config
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),   # also print to console
    ]
)

logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")

# Exception logging
try:
    1 / 0
except ZeroDivisionError:
    logger.exception("Division error!")   # includes traceback

# Level hierarchy: DEBUG < INFO < WARNING < ERROR < CRITICAL
```

### `argparse` — Command-Line Arguments

```python
import argparse

parser = argparse.ArgumentParser(description="Process some files.")
parser.add_argument("input",              help="Input file path")
parser.add_argument("-o", "--output",     help="Output file path", default="output.txt")
parser.add_argument("-n", "--count",      help="Number of items",  type=int, default=10)
parser.add_argument("-v", "--verbose",    action="store_true",     help="Verbose output")
parser.add_argument("--format",           choices=["json", "csv", "txt"], default="txt")

args = parser.parse_args()

print(args.input)
print(args.output)
print(args.count)
print(args.verbose)
print(args.format)

# python script.py data.csv -o result.json -n 20 --verbose --format json
```

### `threading` and `multiprocessing`

```python
import threading
import multiprocessing

# Threading — for I/O-bound tasks (GIL limits CPU parallelism)
def worker(n):
    print(f"Thread {n} starting")
    # ... do I/O work ...
    print(f"Thread {n} done")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Thread with result — use shared state or Queue
import queue

results = queue.Queue()

def compute(n, q):
    q.put(n * n)

t = threading.Thread(target=compute, args=(5, results))
t.start()
t.join()
print(results.get())   # 25

# Multiprocessing — for CPU-bound tasks (no GIL)
def cpu_intensive(n):
    return sum(i*i for i in range(n))

with multiprocessing.Pool(processes=4) as pool:
    results = pool.map(cpu_intensive, [100000, 200000, 300000, 400000])
    print(results)
```

### `concurrent.futures` — High-Level Concurrency

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import urllib.request

# ThreadPoolExecutor for I/O-bound
urls = ["http://example.com", "http://example.org", "http://example.net"]

def fetch(url):
    with urllib.request.urlopen(url) as resp:
        return len(resp.read())

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch, url): url for url in urls}
    for future in futures:
        try:
            size = future.result()
            print(f"{futures[future]}: {size} bytes")
        except Exception as e:
            print(f"Error: {e}")

# ProcessPoolExecutor for CPU-bound
def compute(n):
    return sum(i*i for i in range(n))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(compute, [1000000, 2000000, 3000000]))
```
