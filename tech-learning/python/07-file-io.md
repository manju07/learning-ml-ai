# Python File I/O

## Table of Contents
- [Reading and Writing Files](#reading-and-writing-files)
- [File Modes](#file-modes)
- [Working with Text Files](#working-with-text-files)
- [Working with Binary Files](#working-with-binary-files)
- [CSV Files](#csv-files)
- [JSON Files](#json-files)
- [Pickle — Object Serialization](#pickle)
- [pathlib — Modern File Paths](#pathlib)
- [Temporary Files](#temporary-files)
- [File System Operations](#file-system-operations)
- [Watching Files](#watching-files)

---

## Reading and Writing Files

### The `open()` Function

```python
# Syntax: open(file, mode='r', encoding=None, buffering=-1, ...)

# Always use 'with' — auto-closes file even on exceptions
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()   # read entire file as string
```

---

## File Modes

| Mode | Description |
|------|-------------|
| `'r'` | Read (default). Raises `FileNotFoundError` if missing. |
| `'w'` | Write. Creates file or **truncates** existing. |
| `'a'` | Append. Creates file or appends to existing. |
| `'x'` | Exclusive creation. Raises `FileExistsError` if exists. |
| `'r+'` | Read + write (file must exist). |
| `'w+'` | Read + write (truncates). |
| `'b'` | Binary mode (combine: `'rb'`, `'wb'`). |
| `'t'` | Text mode (default). |

```python
# Read
with open("file.txt", "r") as f: ...

# Write (overwrites)
with open("file.txt", "w") as f: ...

# Append
with open("file.txt", "a") as f: ...

# Binary read
with open("image.png", "rb") as f: ...

# Read + write
with open("file.txt", "r+") as f: ...
```

---

## Working with Text Files

### Reading

```python
# Read entire file as one string
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)

# Read all lines into a list
with open("data.txt") as f:
    lines = f.readlines()   # includes newlines: ['line1\n', 'line2\n']

# Read line by line (memory efficient for large files)
with open("data.txt") as f:
    for line in f:             # file object is iterable
        print(line.strip())    # strip removes trailing newline

# Read one line at a time
with open("data.txt") as f:
    first_line  = f.readline()
    second_line = f.readline()
    print(repr(first_line))    # 'line 1 content\n'
```

### Writing

```python
# Write string
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("Second line\n")

# Write multiple lines at once
lines = ["line 1\n", "line 2\n", "line 3\n"]
with open("output.txt", "w") as f:
    f.writelines(lines)

# Append
with open("log.txt", "a") as f:
    f.write("New log entry\n")

# Using print to write to file
with open("output.txt", "w") as f:
    print("Hello from print!", file=f)
    print("Another line", file=f)
```

### File Position

```python
with open("data.txt", "r+") as f:
    print(f.tell())            # 0 — current position (bytes)

    f.read(10)                 # read 10 chars
    print(f.tell())            # 10

    f.seek(0)                  # go to beginning
    f.seek(0, 2)               # go to end (2 = SEEK_END)
    f.seek(5, 0)               # 5 from start
    f.seek(-3, 2)              # 3 from end
```

---

## Working with Binary Files

```python
# Read a PNG file header
with open("image.png", "rb") as f:
    header = f.read(8)
    print(header)  # b'\x89PNG\r\n\x1a\n'

# Copy a binary file
def copy_file(src, dst, chunk_size=8192):
    with open(src, "rb") as src_f, open(dst, "wb") as dst_f:
        while chunk := src_f.read(chunk_size):
            dst_f.write(chunk)

# Read structured binary data with struct
import struct

# Pack: create binary data
data = struct.pack(">HHI", 1920, 1080, 0xFFFFFF)  # two shorts, one int
print(data)

# Unpack: parse binary data
width, height, color = struct.unpack(">HHI", data)
print(f"{width}x{height}, color={color:#08x}")

# Common format characters
# H = unsigned short (2 bytes)
# I = unsigned int (4 bytes)
# f = float (4 bytes)
# d = double (8 bytes)
# s = char[] string
# > = big-endian, < = little-endian
```

---

## CSV Files

```python
import csv

# Writing CSV
headers = ["name", "age", "city"]
rows = [
    ["Alice", 30, "New York"],
    ["Bob", 25, "London"],
    ["Carol", 35, "Tokyo"],
]

with open("people.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)    # header
    writer.writerows(rows)      # all data rows

# Reading CSV
with open("people.csv", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)       # skip header
    for row in reader:
        print(row)              # ['Alice', '30', 'New York']

# DictWriter — write dicts
with open("people.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age", "city"])
    writer.writeheader()
    writer.writerow({"name": "Alice", "age": 30, "city": "New York"})
    writer.writerows([
        {"name": "Bob",   "age": 25, "city": "London"},
        {"name": "Carol", "age": 35, "city": "Tokyo"},
    ])

# DictReader — read as dicts
with open("people.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])   # Alice 30
```

### CSV with Custom Delimiters

```python
# Tab-separated values (TSV)
with open("data.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["col1", "col2"])

# Semicolon-separated (European style)
with open("data.csv", newline="") as f:
    reader = csv.reader(f, delimiter=";", quotechar='"')
    for row in reader:
        print(row)
```

---

## JSON Files

```python
import json

data = {
    "users": [
        {"name": "Alice", "age": 30, "active": True},
        {"name": "Bob",   "age": 25, "active": False},
    ],
    "total": 2,
}

# Write JSON
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Read JSON
with open("data.json", encoding="utf-8") as f:
    loaded = json.load(f)

print(loaded["users"][0]["name"])  # Alice

# Custom encoder for complex types
from datetime import datetime
import json

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, set):
            return {"__set__": list(obj)}
        return super().default(obj)

def custom_decoder(obj):
    if "__datetime__" in obj:
        return datetime.fromisoformat(obj["__datetime__"])
    if "__set__" in obj:
        return set(obj["__set__"])
    return obj

data = {"created": datetime.now(), "tags": {"python", "code"}}
json_str = json.dumps(data, cls=CustomEncoder)
restored = json.loads(json_str, object_hook=custom_decoder)
```

---

## Pickle — Object Serialization

`pickle` serializes arbitrary Python objects. **Never unpickle untrusted data!**

```python
import pickle

# Complex Python object
class Config:
    def __init__(self):
        self.settings = {"debug": True, "port": 8080}
        self.callbacks = [print, len]  # functions too!

config = Config()

# Serialize to file
with open("config.pkl", "wb") as f:
    pickle.dump(config, f)

# Deserialize from file
with open("config.pkl", "rb") as f:
    loaded_config = pickle.load(f)

print(loaded_config.settings)   # {'debug': True, 'port': 8080}

# Serialize to bytes (in-memory)
data_bytes = pickle.dumps(config)
restored   = pickle.loads(data_bytes)
```

---

## pathlib — Modern File Paths

```python
from pathlib import Path

# Create path objects
home    = Path.home()                    # /home/user
cwd     = Path.cwd()                    # current directory
p       = Path("/usr/local/lib/python3.12/site-packages")
rel     = Path("data/input.csv")        # relative path

# Navigation
print(p.parent)       # /usr/local/lib/python3.12
print(p.parents[2])   # /usr/local
print(p.name)         # site-packages
print(p.stem)         # site-packages (no suffix)
print(p.suffix)       # '' (no extension)

# Building paths
config = home / ".config" / "myapp" / "settings.json"
print(config)   # /home/user/.config/myapp/settings.json

# Check existence
print(config.exists())
print(config.is_file())
print(config.is_dir())

# Create directories
config.parent.mkdir(parents=True, exist_ok=True)

# Read / write
config.write_text('{"theme": "dark"}', encoding="utf-8")
content = config.read_text(encoding="utf-8")
raw     = config.read_bytes()

# Delete
config.unlink()                   # delete file
config.parent.rmdir()             # delete empty dir

# Rename / move
old = Path("old_name.txt")
old.rename("new_name.txt")        # rename (same dir)
old.replace("/tmp/new_name.txt")  # move + rename (cross-device)

# Glob patterns
src = Path("src")
for py in src.glob("*.py"):         # only in src/
    print(py)
for py in src.rglob("*.py"):        # recursively
    print(py)
for py in Path(".").glob("**/*.py"):  # equivalent
    print(py)

# Iterating a directory
for item in Path(".").iterdir():
    if item.is_file():
        print(item.name, item.stat().st_size)

# File metadata
stat = config.stat()
import datetime
mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
print(f"Size: {stat.st_size} bytes, Modified: {mtime}")

# Resolve absolute path
print(rel.resolve())   # /current/working/dir/data/input.csv
```

---

## Temporary Files

```python
import tempfile

# Temporary file — auto-deleted on close
with tempfile.NamedTemporaryFile(
    mode="w",
    suffix=".txt",
    delete=True,
    encoding="utf-8"
) as tmp:
    tmp.write("Temporary content")
    print(tmp.name)   # e.g., /tmp/tmp8ij3xk2a.txt

# Temporary directory — auto-deleted when context exits
with tempfile.TemporaryDirectory() as tmpdir:
    from pathlib import Path
    p = Path(tmpdir) / "data.txt"
    p.write_text("Hello!")
    print(p.read_text())   # Hello!
# tmpdir auto-deleted here

# Manual control
tmp = tempfile.mkstemp(suffix=".db")   # (fd, path)
fd, path = tmp
import os
os.close(fd)    # must close file descriptor
# ... use path ...
os.unlink(path) # manually delete
```

---

## File System Operations

```python
import os
import shutil
from pathlib import Path

# Create
os.makedirs("a/b/c", exist_ok=True)

# Copy
shutil.copy("src.txt", "dst.txt")           # file → file (metadata not copied)
shutil.copy2("src.txt", "dst.txt")          # file → file (with metadata)
shutil.copytree("src_dir", "dst_dir")       # copy entire directory tree

# Move
shutil.move("old_path", "new_path")         # move/rename file or dir

# Delete
os.remove("file.txt")                       # remove file
os.rmdir("empty_dir")                       # remove empty dir
shutil.rmtree("dir_with_contents")          # remove dir + all contents

# Disk usage
total, used, free = shutil.disk_usage("/")
print(f"Free: {free / 1e9:.1f} GB")

# Permissions (Unix)
os.chmod("script.py", 0o755)   # rwxr-xr-x
```

---

## Watching Files

```python
# Using watchdog (pip install watchdog)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f"Modified: {event.src_path}")

    def on_created(self, event):
        print(f"Created: {event.src_path}")

    def on_deleted(self, event):
        print(f"Deleted: {event.src_path}")

observer = Observer()
observer.schedule(MyHandler(), path=".", recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()
```

---

## Best Practices

```python
# 1. Always use 'with' for file handling
with open("file.txt") as f:
    data = f.read()

# 2. Always specify encoding for text files
with open("file.txt", encoding="utf-8") as f:
    ...

# 3. Use pathlib over os.path
from pathlib import Path
config = Path.home() / ".config" / "app.json"

# 4. Handle file errors gracefully
try:
    with open("missing.txt") as f:
        data = f.read()
except FileNotFoundError:
    print("File not found")
except PermissionError:
    print("Permission denied")

# 5. Use newline='' for csv.writer/reader on Windows
with open("data.csv", "w", newline="") as f:
    ...

# 6. For large files, iterate line-by-line
with open("huge.log") as f:
    for line in f:    # memory-efficient
        process(line)
```
