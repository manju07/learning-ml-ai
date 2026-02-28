# Python Concurrency

## Table of Contents
- [Overview](#overview)
- [Threading](#threading)
- [Multiprocessing](#multiprocessing)
- [asyncio — Async/Await](#asyncio)
- [concurrent.futures](#concurrentfutures)
- [Shared State and Synchronization](#shared-state-and-synchronization)
- [Choosing the Right Approach](#choosing-the-right-approach)

---

## Overview

Python has three main concurrency models:

| Model | Best For | GIL Impact | True Parallelism |
|-------|----------|------------|-----------------|
| **Threading** | I/O-bound tasks | Limited by GIL | No (for CPU) |
| **Multiprocessing** | CPU-bound tasks | Bypasses GIL | Yes |
| **asyncio** | High-concurrency I/O | Single thread | No (cooperative) |

**The GIL (Global Interpreter Lock):** CPython allows only one thread to execute Python bytecode at a time. Threading still helps for I/O-bound work because threads release the GIL during I/O waits.

---

## Threading

### Basic Threads

```python
import threading
import time

def worker(name, duration):
    print(f"[{name}] Starting...")
    time.sleep(duration)   # simulates I/O (releases GIL)
    print(f"[{name}] Done after {duration}s")
    return f"{name} result"

# Create and start threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"Thread-{i}", i + 1))
    t.daemon = True       # thread dies when main thread exits
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

print("All threads complete")
```

### Threads with Results

```python
import threading
import queue

result_queue = queue.Queue()

def compute(n, result_q):
    result_q.put(n * n)

threads = [
    threading.Thread(target=compute, args=(i, result_queue))
    for i in range(5)
]
for t in threads: t.start()
for t in threads: t.join()

results = [result_queue.get() for _ in range(5)]
print(sorted(results))  # [0, 1, 4, 9, 16]
```

### Thread-Local Storage

```python
import threading

local = threading.local()

def set_user(user_id):
    local.user_id = user_id
    print(f"Thread {threading.current_thread().name}: user={local.user_id}")

threads = [
    threading.Thread(target=set_user, args=(i,), name=f"Worker-{i}")
    for i in range(3)
]
for t in threads: t.start()
for t in threads: t.join()
# Each thread has its own local.user_id
```

### Thread Pool

```python
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def fetch_url(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return url, len(resp.read())

urls = [
    "https://www.python.org",
    "https://docs.python.org",
    "https://pypi.org",
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(fetch_url, urls)
    for url, size in results:
        print(f"{url}: {size:,} bytes")
```

---

## Multiprocessing

Bypasses the GIL — each process has its own Python interpreter and memory space.

### Basic Processes

```python
import multiprocessing
import os

def worker(number):
    pid = os.getpid()
    result = sum(i**2 for i in range(number))
    print(f"PID {pid}: result={result}")
    return result

if __name__ == "__main__":   # required on Windows!
    processes = []
    for n in [100_000, 200_000, 300_000]:
        p = multiprocessing.Process(target=worker, args=(n,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        print(f"Exit code: {p.exitcode}")
```

### Process Pool

```python
import multiprocessing

def cpu_task(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    data = [100_000, 200_000, 300_000, 400_000]

    # Pool maps tasks to available processes
    with multiprocessing.Pool(processes=4) as pool:
        # map — synchronous
        results = pool.map(cpu_task, data)
        print(results)

        # starmap — multiple args
        args = [(100, 200), (300, 400)]
        def add(a, b): return a + b
        results = pool.starmap(add, args)

        # async variants
        async_result = pool.map_async(cpu_task, data)
        results = async_result.get(timeout=30)

        # imap — lazy/streaming results
        for result in pool.imap(cpu_task, data):
            print(result)
```

### Shared Memory

```python
import multiprocessing

# Value and Array — shared between processes
counter  = multiprocessing.Value("i", 0)    # 'i' = int
arr      = multiprocessing.Array("d", [1.0, 2.0, 3.0])  # 'd' = double

def increment(counter, times):
    for _ in range(times):
        with counter.get_lock():   # thread/process-safe
            counter.value += 1

processes = [
    multiprocessing.Process(target=increment, args=(counter, 1000))
    for _ in range(4)
]
for p in processes: p.start()
for p in processes: p.join()
print(counter.value)   # 4000

# Manager — more flexible shared state
with multiprocessing.Manager() as manager:
    shared_list = manager.list([1, 2, 3])
    shared_dict = manager.dict({"key": "value"})
    # ... use in multiple processes
```

### Inter-Process Communication

```python
import multiprocessing

# Queue — safe for multiple producers/consumers
def producer(q, items):
    for item in items:
        q.put(item)
    q.put(None)   # sentinel

def consumer(q, results):
    while True:
        item = q.get()
        if item is None:
            break
        results.put(item * 2)

if __name__ == "__main__":
    q       = multiprocessing.Queue()
    results = multiprocessing.Queue()

    p = multiprocessing.Process(target=producer, args=(q, [1,2,3,4,5]))
    c = multiprocessing.Process(target=consumer, args=(q, results))
    p.start(); c.start()
    p.join();  c.join()

    while not results.empty():
        print(results.get())   # 2 4 6 8 10

# Pipe — two-way communication between two processes
parent_conn, child_conn = multiprocessing.Pipe()

def child(conn):
    msg = conn.recv()
    conn.send(f"Echo: {msg}")
    conn.close()

p = multiprocessing.Process(target=child, args=(child_conn,))
p.start()
parent_conn.send("Hello!")
print(parent_conn.recv())   # Echo: Hello!
p.join()
```

---

## asyncio

`asyncio` is Python's built-in library for **single-threaded asynchronous** I/O using coroutines.

### Basic Coroutines

```python
import asyncio

async def greet(name, delay):
    print(f"Hello, {name}!")
    await asyncio.sleep(delay)   # non-blocking sleep
    print(f"Goodbye, {name}!")

async def main():
    # Run sequentially (total time: 3s)
    await greet("Alice", 1)
    await greet("Bob", 2)

asyncio.run(main())

# Run concurrently (total time: ~2s)
async def main():
    await asyncio.gather(
        greet("Alice", 1),
        greet("Bob", 2),
    )
```

### Tasks and Gathering

```python
import asyncio

async def fetch_data(url, delay):
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)   # simulate network request
    return f"Data from {url}"

async def main():
    urls = [
        ("https://api.example.com/users", 1),
        ("https://api.example.com/posts", 2),
        ("https://api.example.com/comments", 0.5),
    ]

    # Create tasks — they start running immediately
    tasks = [asyncio.create_task(fetch_data(url, delay)) for url, delay in urls]

    # Wait for all (preserves order)
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)

    # With exception handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            print(f"Error: {r}")
        else:
            print(r)

asyncio.run(main())
```

### Async Context Managers and Iterators

```python
import asyncio

class AsyncDatabase:
    async def __aenter__(self):
        print("Connecting to DB...")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, *args):
        print("Disconnecting from DB...")
        await asyncio.sleep(0.1)

    async def query(self, sql):
        await asyncio.sleep(0.05)
        return [{"id": 1, "name": "Alice"}]


# Async generator
async def stream_records(n):
    for i in range(n):
        await asyncio.sleep(0.1)  # simulate DB fetch
        yield {"id": i, "value": i * 2}


async def main():
    async with AsyncDatabase() as db:
        records = await db.query("SELECT * FROM users")
        print(records)

    # Async for loop over async generator
    async for record in stream_records(5):
        print(record)

    # Async comprehensions
    values = [r["value"] async for r in stream_records(5)]
    print(values)

asyncio.run(main())
```

### Timeouts and Cancellation

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "done"

async def main():
    # Timeout
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("Operation timed out!")

    # Cancel a task
    task = asyncio.create_task(slow_operation())
    await asyncio.sleep(1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("Task was cancelled")

    # as_completed — process results as they arrive
    tasks = [asyncio.create_task(slow_operation()) for _ in range(3)]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(f"Got: {result}")

asyncio.run(main())
```

### Real-World asyncio with aiohttp

```python
import asyncio
import aiohttp   # pip install aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

urls = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
]

results = asyncio.run(fetch_all(urls))
for r in results:
    print(r["title"])
```

### asyncio Event Loop

```python
import asyncio

# Modern way — asyncio.run() handles event loop lifecycle
asyncio.run(main())

# Manual event loop (older style, use only if needed)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    loop.run_until_complete(main())
finally:
    loop.close()

# Inside async context — get running loop
async def task():
    loop = asyncio.get_event_loop()
    loop = asyncio.get_running_loop()   # preferred in 3.10+
```

---

## concurrent.futures

High-level interface for both thread and process pools.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

def slow_square(n):
    time.sleep(0.5)
    return n * n

# ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    # map — maintains order, blocks until all done
    results = list(executor.map(slow_square, range(10)))
    print(results)

    # submit — returns Future objects
    futures = {executor.submit(slow_square, n): n for n in range(10)}
    for future in as_completed(futures):
        n = futures[future]
        try:
            result = future.result()
            print(f"slow_square({n}) = {result}")
        except Exception as e:
            print(f"Error for {n}: {e}")

# ProcessPoolExecutor — same API, separate processes
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(slow_square, range(10)))
```

### Future API

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    future = executor.submit(lambda: 42)

    # Check state
    print(future.done())      # True/False
    print(future.running())   # True/False
    print(future.cancelled()) # True/False

    # Get result (blocks if not done)
    result = future.result(timeout=5.0)

    # Callbacks
    future.add_done_callback(lambda f: print(f"Done: {f.result()}"))

    # Cancel (only if not started)
    future.cancel()
```

---

## Shared State and Synchronization

### Lock

```python
import threading

counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    with lock:              # acquire + release automatically
        counter += 1

threads = [threading.Thread(target=safe_increment) for _ in range(1000)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)   # 1000 (always!)
```

### RLock (Reentrant Lock)

```python
import threading

rlock = threading.RLock()

def outer():
    with rlock:         # acquires lock
        print("outer")
        inner()         # inner can re-acquire the same lock

def inner():
    with rlock:         # works with RLock, would deadlock with Lock!
        print("inner")

outer()
```

### Semaphore

```python
import threading
import time

# Limit concurrent access to a resource
semaphore = threading.Semaphore(3)  # max 3 at once

def access_resource(n):
    with semaphore:
        print(f"Thread {n} accessing resource")
        time.sleep(1)
        print(f"Thread {n} releasing resource")

threads = [threading.Thread(target=access_resource, args=(i,)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()
```

### Event

```python
import threading
import time

event = threading.Event()

def waiter():
    print("Waiter: waiting for event...")
    event.wait()        # blocks until event is set
    print("Waiter: event received!")

def setter():
    time.sleep(2)
    print("Setter: setting event")
    event.set()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)
t1.start(); t2.start()
t1.join();  t2.join()
```

### Queue (Thread-Safe)

```python
import threading
import queue
import time

task_queue = queue.Queue(maxsize=10)
POISON_PILL = None

def producer():
    for i in range(20):
        task_queue.put(i)
        print(f"Produced: {i}")
    task_queue.put(POISON_PILL)  # signal done

def consumer():
    while True:
        item = task_queue.get()
        if item is POISON_PILL:
            break
        time.sleep(0.1)
        print(f"Consumed: {item}")
        task_queue.task_done()

p = threading.Thread(target=producer)
c = threading.Thread(target=consumer)
p.start(); c.start()
p.join();  c.join()

# task_queue.join()  # wait until all tasks processed (task_done called)
```

---

## Choosing the Right Approach

```python
# Rule of thumb:

# 1. I/O-bound (network, disk, database):
#    → asyncio (highest throughput, lowest overhead)
#    → ThreadPoolExecutor (simpler, works with blocking libs)

# 2. CPU-bound (computation, data processing):
#    → ProcessPoolExecutor / multiprocessing

# 3. Mixed (web server with heavy computation):
#    → asyncio for I/O + run_in_executor for CPU tasks

import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_work(data):
    return sum(x**2 for x in data)

async def mixed_workflow():
    loop = asyncio.get_running_loop()

    # Offload CPU work to process pool, await result
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, cpu_bound_work, list(range(1_000_000))
        )
    print(f"CPU result: {result}")

asyncio.run(mixed_workflow())
```

### Quick Decision Guide

```
Is your bottleneck I/O? (network, disk, database)
  ├─ Yes, many concurrent connections needed → asyncio
  ├─ Yes, simpler/legacy code → ThreadPoolExecutor
  └─ No, CPU is the bottleneck → ProcessPoolExecutor

Number of items to process?
  ├─ Few, complex → ProcessPoolExecutor
  ├─ Many, simple → multiprocessing.Pool
  └─ Streaming → asyncio with async generators
```
