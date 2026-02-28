# NumPy — Numerical Python

## Table of Contents
- [Introduction](#introduction)
- [Arrays (ndarray)](#arrays-ndarray)
- [Array Creation](#array-creation)
- [Array Attributes](#array-attributes)
- [Indexing and Slicing](#indexing-and-slicing)
- [Array Operations](#array-operations)
- [Broadcasting](#broadcasting)
- [Universal Functions (ufuncs)](#universal-functions-ufuncs)
- [Linear Algebra](#linear-algebra)
- [Random Module](#random-module)
- [Array Manipulation](#array-manipulation)
- [Performance Tips](#performance-tips)

---

## Introduction

NumPy is the foundational library for numerical computing in Python. It provides:
- A powerful N-dimensional array object (`ndarray`)
- Vectorized math operations (no Python loops needed)
- Linear algebra, Fourier transforms, and random number generation
- Foundation for Pandas, Matplotlib, SciPy, TensorFlow, etc.

```python
import numpy as np
print(np.__version__)   # e.g., 1.26.4
```

---

## Arrays (ndarray)

### Why NumPy Arrays over Python Lists?

```python
import numpy as np
import time

n = 10_000_000

# Python list
lst = list(range(n))
start = time.time()
result = [x * 2 for x in lst]
print(f"List:  {time.time() - start:.3f}s")   # ~1.5s

# NumPy array
arr = np.arange(n)
start = time.time()
result = arr * 2
print(f"NumPy: {time.time() - start:.3f}s")   # ~0.02s (75x faster!)

# Memory comparison
import sys
lst_mem = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in range(100))
arr_mem = arr[:100].nbytes
print(f"List item size: {sys.getsizeof(42)} bytes")    # 28 bytes per int
print(f"NumPy int64:    {arr.itemsize} bytes/element") # 8 bytes per element
```

---

## Array Creation

```python
import numpy as np

# From Python lists
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])

# Dtype control
float_arr = np.array([1, 2, 3], dtype=np.float64)
int_arr   = np.array([1.7, 2.3, 3.9], dtype=np.int32)  # truncates!
bool_arr  = np.array([0, 1, 0, 2], dtype=bool)          # [False, True, False, True]

# Zeros, ones, empty
zeros  = np.zeros((3, 4))              # 3×4 float64 zeros
ones   = np.ones((2, 3, 4))           # 2×3×4 float64 ones
empty  = np.empty((2, 2))             # uninitialized (fast but random values)
full   = np.full((3, 3), 7.0)         # filled with 7.0
eye    = np.eye(4)                    # 4×4 identity matrix
diag   = np.diag([1, 2, 3, 4])       # diagonal matrix

# Like another array's shape
a = np.array([[1, 2], [3, 4]])
b = np.zeros_like(a)       # same shape/dtype, all zeros
c = np.ones_like(a, dtype=float)

# Ranges
arange = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8] — like range()
linspace = np.linspace(0, 1, 11)     # [0., 0.1, ..., 1.0] — 11 evenly spaced
logspace = np.logspace(0, 3, 4)      # [1., 10., 100., 1000.] — log-spaced

# Random (see Random section for full details)
rand  = np.random.rand(3, 4)         # uniform [0, 1)
randn = np.random.randn(3, 4)        # standard normal

# From functions
grid_x = np.fromfunction(lambda i, j: i + j, (3, 3))
arr    = np.fromiter((x**2 for x in range(10)), dtype=int)
```

---

## Array Attributes

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

print(a.ndim)       # 2 — number of dimensions
print(a.shape)      # (2, 3) — (rows, cols)
print(a.size)       # 6 — total number of elements
print(a.dtype)      # float32
print(a.itemsize)   # 4 — bytes per element (float32)
print(a.nbytes)     # 24 — total bytes (size * itemsize)
print(a.T)          # transposed
print(a.real)       # real part (for complex arrays)
print(a.imag)       # imaginary part

# Change dtype
b = a.astype(np.int64)
c = a.astype("float16")   # string shorthand
```

---

## Indexing and Slicing

### 1D Arrays

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70])

print(arr[0])       # 10
print(arr[-1])      # 70
print(arr[2:5])     # [30, 40, 50]
print(arr[::2])     # [10, 30, 50, 70]
print(arr[::-1])    # [70, 60, 50, 40, 30, 20, 10]
```

### 2D Arrays

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(a[0, 0])      # 1 — row 0, col 0
print(a[1, 2])      # 6 — row 1, col 2
print(a[-1, -1])    # 9 — last row, last col

print(a[0])         # [1, 2, 3] — entire row 0
print(a[:, 1])      # [2, 5, 8] — entire col 1

print(a[0:2, 1:3])  # [[2, 3], [5, 6]] — submatrix

# NumPy slices are VIEWS, not copies!
view = a[0:2, 0:2]
view[0, 0] = 99
print(a[0, 0])   # 99 — original modified!

# Explicit copy
copy = a[0:2, 0:2].copy()
```

### Boolean Indexing (Fancy Indexing)

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])

# Boolean mask
mask = arr > 4
print(mask)    # [False False False False  True  True False  True  True False]
print(arr[mask])   # [5, 9, 6, 5]

# Shorthand
print(arr[arr > 4])
print(arr[arr % 2 == 0])   # even numbers

# Compound conditions
print(arr[(arr > 2) & (arr < 6)])   # [3, 4, 5, 5, 3] — AND
print(arr[(arr < 2) | (arr > 7)])   # [1, 1, 9] — OR
print(arr[~(arr > 4)])              # NOT

# np.where — conditional selection or replacement
result = np.where(arr > 4, arr * 2, 0)   # double if > 4 else 0
indices = np.where(arr > 4)              # indices where condition is True
```

### Integer Array Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Select by indices
print(arr[[0, 2, 4]])    # [10, 30, 50]
print(arr[[0, 0, -1]])   # [10, 10, 50] — repetition allowed

# 2D fancy indexing
a = np.array([[1, 2], [3, 4], [5, 6]])
rows = [0, 1, 2]
cols = [0, 1, 0]
print(a[rows, cols])   # [1, 4, 5] — picks a[0,0], a[1,1], a[2,0]
```

---

## Array Operations

### Element-wise Operations

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)    # [6,  8, 10, 12]
print(a - b)    # [-4, -4, -4, -4]
print(a * b)    # [5, 12, 21, 32]
print(a / b)    # [0.2, 0.33, 0.43, 0.5]
print(a ** 2)   # [1, 4, 9, 16]
print(a % 3)    # [1, 2, 0, 1]

# Scalar operations (broadcasting with scalar)
print(a + 10)   # [11, 12, 13, 14]
print(a * 2)    # [2, 4, 6, 8]
```

### Aggregate Functions

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.sum())            # 21 — sum of all
print(a.sum(axis=0))      # [5, 7, 9] — sum along rows (collapse rows)
print(a.sum(axis=1))      # [6, 15] — sum along cols (collapse cols)

print(a.min(), a.max())   # 1, 6
print(a.min(axis=0))      # [1, 2, 3]
print(a.max(axis=1))      # [3, 6]

print(a.mean())           # 3.5
print(a.std())            # standard deviation
print(a.var())            # variance
print(np.median(a))       # 3.5

print(a.cumsum())         # cumulative sum: [1, 3, 6, 10, 15, 21]
print(a.cumprod())        # cumulative product

print(a.argmin())         # index of minimum
print(a.argmax())         # index of maximum
print(a.argmin(axis=0))   # [0, 0, 0] — row index of min per column
```

### Comparison and Logical

```python
a = np.array([1, 2, 3, 4, 5])

print(a > 3)                # [F F F T T]
print(np.all(a > 0))        # True — all elements satisfy
print(np.any(a > 4))        # True — at least one satisfies
print(np.count_nonzero(a > 3))   # 2

# Logical operations on arrays
x = np.array([True, False, True])
y = np.array([False, False, True])
print(np.logical_and(x, y))   # [F F T]
print(np.logical_or(x, y))    # [T F T]
print(np.logical_not(x))      # [F T F]
```

---

## Broadcasting

Broadcasting allows operations on arrays of different shapes.

```python
import numpy as np

# Rule: dimensions are compatible if they are equal OR one of them is 1

# Scalar broadcast
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a + 10)   # adds 10 to every element

# 1D broadcast over 2D
row = np.array([1, 2, 3])        # shape (3,)
mat = np.ones((4, 3))            # shape (4, 3)
print(mat + row)   # adds row to every row of mat (shape: 4×3)

# Column broadcast
col = np.array([[1], [2], [3], [4]])   # shape (4, 1)
print(mat + col)   # adds different value to each row (shape: 4×3)

# Outer product via broadcasting
a = np.array([1, 2, 3])       # shape (3,)
b = np.array([10, 20, 30])    # shape (3,)
# outer = a[:, np.newaxis] * b
outer = a.reshape(-1, 1) * b  # shape (3, 3)
print(outer)
# [[10, 20, 30],
#  [20, 40, 60],
#  [30, 60, 90]]

# Normalize each row to [0, 1]
data = np.random.rand(4, 3)
min_vals = data.min(axis=1, keepdims=True)   # shape (4, 1)
max_vals = data.max(axis=1, keepdims=True)   # shape (4, 1)
normalized = (data - min_vals) / (max_vals - min_vals)
```

---

## Universal Functions (ufuncs)

```python
import numpy as np

a = np.array([1, 4, 9, 16, 25], dtype=float)

# Math ufuncs — element-wise
print(np.sqrt(a))      # [1., 2., 3., 4., 5.]
print(np.exp(a))       # e^x
print(np.log(a))       # natural log
print(np.log2(a))
print(np.log10(a))
print(np.abs(np.array([-1, -2, 3])))   # [1, 2, 3]

# Trig
angles = np.linspace(0, 2 * np.pi, 7)
print(np.sin(angles))
print(np.cos(angles))

# Rounding
b = np.array([1.4, 1.5, 1.6, 2.5])
print(np.round(b))     # [1., 2., 2., 2.] — banker's rounding
print(np.floor(b))     # [1., 1., 1., 2.]
print(np.ceil(b))      # [2., 2., 2., 3.]
print(np.trunc(b))     # [1., 1., 1., 2.]

# Clipping
arr = np.array([-2, -1, 0, 1, 2, 3, 4])
print(np.clip(arr, 0, 2))    # [0, 0, 0, 1, 2, 2, 2]

# Two-input ufuncs
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])
print(np.maximum(a, b))   # [3, 2, 3]
print(np.minimum(a, b))   # [1, 2, 1]
print(np.hypot(a, b))     # sqrt(a² + b²)
print(np.power(a, b))     # [1^3, 2^2, 3^1] = [1, 4, 3]
```

---

## Linear Algebra

```python
import numpy as np

A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

# Matrix multiplication
print(A @ B)                  # matrix product (Python 3.5+)
print(np.matmul(A, B))        # same
print(np.dot(A, B))           # same for 2D

# Element-wise
print(A * B)

# Transpose
print(A.T)
print(A.transpose())

# Determinant, inverse, trace
print(np.linalg.det(A))       # -2.0
print(np.linalg.inv(A))       # inverse
print(np.trace(A))            # sum of diagonal = 5

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)    # [-0.372..., 5.372...]
print(eigenvectors)

# SVD — Singular Value Decomposition
U, S, Vt = np.linalg.svd(A)
print(U, S, Vt)

# Solve linear system Ax = b
b = np.array([5, 6])
x = np.linalg.solve(A, b)    # x such that A @ x == b
print(x)

# Least squares
x_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

# Norms
v = np.array([3, 4])
print(np.linalg.norm(v))          # 5.0 (L2 norm)
print(np.linalg.norm(v, ord=1))   # 7.0 (L1 norm)
print(np.linalg.norm(v, ord=np.inf))  # 4.0 (L-inf norm)

# Matrix norm
print(np.linalg.norm(A, "fro"))   # Frobenius norm
```

---

## Random Module

```python
import numpy as np

rng = np.random.default_rng(seed=42)   # recommended modern API

# Uniform [0, 1)
print(rng.random((3, 4)))

# Integers
print(rng.integers(0, 10, size=(3, 3)))   # [low, high)

# Normal distribution
print(rng.normal(loc=0, scale=1, size=(3, 3)))

# Other distributions
print(rng.uniform(low=-1, high=1, size=5))
print(rng.binomial(n=10, p=0.5, size=100))
print(rng.poisson(lam=3, size=10))
print(rng.exponential(scale=1, size=5))
print(rng.choice([1, 2, 3, 4, 5], size=3, replace=False))

# Shuffle (in-place)
arr = np.arange(10)
rng.shuffle(arr)
print(arr)

# Permutation (returns new array)
perm = rng.permutation(10)   # or rng.permutation(arr)

# Legacy API (less recommended)
np.random.seed(42)
np.random.rand(3, 4)        # uniform [0, 1)
np.random.randn(3, 4)       # standard normal
np.random.randint(0, 10, (3, 3))
np.random.choice(5, 3, replace=False)
np.random.shuffle(arr)
```

---

## Array Manipulation

```python
import numpy as np

# Reshape — new view of same data (must have same total elements)
a = np.arange(12)
b = a.reshape(3, 4)      # 3×4
c = a.reshape(2, 2, 3)   # 2×2×3
d = a.reshape(3, -1)     # -1 = infer (3×4)

# Flatten — always returns a copy
flat = b.flatten()

# Ravel — returns a view if possible
flat = b.ravel()

# Transpose
a = np.arange(24).reshape(2, 3, 4)
print(a.T.shape)              # (4, 3, 2)
print(a.transpose(2, 0, 1).shape)  # (4, 2, 3)

# Adding dimensions
a = np.array([1, 2, 3])
print(a[np.newaxis, :].shape)      # (1, 3) — row vector
print(a[:, np.newaxis].shape)      # (3, 1) — column vector
print(np.expand_dims(a, axis=0).shape)  # (1, 3)

# Squeezing dimensions (remove size-1 dimensions)
a = np.array([[[1, 2, 3]]])  # shape (1, 1, 3)
print(np.squeeze(a).shape)   # (3,)

# Stacking
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.stack([a, b]))          # [[1,2,3],[4,5,6]] — new axis
print(np.stack([a, b], axis=1))  # [[1,4],[2,5],[3,6]]
print(np.vstack([a, b]))         # vertical stack: [[1,2,3],[4,5,6]]
print(np.hstack([a, b]))         # horizontal stack: [1,2,3,4,5,6]
print(np.concatenate([a, b]))    # same as hstack for 1D

# 2D stacking
A = np.ones((2, 3))
B = np.ones((2, 3)) * 2
print(np.vstack([A, B]).shape)   # (4, 3)
print(np.hstack([A, B]).shape)   # (2, 6)

# Splitting
arr = np.arange(12)
parts = np.split(arr, 3)         # 3 equal parts
parts = np.split(arr, [3, 7])    # split at indices 3 and 7

arr2d = np.arange(12).reshape(4, 3)
rows  = np.vsplit(arr2d, 2)      # split into 2 vertical halves
cols  = np.hsplit(arr2d, 3)      # split into 3 horizontal thirds

# Repeat and tile
a = np.array([1, 2, 3])
print(np.repeat(a, 3))           # [1,1,1,2,2,2,3,3,3]
print(np.tile(a, 3))             # [1,2,3,1,2,3,1,2,3]
print(np.tile(a, (2, 3)))        # 2×9 array
```

### Sorting

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

sorted_arr = np.sort(arr)           # returns sorted copy
arr.sort()                          # in-place

indices = np.argsort(arr)           # indices that would sort the array

a = np.array([[3, 1, 2], [6, 4, 5]])
print(np.sort(a, axis=1))           # sort each row
print(np.sort(a, axis=0))           # sort each column

# Structured array sorting
dtype = [("name", "U10"), ("age", int)]
data  = np.array([("Alice", 30), ("Bob", 25), ("Carol", 35)], dtype=dtype)
print(np.sort(data, order="age"))   # sort by age field
```

---

## Performance Tips

```python
import numpy as np
import timeit

# 1. Avoid Python loops — use vectorized operations
n = 1_000_000
a = np.random.rand(n)
b = np.random.rand(n)

# Slow (Python loop)
def dot_loop(a, b):
    return sum(x*y for x, y in zip(a, b))

# Fast (NumPy)
np.dot(a, b)      # ~100x faster

# 2. Preallocate arrays
result = np.empty(n)   # faster than growing array

# 3. Use in-place operations to save memory
a += b         # modifies a in-place
np.add(a, b, out=a)   # explicit in-place with ufunc

# 4. Use contiguous arrays for C functions
arr = np.ascontiguousarray(arr)   # ensures C-contiguous memory

# 5. Choose the right dtype
a_float64 = np.ones(1_000_000)            # 8 MB
a_float32 = np.ones(1_000_000, dtype=np.float32)  # 4 MB

# 6. Use np.einsum for complex contractions
# Matrix multiplication
A = np.random.rand(100, 200)
B = np.random.rand(200, 300)
result = np.einsum("ij,jk->ik", A, B)  # same as A @ B

# Batch matrix multiplication
batch_A = np.random.rand(10, 100, 200)
batch_B = np.random.rand(10, 200, 300)
result  = np.einsum("bij,bjk->bik", batch_A, batch_B)
```

### Structured Arrays

```python
# Like a database table in NumPy
dtype = np.dtype([
    ("name",  "U20"),
    ("age",   np.int32),
    ("score", np.float64),
])

students = np.array([
    ("Alice", 20, 95.5),
    ("Bob",   22, 87.3),
    ("Carol", 21, 91.0),
], dtype=dtype)

print(students["name"])          # ['Alice' 'Bob' 'Carol']
print(students["score"].mean())  # 91.27
print(students[students["age"] > 20])  # filter by age
```
