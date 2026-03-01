# Linear Algebra for Machine Learning: Comprehensive Reference

## Table of Contents
1. [Vectors & Operations](#1-vectors--operations)
2. [Matrices & Operations](#2-matrices--operations)
3. [Linear Transformations](#3-linear-transformations)
4. [Systems of Linear Equations](#4-systems-of-linear-equations)
5. [Vector Spaces, Subspaces & Basis](#5-vector-spaces-subspaces--basis)
6. [Orthogonality & Gram-Schmidt](#6-orthogonality--gram-schmidt)
7. [Eigenvalues & Eigenvectors](#7-eigenvalues--eigenvectors)
8. [Singular Value Decomposition (SVD)](#8-singular-value-decomposition-svd)
9. [Matrix Decompositions: LU, QR, Cholesky](#9-matrix-decompositions-lu-qr-cholesky)
10. [Positive Definite Matrices](#10-positive-definite-matrices)
11. [Norms & Distances](#11-norms--distances)
12. [ML Applications](#12-ml-applications)

---

## 1. Vectors & Operations

### 1.1 Definition & Notation

A vector \( \mathbf{v} \in \mathbb{R}^n \) is an ordered tuple of \( n \) real numbers:
\[
\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}
\]

Geometrically: an arrow from the origin to the point \( (v_1, v_2, \ldots, v_n) \).

```python
import numpy as np

# Column vector (convention in linear algebra)
v = np.array([1.0, 2.0, 3.0])

# All these represent the same mathematical object
v_col = v.reshape(-1, 1)   # Shape (3, 1)
v_row = v.reshape(1, -1)   # Shape (1, 3)

print(f"Vector: {v}")
print(f"Shape: {v.shape}")  # (3,) — 1D
print(f"Column: {v_col.shape}")  # (3, 1)
```

### 1.2 Vector Addition & Scalar Multiplication

**Axioms** (these define a vector space):

\[
\mathbf{u} + \mathbf{v} = \begin{pmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{pmatrix}, \quad
c \mathbf{v} = \begin{pmatrix} cv_1 \\ cv_2 \\ \vdots \end{pmatrix}
\]

Geometric interpretation:
- Addition: **parallelogram rule** (tip-to-tail)
- Scalar: **stretches/shrinks** (or flips if negative)

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -1.0, 0.5])

print(f"a + b = {a + b}")          # [5.  1.  3.5]
print(f"a - b = {a - b}")          # [-3.  3.  2.5]
print(f"3a = {3 * a}")             # [3.  6.  9.]
print(f"-a = {-a}")                # [-1. -2. -3.]
print(f"Linear comb 2a - b = {2*a - b}")  # [-2.  5.  5.5]
```

### 1.3 Dot Product (Inner Product)

**Definition:**
\[
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = \mathbf{a}^T \mathbf{b}
\]

**Geometric formula:**
\[
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
\]

Where \( \theta \) is the angle between the vectors. So:
- \( \mathbf{a} \cdot \mathbf{b} > 0 \): vectors point in same general direction
- \( \mathbf{a} \cdot \mathbf{b} = 0 \): **orthogonal** (perpendicular)
- \( \mathbf{a} \cdot \mathbf{b} < 0 \): opposite directions

**Properties:**
- Commutative: \( \mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a} \)
- Bilinear: \( (\alpha \mathbf{a} + \beta \mathbf{b}) \cdot \mathbf{c} = \alpha(\mathbf{a}\cdot\mathbf{c}) + \beta(\mathbf{b}\cdot\mathbf{c}) \)
- Positive definite: \( \mathbf{v} \cdot \mathbf{v} \geq 0 \), equality iff \( \mathbf{v} = 0 \)

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -1.0, 2.0])

# Multiple ways to compute
dot1 = np.dot(a, b)
dot2 = a @ b
dot3 = np.sum(a * b)
print(f"Dot product: {dot1}")  # 1*4 + 2*(-1) + 3*2 = 8

# Compute angle
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
theta_deg = np.degrees(theta_rad)
print(f"Angle: {theta_deg:.2f}°")

# Projection of a onto b
proj_a_onto_b = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of a onto b: {proj_a_onto_b}")
```

### 1.4 Cross Product (3D only)

\[
\mathbf{a} \times \mathbf{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}
= \begin{pmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{pmatrix}
\]

**Properties:**
- Result is **perpendicular** to both \( \mathbf{a} \) and \( \mathbf{b} \)
- \( \|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\|\|\mathbf{b}\|\sin\theta \) (area of parallelogram)
- Anti-commutative: \( \mathbf{a} \times \mathbf{b} = -\mathbf{b} \times \mathbf{a} \)

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

cross = np.cross(a, b)
print(f"a × b = {cross}")  # [-3.  6. -3.]

# Verify perpendicularity
print(f"(a × b) · a = {np.dot(cross, a):.10f}")  # ~0
print(f"(a × b) · b = {np.dot(cross, b):.10f}")  # ~0

# Magnitude = area of parallelogram
area = np.linalg.norm(cross)
print(f"Parallelogram area: {area:.4f}")
```

### 1.5 Vector Norms

The **p-norm** (Minkowski norm):
\[
\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}
\]

| Norm | \(p\) | Formula | Intuition |
|------|--------|---------|-----------|
| \( \ell_1 \) (Manhattan) | 1 | \( \sum |v_i| \) | City-block distance |
| \( \ell_2 \) (Euclidean) | 2 | \( \sqrt{\sum v_i^2} \) | Straight-line distance |
| \( \ell_p \) | p | \( (\sum |v_i|^p)^{1/p} \) | General |
| \( \ell_\infty \) (Chebyshev) | ∞ | \( \max_i |v_i| \) | Maximum coordinate |

**ML relevance:**
- \( \ell_2 \): Euclidean distance, Ridge regularization, default KNN
- \( \ell_1 \): Lasso regularization, sparse solutions, robust to outliers
- \( \ell_\infty \): Minimax optimization, adversarial robustness

```python
v = np.array([3.0, -4.0, 0.0, 2.0])

l1 = np.linalg.norm(v, ord=1)
l2 = np.linalg.norm(v, ord=2)
l3 = np.linalg.norm(v, ord=3)
linf = np.linalg.norm(v, ord=np.inf)

print(f"v = {v}")
print(f"||v||_1 = {l1:.4f}")     # |3|+|-4|+|0|+|2| = 9
print(f"||v||_2 = {l2:.4f}")     # sqrt(9+16+0+4) = sqrt(29) ≈ 5.385
print(f"||v||_3 = {l3:.4f}")
print(f"||v||_∞ = {linf:.4f}")   # max(3, 4, 0, 2) = 4

# Unit vector (normalized)
v_unit = v / np.linalg.norm(v)
print(f"\nUnit vector: {v_unit.round(4)}")
print(f"||unit||_2 = {np.linalg.norm(v_unit):.10f}")  # = 1

# Norm inequalities: ||v||_∞ ≤ ||v||_2 ≤ ||v||_1 ≤ sqrt(n)*||v||_2
print(f"\nNorm ordering check: {linf:.4f} ≤ {l2:.4f} ≤ {l1:.4f}")
```

---

## 2. Matrices & Operations

### 2.1 Definition & Special Matrices

A matrix \( A \in \mathbb{R}^{m \times n} \) has \( m \) rows and \( n \) columns:
\[
A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}
\]

```python
import numpy as np

# Basic matrices
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# Special matrices
I3 = np.eye(3)                    # Identity
Z = np.zeros((3, 4))              # Zero
O = np.ones((2, 3))               # All-ones
D = np.diag([1, 2, 3])            # Diagonal
D_from_mat = np.diag(np.diag(A))  # Extract diagonal

# Random
A_rand = np.random.randn(4, 3)
A_sym = A_rand @ A_rand.T         # Symmetric positive semi-definite

print(f"Identity:\n{I3}")
print(f"\nDiagonal:\n{D}")
print(f"\nA is symmetric: {np.allclose(A_sym, A_sym.T)}")

# Trace = sum of diagonal
print(f"\ntrace(A) = {np.trace(A):.1f}")  # 1+5+9 = 15

# Matrix dimensions
print(f"A shape: {A.shape}")  # (3, 3)
print(f"A rank: {np.linalg.matrix_rank(A)}")
```

### 2.2 Matrix Operations

**Addition:** \( (A + B)_{ij} = A_{ij} + B_{ij} \) (same shape required)

**Scalar multiplication:** \( (cA)_{ij} = c \cdot A_{ij} \)

**Matrix multiplication:** \( (AB)_{ij} = \sum_k A_{ik} B_{kj} \) (A is \( m \times r \), B is \( r \times n \), result is \( m \times n \))

```python
A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)

# Addition
print(f"A + B:\n{A + B}")

# Matrix multiplication (NOT element-wise)
C = A @ B  # or np.matmul(A, B) or np.dot(A, B)
print(f"\nA @ B:\n{C}")
# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

# Element-wise (Hadamard product)
print(f"\nA * B (element-wise):\n{A * B}")

# NOT commutative in general
print(f"\nA@B == B@A? {np.allclose(A@B, B@A)}")

# Associative and distributive
C_rand = np.random.randn(2, 2)
print(f"(A@B)@C == A@(B@C)? {np.allclose((A@B)@C_rand, A@(B@C_rand))}")
```

### 2.3 Transpose

\[
(A^T)_{ij} = A_{ji}
\]

**Properties:**
- \( (A^T)^T = A \)
- \( (A + B)^T = A^T + B^T \)
- \( (AB)^T = B^T A^T \) (order reverses!)
- \( (ABC)^T = C^T B^T A^T \)

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3

print(f"A:\n{A}")
print(f"\nA.T:\n{A.T}")  # 3×2

# Verify: (AB)^T = B^T A^T
A2 = np.random.randn(3, 4)
B2 = np.random.randn(4, 2)
print(f"\n(A@B)^T == B^T @ A^T: {np.allclose((A2@B2).T, B2.T @ A2.T)}")

# Symmetric matrix: A = A^T
S = np.array([[1, 2, 3], [2, 5, 4], [3, 4, 9]])
print(f"\nS is symmetric: {np.allclose(S, S.T)}")
print(f"A^T @ A is always symmetric: {np.allclose(A2.T @ A2, (A2.T @ A2).T)}")
```

### 2.4 Determinant

**2×2:** \( \det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc \)

**3×3 (Laplace expansion):**
\[
\det(A) = a_{11}M_{11} - a_{12}M_{12} + a_{13}M_{13}
\]
where \( M_{ij} \) is the minor (determinant of submatrix with row \( i \), col \( j \) removed).

**Geometric meaning:** \( |\det(A)| \) = volume scaling factor of the linear transformation represented by \( A \).

**Key properties:**
- \( \det(I) = 1 \)
- \( \det(AB) = \det(A)\det(B) \)
- \( \det(A^T) = \det(A) \)
- \( \det(A^{-1}) = 1/\det(A) \)
- \( \det(cA) = c^n \det(A) \) for \( n \times n \) matrix
- \( \det(A) = 0 \) iff \( A \) is **singular** (not invertible, columns linearly dependent)

```python
A2x2 = np.array([[3, 1], [2, 4]])
det_2x2 = np.linalg.det(A2x2)
print(f"det([[3,1],[2,4]]) = {det_2x2:.1f}")  # 3*4 - 1*2 = 10

A3x3 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
det_3x3 = np.linalg.det(A3x3)
print(f"det(3x3) = {det_3x3:.6f}")  # ≈ 0 (singular!)

# Verify det(AB) = det(A)*det(B)
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)
print(f"\ndet(A)*det(B) = {np.linalg.det(A)*np.linalg.det(B):.6f}")
print(f"det(A@B)     = {np.linalg.det(A@B):.6f}")

# Determinant via eigenvalues: det = product of eigenvalues
eigenvals = np.linalg.eigvals(A)
print(f"\nProduct of eigenvalues = {np.prod(eigenvals).real:.6f}")
print(f"det(A)                 = {np.linalg.det(A):.6f}")
```

### 2.5 Matrix Inverse

For square \( A \): \( A^{-1} \) satisfies \( AA^{-1} = A^{-1}A = I \).

Exists iff \( \det(A) \neq 0 \) (i.e., \( A \) is non-singular / full rank).

**2×2 formula:**
\[
\begin{pmatrix} a & b \\ c & d \end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
\]

**Practical advice:** Never compute inverse explicitly! Use `np.linalg.solve(A, b)` for \( A^{-1}b \). The inverse is slower, less numerically stable, and loses structure (e.g., sparsity).

```python
A = np.array([[2, 1], [5, 3]], dtype=float)

A_inv = np.linalg.inv(A)
print(f"A^(-1):\n{A_inv}")

# Verify
print(f"\nA @ A^(-1):\n{np.round(A @ A_inv, 10)}")  # Should be identity

# Solving linear system
b = np.array([4.0, 7.0])

# SLOW & numerically unstable:  x = np.linalg.inv(A) @ b
# FAST & stable:                x = np.linalg.solve(A, b)
x = np.linalg.solve(A, b)
print(f"\nSolution x = {x}")
print(f"Verification A@x = {A @ x} (should be {b})")

# Inverse of product
A2 = np.random.randn(3, 3)
B2 = np.random.randn(3, 3)
print(f"\n(AB)^(-1) = B^(-1) A^(-1): "
      f"{np.allclose(np.linalg.inv(A2 @ B2), np.linalg.inv(B2) @ np.linalg.inv(A2))}")
```

### 2.6 Trace

\[
\text{tr}(A) = \sum_{i=1}^n A_{ii}
\]

**Properties:**
- \( \text{tr}(A + B) = \text{tr}(A) + \text{tr}(B) \)
- \( \text{tr}(AB) = \text{tr}(BA) \) (cyclic: even \( \text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB) \))
- \( \text{tr}(A) = \sum_i \lambda_i \) (sum of eigenvalues)

```python
A = np.random.randn(4, 4)
B = np.random.randn(4, 4)

print(f"tr(A+B) = {np.trace(A+B):.6f}")
print(f"tr(A)+tr(B) = {np.trace(A)+np.trace(B):.6f}")

print(f"\ntr(AB) = {np.trace(A@B):.6f}")
print(f"tr(BA) = {np.trace(B@A):.6f}")  # Same!

# Frobenius norm via trace: ||A||_F = sqrt(tr(A^T A))
frob = np.linalg.norm(A, 'fro')
frob_trace = np.sqrt(np.trace(A.T @ A))
print(f"\n||A||_F = {frob:.6f}")
print(f"sqrt(tr(A^T A)) = {frob_trace:.6f}")
```

### 2.7 Matrix Rank

The **rank** of \( A \) is the dimension of its column space (= row space).
- Full column rank: \( \text{rank}(A) = n \) → columns linearly independent
- Full row rank: \( \text{rank}(A) = m \)
- **Rank-nullity theorem:** \( \text{rank}(A) + \text{nullity}(A) = n \)

```python
# Full rank matrix
A_full = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(f"Rank of I_3: {np.linalg.matrix_rank(A_full)}")  # 3

# Rank-deficient
A_rank2 = np.array([[1, 2, 3], [4, 5, 6], [5, 7, 9]])  # row3 = row1 + row2
print(f"Rank of rank-deficient: {np.linalg.matrix_rank(A_rank2)}")  # 2

# Using SVD to find rank (numerically stable)
U, S, Vt = np.linalg.svd(A_rank2)
threshold = 1e-10
print(f"Singular values: {S.round(4)}")
print(f"Numerical rank (non-zero SVs): {np.sum(S > threshold)}")
```

---

## 3. Linear Transformations

### 3.1 Definition & Representation

A function \( T: \mathbb{R}^n \to \mathbb{R}^m \) is **linear** if:
1. \( T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) \) (additivity)
2. \( T(c\mathbf{v}) = cT(\mathbf{v}) \) (homogeneity)

**Matrix representation:** Every linear map is represented by a matrix. If \( \{e_1, \ldots, e_n\} \) is a basis, then:
\[
A = [T(e_1) \mid T(e_2) \mid \cdots \mid T(e_n)]
\]

### 3.2 Geometric Transformations in 2D

```python
import numpy as np
import matplotlib.pyplot as plt

def apply_transform(T, points):
    """Apply 2D linear transform to array of column points."""
    return T @ points

# Unit square corners
square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]], dtype=float)

# --- Rotation by angle θ ---
theta = np.pi / 4  # 45 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print(f"Rotation matrix:\n{R.round(4)}")
print(f"det(R) = {np.linalg.det(R):.4f}")  # Always 1 for rotations

# --- Scaling ---
S = np.array([[2, 0], [0, 0.5]])  # Stretch x by 2, compress y by 2
print(f"\nScaling matrix:\n{S}")
print(f"det(S) = {np.linalg.det(S):.4f}")  # Product of scale factors

# --- Reflection across y-axis ---
Ry = np.array([[-1, 0], [0, 1]])
print(f"\nReflection across y-axis, det = {np.linalg.det(Ry):.1f}")  # -1

# --- Shear ---
H = np.array([[1, 0.5], [0, 1]])  # Horizontal shear
print(f"\nShear matrix:\n{H}")
print(f"det(H) = {np.linalg.det(H):.4f}")  # Always 1 (area-preserving)

# --- Projection onto x-axis ---
P = np.array([[1, 0], [0, 0]])
print(f"\nProjection onto x-axis, det = {np.linalg.det(P):.1f}")  # 0 (singular)

# Apply transforms
transforms = {'Rotation 45°': R, 'Scale(2, 0.5)': S, 'Shear': H}
for name, T in transforms.items():
    transformed = apply_transform(T, square)
    print(f"\n{name}: area scales by |det| = {abs(np.linalg.det(T)):.4f}")
```

### 3.3 Composition of Transforms

\( T_2 \circ T_1 \) is represented by matrix product \( A_2 A_1 \) (apply \( A_1 \) first!).

```python
# Rotate then scale ≠ Scale then rotate (in general)
R45 = np.array([[0, -1], [1, 0]])   # 90° rotation
S = np.array([[2, 0], [0, 1]])      # Scale x by 2

v = np.array([1.0, 0.0])

# Rotate then scale
RS = S @ R45  # Apply R45 first, then S
print(f"Scale(Rotate(v)) = {RS @ v}")

# Scale then rotate
SR = R45 @ S  # Apply S first, then R45
print(f"Rotate(Scale(v)) = {SR @ v}")

print(f"RS == SR? {np.allclose(RS, SR)}")  # Usually False
```

### 3.4 Null Space & Image

- **Image (column space):** \( \text{Im}(A) = \{Ax : x \in \mathbb{R}^n\} \)
- **Null space (kernel):** \( \ker(A) = \{x : Ax = 0\} \)

```python
from scipy.linalg import null_space

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [5, 7, 9]], dtype=float)  # rank 2

# Null space basis
null = null_space(A)
print(f"Null space dimension: {null.shape[1]}")
print(f"Null space basis:\n{null}")

# Verify: A @ null ≈ 0
print(f"A @ null_vec ≈ 0: {np.allclose(A @ null, 0, atol=1e-10)}")

# Column space (image) — columns of U with non-zero singular values
U, S, Vt = np.linalg.svd(A, full_matrices=True)
r = np.sum(S > 1e-10)
col_space = U[:, :r]
print(f"\nColumn space dimension: {r}")
```

---

## 4. Systems of Linear Equations

### 4.1 Forms and Solutions

\[
Ax = b \quad (A \in \mathbb{R}^{m \times n},\; x \in \mathbb{R}^n,\; b \in \mathbb{R}^m)
\]

| Case | Condition | # Solutions |
|------|-----------|------------|
| Unique | \( m = n \), \( A \) full rank | 1 |
| No solution | \( b \notin \text{Im}(A) \) | 0 (inconsistent) |
| Infinitely many | \( \ker(A) \neq \{0\} \), \( b \in \text{Im}(A) \) | ∞ |
| Overdetermined | \( m > n \) | Usually 0 (use least squares) |
| Underdetermined | \( m < n \) | ∞ (minimum norm solution) |

### 4.2 Gaussian Elimination

**Algorithm:** Row-reduce augmented matrix \( [A | b] \) to row echelon form using elementary row operations:
1. \( R_i \leftrightarrow R_j \): Row swap
2. \( R_i \leftarrow c R_i \): Scale row by constant
3. \( R_i \leftarrow R_i + c R_j \): Add multiple of one row to another

```python
import numpy as np
from scipy.linalg import lu

# Exact system (unique solution)
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

# Best method: np.linalg.solve
x = np.linalg.solve(A, b)
print(f"Solution: {x}")
print(f"Verification: {np.allclose(A @ x, b)}")

# Manual Gaussian elimination
def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])

    for col in range(n):
        # Partial pivoting: find max element in column
        max_row = np.argmax(np.abs(Ab[col:, col])) + col
        Ab[[col, max_row]] = Ab[[max_row, col]]  # Swap rows

        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row] -= factor * Ab[col]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:]) / Ab[i, i]
    return x

x_manual = gaussian_elimination(A, b)
print(f"\nManual Gaussian: {x_manual}")
```

### 4.3 LU Decomposition

\( A = PLU \) where \( P \) = permutation, \( L \) = lower triangular, \( U \) = upper triangular.

**Efficient for solving multiple systems** with same \( A \) but different \( b \).

```python
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[2, 1, 5],
              [4, 4, -4],
              [1, 3, 1]], dtype=float)

# LU decomposition: A = P L U
P, L, U = lu(A)
print(f"P (permutation):\n{P}")
print(f"\nL (lower triangular):\n{L.round(4)}")
print(f"\nU (upper triangular):\n{U.round(4)}")
print(f"\nP@L@U == A: {np.allclose(P @ L @ U, A)}")

# Solve Ax = b efficiently using LU
lu_piv = lu_factor(A)  # Compute LU once

# Solve for multiple right-hand sides
b1 = np.array([1.0, 2.0, 3.0])
b2 = np.array([4.0, 5.0, 6.0])
x1 = lu_solve(lu_piv, b1)
x2 = lu_solve(lu_piv, b2)
print(f"\nSolution for b1: {x1}")
print(f"Solution for b2: {x2}")
```

### 4.4 Least Squares (Overdetermined Systems)

When \( m > n \) (more equations than unknowns), minimize \( \|Ax - b\|^2 \):
\[
\hat{x} = (A^T A)^{-1} A^T b
\]

This is the **normal equation** — used in linear regression!

```python
# Overdetermined system: more rows than columns
A_over = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], dtype=float)
b_over = np.array([1.9, 3.1, 4.0, 4.9, 6.1])

# Method 1: Normal equation (can be numerically unstable)
x_normal = np.linalg.inv(A_over.T @ A_over) @ A_over.T @ b_over

# Method 2: NumPy least squares (uses QR — more stable)
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A_over, b_over, rcond=None)

print(f"Normal equation: {x_normal.round(4)}")
print(f"lstsq solution:  {x_lstsq.round(4)}")
print(f"Residual ||Ax - b||²: {np.linalg.norm(A_over @ x_lstsq - b_over)**2:.6f}")

# Geometric interpretation: x_lstsq gives point in column space closest to b
b_hat = A_over @ x_lstsq  # Projection of b onto col space of A
residual = b_over - b_hat
print(f"\nResidual ⊥ column space: {np.allclose(A_over.T @ residual, 0, atol=1e-10)}")
```

---

## 5. Vector Spaces, Subspaces & Basis

### 5.1 Vector Space Axioms

A set \( V \) with operations \( + \) and \( \cdot \) is a vector space if:
1. Closure: \( u + v \in V \), \( cv \in V \)
2. Commutativity: \( u + v = v + u \)
3. Associativity: \( (u+v)+w = u+(v+w) \)
4. Zero vector: \( \exists \, 0 \in V: v + 0 = v \)
5. Additive inverse: \( \exists\, -v: v + (-v) = 0 \)
6. Scalar distributivity: \( c(u+v) = cu + cv \), \( (c+d)v = cv + dv \)
7. Scalar associativity: \( c(dv) = (cd)v \)
8. Identity: \( 1 \cdot v = v \)

### 5.2 Subspaces

\( W \subseteq V \) is a subspace if:
1. \( 0 \in W \)
2. Closed under addition: \( u, v \in W \Rightarrow u + v \in W \)
3. Closed under scalar multiplication: \( v \in W, c \in \mathbb{R} \Rightarrow cv \in W \)

**Four fundamental subspaces of matrix \( A \in \mathbb{R}^{m \times n} \):**

| Subspace | Definition | Dimension |
|----------|-----------|-----------|
| Column space \( C(A) \) | \( \{Ax : x \in \mathbb{R}^n\} \) | \( r = \text{rank}(A) \) |
| Row space \( C(A^T) \) | \( \{A^T y : y \in \mathbb{R}^m\} \) | \( r \) |
| Null space \( N(A) \) | \( \{x : Ax = 0\} \) | \( n - r \) |
| Left null space \( N(A^T) \) | \( \{y : A^T y = 0\} \) | \( m - r \) |

```python
from scipy.linalg import null_space
import numpy as np

A = np.array([[1, 0, -1, 0],
              [0, 1, 0, -1],
              [1, 1, -1, -1]], dtype=float)

m, n = A.shape
r = np.linalg.matrix_rank(A)
print(f"A shape: {m}×{n}, rank r={r}")
print(f"Column space dim: {r}")
print(f"Null space dim: {n - r}")
print(f"Left null space dim: {m - r}")

# Null space
N = null_space(A)
print(f"\nNull space basis ({N.shape[1]} vectors):\n{N.round(4)}")

# Verify orthogonality: Column space ⊥ Left null space
U, S, Vt = np.linalg.svd(A, full_matrices=True)
col_space = U[:, :r]         # First r columns of U
left_null = U[:, r:]         # Last m-r columns of U
print(f"\nColumn space ⊥ Left null space: "
      f"{np.allclose(col_space.T @ left_null, 0, atol=1e-10)}")
```

### 5.3 Linear Independence, Span & Basis

**Linear independence:** \( v_1, \ldots, v_k \) are linearly independent if:
\[
c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0 \implies c_1 = c_2 = \cdots = c_k = 0
\]

**Span:** \( \text{span}(v_1, \ldots, v_k) = \{c_1 v_1 + \cdots + c_k v_k : c_i \in \mathbb{R}\} \)

**Basis:** A linearly independent spanning set. Dimension = number of basis vectors.

```python
# Check linear independence via rank
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
v4 = np.array([1, 1, 0])  # v4 = v1 + v2 (dependent!)

# Independent set
V_ind = np.vstack([v1, v2, v3])
print(f"{{v1,v2,v3}} are independent: {np.linalg.matrix_rank(V_ind) == 3}")

# Dependent set
V_dep = np.vstack([v1, v2, v4])
print(f"{{v1,v2,v4}} are independent: {np.linalg.matrix_rank(V_dep) == 3}")

# Find basis from a set of vectors
vectors = np.array([[1, 2, 3],
                    [2, 4, 6],    # = 2 × row 1 (dependent!)
                    [0, 1, 2],
                    [1, 3, 5]], dtype=float)

# Pivots in row echelon form identify basis vectors
# Use QR with pivoting
from scipy.linalg import qr

Q, R, P = qr(vectors.T, pivoting=True)  # P: column permutation
rank = np.sum(np.abs(np.diag(R)) > 1e-10)
basis_indices = P[:rank]
print(f"\nBasis vectors (rows {basis_indices}): shape = {vectors[basis_indices].shape}")
```

---

## 6. Orthogonality & Gram-Schmidt

### 6.1 Orthogonal & Orthonormal Sets

**Orthogonal:** \( \langle v_i, v_j \rangle = 0 \) for \( i \neq j \)

**Orthonormal:** Orthogonal + \( \|v_i\| = 1 \) for all \( i \)

**Orthogonal matrix** \( Q \): \( Q^T Q = Q Q^T = I \), so \( Q^{-1} = Q^T \)

**Properties of orthogonal matrices:**
- \( |\det(Q)| = 1 \) (either +1 or -1)
- Preserve lengths: \( \|Qv\| = \|v\| \)
- Preserve angles: \( \langle Qu, Qv \rangle = \langle u, v \rangle \)
- Represent rotations (\(\det = 1\)) or reflections (\(\det = -1\))

```python
# Verify orthogonal matrix properties
Q = np.array([[1/np.sqrt(2), -1/np.sqrt(2)],
              [1/np.sqrt(2),  1/np.sqrt(2)]])

print(f"Q^T Q = I: {np.allclose(Q.T @ Q, np.eye(2))}")
print(f"Q Q^T = I: {np.allclose(Q @ Q.T, np.eye(2))}")
print(f"det(Q) = {np.linalg.det(Q):.4f}")  # +1 (rotation)
print(f"||Qv|| = ||v||: ", end='')
v = np.array([3.0, 4.0])
print(f"{np.isclose(np.linalg.norm(Q @ v), np.linalg.norm(v))}")
```

### 6.2 Orthogonal Projection

**Projection of \( b \) onto vector \( a \):**
\[
\text{proj}_a b = \frac{a^T b}{a^T a} a
\]

**Projection matrix** onto subspace spanned by columns of \( A \):
\[
P = A(A^T A)^{-1} A^T
\]

Properties of projection matrices:
- \( P^2 = P \) (idempotent)
- \( P^T = P \) (symmetric)
- Eigenvalues are 0 or 1

```python
# Projection onto subspace
A = np.array([[1, 0], [1, 1], [1, 1]], dtype=float)  # 3D → span of 2 vectors

# Projection matrix
P = A @ np.linalg.inv(A.T @ A) @ A.T

# Verify idempotent and symmetric
print(f"P² = P: {np.allclose(P @ P, P)}")
print(f"P^T = P: {np.allclose(P, P.T)}")

# Project b onto column space of A
b = np.array([3.0, 2.0, 1.0])
b_proj = P @ b
residual = b - b_proj

print(f"\nOriginal b: {b}")
print(f"Projected b: {b_proj.round(4)}")
print(f"Residual ⊥ A: {np.allclose(A.T @ residual, 0, atol=1e-10)}")
```

### 6.3 Gram-Schmidt Orthogonalization

**Convert any basis \( \{a_1, \ldots, a_n\} \) to orthonormal basis \( \{q_1, \ldots, q_n\} \):**

\[
u_1 = a_1, \quad q_1 = \frac{u_1}{\|u_1\|}
\]
\[
u_k = a_k - \sum_{j=1}^{k-1} (a_k \cdot q_j) q_j, \quad q_k = \frac{u_k}{\|u_k\|}
\]

```python
def gram_schmidt(A):
    """Modified Gram-Schmidt orthogonalization (more numerically stable)."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].astype(float)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

# Test
A = np.array([[4, 3, -2],
              [0, -1, 2],
              [3, 2, 1]], dtype=float)

Q, R = gram_schmidt(A)
print(f"Q orthonormal: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"A = QR: {np.allclose(Q @ R, A)}")
print(f"\nR (upper triangular):\n{R.round(4)}")

# NumPy has built-in QR
Q_np, R_np = np.linalg.qr(A)
print(f"\nNumPy QR recovers A: {np.allclose(Q_np @ R_np, A)}")
```

---

## 7. Eigenvalues & Eigenvectors

### 7.1 Definition & Characteristic Equation

For square \( A \in \mathbb{R}^{n \times n} \), if:
\[
Av = \lambda v \quad (v \neq 0)
\]

Then \( \lambda \) is an **eigenvalue** and \( v \) is the corresponding **eigenvector**.

**Characteristic polynomial:**
\[
\det(A - \lambda I) = 0
\]

**Geometric meaning:** Eigenvectors are directions that the transformation **only stretches** (by factor \( \lambda \)), not rotates. For a symmetric matrix (e.g., covariance), eigenvalues give the variance along principal directions; the eigenvector with largest \( \lambda \) points in the direction of maximum spread. This is why PCA uses eigendecomposition of the covariance matrix.

**Intuition:** Think of \( A \) as a transformation. Most vectors get rotated *and* stretched. Eigenvectors are special: they only get stretched. The eigenvalue \( \lambda \) tells you the stretch factor (and direction if \( \lambda < 0 \)).

```python
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")   # [5, 2]
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Verify Av = λv
for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
    Av = A @ v
    lambda_v = lam * v
    print(f"\nλ{i+1} = {lam}: Av = {Av.round(4)}, λv = {lambda_v.round(4)}, "
          f"Match: {np.allclose(Av, lambda_v)}")

# Key properties
print(f"\ntrace(A) = {np.trace(A)} = sum(eigenvalues) = {eigenvalues.sum():.4f}")
print(f"det(A) = {np.linalg.det(A):.4f} = prod(eigenvalues) = {eigenvalues.prod():.4f}")
```

### 7.2 Eigenvalue Decomposition (Diagonalization)

For diagonalizable \( A \) (linearly independent eigenvectors):
\[
A = Q\Lambda Q^{-1}
\]

Where \( Q = [v_1 | v_2 | \cdots | v_n] \) (columns = eigenvectors) and \( \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n) \).

**Applications:**
- **Matrix powers:** \( A^k = Q\Lambda^k Q^{-1} \) (fast computation)
- **Matrix exponential:** \( e^A = Q e^\Lambda Q^{-1} \)
- **PCA:** Eigendecomposition of covariance matrix

```python
A = np.array([[4, 1], [2, 3]], dtype=float)
eigenvalues, Q = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)
Q_inv = np.linalg.inv(Q)

# A = Q Λ Q^{-1}
A_reconstructed = Q @ Lambda @ Q_inv
print(f"A = Q Λ Q^(-1): {np.allclose(A, A_reconstructed)}")

# Fast matrix power A^10
A_power10 = Q @ np.diag(eigenvalues**10) @ Q_inv
print(f"A^10 direct: {np.round(np.linalg.matrix_power(A.astype(int), 10), 0)}")
print(f"A^10 via eig: {np.round(A_power10.real, 0)}")

# Symmetric matrices: eigenvalues real, eigenvectors orthogonal
S = np.array([[3, 1, 0], [1, 2, 1], [0, 1, 4]], dtype=float)
lam_s, Q_s = np.linalg.eigh(S)  # Use eigh for symmetric (faster, guaranteed real)
print(f"\nSymmetric: eigenvalues real: {np.isrealobj(lam_s)}")
print(f"Eigenvectors orthogonal: {np.allclose(Q_s.T @ Q_s, np.eye(3))}")
```

### 7.3 Spectral Theorem

**Theorem:** A real symmetric matrix \( A \) has:
1. All **real** eigenvalues
2. **Orthogonal** eigenvectors (even for repeated eigenvalues)
3. Spectral decomposition: \( A = Q\Lambda Q^T \) (since \( Q^{-1} = Q^T \))

This is fundamental for PCA and covariance matrix analysis!

```python
# Spectral decomposition of symmetric matrix
S = np.array([[5, 2, 1],
              [2, 3, 0],
              [1, 0, 4]], dtype=float)

lam, Q = np.linalg.eigh(S)  # Returns eigenvalues sorted ascending
print(f"Eigenvalues: {lam.round(4)}")
print(f"Eigenvectors orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")

# Spectral decomposition: S = sum(λ_i * v_i * v_i^T)
S_reconstructed = sum(lam[i] * np.outer(Q[:, i], Q[:, i]) for i in range(3))
print(f"\nSpectral decomposition: {np.allclose(S, S_reconstructed)}")

# Low-rank approximation via spectral decomposition
# (equivalent to PCA for covariance matrices)
k = 2  # Keep 2 largest eigenvalues
idx = np.argsort(lam)[::-1]
lam_sorted = lam[idx]
Q_sorted = Q[:, idx]

S_approx = sum(lam_sorted[i] * np.outer(Q_sorted[:, i], Q_sorted[:, i]) for i in range(k))
print(f"Rank-{k} approximation error: {np.linalg.norm(S - S_approx, 'fro'):.4f}")
```

---

## 8. Singular Value Decomposition (SVD)

### 8.1 The SVD Theorem

**For any** \( A \in \mathbb{R}^{m \times n} \) (not necessarily square):
\[
A = U \Sigma V^T
\]

Where:
- \( U \in \mathbb{R}^{m \times m} \): left singular vectors (columns are eigenvectors of \( AA^T \))
- \( \Sigma \in \mathbb{R}^{m \times n} \): diagonal with singular values \( \sigma_1 \geq \sigma_2 \geq \cdots \geq 0 \)
- \( V \in \mathbb{R}^{n \times n} \): right singular vectors (columns are eigenvectors of \( A^T A \))

**Connection to eigendecomposition:**
- \( A^T A = V \Sigma^T \Sigma V^T \): eigenvalues are \( \sigma_i^2 \)
- \( A A^T = U \Sigma \Sigma^T U^T \): eigenvalues are \( \sigma_i^2 \)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)  # 4×3

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)
print(f"A shape: {A.shape}")
print(f"U shape: {U.shape}  (left singular vectors)")
print(f"S (singular values): {S.round(4)}")
print(f"Vt shape: {Vt.shape} (right singular vectors)")

# Reconstruct A
Sigma = np.zeros_like(A, dtype=float)
Sigma[:min(A.shape), :min(A.shape)] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt
print(f"\nA = UΣVᵀ: {np.allclose(A, A_reconstructed)}")

# Thin/Economy SVD (more efficient, S shape matches)
U_thin, S_thin, Vt_thin = np.linalg.svd(A, full_matrices=False)
print(f"\nThin SVD:")
print(f"  U_thin: {U_thin.shape}, S: {S_thin.shape}, Vt: {Vt_thin.shape}")
A_thin_recon = U_thin @ np.diag(S_thin) @ Vt_thin
print(f"  Reconstruction: {np.allclose(A, A_thin_recon)}")

# Verify: columns of U orthonormal, columns of V orthonormal
print(f"\nU columns orthonormal: {np.allclose(U.T @ U, np.eye(U.shape[0]))}")
print(f"V columns orthonormal: {np.allclose(Vt.T @ Vt, np.eye(Vt.shape[0]))}")
```

### 8.2 Low-Rank Approximation (Eckart-Young Theorem)

The **best rank-\(k\) approximation** of \( A \) (in Frobenius and spectral norm) is:
\[
A_k = \sum_{i=1}^k \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T
\]

**Error:**
\[
\|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2, \quad \|A - A_k\|_2 = \sigma_{k+1}
\]

```python
# Low-rank approximation
np.random.seed(42)
# Create a "true" rank-3 matrix plus noise
U_true = np.random.randn(100, 3)
V_true = np.random.randn(50, 3)
A_signal = U_true @ V_true.T
A_noisy = A_signal + 0.5 * np.random.randn(100, 50)

# SVD of noisy matrix
U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)

# Errors at different ranks
print("Rank | Frobenius Error | % Variance Explained")
print("-" * 45)
total_var = np.sum(S**2)
for k in [1, 2, 3, 5, 10, 20]:
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A_noisy - A_k, 'fro')
    var_explained = np.sum(S[:k]**2) / total_var
    print(f"  {k:2d} | {error:15.4f} | {var_explained:20.4%}")

# Image compression example
from PIL import Image
import requests
import io

# Generate a synthetic "image" (grayscale)
img_data = np.random.rand(256, 256) * 255
img_data = img_data.astype(np.float64)

U_img, S_img, Vt_img = np.linalg.svd(img_data, full_matrices=False)

for k in [5, 20, 50, 100]:
    compressed = U_img[:, :k] @ np.diag(S_img[:k]) @ Vt_img[:k, :]
    original_size = img_data.size
    compressed_size = U_img[:, :k].size + S_img[:k].size + Vt_img[:k, :].size
    ratio = original_size / compressed_size
    mse = np.mean((img_data - compressed)**2)
    print(f"k={k:3d}: Compression {ratio:.1f}x, MSE={mse:.2f}")
```

### 8.3 SVD Applications: Intuition & Geometry

**Why SVD is universal:** Unlike eigendecomposition (requires square and diagonalizable), SVD works on *any* matrix. \( A = U\Sigma V^T \) decomposes \( A \) into:
1. **\( V^T \)**: rotate/reflect in input space to align with "principal directions"
2. **\( \Sigma \)**: scale along those directions (singular values = "energy" or importance)
3. **\( U \)**: rotate/reflect in output space

**Low-rank approximation (Eckart–Young):** The best rank-\(k\) approximation in Frobenius norm keeps the top-\(k\) singular values and truncates the rest. Geometrically: project onto the \(k\) directions of maximum variance. This underpins PCA, matrix completion, and denoising.

**Worked example (PCA via SVD):** For centered data \( X \in \mathbb{R}^{n \times d} \), the covariance is \( C = \frac{1}{n-1}X^TX \). The principal components are the right singular vectors of \( X \) (columns of \( V \)), and \( \sigma_i^2/(n-1) \) gives the variance along the \(i\)-th PC. You can compute PCA by SVD of \( X \) directly — numerically more stable than eigendecomposition of \( C \) when \( n \ll d \).

### 8.4 SVD Applications Summary Table

| Application | How SVD is Used |
|------------|-----------------|
| PCA | Right singular vectors \( V \) = principal components |
| Linear Regression | \( \hat{x} = V\Sigma^+ U^T b \) (Moore-Penrose pseudoinverse) |
| Image compression | Truncated SVD |
| Collaborative filtering | Matrix factorization |
| Latent Semantic Analysis | SVD of term-document matrix |
| Condition number | \( \sigma_{max} / \sigma_{min} \) |
| Pseudoinverse | \( A^+ = V\Sigma^+ U^T \) |

```python
# Moore-Penrose pseudoinverse via SVD
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)  # 3×2 (overdetermined)

# Pseudoinverse: A+ = V Σ+ U^T
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_pinv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_pinv @ U.T

# Verify against NumPy
A_pinv_np = np.linalg.pinv(A)
print(f"A_pinv matches numpy: {np.allclose(A_pinv, A_pinv_np)}")

# Pseudoinverse gives least-squares solution
b = np.array([1.0, 2.0, 3.0])
x_lstsq = A_pinv @ b
print(f"Least squares solution: {x_lstsq}")
print(f"||Ax - b||: {np.linalg.norm(A @ x_lstsq - b):.6f}")
```

---

## 9. Matrix Decompositions: LU, QR, Cholesky

### 9.1 QR Decomposition

\( A = QR \) where \( Q \) orthogonal, \( R \) upper triangular.

**Applications:**
- Solving linear systems (more stable than LU)
- Gram-Schmidt gives QR
- Eigenvalue algorithms (QR iteration)
- Least squares

```python
from scipy.linalg import qr as scipy_qr

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]], dtype=float)

Q, R = np.linalg.qr(A)
print(f"Q (orthogonal):\n{Q.round(4)}")
print(f"\nR (upper triangular):\n{R.round(4)}")
print(f"\nQ orthogonal: {np.allclose(Q.T @ Q, np.eye(3))}")
print(f"A = QR: {np.allclose(Q @ R, A)}")

# Solve Ax = b using QR
# QRx = b → Rx = Q^T b (back substitution)
b = np.array([1.0, 2.0, 3.0])
QTb = Q.T @ b
from scipy.linalg import solve_triangular
x_qr = solve_triangular(R, QTb)
print(f"\nSolution via QR: {x_qr}")
print(f"Verify: {np.allclose(A @ x_qr, b)}")
```

### 9.2 Cholesky Decomposition

For **symmetric positive definite** (SPD) matrix \( A \):
\[
A = LL^T
\]

Where \( L \) is lower triangular with positive diagonal entries.

**Advantages:**
- 2x faster than LU for SPD matrices
- Used in sampling from multivariate Gaussian
- Kalman filtering, Gaussian processes

```python
# Create SPD matrix
np.random.seed(42)
M = np.random.randn(4, 4)
A_spd = M.T @ M + 4 * np.eye(4)  # Guaranteed SPD

# Cholesky
L = np.linalg.cholesky(A_spd)
print(f"L (lower triangular):\n{L.round(4)}")
print(f"\nA = L L^T: {np.allclose(L @ L.T, A_spd)}")
print(f"All diagonal positive: {np.all(np.diag(L) > 0)}")

# Application: Sample from N(μ, Σ)
def sample_multivariate_gaussian(mu, Sigma, n_samples):
    """Sample using Cholesky — more stable than direct."""
    L = np.linalg.cholesky(Sigma)
    z = np.random.randn(len(mu), n_samples)
    return mu[:, np.newaxis] + L @ z

mu = np.array([1.0, 2.0, 3.0])
Sigma = np.array([[1.0, 0.5, 0.2],
                  [0.5, 2.0, 0.3],
                  [0.2, 0.3, 1.5]])

samples = sample_multivariate_gaussian(mu, Sigma, 10000)
print(f"\nSample mean ≈ μ: {samples.mean(axis=1).round(2)}")
print(f"Sample cov ≈ Σ:\n{np.cov(samples).round(2)}")
```

### 9.3 Comparison of Decompositions

| Decomposition | Formula | Requirement | Main Use |
|--------------|---------|-------------|---------|
| LU | \( A = PLU \) | Any square A | General linear systems |
| QR | \( A = QR \) | Any A | Least squares, eigenvalues |
| Eigendecomp | \( A = Q\Lambda Q^{-1} \) | Square, diagonalizable | Quadratic forms, matrix powers |
| SVD | \( A = U\Sigma V^T \) | Any A | Best! Universal tool |
| Cholesky | \( A = LL^T \) | Symmetric PD | SPD systems, sampling |

---

## 10. Positive Definite Matrices

### 10.1 Definition & Tests

A symmetric matrix \( A \) is **positive definite** (PD) if:
\[
x^T A x > 0 \quad \forall x \neq 0
\]

**Equivalently:**
1. All eigenvalues \( > 0 \)
2. All leading principal minors \( > 0 \) (Sylvester's criterion)
3. Cholesky decomposition exists
4. \( A = B^T B \) for some full-rank \( B \)

| Type | Condition | \( x^T A x \) |
|------|-----------|--------------|
| Positive Definite (PD) | All \( \lambda_i > 0 \) | \( > 0 \) |
| Positive Semi-Definite (PSD) | All \( \lambda_i \geq 0 \) | \( \geq 0 \) |
| Negative Definite (ND) | All \( \lambda_i < 0 \) | \( < 0 \) |
| Indefinite | Mixed signs | Both signs |

```python
def classify_matrix(A):
    """Classify matrix by definiteness."""
    # Check symmetric
    if not np.allclose(A, A.T):
        return "Not symmetric"

    lam = np.linalg.eigvalsh(A)

    if np.all(lam > 1e-10):
        return "Positive Definite"
    elif np.all(lam >= -1e-10):
        return "Positive Semi-Definite"
    elif np.all(lam < -1e-10):
        return "Negative Definite"
    elif np.all(lam <= 1e-10):
        return "Negative Semi-Definite"
    else:
        return "Indefinite"

A_pd = np.array([[4, 2], [2, 3]])         # PD
A_psd = np.array([[1, 1], [1, 1]])        # PSD (rank 1)
A_nd = np.array([[-2, 0], [0, -3]])       # ND
A_indef = np.array([[1, 0], [0, -1]])     # Indefinite (saddle point)

for name, M in [('PD', A_pd), ('PSD', A_psd), ('ND', A_nd), ('Indef', A_indef)]:
    eigs = np.linalg.eigvalsh(M)
    print(f"{name}: {classify_matrix(M)}, eigenvalues={eigs.round(4)}")

# Covariance matrices are always PSD
X = np.random.randn(100, 5)
cov = np.cov(X.T)
print(f"\nCovariance matrix: {classify_matrix(cov)}")
```

### 10.2 ML Relevance

```python
# Hessian definiteness → nature of critical point
# Positive Definite Hessian → local minimum
# Negative Definite Hessian → local maximum
# Indefinite Hessian → saddle point

# For f(x,y) = x^2 + y^2 (bowl shape - convex)
H_convex = np.array([[2, 0], [0, 2]])  # Hessian at any point
print(f"f=x²+y² Hessian: {classify_matrix(H_convex)}")  # PD → minimum

# For f(x,y) = x^2 - y^2 (saddle)
H_saddle = np.array([[2, 0], [0, -2]])  # Hessian at origin
print(f"f=x²-y² Hessian: {classify_matrix(H_saddle)}")  # Indefinite → saddle

# Ridge regression: X^T X + λI is always PD (even if X^T X is PSD)
X_reg = np.random.randn(10, 20)  # Underdetermined: 10 samples, 20 features
XTX = X_reg.T @ X_reg            # PSD (rank ≤ 10)
lambda_reg = 1.0
XTX_reg = XTX + lambda_reg * np.eye(20)  # Add ridge → PD
print(f"\nX^T X:           {classify_matrix(XTX)}")
print(f"X^T X + λI:      {classify_matrix(XTX_reg)}")
```

---

## 11. Norms & Distances

### 11.1 Matrix Norms

**Frobenius norm** (entry-wise L2):
\[
\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^TA)} = \sqrt{\sum_i \sigma_i^2}
\]

**Spectral norm** (induced L2 matrix norm):
\[
\|A\|_2 = \sigma_{\max}(A)
\]

**Nuclear norm** (sum of singular values):
\[
\|A\|_* = \sum_i \sigma_i
\]

```python
A = np.random.randn(4, 5)
U, S, Vt = np.linalg.svd(A)

frob = np.linalg.norm(A, 'fro')
spectral = np.linalg.norm(A, 2)  # = max singular value
nuclear = np.sum(S)

print(f"Frobenius norm: {frob:.4f} = sqrt(sum σ²) = {np.sqrt(np.sum(S**2)):.4f}")
print(f"Spectral norm:  {spectral:.4f} = σ_max = {S[0]:.4f}")
print(f"Nuclear norm:   {nuclear:.4f} = sum(σ)")

# Condition number = σ_max / σ_min
# High condition number → ill-conditioned (sensitive to perturbations)
cond = np.linalg.cond(A)
print(f"\nCondition number: {cond:.4f}")

# Nearly singular matrix
A_illcond = np.array([[1, 1], [1, 1.0001]])
print(f"Ill-conditioned: {np.linalg.cond(A_illcond):.2e}")
```

**Condition numbers: deep dive**

\[
\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}
\]

**Interpretation:** For \( Ax = b \), a relative error \( \varepsilon \) in \( b \) can amplify to \( \kappa(A) \cdot \varepsilon \) in \( x \). If \( \kappa(A) \approx 10^{16} \), the system is effectively singular in float64. **Mitigations:** Use `np.linalg.solve` or QR/LU instead of explicit inverse; add regularization (Ridge: \( X^TX + \lambda I \)) to improve conditioning.

```python
# Demonstrating ill-conditioning
A_ill = np.array([[1, 1], [1, 1.0001]])
b = np.array([2.0, 2.0])
x_true = np.linalg.solve(A_ill, b)

# Perturb b slightly
b_perturbed = b + 1e-8 * np.array([1, -1])
x_perturbed = np.linalg.solve(A_ill, b_perturbed)
print(f"Condition number: {np.linalg.cond(A_ill):.2e}")
print(f"Relative error in x: {np.linalg.norm(x_perturbed - x_true) / np.linalg.norm(x_true):.2e}")
```

### 11.2 Distance Metrics in ML

```python
from scipy.spatial.distance import cdist

# Common distances
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 0.0, 1.0])

euclidean = np.linalg.norm(a - b)
manhattan = np.sum(np.abs(a - b))
chebyshev = np.max(np.abs(a - b))
cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
cosine_dist = 1 - cosine_sim

print(f"Euclidean (L2): {euclidean:.4f}")
print(f"Manhattan (L1): {manhattan:.4f}")
print(f"Chebyshev (L∞): {chebyshev:.4f}")
print(f"Cosine similarity: {cosine_sim:.4f}")
print(f"Cosine distance: {cosine_dist:.4f}")

# Mahalanobis distance (accounts for correlation and scale)
# d(x, y) = sqrt((x-y)^T Σ^(-1) (x-y))
X = np.random.randn(100, 3)
Sigma = np.cov(X.T)
Sigma_inv = np.linalg.inv(Sigma)

x = X[0]
y = X[1]
diff = x - y
mahal_dist = np.sqrt(diff @ Sigma_inv @ diff)
print(f"\nMahalanobis distance: {mahal_dist:.4f}")

# KNN with different metrics (scipy cdist)
points = np.random.randn(5, 3)
dist_matrix = cdist(points, points, metric='euclidean')
print(f"\nDistance matrix shape: {dist_matrix.shape}")
```

---

## 12. ML Applications

### 12.1 PCA from Scratch: Geometry & Covariance Eigendecomposition

**PCA geometry:** Project data onto the directions of maximum variance. The first principal component (PC1) is the direction that maximizes \( \text{Var}(X w) = w^T C w \) subject to \( \|w\| = 1 \), where \( C \) is the covariance matrix. This is equivalent to the eigenvector of \( C \) with largest eigenvalue. Subsequent PCs maximize variance in the subspace orthogonal to previous PCs — hence the eigenvectors in order of decreasing eigenvalues.

**Key insight:** SVD of centered \( X \) gives the same principal directions as eigendecomposition of \( C = \frac{1}{n-1}X^TX \). When \( d \gg n \), SVD of \( X \) is more efficient and numerically stable.

```python
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

# Step 1: Center
X_centered = X - X.mean(axis=0)

# Step 2: Covariance matrix
n = X_centered.shape[0]
C = (X_centered.T @ X_centered) / (n - 1)

# Step 3: Eigendecomposition of C
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Step 4: Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 5: Project
n_components = 2
W = eigenvectors[:, :n_components]  # Principal axes
X_pca = X_centered @ W

# Variance explained
var_explained = eigenvalues / eigenvalues.sum()
print("PCA via covariance matrix:")
print(f"Explained variance ratio: {var_explained.round(4)}")
print(f"Cumulative: {np.cumsum(var_explained).round(4)}")
print(f"Shape: {X_pca.shape}")

# Verify vs sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca_sk = pca.fit_transform(X)
print(f"\nMatches sklearn: {np.allclose(np.abs(X_pca), np.abs(X_pca_sk))}")
# (signs may differ, hence abs)
```

### 12.2 Linear Regression via Normal Equation

```python
# Linear regression: y = Xw, minimize ||y - Xw||²
# Normal equation: w = (X^T X)^{-1} X^T y
# Better: solve (X^T X) w = X^T y

np.random.seed(42)
n, p = 200, 10
X_raw = np.random.randn(n, p)
X_b = np.hstack([np.ones((n, 1)), X_raw])  # Add bias term
w_true = np.random.randn(p + 1)
y = X_b @ w_true + 0.1 * np.random.randn(n)

# Method 1: Normal equation (unstable for large/ill-conditioned systems)
w_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Method 2: np.linalg.solve (uses LU, more stable)
w_solve = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

# Method 3: lstsq via SVD (most stable)
w_lstsq, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

print(f"Normal eq vs solve: {np.allclose(w_normal, w_solve, atol=1e-8)}")
print(f"Normal eq vs lstsq: {np.allclose(w_normal, w_lstsq, atol=1e-8)}")
print(f"Max weight error: {np.max(np.abs(w_lstsq - w_true)):.6f}")
```

### 12.3 Attention Mechanism (Transformers)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (seq_len, d_k) — queries
    K: (seq_len, d_k) — keys
    V: (seq_len, d_v) — values
    """
    d_k = Q.shape[-1]

    # Dot product attention: QK^T / sqrt(d_k)
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)

    # Apply mask (for causal attention in decoders)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Softmax along last dimension
    scores_shifted = scores - scores.max(axis=-1, keepdims=True)  # Numerical stability
    attn_weights = np.exp(scores_shifted)
    attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

    # Weighted sum of values
    output = attn_weights @ V
    return output, attn_weights

# Example
seq_len, d_k, d_v = 5, 16, 32
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, attn = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {attn.shape}")
print(f"Attention weights sum to 1: {np.allclose(attn.sum(axis=-1), 1)}")
```

### 12.4 Recommender Systems via Matrix Factorization

```python
# Matrix factorization: R ≈ U V^T
# R: (n_users, n_items), U: (n_users, k), V: (n_items, k)

# Simplified ALS (Alternating Least Squares) from scratch
def als_factorization(R, k=10, n_iters=50, lam=0.1, seed=42):
    """ALS matrix factorization."""
    rng = np.random.RandomState(seed)
    m, n = R.shape
    U = rng.randn(m, k)
    V = rng.randn(n, k)

    mask = R > 0  # Observed entries

    for _ in range(n_iters):
        # Fix V, solve for U: for each user i,
        # min_u ||R[i,:] - u V^T||² + λ||u||²
        for i in range(m):
            obs_i = mask[i]
            V_obs = V[obs_i]
            R_obs = R[i, obs_i]
            A = V_obs.T @ V_obs + lam * np.eye(k)
            b = V_obs.T @ R_obs
            U[i] = np.linalg.solve(A, b)

        # Fix U, solve for V
        for j in range(n):
            obs_j = mask[:, j]
            U_obs = U[obs_j]
            R_obs = R[obs_j, j]
            A = U_obs.T @ U_obs + lam * np.eye(k)
            b = U_obs.T @ R_obs
            V[j] = np.linalg.solve(A, b)

    return U, V

# Toy ratings matrix (0 = unobserved)
R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]], dtype=float)

U, V = als_factorization(R, k=2, n_iters=100, lam=0.1)
R_pred = U @ V.T

print("Original ratings (0=missing):")
print(R)
print("\nPredicted ratings:")
print(R_pred.round(2))

# Observed RMSE
mask = R > 0
obs_rmse = np.sqrt(np.mean((R[mask] - R_pred[mask])**2))
print(f"\nObserved RMSE: {obs_rmse:.4f}")
```

### 12.5 Common Pitfalls & Numerical Stability

```python
import numpy as np

# Pitfall 1: Using inv() instead of solve()
A = np.random.randn(100, 100)
b = np.random.randn(100)

# BAD: x = np.linalg.inv(A) @ b  — slower, less stable
# GOOD:
x = np.linalg.solve(A, b)
print(f"Solve: {np.allclose(A @ x, b)}")

# Pitfall 2: Row vs column vector confusion
a = np.array([1, 2, 3])   # shape (3,) — neither row nor column!
b = np.array([[1, 2, 3]])  # shape (1, 3) — row vector
# For matmul: (3,) works as (1,3) or (3,1) depending on context
# Explicit: a.reshape(-1, 1) for column, a.reshape(1, -1) for row

# Pitfall 3: Ill-conditioned systems
# X^T X can be singular or near-singular when features are collinear
X = np.random.randn(50, 100)  # 50 samples, 100 features
XTX = X.T @ X
print(f"X^T X condition number: {np.linalg.cond(XTX):.2e}")
# Ridge: XTX + λI improves conditioning
XTX_ridge = XTX + 0.1 * np.eye(100)
print(f"Ridge condition number: {np.linalg.cond(XTX_ridge):.2e}")

# Pitfall 4: eig() vs eigh() for symmetric matrices
S = np.array([[3, 1], [1, 2]])
lam_eig, v_eig = np.linalg.eig(S)   # Can return complex (numerical noise)
lam_eigh, v_eigh = np.linalg.eigh(S)  # Guaranteed real for symmetric
print(f"eigh gives real: {np.isrealobj(lam_eigh)}")
```

**Summary of pitfalls:**
- **Never use `inv(A) @ b`** — use `np.linalg.solve(A, b)` or `lstsq` for least squares.
- **Ill-conditioning:** Check \( \kappa(A) \); use Ridge regularization or QR/SVD.
- **Symmetric matrices:** Use `eigh` instead of `eig` for guaranteed real eigenvalues.
- **Broadcasting:** Be explicit with shapes when debugging matrix dimensions.

---

## Summary: Key Linear Algebra Facts for ML

```
Core Operations:
├── Matrix-vector product Ax: Linear combination of columns of A
├── Dot product a·b = ||a||||b||cos(θ): Measures alignment
├── Projection: p = (a^T b / a^T a) a maps b onto direction of a
└── Least squares: w = (X^T X)^{-1} X^T y = argmin ||y - Xw||²

Key Decompositions:
├── SVD: A = UΣV^T — the most important one (works on ANY matrix)
├── Eigendecomp: A = QΛQ^{-1} — for square matrices, power computations
├── QR: A = QR — least squares, numerically stable solvers
└── Cholesky: A = LL^T — for SPD matrices (covariance, Gram matrices)

Intuitions:
├── Determinant = volume scaling factor of transformation
├── Eigenvalue = stretch factor along eigenvector direction
├── Singular values = "energy" along data directions (for PCA)
├── Rank = intrinsic dimensionality of data
└── Orthogonal matrix = rigid body transformation (rotation/reflection)

ML Connections:
├── PCA = eigendecomp of covariance = SVD of data matrix
├── Linear regression = least squares = pseudoinverse = QR solve
├── Ridge regression = (X^T X + λI)^{-1} X^T y (stabilizes via λI)
├── Attention = QK^T/sqrt(d_k) then softmax (matrix multiplication!)
└── Backprop = chain rule = Jacobian-vector products
```

---

## References

- **Linear Algebra:** Strang, *Linear Algebra and Its Applications*; Axler, *Linear Algebra Done Right*
- **Numerical LA:** Golub & Van Loan, *Matrix Computations*; Trefethen & Bau, *Numerical Linear Algebra*
- **ML:** Bishop, *Pattern Recognition and Machine Learning*; Goodfellow et al., *Deep Learning*
