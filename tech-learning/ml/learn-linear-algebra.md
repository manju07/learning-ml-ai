# Linear Algebra for Machine Learning and Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Vectors](#vectors)
3. [Matrices](#matrices)
4. [Matrix Operations](#matrix-operations)
5. [Linear Transformations](#linear-transformations)
6. [Eigenvalues and Eigenvectors](#eigenvalues)
7. [Matrix Decompositions](#decompositions)
8. [Vector Spaces and Subspaces](#vector-spaces)
9. [Norms and Distances](#norms)
10. [Applications in ML/DL](#applications)
11. [Practical Examples](#examples)

---

## Introduction {#introduction}

Linear algebra is the mathematical foundation of machine learning and deep learning. It provides the language and tools to understand how data is represented, transformed, and processed in ML algorithms.

### Why Linear Algebra Matters in ML/DL
- **Data Representation**: Data is represented as vectors and matrices
- **Transformations**: Neural networks perform linear transformations
- **Dimensionality Reduction**: PCA, SVD reduce feature dimensions
- **Optimization**: Gradient descent operates on vector spaces
- **Efficiency**: Matrix operations enable parallel computation

### Key Concepts Overview
- Vectors: Represent data points and features
- Matrices: Represent datasets and transformations
- Linear transformations: Core operations in neural networks
- Eigenvalues/Eigenvectors: Dimensionality reduction and principal components
- Matrix decompositions: Efficient computation and understanding

---

## Vectors {#vectors}

### Definition
A vector is an ordered collection of numbers (scalars) arranged in a row or column.

```python
import numpy as np

# Column vector
v = np.array([[1], [2], [3]])
print(f"Column vector:\n{v}")

# Row vector
v_row = np.array([1, 2, 3])
print(f"Row vector: {v_row}")

# Vector properties
print(f"Shape: {v.shape}")
print(f"Dimension: {v.shape[0]}")
```

### Vector Operations

#### Addition and Subtraction
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition
c = a + b
print(f"a + b = {c}")  # [5, 7, 9]

# Subtraction
d = a - b
print(f"a - b = {d}")  # [-3, -3, -3]

# Scalar multiplication
e = 2 * a
print(f"2 * a = {e}")  # [2, 4, 6]
```

#### Dot Product (Inner Product)
The dot product measures similarity between vectors and is fundamental in ML.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(a, b)
print(f"a · b = {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

# Geometric interpretation: ||a|| * ||b|| * cos(θ)
# where θ is the angle between vectors
```

**Properties:**
- Commutative: a · b = b · a
- Distributive: a · (b + c) = a · b + a · c
- Scalar multiplication: (k·a) · b = k(a · b)

#### Cross Product (3D)
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

cross_product = np.cross(a, b)
print(f"a × b = {cross_product}")  # [-3, 6, -3]
```

### Vector Norms

#### L2 Norm (Euclidean Norm)
```python
v = np.array([3, 4])
l2_norm = np.linalg.norm(v)
print(f"L2 norm: {l2_norm}")  # √(3² + 4²) = 5

# Manual calculation
l2_manual = np.sqrt(np.sum(v**2))
print(f"Manual L2: {l2_manual}")
```

#### L1 Norm (Manhattan Norm)
```python
v = np.array([3, 4])
l1_norm = np.sum(np.abs(v))
print(f"L1 norm: {l1_norm}")  # |3| + |4| = 7
```

#### L∞ Norm (Max Norm)
```python
v = np.array([3, 4, -5])
linf_norm = np.max(np.abs(v))
print(f"L∞ norm: {linf_norm}")  # max(|3|, |4|, |5|) = 5
```

### Unit Vectors
```python
v = np.array([3, 4])
unit_vector = v / np.linalg.norm(v)
print(f"Unit vector: {unit_vector}")  # [0.6, 0.8]
print(f"Norm of unit vector: {np.linalg.norm(unit_vector)}")  # 1.0
```

---

## Matrices {#matrices}

### Definition
A matrix is a rectangular array of numbers arranged in rows and columns.

```python
# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")  # (3, 3)
print(f"Rows: {A.shape[0]}")
print(f"Columns: {A.shape[1]}")
```

### Special Matrices

#### Identity Matrix
```python
I = np.eye(3)
print(f"Identity matrix:\n{I}")
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]
```

#### Zero Matrix
```python
Z = np.zeros((3, 3))
print(f"Zero matrix:\n{Z}")
```

#### Diagonal Matrix
```python
diag = np.diag([1, 2, 3])
print(f"Diagonal matrix:\n{diag}")
```

#### Symmetric Matrix
```python
A = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
print(f"Is symmetric: {np.allclose(A, A.T)}")
```

#### Orthogonal Matrix
```python
# Columns are orthonormal
Q = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)]])
print(f"Is orthogonal: {np.allclose(Q @ Q.T, np.eye(2))}")
```

---

## Matrix Operations {#matrix-operations}

### Matrix Addition and Subtraction
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
C = A + B
print(f"A + B:\n{C}")

# Subtraction
D = A - B
print(f"A - B:\n{D}")

# Scalar multiplication
E = 2 * A
print(f"2A:\n{E}")
```

### Matrix Multiplication

#### Standard Matrix Multiplication
```python
A = np.array([[1, 2], [3, 4]])  # 2x2
B = np.array([[5, 6], [7, 8]])  # 2x2

# Matrix multiplication
C = A @ B  # or np.dot(A, B)
print(f"A @ B:\n{C}")
# [[1*5+2*7, 1*6+2*8],
#  [3*5+4*7, 3*6+4*8]]
# = [[19, 22],
#    [43, 50]]
```

**Important:** Matrix multiplication is NOT commutative: A @ B ≠ B @ A

#### Element-wise Multiplication (Hadamard Product)
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
C = A * B
print(f"A * B (element-wise):\n{C}")
# [[5, 12],
#  [21, 32]]
```

### Matrix Transpose
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_T = A.T
print(f"A:\n{A}")
print(f"A^T:\n{A_T}")
# A: [[1, 2, 3],
#     [4, 5, 6]]
# A^T: [[1, 4],
#       [2, 5],
#       [3, 6]]
```

**Properties:**
- (A^T)^T = A
- (A + B)^T = A^T + B^T
- (AB)^T = B^T A^T

### Matrix Inverse
```python
A = np.array([[1, 2], [3, 4]])

# Check if invertible (determinant ≠ 0)
det = np.linalg.det(A)
print(f"Determinant: {det}")

if det != 0:
    A_inv = np.linalg.inv(A)
    print(f"A^(-1):\n{A_inv}")
    
    # Verify: A @ A^(-1) = I
    I = A @ A_inv
    print(f"A @ A^(-1):\n{I}")
```

**Properties:**
- (A^(-1))^(-1) = A
- (AB)^(-1) = B^(-1) A^(-1)
- (A^T)^(-1) = (A^(-1))^T

### Determinant
```python
A = np.array([[1, 2], [3, 4]])
det = np.linalg.det(A)
print(f"det(A) = {det}")  # 1*4 - 2*3 = -2

# For 2x2: det([[a,b],[c,d]]) = ad - bc
# For 3x3: More complex (Laplace expansion)
```

**Geometric Interpretation:** Determinant represents the scaling factor of the linear transformation.

### Trace
```python
A = np.array([[1, 2], [3, 4]])
trace = np.trace(A)
print(f"tr(A) = {trace}")  # 1 + 4 = 5
```

---

## Linear Transformations {#linear-transformations}

### Definition
A linear transformation T satisfies:
- T(u + v) = T(u) + T(v)
- T(c·u) = c·T(u)

### Matrix as Linear Transformation
```python
# Rotation matrix (rotate 45 degrees)
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# Apply transformation
v = np.array([1, 0])
v_rotated = R @ v
print(f"Original: {v}")
print(f"Rotated: {v_rotated}")
```

### Common Transformations

#### Scaling
```python
# Scale by factor 2 in x, 3 in y
S = np.array([[2, 0],
              [0, 3]])

v = np.array([1, 1])
v_scaled = S @ v
print(f"Scaled: {v_scaled}")  # [2, 3]
```

#### Reflection
```python
# Reflect across y-axis
R = np.array([[-1, 0],
              [0, 1]])

v = np.array([1, 1])
v_reflected = R @ v
print(f"Reflected: {v_reflected}")  # [-1, 1]
```

#### Shear
```python
# Shear transformation
H = np.array([[1, 1],
              [0, 1]])

v = np.array([1, 1])
v_sheared = H @ v
print(f"Sheared: {v_sheared}")  # [2, 1]
```

---

## Eigenvalues and Eigenvectors {#eigenvalues}

### Definition
For a matrix A, if Av = λv for some scalar λ and vector v ≠ 0, then:
- λ is an eigenvalue
- v is an eigenvector

### Computing Eigenvalues and Eigenvectors
```python
A = np.array([[4, 1],
              [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    result = A @ eigenvec
    expected = eigenval * eigenvec
    print(f"\nEigenvalue {i+1}: {eigenval}")
    print(f"Av = {result}")
    print(f"λv = {expected}")
    print(f"Match: {np.allclose(result, expected)}")
```

### Properties
- Sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)
- Eigenvectors of distinct eigenvalues are linearly independent

### Applications in ML
- **PCA**: Principal components are eigenvectors of covariance matrix
- **PageRank**: Eigenvector of transition matrix
- **Dimensionality Reduction**: Use eigenvectors with largest eigenvalues

---

## Matrix Decompositions {#decompositions}

### Eigenvalue Decomposition
```python
A = np.array([[4, 1],
              [2, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# A = QΛQ^(-1) where Q is eigenvectors, Λ is diagonal eigenvalues
Q = eigenvectors
Lambda = np.diag(eigenvalues)
Q_inv = np.linalg.inv(Q)

A_reconstructed = Q @ Lambda @ Q_inv
print(f"Original A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Match: {np.allclose(A, A_reconstructed)}")
```

### Singular Value Decomposition (SVD)
SVD is one of the most important decompositions in ML.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# SVD: A = UΣV^T
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"U shape: {U.shape}")
print(f"S (singular values): {S}")
print(f"V^T shape: {Vt.shape}")

# Reconstruct A
Sigma = np.diag(S)
A_reconstructed = U @ Sigma @ Vt
print(f"\nOriginal A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Match: {np.allclose(A, A_reconstructed)}")
```

**Applications:**
- **PCA**: SVD of centered data matrix
- **Low-rank approximation**: Keep top k singular values
- **Matrix completion**: Fill missing values
- **Image compression**: Compress images using SVD

#### Low-rank Approximation Example
```python
# Create a matrix
A = np.random.rand(100, 100)

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Keep top 10 singular values (low-rank approximation)
k = 10
U_k = U[:, :k]
S_k = S[:k]
Vt_k = Vt[:k, :]

A_approx = U_k @ np.diag(S_k) @ Vt_k

# Compression ratio
original_size = A.size
compressed_size = U_k.size + S_k.size + Vt_k.size
compression_ratio = original_size / compressed_size
print(f"Compression ratio: {compression_ratio:.2f}x")
```

### QR Decomposition
```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# QR decomposition: A = QR
Q, R = np.linalg.qr(A)

print(f"Q (orthogonal):\n{Q}")
print(f"R (upper triangular):\n{R}")

# Verify: Q is orthogonal (Q^T Q = I)
print(f"\nQ^T Q = I: {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")

# Reconstruct A
A_reconstructed = Q @ R
print(f"\nOriginal A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
```

**Applications:**
- **Solving linear systems**: More numerically stable than direct inversion
- **Least squares**: QR decomposition for overdetermined systems
- **Gram-Schmidt process**: Orthogonalization

### Cholesky Decomposition
For symmetric positive definite matrices: A = LL^T

```python
# Create symmetric positive definite matrix
A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])

# Cholesky decomposition
L = np.linalg.cholesky(A)

print(f"L:\n{L}")
print(f"L^T:\n{L.T}")

# Verify: A = LL^T
A_reconstructed = L @ L.T
print(f"\nOriginal A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Match: {np.allclose(A, A_reconstructed)}")
```

**Applications:**
- **Sampling from multivariate Gaussian**: Generate correlated random variables
- **Solving linear systems**: Faster than LU decomposition for SPD matrices
- **Kalman filters**: Covariance matrix updates

---

## Vector Spaces and Subspaces {#vector-spaces}

### Vector Space
A set V with operations + and · satisfying:
- Closure under addition and scalar multiplication
- Commutativity, associativity, distributivity
- Existence of zero vector and additive inverses

### Subspace
A subset of a vector space that is itself a vector space.

```python
# Check if vectors span a subspace
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])

# Stack vectors as rows
V = np.vstack([v1, v2, v3])

# Check rank (dimension of column space)
rank = np.linalg.matrix_rank(V)
print(f"Rank: {rank}")  # Dimension of subspace spanned
```

### Linear Independence
```python
# Check linear independence
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

V = np.vstack([v1, v2, v3])
rank = np.linalg.matrix_rank(V)
print(f"Rank: {rank}")
print(f"Number of vectors: {V.shape[0]}")
print(f"Linearly independent: {rank == V.shape[0]}")
```

### Basis and Dimension
```python
# Find basis for column space
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Column space basis (using SVD)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Columns of U corresponding to non-zero singular values form basis
rank = np.sum(S > 1e-10)
basis = U[:, :rank]
print(f"Basis vectors:\n{basis}")
print(f"Dimension: {rank}")
```

---

## Norms and Distances {#norms}

### Matrix Norms

#### Frobenius Norm
```python
A = np.array([[1, 2], [3, 4]])
frobenius_norm = np.linalg.norm(A, 'fro')
print(f"Frobenius norm: {frobenius_norm}")
# √(1² + 2² + 3² + 4²) = √30
```

#### Spectral Norm (L2 norm of matrix)
```python
A = np.array([[1, 2], [3, 4]])
spectral_norm = np.linalg.norm(A, 2)
print(f"Spectral norm: {spectral_norm}")
# Largest singular value
```

### Distance Metrics

#### Euclidean Distance
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

euclidean_dist = np.linalg.norm(a - b)
print(f"Euclidean distance: {euclidean_dist}")
```

#### Manhattan Distance
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

manhattan_dist = np.sum(np.abs(a - b))
print(f"Manhattan distance: {manhattan_dist}")
```

#### Cosine Similarity
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity: {cosine_sim}")

# Cosine distance
cosine_dist = 1 - cosine_sim
print(f"Cosine distance: {cosine_dist}")
```

---

## Applications in ML/DL {#applications}

### 1. Data Representation
```python
# Dataset as matrix: rows = samples, columns = features
X = np.random.rand(1000, 784)  # 1000 images, 784 pixels each
print(f"Data matrix shape: {X.shape}")
```

### 2. Linear Regression
```python
# Linear regression: y = Xw + b
# Solution: w = (X^T X)^(-1) X^T y

X = np.random.rand(100, 5)
y = np.random.rand(100)

# Add bias column
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

# Normal equation
w = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
print(f"Learned weights: {w}")
```

### 3. Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA

# Generate data
X = np.random.rand(100, 10)

# Center the data
X_centered = X - X.mean(axis=0)

# PCA using SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Keep top 2 principal components
n_components = 2
components = Vt[:n_components, :].T
X_reduced = X_centered @ components

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {S[:n_components]**2 / np.sum(S**2)}")
```

### 4. Neural Network Layers
```python
# Fully connected layer: y = activation(XW + b)
X = np.random.rand(32, 784)  # 32 samples, 784 features
W = np.random.rand(784, 128)  # Weight matrix
b = np.random.rand(128)      # Bias vector

# Forward pass
z = X @ W + b
y = np.maximum(0, z)  # ReLU activation

print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
```

### 5. Convolution as Matrix Multiplication
```python
# Convolution can be expressed as matrix multiplication
# Using Toeplitz matrix or im2col transformation
from scipy.linalg import toeplitz

# Simple 1D convolution example
signal = np.array([1, 2, 3, 4, 5])
kernel = np.array([1, -1])

# Create Toeplitz matrix for convolution
conv_matrix = toeplitz(np.hstack([kernel, np.zeros(len(signal)-len(kernel))]),
                       np.zeros(len(signal)))
result = conv_matrix @ signal
print(f"Convolution result: {result}")
```

### 6. Attention Mechanism (Simplified)
```python
# Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) V
seq_len = 10
d_k = 64

Q = np.random.rand(seq_len, d_k)
K = np.random.rand(seq_len, d_k)
V = np.random.rand(seq_len, d_k)

# Compute attention scores
scores = Q @ K.T / np.sqrt(d_k)

# Softmax
exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# Apply attention
attention_output = attention_weights @ V
print(f"Attention output shape: {attention_output.shape}")
```

---

## Practical Examples {#examples}

### Example 1: Image Compression with SVD
```python
import matplotlib.pyplot as plt

# Load or create an image (grayscale)
image = np.random.rand(100, 100) * 255

# SVD decomposition
U, S, Vt = np.linalg.svd(image, full_matrices=False)

# Reconstruct with different numbers of components
ranks = [5, 10, 20, 50]
fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(15, 3))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for idx, rank in enumerate(ranks):
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]
    
    compressed = U_k @ np.diag(S_k) @ Vt_k
    
    axes[idx + 1].imshow(compressed, cmap='gray')
    axes[idx + 1].set_title(f'Rank {rank}')
    axes[idx + 1].axis('off')

plt.tight_layout()
plt.show()

# Calculate compression ratio
original_size = image.size
for rank in ranks:
    compressed_size = U[:, :rank].size + S[:rank].size + Vt[:rank, :].size
    ratio = original_size / compressed_size
    print(f"Rank {rank}: Compression ratio = {ratio:.2f}x")
```

### Example 2: PCA for Dimensionality Reduction
```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate 3D data
X, y = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)

# Center the data
X_centered = X - X.mean(axis=0)

# Compute covariance matrix
cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project to 2D using top 2 principal components
W = eigenvectors[:, :2]
X_reduced = X_centered @ W

# Visualize
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
ax1.set_title('Original 3D Data')

ax2 = fig.add_subplot(122)
ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
ax2.set_title('PCA Reduced to 2D')
ax2.set_xlabel(f'PC1 (Var: {eigenvalues[0]:.2f})')
ax2.set_ylabel(f'PC2 (Var: {eigenvalues[1]:.2f})')

plt.tight_layout()
plt.show()

# Variance explained
variance_explained = eigenvalues / np.sum(eigenvalues)
print(f"Variance explained by PC1: {variance_explained[0]:.2%}")
print(f"Variance explained by PC2: {variance_explained[1]:.2%}")
```

### Example 3: Solving Linear Systems
```python
# System: Ax = b
A = np.array([[3, 2, 1],
              [2, 3, 1],
              [1, 1, 4]])

b = np.array([10, 11, 12])

# Method 1: Direct solution
x1 = np.linalg.solve(A, b)
print(f"Solution (direct): {x1}")

# Method 2: Using inverse (less efficient, less stable)
x2 = np.linalg.inv(A) @ b
print(f"Solution (inverse): {x2}")

# Verify
print(f"Verification: {np.allclose(A @ x1, b)}")
```

### Example 4: Matrix Factorization for Recommendation
```python
# User-item rating matrix (simplified)
# R ≈ U @ V^T where U are user factors, V are item factors
R = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])

# SVD for matrix factorization
U, S, Vt = np.linalg.svd(R, full_matrices=False)

# Keep top k factors
k = 2
U_k = U[:, :k] @ np.diag(np.sqrt(S[:k]))
V_k = Vt[:k, :].T @ np.diag(np.sqrt(S[:k]))

# Reconstruct ratings
R_pred = U_k @ V_k.T
print(f"Original ratings:\n{R}")
print(f"\nPredicted ratings:\n{R_pred}")
```

---

## Key Takeaways

1. **Vectors and Matrices**: Fundamental data structures in ML
2. **Matrix Operations**: Enable efficient computation and transformations
3. **Eigenvalues/Eigenvectors**: Essential for dimensionality reduction
4. **SVD**: Powerful decomposition for many ML applications
5. **Norms**: Measure distances and similarities
6. **Linear Transformations**: Core of neural network operations

---

## Additional Resources

- **3Blue1Brown Essence of Linear Algebra**: Visual intuition
- **Gilbert Strang's Linear Algebra**: Comprehensive textbook
- **NumPy Documentation**: Implementation details
- **Matrix Cookbook**: Quick reference for matrix identities

---

**Master linear algebra to deeply understand how ML/DL algorithms work!**

