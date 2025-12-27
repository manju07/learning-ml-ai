# Calculus for Machine Learning and Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Functions and Limits](#functions-limits)
3. [Derivatives](#derivatives)
4. [Partial Derivatives and Gradients](#gradients)
5. [Chain Rule](#chain-rule)
6. [Higher-Order Derivatives](#higher-order)
7. [Optimization](#optimization)
8. [Constrained Optimization](#constrained-optimization)
9. [Integration](#integration)
10. [Applications in ML/DL](#applications)
11. [Practical Examples](#examples)

---

## Introduction {#introduction}

Calculus is essential for understanding and implementing machine learning and deep learning algorithms. It provides the mathematical foundation for:
- **Optimization**: Finding optimal model parameters
- **Gradient Descent**: Core training algorithm
- **Backpropagation**: Computing gradients in neural networks
- **Loss Functions**: Measuring model performance

### Why Calculus Matters in ML/DL
- **Gradient Descent**: Uses derivatives to find minima
- **Backpropagation**: Chain rule for computing gradients
- **Regularization**: Gradient-based penalty terms
- **Hyperparameter Tuning**: Understanding loss landscapes
- **Convergence Analysis**: Understanding optimization behavior

---

## Functions and Limits {#functions-limits}

### Functions
A function f maps inputs x to outputs f(x).

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple function: f(x) = x^2
def f(x):
    return x**2

x = np.linspace(-5, 5, 100)
y = f(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x) = x²')
plt.title('Function f(x) = x²')
plt.grid(True)
plt.show()
```

### Limits
The limit describes the behavior of a function as input approaches a value.

```python
# Limit: lim(x→0) sin(x)/x = 1
x = np.linspace(-0.1, 0.1, 1000)
y = np.sin(x) / x

# Remove division by zero
y[x == 0] = 1

plt.plot(x, y)
plt.axhline(y=1, color='r', linestyle='--', label='Limit = 1')
plt.xlabel('x')
plt.ylabel('sin(x)/x')
plt.title('Limit of sin(x)/x as x→0')
plt.legend()
plt.grid(True)
plt.show()
```

### Continuity
A function is continuous if small changes in input cause small changes in output.

```python
# Continuous function
def continuous_func(x):
    return x**2 + 2*x + 1

# Discontinuous function (step function)
def step_func(x):
    return np.where(x >= 0, 1, 0)

x = np.linspace(-2, 2, 1000)
y1 = continuous_func(x)
y2 = step_func(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x, y1)
ax1.set_title('Continuous Function')
ax1.grid(True)

ax2.plot(x, y2)
ax2.set_title('Discontinuous Function (Step)')
ax2.grid(True)
plt.show()
```

---

## Derivatives {#derivatives}

### Definition
The derivative measures the rate of change of a function:

f'(x) = lim(h→0) [f(x+h) - f(x)] / h

### Numerical Differentiation
```python
def numerical_derivative(f, x, h=1e-5):
    """Compute derivative numerically"""
    return (f(x + h) - f(x)) / h

# Example: f(x) = x^2, f'(x) = 2x
def f(x):
    return x**2

x = np.linspace(-5, 5, 100)
df_dx_numerical = numerical_derivative(f, x)
df_dx_analytical = 2 * x  # True derivative

plt.plot(x, df_dx_numerical, label='Numerical', linestyle='--')
plt.plot(x, df_dx_analytical, label='Analytical', linestyle='-')
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title("Derivative of f(x) = x²")
plt.legend()
plt.grid(True)
plt.show()
```

### Common Derivatives

#### Power Rule
```python
# d/dx(x^n) = n*x^(n-1)
# Example: d/dx(x^3) = 3x^2

def power_rule_example():
    x = np.linspace(-3, 3, 100)
    f = x**3
    df_dx = 3 * x**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x, f)
    ax1.set_title('f(x) = x³')
    ax1.grid(True)
    
    ax2.plot(x, df_dx)
    ax2.set_title("f'(x) = 3x²")
    ax2.grid(True)
    plt.show()

power_rule_example()
```

#### Exponential and Logarithmic
```python
# d/dx(e^x) = e^x
# d/dx(ln(x)) = 1/x

x = np.linspace(0.1, 5, 100)
exp_x = np.exp(x)
ln_x = np.log(x)

d_exp = np.exp(x)  # Derivative of e^x is e^x
d_ln = 1 / x       # Derivative of ln(x) is 1/x

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, exp_x)
axes[0, 0].set_title('f(x) = e^x')
axes[0, 0].grid(True)

axes[0, 1].plot(x, d_exp)
axes[0, 1].set_title("f'(x) = e^x")
axes[0, 1].grid(True)

axes[1, 0].plot(x, ln_x)
axes[1, 0].set_title('f(x) = ln(x)')
axes[1, 0].grid(True)

axes[1, 1].plot(x, d_ln)
axes[1, 1].set_title("f'(x) = 1/x")
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

#### Trigonometric Functions
```python
# d/dx(sin(x)) = cos(x)
# d/dx(cos(x)) = -sin(x)

x = np.linspace(-2*np.pi, 2*np.pi, 100)
sin_x = np.sin(x)
cos_x = np.cos(x)

d_sin = np.cos(x)      # Derivative of sin(x)
d_cos = -np.sin(x)     # Derivative of cos(x)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(x, sin_x, label='sin(x)')
axes[0, 0].plot(x, d_sin, label="d/dx(sin(x)) = cos(x)", linestyle='--')
axes[0, 0].set_title('Sine and its Derivative')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(x, cos_x, label='cos(x)')
axes[0, 1].plot(x, d_cos, label="d/dx(cos(x)) = -sin(x)", linestyle='--')
axes[0, 1].set_title('Cosine and its Derivative')
axes[0, 1].legend()
axes[0, 1].grid(True)

plt.tight_layout()
plt.show()
```

### Derivative Rules

#### Sum Rule
```python
# d/dx(f(x) + g(x)) = f'(x) + g'(x)
def sum_rule_example():
    x = np.linspace(-3, 3, 100)
    f = x**2
    g = np.sin(x)
    
    df_dx = 2*x
    dg_dx = np.cos(x)
    
    # Derivative of sum
    d_sum = df_dx + dg_dx
    
    plt.plot(x, f + g, label='f(x) + g(x)')
    plt.plot(x, d_sum, label="d/dx(f + g)", linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

sum_rule_example()
```

#### Product Rule
```python
# d/dx(f(x) * g(x)) = f'(x)*g(x) + f(x)*g'(x)
def product_rule_example():
    x = np.linspace(-3, 3, 100)
    f = x**2
    g = np.sin(x)
    
    df_dx = 2*x
    dg_dx = np.cos(x)
    
    # Product rule
    d_product = df_dx * g + f * dg_dx
    
    plt.plot(x, f * g, label='f(x) * g(x)')
    plt.plot(x, d_product, label="d/dx(f * g)", linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

product_rule_example()
```

#### Quotient Rule
```python
# d/dx(f(x) / g(x)) = (f'(x)*g(x) - f(x)*g'(x)) / g(x)^2
def quotient_rule_example():
    x = np.linspace(0.1, 5, 100)
    f = x**2
    g = np.sin(x)
    
    df_dx = 2*x
    dg_dx = np.cos(x)
    
    # Quotient rule
    d_quotient = (df_dx * g - f * dg_dx) / (g**2)
    
    plt.plot(x, f / g, label='f(x) / g(x)')
    plt.plot(x, d_quotient, label="d/dx(f / g)", linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

quotient_rule_example()
```

---

## Partial Derivatives and Gradients {#gradients}

### Partial Derivatives
For functions of multiple variables, partial derivatives measure change w.r.t. one variable.

```python
# f(x, y) = x^2 + y^2
# ∂f/∂x = 2x
# ∂f/∂y = 2y

def f(x, y):
    return x**2 + y**2

# Partial derivative w.r.t. x
def df_dx(x, y):
    return 2 * x

# Partial derivative w.r.t. y
def df_dy(x, y):
    return 2 * y

# Visualize
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(15, 5))

# Surface plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('f(x, y) = x² + y²')

# Partial derivative w.r.t. x
ax2 = fig.add_subplot(132)
df_dx_vals = df_dx(X, Y)
ax2.contourf(X, Y, df_dx_vals, levels=20)
ax2.set_title('∂f/∂x = 2x')

# Partial derivative w.r.t. y
ax3 = fig.add_subplot(133)
df_dy_vals = df_dy(X, Y)
ax3.contourf(X, Y, df_dy_vals, levels=20)
ax3.set_title('∂f/∂y = 2y')

plt.tight_layout()
plt.show()
```

### Gradient
The gradient is a vector of all partial derivatives.

```python
# Gradient: ∇f = [∂f/∂x, ∂f/∂y]
def gradient_f(x, y):
    return np.array([2*x, 2*y])

# Visualize gradient field
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Compute gradient at each point
U, V = gradient_f(X, Y)

plt.figure(figsize=(10, 8))
plt.contour(X, Y, f(X, Y), levels=20, cmap='viridis')
plt.quiver(X, Y, U, V, scale=50, color='red', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Field of f(x, y) = x² + y²')
plt.grid(True)
plt.show()
```

### Gradient in Higher Dimensions
```python
# Gradient for n-dimensional function
def gradient_multivariate(f, x, h=1e-5):
    """Compute gradient numerically"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (f(x_plus) - f(x)) / h
    return grad

# Example: f(x, y, z) = x^2 + y^2 + z^2
def f_3d(x):
    return np.sum(x**2)

x = np.array([1.0, 2.0, 3.0])
grad = gradient_multivariate(f_3d, x)
print(f"Gradient at {x}: {grad}")
print(f"Analytical gradient: {2*x}")
```

---

## Chain Rule {#chain-rule}

The chain rule is fundamental for backpropagation in neural networks.

### Single Variable Chain Rule
```python
# If y = f(g(x)), then dy/dx = (dy/dg) * (dg/dx)

# Example: y = sin(x^2)
# Let u = x^2, then y = sin(u)
# dy/dx = (dy/du) * (du/dx) = cos(u) * 2x = cos(x^2) * 2x

def chain_rule_example():
    x = np.linspace(-3, 3, 100)
    
    # y = sin(x^2)
    y = np.sin(x**2)
    
    # Derivative using chain rule
    dy_dx = np.cos(x**2) * 2 * x
    
    plt.plot(x, y, label='y = sin(x²)')
    plt.plot(x, dy_dx, label="dy/dx = 2x·cos(x²)", linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()

chain_rule_example()
```

### Multivariable Chain Rule
```python
# For z = f(g(x, y), h(x, y)):
# ∂z/∂x = (∂z/∂g) * (∂g/∂x) + (∂z/∂h) * (∂h/∂x)
# ∂z/∂y = (∂z/∂g) * (∂g/∂y) + (∂z/∂h) * (∂h/∂y)

# Example: z = (x^2 + y^2)^2
# Let u = x^2 + y^2, then z = u^2
# ∂z/∂x = (∂z/∂u) * (∂u/∂x) = 2u * 2x = 4x(x^2 + y^2)
# ∂z/∂y = (∂z/∂u) * (∂u/∂y) = 2u * 2y = 4y(x^2 + y^2)

def multivariable_chain_rule():
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    u = X**2 + Y**2
    z = u**2
    
    dz_dx = 4 * X * u
    dz_dy = 4 * Y * u
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, z, cmap='viridis')
    ax1.set_title('z = (x² + y²)²')
    
    ax2 = fig.add_subplot(132)
    ax2.contourf(X, Y, dz_dx, levels=20)
    ax2.set_title('∂z/∂x')
    
    ax3 = fig.add_subplot(133)
    ax3.contourf(X, Y, dz_dy, levels=20)
    ax3.set_title('∂z/∂y')
    
    plt.tight_layout()
    plt.show()

multivariable_chain_rule()
```

---

## Higher-Order Derivatives {#higher-order}

### Second Derivative
```python
# Second derivative: f''(x) = d/dx(f'(x))
# Measures curvature (concavity/convexity)

def f(x):
    return x**3 - 3*x**2 + 2

def df_dx(x):
    return 3*x**2 - 6*x

def d2f_dx2(x):
    return 6*x - 6

x = np.linspace(-1, 4, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x, f(x))
axes[0].set_title('f(x)')
axes[0].grid(True)

axes[1].plot(x, df_dx(x))
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_title("f'(x)")
axes[1].grid(True)

axes[2].plot(x, d2f_dx2(x))
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_title("f''(x)")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Find inflection points (where f''(x) = 0)
inflection_point = 1.0  # where 6x - 6 = 0
print(f"Inflection point at x = {inflection_point}")
```

### Hessian Matrix
For multivariable functions, the Hessian contains all second-order partial derivatives.

```python
# Hessian: H_ij = ∂²f/∂x_i∂x_j
# For f(x, y) = x^2 + y^2:
# H = [[2, 0],
#      [0, 2]]

def hessian_f(x, y):
    """Hessian matrix for f(x, y) = x^2 + y^2"""
    return np.array([[2, 0],
                     [0, 2]])

# For f(x, y) = x^2*y + y^2:
# H = [[2y, 2x],
#      [2x, 2]]

def hessian_f2(x, y):
    """Hessian matrix for f(x, y) = x^2*y + y^2"""
    return np.array([[2*y, 2*x],
                     [2*x, 2]])

x_val, y_val = 1.0, 2.0
H = hessian_f2(x_val, y_val)
print(f"Hessian at ({x_val}, {y_val}):\n{H}")

# Eigenvalues of Hessian indicate curvature
eigenvals = np.linalg.eigvals(H)
print(f"Eigenvalues: {eigenvals}")
print(f"Positive definite: {np.all(eigenvals > 0)}")  # Convex function
```

---

## Optimization {#optimization}

### Critical Points
Points where gradient is zero (local minima, maxima, or saddle points).

```python
# Find critical points: ∇f = 0
# For f(x, y) = x^2 + y^2, critical point at (0, 0)

def find_critical_points():
    # f(x, y) = x^2 + y^2 - 2xy
    # ∇f = [2x - 2y, 2y - 2x] = [0, 0]
    # Solution: x = y (line of critical points)
    
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    f = X**2 + Y**2 - 2*X*Y
    
    plt.contourf(X, Y, f, levels=20, cmap='viridis')
    plt.plot(x, x, 'r-', linewidth=2, label='Critical points (x=y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Critical Points')
    plt.legend()
    plt.colorbar()
    plt.show()

find_critical_points()
```

### Gradient Descent
The core optimization algorithm in ML.

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.1, n_iterations=100):
    """Simple gradient descent"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        x = x - learning_rate * grad
        history.append(x.copy())
    
    return x, np.array(history)

# Minimize f(x) = x^2
def f_simple(x):
    return x**2

def grad_f_simple(x):
    return 2 * x

x0 = np.array([5.0])
x_opt, history = gradient_descent(f_simple, grad_f_simple, x0, 
                                    learning_rate=0.1, n_iterations=50)

x_vals = np.linspace(-6, 6, 100)
plt.plot(x_vals, f_simple(x_vals), label='f(x) = x²')
plt.plot(history, f_simple(history), 'ro-', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal x: {x_opt[0]:.6f}")
print(f"True minimum: 0.0")
```

### Learning Rate Impact
```python
def compare_learning_rates():
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, lr in enumerate(learning_rates):
        x0 = np.array([5.0])
        x_opt, history = gradient_descent(f_simple, grad_f_simple, x0,
                                          learning_rate=lr, n_iterations=50)
        
        x_vals = np.linspace(-6, 6, 100)
        axes[idx].plot(x_vals, f_simple(x_vals), 'b-', alpha=0.3)
        axes[idx].plot(history, f_simple(history), 'ro-', markersize=4)
        axes[idx].set_title(f'Learning Rate = {lr}')
        axes[idx].grid(True)
    
    plt.tight_layout()
    plt.show()

compare_learning_rates()
```

### Momentum
```python
def gradient_descent_momentum(f, grad_f, x0, learning_rate=0.1, 
                              momentum=0.9, n_iterations=100):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like(x)  # Velocity
    history = [x.copy()]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        v = momentum * v - learning_rate * grad
        x = x + v
        history.append(x.copy())
    
    return x, np.array(history)

# Compare with and without momentum
x0 = np.array([5.0])
x_gd, hist_gd = gradient_descent(f_simple, grad_f_simple, x0, 
                                 learning_rate=0.1, n_iterations=50)
x_mom, hist_mom = gradient_descent_momentum(f_simple, grad_f_simple, x0,
                                            learning_rate=0.1, momentum=0.9,
                                            n_iterations=50)

x_vals = np.linspace(-6, 6, 100)
plt.plot(x_vals, f_simple(x_vals), 'b-', alpha=0.3, label='f(x)')
plt.plot(hist_gd, f_simple(hist_gd), 'ro-', label='Gradient Descent', markersize=4)
plt.plot(hist_mom, f_simple(hist_mom), 'go-', label='With Momentum', markersize=4)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent: With vs Without Momentum')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Constrained Optimization {#constrained-optimization}

### Lagrange Multipliers
For optimizing f(x, y) subject to g(x, y) = 0.

```python
# Minimize f(x, y) = x^2 + y^2 subject to x + y = 1
# Using Lagrange multiplier: ∇f = λ∇g

from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

# Solve
result = minimize(objective, [0, 0], constraints={'type': 'eq', 'fun': constraint})
print(f"Optimal point: ({result.x[0]:.4f}, {result.x[1]:.4f})")
print(f"Constraint satisfied: {constraint(result.x):.6f}")
```

---

## Integration {#integration}

### Definite Integral
```python
from scipy import integrate

# ∫[0 to 1] x^2 dx = [x^3/3]_0^1 = 1/3
def f(x):
    return x**2

result, error = integrate.quad(f, 0, 1)
print(f"∫₀¹ x² dx = {result:.6f}")
print(f"True value: 1/3 = {1/3:.6f}")
print(f"Error: {error:.2e}")
```

### Applications: Expected Value
```python
# Expected value: E[X] = ∫ x·p(x) dx
# For probability density p(x) = 2x on [0, 1]

def pdf(x):
    return 2 * x

def x_times_pdf(x):
    return x * pdf(x)

expected_value, _ = integrate.quad(x_times_pdf, 0, 1)
print(f"Expected value: {expected_value:.6f}")
print(f"True value: 2/3 = {2/3:.6f}")
```

---

## Applications in ML/DL {#applications}

### 1. Loss Function Derivatives
```python
# Mean Squared Error: L = (1/n)Σ(y_pred - y_true)²
# dL/dw = (2/n)Σ(y_pred - y_true) * x

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_gradient(y_pred, y_true, X):
    n = len(y_true)
    return (2/n) * X.T @ (y_pred - y_true)

# Example
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.1, 1.9, 3.2, 3.8])
X = np.array([[1], [2], [3], [4]])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true, X)
print(f"MSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")
```

### 2. Cross-Entropy Loss
```python
# Cross-entropy: L = -Σ y_true * log(y_pred)
# dL/dz = y_pred - y_true (for softmax)

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_gradient(y_pred, y_true):
    return y_pred - y_true

# Example
logits = np.array([2.0, 1.0, 0.1])
y_pred = softmax(logits)
y_true = np.array([1, 0, 0])

loss = cross_entropy_loss(y_pred, y_true)
grad = cross_entropy_gradient(y_pred, y_true)
print(f"Cross-entropy loss: {loss:.4f}")
print(f"Gradient: {grad}")
```

### 3. Backpropagation (Simplified)
```python
# Simple 2-layer network: y = σ(W2 * σ(W1 * x + b1) + b2)
# Using chain rule to compute gradients

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Forward pass
def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward pass (simplified)
def backward(x, y_true, z1, a1, z2, a2, W1, W2):
    # Output layer gradient
    dz2 = (a2 - y_true) * sigmoid_derivative(z2)
    
    # Hidden layer gradient (chain rule)
    da1 = W2.T @ dz2
    dz1 = da1 * sigmoid_derivative(z1)
    
    # Weight gradients
    dW2 = np.outer(dz2, a1)
    dW1 = np.outer(dz1, x)
    
    return dW1, dW2

# Example
x = np.array([0.5, 0.3])
W1 = np.random.rand(3, 2)
b1 = np.random.rand(3)
W2 = np.random.rand(1, 3)
b2 = np.random.rand(1)
y_true = np.array([1.0])

z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
dW1, dW2 = backward(x, y_true, z1, a1, z2, a2, W1, W2)

print(f"Output: {a2[0]:.4f}")
print(f"Weight gradients computed using chain rule")
```

---

## Practical Examples {#examples}

### Example 1: Linear Regression from Scratch
```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Predictions
            y_pred = X @ self.weights + self.bias
            
            # Gradients (derivatives of MSE)
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return X @ self.weights + self.bias

# Test
X = np.random.rand(100, 1)
y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)

model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, y_pred, 'r-', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Example 2: Finding Optimal Learning Rate
```python
def find_optimal_learning_rate(f, grad_f, x0, lr_range, n_iterations=100):
    """Test different learning rates"""
    results = []
    
    for lr in lr_range:
        x, history = gradient_descent(f, grad_f, x0, lr, n_iterations)
        final_loss = f(x)
        results.append((lr, final_loss, history))
    
    return results

# Test
lr_range = np.logspace(-3, 0, 20)
results = find_optimal_learning_rate(f_simple, grad_f_simple, 
                                     np.array([5.0]), lr_range)

learning_rates = [r[0] for r in results]
final_losses = [r[1][0] for r in results]

plt.semilogx(learning_rates, final_losses, 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('Final Loss')
plt.title('Learning Rate vs Final Loss')
plt.grid(True)
plt.show()

optimal_lr = learning_rates[np.argmin(final_losses)]
print(f"Optimal learning rate: {optimal_lr:.4f}")
```

---

## Key Takeaways

1. **Derivatives**: Measure rate of change, essential for optimization
2. **Gradients**: Vector of partial derivatives, points to steepest ascent
3. **Chain Rule**: Fundamental for backpropagation
4. **Gradient Descent**: Core optimization algorithm
5. **Hessian**: Second-order information for advanced optimization
6. **Integration**: Useful for probability and expected values

---

## Additional Resources

- **3Blue1Brown Essence of Calculus**: Visual intuition
- **Khan Academy Calculus**: Comprehensive tutorials
- **Calculus on Manifolds**: Advanced topics
- **Numerical Recipes**: Implementation details

---

**Master calculus to understand how ML/DL algorithms optimize and learn!**

