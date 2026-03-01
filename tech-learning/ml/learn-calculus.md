# Calculus for Machine Learning: Comprehensive Reference

## Table of Contents
1. [Limits & Continuity](#1-limits--continuity)
2. [Derivatives: Definition & Rules](#2-derivatives-definition--rules)
3. [Partial Derivatives & Gradients](#3-partial-derivatives--gradients)
4. [Directional Derivatives](#4-directional-derivatives)
5. [Jacobian & Hessian Matrices](#5-jacobian--hessian-matrices)
6. [Taylor Series](#6-taylor-series)
7. [Integration](#7-integration)
8. [Multivariate Calculus](#8-multivariate-calculus)
9. [Optimization Theory](#9-optimization-theory)
10. [Automatic Differentiation](#10-automatic-differentiation)
11. [Calculus in Machine Learning](#11-calculus-in-machine-learning)

---

## 1. Limits & Continuity

### 1.1 Limits

The **limit** of \( f(x) \) as \( x \to a \) is \( L \) if \( f(x) \) can be made arbitrarily close to \( L \) by taking \( x \) sufficiently close to \( a \):
\[
\lim_{x \to a} f(x) = L
\]

**Formal (ε-δ) definition:** \( \forall \varepsilon > 0, \exists \delta > 0 : 0 < |x - a| < \delta \Rightarrow |f(x) - L| < \varepsilon \)

**Limit laws:**
- \( \lim (f + g) = \lim f + \lim g \)
- \( \lim (fg) = \lim f \cdot \lim g \)
- \( \lim (f/g) = \lim f / \lim g \) (if \( \lim g \neq 0 \))
- \( \lim f(g(x)) = f(\lim g(x)) \) (if \( f \) continuous)

**Important limits:**
\[
\lim_{x \to 0} \frac{\sin x}{x} = 1, \quad
\lim_{x \to 0} \frac{e^x - 1}{x} = 1, \quad
\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e
\]

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerically verify lim(x→0) sin(x)/x = 1
x_vals = np.array([0.1, 0.01, 0.001, 0.0001, 1e-5])
for x in x_vals:
    val = np.sin(x) / x
    print(f"x={x:.0e}: sin(x)/x = {val:.10f}")

# L'Hopital's Rule: if lim f(x)/g(x) = 0/0 or ∞/∞,
# then lim f(x)/g(x) = lim f'(x)/g'(x)
# Example: lim(x→0) (1 - cos(x)) / x^2 = ?
# By L'Hopital: lim sin(x) / (2x) = 1/2 (another 0/0)
# Apply again: lim cos(x) / 2 = 1/2

x_vals = np.array([1.0, 0.1, 0.01, 0.001])
print("\nLim (1-cos(x))/x² → 0.5:")
for x in x_vals:
    val = (1 - np.cos(x)) / x**2
    print(f"  x={x}: {val:.6f}")

# One-sided limits
def f_sided(x):
    return np.where(x > 0, np.sqrt(x), -np.sqrt(-x))

x_right = np.array([0.1, 0.01, 0.001])
x_left = -x_right
print("\nOne-sided limits of f(x):")
print("Right:", [f_sided(x) for x in x_right])
print("Left: ", [f_sided(x) for x in x_left])
```

### 1.2 Continuity

\( f \) is **continuous at \( a \)** if:
1. \( f(a) \) is defined
2. \( \lim_{x \to a} f(x) \) exists
3. \( \lim_{x \to a} f(x) = f(a) \)

**Types of discontinuities:**
- **Removable:** Limit exists, but ≠ f(a) (can "fix" by redefining f(a))
- **Jump:** Left and right limits exist but differ
- **Essential:** Limit doesn't exist at all (e.g., \( \sin(1/x) \) at 0)

**ML relevance:** Activation functions' continuity affects gradient computation:
- ReLU: continuous but not differentiable at 0 (subgradient used)
- Sigmoid, tanh: everywhere smooth (infinitely differentiable)

```python
# Common activation functions and their continuity/smoothness
def relu(x): return np.maximum(0, x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def tanh_fn(x): return np.tanh(x)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def swish(x): return x * sigmoid(x)

x = np.linspace(-3, 3, 1000)
activations = {
    'ReLU': relu(x),
    'Leaky ReLU': leaky_relu(x),
    'Sigmoid': sigmoid(x),
    'Tanh': tanh_fn(x),
    'GELU': gelu(x),
    'Swish': swish(x),
}

# Check smoothness at x=0
for name, fn_vals in activations.items():
    h = 1e-5
    x0 = 0.0
    deriv = (fn_vals[500 + 1] - fn_vals[500 - 1]) / (2 * h * (x[1] - x[0]))
    print(f"{name}: approx derivative at 0 ≈ {deriv:.4f}")
```

---

## 2. Derivatives: Definition & Rules

### 2.1 Definition of Derivative

\[
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\]

**Geometric meaning:** Slope of tangent line to \( f \) at \( x \).

**Physical meaning:** Instantaneous rate of change.

**Intuition (local linear approximation):** The derivative is the *best linear approximation* to \( f \) near \( x \). For small \( h \), \( f(x+h) \approx f(x) + f'(x)h \). The derivative tells you: "If I move a tiny step \( h \), how much does \( f \) change?" — the answer is \( f'(x) \cdot h \). This is why gradients guide optimization: they predict local change.

Three equivalent notations: \( f'(x) \), \( \frac{df}{dx} \), \( Df(x) \)

```python
def numerical_derivative(f, x, h=1e-5):
    """Forward difference: (f(x+h) - f(x)) / h — O(h) error"""
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h=1e-5):
    """Central difference: (f(x+h) - f(x-h)) / (2h) — O(h²) error, more accurate!"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Compare accuracy
def f(x): return np.sin(x)
def df(x): return np.cos(x)  # True derivative

x = 1.0
h_vals = [0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-8]

print(f"True derivative: {df(x):.10f}")
print(f"\n{'h':>10} {'Forward':>15} {'Central':>15} {'FD Error':>12} {'CD Error':>12}")
for h in h_vals:
    fd = numerical_derivative(f, x, h)
    cd = central_difference(f, x, h)
    print(f"{h:10.1e} {fd:15.10f} {cd:15.10f} {abs(fd-df(x)):12.2e} {abs(cd-df(x)):12.2e}")
# Central difference is O(h²), forward is O(h)
# Numerical stability pitfall: h too small causes subtractive cancellation!
# For float64, h ~ 1e-8 is often optimal; h=1e-15 gives garbage.
```

### 2.2 Differentiation Rules

#### Power, Exponential & Logarithm Rules

\[
\frac{d}{dx} x^n = nx^{n-1}, \quad
\frac{d}{dx} e^x = e^x, \quad
\frac{d}{dx} a^x = a^x \ln a, \quad
\frac{d}{dx} \ln x = \frac{1}{x}
\]

#### Product & Quotient Rules

\[
(fg)' = f'g + fg' \qquad \text{(Product rule)}
\]
\[
\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2} \qquad \text{(Quotient rule)}
\]

#### Chain Rule

\[
\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x) = \frac{df}{dg} \cdot \frac{dg}{dx}
\]

**Intuition:** Think of a chain of effects: \( x \) affects \( g \), which affects \( f \). The total rate of change = (how fast \( f \) changes w.r.t. \( g \)) × (how fast \( g \) changes w.r.t. \( x \)). In backprop, we multiply local Jacobians along the computation graph.

**This is the foundation of backpropagation!**

**Common chain rule pitfalls:**
- **Wrong variable:** \( \frac{d}{dx}[f(g(x))] \neq f'(x) \cdot g'(x) \) — you must evaluate \( f' \) at \( g(x) \), not at \( x \).
- **Forgetting inner derivative:** \( \frac{d}{dx}\left[\sin(x^2)\right] = \cos(x^2) \cdot 2x \), not \( \cos(x^2) \).
- **Multivariable chain rule:** For \( z = f(u,v) \), \( u = g(x,y) \), use \( \frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x} \) — sum over all paths from \( x \) to \( z \).

**Worked example:** Compute \( \frac{d}{dx}\left[\ln(1 + e^{2x})\right] \):
- Outer: \( \frac{d}{du}\ln u = \frac{1}{u} \), so \( \frac{1}{1+e^{2x}} \)
- Inner: \( \frac{d}{dx}(1 + e^{2x}) = 2e^{2x} \)
- Result: \( \frac{2e^{2x}}{1 + e^{2x}} = \frac{2}{1 + e^{-2x}} \) (alternative form)

```python
import sympy as sp

x = sp.Symbol('x')

# Power rule: d/dx[x^n] = n*x^(n-1)
print("Power rule:")
for n in [2, 3, 0.5, -1]:
    expr = x**n
    deriv = sp.diff(expr, x)
    print(f"  d/dx[x^{n}] = {deriv}")

# Chain rule examples
print("\nChain rule examples:")
exprs = [
    sp.sin(x**2),         # d/dx[sin(x²)] = 2x*cos(x²)
    sp.exp(-x**2),        # d/dx[e^(-x²)] = -2x*e^(-x²)
    sp.log(1 + x**2),     # d/dx[ln(1+x²)] = 2x/(1+x²)
    sp.sqrt(x**2 + 1),    # d/dx[√(x²+1)] = x/√(x²+1)
]
for expr in exprs:
    deriv = sp.diff(expr, x)
    print(f"  d/dx[{expr}] = {sp.simplify(deriv)}")

# Derivative of sigmoid (important in backprop!)
sigma = 1 / (1 + sp.exp(-x))
d_sigma = sp.diff(sigma, x)
print(f"\nd/dx[sigmoid] = {sp.simplify(d_sigma)}")
# = σ(x)(1 - σ(x))
```

### 2.3 Important Derivatives for ML

```python
import numpy as np

# Common activation function derivatives
def d_sigmoid(x):
    s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    return s * (1 - s)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def d_relu(x):
    return (x > 0).astype(float)  # subgradient: 0 at x=0

def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def d_softplus(x):
    """Softplus: log(1+e^x), derivative = sigmoid(x)"""
    return 1 / (1 + np.exp(-x))

x = np.linspace(-4, 4, 1000)
print("Derivatives at x=0:")
print(f"  d/dx[sigmoid] at 0: {d_sigmoid(0):.4f} (max = 0.25)")
print(f"  d/dx[tanh]    at 0: {d_tanh(0):.4f}   (max = 1)")
print(f"  d/dx[ReLU]    at 0: {d_relu(0):.4f}   (subgradient = 0)")

# Vanishing gradient problem: sigmoid saturates
print("\nSigmoid derivative saturation:")
for x_val in [-10, -5, 0, 5, 10]:
    print(f"  x={x_val:4d}: d_sigmoid = {d_sigmoid(x_val):.6f}")
```

---

## 3. Partial Derivatives & Gradients

### 3.1 Partial Derivatives

For \( f: \mathbb{R}^n \to \mathbb{R} \), the **partial derivative** w.r.t. \( x_i \):
\[
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}
\]

**Interpretation:** Rate of change of \( f \) when varying **only \( x_i \)** (holding others fixed).

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

# f(x, y) = x²y + y³ - 2xy²
f = x**2 * y + y**3 - 2*x*y**2

df_dx = sp.diff(f, x)  # ∂f/∂x
df_dy = sp.diff(f, y)  # ∂f/∂y

print(f"f(x,y) = {f}")
print(f"∂f/∂x = {df_dx}")   # = 2xy - 2y²
print(f"∂f/∂y = {df_dy}")   # = x² + 3y² - 4xy

# Evaluate at (x=2, y=1)
point = {x: 2, y: 1}
print(f"\nAt (2,1):")
print(f"  ∂f/∂x = {df_dx.subs(point)}")
print(f"  ∂f/∂y = {df_dy.subs(point)}")

# 3D example: f(x,y,z) = x²yz + xy²z²
f3 = x**2 * y * z + x * y**2 * z**2
print(f"\nf(x,y,z) = {f3}")
print(f"∂f/∂x = {sp.diff(f3, x)}")
print(f"∂f/∂y = {sp.diff(f3, y)}")
print(f"∂f/∂z = {sp.diff(f3, z)}")
```

### 3.2 The Gradient

The **gradient** is the vector of all partial derivatives:
\[
\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}
\]

**Key properties:**
1. \( \nabla f(x) \) points in the direction of **steepest ascent**
2. \( -\nabla f(x) \) points in the direction of **steepest descent**
3. \( \nabla f(x) \perp \) level curves (contour lines)
4. \( \|\nabla f(x)\| \) = rate of change in steepest direction

**Intuition:** The gradient answers: "Which direction should I move to increase \( f \) most?" Since the directional derivative is \( D_d f = \nabla f \cdot \hat{d} \), it's maximized when \( \hat{d} \) aligns with \( \nabla f \). The gradient is perpendicular to level curves because along a level curve \( f \) doesn't change, so the direction of change must be orthogonal to it.

```python
import numpy as np

# Numerical gradient (for any function)
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient using central differences."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Example: f(x, y) = x² + 2y² - xy
def f(x_vec):
    x, y = x_vec
    return x**2 + 2*y**2 - x*y

def grad_f_analytic(x_vec):
    x, y = x_vec
    return np.array([2*x - y, 4*y - x])

test_points = [np.array([1.0, 1.0]),
               np.array([2.0, -1.0]),
               np.array([0.0, 3.0])]

print("Gradient comparison (numeric vs analytic):")
for pt in test_points:
    g_num = numerical_gradient(f, pt)
    g_ana = grad_f_analytic(pt)
    print(f"  x={pt}: numerical={g_num}, analytic={g_ana}, "
          f"error={np.max(np.abs(g_num - g_ana)):.2e}")

# Gradient field visualization concept
print("\nGradient at minimum:")
# At minimum, gradient = 0
# Solve: 2x - y = 0, 4y - x = 0
# From first: y = 2x. Sub: 4(2x) - x = 7x = 0 → x=0, y=0
# So minimum is at origin (makes sense for f = x² + 2y² - xy)
print(f"  grad_f(0,0) = {grad_f_analytic(np.array([0., 0.]))}")
```

### 3.3 Gradient Descent

\[
x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)})
\]

**Why -gradient?** We want to minimize \( f \), so we move **opposite** to the direction of steepest ascent.

**Step size (learning rate) \( \eta \):**
- Too large: diverge or oscillate
- Too small: converge slowly
- Optimal (Armijo/Wolfe line search): adapts step size

```python
def gradient_descent(f, grad_f, x0, lr=0.1, n_iters=200, tol=1e-8):
    """Vanilla gradient descent with convergence tracking."""
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    losses = [f(x)]

    for i in range(n_iters):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            print(f"Converged at iteration {i}")
            break
        x = x - lr * g
        trajectory.append(x.copy())
        losses.append(f(x))

    return x, np.array(trajectory), losses

# Minimize f(x,y) = x² + 2y² - xy
x_opt, traj, losses = gradient_descent(f, grad_f_analytic, [3.0, 2.0], lr=0.1)
print(f"Minimum at: {x_opt} (should be [0, 0])")
print(f"Function value: {f(x_opt):.8f}")
print(f"Gradient at min: {grad_f_analytic(x_opt)}")
print(f"Convergence: {len(losses)} iterations")
```

---

## 4. Directional Derivatives

### 4.1 Definition

The **directional derivative** of \( f \) at \( x \) in direction \( d \):
\[
D_d f(x) = \lim_{h \to 0} \frac{f(x + hd) - f(x)}{h} = \nabla f(x) \cdot \hat{d}
\]

Where \( \hat{d} = d/\|d\| \) is the unit direction vector.

**Key insight:** \( D_d f(x) = \|\nabla f\| \cos\theta \) where \( \theta \) is angle between \( \nabla f \) and \( d \).

Maximum directional derivative = \( \|\nabla f\| \) (achieved when \( d \parallel \nabla f \)).

```python
import numpy as np

# f(x, y) = x² + y²
def grad_circle(x_vec):
    return 2 * x_vec  # [2x, 2y]

point = np.array([1.0, 2.0])
grad = grad_circle(point)

# Directional derivatives in various directions
directions = {
    'gradient direction': grad / np.linalg.norm(grad),
    'neg gradient':       -grad / np.linalg.norm(grad),
    'x-axis':             np.array([1.0, 0.0]),
    'y-axis':             np.array([0.0, 1.0]),
    '45° diagonal':       np.array([1.0, 1.0]) / np.sqrt(2),
}

print(f"Point: {point}, Gradient: {grad}, ||∇f|| = {np.linalg.norm(grad):.4f}")
print("\nDirectional derivatives:")
for name, d_unit in directions.items():
    D_d = np.dot(grad, d_unit)
    print(f"  {name:20s}: {D_d:.4f}")

# Maximum directional derivative = ||gradient||
print(f"\nMax directional deriv = ||∇f|| = {np.linalg.norm(grad):.4f}")
```

---

## 5. Jacobian & Hessian Matrices

### 5.1 Jacobian Matrix

For \( f: \mathbb{R}^n \to \mathbb{R}^m \), the **Jacobian** is:
\[
J = \frac{\partial f}{\partial x} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix} \in \mathbb{R}^{m \times n}
\]

**Row \( i \)**: Gradient of \( f_i \) w.r.t. all inputs.
**Column \( j \)**: How all outputs change w.r.t. \( x_j \).

**Chain rule in matrix form:**
\[
\frac{\partial(g \circ f)}{\partial x} = J_g(f(x)) \cdot J_f(x)
\]

```python
import sympy as sp
import numpy as np

x1, x2 = sp.symbols('x1 x2')

# f: R² → R³
f1 = x1**2 + x2
f2 = x1 * x2**2
f3 = sp.sin(x1) + sp.cos(x2)

# Jacobian
J = sp.Matrix([
    [sp.diff(f1, x1), sp.diff(f1, x2)],
    [sp.diff(f2, x1), sp.diff(f2, x2)],
    [sp.diff(f3, x1), sp.diff(f3, x2)],
])
print("Jacobian J(f):")
print(J)

# Evaluate at (1, 2)
J_eval = J.subs({x1: 1, x2: 2})
print(f"\nJ at (1,2):\n{np.array(J_eval.tolist(), dtype=float)}")

# Numerical Jacobian
def numerical_jacobian(f_vec, x, h=1e-5):
    """f_vec: function returning vector, x: input vector."""
    n = len(x)
    f0 = np.array(f_vec(x), dtype=float)
    m = len(f0)
    J_num = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += h
        x_minus = x.copy()
        x_minus[j] -= h
        J_num[:, j] = (np.array(f_vec(x_plus)) - np.array(f_vec(x_minus))) / (2*h)
    return J_num

def f_vec(x):
    return [x[0]**2 + x[1], x[0]*x[1]**2, np.sin(x[0]) + np.cos(x[1])]

J_numerical = numerical_jacobian(f_vec, np.array([1.0, 2.0]))
print(f"\nNumerical Jacobian at (1,2):\n{J_numerical}")
```

### 5.2 Hessian Matrix

For \( f: \mathbb{R}^n \to \mathbb{R} \), the **Hessian** contains all second-order partials:
\[
H(x) = \nabla^2 f(x) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & & \ddots
\end{pmatrix} \in \mathbb{R}^{n \times n}
\]

**Symmetry:** If \( f \) has continuous second partial derivatives (Schwarz's theorem):
\[
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
\]

**Critical point classification:**
- \( H \succ 0 \) (positive definite): **local minimum**
- \( H \prec 0 \) (negative definite): **local maximum**
- \( H \) indefinite: **saddle point**
- \( H \succeq 0 \) or \( H \preceq 0 \): inconclusive (could be flat)

```python
import sympy as sp
import numpy as np

x, y = sp.symbols('x y')

# Example: f(x,y) = x³ - 3xy + y³
f_sym = x**3 - 3*x*y + y**3

# First-order partials (gradient)
df_dx = sp.diff(f_sym, x)
df_dy = sp.diff(f_sym, y)
print(f"∂f/∂x = {df_dx}")   # 3x² - 3y
print(f"∂f/∂y = {df_dy}")   # -3x + 3y²

# Hessian
H11 = sp.diff(f_sym, x, 2)
H12 = sp.diff(sp.diff(f_sym, x), y)
H21 = sp.diff(sp.diff(f_sym, y), x)
H22 = sp.diff(f_sym, y, 2)

H_sym = sp.Matrix([[H11, H12], [H21, H22]])
print(f"\nHessian:\n{H_sym}")

# Find critical points: solve ∇f = 0
critical = sp.solve([df_dx, df_dy], [x, y])
print(f"\nCritical points: {critical}")

# Classify each critical point
for cp in critical:
    H_at_cp = H_sym.subs({x: cp[0], y: cp[1]})
    H_np = np.array(H_at_cp.tolist(), dtype=float)
    eigvals = np.linalg.eigvals(H_np)
    det_H = float(H_at_cp.det())
    trace_H = float(H_at_cp.trace())
    print(f"\nAt {cp}:")
    print(f"  H = {H_np}")
    print(f"  det(H)={det_H:.4f}, trace(H)={trace_H:.4f}")
    print(f"  eigenvalues: {eigvals.real}")
    if det_H > 0 and trace_H > 0:
        print("  → LOCAL MINIMUM")
    elif det_H > 0 and trace_H < 0:
        print("  → LOCAL MAXIMUM")
    elif det_H < 0:
        print("  → SADDLE POINT")
    else:
        print("  → INCONCLUSIVE")
```

### 5.3 Newton's Method (Using Hessian)

**Update rule:**
\[
x^{(t+1)} = x^{(t)} - H(x^{(t)})^{-1} \nabla f(x^{(t)})
\]

**Convergence:** Quadratic (vs linear for gradient descent). But computing/inverting Hessian is \( O(n^3) \) — impractical for large \( n \).

**Quasi-Newton methods** (L-BFGS, BFGS): Approximate \( H^{-1} \) efficiently.

```python
def newtons_method(f, grad_f, hessian_f, x0, n_iters=50, tol=1e-8):
    """Newton's method for optimization."""
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for i in range(n_iters):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            print(f"Converged at iteration {i}")
            break
        H = hessian_f(x)
        # Solve H * delta = g (instead of computing H^{-1})
        delta = np.linalg.solve(H, g)
        x = x - delta
        history.append(x.copy())

    return x, history

# Minimize f(x,y) = (x-2)² + (y+1)² + xy (bowl with cross-term)
def f_quad(v):
    x, y = v
    return (x-2)**2 + (y+1)**2 + x*y

def grad_quad(v):
    x, y = v
    return np.array([2*(x-2) + y, 2*(y+1) + x])

def hess_quad(v):
    return np.array([[2, 1], [1, 2]])

x_gd, traj_gd, _ = gradient_descent(f_quad, grad_quad, [5.0, 5.0], lr=0.1)
x_nt, traj_nt = newtons_method(f_quad, grad_quad, hess_quad, [5.0, 5.0])

print(f"Gradient descent: {x_gd.round(6)} in {len(traj_gd)} steps")
print(f"Newton's method:  {x_nt.round(6)} in {len(traj_nt)} steps")
print("(Newton converges in fewer steps, but each step is more expensive)")
```

---

## 6. Taylor Series

### 6.1 Univariate Taylor Series

\[
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n = f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2 + \cdots
\]

**Maclaurin series** (a = 0):
\[
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}, \quad
\sin x = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}, \quad
\cos x = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}
\]
\[
\ln(1+x) = \sum_{n=1}^{\infty} \frac{(-1)^{n-1} x^n}{n} \quad (|x| < 1)
\]

```python
import numpy as np
import sympy as sp

# Taylor approximations to e^x around x=0
def taylor_exp(x, n_terms):
    """n-th order Taylor approximation of e^x."""
    result = 0.0
    factorial = 1
    for n in range(n_terms):
        if n > 0:
            factorial *= n
        result += x**n / factorial
    return result

x_val = 1.5
print(f"True e^{x_val} = {np.exp(x_val):.10f}")
for n in [1, 2, 3, 5, 10]:
    approx = taylor_exp(x_val, n)
    error = abs(approx - np.exp(x_val))
    print(f"  {n} terms: {approx:.10f}, error={error:.2e}")

# Symbolic Taylor series
x = sp.Symbol('x')
f_exp = sp.exp(x)
print("\nSymbolic Taylor expansion of e^x:")
print(sp.series(f_exp, x, 0, n=6))

# Taylor expansion of sin(x) around x=π/4
a = sp.pi / 4
print(f"\nTaylor expansion of sin(x) around π/4:")
print(sp.series(sp.sin(x), x, a, n=5))
```

### 6.2 Multivariate Taylor Series

For \( f: \mathbb{R}^n \to \mathbb{R} \), the second-order Taylor approximation around \( a \):
\[
f(x) \approx f(a) + \nabla f(a)^T (x-a) + \frac{1}{2}(x-a)^T H(a)(x-a)
\]

**ML applications:**
- **Quadratic approximation** of loss landscape
- **Newton's method** derives from this
- **Curvature analysis** for optimization
- **Natural gradient** uses Fisher information matrix (a second-order method)

```python
# Second-order Taylor approximation
def f_2d(v):
    x, y = v
    return np.sin(x) * np.cos(y)

def grad_2d(v):
    x, y = v
    return np.array([np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y)])

def hess_2d(v):
    x, y = v
    return np.array([
        [-np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)],
        [-np.cos(x) * np.sin(y), -np.sin(x) * np.cos(y)]
    ])

a = np.array([np.pi/4, np.pi/4])

def taylor_2nd_order(x_vec, a_vec, f, grad_f, hess_f):
    """Second-order Taylor approximation of f at a."""
    d = x_vec - a_vec
    return (f(a_vec) +
            np.dot(grad_f(a_vec), d) +
            0.5 * d @ hess_f(a_vec) @ d)

# Compare Taylor approximation to true value
test_points = [a + np.array([0.1, 0.1]),
               a + np.array([0.5, 0.3]),
               a + np.array([1.0, 1.0])]

print("Taylor approximation accuracy:")
for p in test_points:
    true_val = f_2d(p)
    taylor_val = taylor_2nd_order(p, a, f_2d, grad_2d, hess_2d)
    linear_val = f_2d(a) + np.dot(grad_2d(a), p - a)  # 1st order
    print(f"  ||d||={np.linalg.norm(p-a):.2f}: "
          f"true={true_val:.6f}, taylor2={taylor_val:.6f} "
          f"(err={abs(true_val-taylor_val):.2e}), "
          f"linear err={abs(true_val-linear_val):.2e}")
```

---

## 7. Integration

### 7.1 Definite & Indefinite Integrals

**Fundamental Theorem of Calculus:**
\[
\int_a^b f'(x)\, dx = f(b) - f(a), \quad \frac{d}{dx}\int_a^x f(t)\, dt = f(x)
\]

**Common integrals:**
\[
\int x^n\, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)
\]
\[
\int e^x\, dx = e^x + C, \quad \int \frac{1}{x}\, dx = \ln|x| + C
\]
\[
\int \sin x\, dx = -\cos x + C, \quad \int \cos x\, dx = \sin x + C
\]

```python
from scipy import integrate
import sympy as sp
import numpy as np

x = sp.Symbol('x')

# Symbolic integration
exprs = [x**3, sp.exp(-x**2/2), sp.sin(x)**2, 1/(1+x**2)]
print("Indefinite integrals:")
for expr in exprs:
    integral = sp.integrate(expr, x)
    print(f"  ∫ {expr} dx = {integral}")

# Definite integrals
print("\nDefinite integrals:")
definite = [
    (x**2, 0, 1),         # = 1/3
    (sp.sin(x), 0, sp.pi), # = 2
    (sp.exp(-x), 0, sp.oo), # = 1
]
for expr, a, b in definite:
    val = sp.integrate(expr, (x, a, b))
    print(f"  ∫_{{{a}}}^{{{b}}} {expr} dx = {val}")

# Numerical integration
from scipy.integrate import quad, dblquad, trapezoid

def f_numeric(x): return np.exp(-x**2)  # Gaussian integral

result, error = quad(f_numeric, -np.inf, np.inf)
print(f"\n∫_(-∞)^(∞) e^(-x²) dx = {result:.8f} (= √π = {np.sqrt(np.pi):.8f})")

# Monte Carlo integration
def monte_carlo_integrate(f, a, b, n=100000):
    """Monte Carlo: E[f(X)] ≈ (b-a) * mean(f(U)) where U~Uniform[a,b]"""
    samples = np.random.uniform(a, b, n)
    return (b - a) * np.mean(f(samples))

mc_result = monte_carlo_integrate(lambda x: x**2, 0, 1)
print(f"\nMonte Carlo ∫₀¹ x² dx ≈ {mc_result:.6f} (true = 1/3 ≈ {1/3:.6f})")
```

### 7.2 Integration Techniques

```python
import sympy as sp

x, t, u = sp.symbols('x t u', positive=True)

# Integration by parts: ∫ u dv = uv - ∫ v du
# ∫ x*e^x dx
f_ibp = x * sp.exp(x)
result_ibp = sp.integrate(f_ibp, x)
print(f"∫ x·eˣ dx = {result_ibp}")

# U-substitution: ∫ 2x * cos(x²) dx, u = x², du = 2x dx
# = ∫ cos(u) du = sin(u) = sin(x²)
f_sub = 2*x * sp.cos(x**2)
print(f"∫ 2x·cos(x²) dx = {sp.integrate(f_sub, x)}")

# Partial fractions: ∫ 1/(x²-1) dx
x_sym = sp.Symbol('x')
f_pf = 1 / (x_sym**2 - 1)
apart_decomp = sp.apart(f_pf, x_sym)
print(f"Partial fraction: 1/(x²-1) = {apart_decomp}")
print(f"∫ 1/(x²-1) dx = {sp.integrate(f_pf, x_sym)}")
```

### 7.3 Integration in ML: Expected Values

```python
from scipy import integrate
import numpy as np
from scipy.stats import norm

# Expected value: E[X] = ∫ x·p(x) dx
# E[X²] = ∫ x²·p(x) dx
# Var[X] = E[X²] - (E[X])²

mu, sigma = 2.0, 1.5
gaussian = norm(mu, sigma)

# E[X] for X ~ N(μ, σ²)
E_X, _ = integrate.quad(lambda x: x * gaussian.pdf(x), -np.inf, np.inf)
print(f"E[X] = {E_X:.6f} (true: {mu})")

# E[X²]
E_X2, _ = integrate.quad(lambda x: x**2 * gaussian.pdf(x), -np.inf, np.inf)
Var_X = E_X2 - E_X**2
print(f"Var[X] = {Var_X:.6f} (true: {sigma**2})")

# Entropy: H(X) = -∫ p(x) log p(x) dx
# For Gaussian: H = 0.5 * log(2πeσ²)
H_X, _ = integrate.quad(lambda x: -gaussian.pdf(x) * np.log(gaussian.pdf(x) + 1e-300),
                         mu - 10*sigma, mu + 10*sigma)
H_true = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
print(f"Entropy H(X) = {H_X:.6f} (true: {H_true:.6f})")

# KL divergence: KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
P = norm(0, 1)
Q = norm(0.5, 1.5)

KL_PQ, _ = integrate.quad(
    lambda x: P.pdf(x) * (np.log(P.pdf(x) + 1e-300) - np.log(Q.pdf(x) + 1e-300)),
    -10, 10
)
print(f"\nKL(N(0,1) || N(0.5,1.5)) = {KL_PQ:.6f}")
```

---

## 8. Multivariate Calculus

### 8.1 Gradient, Divergence, Curl

For scalar field \( f: \mathbb{R}^3 \to \mathbb{R} \) and vector field \( F = (F_1, F_2, F_3): \mathbb{R}^3 \to \mathbb{R}^3 \):

**Gradient** (scalar → vector): \( \nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}\right) \)

**Divergence** (vector → scalar): \( \nabla \cdot F = \frac{\partial F_1}{\partial x} + \frac{\partial F_2}{\partial y} + \frac{\partial F_3}{\partial z} \)

**Curl** (vector → vector): \( \nabla \times F = \left(\frac{\partial F_3}{\partial y} - \frac{\partial F_2}{\partial z},\; \frac{\partial F_1}{\partial z} - \frac{\partial F_3}{\partial x},\; \frac{\partial F_2}{\partial x} - \frac{\partial F_1}{\partial y}\right) \)

**Laplacian** (scalar → scalar): \( \Delta f = \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2} \)

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

# Scalar field: f(x,y,z) = x²yz + e^(xy)
f_3d = x**2 * y * z + sp.exp(x*y)

# Gradient
grad_f = [sp.diff(f_3d, var) for var in [x, y, z]]
print("Gradient of f:")
for var, g in zip(['∂f/∂x', '∂f/∂y', '∂f/∂z'], grad_f):
    print(f"  {var} = {g}")

# Laplacian
laplacian = sum(sp.diff(f_3d, var, 2) for var in [x, y, z])
print(f"\nLaplacian ∇²f = {sp.simplify(laplacian)}")

# Vector field: F(x,y,z) = (x², xy, xz)
F1, F2, F3 = x**2, x*y, x*z

# Divergence
div_F = sp.diff(F1, x) + sp.diff(F2, y) + sp.diff(F3, z)
print(f"\nDivergence ∇·F = {div_F}")  # = 2x + x + x = 4x

# Curl
curl_F = [
    sp.diff(F3, y) - sp.diff(F2, z),
    sp.diff(F1, z) - sp.diff(F3, x),
    sp.diff(F2, x) - sp.diff(F1, y),
]
print(f"Curl ∇×F = ({curl_F[0]}, {curl_F[1]}, {curl_F[2]})")
```

### 8.2 Multivariable Chain Rule (General Form)

If \( z = f(u, v) \) where \( u = g(x, y) \) and \( v = h(x, y) \):
\[
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial z}{\partial v}\frac{\partial v}{\partial x}
\]

In **Jacobian matrix form** (composition \( h = f \circ g \)):
\[
J_h(x) = J_f(g(x)) \cdot J_g(x)
\]

This is exactly **backpropagation** — chain rule applied layer by layer.

```python
import numpy as np

# Backprop example: y = sigmoid(w @ x + b)
# dy/dw = sigmoid'(z) * x, where z = w @ x + b

def sigmoid(z): return 1 / (1 + np.exp(-z))
def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward pass
w = np.array([0.5, -0.3, 0.8])
x = np.array([1.0, 2.0, 0.5])
b = 0.1

z = w @ x + b                  # Linear combination
y = sigmoid(z)                  # Activation

# Backward pass (chain rule)
dL_dy = 2 * (y - 1.0)         # dLoss/dy for MSE, y_true=1
dy_dz = d_sigmoid(z)           # ∂y/∂z
dz_dw = x                      # ∂z/∂w
dz_db = 1.0                    # ∂z/∂b

# Chain rule: dL/dw = (dL/dy) * (dy/dz) * (dz/dw)
dL_dw = dL_dy * dy_dz * dz_dw
dL_db = dL_dy * dy_dz * dz_db

print(f"Forward: z={z:.4f}, y={y:.4f}")
print(f"Backward: dL/dy={dL_dy:.4f}, dy/dz={dy_dz:.4f}")
print(f"dL/dw = {dL_dw}")
print(f"dL/db = {dL_db:.4f}")

# Verify with numerical gradient
def loss(w, x, b, y_true):
    z = w @ x + b
    y = sigmoid(z)
    return (y - y_true)**2

h = 1e-5
y_true = 1.0
grad_w_numerical = np.zeros_like(w)
for i in range(len(w)):
    w_plus = w.copy(); w_plus[i] += h
    w_minus = w.copy(); w_minus[i] -= h
    grad_w_numerical[i] = (loss(w_plus, x, b, y_true) - loss(w_minus, x, b, y_true)) / (2*h)

print(f"\nAnalytic  dL/dw: {dL_dw}")
print(f"Numerical dL/dw: {grad_w_numerical}")
print(f"Max error: {np.max(np.abs(dL_dw - grad_w_numerical)):.2e}")
```

---

## 9. Optimization Theory

### 9.1 Convexity

**Convex function:** \( f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y) \) for all \( \lambda \in [0,1] \)

Equivalently (if \( f \) differentiable): \( f(y) \geq f(x) + \nabla f(x)^T(y-x) \) (tangent line lower bounds)

Equivalently (if \( f \) twice differentiable): \( H(x) \succeq 0 \) for all \( x \)

**Why convexity matters in ML:**
- **Any local minimum = global minimum** for convex problems
- Gradient descent guaranteed to converge (with appropriate lr)
- Linear/logistic regression with convex loss → globally optimal solution

```python
import numpy as np

def is_convex_check(f, x_range, n_samples=100):
    """Probabilistic check of convexity via Jensen's inequality."""
    rng = np.random.RandomState(42)
    violations = 0
    for _ in range(1000):
        x1 = rng.uniform(*x_range)
        x2 = rng.uniform(*x_range)
        lam = rng.uniform(0, 1)
        lhs = f(lam * x1 + (1 - lam) * x2)
        rhs = lam * f(x1) + (1 - lam) * f(x2)
        if lhs > rhs + 1e-8:
            violations += 1
    return violations == 0

# Convex functions
print(f"f(x)=x² convex?     {is_convex_check(lambda x: x**2, (-5, 5))}")
print(f"f(x)=e^x convex?    {is_convex_check(lambda x: np.exp(x), (-5, 5))}")
print(f"f(x)=|x| convex?    {is_convex_check(np.abs, (-5, 5))}")

# Non-convex functions
print(f"f(x)=sin(x) convex? {is_convex_check(np.sin, (0, 2*np.pi))}")
print(f"f(x)=x³ convex?     {is_convex_check(lambda x: x**3, (-5, 5))}")
```

### 9.2 Constrained Optimization: Lagrange Multipliers

**Problem:** Minimize \( f(x) \) subject to \( g(x) = 0 \)

**Lagrangian:** \( \mathcal{L}(x, \lambda) = f(x) - \lambda g(x) \)

**KKT conditions** (necessary for optimality):
\[
\nabla_x f(x^*) = \lambda \nabla_x g(x^*) \qquad \text{(stationarity)}
\]
\[
g(x^*) = 0 \qquad \text{(primal feasibility)}
\]

For **inequality constraints** \( g(x) \leq 0 \):
- \( \nabla_x f = \lambda \nabla_x g \)
- \( g(x^*) \leq 0 \) (primal feasibility)
- \( \lambda \geq 0 \) (dual feasibility)
- \( \lambda g(x^*) = 0 \) (complementary slackness)

```python
from scipy.optimize import minimize
import numpy as np

# Minimize f(x,y) = x² + y² subject to x + y = 1
def objective(v): return v[0]**2 + v[1]**2
def constraint_eq(v): return v[0] + v[1] - 1

result = minimize(objective, [0.5, 0.5],
                  constraints={'type': 'eq', 'fun': constraint_eq},
                  method='SLSQP')
print(f"Constrained min on x+y=1:")
print(f"  Solution: x={result.x[0]:.6f}, y={result.x[1]:.6f}")
print(f"  Value: f={result.fun:.6f}")
print(f"  Analytic: (0.5, 0.5), f=0.5")

# Inequality constraint: min x²+y² s.t. x+y ≥ 2
def constraint_ineq(v): return v[0] + v[1] - 2

result_ineq = minimize(objective, [1.0, 1.0],
                        constraints={'type': 'ineq', 'fun': constraint_ineq},
                        method='SLSQP')
print(f"\nConstrained min on x+y≥2:")
print(f"  Solution: x={result_ineq.x[0]:.6f}, y={result_ineq.x[1]:.6f}")

# SVM as constrained optimization
# min 0.5||w||² s.t. y_i(w^T x_i + b) ≥ 1
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X_svm, y_svm = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                    random_state=42)
y_svm = 2 * y_svm - 1  # Convert to {-1, +1}
svm = SVC(kernel='linear', C=1e6)  # Hard margin (large C)
svm.fit(X_svm, y_svm)
print(f"\nSVM weights (from constrained opt): w={svm.coef_[0].round(4)}")
```

### 9.3 Gradient Descent Variants

```python
import numpy as np

def sgd(X, y, loss_grad, w0, lr=0.01, n_epochs=100, batch_size=32):
    """Stochastic/Mini-batch Gradient Descent."""
    w = w0.copy()
    n = len(y)
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]

        epoch_loss = 0
        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]
            grad, loss = loss_grad(X_batch, y_batch, w)
            w -= lr * grad
            epoch_loss += loss * len(y_batch)

        losses.append(epoch_loss / n)
    return w, losses

def adam(X, y, loss_grad, w0, lr=0.001, beta1=0.9, beta2=0.999,
         eps=1e-8, n_epochs=100, batch_size=32):
    """Adam optimizer."""
    w = w0.copy()
    m = np.zeros_like(w)  # First moment (mean)
    v = np.zeros_like(w)  # Second moment (variance)
    t = 0
    n = len(y)
    losses = []

    for epoch in range(n_epochs):
        perm = np.random.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]
        epoch_loss = 0

        for i in range(0, n, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]
            grad, loss = loss_grad(X_batch, y_batch, w)

            t += 1
            m = beta1 * m + (1 - beta1) * grad          # Update biased first moment
            v = beta2 * v + (1 - beta2) * grad**2       # Update biased second moment
            m_hat = m / (1 - beta1**t)                   # Bias correction
            v_hat = v / (1 - beta2**t)                   # Bias correction

            w -= lr * m_hat / (np.sqrt(v_hat) + eps)
            epoch_loss += loss * len(y_batch)

        losses.append(epoch_loss / n)
    return w, losses

# Compare optimizers on logistic regression
from sklearn.datasets import make_classification

np.random.seed(42)
X_data, y_data = make_classification(n_samples=1000, n_features=10, random_state=42)
X_data = np.hstack([np.ones((len(X_data), 1)), X_data])  # Add bias

def logistic_loss_grad(X, y, w):
    """Returns (gradient, loss) for logistic regression."""
    z = X @ w
    pred = 1 / (1 + np.exp(-z))
    loss = -np.mean(y * np.log(pred + 1e-10) + (1-y) * np.log(1-pred + 1e-10))
    grad = X.T @ (pred - y) / len(y)
    return grad, loss

w0 = np.zeros(X_data.shape[1])

w_sgd, losses_sgd = sgd(X_data, y_data, logistic_loss_grad, w0, lr=0.1, n_epochs=50)
w_adam, losses_adam = adam(X_data, y_data, logistic_loss_grad, w0, lr=0.01, n_epochs=50)

print(f"SGD  final loss: {losses_sgd[-1]:.6f}")
print(f"Adam final loss: {losses_adam[-1]:.6f}")
```

---

## 10. Automatic Differentiation

**Connection to calculus:** AD computes *exact* derivatives by applying the chain rule mechanically to the computational graph. Unlike symbolic differentiation, AD doesn't expand expressions into huge formulas; it evaluates derivatives alongside function values. Unlike finite differences, AD has no truncation error and costs only a small constant factor more than the forward pass.

### 10.1 The Three Ways to Differentiate

| Method | How | Exact? | Speed | Memory | Use |
|--------|-----|--------|-------|--------|-----|
| Symbolic | Algebraic manipulation | Yes | Slow | High | Simple expressions |
| Numerical | Finite differences | No (approx) | Medium | Low | Gradient checking |
| Automatic | Computational graph | Yes | Fast | Medium | ML training |

### 10.2 Forward Mode AD

Track \( (f(x), f'(x)) \) simultaneously as dual numbers.

For input \( x = a + \dot{a}\epsilon \) (where \( \epsilon^2 = 0 \)):
\[
f(x) = f(a) + f'(a)\dot{a}\epsilon
\]

**Cost:** One pass computes one directional derivative. Efficient when \( n \ll m \) (few inputs, many outputs).

```python
class DualNumber:
    """Dual number for forward-mode AD."""
    def __init__(self, val, deriv=0.0):
        self.val = val    # Real part
        self.deriv = deriv  # Infinitesimal part (derivative)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.val + other, self.deriv)
        return DualNumber(self.val + other.val, self.deriv + other.deriv)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.val * other, self.deriv * other)
        # Product rule: (uv)' = u'v + uv'
        return DualNumber(self.val * other.val,
                         self.deriv * other.val + self.val * other.deriv)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return DualNumber(self.val - other, self.deriv)
        return DualNumber(self.val - other.val, self.deriv - other.deriv)

    def __pow__(self, n):
        # Power rule: d/dx[x^n] = n*x^(n-1)
        return DualNumber(self.val**n, n * self.val**(n-1) * self.deriv)

    def __repr__(self):
        return f"DualNumber(val={self.val:.4f}, deriv={self.deriv:.4f})"

def dual_exp(d): return DualNumber(np.exp(d.val), np.exp(d.val) * d.deriv)
def dual_sin(d): return DualNumber(np.sin(d.val), np.cos(d.val) * d.deriv)
def dual_cos(d): return DualNumber(np.cos(d.val), -np.sin(d.val) * d.deriv)

# Compute d/dx [x³ + sin(x²)] at x=2
x = DualNumber(2.0, 1.0)  # x=2, dx/dx=1

# f(x) = x³ + sin(x²)
f = x**3 + dual_sin(x**2)
print(f"f(2) = {f.val:.6f}")
print(f"f'(2) = {f.deriv:.6f}")

# True derivative: 3x² + 2x*cos(x²) at x=2
x_val = 2.0
true_deriv = 3*x_val**2 + 2*x_val*np.cos(x_val**2)
print(f"True f'(2) = {true_deriv:.6f}")
```

### 10.3 Reverse Mode AD (Backpropagation)

**Key insight:** Compute \( \frac{\partial L}{\partial w} \) for ALL weights in ONE backward pass.

**Algorithm:**
1. Forward pass: Compute output, save intermediate values
2. Backward pass: Propagate gradients from output to input using chain rule

**Cost:** One backward pass = one forward pass. Efficient for \( m \ll n \) (one loss, many parameters).

```python
class Variable:
    """Simple autograd implementation (like PyTorch's tensor)."""
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=float)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None  # Default: leaf node
        self._prev = set()             # Predecessors in computation graph

    def backward(self):
        """Topological sort + reverse-mode backprop."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        out = Variable(self.data + other.data)
        out._prev = {self, other}
        def backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else 0) + out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        out = Variable(self.data * other.data)
        out._prev = {self, other}
        def backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + other.data * out.grad
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else 0) + self.data * out.grad
        out._backward = backward
        return out

    def relu(self):
        out = Variable(np.maximum(0, self.data))
        out._prev = {self}
        def backward():
            if self.requires_grad:
                mask = (self.data > 0).astype(float)
                self.grad = (self.grad if self.grad is not None else 0) + mask * out.grad
        out._backward = backward
        return out

# Example: Simple computation graph
# L = (w₁x + w₂y)²  → dL/dw₁, dL/dw₂
w1 = Variable(2.0)
w2 = Variable(-1.0)
x = Variable(3.0, requires_grad=False)
y = Variable(1.0, requires_grad=False)

z = w1 * x + w2 * y   # 2*3 + (-1)*1 = 5
L = z * z              # 25

L.backward()

print(f"L = {L.data}")
print(f"dL/dw1 = {w1.grad}  (expected: 2*z*x = 2*5*3 = 30)")
print(f"dL/dw2 = {w2.grad}  (expected: 2*z*y = 2*5*1 = 10)")

# Verify numerically
h = 1e-5
def L_fn(w1_val, w2_val):
    z = w1_val * 3 + w2_val * 1
    return z**2

print(f"\nNumerical dL/dw1 = {(L_fn(2+h, -1) - L_fn(2-h, -1))/(2*h):.4f}")
print(f"Numerical dL/dw2 = {(L_fn(2, -1+h) - L_fn(2, -1-h))/(2*h):.4f}")
```

### 10.4 Automatic Differentiation: Computational Graph View

In AD, every operation is a node; we track how the output depends on each input. **Forward mode** pushes derivatives along with values (dual numbers); **reverse mode** (backprop) propagates gradients backward from the loss. For ML with \( n \gg 1 \) parameters and one scalar loss, reverse mode is \( O(n) \) per backward pass vs. \( O(n) \) forward passes for forward mode — hence backprop dominates.

**Key calculus insight:** Each node applies the chain rule. If \( y = f(u,v) \) and \( u,v \) depend on \( x \), then \( \frac{dy}{dx} = \frac{\partial y}{\partial u}\frac{du}{dx} + \frac{\partial y}{\partial v}\frac{dv}{dx} \). AD implements this by storing local Jacobians and composing them.

### 10.5 Using PyTorch Autograd

```python
import torch

# PyTorch example: full autograd
w = torch.tensor([2.0, -1.0], requires_grad=True)
x = torch.tensor([3.0, 1.0])

# L = (w · x)²
z = torch.dot(w, x)   # 2*3 + (-1)*1 = 5
L = z**2              # 25

L.backward()

print(f"PyTorch dL/dw = {w.grad}")  # [30., 10.]

# Neural network training loop
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# One training step
X_t = torch.randn(32, 10)  # Batch of 32
y_t = torch.randint(0, 2, (32, 1)).float()

optimizer.zero_grad()      # Clear previous gradients
pred = model(X_t)          # Forward pass
loss = criterion(pred, y_t) # Compute loss
loss.backward()             # Backward pass (auto-differentiation!)
optimizer.step()            # Update parameters

print(f"\nPyTorch training step loss: {loss.item():.4f}")
```

---

## 11. Calculus in Machine Learning

### 11.1 Loss Functions and Their Gradients

```python
import numpy as np

# --- Mean Squared Error ---
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def mse_gradient(y_pred, y_true):
    """dL/dy_pred = 2(y_pred - y_true) / n"""
    return 2 * (y_pred - y_true) / len(y_true)

# --- Binary Cross-Entropy ---
def bce_loss(y_pred, y_true, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_gradient(y_pred, y_true, eps=1e-7):
    """dL/dy_pred = -(y_true/y_pred - (1-y_true)/(1-y_pred)) / n"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)

# --- Softmax Cross-Entropy (combined for stability) ---
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def softmax_ce_loss(logits, y_true_indices):
    probs = softmax(logits)
    n = len(y_true_indices)
    return -np.mean(np.log(probs[range(n), y_true_indices] + 1e-10))

def softmax_ce_gradient(logits, y_true_indices):
    """dL/dlogits = (softmax(logits) - one_hot(y_true)) / n"""
    n = len(y_true_indices)
    probs = softmax(logits)
    probs[range(n), y_true_indices] -= 1
    return probs / n

# Test
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
y_true = np.array([0, 1])
loss = softmax_ce_loss(logits, y_true)
grad = softmax_ce_gradient(logits, y_true)
print(f"Softmax CE loss: {loss:.4f}")
print(f"Gradient:\n{grad}")
```

### 11.2 Complete Backpropagation Derivation

```python
import numpy as np

class TwoLayerNet:
    """2-layer neural network with explicit backprop."""
    def __init__(self, n_input, n_hidden, n_output, lr=0.01):
        # He initialization
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2 / n_input)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2 / n_hidden)
        self.b2 = np.zeros(n_output)
        self.lr = lr

    def relu(self, z): return np.maximum(0, z)
    def d_relu(self, z): return (z > 0).astype(float)

    def softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass. Save intermediate values for backward."""
        self.X = X                                    # Input
        self.Z1 = X @ self.W1 + self.b1              # Pre-activation 1
        self.A1 = self.relu(self.Z1)                  # Activation 1
        self.Z2 = self.A1 @ self.W2 + self.b2        # Pre-activation 2
        self.A2 = self.softmax(self.Z2)              # Output probs
        return self.A2

    def backward(self, y_true_idx):
        """Backward pass: compute gradients via chain rule."""
        n = len(y_true_idx)

        # --- Output layer ---
        # dL/dZ2: gradient of softmax-cross-entropy (combined form)
        dZ2 = self.A2.copy()
        dZ2[range(n), y_true_idx] -= 1
        dZ2 /= n                                     # Shape: (n, n_output)

        # dL/dW2 = A1^T dZ2
        dW2 = self.A1.T @ dZ2                        # Shape: (n_hidden, n_output)
        db2 = dZ2.sum(axis=0)                        # Shape: (n_output,)

        # --- Hidden layer ---
        # dL/dA1 = dZ2 @ W2^T
        dA1 = dZ2 @ self.W2.T                        # Shape: (n, n_hidden)

        # dL/dZ1 = dA1 * ReLU'(Z1)  (elementwise: chain rule)
        dZ1 = dA1 * self.d_relu(self.Z1)             # Shape: (n, n_hidden)

        # dL/dW1 = X^T dZ1
        dW1 = self.X.T @ dZ1                         # Shape: (n_input, n_hidden)
        db1 = dZ1.sum(axis=0)                        # Shape: (n_hidden,)

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return dW1, db1, dW2, db2

# Training loop
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X_iris, y_iris = load_iris(return_X_y=True)
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)

net = TwoLayerNet(n_input=4, n_hidden=16, n_output=3, lr=0.01)
n_epochs = 500

for epoch in range(n_epochs):
    probs = net.forward(X_iris)
    loss = -np.mean(np.log(probs[range(len(y_iris)), y_iris] + 1e-10))
    net.backward(y_iris)

    if epoch % 100 == 0:
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y_iris)
        print(f"Epoch {epoch:4d}: Loss={loss:.4f}, Accuracy={acc:.4f}")
```

### 11.3 Gradient Flow & Vanishing/Exploding Gradients

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_gradient_flow(n_layers, activation='sigmoid'):
    """Simulate gradient magnitudes through deep network."""
    layer_grad_norms = []
    grad = np.ones(100)  # Initial gradient (from loss)

    for layer in range(n_layers):
        if activation == 'sigmoid':
            # Sigmoid derivative is at most 0.25
            # At saturation (x >> 0), it approaches 0
            a_vals = np.random.randn(100)  # Pre-activations
            d_act = 1 / (1 + np.exp(-a_vals)) * (1 - 1 / (1 + np.exp(-a_vals)))
        elif activation == 'relu':
            a_vals = np.random.randn(100)
            d_act = (a_vals > 0).astype(float)  # Exactly 0 or 1
        elif activation == 'tanh':
            a_vals = np.random.randn(100)
            d_act = 1 - np.tanh(a_vals)**2

        W = np.random.randn(100, 100) * 0.1  # Small weights
        grad = (W.T @ (grad * d_act))  # Backprop through one layer
        layer_grad_norms.append(np.linalg.norm(grad))

    return layer_grad_norms

# Compare activations
print("Gradient norms (layer by layer, deeper = earlier):")
print(f"{'Layer':>6} {'Sigmoid':>12} {'ReLU':>12} {'Tanh':>12}")

sigmoid_norms = analyze_gradient_flow(20, 'sigmoid')
relu_norms = analyze_gradient_flow(20, 'relu')
tanh_norms = analyze_gradient_flow(20, 'tanh')

for i in [0, 4, 9, 14, 19]:
    print(f"{i:6d} {sigmoid_norms[i]:12.6f} {relu_norms[i]:12.6f} {tanh_norms[i]:12.6f}")

print("\nNote: Sigmoid gradients vanish rapidly (vanishing gradient problem)")
print("ReLU helps but can 'die' (dying ReLU problem)")
print("Batch Norm, ResNets, careful init help address these issues")
```

### 11.4 Common Pitfalls & Numerical Stability

```python
import numpy as np

# Pitfall 1: Numerical derivative — h too small causes cancellation
def bad_numerical_deriv(f, x, h=1e-15):
    return (f(x + h) - f(x)) / h

def good_numerical_deriv(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

f = lambda x: np.sin(x)
x = 1.0
print("Numerical derivative at x=1 (true = cos(1) ≈ 0.540):")
for h in [1e-2, 1e-5, 1e-8, 1e-12, 1e-15]:
    bad = bad_numerical_deriv(f, x, h)
    good = good_numerical_deriv(f, x, h)
    print(f"  h={h:.0e}: forward={bad:.6f}, central={good:.6f}")

# Pitfall 2: Softmax/sigmoid overflow — use log-sum-exp trick
def softmax_naive(z):
    return np.exp(z) / np.exp(z).sum()  # Overflow for large z!

def softmax_stable(z):
    z_shifted = z - z.max()
    return np.exp(z_shifted) / np.exp(z_shifted).sum()

z_large = np.array([1000.0, 1001.0, 1002.0])
print("\nSoftmax overflow:")
try:
    print(f"  Naive: {softmax_naive(z_large)}")
except:
    print("  Naive: overflow!")
print(f"  Stable: {softmax_stable(z_large)}")

# Pitfall 3: log(0) or division by zero in BCE
eps = 1e-7
y_pred = np.array([0.0, 1.0, 0.001])
y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
print("\nBCE: always clip y_pred to [eps, 1-eps] to avoid log(0)")
```

**Summary of pitfalls:**
- **Finite differences:** Use central difference; avoid \( h \) too small (machine epsilon ~1e-16 causes cancellation).
- **Chain rule:** Always evaluate outer derivative at the *inner* function's value.
- **Overflow:** Use log-sum-exp for softmax; clip exponentials in sigmoid.
- **Gradient magnitude:** Vanishing (sigmoid) vs exploding (deep nets) — use BatchNorm, ResNets, gradient clipping.

---

## Summary: Calculus for ML

```
Derivatives (the backbone of ML):
├── Definition: f'(x) = lim[h→0] (f(x+h) - f(x))/h
├── Chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)  → BACKPROP
├── Partial derivative: hold other vars constant
├── Gradient ∇f: vector of partials, points uphill
└── Hessian H(f): matrix of second derivatives, encodes curvature

Key Rules:
├── Power:    d/dx[x^n] = n·x^(n-1)
├── Exp:      d/dx[e^x] = e^x
├── Log:      d/dx[ln x] = 1/x
├── Product:  (fg)' = f'g + fg'
├── Quotient: (f/g)' = (f'g - fg') / g²
└── Chain:    d/dx[f(g(x))] = f'(g(x)) · g'(x)

Optimization:
├── Gradient descent: x ← x - η∇f(x)
├── Newton's method: x ← x - H(x)^{-1}∇f(x)  [quadratic convergence]
├── Adam: adaptive learning rates (first + second moments)
├── Lagrange multipliers: constrained optimization (SVM dual)
└── Convex problems: any local min = global min

Automatic Differentiation:
├── Forward mode: propagate (value, derivative) pairs
│   Efficient for few inputs, many outputs
├── Reverse mode (backprop): one backward pass for ALL gradients
│   Efficient for one output (loss), many parameters
└── Both are EXACT (not approximations like finite differences)

Taylor Series (understanding approximation):
├── f(x) ≈ f(a) + f'(a)(x-a)              [linear/first order]
├── f(x) ≈ f(a) + ∇f(a)^T(x-a) + ½(x-a)^T H(a)(x-a)  [quadratic]
└── Used in Newton's method, natural gradient, second-order methods
```

---

## References

- **Calculus:** Stewart, *Calculus*; Apostol, *Calculus Vol. I & II*
- **Optimization:** Boyd & Vandenberghe, *Convex Optimization*; Nocedal & Wright, *Numerical Optimization*
- **Automatic Differentiation:** Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018)
- **ML:** Goodfellow et al., *Deep Learning* (Ch. 6: Derivatives, Ch. 8: Optimization)
