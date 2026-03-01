# Optimization for Machine Learning and Deep Learning: Comprehensive Guide

## Table of Contents
1. [Mathematical Foundations](#mathematical-foundations)
2. [Convexity Theory](#convexity-theory)
3. [Gradient Descent Variants](#gradient-descent-variants)
4. [Momentum Methods](#momentum-methods)
5. [Adaptive Learning Rate Methods](#adaptive-learning-rate-methods)
6. [Learning Rate Scheduling](#learning-rate-scheduling)
7. [Second-Order Methods](#second-order-methods)
8. [Constrained Optimization and KKT Conditions](#constrained-optimization-and-kkt-conditions)
9. [Regularization from Optimization Perspective](#regularization-from-optimization-perspective)
10. [Loss Landscapes in Deep Learning](#loss-landscapes-in-deep-learning)
11. [Gradient Clipping](#gradient-clipping)
12. [Mixed Precision Training](#mixed-precision-training)
13. [Distributed Training](#distributed-training)
14. [Hyperparameter Tuning for Optimizers](#hyperparameter-tuning-for-optimizers)
15. [Full Code Examples — Optimizer Comparison](#full-code-examples--optimizer-comparison)
16. [Common Pitfalls](#common-pitfalls-in-optimization)

---

## Mathematical Foundations

### Optimization Problem Formulation

The general unconstrained ML optimization problem:
\[
\theta^* = \arg\min_{\theta \in \mathbb{R}^d} \mathcal{L}(\theta)
\]

Where \( \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i) \) is the empirical risk.

**Constrained** form:
\[
\min_\theta \mathcal{L}(\theta) \quad \text{subject to } g_i(\theta) \leq 0, \; h_j(\theta) = 0
\]

### Gradient, Jacobian, Hessian

**Gradient** of scalar \( f : \mathbb{R}^d \to \mathbb{R} \):
\[
\nabla f(\theta) = \left[\frac{\partial f}{\partial \theta_1}, \ldots, \frac{\partial f}{\partial \theta_d}\right]^\top \in \mathbb{R}^d
\]

**Jacobian** of vector \( F : \mathbb{R}^d \to \mathbb{R}^m \):
\[
J_F(\theta) = \frac{\partial F}{\partial \theta} \in \mathbb{R}^{m \times d}, \quad J_{ij} = \frac{\partial F_i}{\partial \theta_j}
\]

**Hessian** of scalar \( f \):
\[
H_f(\theta) = \nabla^2 f(\theta) \in \mathbb{R}^{d \times d}, \quad H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}
\]

### First-Order Optimality Condition

A necessary condition for \( \theta^* \) to be a local minimum (unconstrained):
\[
\nabla \mathcal{L}(\theta^*) = \mathbf{0}
\]

**Second-order** sufficient condition: \( H_\mathcal{L}(\theta^*) \succ 0 \) (positive definite)

### Taylor Expansion of Loss

\[
\mathcal{L}(\theta + \Delta\theta) \approx \mathcal{L}(\theta) + \nabla \mathcal{L}(\theta)^\top \Delta\theta + \frac{1}{2} \Delta\theta^\top H \Delta\theta
\]

The optimal step (Newton): \( \Delta\theta^* = -H^{-1} \nabla \mathcal{L}(\theta) \)

---

## Convexity Theory

### Convex Sets

A set \( \mathcal{C} \subseteq \mathbb{R}^d \) is **convex** if for any \( x, y \in \mathcal{C} \) and \( \lambda \in [0,1] \):
\[
\lambda x + (1-\lambda)y \in \mathcal{C}
\]

Examples: balls, polytopes (intersection of half-spaces), affine subspaces.

### Convex Functions

A function \( f : \mathcal{C} \to \mathbb{R} \) is **convex** if for all \( x, y \in \mathcal{C} \), \( \lambda \in [0,1] \):
\[
f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)
\]

**Geometrically**: the function lies below any chord.

**Equivalent conditions** (differentiable f):
1. **First-order**: \( f(y) \geq f(x) + \nabla f(x)^\top (y - x) \) for all \( x, y \) (function lies above its tangent)
2. **Second-order**: \( \nabla^2 f(x) \succeq 0 \) for all \( x \) (Hessian is positive semi-definite)

### Strongly Convex Functions

\( f \) is **\( \mu \)-strongly convex** if:
\[
f(y) \geq f(x) + \nabla f(x)^\top (y - x) + \frac{\mu}{2}\|y - x\|^2
\]

Implies: unique global minimum, faster convergence for GD.

### Lipschitz Smoothness

\( f \) is **\( L \)-smooth** if gradients are \( L \)-Lipschitz:
\[
\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|
\]

Equivalently: \( f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2 \)

The optimal learning rate for gradient descent on an L-smooth convex function is \( \eta = 1/L \).

### Key Result: GD Convergence on Convex Functions

For \( L \)-smooth convex \( f \) with learning rate \( \eta = 1/L \):
\[
f(\theta_T) - f(\theta^*) \leq \frac{L \|\theta_0 - \theta^*\|^2}{2T}
\]
Convergence rate: \( O(1/T) \) (sublinear)

For \( \mu \)-strongly convex, \( L \)-smooth \( f \) with \( \eta = 1/L \):
\[
\|\theta_T - \theta^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^T \|\theta_0 - \theta^*\|^2
\]
Convergence rate: \( O(\exp(-T)) \) (linear / geometric)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def check_convexity(f, x_range=(-5, 5), n_samples=1000):
    """Numerically verify convexity via the chord condition."""
    x = np.linspace(*x_range, n_samples)
    y = f(x)
    violations = 0
    for _ in range(500):
        i, j = np.random.choice(n_samples, 2, replace=False)
        lam = np.random.uniform(0, 1)
        x_mid = lam * x[i] + (1 - lam) * x[j]
        f_mid_interp = lam * y[i] + (1 - lam) * y[j]
        # Approximate f at x_mid
        idx = np.argmin(np.abs(x - x_mid))
        f_mid_actual = y[idx]
        if f_mid_actual > f_mid_interp + 1e-6:
            violations += 1
    print(f"Convexity check: {violations}/500 violations (0 = convex)")

# Convex: f(x) = x^2
check_convexity(lambda x: x**2)

# Non-convex: f(x) = sin(x)
check_convexity(lambda x: np.sin(x), x_range=(0, 10))

# Visualize convex vs non-convex
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x = np.linspace(-3, 3, 200)
fns = [("x² (convex)", x**2), ("x⁴-4x² (non-convex)", x**4 - 4*x**2), ("exp(x) (convex)", np.exp(x))]
for ax, (name, y) in zip(axes, fns):
    ax.plot(x, y, linewidth=2)
    ax.set_title(name); ax.grid(True, alpha=0.4)
plt.tight_layout(); plt.show()
```

---

## Gradient Descent Variants

### 1. Batch Gradient Descent

Uses the **entire dataset** to compute one gradient update:
\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t) = \theta_t - \frac{\eta}{n} \sum_{i=1}^n \nabla_\theta \ell(\theta_t; x_i, y_i)
\]

- **Pro**: Exact gradient, stable convergence, deterministic
- **Con**: Very slow for large datasets (one update per full pass), doesn't escape shallow local minima

**Convergence** (L-smooth, convex): \( O(1/T) \); (strongly convex): \( O(\exp(-T)) \)

### 2. Stochastic Gradient Descent (SGD)

Uses a **single random sample** per update:
\[
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \ell(\theta_t; x_{i_t}, y_{i_t})
\]

Where \( i_t \) is drawn uniformly at random.

- **Pro**: Very fast per-iteration, noisy updates help escape local minima, works for online learning
- **Con**: High variance, oscillatory behavior, requires decaying LR for convergence

**Convergence** (non-convex): \( O(1/\sqrt{T}) \) in expectation; requires \( \sum \eta_t = \infty, \sum \eta_t^2 < \infty \)

### 3. Mini-Batch SGD

The standard in practice: uses a random batch \( \mathcal{B}_t \subset \{1, \ldots, n\} \) of size \( B \):
\[
\theta_{t+1} = \theta_t - \frac{\eta}{B} \sum_{i \in \mathcal{B}_t} \nabla_\theta \ell(\theta_t; x_i, y_i)
\]

**Why it works well**:
- Variance: \( \text{Var}[\hat{g}] = \text{Var}[g_i] / B \) — reduces noise vs single-sample SGD
- Hardware: batch operations are efficient on GPU (BLAS level-3)
- Gradient noise has a *regularizing* effect in deep learning

### SGD Implementation with Full Convergence Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class OptResult:
    params: np.ndarray
    losses: List[float]
    grad_norms: List[float]
    name: str


def batch_gd(grad_fn, theta_init, lr=0.1, n_iters=200, data=None) -> OptResult:
    """Full-batch gradient descent."""
    theta = theta_init.copy()
    losses, grad_norms = [], []
    X, y = data

    for t in range(n_iters):
        g = grad_fn(theta, X, y)
        theta = theta - lr * g
        losses.append(loss_fn(theta, X, y))
        grad_norms.append(np.linalg.norm(g))

    return OptResult(theta, losses, grad_norms, "Batch GD")


def sgd(grad_fn, theta_init, lr=0.1, n_iters=200, batch_size=32, data=None,
        lr_decay=None) -> OptResult:
    """Mini-batch SGD with optional learning rate decay."""
    theta = theta_init.copy()
    losses, grad_norms = [], []
    X, y = data
    n = X.shape[0]

    for t in range(n_iters):
        idx = np.random.choice(n, batch_size, replace=False)
        Xb, yb = X[idx], y[idx]
        g = grad_fn(theta, Xb, yb)
        eta = lr / (1 + lr_decay * t) if lr_decay else lr
        theta = theta - eta * g
        losses.append(loss_fn(theta, X, y))
        grad_norms.append(np.linalg.norm(g))

    return OptResult(theta, losses, grad_norms, f"SGD (B={batch_size})")


# ── Linear Regression example ──
np.random.seed(42)
n, d = 2000, 10
theta_true = np.random.randn(d)
X = np.random.randn(n, d)
y = X @ theta_true + 0.1 * np.random.randn(n)

def loss_fn(theta, X, y):
    return 0.5 * np.mean((X @ theta - y)**2)

def grad_fn(theta, X, y):
    return X.T @ (X @ theta - y) / len(y)

theta0 = np.zeros(d)

res_batch = batch_gd(grad_fn, theta0, lr=0.05, n_iters=300, data=(X, y))
res_sgd32 = sgd(grad_fn, theta0, lr=0.05, n_iters=300, batch_size=32, data=(X, y))
res_sgd1  = sgd(grad_fn, theta0, lr=0.02, n_iters=300, batch_size=1, data=(X, y))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for res in [res_batch, res_sgd32, res_sgd1]:
    axes[0].semilogy(res.losses, label=res.name)
    axes[1].semilogy(res.grad_norms, label=res.name)
axes[0].set_title("Loss Convergence"); axes[0].legend(); axes[0].grid(True)
axes[1].set_title("Gradient Norm"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.show()

print(f"\nTrue    θ norm: {np.linalg.norm(theta_true):.4f}")
print(f"Batch GD error: {np.linalg.norm(res_batch.params - theta_true):.6f}")
print(f"SGD-32  error:  {np.linalg.norm(res_sgd32.params - theta_true):.6f}")
```

---

## Momentum Methods

### Classical Momentum (Heavy Ball)

The momentum method accumulates a velocity vector in the direction of gradient descent:

\[
v_{t+1} = \beta v_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
\]
\[
\theta_{t+1} = \theta_t + v_{t+1}
\]

**Intuition**: Like a ball rolling down a hill. The \( \beta v_t \) term carries accumulated velocity, helping navigate ravines (directions of small curvature that slow down GD).

**Effect on curvature**: In the direction with eigenvalue \( \lambda \) of the Hessian, effective step size is:
\[
\eta_{\text{eff}} = \frac{\eta}{1 - \beta} \cdot \frac{1}{1 + \frac{\beta}{(1-\beta)^2} \lambda}
\]

For small \( \lambda \) (flat directions): step ≈ \( \frac{\eta}{1-\beta} \) — **amplified** by \( 1/(1-\beta) \)  
For large \( \lambda \) (steep directions): step damped — prevents oscillations

**Convergence** (\( \mu \)-strongly convex, L-smooth):
\[
\left(1 - \sqrt{\frac{\mu}{L}}\right)^T \quad \text{vs GD's} \quad \left(1 - \frac{\mu}{L}\right)^T
\]

Momentum achieves **linear convergence** with rate depending on \( \sqrt{\kappa} \) instead of \( \kappa \) (condition number).

### Nesterov Accelerated Gradient (NAG)

Nesterov (1983) proposed computing the gradient at the **lookahead** position:

\[
\theta_{\text{look}} = \theta_t + \beta v_t
\]
\[
v_{t+1} = \beta v_t - \eta \nabla_\theta \mathcal{L}(\theta_{\text{look}})
\]
\[
\theta_{t+1} = \theta_t + v_{t+1}
\]

**Key difference**: Momentum GD first updates, then corrects; NAG first looks ahead, then updates — providing a more informative gradient.

**Optimal convergence**: \( O(1/T^2) \) for convex functions (vs \( O(1/T) \) for GD).

```python
class Optimizer:
    """Base class for optimizers."""
    def __init__(self, lr=0.01):
        self.lr = lr
        self.t = 0

    def step(self, theta, grad): raise NotImplementedError


class SGDOptimizer(Optimizer):
    def step(self, theta, grad):
        self.t += 1
        return theta - self.lr * grad


class MomentumOptimizer(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def step(self, theta, grad):
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v - self.lr * grad
        return theta + self.v


class NesterovOptimizer(Optimizer):
    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def lookahead(self, theta):
        return theta + self.beta * self.v if self.v is not None else theta

    def step(self, theta, grad_at_lookahead):
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v - self.lr * grad_at_lookahead
        return theta + self.v


def run_optimizer(opt, loss_fn, grad_fn, theta_init, n_iters=100, data=None):
    theta = theta_init.copy()
    losses = [loss_fn(theta, *data)]

    if isinstance(opt, NesterovOptimizer):
        for _ in range(n_iters):
            lookahead = opt.lookahead(theta)
            g = grad_fn(lookahead, *data)
            theta = opt.step(theta, g)
            losses.append(loss_fn(theta, *data))
    else:
        for _ in range(n_iters):
            g = grad_fn(theta, *data)
            theta = opt.step(theta, g)
            losses.append(loss_fn(theta, *data))

    return theta, losses


theta0 = np.zeros(d)
data = (X, y)

_, losses_sgd = run_optimizer(SGDOptimizer(0.05), loss_fn, grad_fn, theta0, 200, data)
_, losses_mom = run_optimizer(MomentumOptimizer(0.05, 0.9), loss_fn, grad_fn, theta0, 200, data)
_, losses_nag = run_optimizer(NesterovOptimizer(0.05, 0.9), loss_fn, grad_fn, theta0, 200, data)

plt.figure(figsize=(10, 5))
plt.semilogy(losses_sgd, label="SGD", linewidth=2)
plt.semilogy(losses_mom, label="Momentum (β=0.9)", linewidth=2)
plt.semilogy(losses_nag, label="NAG (β=0.9)", linewidth=2)
plt.xlabel("Iteration"); plt.ylabel("Loss (log scale)")
plt.title("SGD vs Momentum vs Nesterov")
plt.legend(); plt.grid(True); plt.show()
```

---

## Adaptive Learning Rate Methods

### The Problem with Fixed LR

A single learning rate is applied to all parameters equally. But:
- Parameters with sparse gradients (NLP embeddings) need larger steps
- Parameters with frequent large gradients need smaller steps
- Different layers may have very different gradient scales

### 1. AdaGrad (Duchi et al., 2011)

Accumulates **sum of squared gradients** and divides by its square root — parameters with large historical gradients get smaller updates:

\[
G_{t+1} = G_t + g_t^2 \quad \text{(element-wise)}
\]
\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \odot g_t
\]

- **Pro**: No LR tuning for sparse features; good for NLP with rare words
- **Con**: \( G_t \) grows monotonically → effective LR shrinks to zero → training stops

### 2. RMSprop (Hinton, 2012)

Fixes AdaGrad's dying LR by using **exponential moving average** of squared gradients:

\[
v_{t+1} = \rho v_t + (1 - \rho) g_t^2
\]
\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1} + \epsilon}} \odot g_t
\]

Where \( \rho \approx 0.9 \). The window focuses on recent gradients, preventing vanishing LR.

### 3. Adam (Kingma & Ba, 2014)

Combines momentum (first moment) with RMSprop (second moment), with **bias correction**:

**First moment** (mean of gradients):
\[
m_{t+1} = \beta_1 m_t + (1 - \beta_1) g_t
\]

**Second moment** (uncentered variance):
\[
v_{t+1} = \beta_2 v_t + (1 - \beta_2) g_t^2
\]

**Bias correction** (crucial at early timesteps where moments are underestimated):
\[
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \quad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
\]

**Update rule**:
\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}
\]

**Typical hyperparameters**: \( \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}, \eta = 10^{-3} \)

**Effective step size analysis**:
\[
|\Delta\theta| \approx \eta \cdot \frac{|\hat{m}|}{\sqrt{\hat{v}}} \approx \eta \cdot \text{sign}(g) \cdot \text{SNR}(g)
\]

If gradient is consistent: \( |\hat{m}| \approx \sqrt{\hat{v}} \) → step ≈ \( \eta \)  
If gradient is noisy: \( |\hat{m}| \ll \sqrt{\hat{v}} \) → step \( \ll \eta \) (adaptive dampening)

### 4. AdamW (Loshchilov & Hutter, 2017)

**Critical insight**: L2 regularization ≠ weight decay in adaptive methods.

In Adam with L2 reg, the gradient becomes \( g_t + \lambda \theta_t \). But the adaptive scaling \( 1/\sqrt{\hat{v}} \) also divides the regularization term, making regularization effectively weaker for parameters with large gradients.

**AdamW** decouples weight decay from the gradient update:
\[
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} - \eta \lambda \theta_t
\]

The \( -\eta \lambda \theta_t \) term is applied directly, independently of gradient scale. This is the **correct** implementation of weight decay.

**AdamW is the de-facto standard optimizer** for training Transformers and modern large models.

### 5. AMSGrad (Reddi et al., 2018)

Adam can fail to converge in some cases because \( \hat{v}_t \) can decrease. AMSGrad uses the maximum of past \( \hat{v} \) values:

\[
\hat{V}_{t+1} = \max(\hat{V}_t, \hat{v}_{t+1})
\]
\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{V}_{t+1}} + \epsilon} \hat{m}_{t+1}
\]

### 6. LAMB (You et al., 2019) — Large Batch Training

Designed for very large batch sizes (global batch ~32K). Clips the layer-wise ratio of gradient norm to weight norm:

\[
r_l = \frac{\|\theta_l\|}{\|\hat{m}_l / \sqrt{\hat{v}_l}\|}
\]
\[
\theta_l \leftarrow \theta_l - \eta \cdot r_l \cdot \frac{\hat{m}_l}{\sqrt{\hat{v}_l} + \epsilon}
\]

Used to train BERT in 77 minutes (vs 3 days with Adam).

### 7. LARS (Large Batch SGD) (You et al., 2017)

Layer-wise adaptive rate scaling for SGD:
\[
\eta_l = \alpha \cdot \frac{\|\theta_l\|}{\|\nabla_l \mathcal{L}\| + \beta \|\theta_l\|}
\]

### Full Optimizer Implementation

```python
import numpy as np

class AdamOptimizer:
    """Adam optimizer with AMSGrad option."""
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}

    def step(self, params_grads):
        """
        params_grads: list of (param_name, param, grad) tuples
        Returns updated parameters.
        """
        updated = []
        for name, theta, g in params_grads:
            if name not in self.state:
                self.state[name] = {
                    "t": 0,
                    "m": np.zeros_like(theta),
                    "v": np.zeros_like(theta),
                    "v_max": np.zeros_like(theta),
                }

            s = self.state[name]
            s["t"] += 1
            t = s["t"]

            # Weight decay (decoupled — AdamW style)
            if self.weight_decay > 0:
                g = g + self.weight_decay * theta

            # Moment updates
            s["m"] = self.beta1 * s["m"] + (1 - self.beta1) * g
            s["v"] = self.beta2 * s["v"] + (1 - self.beta2) * g**2

            # Bias correction
            m_hat = s["m"] / (1 - self.beta1**t)
            v_hat = s["v"] / (1 - self.beta2**t)

            if self.amsgrad:
                s["v_max"] = np.maximum(s["v_max"], v_hat)
                denom = np.sqrt(s["v_max"]) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps

            theta_new = theta - self.lr * m_hat / denom
            updated.append((name, theta_new))

        return updated


class RMSpropOptimizer:
    def __init__(self, lr=1e-3, rho=0.9, eps=1e-8, momentum=0.0):
        self.lr, self.rho, self.eps, self.momentum = lr, rho, eps, momentum
        self.state = {}

    def step(self, params_grads):
        updated = []
        for name, theta, g in params_grads:
            if name not in self.state:
                self.state[name] = {"v": np.zeros_like(theta), "buf": np.zeros_like(theta)}
            s = self.state[name]
            s["v"] = self.rho * s["v"] + (1 - self.rho) * g**2
            buf = self.momentum * s["buf"] + self.lr * g / (np.sqrt(s["v"]) + self.eps)
            s["buf"] = buf
            updated.append((name, theta - buf))
        return updated
```

---

## Learning Rate Scheduling

### 1. Step Decay

Multiply LR by factor \( \gamma \) every \( k \) epochs:
\[
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}
\]

### 2. Cosine Annealing (Loshchilov & Hutter, 2016)

Smoothly decays LR following a cosine schedule:
\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T_{\max}}\right)
\]

**Cosine Annealing with Warm Restarts (SGDR)**: Reset LR to max every \( T_i \) steps with geometric increase \( T_{i+1} = T_{\text{mult}} \cdot T_i \).

### 3. Linear Warmup

Start with \( \eta_{\text{small}} \) and linearly increase to \( \eta_{\text{target}} \) over \( w \) steps:
\[
\eta_t = \eta_{\text{target}} \cdot \frac{t}{w} \quad \text{for } t \leq w
\]

Critical for Transformers: avoids instability in early training when gradients are large and inconsistent.

### 4. One-Cycle Policy (Smith, 2018)

Two phases:
1. **Warmup** (from \( \eta_{\min} \) to \( \eta_{\max} \)): ~30% of training
2. **Annealing** (\( \eta_{\max} \) to \( \eta_{\min}/100 \)): ~70% of training

Often combined with a similar cycle for momentum (decrease during warmup, increase during annealing). Often finds better solutions faster than step decay.

### 5. Polynomial Decay

\[
\eta_t = (\eta_0 - \eta_{\text{end}}) \left(1 - \frac{t}{T}\right)^p + \eta_{\text{end}}
\]

### 6. Reduce on Plateau

Monitor a metric (e.g., val loss). If no improvement for `patience` steps, multiply LR by `factor`.

### 7. Warmup + Cosine (Transformer Standard)

\[
\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\; t \cdot w^{-1.5})
\]

```python
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def visualize_schedules(n_steps=1000, warmup_steps=100):
    """Plot common LR schedules."""
    model = torch.nn.Linear(10, 1)
    base_lr = 1e-3
    schedules = {}

    # Step decay
    opt = optim.SGD(model.parameters(), lr=base_lr)
    sch = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)
    lrs = []
    for _ in range(n_steps): lrs.append(opt.param_groups[0]["lr"]); sch.step()
    schedules["Step Decay"] = lrs

    # Cosine Annealing
    opt = optim.SGD(model.parameters(), lr=base_lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=1e-6)
    lrs = []
    for _ in range(n_steps): lrs.append(opt.param_groups[0]["lr"]); sch.step()
    schedules["Cosine Annealing"] = lrs

    # Cosine with Warm Restarts
    opt = optim.SGD(model.parameters(), lr=base_lr)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2)
    lrs = []
    for _ in range(n_steps): lrs.append(opt.param_groups[0]["lr"]); sch.step()
    schedules["Cosine + Warm Restarts"] = lrs

    # One Cycle
    opt = optim.SGD(model.parameters(), lr=base_lr/10)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=base_lr,
                                          total_steps=n_steps, pct_start=0.3)
    lrs = []
    for _ in range(n_steps): lrs.append(opt.param_groups[0]["lr"]); sch.step()
    schedules["One Cycle"] = lrs

    # Transformer warmup-cosine
    def transformer_schedule(step, d_model=512, warmup=warmup_steps):
        step = max(step, 1)
        return d_model**(-0.5) * min(step**(-0.5), step * warmup**(-1.5)) * 1e4
    schedules["Transformer Warmup"] = [transformer_schedule(t) for t in range(1, n_steps+1)]

    # Reduce on Plateau (conceptual visualization)
    lrs = [base_lr] * n_steps
    for milestone in [300, 500, 700]:
        for t in range(milestone, n_steps):
            lrs[t] *= 0.5 if t == milestone else 1
    schedules["Reduce on Plateau"] = lrs

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, lr_list) in zip(axes.flatten(), schedules.items()):
        ax.plot(lr_list, linewidth=2, color="steelblue")
        ax.set_title(name, fontsize=12); ax.set_xlabel("Step"); ax.set_ylabel("LR")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Learning Rate Schedules", fontsize=15)
    plt.tight_layout(); plt.show()

visualize_schedules()
```

---

## Second-Order Methods

Second-order methods use **curvature** (Hessian or approximations) to adapt step size per direction. They converge faster per iteration but have higher per-iteration cost. Best for: moderate-dimensional, smooth, well-conditioned problems.

### Newton's Method

Uses curvature information (Hessian) for the exact optimal step:
\[
\theta_{t+1} = \theta_t - H_t^{-1} \nabla \mathcal{L}(\theta_t)
\]

**Convergence**: Quadratic near the optimum (error squared at each step). Reaches machine precision in very few iterations.

**Problems**:
1. Computing \( H \): \( O(d^2) \) memory (e.g., \( d = 10^8 \) → \( 10^{16} \) entries!)
2. Inverting \( H \): \( O(d^3) \) per step
3. Hessian may not be PD (saddle points)

### Gauss-Newton Method

Approximates the Hessian using Jacobians (avoids second derivatives):
\[
H \approx J^\top J
\]
Always positive semi-definite! Used for least-squares problems.

### L-BFGS (Limited-memory BFGS)

Quasi-Newton method: approximates \( H^{-1} \) implicitly using the last \( m \) gradient pairs \( (s_k, y_k) \) where \( s_k = \theta_{k+1} - \theta_k \), \( y_k = g_{k+1} - g_k \):

\[
H_{k+1}^{-1} \approx \left(I - \rho_k s_k y_k^\top\right) H_k^{-1} \left(I - \rho_k y_k s_k^\top\right) + \rho_k s_k s_k^\top
\]

Where \( \rho_k = 1/(y_k^\top s_k) \).

Memory: \( O(md) \) instead of \( O(d^2) \). Convergence: superlinear.

**Used in**: Traditional ML (SciPy `minimize`), shallow networks, when full-batch possible.

### Natural Gradient (Amari, 1998)

Follows the steepest descent in the **Riemannian manifold** of probability distributions, using the Fisher Information Matrix \( F \) as metric:

\[
\tilde{\nabla} \mathcal{L}(\theta) = F(\theta)^{-1} \nabla \mathcal{L}(\theta)
\]
\[
F(\theta) = \mathbb{E}_{x \sim p_\theta} \left[\nabla \log p_\theta(x) \cdot \nabla \log p_\theta(x)^\top\right]
\]

The update is **invariant to reparameterization** — a fundamental advantage.

**Approximations** (exact F is too expensive):
- **K-FAC** (Kronecker-Factored Approximate Curvature): block-diagonal Fisher approximation
- **EKFAC**: Eigenvalue-corrected K-FAC

```python
from scipy.optimize import minimize
import numpy as np

# L-BFGS via scipy
def rosenbrock(x):
    return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

def rosenbrock_grad(x):
    g = np.zeros_like(x)
    for i in range(len(x)-1):
        g[i]   += -400*x[i]*(x[i+1]-x[i]**2) - 2*(1-x[i])
        g[i+1] += 200*(x[i+1]-x[i]**2)
    return g

x0 = np.array([-1.5, 1.0, -1.5, 1.0])

# Compare convergence speeds
from scipy.optimize import minimize

results = {}
for method in ["L-BFGS-B", "CG", "Nelder-Mead"]:
    count = {"n": 0}
    def tracked_f(x):
        count["n"] += 1
        return rosenbrock(x)
    r = minimize(tracked_f, x0, method=method, jac=rosenbrock_grad if method != "Nelder-Mead" else None)
    results[method] = {"fun": r.fun, "nit": r.nit, "nfev": count["n"], "success": r.success}

for method, r in results.items():
    print(f"{method:<15}: loss={r['fun']:.2e}, iters={r['nit']}, fun_evals={r['nfev']}, ok={r['success']}")
```

---

## Constrained Optimization and KKT Conditions

### Lagrange Multipliers (Equality Constraints)

For \( \min_\theta f(\theta) \) subject to \( h(\theta) = 0 \):

The **Lagrangian**:
\[
\mathcal{L}(\theta, \nu) = f(\theta) + \nu^\top h(\theta)
\]

At optimality: \( \nabla_\theta \mathcal{L} = 0 \) and \( h(\theta) = 0 \)

**Dual problem**: \( \max_\nu \min_\theta \mathcal{L}(\theta, \nu) \)

### KKT Conditions (Inequality + Equality Constraints)

For the problem:
\[
\min_\theta f(\theta) \quad \text{s.t. } g_i(\theta) \leq 0, \; h_j(\theta) = 0
\]

The **KKT conditions** are (necessary for optimality under constraint qualifications):

1. **Stationarity**: \( \nabla f(\theta^*) + \sum_i \lambda_i^* \nabla g_i(\theta^*) + \sum_j \nu_j^* \nabla h_j(\theta^*) = 0 \)
2. **Primal feasibility**: \( g_i(\theta^*) \leq 0 \), \( h_j(\theta^*) = 0 \)
3. **Dual feasibility**: \( \lambda_i^* \geq 0 \)
4. **Complementary slackness**: \( \lambda_i^* g_i(\theta^*) = 0 \) (either constraint is active or multiplier is zero)

KKT conditions are **sufficient** for convex problems. SVMs are solved via their KKT conditions.

```python
from scipy.optimize import minimize

def demo_constrained_optimization():
    """
    Minimize f(x,y) = (x-3)² + (y-2)² 
    subject to: x + y ≤ 4, x ≥ 0, y ≥ 0
    (Closest point to (3,2) inside triangle)
    """
    f = lambda x: (x[0]-3)**2 + (x[1]-2)**2
    jac = lambda x: np.array([2*(x[0]-3), 2*(x[1]-2)])

    constraints = [
        {"type": "ineq", "fun": lambda x: 4 - x[0] - x[1]},  # x+y <= 4
        {"type": "ineq", "fun": lambda x: x[0]},              # x >= 0
        {"type": "ineq", "fun": lambda x: x[1]},              # y >= 0
    ]

    x0 = np.array([0.5, 0.5])
    result = minimize(f, x0, method="SLSQP", jac=jac, constraints=constraints)

    print(f"Optimal point: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
    print(f"Optimal value: {result.fun:.4f}")
    print(f"Constraint x+y: {result.x[0]+result.x[1]:.4f} (should be ≤ 4)")

    # Visualize
    x = np.linspace(0, 5, 200); y = np.linspace(0, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = (X-3)**2 + (Y-2)**2
    feasible = (X + Y <= 4) & (X >= 0) & (Y >= 0)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.5)
    plt.contourf(X, Y, feasible.astype(float), levels=[0.5, 1.5], colors=["lightgreen"], alpha=0.3)
    plt.plot(*result.x, "r*", markersize=15, label=f"Optimum {result.x.round(2)}")
    plt.plot(3, 2, "ko", markersize=10, label="Unconstrained min (3,2)")
    plt.legend(); plt.title("Constrained Optimization"); plt.grid(True, alpha=0.3)
    plt.show()

demo_constrained_optimization()


# Dual Ascent / ADMM (Alternating Direction Method of Multipliers)
def admm_lasso(A, b, lam, rho=1.0, n_iter=200):
    """
    ADMM for LASSO: min ||Ax-b||² + lam*||z||₁  s.t. x=z
    Variables: x (primal), z (auxiliary), u (scaled dual)
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(AtA + rho * np.eye(n))

    x, z, u = np.zeros(n), np.zeros(n), np.zeros(n)
    losses = []

    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)

    for _ in range(n_iter):
        # x-update: solve linear system
        x = np.linalg.solve(L.T, np.linalg.solve(L, Atb + rho*(z-u)))
        # z-update: soft thresholding (proximal for L1)
        z = soft_threshold(x + u, lam/rho)
        # u-update: dual ascent
        u = u + x - z
        losses.append(0.5*np.linalg.norm(A@x-b)**2 + lam*np.linalg.norm(z, 1))

    return x, losses

np.random.seed(0)
m, n = 100, 200
A_lasso = np.random.randn(m, n)
x_true = np.zeros(n); x_true[:10] = np.random.randn(10)  # sparse true solution
b_lasso = A_lasso @ x_true + 0.01*np.random.randn(m)

x_admm, losses_admm = admm_lasso(A_lasso, b_lasso, lam=0.1)
print(f"True support: {(x_true != 0).sum()} nonzeros")
print(f"ADMM solution: {(np.abs(x_admm) > 1e-3).sum()} nonzeros")
```

### Projected Gradient Example

```python
def projected_gradient_descent(grad_fn, x0, lr=0.1, n_iters=100, box_bounds=None):
    """Box-constrained: x in [low, high]."""
    x = x0.copy()
    low, high = box_bounds if box_bounds else (-np.inf, np.inf)
    for _ in range(n_iters):
        g = grad_fn(x)
        x = x - lr * g
        x = np.clip(x, low, high)
    return x
```

---

## Regularization from Optimization Perspective

### L2 as Gaussian Prior

L2 regularization corresponds to a **Gaussian prior** on weights under the Bayesian perspective:

\[
p(\theta) = \mathcal{N}(0, \lambda^{-1} I)
\]

MAP estimation:
\[
\arg\max_\theta \log p(D|\theta) + \log p(\theta) = \arg\min_\theta \mathcal{L}(\theta) + \frac{\lambda}{2}\|\theta\|^2
\]

### L1 as Laplace Prior

\[
p(\theta) \propto \exp(-\lambda |\theta|)
\]

Promotes **sparsity** because the non-differentiable point at 0 creates a "kink" that makes sparse solutions optimal.

### Dropout as Adaptive Regularization

Dropout multiplies the effective loss by an exponential number of sub-network losses, approximately equivalent to an L2 penalty with data-dependent coefficient. The variance of the stochastic gradient acts as an adaptive regularizer.

### Implicit Regularization of SGD

SGD with small batches has a regularizing effect even without explicit penalties. The noise in stochastic gradient can be shown to prefer **flat minima** over sharp ones. Flat minima generalize better (large basin of attraction).

The **sharpness** of a minimum can be measured by the largest eigenvalue of the Hessian \( \lambda_{\max}(H) \). SGD with large LR implicitly minimizes:
\[
\mathcal{L}(\theta) + \frac{\eta B}{4} \mathbb{E}[\|\nabla \ell(\theta; \xi)\|^2]
\]

---

## Loss Landscapes in Deep Learning

### Critical Points

- **Saddle points**: \( \nabla \mathcal{L} = 0 \), Hessian has positive and negative eigenvalues. Much more common than local minima in high dimensions.
- **Local minima**: \( \nabla \mathcal{L} = 0 \), Hessian PD. In overparameterized networks, most local minima have similar loss (Goodfellow et al., 2015).
- **Degenerate saddles (plateaus)**: near-zero gradients but not at a minimum. Can cause slow training.

### The Mode Connectivity Hypothesis (Garipov et al., 2018)

Different good solutions found by SGD from different initializations can be connected by paths of low loss in parameter space. The loss landscape is not a bowl with many isolated wells, but has high-dimensional "mountain ranges" surrounding connected valleys.

### Sharp vs Flat Minima

**Flat minima** (small \( \lambda_{\max}(H) \)): robust to parameter perturbations, better generalization.  
**Sharp minima** (large \( \lambda_{\max}(H) \)): sensitive to perturbations, often overfits.

**Sharpness-Aware Minimization (SAM, Foret et al., 2021)**:
\[
\min_\theta \max_{\|\epsilon\| \leq \rho} \mathcal{L}(\theta + \epsilon)
\]

SAM finds flat minima by explicitly optimizing for robustness to perturbations.

```python
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer wrapper."""
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute perturbation: max_||e||<=rho L(θ+e)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to perturbation
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Return to original point and take SGD step with perturbed gradient."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # return to original
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2).to(shared_device)
                         for group in self.param_groups
                         for p in group["params"] if p.grad is not None]),
            p=2)
        return norm
```

---

## Gradient Clipping

### By Norm (Recommended)

Rescales all gradients so the global norm \( \leq \) `max_norm`:
\[
\text{if } \|\mathbf{g}\| > c : \quad \mathbf{g} \leftarrow \mathbf{g} \cdot \frac{c}{\|\mathbf{g}\|}
\]

Preserves gradient direction (unlike by-value clipping). Critical for RNNs and Transformers.

### By Value

Clips each gradient element independently: \( g_i \leftarrow \text{clip}(g_i, -v, v) \). Changes gradient direction, generally less preferred.

```python
import torch
import torch.nn as nn

def demo_gradient_clipping():
    """Show effect of gradient clipping on training stability."""
    import torch

    # Simulate a model with exploding gradients
    class ExplodingNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Intentionally bad initialization to cause explosion
            self.layers = nn.Sequential(*[nn.Linear(10, 10) for _ in range(20)])
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -2, 2)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x.sum()

    model = ExplodingNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(1, 10)

    grad_norms_unclipped = []
    grad_norms_clipped = []

    for step in range(50):
        optimizer.zero_grad()
        loss = model(x)
        loss.backward()

        # Record unclipped norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()**2
        grad_norms_unclipped.append(total_norm**0.5)

        # Clip and record
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_norm_clipped = sum(p.grad.data.norm(2).item()**2
                                  for p in model.parameters() if p.grad is not None)**0.5
        grad_norms_clipped.append(total_norm_clipped)

        optimizer.step()

    plt.figure(figsize=(10, 4))
    plt.semilogy(grad_norms_unclipped, label="Before clipping", alpha=0.7)
    plt.semilogy(grad_norms_clipped, label="After clipping (max_norm=1)", alpha=0.7)
    plt.xlabel("Step"); plt.ylabel("Gradient Norm (log)")
    plt.title("Effect of Gradient Clipping"); plt.legend(); plt.grid(True)
    plt.show()

demo_gradient_clipping()
```

---

## Mixed Precision Training

### Why Mixed Precision?

Modern GPUs (V100, A100, H100) have specialized hardware for 16-bit operations (Tensor Cores):
- **2× memory efficiency**: store activations/gradients in FP16
- **2–8× throughput**: FP16/BF16 matrix multiplications
- **Larger batch sizes**: fit more data in GPU memory

### FP16 vs BF16

| Format | Sign | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits |
| FP16 | 1 | 5 | 10 | ±65504 | ~3 decimal digits |
| BF16 | 1 | 8 | 7  | ±3.4×10³⁸ | ~2 decimal digits |

**BF16**: Same range as FP32, less precision — numerically safer, no underflow/overflow.  
**FP16**: Higher precision, narrow range — requires **loss scaling** to avoid underflow of small gradients.

### Loss Scaling

Small gradients in FP16 can underflow (round to 0). Solution: multiply loss by a large scale \( S \) before backward pass, then divide gradients by \( S \) before optimizer step.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler(
            init_scale=2**16,     # initial loss scale
            growth_factor=2.0,    # scale up if no overflow for growth_interval steps
            backoff_factor=0.5,   # scale down on overflow
            growth_interval=2000, # steps between scale-up attempts
        )
        self.step_count = 0

    def train_step(self, x, y, criterion):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass in FP16/BF16
        with autocast(device_type="cuda", dtype=torch.float16):
            logits = self.model(x)
            loss = criterion(logits, y)

        # Backward pass: scale → backward → unscale → clip → step
        self.scaler.scale(loss).backward()

        # Unscale before clip
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping (optional but common)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step (skips if grads contain inf/nan due to overflow)
        self.scaler.step(self.optimizer)
        self.scaler.update()  # adjust loss scale

        self.step_count += 1
        current_scale = self.scaler.get_scale()
        return loss.item(), current_scale

# BF16 training (no loss scaling needed, simpler)
def train_step_bf16(model, optimizer, x, y, criterion):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = criterion(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
```

---

## Distributed Training

### Data Parallelism

Split the **data** across multiple GPUs; each GPU has a full model copy.

**Synchronous**: all GPUs compute gradients on their mini-batch, then average gradients before the update (DDP — standard in PyTorch).

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def ddp_training(rank, world_size, dataset, model_fn, n_epochs=10):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model = model_fn().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                  rank=rank, shuffle=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                          sampler=sampler, num_workers=4,
                                          pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)  # ensures different shuffling each epoch
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            # DDP automatically averages gradients across all ranks via all-reduce
            scaler.step(optimizer); scaler.update()

    cleanup_ddp()

# Launch: torchrun --nproc_per_node=4 train.py
```

### Model Parallelism

When the model is too large for one GPU, split the **model** across devices:

```python
class ModelParallelResNet(torch.nn.Module):
    """Split model across two GPUs."""
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        base = resnet50()
        # First half on GPU 0
        self.part1 = torch.nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2
        ).to("cuda:0")
        # Second half on GPU 1
        self.part2 = torch.nn.Sequential(
            base.layer3, base.layer4,
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            base.fc
        ).to("cuda:1")

    def forward(self, x):
        x = self.part1(x.to("cuda:0"))
        x = self.part2(x.to("cuda:1"))  # moves tensor between GPUs
        return x
```

### Gradient Accumulation

Simulate larger effective batch size with limited GPU memory:

```python
def train_with_accumulation(model, loader, optimizer, criterion,
                              accumulation_steps=8, device="cuda"):
    """
    Effective batch size = loader.batch_size * accumulation_steps.
    Useful when large batch needed for stable training but GPU memory is limited.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        is_update_step = (step + 1) % accumulation_steps == 0

        # Optionally: disable grad sync on non-update steps for DDP efficiency
        context = model.no_sync() if (hasattr(model, "no_sync") and not is_update_step) \
                  else contextlib.nullcontext()

        with context:
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y) / accumulation_steps  # normalize
            scaler.scale(loss).backward()

        if is_update_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

---

## Hyperparameter Tuning for Optimizers

### Key Hyperparameters

| Optimizer | Critical HPs | Typical Range |
|---|---|---|
| SGD | `lr`, `momentum`, `weight_decay` | lr: 0.001–0.1, β: 0.9–0.99 |
| Adam | `lr`, `beta1`, `beta2`, `eps`, `weight_decay` | lr: 1e-4 to 1e-2 |
| AdamW | same + `weight_decay` | weight_decay: 0.01–0.1 |
| RMSprop | `lr`, `alpha` (rho), `eps` | lr: 1e-4 to 1e-2 |

### LR Sensitivity Rules of Thumb

1. **Scale LR with batch size**: If batch doubles, multiply LR by \( \sqrt{2} \) (variance rule) or simply 2 (linear scaling — works for SGD, not always Adam)
2. **Adam default lr=3e-4**: Often a good starting point (Karpathy's constant)
3. **SGD default lr=0.1**: For CIFAR with cosine schedule

### Optimizer Selection Guide

| Task | Recommended Optimizer |
|---|---|
| NLP / Transformers | AdamW + warmup + cosine |
| Image classification (from scratch) | SGD + cosine / OneCycle |
| Image classification (transfer learning) | AdamW |
| RNN / LSTM | Adam or RMSprop |
| GANs | Adam (β₁=0.5) |
| Large batch (>8K) | LAMB or LARS |
| Fine-grained control | AdamW + custom schedule |

---

## Full Code Examples — Optimizer Comparison

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class QuadraticProblem:
    """
    Ill-conditioned quadratic: f(x,y) = 100*(x - x*)² + (y - y*)²
    Condition number = 100 → challenging for vanilla GD.
    """
    def __init__(self, x_star=1.0, y_star=2.0):
        self.x_star = torch.tensor([x_star, y_star], dtype=torch.float32)

    def loss(self, theta):
        d = theta - self.x_star
        return 100 * d[0]**2 + d[1]**2

    def __call__(self, theta):
        return self.loss(theta)


def compare_optimizers_quadratic():
    problem = QuadraticProblem()
    theta_init = torch.tensor([5.0, 5.0])

    optimizers_cfg = {
        "SGD (lr=0.01)":          lambda p: torch.optim.SGD([p], lr=0.01),
        "SGD+Momentum (lr=0.01)": lambda p: torch.optim.SGD([p], lr=0.01, momentum=0.9),
        "SGD+Nesterov":           lambda p: torch.optim.SGD([p], lr=0.01, momentum=0.9, nesterov=True),
        "RMSprop":                lambda p: torch.optim.RMSprop([p], lr=0.01),
        "Adam (lr=0.1)":          lambda p: torch.optim.Adam([p], lr=0.1),
        "AdamW (lr=0.1)":         lambda p: torch.optim.AdamW([p], lr=0.1, weight_decay=1e-3),
    }

    n_steps = 300
    results = {}

    for name, opt_fn in optimizers_cfg.items():
        theta = theta_init.clone().requires_grad_(True)
        optimizer = opt_fn(theta)
        losses = []

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = problem(theta)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[name] = losses
        print(f"{name:<30}: final loss = {losses[-1]:.6f}, "
              f"converged (< 1e-4) at step {next((i for i,l in enumerate(losses) if l<1e-4), '>300')}")

    plt.figure(figsize=(12, 6))
    for name, losses in results.items():
        plt.semilogy(losses, label=name, linewidth=2)
    plt.xlabel("Step"); plt.ylabel("Loss (log scale)")
    plt.title("Optimizer Comparison on Ill-Conditioned Quadratic")
    plt.legend(loc="upper right"); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()
    return results


def compare_optimizers_neural_net():
    """Compare optimizers on MNIST classification."""
    import torchvision
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

    def make_model():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(),
            nn.Linear(128, 10),
        ).to(device)

    optimizers_cfg = {
        "SGD (cosine)":  lambda p: (torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=1e-4),
                                     "cosine"),
        "Adam":          lambda p: (torch.optim.Adam(p, lr=1e-3), None),
        "AdamW":         lambda p: (torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01), None),
        "RMSprop":       lambda p: (torch.optim.RMSprop(p, lr=1e-3, momentum=0.9), None),
    }

    n_epochs = 5
    criterion = nn.CrossEntropyLoss()
    all_train_losses = {}

    for name, opt_fn in optimizers_cfg.items():
        model = make_model()
        opt, sched_type = opt_fn(model.parameters())
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
                     if sched_type == "cosine" else None)
        epoch_losses = []

        for epoch in range(n_epochs):
            model.train(); running_loss = 0; n = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward(); opt.step()
                running_loss += loss.item(); n += 1
            if scheduler: scheduler.step()
            epoch_losses.append(running_loss / n)
            print(f"  {name} | Epoch {epoch+1}: loss={running_loss/n:.4f}")

        all_train_losses[name] = epoch_losses

    plt.figure(figsize=(10, 5))
    for name, losses in all_train_losses.items():
        plt.plot(range(1, n_epochs+1), losses, "o-", linewidth=2, label=name)
    plt.xlabel("Epoch"); plt.ylabel("Train Loss")
    plt.title("Optimizer Comparison on MNIST"); plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()

# Run comparisons:
# quad_results = compare_optimizers_quadratic()
# compare_optimizers_neural_net()
```

---

## Common Pitfalls in Optimization

1. **Learning rate too high**: Divergence or oscillation. Use LR warmup and schedule.
2. **L2 reg ≠ weight decay in Adam**: Use AdamW for proper decoupled weight decay.
3. **Batch size vs LR**: Larger batches → scale LR (linear or sqrt). Don't naively keep same LR.
4. **Gradient clipping**: Essential for RNNs/Transformers; clip by **norm**, not by value.
5. **Local minima vs saddle points**: In high dimensions, saddles dominate; momentum helps escape.
6. **Forgetting to zero grad**: `optimizer.zero_grad()` before backward, or gradients accumulate.
7. **Evaluating in training mode**: `model.eval()` disables dropout/BatchNorm training behavior.
8. **Ill-conditioning**: Use preconditioning (e.g., Adam's per-parameter scaling) or second-order methods.

---

## Summary Reference Table

| Method | Per-Step Cost | Convergence | Best Use Case |
|---|---|---|---|
| Batch GD | \(O(n)\) | \(O(1/T)\) convex | Small datasets, strict reproducibility |
| SGD | \(O(1)\) | \(O(1/\sqrt{T})\) | Large datasets, generalization |
| Momentum | \(O(1)\) | \(O(\sqrt{\kappa})\) better | General DL, CNNs |
| NAG | \(O(1)\) | \(O(1/T^2)\) convex | Convex problems |
| AdaGrad | \(O(d)\) | — | Sparse features (NLP) |
| RMSprop | \(O(d)\) | — | RNNs, non-stationary |
| Adam | \(O(d)\) | — | Default for DL |
| AdamW | \(O(d)\) | — | Transformers (standard) |
| L-BFGS | \(O(md)\) | Superlinear | Full-batch, shallow |
| Newton | \(O(d^3)\) | Quadratic | Very small models |

---

## Resources

### Books
- **Convex Optimization** (Boyd & Vandenberghe): web.stanford.edu/~boyd/cvxbook
- **Numerical Optimization** (Nocedal & Wright): Classic reference for second-order methods, L-BFGS, constrained optimization
- **Adam paper**: arxiv.org/abs/1412.6980
- **AdamW paper**: arxiv.org/abs/1711.05101
- **SAM paper**: arxiv.org/abs/2010.01412
- **LAMB paper**: arxiv.org/abs/1904.00962
- **Understanding Adam**: arxiv.org/abs/1902.09843
- **PyTorch Optimizer docs**: pytorch.org/docs/stable/optim.html

### References
- Nocedal & Wright (2006): *Numerical Optimization*, 2nd ed. Springer
- Boyd et al. (2011): Distributed Optimization and Statistical Learning via ADMM, *Foundations and Trends*
