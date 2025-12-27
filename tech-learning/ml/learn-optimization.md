# Optimization for Machine Learning and Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Convex Optimization](#convex-optimization)
3. [Gradient-Based Methods](#gradient-methods)
4. [Second-Order Methods](#second-order)
5. [Stochastic Optimization](#stochastic)
6. [Constrained Optimization](#constrained)
7. [Non-Convex Optimization](#non-convex)
8. [Hyperparameter Optimization](#hyperparameter)
9. [Applications in ML/DL](#applications)
10. [Practical Examples](#examples)

---

## Introduction {#introduction}

Optimization is at the heart of machine learning and deep learning. Most ML problems reduce to finding parameters that minimize a loss function.

### Why Optimization Matters
- **Training**: Finding optimal model parameters
- **Regularization**: Balancing fit and complexity
- **Hyperparameter Tuning**: Optimizing learning rates, architectures
- **Feature Selection**: Optimizing model complexity
- **Neural Architecture Search**: Finding optimal architectures

### Optimization Problem Formulation
```
minimize: f(θ)
subject to: g_i(θ) ≤ 0, i = 1, ..., m
            h_j(θ) = 0, j = 1, ..., p
```

---

## Convex Optimization {#convex-optimization}

### Convex Functions
A function f is convex if:
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for all x, y and λ ∈ [0, 1]

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Convex function: f(x) = x^2
def convex_func(x):
    return x**2

# Non-convex function: f(x) = x^4 - 4x^2
def non_convex_func(x):
    return x**4 - 4*x**2

x = np.linspace(-3, 3, 100)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, convex_func(x))
plt.title('Convex Function: f(x) = x²')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, non_convex_func(x))
plt.title('Non-Convex Function: f(x) = x⁴ - 4x²')
plt.grid(True)
plt.show()

# Check convexity using second derivative
# f''(x) ≥ 0 for all x → convex
x_check = np.linspace(-3, 3, 100)
d2_convex = 2  # Constant positive
d2_nonconvex = 12*x_check**2 - 8  # Can be negative

print(f"Convex function second derivative: {d2_convex} (always ≥ 0)")
print(f"Non-convex function second derivative at x=0: {12*0**2 - 8} (negative)")
```

### Convex Sets
```python
# A set C is convex if for any x, y ∈ C and λ ∈ [0, 1]:
# λx + (1-λ)y ∈ C

# Example: Unit circle (convex)
theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

plt.plot(circle_x, circle_y, 'b-', linewidth=2)
plt.fill(circle_x, circle_y, alpha=0.3)
plt.axis('equal')
plt.title('Convex Set: Unit Circle')
plt.grid(True)
plt.show()
```

### Properties of Convex Optimization
- **Global Optimum**: Any local minimum is global
- **Efficient Algorithms**: Many polynomial-time algorithms exist
- **Duality**: Strong duality holds under certain conditions

---

## Gradient-Based Methods {#gradient-methods}

### Gradient Descent
```python
def gradient_descent(f, grad_f, x0, learning_rate=0.1, n_iterations=100, 
                    tolerance=1e-6):
    """Standard gradient descent"""
    x = x0.copy()
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        
        # Check convergence
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged at iteration {i}")
            break
        
        # Update
        x = x - learning_rate * grad
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses

# Example: Minimize f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1

def grad_f(x):
    return 2*x + 2

x0 = np.array([5.0])
x_opt, history, losses = gradient_descent(f, grad_f, x0, 
                                           learning_rate=0.1, n_iterations=100)

x_vals = np.linspace(-2, 6, 100)
plt.plot(x_vals, f(x_vals), 'b-', alpha=0.3, label='f(x)')
plt.plot(history, losses, 'ro-', markersize=4, label='GD Path')
plt.axvline(x=-1, color='g', linestyle='--', label='Optimum (x=-1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal x: {x_opt[0]:.6f}")
print(f"True optimum: -1.0")
```

### Learning Rate Selection
```python
def test_learning_rates(f, grad_f, x0, lr_range, n_iterations=50):
    """Test different learning rates"""
    results = []
    
    for lr in lr_range:
        x_opt, history, losses = gradient_descent(f, grad_f, x0, lr, n_iterations)
        results.append({
            'lr': lr,
            'final_loss': losses[-1],
            'converged': len(losses) < n_iterations,
            'losses': losses
        })
    
    return results

lr_range = np.logspace(-3, 0, 20)
results = test_learning_rates(f, grad_f, x0, lr_range)

final_losses = [r['final_loss'] for r in results]
learning_rates = [r['lr'] for r in results]

plt.semilogx(learning_rates, final_losses, 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('Final Loss')
plt.title('Learning Rate vs Final Loss')
plt.grid(True)
plt.show()

# Find optimal learning rate
optimal_idx = np.argmin(final_losses)
print(f"Optimal learning rate: {learning_rates[optimal_idx]:.4f}")
```

### Momentum
```python
def gradient_descent_momentum(f, grad_f, x0, learning_rate=0.1, 
                              momentum=0.9, n_iterations=100):
    """Gradient descent with momentum"""
    x = x0.copy()
    v = np.zeros_like(x)  # Velocity
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        v = momentum * v - learning_rate * grad
        x = x + v
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses

# Compare with and without momentum
x_gd, hist_gd, loss_gd = gradient_descent(f, grad_f, x0, 0.1, 50)
x_mom, hist_mom, loss_mom = gradient_descent_momentum(f, grad_f, x0, 0.1, 0.9, 50)

plt.plot(loss_gd, label='Gradient Descent', linewidth=2)
plt.plot(loss_mom, label='With Momentum', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Gradient Descent: With vs Without Momentum')
plt.legend()
plt.grid(True)
plt.show()
```

### Nesterov Accelerated Gradient (NAG)
```python
def nesterov_accelerated_gradient(f, grad_f, x0, learning_rate=0.1,
                                  momentum=0.9, n_iterations=100):
    """Nesterov Accelerated Gradient"""
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(n_iterations):
        # Look ahead
        x_lookahead = x + momentum * v
        grad = grad_f(x_lookahead)
        v = momentum * v - learning_rate * grad
        x = x + v
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses

x_nag, hist_nag, loss_nag = nesterov_accelerated_gradient(f, grad_f, x0, 0.1, 0.9, 50)

plt.plot(loss_gd, label='GD', linewidth=2)
plt.plot(loss_mom, label='Momentum', linewidth=2)
plt.plot(loss_nag, label='NAG', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Comparison of Optimization Methods')
plt.legend()
plt.grid(True)
plt.show()
```

### Adaptive Learning Rates

#### AdaGrad
```python
def adagrad(f, grad_f, x0, learning_rate=0.1, epsilon=1e-8, n_iterations=100):
    """AdaGrad: Adaptive learning rate per parameter"""
    x = x0.copy()
    G = np.zeros_like(x)  # Sum of squared gradients
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        G += grad**2
        x = x - learning_rate * grad / (np.sqrt(G) + epsilon)
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses
```

#### RMSprop
```python
def rmsprop(f, grad_f, x0, learning_rate=0.01, beta=0.9, epsilon=1e-8, 
            n_iterations=100):
    """RMSprop: Exponential moving average of squared gradients"""
    x = x0.copy()
    v = np.zeros_like(x)  # Moving average of squared gradients
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        v = beta * v + (1 - beta) * grad**2
        x = x - learning_rate * grad / (np.sqrt(v) + epsilon)
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses
```

#### Adam
```python
def adam(f, grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999,
         epsilon=1e-8, n_iterations=100):
    """Adam: Adaptive Moment Estimation"""
    x = x0.copy()
    m = np.zeros_like(x)  # First moment (mean)
    v = np.zeros_like(x)  # Second moment (variance)
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(1, n_iterations + 1):
        grad = grad_f(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * grad**2
        
        # Bias correction
        m_hat = m / (1 - beta1**i)
        v_hat = v / (1 - beta2**i)
        
        # Update parameters
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses

# Compare adaptive methods
x_adam, hist_adam, loss_adam = adam(f, grad_f, x0, 0.1, n_iterations=50)
x_rms, hist_rms, loss_rms = rmsprop(f, grad_f, x0, 0.1, n_iterations=50)

plt.plot(loss_gd[:50], label='GD', linewidth=2)
plt.plot(loss_adam, label='Adam', linewidth=2)
plt.plot(loss_rms, label='RMSprop', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Adaptive Learning Rate Methods')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Second-Order Methods {#second-order}

### Newton's Method
```python
def newton_method(f, grad_f, hessian_f, x0, n_iterations=10):
    """Newton's method using second-order information"""
    x = x0.copy()
    history = [x.copy()]
    losses = [f(x)]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        H = hessian_f(x)
        
        # Solve H * Δx = -grad
        try:
            delta_x = np.linalg.solve(H, -grad)
            x = x + delta_x
        except np.linalg.LinAlgError:
            print(f"Hessian singular at iteration {i}")
            break
        
        history.append(x.copy())
        losses.append(f(x))
    
    return x, np.array(history), losses

# Example: f(x) = x^4 - 4x^2 + 2x
def f_newton(x):
    return x**4 - 4*x**2 + 2*x

def grad_f_newton(x):
    return 4*x**3 - 8*x + 2

def hessian_f_newton(x):
    return 12*x**2 - 8

x0_newton = np.array([2.0])
x_newton, hist_newton, loss_newton = newton_method(f_newton, grad_f_newton, 
                                                    hessian_f_newton, x0_newton)

x_vals = np.linspace(-2, 2, 100)
plt.plot(x_vals, f_newton(x_vals), 'b-', alpha=0.3, label='f(x)')
plt.plot(hist_newton, loss_newton, 'ro-', markersize=6, label='Newton Method')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title("Newton's Method")
plt.legend()
plt.grid(True)
plt.show()
```

### Quasi-Newton Methods (BFGS)
```python
from scipy.optimize import minimize

# BFGS (Broyden-Fletcher-Goldfarb-Shanno) approximation
result = minimize(f_newton, x0_newton, method='BFGS', jac=grad_f_newton)
print(f"BFGS optimal x: {result.x[0]:.6f}")
print(f"Converged: {result.success}")
```

---

## Stochastic Optimization {#stochastic}

### Stochastic Gradient Descent (SGD)
```python
def stochastic_gradient_descent(f_batch, grad_f_batch, X, y, x0, 
                               learning_rate=0.01, batch_size=32, 
                               n_epochs=10):
    """Stochastic Gradient Descent"""
    n_samples = len(X)
    x = x0.copy()
    history = [x.copy()]
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]
            
            # Compute gradient on batch
            grad = grad_f_batch(x, batch_X, batch_y)
            
            # Update
            x = x - learning_rate * grad
            history.append(x.copy())
    
    return x, np.array(history)

# Example: Linear regression
def mse_loss_batch(w, X, y):
    y_pred = X @ w
    return np.mean((y_pred - y)**2)

def mse_grad_batch(w, X, y):
    y_pred = X @ w
    return (2/len(y)) * X.T @ (y_pred - y)

# Generate data
np.random.seed(42)
X = np.random.randn(1000, 5)
true_w = np.array([1, 2, 3, 4, 5])
y = X @ true_w + 0.1 * np.random.randn(1000)

w0 = np.zeros(5)
w_sgd, hist_sgd = stochastic_gradient_descent(mse_loss_batch, mse_grad_batch,
                                               X, y, w0, learning_rate=0.01,
                                               batch_size=32, n_epochs=10)

print(f"True weights: {true_w}")
print(f"SGD weights: {w_sgd}")
print(f"Error: {np.linalg.norm(w_sgd - true_w):.6f}")
```

### Mini-Batch Gradient Descent
```python
def compare_batch_sizes():
    """Compare different batch sizes"""
    batch_sizes = [1, 10, 32, 100, 1000]
    results = []
    
    for batch_size in batch_sizes:
        w, hist = stochastic_gradient_descent(mse_loss_batch, mse_grad_batch,
                                             X, y, w0, learning_rate=0.01,
                                             batch_size=batch_size, n_epochs=10)
        results.append({
            'batch_size': batch_size,
            'weights': w,
            'error': np.linalg.norm(w - true_w)
        })
    
    return results

results = compare_batch_sizes()
for r in results:
    print(f"Batch size {r['batch_size']:4d}: Error = {r['error']:.6f}")
```

---

## Constrained Optimization {#constrained}

### Lagrange Multipliers
```python
from scipy.optimize import minimize

# Minimize f(x, y) = x^2 + y^2 subject to x + y = 1
def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

# Solve
result = minimize(objective, [0, 0], 
                 constraints={'type': 'eq', 'fun': constraint})
print(f"Optimal point: ({result.x[0]:.4f}, {result.x[1]:.4f})")
print(f"Constraint satisfied: {constraint(result.x):.6f}")
print(f"Optimal value: {result.fun:.4f}")
```

### Projected Gradient Descent
```python
def projected_gradient_descent(f, grad_f, x0, projection, learning_rate=0.1,
                               n_iterations=100):
    """Gradient descent with projection onto constraint set"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(n_iterations):
        grad = grad_f(x)
        x = x - learning_rate * grad
        x = projection(x)  # Project onto constraint set
        history.append(x.copy())
    
    return x, np.array(history)

# Example: Project onto unit ball
def project_unit_ball(x):
    norm = np.linalg.norm(x)
    if norm > 1:
        return x / norm
    return x

# Minimize f(x) = ||x - [2, 0]||^2 subject to ||x|| ≤ 1
def f_constrained(x):
    return np.sum((x - np.array([2, 0]))**2)

def grad_f_constrained(x):
    return 2 * (x - np.array([2, 0]))

x0_constrained = np.array([2.0, 0.0])
x_opt, hist = projected_gradient_descent(f_constrained, grad_f_constrained,
                                        x0_constrained, project_unit_ball)

print(f"Optimal x: {x_opt}")
print(f"Constraint satisfied: ||x|| = {np.linalg.norm(x_opt):.4f} ≤ 1")
```

---

## Non-Convex Optimization {#non-convex}

### Local Minima Problem
```python
# Non-convex function with multiple local minima
def non_convex_func(x):
    return x**4 - 4*x**2 + 2*x

def grad_non_convex(x):
    return 4*x**3 - 8*x + 2

x = np.linspace(-2.5, 2.5, 100)
y = non_convex_func(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Non-Convex Function with Multiple Local Minima')
plt.grid(True)
plt.show()

# Try different starting points
starting_points = [-2, 0, 2]
for x0 in starting_points:
    x_opt, hist, losses = gradient_descent(non_convex_func, grad_non_convex,
                                          np.array([x0]), learning_rate=0.01,
                                          n_iterations=1000)
    plt.plot(hist, losses, 'o-', markersize=3, label=f'Start at {x0}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent from Different Starting Points')
plt.legend()
plt.grid(True)
plt.show()
```

### Simulated Annealing
```python
def simulated_annealing(f, x0, T0=100, cooling_rate=0.95, n_iterations=1000):
    """Simulated annealing for global optimization"""
    x = x0.copy()
    best_x = x.copy()
    best_f = f(x)
    T = T0
    
    history = [x.copy()]
    best_history = [best_f]
    
    for i in range(n_iterations):
        # Generate neighbor
        x_new = x + np.random.randn(*x.shape) * 0.1
        
        # Accept or reject
        delta_f = f(x_new) - f(x)
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / T):
            x = x_new
            if f(x) < best_f:
                best_x = x.copy()
                best_f = f(x)
        
        T *= cooling_rate
        history.append(x.copy())
        best_history.append(best_f)
    
    return best_x, np.array(history), best_history

x_sa, hist_sa, best_sa = simulated_annealing(non_convex_func, np.array([0.0]))

x_vals = np.linspace(-2.5, 2.5, 100)
plt.plot(x_vals, non_convex_func(x_vals), 'b-', alpha=0.3)
plt.plot(hist_sa, [non_convex_func(x) for x in hist_sa], 'ro', markersize=2, alpha=0.5)
plt.plot(x_sa, non_convex_func(x_sa), 'g*', markersize=15, label='Best')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Simulated Annealing')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Hyperparameter Optimization {#hyperparameter}

### Grid Search
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define parameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Random Search
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(10, 200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(rf, param_distributions, n_iter=50,
                                   cv=5, scoring='accuracy', random_state=42)
random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

### Bayesian Optimization
```python
from skopt import gp_minimize
from skopt.space import Real, Integer

# Define search space
space = [
    Integer(10, 200, name='n_estimators'),
    Integer(3, 20, name='max_depth'),
    Integer(2, 20, name='min_samples_split')
]

def objective(params):
    n_est, max_d, min_split = params
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                min_samples_split=min_split, random_state=42)
    scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    return -scores.mean()  # Minimize negative accuracy

result = gp_minimize(objective, space, n_calls=50, random_state=42)
print(f"Best parameters: {result.x}")
print(f"Best score: {-result.fun:.4f}")
```

---

## Applications in ML/DL {#applications}

### 1. Training Neural Networks
```python
# Adam optimizer is commonly used
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
# Training uses Adam optimizer automatically
```

### 2. Regularized Regression
```python
# Ridge regression: L2 regularization
# Lasso regression: L1 regularization

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)  # L2 regularization
lasso = Lasso(alpha=1.0)   # L1 regularization

# Both solve optimization problems with constraints
```

---

## Practical Examples {#examples}

### Example 1: Logistic Regression from Scratch
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # Gradients
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > 0.5).astype(int)

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, random_state=42)
lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
lr.fit(X, y)
y_pred = lr.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Key Takeaways

1. **Convex Optimization**: Guaranteed global optimum
2. **Gradient Descent**: Foundation of ML optimization
3. **Adaptive Methods**: Adam, RMSprop for better convergence
4. **Stochastic Methods**: SGD for large datasets
5. **Second-Order Methods**: Faster convergence but expensive
6. **Constrained Optimization**: Handle constraints with Lagrange multipliers
7. **Hyperparameter Optimization**: Grid search, random search, Bayesian optimization

---

## Additional Resources

- **Convex Optimization (Boyd)**: Comprehensive textbook
- **Numerical Optimization (Nocedal & Wright)**: Advanced methods
- **Deep Learning Book (Goodfellow)**: Optimization in DL context

---

**Master optimization to train better ML/DL models!**

