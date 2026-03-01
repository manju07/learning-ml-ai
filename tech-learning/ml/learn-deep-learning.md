# Deep Learning: Comprehensive Guide from Foundations to Advanced

## Table of Contents
1. [Introduction to Deep Learning](#introduction-to-deep-learning)
2. [Perceptrons and the Linear Unit](#perceptrons-and-the-linear-unit)
3. [Multi-Layer Perceptrons (MLPs)](#multi-layer-perceptrons-mlps)
4. [Activation Functions — Complete Reference](#activation-functions--complete-reference)
5. [Backpropagation — Full Mathematical Derivation](#backpropagation--full-mathematical-derivation)
6. [Weight Initialization](#weight-initialization)
7. [Normalization Layers](#normalization-layers)
8. [Regularization Techniques](#regularization-techniques)
9. [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
10. [Residual Connections and Skip Connections](#residual-connections-and-skip-connections)
11. [Attention Mechanisms](#attention-mechanisms)
12. [Sequence Models: RNN, LSTM, GRU](#sequence-models-rnn-lstm-gru)
13. [Modern Architecture Overview](#modern-architecture-overview)
14. [Transfer Learning and Fine-Tuning](#transfer-learning-and-fine-tuning)
15. [Training Tricks and Practical Tips](#training-tricks-and-practical-tips)
16. [Full PyTorch Training Pipeline](#full-pytorch-training-pipeline)

---

## Introduction to Deep Learning

Deep learning is a branch of machine learning that uses artificial neural networks with many layers (hence *deep*) to learn hierarchical representations of data. Unlike classical ML methods that rely on hand-crafted features, deep networks learn features automatically, layer by layer, from raw input.

### Why Deep Learning Works

The **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991) guarantees that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision. However, deep networks can represent exponentially more functions efficiently than shallow ones.

**Key insights:**
- Each layer learns a different level of abstraction
- Depth allows composition of non-linear transformations
- Gradient-based optimization finds good solutions in practice despite non-convexity
- Stochastic training + large models generalize surprisingly well

### Mathematical Notation

Throughout this document we use:
- \( x \in \mathbb{R}^d \): input vector
- \( W^{(l)} \): weight matrix for layer \( l \)
- \( b^{(l)} \): bias vector for layer \( l \)
- \( a^{(l)} \): pre-activation at layer \( l \) (also called logits)
- \( h^{(l)} \): post-activation at layer \( l \) (hidden state)
- \( \sigma(\cdot) \): activation function
- \( \hat{y} \): model output / prediction
- \( \mathcal{L} \): loss function

---

## Perceptrons and the Linear Unit

### The Biological Inspiration

The perceptron (Rosenblatt, 1958) is modeled after neurons: it receives multiple signals, sums them (weighted), and fires if the total exceeds a threshold.

### Mathematical Formulation

Given input \( x = [x_1, x_2, \ldots, x_d]^\top \) and weights \( w = [w_1, w_2, \ldots, w_d]^\top \):

\[
z = w^\top x + b = \sum_{i=1}^{d} w_i x_i + b
\]

\[
\hat{y} = \text{step}(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}
\]

**Perceptron update rule** (for misclassified sample \( (x_i, y_i) \)):
\[
w \leftarrow w + \eta (y_i - \hat{y}_i) x_i
\]
\[
b \leftarrow b + \eta (y_i - \hat{y}_i)
\]

**Limitation**: The perceptron convergence theorem guarantees convergence only for *linearly separable* data.

### Perceptron Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """Single-layer perceptron with step activation."""

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self._predict_single(xi)
                delta = self.lr * (yi - pred)
                self.weights += delta * xi
                self.bias += delta
                errors += int(delta != 0)
            self.errors_per_epoch.append(errors)
            if errors == 0:
                break

    def _predict_single(self, x):
        return 1 if (np.dot(x, self.weights) + self.bias) >= 0 else 0

    def predict(self, X):
        return np.array([self._predict_single(xi) for xi in X])

    def plot_errors(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.errors_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel("Misclassifications")
        plt.title("Perceptron Learning Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# --- AND gate example ---
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

p = Perceptron(learning_rate=0.1)
p.fit(X_and, y_and)
print("AND gate predictions:", p.predict(X_and))  # [0 0 0 1]

# --- XOR — NOT linearly separable (perceptron fails) ---
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
p_xor = Perceptron(learning_rate=0.1, n_iterations=100)
p_xor.fit(X_xor, y_xor)
print("XOR predictions (will fail):", p_xor.predict(X_xor))  # Not [0 1 1 0]
```

---

## Multi-Layer Perceptrons (MLPs)

### Architecture

An MLP stacks multiple layers of linear units followed by non-linear activations:

```
Input (x) → [W₁, b₁] → σ → [W₂, b₂] → σ → ... → [Wₗ, bₗ] → output (ŷ)
```

### Forward Pass — Full Mathematics

For a network with \( L \) layers:

**Layer 0** (input): \( h^{(0)} = x \)

**Layers 1 through L**:
\[
a^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}
\]
\[
h^{(l)} = \sigma\left(a^{(l)}\right)
\]

**Output**: \( \hat{y} = h^{(L)} \)

Where:
- \( W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}} \)
- \( b^{(l)} \in \mathbb{R}^{n_l} \)
- \( \sigma \) is an element-wise activation function

### Vectorized Batch Forward Pass

For a batch of \( m \) samples, \( X \in \mathbb{R}^{m \times d} \):

\[
A^{(l)} = H^{(l-1)} W^{(l)\top} + \mathbf{1} b^{(l)\top}
\]
\[
H^{(l)} = \sigma\left(A^{(l)}\right)
\]

### MLP from Scratch (NumPy)

```python
import numpy as np

class MLP:
    """
    Multi-Layer Perceptron implemented from scratch.
    Supports arbitrary depth, ReLU hidden activations, sigmoid output.
    """

    def __init__(self, layer_sizes: list, learning_rate: float = 0.01):
        """
        layer_sizes: e.g. [2, 4, 4, 1]  → input dim=2, two hidden layers of 4, output dim=1
        """
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        # He initialization for ReLU layers
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        self.activations = []  # store for backprop
        self.z_values = []

    # ---- Activation functions ----
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def sigmoid_deriv(z):
        s = MLP.sigmoid(z)
        return s * (1.0 - s)

    # ---- Forward pass ----
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        H = X
        L = len(self.weights)

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = H @ W + b
            self.z_values.append(Z)
            if i < L - 1:
                H = self.relu(Z)
            else:
                H = self.sigmoid(Z)  # output layer
            self.activations.append(H)

        return H

    # ---- Backward pass ----
    def backward(self, X, y):
        m = X.shape[0]
        L = len(self.weights)
        output = self.activations[-1]

        # Output layer delta (binary cross-entropy + sigmoid)
        delta = (output - y) / m  # shape: (m, output_dim)

        grads_W = [None] * L
        grads_b = [None] * L

        for i in reversed(range(L)):
            grads_W[i] = self.activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)

            if i > 0:
                # Backpropagate through ReLU
                delta = delta @ self.weights[i].T * self.relu_deriv(self.z_values[i - 1])

        # Update
        for i in range(L):
            self.weights[i] -= self.lr * grads_W[i]
            self.biases[i] -= self.lr * grads_b[i]

    def binary_cross_entropy(self, y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X, y, epochs: int = 2000, print_every: int = 200):
        history = []
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = self.binary_cross_entropy(y_pred, y)
            self.backward(X, y)
            history.append(loss)
            if epoch % print_every == 0:
                acc = np.mean((y_pred > 0.5).astype(int) == y)
                print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        return history


# --- XOR problem (needs hidden layers) ---
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

mlp = MLP(layer_sizes=[2, 8, 8, 1], learning_rate=0.1)
history = mlp.train(X_xor, y_xor, epochs=5000, print_every=1000)
preds = mlp.forward(X_xor)
print("\nXOR predictions:", (preds > 0.5).astype(int).flatten())
```

---

## Activation Functions — Complete Reference

Activation functions introduce non-linearity, enabling networks to approximate complex functions. Without them, a deep network collapses to a single linear transformation.

### 1. Sigmoid

\[
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
\]

- **Range**: (0, 1)
- **Pros**: Smooth, interpretable as probability, bounded output
- **Cons**: Vanishing gradients (gradient < 0.25 always), not zero-centered, slow convergence
- **Use**: Output layer for binary classification, gates in LSTM

### 2. Tanh

\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)
\]

- **Range**: (-1, 1)
- **Pros**: Zero-centered (better gradient flow than sigmoid), stronger gradients than sigmoid
- **Cons**: Still suffers from vanishing gradients for large |x|
- **Use**: Hidden layers in RNNs/LSTMs, shallow networks

### 3. ReLU (Rectified Linear Unit)

\[
\text{ReLU}(x) = \max(0, x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
\]

- **Range**: [0, ∞)
- **Pros**: No vanishing gradient for positive inputs, computationally efficient, sparse activations (biological plausibility), fast convergence
- **Cons**: Dying ReLU problem (neurons that output 0 always can stop learning), not zero-centered, unbounded
- **Use**: Default choice for hidden layers in CNNs and MLPs

### 4. Leaky ReLU

\[
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}, \quad \alpha \approx 0.01
\]

- **Range**: (-∞, ∞)
- **Pros**: Fixes dying ReLU, allows small negative gradient
- **Cons**: The \( \alpha \) hyperparameter must be tuned; Parametric ReLU (PReLU) learns \( \alpha \)
- **Use**: When dying ReLU is a concern

### 5. ELU (Exponential Linear Unit)

\[
\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
\]

- **Range**: (−α, ∞)
- **Pros**: Smooth everywhere, negative values push mean activations closer to zero (self-normalizing property), robust to noisy inputs
- **Cons**: Computationally more expensive than ReLU
- **Use**: Deeper networks, alternative to batch normalization

### 6. GELU (Gaussian Error Linear Unit)

\[
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
\]

Approximation: \( \text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right) \)

- **Range**: ≈ (-0.17, ∞)
- **Pros**: State-of-the-art in Transformers (BERT, GPT), smooth stochastic gating (weights input by its quantile), better than ReLU for NLP
- **Cons**: More expensive, not straightforward to interpret
- **Use**: Transformer models, NLP

### 7. Swish

\[
\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}
\]

When \( \beta = 1 \): \( \text{Swish}(x) = x \cdot \sigma(x) \)

- **Range**: ≈ (-0.28, ∞)
- **Pros**: Smooth, non-monotonic, generally outperforms ReLU on deeper networks, discovered via neural architecture search
- **Cons**: Slightly more expensive than ReLU
- **Use**: EfficientNet, recent vision models

### 8. Mish

\[
\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))
\]

- **Range**: ≈ (-0.31, ∞)
- **Pros**: Smooth, non-monotonic, self-gated, slightly better than Swish in practice
- **Cons**: Most expensive of common activations
- **Use**: YOLOv4, recent object detection models

### 9. Softmax (output layer only)

\[
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

Used in the output layer for multi-class classification. Converts logits to a probability distribution.

### Comparison Code

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 400)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.minimum(x, 0)) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def mish(x):
    return x * np.tanh(np.log1p(np.exp(np.minimum(x, 20))))

activations = {
    "Sigmoid": sigmoid,
    "Tanh": tanh,
    "ReLU": relu,
    "Leaky ReLU": leaky_relu,
    "ELU": elu,
    "GELU": gelu,
    "Swish": swish,
    "Mish": mish,
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for ax, (name, fn) in zip(axes, activations.items()):
    ax.plot(x, fn(x), linewidth=2, color="steelblue")
    ax.set_title(name, fontsize=13)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlim(-4, 4)
    ax.grid(True, alpha=0.3)

plt.suptitle("Activation Functions Comparison", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("activation_functions.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Quick Selection Guide

| Problem Type | Output Activation | Hidden Activation |
|---|---|---|
| Binary classification | Sigmoid | ReLU / GELU |
| Multi-class classification | Softmax | ReLU / GELU |
| Regression | Linear (none) | ReLU / ELU |
| NLP / Transformers | Softmax | GELU |
| Object detection | Sigmoid / Softmax | LeakyReLU / Mish |

---

## Backpropagation — Full Mathematical Derivation

Backpropagation (Rumelhart et al., 1986) is an efficient algorithm for computing the gradient of the loss \( \mathcal{L} \) with respect to all parameters using the **chain rule of calculus**.

### The Chain Rule

If \( f = f(g(x)) \), then:
\[
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
\]

For vectors (Jacobians):
\[
\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}}{\partial \mathbf{g}} \cdot \frac{\partial \mathbf{g}}{\partial \mathbf{x}}
\]

### Forward Pass (2-layer network)

Consider a 2-layer network with:
- \( z^{(1)} = W^{(1)} x + b^{(1)} \)
- \( h^{(1)} = \text{ReLU}(z^{(1)}) \)
- \( z^{(2)} = W^{(2)} h^{(1)} + b^{(2)} \)
- \( \hat{y} = \sigma(z^{(2)}) \) (sigmoid for binary classification)
- \( \mathcal{L} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})] \) (binary cross-entropy)

### Backward Pass (Derivation Step by Step)

**Step 1**: \( \frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} \)

**Step 2**: \( \frac{\partial \hat{y}}{\partial z^{(2)}} = \hat{y}(1 - \hat{y}) \)

Therefore: \( \frac{\partial \mathcal{L}}{\partial z^{(2)}} = \hat{y} - y \) (elegant result!)

**Step 3** (gradients w.r.t. output layer):
\[
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \frac{\partial \mathcal{L}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial W^{(2)}} = (\hat{y} - y) \cdot h^{(1)\top}
\]
\[
\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \hat{y} - y
\]

**Step 4** (gradient flowing back to hidden layer):
\[
\frac{\partial \mathcal{L}}{\partial h^{(1)}} = W^{(2)\top} \cdot \frac{\partial \mathcal{L}}{\partial z^{(2)}} = W^{(2)\top} (\hat{y} - y)
\]

**Step 5**: ReLU backward:
\[
\frac{\partial \mathcal{L}}{\partial z^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(1)}} \odot \mathbf{1}[z^{(1)} > 0]
\]

**Step 6** (gradients w.r.t. first layer):
\[
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial z^{(1)}} \cdot x^\top
\]
\[
\frac{\partial \mathcal{L}}{\partial b^{(1)}} = \frac{\partial \mathcal{L}}{\partial z^{(1)}}
\]

### The δ (Error Signal) Notation

Define the *error signal* at layer \( l \):
\[
\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial a^{(l)}}
\]

Then the recurrence is:
\[
\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L}
\]
\[
\delta^{(l)} = \left(W^{(l+1)\top} \delta^{(l+1)}\right) \odot \sigma'\left(a^{(l)}\right)
\]

And gradients:
\[
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot h^{(l-1)\top}
\]
\[
\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}
\]

### Computational Graph and Automatic Differentiation

Modern frameworks (PyTorch, JAX) build computational graphs and apply the chain rule automatically using **reverse-mode automatic differentiation** (also called backprop). Each operation records its local gradient, and gradients are accumulated in reverse topological order.

```python
import torch
import torch.nn as nn

# PyTorch autograd demonstration
x = torch.tensor([[0.5, -0.3]], requires_grad=False)
W1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# Forward pass
z1 = x @ W1 + b1
h1 = torch.relu(z1)
z2 = h1 @ W2 + b2
y_pred = torch.sigmoid(z2)

# Loss
y_true = torch.tensor([[1.0]])
loss = nn.BCELoss()(y_pred, y_true)

# Backward pass — PyTorch computes all gradients
loss.backward()

print("dL/dW1:", W1.grad.shape)  # (2, 4)
print("dL/dW2:", W2.grad.shape)  # (4, 1)
```

---

## Weight Initialization

Proper initialization prevents vanishing/exploding gradients from the very start of training.

### Why Initialization Matters

If weights are too large → activations explode → gradients explode  
If weights are too small → activations vanish → gradients vanish  
If all weights are equal → all neurons learn the same function (symmetry breaking)

### 1. Xavier / Glorot Initialization (2010)

Designed for **sigmoid** and **tanh** activations. Maintains variance of activations and gradients across layers.

**Condition**: Var(output) = Var(input)

**Formula**: 
\[
W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
\]
or equivalently:
\[
W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)
\]

### 2. He / Kaiming Initialization (2015)

Designed for **ReLU** activations. Accounts for the fact that ReLU zeros out half the neurons.

\[
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
\]

**Fan-in mode** (recommended for ReLU): \( \text{std} = \sqrt{2 / n_{in}} \)  
**Fan-out mode**: \( \text{std} = \sqrt{2 / n_{out}} \)

### 3. Orthogonal Initialization

Initializes weight matrices as (approximately) orthogonal matrices via SVD. Preserves gradient norms exactly for linear networks and helps deep RNNs.

\[
W = UV^\top \quad \text{from SVD: } A = U \Sigma V^\top
\]

### 4. LeCun Initialization

\[
W \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right)
\]
Original suggestion by LeCun (1998), appropriate for SELU activations (self-normalizing).

### Implementation Comparison

```python
import torch
import torch.nn as nn

def compare_initializations(activation="relu"):
    """Compare gradient magnitudes under different initializations."""
    layers_per_init = {
        "zeros": lambda t: nn.init.zeros_(t),
        "uniform": lambda t: nn.init.uniform_(t, -0.5, 0.5),
        "xavier_uniform": nn.init.xavier_uniform_,
        "xavier_normal": nn.init.xavier_normal_,
        "he_uniform": lambda t: nn.init.kaiming_uniform_(t, nonlinearity=activation),
        "he_normal": lambda t: nn.init.kaiming_normal_(t, nonlinearity=activation),
        "orthogonal": nn.init.orthogonal_,
    }

    act_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
    n_layers = 10
    dim = 256

    print(f"\nActivation: {activation}")
    print(f"{'Initialization':<20} {'Mean std across layers':>25}")
    print("-" * 50)

    for name, init_fn in layers_per_init.items():
        x = torch.randn(32, dim)
        stds = [x.std().item()]

        for _ in range(n_layers):
            W = torch.empty(dim, dim)
            init_fn(W)
            x = act_fn(x @ W)
            stds.append(x.std().item())

        print(f"{name:<20} {' -> '.join(f'{s:.3f}' for s in stds[::3])}")

compare_initializations("relu")
compare_initializations("tanh")


# PyTorch built-in initialization
class InitializedMLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
```

---

## Normalization Layers

Normalization layers stabilize and accelerate training by ensuring activations have consistent scale.

### 1. Batch Normalization (Ioffe & Szegedy, 2015)

Normalizes over the **batch dimension** for each feature.

**Forward pass** (for feature dimension \( j \), over batch \( B \)):

\[
\mu_j = \frac{1}{|B|} \sum_{i \in B} x_{i,j}
\]
\[
\sigma_j^2 = \frac{1}{|B|} \sum_{i \in B} (x_{i,j} - \mu_j)^2
\]
\[
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
\]
\[
y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j
\]

Where \( \gamma_j, \beta_j \) are **learnable** scale and shift parameters.

**At inference**: uses running statistics (exponential moving average of \( \mu \) and \( \sigma^2 \)):
\[
\mu_{\text{run}} \leftarrow (1 - m) \cdot \mu_{\text{run}} + m \cdot \mu_B
\]

**Effects**: Reduces internal covariate shift, acts as regularizer, allows higher learning rates, reduces sensitivity to initialization.

**Limitation**: Problematic with small batch sizes; behavior differs between train and test.

### 2. Layer Normalization (Ba et al., 2016)

Normalizes over the **feature dimension** for each sample (not across batch).

\[
\mu_i = \frac{1}{D} \sum_{j=1}^{D} x_{i,j}, \quad \sigma_i^2 = \frac{1}{D} \sum_{j=1}^{D} (x_{i,j} - \mu_i)^2
\]
\[
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j
\]

**Advantages over BN**: Works identically during train and inference, works with batch size 1, standard for **Transformers and RNNs**.

### 3. Group Normalization (Wu & He, 2018)

Divides channels into \( G \) groups and normalizes within each group. A middle ground between BN (batch) and LN (layer).

\[
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_{i,g}}{\sqrt{\sigma_{i,g}^2 + \epsilon}}
\]

Where \( g = \lfloor j \cdot G / C \rfloor \) is the group index. Reduces to LN when G=1, to Instance Norm when G=C.

**Used in**: Object detection (Mask R-CNN), when batch size is small.

### 4. Instance Normalization

Normalizes over spatial (H, W) dimensions per-channel per-sample. Used in style transfer.

### Normalization Comparison Code

```python
import torch
import torch.nn as nn

batch_size, channels, height, width = 4, 8, 32, 32

x = torch.randn(batch_size, channels, height, width)

# Batch Normalization
bn = nn.BatchNorm2d(channels)

# Layer Normalization (over all features per sample)
ln = nn.LayerNorm([channels, height, width])

# Group Normalization (G=4 groups of 2 channels each)
gn = nn.GroupNorm(num_groups=4, num_channels=channels)

# Instance Normalization
inst_n = nn.InstanceNorm2d(channels, affine=True)

for name, norm in [("BatchNorm", bn), ("LayerNorm", ln),
                    ("GroupNorm", gn), ("InstanceNorm", inst_n)]:
    out = norm(x)
    print(f"{name:<15}: input std={x.std():.4f}, output std={out.std():.4f} (detached)")
```

### When to Use Which

| Scenario | Recommended Normalization |
|---|---|
| CNNs with large batches | Batch Normalization |
| Transformers, NLP | Layer Normalization |
| Object detection, small batches | Group Normalization |
| Style transfer | Instance Normalization |
| RNNs | Layer Normalization |

---

## Regularization Techniques

Regularization reduces overfitting by constraining or modifying the training process.

### 1. L2 Regularization (Weight Decay)

Adds a penalty proportional to the squared magnitude of weights:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \sum_{l} \|W^{(l)}\|_F^2
\]

The gradient update becomes:
\[
W \leftarrow W - \eta \left(\nabla_W \mathcal{L} + \lambda W\right) = (1 - \eta\lambda) W - \eta \nabla_W \mathcal{L}
\]

This **shrinks** weights toward zero at each step, hence "weight decay." In Adam, L2 regularization ≠ weight decay (see AdamW).

### 2. L1 Regularization (Lasso)

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum_{l} \|W^{(l)}\|_1
\]

Gradient: \( \lambda \cdot \text{sign}(W) \) — promotes **sparse** solutions (many weights go to exactly zero).

### 3. Dropout (Srivastava et al., 2014)

During training, randomly set activations to zero with probability \( p \):

\[
h_i^{\text{train}} = \begin{cases} 0 & \text{with probability } p \\ \frac{h_i}{1-p} & \text{with probability } 1-p \end{cases}
\]

The \( \frac{1}{1-p} \) factor (inverted dropout) ensures expected output is unchanged.

At **test time**, use all neurons (no dropout). The network has implicitly trained an exponential ensemble of \( 2^N \) sub-networks.

**Effect**: Prevents co-adaptation of neurons, acts like bagging of sub-networks.

### 4. DropConnect

Randomly drops weights (rather than activations). Generalizes dropout.

### 5. Spatial Dropout (Dropout2D)

For CNNs: drops entire feature maps (2D channels) rather than individual neurons.

### 6. Early Stopping

Monitor validation loss; stop training when it stops improving (with patience):

```
if val_loss improves:
    save checkpoint
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= patience:
        stop training, restore best checkpoint
```

### 7. Data Augmentation

Artificially increases dataset size by applying label-preserving transformations. Effectively injects domain-specific prior knowledge.

**Image augmentations**: horizontal flip, rotation, crop, color jitter, CutOut, Mixup, CutMix, AutoAugment, RandAugment.

**Text augmentations**: synonym replacement, back-translation, insertion, deletion, swap.

### 8. Label Smoothing

Instead of hard one-hot labels, use:

\[
y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{onehot}} + \frac{\epsilon}{K}
\]

Prevents the model from becoming over-confident, improves calibration.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Dropout demonstration ----
class RegularizedMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# L2 regularization via weight_decay in optimizer
model = RegularizedMLP(784, 256, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Label smoothing loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, logits, targets):
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.cls - 1)
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        smooth_labels = one_hot * confidence + (1 - one_hot) * smooth_val
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss

# Data augmentation with torchvision
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    """Apply MixUp: randomly interpolate between two samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

## Vanishing and Exploding Gradients

### The Problem

During backpropagation, gradients are multiplied across \( L \) layers:
\[
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \cdot \prod_{l=2}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}}
\]

Each factor \( \frac{\partial h^{(l)}}{\partial h^{(l-1)}} = W^{(l)\top} \text{diag}(\sigma'(a^{(l)})) \).

**If** eigenvalues of these Jacobians are consistently < 1: gradients **vanish** exponentially with depth.  
**If** eigenvalues > 1: gradients **explode** exponentially.

For sigmoid: \( \sigma'(x) \leq 0.25 \) always → vanishing almost guaranteed in deep sigmoid networks.

### Causes

1. **Activation functions with saturating gradients** (sigmoid, tanh)
2. **Poor weight initialization** (too large or too small)
3. **Very deep networks** without residual connections

### Solutions

| Problem | Solution |
|---|---|
| Vanishing (sigmoid/tanh) | Use ReLU, GELU, or other non-saturating activations |
| Vanishing (deep nets) | Residual connections (ResNet), Highway networks |
| Exploding gradients | Gradient clipping |
| Both | Proper initialization (He, Xavier) |
| Training instability | Batch/Layer normalization |
| Long sequences | LSTM, GRU (gating), attention mechanisms |

### Gradient Clipping

```python
import torch

# Clip by norm (recommended for most cases)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# In training loop:
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

### Detecting Gradient Problems

```python
def check_gradients(model):
    """Check gradient statistics after backward pass."""
    print(f"\n{'Layer':<30} {'Mean grad':<15} {'Max grad':<15} {'Norm':<10}")
    print("-" * 70)
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.data
            print(f"{name:<30} {g.mean().item():<15.6f} {g.max().item():<15.6f} {g.norm().item():<10.4f}")
```

---

## Residual Connections and Skip Connections

### The Core Idea (He et al., 2016)

Instead of learning \( H(x) \) directly, learn the *residual*:
\[
H(x) = F(x) + x
\]
where \( F(x) = H(x) - x \) is the **residual function**.

If the optimal transformation is close to identity, it's easier to learn \( F(x) \approx 0 \) than \( H(x) \approx x \).

### Why Residuals Help

**Gradient flow**: The identity shortcut provides a direct path for gradients to flow backward without going through non-linear transformations:
\[
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left(\frac{\partial F}{\partial x} + 1\right)
\]

The "+1" guarantees gradients always flow, even when \( \frac{\partial F}{\partial x} \approx 0 \).

### ResNet Basic Block

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Standard ResNet residual block with two 3x3 convolutions.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (projection if dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)  # ← the residual addition
        return out


class BottleneckBlock(nn.Module):
    """
    ResNet-50/101/152 bottleneck block: 1x1 → 3x3 → 1x1 convolutions.
    More efficient than BasicBlock for deep networks.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid = out_channels
        expanded = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, expanded, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(expanded)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != expanded:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded, 1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + identity)
        return out
```

### Dense Connections (DenseNet, Huang et al., 2017)

Each layer receives feature maps from **all** preceding layers:
\[
h^{(l)} = H_l\left([h^{(0)}, h^{(1)}, \ldots, h^{(l-1)}]\right)
\]

Where \( [\cdot] \) is channel-wise concatenation.

---

## Attention Mechanisms

### Scaled Dot-Product Attention (Vaswani et al., 2017)

The fundamental building block of the Transformer architecture.

**Inputs**:
- Queries \( Q \in \mathbb{R}^{n \times d_k} \)
- Keys \( K \in \mathbb{R}^{m \times d_k} \)
- Values \( V \in \mathbb{R}^{m \times d_v} \)

**Formula**:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

**Why divide by** \( \sqrt{d_k} \)? The dot products grow with \( d_k \), pushing softmax into regions with tiny gradients. Dividing by \( \sqrt{d_k} \) stabilizes gradients.

### Multi-Head Attention

Rather than performing a single attention, run \( h \) attention functions in parallel with different learned projections:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\]
\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, S, D = x.shape
        x = x.view(B, S, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, heads, S, d_k)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        x, attn_weights = self.attention(Q, K, V, mask=mask)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        output = self.W_o(x)
        return output, attn_weights


# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
out, weights = mha(x, x, x)  # self-attention
print("Output shape:", out.shape)     # (2, 10, 512)
print("Attn weights:", weights.shape) # (2, 8, 10, 10)
```

---

## Sequence Models: RNN, LSTM, GRU

### Vanilla RNN

The fundamental recurrence:
\[
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
\]
\[
\hat{y}_t = W_{hy} h_t + b_y
\]

**Problem**: Suffers from vanishing gradients over long sequences because \( \frac{\partial h_t}{\partial h_{t-k}} \) involves products of Jacobians.

### LSTM (Long Short-Term Memory — Hochreiter & Schmidhuber, 1997)

LSTMs introduce a **cell state** \( c_t \) (the "memory") alongside the hidden state \( h_t \), controlled by three gates:

**Forget gate** — what to erase from cell state:
\[
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
\]

**Input gate** — what new information to write:
\[
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
\]
\[
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
\]

**Cell state update**:
\[
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
\]

**Output gate** — what to expose from cell state:
\[
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
\]
\[
h_t = o_t \odot \tanh(c_t)
\]

The **linear update** of \( c_t \) (no sigmoid/tanh squashing) ensures gradients can flow through without vanishing.

### GRU (Gated Recurrent Unit — Cho et al., 2014)

A simpler alternative to LSTM with two gates instead of three:

**Reset gate** — how much past hidden state to forget:
\[
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
\]

**Update gate** — interpolation between old and new hidden state:
\[
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
\]

**Candidate hidden state**:
\[
\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t] + b)
\]

**New hidden state**:
\[
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\]

GRU has fewer parameters than LSTM and often performs comparably.

### Implementation in PyTorch

```python
import torch
import torch.nn as nn

class LSTMFromScratch(nn.Module):
    """LSTM cell implemented manually (for understanding)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # All gates in one matrix multiplication for efficiency
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, states=None):
        """
        x: (batch, seq_len, input_size)
        Returns: output (batch, seq_len, hidden_size), (h_n, c_n)
        """
        batch, seq_len, _ = x.shape
        if states is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)
            c = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h, c = states

        outputs = []
        for t in range(seq_len):
            combined = torch.cat([h, x[:, t, :]], dim=1)
            gates = self.W(combined)  # (batch, 4 * hidden_size)

            f, i, g, o = gates.chunk(4, dim=1)
            f = torch.sigmoid(f)  # forget gate
            i = torch.sigmoid(i)  # input gate
            g = torch.tanh(g)    # cell gate (candidate)
            o = torch.sigmoid(o) # output gate

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1), (h, c)


# Using PyTorch built-in LSTM
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))  # (batch, seq, embed)
        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # Concatenate last forward and backward hidden states
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(self.dropout(hidden))
```

---

## Modern Architecture Overview

### CNN (Convolutional Neural Networks)
- **Inductive bias**: translation equivariance, local connectivity
- **Key idea**: shared weight filters + pooling for spatial hierarchy
- **Strengths**: Images, video, audio spectrograms
- **Landmark models**: LeNet, AlexNet, VGG, ResNet, EfficientNet, ConvNeXt

### RNN / LSTM / GRU
- **Inductive bias**: sequential order, shared parameters across time
- **Key idea**: recurrent hidden state captures temporal dependencies
- **Strengths**: NLP (pre-Transformer), time series, speech
- **Limitation**: Sequential computation, can't parallelize across time

### Transformer
- **Inductive bias**: permutation equivariance (position must be added explicitly)
- **Key idea**: attention mechanism relates any two positions in O(1)
- **Strengths**: NLP (BERT, GPT), vision (ViT), multimodal
- **Limitation**: Quadratic complexity in sequence length for full attention

### GNN (Graph Neural Networks)
- **Inductive bias**: permutation equivariance over nodes, local neighborhood aggregation
- **Key idea**: message passing — each node aggregates features from neighbors
- **Strengths**: Social networks, molecular properties, knowledge graphs
- **Common models**: GCN, GraphSAGE, GAT, GIN

---

## Transfer Learning and Fine-Tuning

### Why Transfer Learning Works

Pre-trained models have learned general representations (edges, textures, shapes for vision; grammar, semantics for NLP) from massive datasets. Fine-tuning adapts these representations to a new task with far less data.

### Strategies

**1. Feature extraction**: Freeze pre-trained layers, train only the new head.

**2. Full fine-tuning**: Unfreeze all layers, train with small learning rate.

**3. Gradual unfreezing** (ULMFiT strategy): Unfreeze one layer at a time, from last to first.

**4. Layer-wise learning rate decay**: Lower LR for earlier layers, higher for later layers.

### Fine-tuning with PyTorch

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# ── Strategy 1: Feature Extraction (frozen backbone) ──
def create_feature_extractor(num_classes: int, model_name: str = "resnet50"):
    backbone = getattr(models, model_name)(weights="IMAGENET1K_V2")

    # Freeze all parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Replace the final classifier
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return backbone

# ── Strategy 2: Full fine-tuning with differential LR ──
def create_fine_tuned_model(num_classes: int):
    backbone = models.resnet50(weights="IMAGENET1K_V2")
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)

    # Different LR for backbone vs head
    param_groups = [
        {"params": [p for n, p in backbone.named_parameters()
                    if "fc" not in n], "lr": 1e-5},
        {"params": backbone.fc.parameters(), "lr": 1e-3},
    ]
    return backbone, param_groups

backbone, param_groups = create_fine_tuned_model(num_classes=10)
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

# ── Strategy 3: Gradual unfreezing ──
def gradual_unfreeze(model, epoch, unfreeze_schedule):
    """
    unfreeze_schedule: {epoch: layer_pattern_to_unfreeze}
    Example: {0: "layer4", 2: "layer3", 4: "layer2"}
    """
    if epoch in unfreeze_schedule:
        pattern = unfreeze_schedule[epoch]
        unfrozen = 0
        for name, param in model.named_parameters():
            if pattern in name:
                param.requires_grad = True
                unfrozen += param.numel()
        print(f"Epoch {epoch}: Unfroze layers matching '{pattern}' ({unfrozen:,} params)")
```

---

## Training Tricks and Practical Tips

### 1. Learning Rate Scheduling

**Step decay**: Reduce LR by factor \( \gamma \) every \( k \) epochs:
\[
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}
\]

**Cosine annealing** (Loshchilov & Hutter, 2016):
\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi t}{T}\right)
\]

**Warmup**: Start with small LR, linearly increase to target over \( w \) steps. Critical for Transformers to stabilize early training.

**One-Cycle Policy** (Smith, 2018): Linear warmup → cosine annealing, often finds better solutions faster.

### 2. Mixed Precision Training (FP16/BF16)

Use 16-bit floating point for forward/backward pass (faster, less memory), but maintain 32-bit master weights for numerical stability. Typical speedup: 1.5–2× on Ampere+ GPUs.

**FP16 vs BF16**: 
- **FP16** (float16): 5 exponent, 10 mantissa bits. Small dynamic range → risk of overflow/underflow; requires loss scaling (GradScaler). Common on older GPUs.
- **BF16** (bfloat16): 8 exponent (same as FP32), 7 mantissa bits. Same dynamic range as FP32 → no loss scaling needed; more stable. Preferred on A100, H100, Intel Gaudi, Apple M-series.

**GradScaler** (required for FP16, optional for BF16): Scale loss before backward to avoid underflow of small gradients; unscale before optimizer step; adjust scale factor based on gradient infinity checks.

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    with autocast():  # Forward in FP16/BF16
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()  # Scaled backward
    scaler.unscale_(optimizer)    # Unscale gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)        # Step with unscaled gradients
    scaler.update()               # Update scale factor for next iter
```

**When to use**: BF16 if available (A100+); FP16 + GradScaler on V100 and older. Disable for layers sensitive to precision (e.g., some normalization, small batch LayerNorm).

### 3. Gradient Accumulation

Simulate larger batch sizes when memory is limited:

```python
accumulation_steps = 4  # effective batch = actual_batch * accumulation_steps

for step, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 4. Additional Optimization Tricks

**Exponential Moving Average (EMA)** of weights — often improves generalization:
```python
from copy import deepcopy
ema_model = deepcopy(model)
ema_decay = 0.999  # Per step: ema = decay * ema + (1 - decay) * param
@torch.no_grad()
def update_ema():
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
# Call update_ema() after each optimizer.step()
```

**Gradient checkpointing** — trade compute for memory; recompute activations during backward:
```python
from torch.utils.checkpoint import checkpoint
def forward_with_checkpoint(self, x):
    return checkpoint(self._forward_block, x, use_reentrant=False)
# Use for large models when OOM; 30–50% memory savings, ~20% slower.
```

**torch.compile** (PyTorch 2.0+) — JIT compile for speed:
```python
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune"
# 20–30% speedup on many models with minimal code change.
```

### 5. Learning Rate Finder

```python
def lr_finder(model, optimizer, criterion, dataloader,
              start_lr=1e-7, end_lr=10, num_iter=100):
    """Find optimal LR by exponentially increasing it and tracking loss."""
    lrs, losses = [], []
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    for param_group in optimizer.param_groups:
        param_group["lr"] = start_lr

    for i, (x, y) in enumerate(dataloader):
        if i >= num_iter:
            break
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_lr = start_lr * (lr_mult ** i)
        lrs.append(current_lr)
        losses.append(loss.item())

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    return lrs, losses
```

---

## Full PyTorch Training Pipeline

A production-quality training loop with all best practices:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
import time
import os

# ── Model ──
class DeepMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.3, use_bn: bool = True):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── Custom Dataset ──
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training / Evaluation Functions ──
def train_epoch(model, loader, optimizer, criterion, scaler, device, clip_norm=1.0):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():  # Mixed precision
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


# ── Full Training Script ──
def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    import numpy as np
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=10_000, n_features=50,
                               n_informative=30, n_classes=5, random_state=42)

    dataset = TabularDataset(X, y)
    n_val = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"] * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = DeepMLP(input_dim=50, hidden_dims=[256, 256, 128],
                    output_dim=5, dropout=0.3).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer + Scheduler + Loss
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                            weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"], pct_start=0.3
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                             criterion, scaler, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_acc": val_acc}, "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"Early stopping at epoch {epoch} (best val acc: {best_val_acc:.4f})")
            break

    return history, best_val_acc


config = {
    "epochs": 50,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 10,
}

# Uncomment to run:
# history, best_acc = train(config)
# print(f"\nBest validation accuracy: {best_acc:.4f}")
```

---

## Common Pitfalls and Debugging

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Forgetting `model.train()` / `model.eval()`** | Dropout/Batchnorm behave wrong at test | Call `eval()` before inference; `train()` before training |
| **Using `inplace=True` with checkpointing** | Errors or wrong gradients | Avoid inplace ops in checkpointed regions |
| **Wrong batch axis** | NaNs or weird loss | Ensure batch dim is 0; use `batch_first=True` in LSTM/Transformer |
| **No gradient clipping in RNNs** | Exploding gradients | `clip_grad_norm_(params, 1.0)` after backward |
| **Learning rate too high** | Loss spikes, NaNs | Use warmup; try 1e-4 for fine-tuning, 1e-3 for scratch |
| **Data not on same device as model** | RuntimeError device mismatch | `.to(device)` for both model and batch |
| **Shuffling validation/test set** | Misleading metrics | Don't shuffle val/test; use fixed seed for reproducibility |
| **BatchNorm with batch size 1** | Unstable training | Use LayerNorm or GroupNorm; or larger batch |
| **Mixed precision without GradScaler (FP16)** | Loss = NaN | Use `GradScaler` with FP16; or switch to BF16 |
| **Leaving `requires_grad=True` on frozen params** | Wasted compute, accidental updates | `param.requires_grad = False` for frozen layers |

**Debugging tips**: Log gradient norms; use `torch.autograd.detect_anomaly()` to find NaNs; profile with `torch.profiler`; check data shapes with `print(x.shape)` in the first batch.

---

## Summary Table

| Concept | Key Formula | Use Case |
|---|---|---|
| Forward pass | \( h^{(l)} = \sigma(W^{(l)}h^{(l-1)} + b^{(l)}) \) | All networks |
| Backprop | \( \delta^{(l)} = (W^{(l+1)\top}\delta^{(l+1)}) \odot \sigma'(a^{(l)}) \) | Training |
| Xavier init | \( \text{std} = \sqrt{2/(n_{in}+n_{out})} \) | tanh/sigmoid |
| He init | \( \text{std} = \sqrt{2/n_{in}} \) | ReLU |
| Batch Norm | Normalize over batch, learn γ, β | CNNs |
| Layer Norm | Normalize over features, learn γ, β | Transformers/RNN |
| Residual | \( H(x) = F(x) + x \) | ResNets, Transformers |
| LSTM cell update | \( c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \) | Sequences |
| Attention | \( \text{softmax}(QK^\top/\sqrt{d_k})V \) | Transformers |
| Dropout | zero activations with prob \( p \), scale by \( 1/(1-p) \) | Regularization |

---

## Resources

- **Deep Learning Book** (Goodfellow, Bengio, Courville): deeplearningbook.org
- **Dive into Deep Learning**: d2l.ai
- **fast.ai Practical Deep Learning**: fast.ai
- **Stanford CS231n** (CNNs): cs231n.stanford.edu
- **Stanford CS224n** (NLP): cs224n.stanford.edu
- **Papers with Code**: paperswithcode.com
- **The Annotated Transformer**: nlp.seas.harvard.edu
- **PyTorch Mixed Precision**: pytorch.org/docs/stable/amp.html
- **Gradient Accumulation**: Simulate larger batches when memory-limited
