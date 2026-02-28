# Neural Architecture Search (NAS): Complete Guide

## Table of Contents
1. [Introduction to NAS](#introduction-to-nas)
2. [Search Space Design](#search-space-design)
3. [Search Strategies](#search-strategies)
4. [DARTS: Differentiable Architecture Search](#darts-differentiable-architecture-search)
5. [ENAS and Weight Sharing](#enas-and-weight-sharing)
6. [EfficientNet and Compound Scaling](#efficientnet-and-compound-scaling)
7. [NAS for Transformers](#nas-for-transformers)
8. [AutoML and HPO vs NAS](#automl-and-hpo-vs-nas)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)

---

## Introduction to NAS

**Neural Architecture Search (NAS)** automates the design of neural network architectures. Instead of hand-designing layers and connections, NAS searches over a space of architectures to find high-performing ones.

### Why NAS?

- **Manual design** is time-consuming and requires expertise
- **Optimal architecture** depends on dataset, task, and hardware
- **NAS** can discover novel architectures (e.g., EfficientNet, NASNet)

### NAS Components

1. **Search space**: Set of possible architectures
2. **Search strategy**: How to explore the space (RL, evolution, gradient)
3. **Performance estimation**: How to evaluate (train, proxy, predictor)

### Cost Challenge

Training each candidate is expensive. Key insight: **weight sharing** (one supernet, many subnets) and **differentiable search** reduce cost.

---

## Search Space Design

### Cell-Based Search (NASNet, DARTS)

Search for a **cell** (repeated block); stack cells to form full network.

```python
# Cell: Directed graph of nodes (feature maps) and edges (operations)
# Node i = aggregation of outputs from nodes j < i
# Edge (i, j): apply one of {conv 3x3, conv 5x5, sep_conv, max_pool, ...}

OPS = [
    'none',
    'skip_connect',
    'conv_3x3',
    'conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'max_pool_3x3',
    'avg_pool_3x3'
]
```

### Macro Search

Search full architecture: number of layers, filter sizes, etc. Larger space, harder search.

### Hierarchical Search (NAS-Bench-301)

Search at multiple levels: outer (blocks), inner (cells).

---

## Search Strategies

### 1. Random Search

Baseline: sample architectures, train, pick best.

### 2. Reinforcement Learning (NAS-Net, ENAS)

- **Controller RNN**: Outputs architecture as sequence
- **Reward**: Validation accuracy
- **Policy gradient** to update controller

### 3. Evolution (AmoebaNet)

- Population of architectures
- Mutate/crossover, select by fitness
- Train offspring, replace worst

### 4. Bayesian Optimization

- Surrogate model (GP, neural predictor)
- Acquire next architecture via acquisition function (EI, UCB)

### 5. Differentiable (DARTS)

- Relax architecture to continuous
- Optimize with gradient descent
- Derive discrete architecture at end

---

## DARTS: Differentiable Architecture Search

**DARTS** (Liu et al., 2019) makes the architecture **differentiable** by mixing operations with softmax weights.

### Mixed Operation

For each edge (i, j), instead of one op:
ō^(i,j)(x) = Σ_k (exp(α_k) / Σ_l exp(α_l)) · o_k(x)

α_k are **architecture parameters** (learned).

### Bilevel Optimization

- **Inner**: Train weights w (on train set)
- **Outer**: Update α (on validation set)

∇_α L_val(w*(α), α) ≈ ∇_α L_val(w - ξ∇_w L_train, α)

### DARTS Implementation Sketch

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class MixedOp(nn.Module):
    """Mixed operation: weighted sum of candidate ops"""
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'max_pool_3x3']:
            op = OPS[primitive](C_in, C_out, stride)
            self._ops.append(op)
    
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class DARTSCell(nn.Module):
    def __init__(self, steps, C_prev, C, C_prev_prev=None):
        super().__init__()
        self.steps = steps  # Number of intermediate nodes
        self._ops = nn.ModuleList()
        for i in range(steps):
            for j in range(i + 2):  # Connect from prev + 2 inputs
                stride = 1 if j < 2 else 2
                op = MixedOp(C if j < 2 else C, C, stride)
                self._ops.append(op)
    
    def forward(self, s0, s1, weights):
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) 
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.steps:], dim=1)

# Architecture parameters α (one per op per edge)
# num_edges = sum(i+2 for i in range(steps))
# num_ops = 5
# alpha = nn.Parameter(torch.randn(num_edges, num_ops) * 1e-3)
# weights = F.softmax(alpha, dim=-1)
```

### Discretization

After search: for each edge, pick op with highest α.

```python
def discretize(alpha):
    return alpha.argmax(dim=-1)
```

### DARTS Caveats

- **Overfitting to search**: α may overfit validation
- **Collapse**: Often selects skip_connect (easy)
- **Second-order** approx can be unstable → use first-order

---

## ENAS and Weight Sharing

**ENAS** (Efficient NAS): One **shared** set of weights; different architectures are subgraphs. Train by sampling architectures and updating shared weights.

### Weight Sharing

- **Supernet**: Graph with all possible ops
- **Subnet**: Mask to select one op per edge
- Train supernet; subnets inherit weights
- Reduces search cost from O(N × train) to O(1 × train)

### ENAS Controller

```python
# Controller outputs: for each edge, which op; for each node, which prev node to connect
# RL: reward = validation accuracy of sampled arch
# Policy gradient to update controller
```

---

## EfficientNet and Compound Scaling

**EfficientNet** uses NAS to find baseline, then **compound scaling**:

- **Depth** (d): More layers
- **Width** (w): More channels
- **Resolution** (r): Larger input

Constrain: d^α · w^β · r^γ = 2 with α+β+γ=1.

### Scaling Formula

```python
def compound_scale(base_config, phi):
    """phi = 0: baseline; phi=1,2,...: scaled"""
    alpha, beta, gamma = 1.2, 1.1, 1.15
    d = base_config['depth'] * (alpha ** phi)
    w = base_config['width'] * (beta ** phi)
    r = base_config['resolution'] * (gamma ** phi)
    return int(d), int(w), int(r)
```

---

## NAS for Transformers

### Search Dimensions

- **Depth**: Number of layers
- **Width**: Hidden size, FFN size
- **Heads**: Number of attention heads
- **Kernel size**: Local attention
- **Activation**: GELU, ReLU, Swish

### AutoFormer, NAS-BERT

- Search for transformer blocks
- Weight sharing across subnets
- Search on proxy task (e.g., MLM on small data)

```python
# Example: search attention dim
class SearchableAttention(nn.Module):
    def __init__(self, max_dims):
        self.dims = list(range(64, max_dims+1, 64))
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(d, num_heads=8) for d in self.dims
        ])
    
    def forward(self, x, arch_param):
        idx = arch_param.argmax()
        return self.attentions[idx](x, x, x)[0]
```

---

## AutoML and HPO vs NAS

| | HPO | NAS |
|---|-----|-----|
| **Search** | Hyperparameters (lr, layers) | Architecture (ops, connections) |
| **Space** | Continuous/categorical | Combinatorial, graph |
| **Tools** | Optuna, Ray Tune | NNI, AutoKeras, DARTS |
| **Overlap** | Joint search (NAS + HPO) | |

---

## Practical Examples

### Example 1: DARTS with torch-darts

```python
# pip install torch
# DARTS implementation: github.com/quark0/darts

# Simplified: use Optuna for architecture search
import optuna

def create_model(trial):
    n_layers = trial.suggest_int("n_layers", 2, 8)
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    return MyModel(n_layers=n_layers, hidden=hidden)

def objective(trial):
    model = create_model(trial)
    train(model, train_loader)
    return evaluate(model, val_loader)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best = study.best_params
```

### Example 2: AutoKeras

```python
# pip install autokeras
import autokeras as ak

clf = ak.ImageClassifier(overwrite=True, max_trials=10)
clf.fit(x_train, y_train, epochs=10)
model = clf.export_model()
```

### Example 3: NAS with NNI

```python
# Microsoft NNI
# nni.create_experiment(config)
# Define search space in YAML
# search_space:
#   n_layers: {_type: randint, _value: [2, 8]}
#   hidden: {_type: choice, _value: [64, 128, 256]}
```

---

## Best Practices

1. **Start small**: Search on subset of data
2. **Use proxy**: Shorter training, smaller model for search
3. **Weight sharing**: When possible (ENAS, DARTS)
4. **Validate**: Retrain best arch from scratch
5. **Hardware**: Consider latency, FLOPs in search objective
6. **DARTS**: Watch for skip-connect collapse; use regularization

---

## Summary

| Method | Key Idea | Cost |
|--------|----------|------|
| RL (ENAS) | Controller + weight sharing | Medium |
| Evolution | Population + mutate | High |
| DARTS | Differentiable, gradient | Low |
| EfficientNet | NAS + compound scaling | - |
| AutoKeras | Keras-integrated NAS | Medium |

**Libraries**: `torch`, `nni`, `autokeras`, `optuna`
