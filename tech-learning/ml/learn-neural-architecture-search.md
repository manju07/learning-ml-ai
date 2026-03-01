# Neural Architecture Search (NAS): Complete Guide

## Table of Contents
1. [Introduction to NAS](#introduction-to-nas)
2. [Search Space Design](#search-space-design)
3. [Search Strategies](#search-strategies)
4. [DARTS: Differentiable Architecture Search](#darts-differentiable-architecture-search)
5. [PC-DARTS and NAS-Bench](#pc-darts-and-nas-bench)
6. [ENAS and Weight Sharing](#enas-and-weight-sharing)
7. [EfficientNet and Compound Scaling](#efficientnet-and-compound-scaling)
8. [NAS for Transformers](#nas-for-transformers)
9. [AutoML and HPO vs NAS](#automl-and-hpo-vs-nas)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)
12. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Further Reading and References](#further-reading-and-references)

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

Search for a **cell** (repeated block); stack cells to form the full network. A cell is a directed acyclic graph (DAG): nodes are feature maps, edges are operations. Node \(i\) aggregates outputs from nodes \(j < i\). The cell is repeated (with possible stride) to build the full net.

**Operations** (per edge): `none`, `skip_connect`, `conv_3x3`, `conv_5x5`, `sep_conv_3x3`, `sep_conv_5x5`, `dil_conv_3x3`, `dil_conv_5x5`, `max_pool_3x3`, `avg_pool_3x3`

**Search space size**: For \(N\) edges and \(K\) ops per edge: \(K^N\) architectures (e.g., \(8^{14} \approx 4 \times 10^{12}\) for DARTS).

### Macro Search

Search full architecture: number of layers, filter sizes, kernel sizes. Larger space, harder search—often combined with evolution or RL.

### Hierarchical Search (NAS-Bench-101/201/301)

- **NAS-Bench-101**: Small cell space; full training results precomputed → instant lookup
- **NAS-Bench-201**: CIFAR-100; 15,625 architectures; used for predictor/weight-sharing research
- **NAS-Bench-301**: NAS-Bench-201 extended via learned surrogate (e.g., DARTS); scales to larger spaces

---

## Search Strategies

### 1. Random Search

Baseline: sample architectures from \(\mathcal{A}\), train each, pick best. Surprisingly strong for moderate search spaces; serves as lower bound.

### 2. Reinforcement Learning (NAS-Net, ENAS)

- **Controller**: RNN outputs architecture as a sequence (e.g., op per edge)
- **Reward**: \(R = \text{val\_accuracy}\) (or validation loss)
- **Policy gradient** (REINFORCE): \(\nabla_\phi \mathbb{E}_{a \sim \pi_\phi}[R(a)] \approx R(a) \nabla_\phi \log \pi_\phi(a)\)
- **Credit assignment**: Reward shaping (e.g., early stopping reward) improves sample efficiency

### 3. Evolution (AmoebaNet, Regularized Evolution)

- Population of \(P\) architectures; fitness = validation accuracy
- **Mutate**: Change one op or connection
- **Crossover**: Combine two parents (e.g., mix cells)
- **Selection**: Keep top-k; replace worst with offspring
- Regularized evolution: age regularization favors younger architectures to avoid local optima

### 4. Bayesian Optimization

- **Surrogate**: GP or neural predictor \(f(a) \approx R(a)\) trained on \((a_i, R_i)\)
- **Acquisition**: EI (Expected Improvement), UCB; trade off exploration vs exploitation
- **Limitation**: Doesn't scale to very large discrete spaces without embeddings

### 5. Differentiable (DARTS)

- **Relax**: \(o^{(i,j)}(x) = \sum_k \frac{\exp(\alpha_k)}{\sum_l \exp(\alpha_l)} o_k(x)\)—soft selection over ops
- **Joint optimization**: \(\alpha\) (architecture) and \(w\) (weights) via bilevel or alternating updates
- **Discretize**: At the end, pick \(o = \arg\max_k \alpha_k\) per edge

---

## DARTS: Differentiable Architecture Search

**DARTS** (Liu et al., 2019) makes the architecture **differentiable** by replacing the discrete choice "pick one op" with a **soft mixture** of all ops, weighted by learned parameters \(\alpha\).

### Mixed Operation

For each edge \((i, j)\), instead of selecting a single operation \(o_k\), use:

\[
\bar{o}^{(i,j)}(x) = \sum_{k=1}^{K} \frac{\exp(\alpha_k^{(i,j)})}{\sum_{l=1}^{K} \exp(\alpha_l^{(i,j)})} \cdot o_k(x)
\]

where \(\alpha^{(i,j)} \in \mathbb{R}^K\) are **architecture parameters** (learned). At search end, discretize: \(o^* = \arg\max_k \alpha_k^{(i,j)}\).

### Bilevel Optimization

DARTS formulates a **bilevel** problem:
- **Inner**: \(w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)\)
- **Outer**: \(\min_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha)\)

Gradient w.r.t. \(\alpha\) requires \(\nabla_\alpha w^*\), which is expensive. **First-order approximation**:

\[
\nabla_\alpha \mathcal{L}_{\text{val}} \approx \nabla_\alpha \mathcal{L}_{\text{val}}(w - \xi \nabla_w \mathcal{L}_{\text{train}}, \alpha)
\]

i.e., assume \(w\) is one gradient step from \(w^*\); no second-order terms. **Second-order** uses implicit differentiation; more accurate but slower and sometimes unstable.

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

**EfficientNet** (Tan & Le, 2019) first uses NAS to find a strong **baseline** (EfficientNet-B0), then applies **compound scaling** to scale depth, width, and resolution together.

### Scaling Dimensions

- **Depth** \(d\): More layers (e.g., more MBConv blocks)
- **Width** \(w\): More channels per layer
- **Resolution** \(r\): Larger input (e.g., 224 → 336)

**Constraint**: \(d^\alpha \cdot w^\beta \cdot r^\gamma = 2\) with \(\alpha + \beta + \gamma = 1\). Grid search on small models found \(\alpha \approx 1.2\), \(\beta \approx 1.1\), \(\gamma \approx 1.15\).

### Scaling Formula

\[
d = d_0 \cdot \phi^\alpha, \quad w = w_0 \cdot \phi^\beta, \quad r = r_0 \cdot \phi^\gamma
\]

 where \(\phi\) is the compound coefficient (e.g., 1, 2, 3 for B0, B1, B2...).

```python
def compound_scale(base_config: dict, phi: float) -> tuple[int, int, int]:
    """
    phi=0: baseline (B0); phi=1,2,3: B1, B2, B3...
    EfficientNet-B0: depth=1.0, width=1.0, resolution=224
    """
    alpha, beta, gamma = 1.2, 1.1, 1.15
    d = int(base_config['depth'] * (alpha ** phi))
    w = int(base_config['width'] * (beta ** phi))
    r = int(base_config['resolution'] * (gamma ** phi))
    return d, w, r
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

1. **Start small**: Search on subset of data (e.g., CIFAR-10) or fewer epochs
2. **Use proxy**: Shorter training (e.g., 50 epochs), smaller model; correlation with full training is often good
3. **Weight sharing**: When possible (ENAS, DARTS)—dramatically reduces cost
4. **Validate**: Retrain best architecture from scratch; search may overfit proxy
5. **Hardware-aware**: Include latency, FLOPs, or energy in search objective (e.g., ProxylessNAS)
6. **DARTS**: Watch for skip-connect collapse; use regularization (e.g., DropEdge) or PC-DARTS

---

## Common Pitfalls and Troubleshooting

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **DARTS skip collapse** | Most \(\alpha\) mass on skip_connect | L2 on \(\alpha\); limit skip; use PC-DARTS |
| **Search overfitting** | Good val acc during search, poor retrain | Early stopping; more data; regularize \(\alpha\) |
| **Memory OOM** | Search fails on GPU | PC-DARTS; reduce batch size; gradient checkpointing |
| **Unstable bilevel** | Loss spikes, NaN | Use first-order approx; smaller \(\xi\); warmup |
| **Transfer gap** | CIFAR arch poor on ImageNet | Search on target dataset or larger proxy |
| **Wrong metric** | Best arch has high latency | Add latency term to objective |

---

## Performance Benchmarks

| Method | CIFAR-10 | ImageNet Top-1 | Search Cost (GPU-days) |
|--------|----------|----------------|------------------------|
| **DARTS (2nd)** | 97.2% | 73.3% | ~1 |
| **PC-DARTS** | 97.4% | 74.0% | ~0.5 |
| **EfficientNet-B0** | - | 77.1% | - |
| **NASNet** | 97.4% | 74.0% | 2000 |
| **ENAS** | 97.1% | - | 0.5 |

*Approximate; see papers for exact setup.*

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

---

## Further Reading and References

### Foundational Papers

- Zoph & Le (2017). *Neural Architecture Search with Reinforcement Learning*. ICLR. [NAS-Net]
- Pham et al. (2018). *Efficient Neural Architecture Search via Parameter Sharing*. ICML. [ENAS]
- Liu et al. (2019). *DARTS: Differentiable Architecture Search*. ICLR.
- Xu et al. (2020). *PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search*. ICLR.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.

### NAS-Bench

- Ying et al. (2019). *NAS-Bench-101*. ICLR.
- Dong et al. (2020). *NAS-Bench-201*. ICLR.
- Siems et al. (2020). *NAS-Bench-301*. NeurIPS.

### Surveys

- Elsken et al. (2019). *Neural Architecture Search: A Survey*. JMLR.
