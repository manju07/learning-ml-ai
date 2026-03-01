# Continual Learning: Complete Guide

## Table of Contents
1. [Introduction to Continual Learning](#introduction-to-continual-learning)
2. [Catastrophic Forgetting](#catastrophic-forgetting)
3. [Regularization-Based Methods](#regularization-based-methods)
4. [Elastic Weight Consolidation (EWC)](#elastic-weight-consolidation-ewc)
5. [Memory-Aware Synapses (MAS)](#memory-aware-synapses-mas)
6. [Replay-Based Methods](#replay-based-methods)
7. [Architectural Methods](#architectural-methods)
8. [Meta-Learning for Continual Learning](#meta-learning-for-continual-learning)
9. [Task-Incremental vs Class-Incremental](#task-incremental-vs-class-incremental)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)
12. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Further Reading and References](#further-reading-and-references)

---

## Introduction to Continual Learning

**Continual (Lifelong) Learning** enables models to learn new tasks over time **without forgetting** previous ones. Unlike traditional training on fixed datasets, continual learning assumes a stream of tasks arriving sequentially—mirroring how humans and real-world systems learn: we acquire new skills (language, driving, new domains) while retaining previous knowledge.

### Why Continual Learning?

- **Data streams**: New data arrives over time (user behavior, new products, new sensor deployments). Retraining from scratch is often infeasible.
- **Privacy**: Cannot store all past data (GDPR, medical records). Continual learning allows learning from data streams without permanent retention.
- **Efficiency**: Avoid retraining from scratch—computational and energy cost of full retraining grows with every new task.
- **Adaptability**: Deploy once, keep improving. Critical for edge devices, personalized models, and production systems that evolve.

### The Core Challenge

**Catastrophic forgetting**: When learning task B, performance on task A degrades dramatically. This occurs because gradient descent tends to overwrite weights that were optimal for previous tasks when minimizing the loss on the new task—there is no explicit mechanism to preserve old solutions.

### Continual Learning Scenarios

| Scenario | Task ID at Test | Example |
|----------|-----------------|---------|
| **Task-incremental** | Known | Router knows which task; separate output heads per task |
| **Domain-incremental** | Unknown but same output space | Different input distributions (day/night, different cameras) |
| **Class-incremental** | Unknown, new classes | Add new classes over time; single head over all classes |

---

## Catastrophic Forgetting

### Why It Happens

- **Overwriting**: The gradient \(\nabla_\theta \mathcal{L}_B\) for task B points toward parameters that minimize \(\mathcal{L}_B\); these may conflict with \(\theta^*_A\) that minimized \(\mathcal{L}_A\). No explicit constraint prevents overwriting.
- **Representation shift**: Shared layers learn representations beneficial for the current task; when the task changes, the representation space drifts, invalidating task-specific heads.
- **Interference**: The optimal solution for task A and B may lie in different regions of parameter space; SGD finds a compromise that often degrades both.

### Mathematical View

Let \(\theta^*_A\) minimize \(\mathcal{L}_A(\theta)\). After training on task B:

\[
\theta_{\text{new}} = \theta^*_A - \eta \nabla_\theta \mathcal{L}_B(\theta^*_A) + \mathcal{O}(\eta^2)
\]

If \(\nabla_\theta \mathcal{L}_A\) and \(\nabla_\theta \mathcal{L}_B\) point in conflicting directions (negative inner product), updating toward B increases \(\mathcal{L}_A\)—forgetting.

### Measuring Forgetting

**Average Accuracy (ACC)**: Mean accuracy across all tasks after learning the final task.

**Forgetting (BWF—Backward Transfer)**: Per-task drop in accuracy after learning subsequent tasks:

\[
F_i = \max_{j \leq i} a_{i,j} - a_{i,T}
\]

where \(a_{i,j}\) is accuracy on task \(i\) after learning up to task \(j\), and \(T\) is the final task.

**Forward Transfer (FWT)**: Improvement on future tasks from learning past tasks.

```python
import numpy as np

def average_accuracy(all_tasks_acc: list[float]) -> float:
    """Mean accuracy across all tasks after final training."""
    return np.mean(all_tasks_acc)

def forgetting_measure(acc_matrix: np.ndarray) -> float:
    """
    Per-task forgetting: drop in accuracy for each task.
    acc_matrix[i, j] = accuracy on task i after learning through task j.
    """
    n_tasks = acc_matrix.shape[0]
    forgets = []
    for i in range(n_tasks - 1):
        best_so_far = np.max(acc_matrix[i, :i+1]) if i > 0 else acc_matrix[i, 0]
        final_acc = acc_matrix[i, -1]
        forgets.append(max(0, best_so_far - final_acc))
    return np.mean(forgets)

def backward_transfer(acc_before: np.ndarray, acc_after: np.ndarray) -> float:
    """Simple forgetting: mean drop per task."""
    return np.mean(np.maximum(0, acc_before - acc_after))
```

---

## Regularization-Based Methods

Add a **penalty** to the loss to prevent important weights from changing. The general form:

\[
\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \lambda \sum_i \Omega_i (\theta_i - \theta^*_i)^2
\]

where \(\Omega_i\) is an **importance weight** for parameter \(i\), and \(\theta^*_i\) is the optimal value after previous tasks. The key design choice: how to compute \(\Omega_i\).

### L2 Regularization (Fine-tuning)

Penalize deviation from initial weights:
\[
\mathcal{L} = \mathcal{L}_{\text{new}}(\theta) + \lambda \|\theta - \theta_0\|^2
\]

**Problem**: Treats all weights equally; doesn't identify which parameters are critical for past tasks. Low-capacity parameters may be over-regularized while high-impact ones change freely.

---

## Elastic Weight Consolidation (EWC)

**EWC** (Kirkpatrick et al., 2017) uses **Fisher Information** to measure the importance of each parameter for previous tasks. Intuition: parameters with high Fisher information have high curvature of the log-likelihood—changing them significantly affects the model's predictions, so we should constrain them.

### Idea

- **Important weights** (high Fisher): Changing them would materially change the output distribution → penalize change heavily.
- **Unimportant weights** (low Fisher): Output is relatively insensitive → allow more freedom for new tasks.

### EWC Loss

\[
\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_i)^2
\]

- \(F_i\): Diagonal of Fisher information matrix for the previous task
- \(\theta^*_i\): Optimal weights after the previous task (stored after each task)

### Fisher Information

The Fisher information matrix quantifies how much each parameter affects the model's predictive distribution:

\[
\mathcal{F}_i = \mathbb{E}_{x,y \sim p_{\text{data}}} \left[ \left( \frac{\partial \log p(y|x,\theta)}{\partial \theta_i} \right)^2 \right]
\]

For classification with softmax, we approximate using the model's predicted distribution (labels from model, not ground truth):

\[
F_i \approx \frac{1}{N} \sum_{n=1}^N \sum_{c} p(y=c|x_n,\theta) \left( \frac{\partial \log p(y=c|x_n,\theta)}{\partial \theta_i} \right)^2
\]

Or the simpler **empirical Fisher** (gradient of log-prob of *true* label squared):
\[
F_i \approx \frac{1}{N} \sum_{n=1}^N \left( \frac{\partial \log p(y_n|x_n,\theta)}{\partial \theta_i} \right)^2
\]

```python
import torch
import torch.nn as nn
import copy

def compute_fisher(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_samples: int = 1000,
) -> dict[str, torch.Tensor]:
    """
    Compute diagonal Fisher information for each parameter.
    Uses empirical Fisher: gradient of log p(y|x,θ) w.r.t. θ, squared and averaged.
    """
    model.eval()
    fisher = {
        n: torch.zeros_like(p, device=p.device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    n_seen = 0
    for x, y in dataloader:
        if n_seen >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        for i in range(x.size(0)):
            if n_seen >= num_samples:
                break
            model.zero_grad()
            logits = model(x[i : i + 1])
            log_prob = nn.functional.log_softmax(logits, dim=1)[0, y[i]]
            log_prob.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            n_seen += 1
    # Normalize by sample count
    for name in fisher:
        fisher[name] /= max(n_seen, 1)
    return fisher


def ewc_loss(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
    old_params: dict[str, torch.Tensor],
    lamb: float = 1000.0,
) -> torch.Tensor:
    """EWC penalty: penalize deviation of important (high-Fisher) parameters."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in fisher and name in old_params:
            diff = param - old_params[name]
            loss = loss + (fisher[name] * diff**2).sum()
    return lamb * loss


# ============ Training loop for task 2 ============
# 1. After task 1:  fisher_1 = compute_fisher(model, task1_loader, device)
#                  params_1 = {n: p.clone().detach() for n, p in model.named_parameters()}
# 2. For task 2:   total_loss = ce_loss + ewc_loss(model, fisher_1, params_1, lamb=1000)
```

### Online EWC

For **many tasks** (T > 2), storing separate Fisher and params per task is costly. **Online EWC** (Schwarz et al., 2018) maintains a running average:

\[
\widetilde{F}_i^{(t)} = \gamma \widetilde{F}_i^{(t-1)} + F_i^{(t)}, \quad \theta^*_i = \theta_i^{(t-1)}
\]

Only the *previous* task's optimal params are used; Fisher is accumulated. Typically \(\gamma \in [0.9, 1)\).

---

## Memory-Aware Synapses (MAS)

**MAS** (Aljundi et al., 2018) estimates parameter importance **without labels**—using the sensitivity of the model's output (e.g., squared L2 norm of representation) to each parameter. Useful when ground-truth labels for old tasks are unavailable.

### Importance Measure

\[
\Omega_i = \frac{1}{N} \sum_{n=1}^N \left\| \frac{\partial g(x_n; \theta)}{\partial \theta_i} \right\|^2
\]

where \(g(x;\theta)\) is the model output (e.g., representation before classifier). High \(\Omega_i\) means the output changes a lot when \(\theta_i\) changes → important for the current representation.

### MAS vs EWC

| | EWC | MAS |
|---|-----|-----|
| **Requires labels** | Yes | No |
| **Basis** | Fisher (log-likelihood curvature) | Output sensitivity |
| **Use case** | Classification | Any representation learning |

```python
def compute_mas_importance(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_samples: int = 1000,
    output_layer_name: str = "fc",  # Name of layer whose output we measure
) -> dict[str, torch.Tensor]:
    """Compute MAS importance: sensitivity of model output to each parameter."""
    model.eval()
    importance = {
        n: torch.zeros_like(p, device=p.device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    n_seen = 0
    for x, _ in dataloader:
        if n_seen >= num_samples:
            break
        x = x.to(device)
        # Use representation (e.g., before classifier) as g(x)
        out = model(x)  # Assume model returns representation or we use intermediate
        target = (out ** 2).sum()
        target.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach() ** 2
        model.zero_grad()
        n_seen += x.size(0)
    for name in importance:
        importance[name] /= max(n_seen, 1)
    return importance
```

---

## Replay-Based Methods

Store a **buffer** of old task examples; interleave with new task during training. Intuition: rehearsal prevents the decision boundary from drifting—the model continually "sees" old examples.

### Experience Replay

**Reservoir sampling**: When buffer is full, replace a random element with probability = capacity/N (preserves uniform distribution over the stream).

```python
import torch
import numpy as np

class ReplayBuffer:
    """Buffer for experience replay with reservoir sampling."""

    def __init__(self, capacity: int = 1000):
        self.buffer: list = []
        self.capacity = capacity
        self.seen = 0

    def add(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        """Add sample; use reservoir sampling when at capacity."""
        self.seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((x.detach(), y.detach(), task_id))
        else:
            idx = np.random.randint(0, self.seen)
            if idx < self.capacity:
                self.buffer[idx] = (x.detach(), y.detach(), task_id)

    def sample(self, batch_size: int):
        """Sample min(batch_size, len) random examples from buffer."""
        n = min(batch_size, len(self.buffer))
        if n == 0:
            return None, None
        indices = np.random.choice(len(self.buffer), n, replace=False)
        samples = [self.buffer[i] for i in indices]
        x = torch.cat([s[0] for s in samples])
        y = torch.cat([s[1] for s in samples])
        return x, y

# Training: sample from buffer + new task data; mix and train
for x_new, y_new in new_task_loader:
    x_old, y_old = buffer.sample(32)
    if x_old is not None:
        x = torch.cat([x_new, x_old])
        y = torch.cat([y_new, y_old])
    else:
        x, y = x_new, y_new
    loss = F.cross_entropy(model(x), y)
```

### Generative Replay

Train a **generator** (VAE, GAN) to produce synthetic old task data; no raw storage. Labels from frozen old classifier.

```python
# 1. After task 1: train VAE on task 1; keep classifier C_1
# 2. For task 2: x_gen = VAE.sample(); y_gen = C_1(x_gen)
#    Train on (real task 2) + (x_gen, y_gen)
# Challenge: generator quality degrades for very old tasks
```

### Gradient Episodic Memory (GEM)

Store exemplars per task; constrain new task gradient so it does not increase loss on exemplars. Project gradient onto feasible cone.

---

## Architectural Methods

### Progressive Neural Networks

- New task → new copy of network; **lateral connections** from previous task columns feed into the new column
- **No forgetting**: Separate params per task
- **Con**: Grows linearly with tasks; parameters scale as \(O(T \cdot |\theta|)\)

### PackNet

- After each task: **prune** low-magnitude weights; **freeze** pruned weights permanently
- Remaining capacity for new task
- Bounded growth: total params fixed; capacity is "reallocated" via pruning

### DER (Dynamically Expandable Representation)

- Expand network when needed; route different tasks to different subnets
- Each task gets a dedicated subnet; shared backbone for efficiency

```python
# Conceptual: task routing via masks
# task 1: params[mask_1]; task 2: params[mask_2] (+ shared)
# At inference: need task ID for routing (task-incremental)
```

---

## Meta-Learning for Continual Learning

**MAML** (Model-Agnostic Meta-Learning) and **ANML** (A Neuromodal Meta-Learner) learn **initializations** that adapt quickly to new tasks with few gradient steps. The meta-objective encourages \(\theta\) such that after a few steps on task \(\mathcal{T}\), the model performs well on \(\mathcal{T}\).

### Formulation

\[
\min_\theta \sum_{\mathcal{T}} \mathcal{L}_{\mathcal{T}}\big(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}^{\text{tr}}(\theta)\big)
\]

Inner loop: adapt on task training data. Outer loop: optimize init so that adapted params generalize on task validation data. For continual learning: new tasks are few-shot; meta-learned init enables fast adaptation without forgetting (if combined with EWC/replay).

---

## Task-Incremental vs Class-Incremental

### Task-Incremental

- At test: given task ID
- Each task can have separate head
- Easier: model can route to correct head

### Class-Incremental

- At test: no task ID; predict over all classes
- Single head; must distinguish all classes
- Harder: more interference, needs replay or strong regularization

```python
# Class-incremental: output dim grows
# Task 1: classes 0-4
# Task 2: add classes 5-9, total 10
# Need: rehearsal or fix old representations
```

---

## Practical Examples

### Example 1: EWC on MNIST Split

```python
# Split MNIST: task 1 = digits 0-4, task 2 = digits 5-9
def train_task(model, loader, optimizer, epoch=5):
    for e in range(epoch):
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Task 1
train_task(model, task1_loader, optimizer)
fisher_1 = compute_fisher(model, task1_loader, device)
params_1 = {n: p.clone().detach() for n, p in model.named_parameters()}

# Task 2 with EWC
for x, y in task2_loader:
    ce = F.cross_entropy(model(x), y)
    ewc = ewc_loss(model, fisher_1, params_1, lamb=1000)
    loss = ce + ewc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 2: Replay on CIFAR-100 (20 tasks, 5 classes each)

```python
buffer = ReplayBuffer(capacity=2000)
for task_id, (train_loader, test_loader) in enumerate(tasks):
    for x, y in train_loader:
        # Add to buffer (reservoir sampling)
        buffer.add(x, y, task_id)
        # Train with replay
        replay_batch = buffer.sample(32)
        # ... mix and train
```

### Example 3: Using Avalanche

```python
# pip install avalanche-lib
from avalanche.training import EWC, Naive
from avalanche.models import SimpleMLP
from avalanche.benchmarks import SplitMNIST

benchmark = SplitMNIST(n_experiences=5)
model = SimpleMLP(num_classes=10)
strategy = EWC(model, optimizer, ewc_lambda=0.4)
for exp in benchmark.train_stream:
    strategy.train(exp)
    strategy.eval(benchmark.test_stream)
```

---

## Best Practices

1. **Start with replay**: Often best accuracy; if storage allowed
2. **EWC**: Good when replay not possible; tune \(\lambda\) (typically 100–10000)
3. **Evaluate on all tasks**: Report average accuracy, forgetting (BWF)
4. **Buffer size**: Larger = less forgetting, more memory; 1–2 exemplars per class often sufficient
5. **Exemplar selection**: Herding, K-means for diverse replay; avoid redundant samples
6. **Architecture**: More capacity helps; expansion methods for many tasks

---

## Common Pitfalls and Troubleshooting

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **EWC λ too low** | Severe forgetting | Increase λ (e.g., 1000 → 10000); use grid search |
| **EWC λ too high** | Poor plasticity (new task does not learn) | Decrease λ |
| **Fisher on too few samples** | Unstable importance estimates | Use 500–2000 samples per task |
| **Replay buffer too small** | Still forgetting | Increase capacity; use 20–50 per class |
| **Class-imbalanced replay** | Bias toward old classes | Balanced sampling from buffer |
| **Task-order sensitivity** | Results vary with task sequence | Report mean ± std over multiple orderings |
| **Overfitting to replay** | Train loss low, val degrades | Reduce replay ratio; use stronger aug |

**Debugging**: Log loss on old-task exemplars during new-task training. If it spikes, regularization/replay is insufficient.

---

## Performance Benchmarks

Common benchmarks and approximate SOTA (as of 2023):

| Benchmark | Setup | Strong Methods | ACC |
|-----------|-------|----------------|-----|
| **Split-MNIST** (5 tasks) | 2 digits/task | ER + DER, iCaRL | ~98% |
| **Split-CIFAR100** (20 tasks) | 5 classes/task | DER++, ER-ACE | ~75% |
| **Sequential CIFAR-100** | 20 tasks | CoPE, REMIND | ~70% |
| **Streaming** (e.g., CLOC) | No task boundary | SCR, EWC-Online | varies |

**Libraries**: `avalanche-lib` (unified CL toolkit), `continuum`, `pytorch`

---

## Summary

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **EWC** | Fisher penalty | No storage | Tune λ, quadratic |
| **Replay** | Store exemplars | Strong | Storage, privacy |
| **Generative Replay** | Synthetic data | No raw storage | Generator quality |
| **Progressive** | New columns | No forgetting | Grows with tasks |
| **PackNet** | Prune & freeze | Bounded size | Pruning overhead |

**Libraries**: `avalanche-lib`, `continuum`, `pytorch`

---

## Further Reading and References

### Foundational Papers

- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS. [EWC]
- Aljundi et al. (2018). *Memory Aware Synapses: Learning what (not) to forget*. ECCV. [MAS]
- Lopez-Paz & Ranzato (2017). *Gradient Episodic Memory for Continual Learning*. NeurIPS. [GEM]
- Rusu et al. (2016). *Progressive Neural Networks*. arXiv. [Progressive Nets]

### Surveys and Books

- De Lange et al. (2021). *A continual learning survey: Defying forgetting in classification tasks*. IEEE TPAMI.
- Parisi et al. (2019). *Continual lifelong learning with neural networks: A review*. Neural Networks.

### Code and Toolkits

- [Avalanche](https://avalanche.continualai.org/) — Continual learning library
- [Continuum](https://github.com/ContinualAI/continuum) — Data loaders for CL
