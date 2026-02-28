# Continual Learning: Complete Guide

## Table of Contents
1. [Introduction to Continual Learning](#introduction-to-continual-learning)
2. [Catastrophic Forgetting](#catastrophic-forgetting)
3. [Regularization-Based Methods](#regularization-based-methods)
4. [Elastic Weight Consolidation (EWC)](#elastic-weight-consolidation-ewc)
5. [Replay-Based Methods](#replay-based-methods)
6. [Architectural Methods](#architectural-methods)
7. [Task-Incremental vs Class-Incremental](#task-incremental-vs-class-incremental)
8. [Practical Examples](#practical-examples)
9. [Best Practices](#best-practices)

---

## Introduction to Continual Learning

**Continual (Lifelong) Learning** enables models to learn new tasks over time **without forgetting** previous ones. Unlike traditional training on fixed datasets, continual learning assumes a stream of tasks.

### Why Continual Learning?

- **Data streams**: New data arrives over time (user behavior, new products)
- **Privacy**: Cannot store all past data
- **Efficiency**: Avoid retraining from scratch
- **Adaptability**: Deploy once, keep improving

### The Core Challenge

**Catastrophic forgetting**: When learning task B, performance on task A degrades dramatically.

### Continual Learning Scenarios

| Scenario | Task ID at Test | Example |
|----------|-----------------|---------|
| **Task-incremental** | Known | Router knows which task |
| **Domain-incremental** | Unknown but same output space | Different input distributions |
| **Class-incremental** | Unknown, new classes | Add new classes over time |

---

## Catastrophic Forgetting

### Why It Happens

- **Overwriting**: New task gradient overwrites weights important for old tasks
- **Representation shift**: Shared representation drifts
- **Interference**: New task conflicts with old task's optimal weights

### Measuring Forgetting

```python
def average_accuracy(all_tasks_acc_before, all_tasks_acc_after):
    """Accuracy before/after learning new tasks"""
    return np.mean(all_tasks_acc_before), np.mean(all_tasks_acc_after)

def forgetting_measure(acc_before, acc_after):
    """Per-task forgetting: drop in accuracy"""
    return np.mean(np.maximum(0, acc_before - acc_after))
```

---

## Regularization-Based Methods

Add a **penalty** to prevent important weights from changing.

### L2 Regularization (Fine-tuning)

Penalize deviation from initial weights:
L = L_new + λ ||θ - θ_0||²

Problem: Treats all weights equally; doesn't identify "important" ones.

---

## Elastic Weight Consolidation (EWC)

**EWC** (Kirkpatrick et al., 2017) uses **Fisher Information** to measure importance of each parameter for previous tasks.

### Idea

- Important weights: High Fisher → penalize change
- Unimportant weights: Low Fisher → allow change

### EWC Loss

L = L_new(θ) + (λ/2) Σ_i F_i (θ_i - θ*_i)²

- F_i: Diagonal of Fisher information matrix for task A
- θ*_i: Optimal weights after task A

### Fisher Information

F_i ≈ E[(∂log p(y|x,θ)/∂θ_i)²]

Approximated by: F_i ≈ (1/N) Σ_n (∂log p(y_n|x_n,θ)/∂θ_i)² over task A data.

```python
import torch
import torch.nn as nn
import copy

def compute_fisher(model, dataloader, device, num_samples=1000):
    """Compute diagonal Fisher for each parameter"""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    n = 0
    for x, y in dataloader:
        if n >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        for i in range(x.size(0)):
            if n >= num_samples:
                break
            model.zero_grad()
            log_probs[i, y[i]].backward(retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad ** 2
            n += 1
    for n in fisher:
        fisher[n] /= n
    return fisher

def ewc_loss(model, fisher, old_params, lamb=1000):
    """EWC penalty"""
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher and name in old_params:
            loss += (fisher[name] * (param - old_params[name]) ** 2).sum()
    return lamb * loss

# Training loop for task 2
# 1. After task 1: store fisher_1, params_1
# 2. For task 2: total_loss = ce_loss + ewc_loss(model, fisher_1, params_1)
```

### Online EWC

For many tasks: Fisher is running average; old params = previous task's params.

---

## Replay-Based Methods

Store a **buffer** of old task examples; interleave with new task during training.

### Experience Replay

```python
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity
    
    def add(self, x, y, task_id):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(np.random.randint(len(self.buffer)))
        self.buffer.append((x, y, task_id))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)))
        return [self.buffer[i] for i in indices]

# Training: sample from buffer + new task data
for x_new, y_new in new_task_loader:
    x_old, y_old = replay.sample(32)
    x = torch.cat([x_new, x_old])
    y = torch.cat([y_new, y_old])
    loss = criterion(model(x), y)
```

### Generative Replay

Train a **generator** to produce synthetic old task data; no raw storage.

```python
# Train VAE/GAN on task 1 data
# For task 2: sample from generator, label with task 1 classifier
# Train on: (real task 2 data) + (generated task 1 data, labeled)
```

### Gradient Episodic Memory (GEM)

Store exemplars; constrain new task gradient to not increase loss on exemplars.

---

## Architectural Methods

### Progressive Neural Networks

- New task → new copy of network; lateral connections from previous task columns
- No forgetting (separate params) but grows with tasks

### PackNet

- Prune after each task; freeze pruned weights
- Remaining capacity for new task

### DER (Dynamically Expandable Representation)

- Expand network when needed
- Route different tasks to different subnets

```python
# Concept: mask or routing
# task 1 uses params[0:1000]
# task 2 uses params[1000:2000] + possibly shared
```

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
2. **EWC**: Good when replay not possible; tune λ
3. **Evaluate on all tasks**: Report average accuracy, forgetting
4. **Buffer size**: Larger = less forgetting, more memory
5. **Exemplar selection**: Herding, K-means for diverse replay
6. **Architecture**: More capacity helps; expansion methods for many tasks

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
