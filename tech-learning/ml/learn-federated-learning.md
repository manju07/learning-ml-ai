# Federated Learning: Complete Guide

## Table of Contents
1. [Introduction to Federated Learning](#introduction-to-federated-learning)
2. [Core Concepts](#core-concepts)
3. [Federated Averaging (FedAvg)](#federated-averaging-fedavg)
4. [Privacy and Security](#privacy-and-security)
5. [Differential Privacy](#differential-privacy)
6. [Secure Aggregation](#secure-aggregation)
7. [Handling Non-IID Data](#handling-non-iid-data)
8. [Communication Efficiency](#communication-efficiency)
9. [Federated Learning Frameworks](#federated-learning-frameworks)
10. [Practical Examples](#practical-examples)
11. [Advanced Topics](#advanced-topics)
12. [Best Practices](#best-practices)

---

## Introduction to Federated Learning

Federated Learning (FL) enables **training machine learning models across decentralized data** without centralizing the data. Data stays on-device (phones, hospitals, institutions); only model updates are shared.

### Why Federated Learning?

| Challenge | Traditional ML | Federated Learning |
|-----------|----------------|-------------------|
| **Data location** | Central server holds all data | Data stays at source |
| **Privacy** | Raw data transmitted | Only gradients/updates shared |
| **Regulations** | GDPR, HIPAA concerns | Compliant by design |
| **Data silos** | Requires data consolidation | Works with distributed data |

### Real-World Applications

- **Mobile keyboard** (Google Gboard): Learn typing patterns without sending keystrokes
- **Healthcare**: Train on patient data across hospitals without sharing records
- **Financial services**: Collaborative fraud detection across banks
- **IoT devices**: Smart home models without cloud data
- **Autonomous vehicles**: Learn from fleet without centralizing driving data

### Key Terminology

- **Server**: Coordinates training, aggregates updates
- **Client**: Holds local data, computes updates
- **Round**: One cycle of client updates + server aggregation
- **Non-IID**: Data distribution differs across clients (common in FL)

---

## Core Concepts

### Federated Learning Flow

```
1. Server broadcasts global model M_t to clients
2. Each client k trains on local data D_k, produces update Δ_k
3. Clients send updates to server (not raw data)
4. Server aggregates: M_{t+1} = Aggregate({Δ_1, ..., Δ_K})
5. Repeat until convergence
```

### Formal Setup

- **K clients** with local datasets D_1, ..., D_K
- **Goal**: Minimize F(w) = Σ_{k=1}^K (n_k/n) · F_k(w)
- **F_k(w)**: Local objective on client k (e.g., cross-entropy loss)
- **n_k**: Size of D_k, n = Σ n_k

```python
# Pseudocode: Federated Learning
def federated_learning(server_model, clients_data, num_rounds):
    for round in range(num_rounds):
        client_updates = []
        client_weights = []
        
        for client_id, local_data in clients_data.items():
            # 1. Client downloads global model
            local_model = copy(server_model)
            
            # 2. Train on local data
            for batch in local_data:
                loss = compute_loss(local_model, batch)
                local_model.backward(loss)
            
            # 3. Compute update (delta from global)
            delta = local_model.params - server_model.params
            client_updates.append(delta)
            client_weights.append(len(local_data))
        
        # 4. Server aggregates (weighted average)
        server_model.params = server_model.params + weighted_average(client_updates, client_weights)
```

### Client Selection

Not all clients participate each round (availability, battery, bandwidth):

```python
import random

def select_clients(all_clients, fraction=0.1, min_clients=10):
    """Select subset of clients for this round"""
    num_selected = max(min_clients, int(len(all_clients) * fraction))
    return random.sample(all_clients, num_selected)

# Stratified sampling: ensure diversity
def stratified_client_selection(clients_by_type, samples_per_type=5):
    selected = []
    for client_type, client_list in clients_by_type.items():
        selected.extend(random.sample(client_list, min(samples_per_type, len(client_list))))
    return selected
```

---

## Federated Averaging (FedAvg)

FedAvg (McMahan et al., 2017) is the foundational FL algorithm.

### Algorithm

1. **Initialize**: Server has w_0
2. **Each round t**:
   - Server sends w_t to clients S_t
   - Each client k ∈ S_t: w_{t+1}^k = w_t - η · ∇F_k(w_t) (local SGD)
   - Server aggregates: w_{t+1} = Σ_k (n_k/n_t) · w_{t+1}^k

### Implementation

```python
import torch
import torch.nn as nn
import copy

def fedavg_aggregate(server_weights, client_weights_list, client_sizes):
    """
    Weighted average of client model weights.
    server_weights: current global model state dict
    client_weights_list: list of client model state dicts
    client_sizes: list of data sizes per client
    """
    total_size = sum(client_sizes)
    aggregated = {}
    
    for key in server_weights.keys():
        aggregated[key] = torch.zeros_like(server_weights[key])
        for client_weights, size in zip(client_weights_list, client_sizes):
            aggregated[key] += client_weights[key] * (size / total_size)
    
    return aggregated

# Client-side training
def client_update(model, train_loader, optimizer, num_epochs=1):
    """Single client local training"""
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = nn.functional.cross_entropy(output, batch_y)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

# Full FedAvg loop
def run_fedavg(server_model, client_dataloaders, num_rounds=100, lr=0.01):
    optimizer = torch.optim.SGD(server_model.parameters(), lr=lr)
    
    for round in range(num_rounds):
        client_models = []
        client_sizes = []
        
        for client_id, loader in client_dataloaders.items():
            local_model = copy.deepcopy(server_model)
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            weights = client_update(local_model, loader, local_optimizer)
            client_models.append(weights)
            client_sizes.append(len(loader.dataset))
        
        # Aggregate
        aggregated = fedavg_aggregate(
            server_model.state_dict(),
            client_models,
            client_sizes
        )
        server_model.load_state_dict(aggregated)
        
        if round % 10 == 0:
            print(f"Round {round} complete")
```

### FedProx: Handling System Heterogeneity

FedProx adds a proximal term to handle stragglers and varying client data:

**Objective**: min_w F_k(w) + (μ/2) ||w - w_t||²

The proximal term penalizes updates that drift too far from the global model.

```python
def fedprox_loss(local_model, global_weights, batch, mu=0.01):
    """FedProx loss with proximal term"""
    ce_loss = nn.functional.cross_entropy(local_model(batch_x), batch_y)
    prox_term = 0
    for (name, local_param), (_, global_param) in zip(
        local_model.named_parameters(),
        [(n, p) for n, p in global_weights.items()]
    ):
        prox_term += torch.sum((local_param - global_param) ** 2)
    return ce_loss + (mu / 2) * prox_term
```

---

## Privacy and Security

### Privacy Attacks in FL

1. **Model inversion**: Reconstruct training data from gradients
2. **Membership inference**: Determine if specific sample was in training set
3. **Property inference**: Extract demographic/property of training population

### Defenses Overview

| Defense | Privacy Guarantee | Utility Cost |
|---------|-------------------|--------------|
| Differential Privacy | Formal guarantee | Moderate |
| Secure Aggregation | Server never sees individual updates | Low |
| Homomorphic Encryption | Compute on encrypted data | High |
| Gradient clipping | Limits influence of single sample | Low |

---

## Differential Privacy

Differential Privacy (DP) provides **formal privacy guarantees**: output distribution changes little whether or not any single record is in the dataset.

### (ε, δ)-Differential Privacy

A mechanism M is (ε, δ)-DP if for any adjacent datasets D, D' (differ by one record):

P(M(D) ∈ S) ≤ e^ε · P(M(D') ∈ S) + δ

### DP-SGD for Federated Learning

Add calibrated Gaussian noise to gradients:

```python
import torch
import numpy as np

def add_dp_noise(gradients, noise_multiplier, max_grad_norm, sensitivity):
    """
    Add differential privacy noise to gradients.
    noise_multiplier: σ in Gaussian mechanism
    max_grad_norm: clip gradient norm to this (sensitivity)
    """
    # 1. Clip gradients (limit sensitivity)
    grad_norm = torch.norm(torch.stack([g.norm() for g in gradients]))
    if grad_norm > max_grad_norm:
        scale = max_grad_norm / (grad_norm + 1e-6)
        gradients = [g * scale for g in gradients]
    
    # 2. Add Gaussian noise
    noise_scale = sensitivity * noise_multiplier
    noisy_gradients = [
        g + torch.randn_like(g) * noise_scale
        for g in gradients
    ]
    return noisy_gradients

# Privacy accounting: track (ε, δ) budget
def compute_epsilon(delta, noise_multiplier, steps, batch_size, dataset_size):
    """Compute privacy budget (simplified)"""
    # Advanced: use RDP or moments accountant
    q = batch_size / dataset_size  # Sampling probability
    sigma = noise_multiplier
    return np.sqrt(2 * np.log(1.25 / delta)) * (q * np.sqrt(steps) / sigma)
```

### Opacus / TensorFlow Privacy

```python
# Using Opacus for DP training
# pip install opacus

from opacus import PrivacyEngine

model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Train as usual - DP is automatic
for epoch in range(epochs):
    for batch in train_loader:
        loss = model(batch)...
        loss.backward()
        optimizer.step()
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Privacy budget: ε = {epsilon:.2f}")
```

---

## Secure Aggregation

**Goal**: Server learns only the **aggregated** model, not individual client updates.

### Protocol (High Level)

1. Clients share pairwise secrets (key agreement)
2. Each client masks update: Δ_k + mask_k where Σ mask_k = 0
3. Server receives masked updates, sums them
4. Masks cancel: Σ(Δ_k + mask_k) = Σ Δ_k

```python
# Simplified secure aggregation (real implementations use crypto libs)
from dataclasses import dataclass
import hashlib

@dataclass
class SecureAggregationClient:
    client_id: int
    secret_keys: dict  # Shared with other clients
    
    def mask_update(self, update, all_client_ids):
        """Mask update so sum of masks = 0 across clients"""
        mask = torch.zeros_like(update)
        for other_id in all_client_ids:
            if other_id != self.client_id:
                # In real impl: use PRNG with shared key
                seed = hashlib.sha256(f"{self.client_id}-{other_id}".encode()).digest()
                torch.manual_seed(int.from_bytes(seed[:4], 'big'))
                if other_id < self.client_id:
                    mask += torch.randn_like(update)
                else:
                    mask -= torch.randn_like(update)
        return update + mask

# Server: receives masked updates, sum = aggregated (masks cancel)
def secure_aggregate(masked_updates):
    return sum(masked_updates) / len(masked_updates)
```

### Practical Tools

- **Google's Secure Aggregation**: Used in production FL
- **PySyft**: Python library for secure FL
- **TF Encrypted**: TensorFlow-based secure computation

---

## Handling Non-IID Data

**Non-IID** (Non-Identically and Independently Distributed): Client data distributions differ—e.g., hospital A has mostly cardiology, hospital B mostly oncology.

### Challenges

- **Weight divergence**: Local models drift toward local data
- **Slow convergence**: Conflicting gradients
- **Poor generalization**: Model overfits to participating clients

### Solutions

**1. FedNova**: Normalizes client updates by local steps

```python
# Clients may do different # of steps - normalize in aggregation
def fednova_aggregate(client_updates, client_steps, client_sizes):
    total_size = sum(client_sizes)
    # Normalize by effective steps
    normalized = [u / s for u, s in zip(client_updates, client_steps)]
    return sum(n * (sz/total_size) for n, sz in zip(normalized, client_sizes))
```

**2. SCAFFOLD**: Uses variance reduction with control variates

```python
# Client k maintains c_k (control variate)
# Reduces client drift
# c_k = c_k - c_global + (1/(K*eta)) * (w_global - w_k)
```

**3. FedBN**: Batch normalization layers stay local (not aggregated)

```python
# Don't aggregate BN statistics - they're data-dependent
def fedbn_aggregate(server_state, client_states, exclude_keys=['bn', 'batch_norm']):
    aggregated = copy.deepcopy(server_state)
    for key in aggregated:
        if not any(ex in key for ex in exclude_keys):
            aggregated[key] = average([c[key] for c in client_states])
    return aggregated
```

**4. Personalized FL**: Each client gets personalized model

```python
# Option A: Local fine-tuning after aggregation
# Option B: MAML-style meta-learning for fast adaptation
# Option C: Multi-task learning (shared + client-specific layers)
```

---

## Communication Efficiency

FL is often **communication-bound** (sending full model each round is expensive).

### Gradient Compression

```python
# 1. Top-k sparsification: Send only top-k gradients by magnitude
def topk_compress(gradients, k=0.01):
    flat = torch.cat([g.flatten() for g in gradients])
    k_val = max(1, int(flat.numel() * k))
    _, indices = torch.topk(flat.abs(), k_val)
    mask = torch.zeros_like(flat)
    mask[indices] = flat[indices]
    return unflatten(mask, gradients)

# 2. Quantization: Reduce precision (32-bit → 8-bit)
def quantize_ gradients(gradients, num_bits=8):
    scale = 2 ** num_bits - 1
    return [torch.round(g * scale).to(torch.int8) / scale for g in gradients]
```

### Local Steps vs Communication

- **Increase local epochs**: Fewer rounds, but may diverge
- **Adaptive aggregation**: Only aggregate when significant change
- **Async FL**: Don't wait for all clients (handle stragglers)

---

## Federated Learning Frameworks

### Flower (flwr)

```python
# pip install flwr

import flwr as fl
import torch

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
    
    def get_parameters(self, config):
        return [p.detach().numpy() for p in self.model.parameters()]
    
    def fit(self, parameters, config):
        # Load server parameters
        for p, param in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(param)
        # Train
        train(self.model, self.train_loader, epochs=1)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        for p, param in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(param)
        loss, accuracy = test(self.model, self.test_loader)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

# Server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
    )
)

# Client (run on each machine)
fl.client.start_numpy_client(server_address="localhost:8080", client=MNISTClient(model, train_loader))
```

### TensorFlow Federated (TFF)

```python
import tensorflow_federated as tff

# Define model and data
def create_keras_model():
    return tf.keras.models.Sequential([...])

def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Build iterative process
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0)
)

# Run training
state = trainer.initialize()
for round in range(100):
    state, metrics = trainer.next(state, federated_train_data)
    print(f"Round {round}, loss: {metrics['train']['loss']}")
```

---

## Practical Examples

### Complete FedAvg with PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

# Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_client_dataloaders(num_clients=10, data_per_client=500):
    """Create synthetic non-IID data: each client has 2-3 digits"""
    from torchvision import datasets
    train_data = datasets.MNIST('./data', train=True, download=True)
    
    loaders = {}
    for k in range(num_clients):
        # Assign subset of digits to client (non-IID)
        digit_subset = (k % 10, (k + 1) % 10)
        indices = [i for i in range(len(train_data)) if train_data[i][1] in digit_subset]
        indices = indices[:data_per_client]
        subset = torch.utils.data.Subset(train_data, indices)
        loaders[k] = DataLoader(subset, batch_size=32, shuffle=True)
    return loaders

def run_federated_training(num_rounds=50):
    model = SimpleMLP()
    client_loaders = create_client_dataloaders(num_clients=5)
    
    for round in range(num_rounds):
        client_weights = []
        client_sizes = []
        
        for client_id, loader in client_loaders.items():
            local_model = copy.deepcopy(model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01)
            
            local_model.train()
            for x, y in loader:
                x = x.view(x.size(0), -1)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(local_model(x), y)
                loss.backward()
                optimizer.step()
            
            client_weights.append(copy.deepcopy(local_model.state_dict()))
            client_sizes.append(len(loader.dataset))
        
        # FedAvg aggregation
        total = sum(client_sizes)
        new_state = {}
        for key in model.state_dict():
            new_state[key] = sum(
                client_weights[i][key] * (client_sizes[i] / total)
                for i in range(len(client_weights))
            )
        model.load_state_dict(new_state)
    
    return model
```

---

## Advanced Topics

### Byzantine-Robust Aggregation

Malicious clients may send arbitrary updates. Use robust aggregation:
- **Krum**: Select update closest to others
- **Trimmed mean**: Remove outliers before averaging
- **Median**: Coordinate-wise median

### Asynchronous FL

Clients update at different times. Server uses stale updates—weight by staleness.

### Vertical vs Horizontal FL

- **Horizontal**: Same feature space, different samples (hospitals with same patient fields)
- **Vertical**: Same samples, different features (bank + ecommerce on same users)
- **Federated Transfer Learning**: Different samples and features

---

## Best Practices

1. **Start with FedAvg** before trying FedProx/SCAFFOLD
2. **Tune local epochs**: 1-5 typically; more for non-IID
3. **Client sampling**: 10-30% per round often sufficient
4. **Differential privacy**: Start with high ε (e.g., 10), decrease as needed
5. **Validate on held-out clients** to test generalization
6. **Monitor divergence**: Track variance of client updates

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| FedAvg | Weighted average of client updates |
| Privacy | DP noise, Secure Aggregation |
| Non-IID | FedProx, FedBN, personalization |
| Communication | Compression, more local steps |
| Frameworks | Flower, TFF, PySyft |

**When to use FL**: Data cannot leave source (privacy, regulation, logistics).
