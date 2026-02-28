# State Space Models & Mamba: Complete Guide

## Table of Contents
1. [Introduction to State Space Models](#introduction-to-state-space-models)
2. [Linear State Space Layers (S4)](#linear-state-space-layers-s4)
3. [Mamba: Selective State Spaces](#mamba-selective-state-spaces)
4. [Mamba Architecture Deep Dive](#mamba-architecture-deep-dive)
5. [Mamba vs Transformers](#mamba-vs-transformers)
6. [Mamba-2 and Hybrid Models](#mamba-2-and-hybrid-models)
7. [Mixture of Experts (MoE)](#mixture-of-experts-moe)
8. [Practical Examples](#practical-examples)
9. [Best Practices](#best-practices)

---

## Introduction to State Space Models

**State Space Models (SSMs)** are sequence models that map input sequences to output sequences through a latent state. They offer an alternative to Transformers with **linear** complexity in sequence length instead of quadratic.

### Why SSMs?

| Architecture | Complexity | Long Context | Training | Inference |
|-------------|------------|--------------|----------|-----------|
| **Transformer** | O(N²) attention | Limited (context window) | Parallel | O(N) per token |
| **RNN/LSTM** | O(N) | Vanishing gradient | Sequential | O(1) per token |
| **SSM (Mamba)** | O(N) | Excellent | Parallel (conv mode) | O(1) per token (recurrent) |

### Core SSM Equation

Continuous-time:
- h'(t) = A·h(t) + B·x(t)
- y(t) = C·h(t) + D·x(t)

Where:
- x(t): input
- h(t): hidden state (N-dimensional)
- y(t): output
- A, B, C, D: learnable matrices

### Discretization

Convert continuous to discrete for digital processing:

- h_k = Ā·h_{k-1} + B̄·x_k
- y_k = C·h_k

Where Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB. Δ = step size.

```python
import torch
import torch.nn as nn
import numpy as np

def discretize_zoh(A, B, delta):
    """Zero-order hold discretization"""
    I = torch.eye(A.shape[0], device=A.device)
    dA = torch.exp(delta * A)  # Simplified for diagonal A
    dB = (dA - I) @ torch.inverse(A) @ B if A.det() != 0 else delta * B
    return dA, dB
```

---

## Linear State Space Layers (S4)

**S4** (Structured State Spaces for Sequence Modeling, Gu et al., 2022) introduced:
- **HiPPO initialization**: A matrix that captures long-range dependencies
- **Diagonal approximation**: Efficient computation
- **Convolutional mode**: Parallel training via convolution

### HiPPO Matrix

Initializes A to optimally approximate recent history:

```python
def make_hippo(N):
    """HiPPO matrix initialization for optimal memory"""
    P = np.sqrt(1 + 2 * np.arange(N))
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i > j:
                A[i, j] = P[i] * P[j]
            elif i == j:
                A[i, j] = i + 1
    return -A
```

### Dual Mode: Convolution and Recurrence

- **Training (convolution)**: Compute kernel K = C · Ā^k · B̄ for all k, then convolve
- **Inference (recurrence)**: h_k = Ā·h_{k-1} + B̄·x_k (O(1) per step)

```python
def compute_ssm_kernel(A, B, C, L):
    """Compute convolution kernel for SSM"""
    # K[k] = C @ A^k @ B for k = 0, ..., L-1
    kernel = []
    Ak = torch.eye(A.shape[0], device=A.device)
    for _ in range(L):
        kernel.append((C @ Ak @ B).squeeze())
        Ak = Ak @ A
    return torch.stack(kernel)

# Training: y = conv1d(x, kernel)
# Inference: recurrence (no need to store full sequence)
```

### S4D (Diagonal)

When A is diagonal, everything simplifies:

```python
class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Diagonal A: complex eigenvalues for oscillation
        A_real = -0.5 * torch.ones(d_state)
        A_imag = torch.randn(d_state) * np.pi
        self.A = nn.Parameter(torch.complex(A_real, A_imag))
        self.B = nn.Parameter(torch.randn(d_state))
        self.C = nn.Parameter(torch.randn(d_state))
        self.D = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        # x: [B, L, D]
        L = x.shape[1]
        # Convolutional mode
        dA = torch.exp(self.delta * self.A)
        kernel = self.C * (dA.unsqueeze(0) ** torch.arange(L, device=x.device).unsqueeze(1)) @ self.B
        # FFT-based convolution
        y = torch.fft.ifft(torch.fft.fft(x, n=2*L, dim=1) * torch.fft.fft(kernel.unsqueeze(0), n=2*L, dim=0), dim=1)[:, :L]
        return y.real + self.D * x
```

---

## Mamba: Selective State Spaces

**Mamba** (Gu & Dao, 2023) adds **input-dependent selection** to SSMs: B, C, and Δ depend on the input. This gives the model the ability to **selectively** remember or forget information.

### Key Innovation: Selectivity

In S4, A, B, C are fixed (time-invariant). In Mamba, B, C, Δ are functions of x:

- B_t = Linear(x_t)
- C_t = Linear(x_t)
- Δ_t = softplus(Linear(x_t))

This makes the model content-aware—it can decide what to remember based on the input.

### Mamba Block

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        
        # Selection projections (input-dependent)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        
        # A parameter (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
    
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x_branch = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = torch.silu(x_branch)
        
        # Selection: input-dependent B, C, delta
        x_dbl = self.x_proj(x_branch)
        delta = torch.softplus(x_dbl[..., :1])  # Step size
        B_sel = x_dbl[..., 1:1+self.d_state]
        C_sel = x_dbl[..., 1+self.d_state:]
        
        # SSM recurrence (simplified; real uses selective scan CUDA kernel)
        A = -torch.exp(self.A_log)
        y = selective_scan(x_branch, delta, A, B_sel, C_sel, self.D)
        
        # Gate and project
        y = y * torch.silu(z)
        return self.out_proj(y)

def selective_scan(x, delta, A, B, C, D):
    """Selective scan (simplified recurrence)"""
    B_batch, L, d_inner = x.shape
    d_state = A.shape[1]
    h = torch.zeros(B_batch, d_inner, d_state, device=x.device)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t:t+1, :] * A.unsqueeze(0))  # [B, d_inner, d_state]
        dB = delta[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)
        h = dA * h + dB * x[:, t, :].unsqueeze(-1)
        y = (h * C[:, t, :].unsqueeze(1)).sum(-1)
        ys.append(y)
    y = torch.stack(ys, dim=1) + D * x
    return y
```

### Hardware-Aware Selective Scan

Real Mamba uses a **CUDA kernel** with:
- Kernel fusion (no materializing intermediate states)
- IO-aware tiling (minimize HBM reads)
- Parallel scan in shared memory

---

## Mamba Architecture Deep Dive

### Full Mamba Model

```python
class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                MambaBlock(d_model, d_state, d_conv, expand)
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)  # Residual
        x = self.norm(x)
        return self.lm_head(x)
```

### Using Pretrained Mamba

```python
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b")

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
```

---

## Mamba vs Transformers

| Feature | Transformer | Mamba |
|---------|-------------|-------|
| **Attention** | O(N²) full attention | No attention; selective scan O(N) |
| **Long sequences** | Expensive (128K+ costs grow) | Efficient (linear) |
| **Inference** | KV cache grows with context | Constant state (~16 floats/layer) |
| **Training** | Parallel via attention | Parallel via conv/scan |
| **In-context learning** | Strong | Good (selective mechanism) |
| **Retrieval** | Precise (attend to any token) | Compressed state (can miss) |

### When to Choose Mamba

- Very long sequences (DNA, audio, time series)
- Inference on long context (O(1) vs O(N) per token)
- Edge devices (smaller state than KV cache)

### When Transformers Win

- Tasks requiring precise retrieval from context
- Short-to-medium sequences (attention overhead is fine)
- Maturity of ecosystem and tooling

---

## Mamba-2 and Hybrid Models

### Mamba-2

- Connections between SSMs and structured attention
- More efficient kernel implementations
- Better parallelism

### Hybrid: Transformer + Mamba

Combine Transformer layers (for precise retrieval) with Mamba layers (for efficiency):

```python
class HybridBlock(nn.Module):
    def __init__(self, d_model, use_attention=False):
        super().__init__()
        if use_attention:
            self.block = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        else:
            self.block = MambaBlock(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.use_attention = use_attention
    
    def forward(self, x):
        if self.use_attention:
            x_norm = self.norm(x)
            attn_out, _ = self.block(x_norm, x_norm, x_norm)
            return x + attn_out
        else:
            return x + self.block(self.norm(x))

# Example: Attention every 4th layer
layers = [HybridBlock(d_model, use_attention=(i % 4 == 0)) for i in range(24)]
```

### Jamba (AI21)

- Interleaves Mamba + Transformer + MoE layers
- 256K context window
- Efficient inference

---

## Mixture of Experts (MoE)

**MoE** activates only a subset of parameters per token, achieving larger model capacity without proportional compute cost.

### MoE Architecture

Replace the FFN in each Transformer/Mamba layer with N experts + a router:

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ffn, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(), nn.Linear(d_ffn, d_model))
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
        self.top_k = top_k
    
    def forward(self, x):
        # x: [B, L, D]
        router_logits = self.router(x)  # [B, L, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += top_k_probs[:, :, k][mask].unsqueeze(-1) * expert_output
        return output
```

### Router and Load Balancing

Prevent "expert collapse" (all tokens routed to one expert):

```python
def load_balancing_loss(router_probs, expert_counts, num_experts):
    """Auxiliary loss to balance expert usage"""
    # Want: each expert gets 1/num_experts fraction
    fraction_per_expert = expert_counts.float() / expert_counts.sum()
    target = 1.0 / num_experts
    return num_experts * (fraction_per_expert * router_probs.mean(dim=(0,1))).sum()
```

### Mixtral (Mistral AI)

- 8 experts, top-2 routing per token
- 47B total params, ~13B active per token
- Matches/beats much larger dense models

### DeepSeek-MoE

- Fine-grained experts (more, smaller experts)
- Shared expert (always active) + routed experts
- Better load balancing

---

## Practical Examples

### Example 1: Mamba for Long Sequence Classification

```python
from transformers import MambaForSequenceClassification, AutoTokenizer

model = MambaForSequenceClassification.from_pretrained(
    "state-spaces/mamba-130m",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m")

# Works on very long inputs efficiently
inputs = tokenizer("Very long document...", return_tensors="pt", max_length=16384, truncation=True)
outputs = model(**inputs)
```

### Example 2: Simple SSM from Scratch

```python
class SimpleSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.A = nn.Parameter(-torch.rand(d_model, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))
        self.delta = nn.Parameter(torch.ones(d_model) * 0.1)
    
    def forward(self, x):
        B_size, L, D = x.shape
        dA = torch.exp(self.delta.unsqueeze(-1) * self.A)
        dB = self.delta.unsqueeze(-1) * self.B
        h = torch.zeros(B_size, D, self.A.shape[1], device=x.device)
        outputs = []
        for t in range(L):
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)
            y = (h * self.C).sum(-1) + self.D * x[:, t]
            outputs.append(y)
        return torch.stack(outputs, dim=1)
```

---

## Best Practices

1. **Use pretrained Mamba** for NLP; train from scratch for domain-specific long sequences
2. **Hybrid**: Combine with attention for tasks needing precise retrieval
3. **MoE**: Use load-balancing loss; monitor expert utilization
4. **Long context**: Mamba excels at 16K+ sequences
5. **Inference**: Mamba constant memory; no KV cache needed

---

## Summary

| Model | Key Idea | Complexity | Best For |
|-------|----------|------------|----------|
| S4 | Structured SSM, HiPPO | O(N) | Long-range dependencies |
| Mamba | Selective SSM | O(N) | LLMs, long context |
| MoE | Sparse experts | O(N·k/E) active | Large capacity, efficient |
| Hybrid | Attention + Mamba | Mixed | Best of both |
| Jamba | Mamba + Transformer + MoE | O(N) mostly | Production LLMs |

**Libraries**: `mamba-ssm`, `transformers`, `causal-conv1d`
