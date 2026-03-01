# PyTorch: Comprehensive Deep Learning Guide

## Table of Contents
1. [Introduction and Installation](#1-introduction-and-installation)
2. [Tensors: Creation, dtype, Device, Memory Layout](#2-tensors-creation-dtype-device-memory-layout)
3. [Tensor Operations: Indexing, Slicing, Broadcasting, In-place](#3-tensor-operations-indexing-slicing-broadcasting-in-place)
4. [Autograd: Computational Graphs and Automatic Differentiation](#4-autograd-computational-graphs-and-automatic-differentiation)
5. [Custom Autograd Functions](#5-custom-autograd-functions)
6. [nn.Module: Defining Models, Parameters, Buffers, Hooks](#6-nnmodule-defining-models-parameters-buffers-hooks)
7. [Common Layers: Linear, Conv, BN, LN, Dropout, Embedding, LSTM, Transformer](#7-common-layers)
8. [Loss Functions](#8-loss-functions)
9. [Optimizers and Learning Rate Schedulers](#9-optimizers-and-learning-rate-schedulers)
10. [Dataset and DataLoader](#10-dataset-and-dataloader)
11. [Complete Training Loop with Validation and Checkpointing](#11-complete-training-loop)
12. [GPU Training: Mixed Precision, DataParallel, DDP](#12-gpu-training)
13. [torch.compile, functorch, and TorchScript](#13-torchcompile-functorch-and-torchscript)
14. [Model Saving and Loading](#14-model-saving-and-loading)
15. [Hooks for Feature Extraction and Gradient Inspection](#15-hooks)
16. [PyTorch Lightning](#16-pytorch-lightning)
17. [Profiling with torch.profiler](#17-profiling)
18. [Full Examples: MLP, CNN, RNN, Transformer](#18-full-examples)
19. [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)
20. [Production Deployment Notes](#production-deployment-notes)

---

## 1. Introduction and Installation

PyTorch is a dynamic deep learning framework developed by Meta AI Research. Its **define-by-run** (dynamic computational graph) philosophy makes debugging natural, as the graph is constructed at runtime using standard Python control flow.

**Core design principles:**
- Dynamic graphs (eager execution by default)
- NumPy-like tensor API with GPU support
- First-class autograd engine
- Production paths: TorchScript, `torch.compile`, TorchServe

```bash
# CPU-only
pip install torch torchvision torchaudio

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 2. Tensors: Creation, dtype, Device, Memory Layout

### 2.1 Tensor Creation

A `torch.Tensor` is a multi-dimensional array. Unlike NumPy, tensors can reside on GPU and participate in autograd.

```python
import torch

# ---- From Python data ----
t_list  = torch.tensor([1, 2, 3])                        # infer dtype (int64)
t_float = torch.tensor([1.0, 2.0, 3.0])                  # float32
t_2d    = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# ---- Factory functions ----
zeros    = torch.zeros(3, 4)                              # shape (3,4), filled 0
ones     = torch.ones(3, 4)
full     = torch.full((3, 4), fill_value=7.0)
eye      = torch.eye(4)                                   # identity matrix
empty    = torch.empty(3, 4)                              # uninitialized memory

# ---- Random tensors ----
randn    = torch.randn(3, 4)                              # N(0,1)
rand     = torch.rand(3, 4)                               # Uniform[0,1)
randint  = torch.randint(0, 10, (3, 4))                   # integers [0,10)
bernoulli= torch.bernoulli(torch.full((3, 4), 0.5))       # binary 0/1

# ---- Range-based ----
arange   = torch.arange(0, 10, step=2)                   # [0,2,4,6,8]
linspace = torch.linspace(0, 1, steps=11)                 # 11 points in [0,1]

# ---- From NumPy (zero-copy shared memory) ----
import numpy as np
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t   = torch.from_numpy(arr)   # shares memory
arr[0] = 99.0
print(t[0])   # 99.0 — same memory

# Safe copy
t_copy = torch.tensor(arr)  # copies data
```

### 2.2 Data Types (dtype)

```python
# All dtypes
dtypes = {
    'float16':  torch.float16,   # half precision, 2 bytes
    'bfloat16': torch.bfloat16,  # brain float16, 2 bytes (better range than float16)
    'float32':  torch.float32,   # single precision, 4 bytes (default for float)
    'float64':  torch.float64,   # double precision, 8 bytes
    'int8':     torch.int8,
    'int16':    torch.int16,
    'int32':    torch.int32,
    'int64':    torch.int64,     # default for integer tensor
    'bool':     torch.bool,
    'complex64':torch.complex64,
}

x = torch.randn(3, 4)
print(x.dtype)            # torch.float32

# Type casting
x_half = x.half()         # → float16
x_bf16 = x.to(torch.bfloat16)
x_int  = x.int()          # truncates to int32
x_f64  = x.double()       # → float64

# Check type
print(x.is_floating_point())   # True
print(x.is_complex())          # False
```

### 2.3 Devices: CPU and GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating tensors directly on device
t_gpu = torch.randn(3, 4, device=device)
t_gpu2 = torch.zeros(3, 4, device='cuda:0')  # specific GPU

# Moving tensors
t_cpu = torch.randn(3, 4)
t_gpu = t_cpu.to(device)         # non-blocking by default
t_gpu = t_cpu.cuda()             # shorthand
t_cpu2 = t_gpu.cpu()             # back to CPU

# Non-blocking transfer (overlaps with computation)
t_gpu = t_cpu.to(device, non_blocking=True)

# Pin memory for fast CPU→GPU transfer
pinned = torch.zeros(1000, 1000).pin_memory()
gpu_t  = pinned.to(device, non_blocking=True)

# Multi-GPU: access specific GPU
t_gpu1 = torch.randn(3, 4, device='cuda:1')

# Querying device info
print(torch.cuda.current_device())
print(torch.cuda.get_device_properties(0))
print(torch.cuda.memory_allocated(0) / 1e9, 'GB')  # current allocation
print(torch.cuda.max_memory_allocated(0) / 1e9, 'GB')

# Context manager to set default GPU
with torch.cuda.device(1):
    t = torch.randn(3, 4)  # on cuda:1
```

### 2.4 Tensor Attributes and Memory Layout

```python
x = torch.randn(3, 4)

# Key attributes
print(x.shape)          # torch.Size([3, 4])
print(x.size())         # same as shape
print(x.ndim)           # 2
print(x.dtype)          # torch.float32
print(x.device)         # device(type='cpu')
print(x.requires_grad)  # False
print(x.is_contiguous())# True — C-contiguous layout

# Strides: bytes between elements along each dim
print(x.stride())       # (4, 1) — step 4 to next row, step 1 to next col

# Non-contiguous tensors (e.g., after transpose)
xt = x.t()
print(xt.is_contiguous())  # False
print(xt.stride())          # (1, 4) — transposed strides

# Make contiguous
xt_c = xt.contiguous()
print(xt_c.is_contiguous()) # True

# Underlying storage
print(x.storage())
print(x.data_ptr())  # memory address
```

---

## 3. Tensor Operations: Indexing, Slicing, Broadcasting, In-place

### 3.1 Indexing and Slicing

```python
x = torch.arange(24).reshape(2, 3, 4).float()

# Basic indexing
print(x[0])          # first matrix (3×4)
print(x[0, 1])       # second row of first matrix (length 4)
print(x[0, 1, 2])    # scalar element

# Slicing (like NumPy)
print(x[:, :, 1:3])  # columns 1 and 2 of all matrices
print(x[1, ::2])     # every other row of second matrix
print(x[..., -1])    # last column via Ellipsis

# Boolean indexing
mask = x > 10
print(x[mask])       # 1D tensor of elements > 10
x[mask] = 0          # in-place assignment

# Fancy/advanced indexing
idx = torch.tensor([0, 2])
print(x[:, idx])     # select columns 0 and 2

# torch.gather: advanced selection
# Picks from x along dim=1 using index tensor
values = torch.tensor([[10.0, 20.0, 30.0],
                        [40.0, 50.0, 60.0]])
idx    = torch.tensor([[2, 0], [1, 2]])
out    = torch.gather(values, dim=1, index=idx)
# out[i][j] = values[i][idx[i][j]]
print(out)

# torch.index_select
print(torch.index_select(values, dim=0, index=torch.tensor([1])))

# Scatter: inverse of gather
src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
out = torch.zeros(2, 3)
out.scatter_(1, idx, src)  # in-place scatter
```

### 3.2 Reshaping and Viewing

```python
x = torch.arange(12)

# view: shares memory (must be contiguous)
y = x.view(3, 4)
y = x.view(3, -1)      # infer last dim

# reshape: copies if needed
z = x.reshape(3, 4)

# squeeze / unsqueeze
a = torch.randn(1, 3, 1, 4)
print(a.squeeze().shape)         # (3, 4) — remove all size-1 dims
print(a.squeeze(0).shape)        # (3, 1, 4)
print(a.unsqueeze(2).shape)      # (1, 3, 1, 1, 4)

# flatten
b = torch.randn(2, 3, 4)
print(b.flatten().shape)          # (24,)
print(b.flatten(1).shape)         # (2, 12) — flatten from dim 1 onward

# permute: reorder dimensions
c = torch.randn(2, 3, 4)
print(c.permute(0, 2, 1).shape)   # (2, 4, 3)

# transpose (2D shortcut)
print(c.transpose(1, 2).shape)    # (2, 4, 3)

# contiguous after permute
d = c.permute(0, 2, 1).contiguous()
```

### 3.3 Broadcasting

Broadcasting follows NumPy rules: dimensions are compared right-to-left; each dim must be equal, 1, or missing.

```python
a = torch.ones(5, 3)    # (5, 3)
b = torch.ones(3)       # (3,) → broadcast to (5, 3)
c = a + b               # fine

# Explicit broadcasting
a = torch.randn(4, 1, 3)
b = torch.randn(   5, 3)
print((a + b).shape)    # (4, 5, 3)

# torch.broadcast_to
x = torch.tensor([1.0, 2.0, 3.0])
print(torch.broadcast_to(x, (4, 3)).shape)  # (4,3)

# Explicit expand (shares storage, no copy)
x = torch.randn(1, 4)
y = x.expand(3, 4)       # (3, 4), no copy
z = x.expand(-1, 4)      # -1 means keep that dimension

# repeat (copies data)
r = x.repeat(3, 2)       # (3, 8)
```

### 3.4 Math Operations

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Element-wise
print(a + b)          # or torch.add(a, b)
print(a - b)
print(a * b)          # Hadamard product
print(a / b)
print(a ** 2)
print(torch.exp(a))
print(torch.log(a.abs() + 1e-8))
print(torch.sqrt(a.abs()))
print(torch.sin(a))

# Matrix multiplication
m1 = torch.randn(3, 4)
m2 = torch.randn(4, 5)
print(m1 @ m2)              # (3,5)  — Python 3.5+ matmul operator
print(torch.matmul(m1, m2)) # same
print(torch.mm(m1, m2))     # 2D only version

# Batched matmul
bm1 = torch.randn(10, 3, 4)
bm2 = torch.randn(10, 4, 5)
print(torch.bmm(bm1, bm2).shape)   # (10, 3, 5)
print((bm1 @ bm2).shape)           # same via broadcasting matmul

# einsum: Einstein summation (very powerful)
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum('ij,jk->ik', A, B)  # matmul

# Batch dot product
x = torch.randn(10, 4)
y = torch.randn(10, 4)
dots = torch.einsum('bi,bi->b', x, y)  # (10,)

# Outer product
u = torch.randn(3)
v = torch.randn(4)
outer = torch.einsum('i,j->ij', u, v)  # (3,4)

# Reductions
print(a.sum())                # all elements
print(a.sum(dim=0))           # sum along rows → shape (4,)
print(a.sum(dim=1, keepdim=True))  # keep dim → (3,1)
print(a.mean(), a.std(), a.var())
print(a.max(), a.min())
print(a.argmax(), a.argmin())
print(a.max(dim=0))           # returns (values, indices)

# Clamp
print(a.clamp(min=-1.0, max=1.0))

# Comparison
print(torch.eq(a, b))         # element-wise ==
print(torch.gt(a, 0))         # element-wise >
print(torch.allclose(a, b, atol=1e-4))
```

### 3.5 In-place Operations

In-place operations modify tensors directly. They are suffixed with `_` and are forbidden on tensors that require gradients (breaks the computational graph).

```python
x = torch.randn(3, 4)

x.add_(1.0)        # x = x + 1
x.mul_(2.0)        # x = x * 2
x.zero_()          # x = 0
x.fill_(5.0)       # x = 5
x.clamp_(0, 1)     # in-place clamp

# Copy
y = torch.zeros_like(x)
y.copy_(x)         # copies x into y

# CAUTION: in-place on leaf with requires_grad
w = torch.randn(3, requires_grad=True)
# w.add_(1)  # RuntimeError! Use:
with torch.no_grad():
    w.add_(1)
```

### 3.6 Concatenation and Stacking

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

cat_0  = torch.cat([a, b], dim=0)   # (4, 3) — concatenate along existing dim
cat_1  = torch.cat([a, b], dim=1)   # (2, 6)

stack  = torch.stack([a, b], dim=0) # (2, 2, 3) — new dim
stack2 = torch.stack([a, b], dim=1) # (2, 2, 3) different axis

# chunk: split into N pieces
chunks = torch.chunk(cat_0, chunks=4, dim=0)  # list of 4 tensors

# split: split at specific sizes
parts  = torch.split(cat_0, split_size_or_sections=1, dim=0)
```

---

## 4. Autograd: Computational Graphs and Automatic Differentiation

### 4.1 Core Concepts

PyTorch builds a **dynamic computational graph** during the forward pass. Each node is a `Function` that knows how to compute its output and its gradient contribution (backward). When you call `.backward()`, gradients flow from outputs to leaf tensors via chain rule.

```
Loss
  |
 (backward)
  |
Function nodes (recorded during forward)
  |
Leaf tensors (requires_grad=True) ← gradients accumulate here
```

```python
# Basic gradient computation
x = torch.tensor(3.0, requires_grad=True)   # leaf tensor
y = x ** 2                                  # y = x^2
z = 2 * y + 1                               # z = 2x^2 + 1

z.backward()             # dz/dx = 4x
print(x.grad)            # tensor(12.) — at x=3, dz/dx = 12
```

### 4.2 Gradient Accumulation and Zeroing

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

for _ in range(3):
    y = (x ** 2).sum()
    y.backward()
    print(x.grad)   # accumulates! [2,4,6], [4,8,12], [6,12,18]
    # Don't forget to zero:
    x.grad.zero_()
```

### 4.3 Vector-Valued Backward (Jacobian-vector product)

`.backward()` computes J^T v, where J is the Jacobian and v is `gradient` arg.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2   # y = [x0^2, x1^2, x2^2]

# Pass external gradient (for non-scalar outputs)
v = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient=v)   # equiv to (y * v).sum().backward()
print(x.grad)            # [2x0, 2x1, 2x2] = [2, 4, 6]
```

### 4.4 Gradient Flow Control

```python
x = torch.randn(3, requires_grad=True)
w = torch.randn(3, requires_grad=True)

# No gradient tracking
with torch.no_grad():
    y = x @ w   # fast, no graph built

# Detach from graph (returns tensor sharing data, no grad)
x_detached = x.detach()  # requires_grad=False, same storage

# Check if gradient is tracked
print(x.requires_grad)       # True
print(x.grad_fn)             # None (leaf)

y = x * w
print(y.grad_fn)             # MulBackward0 (non-leaf)
print(y.requires_grad)       # True (inherits from x, w)

# Stop gradient through part of network (detach feature maps)
feature = encoder(input)
target_feature = encoder_target(input).detach()  # stop gradient to target encoder
loss = F.mse_loss(feature, target_feature)
```

### 4.5 Higher-Order Gradients

```python
x = torch.tensor(2.0, requires_grad=True)

# First derivative
y = x ** 3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

# Second derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)  # 6.0*x at x=2 → 12.0

# grad() function (alternative to .backward())
loss = (x ** 2 + 2 * x + 1)
grads = torch.autograd.grad(loss, x)
print(grads[0])  # 2x + 2 = 6
```

### 4.6 Jacobian and Hessian

```python
from torch.autograd.functional import jacobian, hessian

def f(x):
    return torch.stack([x[0] ** 2, x[0] * x[1], x[1] ** 3])

x = torch.tensor([1.0, 2.0])
J = jacobian(f, x)   # (3, 2) Jacobian
H = hessian(lambda x: x.sum(), x)  # full Hessian
print(J)
```

---

## 5. Custom Autograd Functions

Implement custom forward/backward passes for operations not in PyTorch, or to optimize numerics.

```python
class SigmoidFunction(torch.autograd.Function):
    """
    Manual sigmoid with numerically stable backward.
    Forward:  σ(x) = 1 / (1 + exp(-x))
    Backward: dL/dx = dL/dσ * σ(1 - σ)
    """
    @staticmethod
    def forward(ctx, x):
        sigmoid = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid)  # save for backward pass
        return sigmoid

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid, = ctx.saved_tensors
        return grad_output * sigmoid * (1 - sigmoid)

# Usage (via .apply())
x = torch.randn(3, requires_grad=True)
y = SigmoidFunction.apply(x)
y.sum().backward()
print(x.grad)

# Verify against PyTorch's sigmoid
x2 = x.detach().requires_grad_(True)
y2 = torch.sigmoid(x2)
y2.sum().backward()
print(torch.allclose(x.grad, x2.grad))  # True
```

```python
class GradientClipFunction(torch.autograd.Function):
    """Clip gradients during backward."""
    @staticmethod
    def forward(ctx, x, clip_val):
        ctx.save_for_backward(torch.tensor(clip_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        clip_val, = ctx.saved_tensors
        return grad_output.clamp(-clip_val.item(), clip_val.item()), None

# Straight-through estimator (for discrete operations)
class STEQuantize(torch.autograd.Function):
    """Quantize forward, pass gradient straight through."""
    @staticmethod
    def forward(ctx, x, num_levels=256):
        scale = (num_levels - 1)
        return (x * scale).round() / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through

x = torch.randn(4, requires_grad=True)
q = STEQuantize.apply(x)
q.sum().backward()
print(x.grad)  # all ones (straight-through)
```

---

## 6. nn.Module: Defining Models, Parameters, Buffers, Hooks

### 6.1 Basic nn.Module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 512, 10)
print(model)

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total: {total:,}, Trainable: {trainable:,}')
```

### 6.2 Parameters vs Buffers

- **Parameters** (`nn.Parameter`): Trainable, included in `model.parameters()`, saved in `state_dict`.
- **Buffers** (`register_buffer`): Non-trainable tensors (e.g., running stats in BatchNorm), saved in `state_dict`, moved with `.to(device)`.

```python
class CustomNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        # Trainable scale and shift
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))
        # Non-trainable running stats (buffer)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var  = x.var(dim=0)
            # Update running stats (in-place, no grad)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var
        x_norm = (x - mean) / (var + 1e-5).sqrt()
        return self.weight * x_norm + self.bias

norm = CustomNorm(10)
x = torch.randn(32, 10)
print(norm(x).shape)

# Buffers move with model
norm.cuda()
print(norm.running_mean.device)  # cuda:0
```

### 6.3 Accessing Parameters and State

```python
model = MLP(784, 512, 10)

# All parameters (includes nested modules)
for name, param in model.named_parameters():
    print(name, param.shape)

# All buffers
for name, buf in model.named_buffers():
    print(name, buf.shape)

# Named modules (flat list)
for name, module in model.named_modules():
    print(name, type(module).__name__)

# Children (immediate sub-modules only)
for name, child in model.named_children():
    print(name, child)

# State dict: all parameters + buffers
sd = model.state_dict()
for k, v in sd.items():
    print(k, v.shape)
```

### 6.4 Forward and Backward Hooks

```python
# Forward hook: inspects inputs/outputs during forward pass
activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register on specific layer
model = MLP(784, 512, 10)
handle = model.net[0].register_forward_hook(make_hook('fc1'))

x = torch.randn(4, 784)
_ = model(x)
print(activations['fc1'].shape)  # (4, 512)
handle.remove()  # Always remove hooks when done

# Pre-forward hook (modifies input before layer)
def pre_hook(module, input):
    return (input[0] * 0.5,)  # halve the input

handle2 = model.net[0].register_forward_pre_hook(pre_hook)

# Backward hook: inspect/modify gradients
grad_info = {}

def backward_hook(module, grad_input, grad_output):
    grad_info['grad_output'] = grad_output[0].detach()

handle3 = model.net[0].register_full_backward_hook(backward_hook)

loss = model(x).sum()
loss.backward()
print(grad_info['grad_output'].shape)

# Clean up
handle2.remove()
handle3.remove()
```

---

## 7. Common Layers

### 7.1 Linear (Fully Connected)

```
y = xW^T + b,   W ∈ R^{out×in},  b ∈ R^{out}
```

```python
linear = nn.Linear(in_features=256, out_features=128, bias=True)
x = torch.randn(32, 256)
print(linear(x).shape)  # (32, 128)

# Weight initialization
nn.init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
nn.init.zeros_(linear.bias)
```

### 7.2 Convolutional Layers

```
Output size: floor((H + 2P - D(K-1) - 1) / S + 1)
H=height, P=padding, D=dilation, K=kernel, S=stride
```

```python
# Conv2d
conv = nn.Conv2d(
    in_channels=3, out_channels=64,
    kernel_size=3, stride=1, padding=1,
    dilation=1, groups=1, bias=False
)
x = torch.randn(8, 3, 32, 32)
print(conv(x).shape)   # (8, 64, 32, 32)

# Depthwise separable convolution
dw = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)  # depthwise
pw = nn.Conv2d(64, 128, kernel_size=1)                         # pointwise

# Transposed convolution (upsampling)
conv_t = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
x = torch.randn(8, 64, 16, 16)
print(conv_t(x).shape)  # (8, 32, 32, 32)

# Conv1d (sequences)
conv1d = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
x = torch.randn(16, 128, 50)  # (batch, channels, length)
print(conv1d(x).shape)         # (16, 256, 50)

# Pooling
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
gap     = nn.AdaptiveAvgPool2d((1, 1))  # global average pool → (B, C, 1, 1)
```

### 7.3 Batch Normalization and Layer Normalization

**BatchNorm** normalizes across the batch dimension. During training, uses batch stats; during eval, uses running stats.

```
BatchNorm1d: normalizes over (N, L) for each feature
BatchNorm2d: normalizes over (N, H, W) for each channel
```

```python
# BatchNorm2d for CNN feature maps
bn = nn.BatchNorm2d(num_features=64, eps=1e-5, momentum=0.1, affine=True)
x = torch.randn(8, 64, 16, 16)
print(bn(x).shape)  # (8, 64, 16, 16)

# BatchNorm1d for sequences or MLP
bn1d = nn.BatchNorm1d(128)
x = torch.randn(32, 128)
print(bn1d(x).shape)  # (32, 128)

# LayerNorm: normalizes over last D dims (good for Transformers)
# Normalizes over features, not batch — works with any batch size
ln = nn.LayerNorm(normalized_shape=512)
x = torch.randn(4, 10, 512)  # (batch, seq_len, d_model)
print(ln(x).shape)            # (4, 10, 512)

# GroupNorm: divides channels into groups, normalizes within each group
gn = nn.GroupNorm(num_groups=8, num_channels=64)
x = torch.randn(4, 64, 32, 32)
print(gn(x).shape)

# InstanceNorm: normalizes per sample per channel
inst = nn.InstanceNorm2d(64)
```

### 7.4 Dropout

```python
# Standard dropout (zeros random elements)
dropout = nn.Dropout(p=0.5)   # p = probability of zeroing

# 2D dropout (zeros entire feature maps — better for CNNs)
dropout2d = nn.Dropout2d(p=0.5)

# NOTE: dropout is identity at eval time
model.train()
x = torch.ones(4, 10)
print(dropout(x))  # some zeros

model.eval()
print(dropout(x))  # all ones
```

### 7.5 Embedding Layer

Embedding maps integer token IDs to dense vectors. Think of it as a lookup table with learned weights.

```python
# Embedding(vocab_size, embedding_dim)
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=256, padding_idx=0)

# Input: (batch, seq_len) of integer token IDs
tokens = torch.randint(0, 10000, (8, 50))
embedded = embedding(tokens)
print(embedded.shape)  # (8, 50, 256)

# EmbeddingBag: efficient embedding + reduction
emb_bag = nn.EmbeddingBag(10000, 256, mode='mean')

# Positional encoding (sinusoidal — not learned)
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)   # (1, seq_len, d_model)

pe = positional_encoding(100, 512)
print(pe.shape)  # (1, 100, 512)
```

### 7.6 LSTM and GRU

```
LSTM gates (at each timestep):
  i_t = σ(W_i [h_{t-1}, x_t] + b_i)  — input gate
  f_t = σ(W_f [h_{t-1}, x_t] + b_f)  — forget gate
  g_t = tanh(W_g [h_{t-1}, x_t] + b_g) — cell gate
  o_t = σ(W_o [h_{t-1}, x_t] + b_o)  — output gate
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
  h_t = o_t ⊙ tanh(c_t)
```

```python
lstm = nn.LSTM(
    input_size=128,   # feature dim of each token
    hidden_size=256,
    num_layers=2,
    batch_first=True, # (batch, seq, feature) instead of (seq, batch, feature)
    dropout=0.3,      # dropout between layers (not on last layer)
    bidirectional=True
)

x = torch.randn(8, 50, 128)  # (batch, seq_len, input_size)
h0 = torch.zeros(2*2, 8, 256)  # (num_layers * num_dirs, batch, hidden)
c0 = torch.zeros(2*2, 8, 256)

output, (hn, cn) = lstm(x, (h0, c0))
print(output.shape)  # (8, 50, 512) — 256 * 2 directions
print(hn.shape)      # (4, 8, 256) — all layers, both dirs

# GRU (simpler: no cell state)
gru = nn.GRU(128, 256, num_layers=2, batch_first=True, bidirectional=True)
output, hn = gru(x)
print(output.shape)  # (8, 50, 512)
```

### 7.7 Transformer

```python
# Multi-head self-attention
attn = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)

q = k = v = torch.randn(4, 20, 512)  # (batch, seq_len, embed_dim)
out, weights = attn(q, k, v)
print(out.shape)     # (4, 20, 512)
print(weights.shape) # (4, 20, 20)  attention weights

# Full Transformer encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048,
    dropout=0.1, activation='relu', batch_first=True,
    norm_first=True  # Pre-LN (more stable training)
)

# Stack N encoder layers
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,
                                 norm=nn.LayerNorm(512))

src = torch.randn(4, 20, 512)
src_key_padding_mask = torch.zeros(4, 20).bool()  # False = attend
out = encoder(src, src_key_padding_mask=src_key_padding_mask)
print(out.shape)  # (4, 20, 512)

# Full Transformer (encoder + decoder)
transformer = nn.Transformer(
    d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1, batch_first=True
)
src = torch.randn(4, 20, 512)
tgt = torch.randn(4, 15, 512)
out = transformer(src, tgt)
print(out.shape)  # (4, 15, 512)
```

---

## 8. Loss Functions

### 8.1 Regression Losses

```python
pred = torch.randn(32, 1)
target = torch.randn(32, 1)

# MSE: L = (1/N) Σ (y_i - ŷ_i)²
mse = nn.MSELoss(reduction='mean')
print(mse(pred, target))

# MAE (L1): L = (1/N) Σ |y_i - ŷ_i|
mae = nn.L1Loss()
print(mae(pred, target))

# Huber loss (smooth L1): quadratic for small errors, linear for large
# δ * (sqrt(1 + (x/δ)^2) - 1)
huber = nn.HuberLoss(delta=1.0)
print(huber(pred, target))

# SmoothL1Loss
smooth = nn.SmoothL1Loss(beta=1.0)
```

### 8.2 Classification Losses

```python
logits = torch.randn(32, 10)   # raw unnormalized scores
labels = torch.randint(0, 10, (32,))

# CrossEntropyLoss = LogSoftmax + NLLLoss
# = -log(exp(logit_y) / Σ exp(logit_k))
ce = nn.CrossEntropyLoss(
    weight=None,       # class weights for imbalanced data
    reduction='mean',
    label_smoothing=0.1  # regularization: soft targets
)
loss = ce(logits, labels)

# NLLLoss: expects log-probabilities (after log_softmax)
log_probs = F.log_softmax(logits, dim=-1)
nll = nn.NLLLoss()
loss = nll(log_probs, labels)

# Binary cross-entropy with logits (numerically stable)
# L = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
bin_logits = torch.randn(32, 5)
bin_labels = torch.randint(0, 2, (32, 5)).float()
bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([2.0]*5)  # weight for positive class
)
loss = bce(bin_logits, bin_labels)

# KL divergence
kl_div = nn.KLDivLoss(reduction='batchmean')
log_probs = F.log_softmax(torch.randn(32, 10), dim=-1)
target_probs = F.softmax(torch.randn(32, 10), dim=-1)
loss = kl_div(log_probs, target_probs)
```

### 8.3 Custom Loss Functions

```python
class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    FL(p_t) = -α_t (1-p_t)^γ log(p_t)
    γ > 0 reduces the relative loss for well-classified examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)                          # p_t = e^{-CE}
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = F.pairwise_distance(emb1, emb2)
        loss = (label * dist.pow(2) +
                (1 - label) * F.relu(self.margin - dist).pow(2))
        return loss.mean()


class LabelSmoothingCE(nn.Module):
    """Label smoothing cross-entropy."""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_prob = F.log_softmax(pred, dim=-1)
        # Smooth labels
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob, self.smoothing / (self.num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        return -(smooth_target * log_prob).sum(dim=-1).mean()
```

---

## 9. Optimizers and Learning Rate Schedulers

### 9.1 Optimizers

```python
model = MLP(784, 512, 10)

# SGD with momentum and weight decay
sgd = optim.SGD(
    model.parameters(), lr=0.01,
    momentum=0.9, weight_decay=1e-4, nesterov=True
)

# Adam: adaptive learning rates per parameter
adam = optim.Adam(
    model.parameters(), lr=1e-3,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=0
)

# AdamW: decouples weight decay from gradient update
# Recommended over Adam for transformer training
adamw = optim.AdamW(
    model.parameters(), lr=1e-3,
    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
)

# Different lr per layer group (common for fine-tuning)
optimizer = optim.AdamW([
    {'params': model.net[0].parameters(), 'lr': 1e-4},   # lower for early layers
    {'params': model.net[-1].parameters(), 'lr': 1e-3},  # higher for head
], weight_decay=0.01)

# Gradient clipping (prevents exploding gradients in RNNs/Transformers)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

### 9.2 Learning Rate Schedulers

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Step decay: lr = lr * gamma every step_size epochs
step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# MultiStep: decay at specific epochs
multi_step = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# Cosine annealing: lr follows cosine curve from lr to eta_min
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Cosine with warm restarts (SGDR)
cosine_wr = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ReduceLROnPlateau: reduce lr when metric stops improving
plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

# Warmup + cosine (for Transformers)
from torch.optim.lr_scheduler import LambdaLR

def warmup_cosine(step, warmup_steps=1000, total_steps=10000):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)

# OneCycleLR (super-convergence)
one_cycle = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1,
    steps_per_epoch=len(train_loader), epochs=30,
    pct_start=0.3, anneal_strategy='cos'
)

# Usage in training loop
for epoch in range(100):
    train(...)
    val_loss = validate(...)

    # Most schedulers: call after epoch
    step_lr.step()

    # ReduceLROnPlateau: pass metric
    plateau.step(val_loss)
```

---

## 10. Dataset and DataLoader

### 10.1 Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, pandas as pd

class ImageCSVDataset(Dataset):
    """
    Custom dataset loading images from paths listed in a CSV.
    CSV format: image_path, label
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(self.img_dir, row['path'])).convert('RGB')
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### 10.2 DataLoader with Advanced Options

```python
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler

# Basic DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # parallel data loading
    pin_memory=True,      # pin CPU memory for faster GPU transfer
    drop_last=True,       # drop last incomplete batch
    persistent_workers=True,  # keep workers alive between epochs
)

# Weighted sampler for class imbalance
class_counts = [100, 500, 50, 200]
weights = [1.0 / c for c in class_counts]
sample_weights = [weights[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
balanced_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

# Custom collate_fn (for variable-length sequences)
def collate_fn(batch):
    """Pad sequences to max length in batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded  = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels), lengths

seq_loader = DataLoader(seq_dataset, batch_size=32, collate_fn=collate_fn)

# Pack for LSTM
for seqs, labels, lengths in seq_loader:
    packed = nn.utils.rnn.pack_padded_sequence(seqs, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, (hn, cn) = lstm(packed)
    output_padded, lengths_out = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
```

---

## 11. Complete Training Loop

```python
import torch, os, math
from torch.cuda.amp import GradScaler, autocast

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        with autocast():  # mixed precision
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


def save_checkpoint(state, filepath):
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    ckpt = torch.load(filepath, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_acc', 0.0)


def full_training_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model     = MLP(784, 512, 10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    scaler    = GradScaler()

    start_epoch, best_acc = 0, 0.0
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Optionally resume
    last_ckpt = os.path.join(checkpoint_dir, 'last.pth')
    if os.path.exists(last_ckpt):
        start_epoch, best_acc = load_checkpoint(last_ckpt, model, optimizer, scheduler)
        print(f'Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%')

    for epoch in range(start_epoch, 100):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Epoch {epoch+1:3d} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | '
              f'LR: {scheduler.get_last_lr()[0]:.2e}')

        # Save last checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_acc': best_acc,
        }, last_ckpt)

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'model': model.state_dict()},
                            os.path.join(checkpoint_dir, 'best.pth'))
            print(f'  → New best: {best_acc:.2f}%')

    print(f'Training complete. Best val accuracy: {best_acc:.2f}%')
```

---

## 12. GPU Training

### 12.1 Single GPU

```python
device = torch.device('cuda:0')
model  = model.to(device)

# Or use .cuda()
model.cuda()   # defaults to cuda:0

# Move data
inputs, labels = inputs.to(device), labels.to(device)
```

### 12.2 Mixed Precision Training (AMP)

Mixed precision uses **float16** for most ops and **float32** for critical operations. Up to 3x faster on modern GPUs with Tensor Cores.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # scales loss to avoid underflow in float16

for inputs, labels in train_loader:
    inputs, labels = inputs.cuda(), labels.cuda()
    optimizer.zero_grad()

    # Forward in float16
    with autocast():
        outputs = model(inputs)
        loss    = criterion(outputs, labels)

    # Backward: unscale gradients before clipping
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update + unscale if not already done
    scaler.step(optimizer)
    scaler.update()

# bfloat16 (better numerical range, available on Ampere+)
with autocast(dtype=torch.bfloat16):
    outputs = model(inputs)
```

### 12.3 DataParallel (single machine, multi-GPU)

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.cuda()

# Accessing wrapped model's attributes
model.module.some_attribute

# LIMITATION: DataParallel has overhead due to Python GIL and
# data scatter/gather. Prefer DDP for multi-GPU.
```

### 12.4 DistributedDataParallel (DDP)

DDP is the recommended way for multi-GPU and multi-node training.

```python
# launch_script.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)

    model = MLP(784, 512, 10).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # DistributedSampler ensures non-overlapping data per process
    sampler      = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              sampler=sampler, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        sampler.set_epoch(epoch)   # shuffle differently each epoch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()          # gradients automatically synced across GPUs
            optimizer.step()

        if rank == 0:
            print(f'Epoch {epoch} done')

    cleanup()

# Launch: torchrun --nproc_per_node=4 launch_script.py
if __name__ == '__main__':
    import torch.multiprocessing as mp
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)
```

---

## 13. torch.compile, functorch, and TorchScript

### 13.1 torch.compile (PyTorch 2.0+)

`torch.compile` traces and compiles your model using TorchInductor (generates optimized Triton/C++/CUDA kernels). Typically 1.5-3× speedup with no code changes.

**Compilation modes**:
- `default`: Best balance; some graph breaks allowed.
- `reduce-overhead`: Minimizes Python overhead; good for small models or many small batches.
- `max-autotune`: Longer compile (minutes), maximum runtime speed; use for production inference.

**Backends**: `inductor` (default, best for most cases), `cudagraphs` (captures CUDA streams), `aot_eager`/`aot_inductor` (AOT compilation), `onnxrt` (ONNX Runtime).

**Gotchas**: First call triggers tracing and compile (cold start 10s–2min). Different input shapes trigger recompilation. Avoid data-dependent control flow, in-place ops on params, or Python objects inside the graph. Use `dynamic=True` for variable-length sequences.

```python
import torch

model = MLP(784, 512, 10).cuda()
compiled_model = torch.compile(model, mode='default', fullgraph=False, dynamic=False)
x = torch.randn(32, 784, device='cuda')
y = compiled_model(x)  # first call: trace + compile; later calls: fast
with torch.compiler.disable():
    y = compiled_model(x)  # skip compilation for this call
```

### 13.2 functorch / torch.func (PyTorch 2.0+)

`torch.func` (formerly functorch) provides composable function transforms for advanced use cases.

```python
from torch.func import vmap, grad, jacrev, jacfwd, functional_call

# vmap: vectorize over batch dim — avoid manual batching
def f(x):
    return x.sum(dim=-1)
batched_f = vmap(f)  # f: (d,) -> (),  batched_f: (B, d) -> (B,)
out = batched_f(torch.randn(32, 10))

# grad: single-arg gradient
g = grad(lambda p: (model(p) - target).pow(2).sum())(params)

# jacrev / jacfwd: Jacobian (reverse or forward mode)
J = jacrev(model)(x)  # (out_dim, in_dim)

# functional_call: call module with different weights (e.g. MAML, hypernetworks)
from torch.func import functional_call
params_dict = dict(model.named_parameters())
logits = functional_call(model, params_dict, (x,))
```

**Use cases**: MAML/meta-learning, vectorized parallel evaluation, per-sample gradients, efficient Jacobian/Hessian computation.

### 13.3 TorchScript

TorchScript compiles models to an intermediate representation that can run without Python (C++ deployment).

```python
# Method 1: torch.jit.trace (for static control flow)
model.eval()
example_input = torch.randn(1, 784)
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')

# Load and run (no Python needed)
loaded = torch.jit.load('model_traced.pt')
output = loaded(torch.randn(1, 784))

# Method 2: torch.jit.script (handles dynamic control flow)
@torch.jit.script
def compute_fn(x: torch.Tensor, threshold: float) -> torch.Tensor:
    if x.sum() > threshold:
        return x * 2
    return x + 1

class ScriptableModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TorchScript-compatible Python subset
        result = torch.zeros_like(x)
        for i in range(x.shape[0]):
            result[i] = x[i] * 2
        return result

scripted = torch.jit.script(ScriptableModel())
scripted.save('scripted.pt')
```

---

## 14. Model Saving and Loading

```python
model = MLP(784, 512, 10)

# ---- Recommended: state_dict ----
# Save
torch.save(model.state_dict(), 'model.pth')

# Load (architecture must be defined)
model2 = MLP(784, 512, 10)
model2.load_state_dict(torch.load('model.pth', map_location='cpu'))
model2.eval()

# Partial load (e.g., transfer learning)
pretrained = torch.load('pretrained.pth')
model_dict = model.state_dict()
# Filter: only load matching keys/shapes
pretrained = {k: v for k, v in pretrained.items()
              if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained)
model.load_state_dict(model_dict, strict=False)

# ---- Full model (includes architecture) ----
torch.save(model, 'full_model.pth')
model3 = torch.load('full_model.pth')  # requires model class definition

# ---- Checkpoint (all training state) ----
checkpoint = {
    'epoch': 42,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_val_loss': 0.032,
    'config': {'lr': 1e-3, 'batch_size': 64},
}
torch.save(checkpoint, 'checkpoint.pth')

# Resume
ckpt = torch.load('checkpoint.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
start_epoch = ckpt['epoch']
```

---

## 15. Hooks

### 15.1 Feature Extraction

```python
class FeatureExtractor(nn.Module):
    """Extract intermediate features using hooks."""
    def __init__(self, model, layer_names):
        super().__init__()
        self.model    = model
        self.features = {}
        self.handles  = []

        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def forward(self, x):
        self.features.clear()
        return self.model(x)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

import torchvision.models as models
resnet = models.resnet50(pretrained=True)
extractor = FeatureExtractor(resnet, ['layer2', 'layer3', 'layer4'])
extractor.eval()

x = torch.randn(4, 3, 224, 224)
out = extractor(x)
for name, feat in extractor.features.items():
    print(name, feat.shape)

extractor.remove_hooks()
```

### 15.2 Gradient Inspection (CAM, GradCAM)

```python
class GradCAM:
    """Gradient-weighted Class Activation Maps."""
    def __init__(self, model, target_layer):
        self.model        = model
        self.gradients    = None
        self.activations  = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        self.model.zero_grad()
        # Backprop for target class
        one_hot = torch.zeros_like(logits)
        one_hot[range(len(class_idx)), class_idx] = 1
        logits.backward(gradient=one_hot)

        # Pool gradients over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return cam

model   = models.resnet50(pretrained=True)
gradcam = GradCAM(model, model.layer4[-1].conv3)
x       = torch.randn(1, 3, 224, 224, requires_grad=True)
cam     = gradcam(x)
print(cam.shape)  # (1, 1, 7, 7)
```

---

## 16. PyTorch Lightning

PyTorch Lightning removes boilerplate while preserving full flexibility.

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

class LightningMLP(pl.LightningModule):
    def __init__(self, in_dim=784, hidden=512, num_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model     = MLP(in_dim, hidden, num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y  = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        acc    = (logits.argmax(1) == y).float().mean()
        self.log_dict({'train/loss': loss, 'train/acc': acc},
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits  = self(x)
        loss    = self.criterion(logits, y)
        acc     = (logits.argmax(1) == y).float().mean()
        self.log_dict({'val/loss': loss, 'val/acc': acc}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits  = self(x)
        acc     = (logits.argmax(1) == y).float().mean()
        self.log('test/acc', acc)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss'},
        }


# Callbacks
checkpoint_cb = ModelCheckpoint(
    monitor='val/acc', mode='max', save_top_k=3,
    filename='epoch{epoch:02d}-acc{val/acc:.3f}'
)
early_stop_cb = EarlyStopping(monitor='val/loss', patience=10, mode='min')
lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

# Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu', devices=2,    # use 2 GPUs
    strategy='ddp',
    precision='16-mixed',            # automatic mixed precision
    callbacks=[checkpoint_cb, early_stop_cb, lr_monitor_cb],
    logger=TensorBoardLogger('lightning_logs', name='mlp'),
    gradient_clip_val=1.0,
    log_every_n_steps=10,
)

model   = LightningMLP()
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

# Load best checkpoint
best_model = LightningMLP.load_from_checkpoint(checkpoint_cb.best_model_path)
```

---

## 17. Profiling with torch.profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

model = MLP(784, 512, 10).cuda()
x = torch.randn(64, 784, device='cuda')

# Profile CPU and CUDA activity
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    profile_memory=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof_logs'),
) as prof:
    for step in range(10):
        with record_function("forward"):
            y = model(x)
        with record_function("backward"):
            y.sum().backward()
        prof.step()

# Print summary
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

# Export chrome trace
prof.export_chrome_trace('trace.json')
# Open at chrome://tracing
```

---

## 18. Full Examples

### 18.1 MLP for Tabular Data

```python
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data
X, y = make_classification(n_samples=5000, n_features=20, n_classes=3, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

tr_ds  = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

tr_loader  = DataLoader(tr_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model     = TabularMLP(20, [256, 128, 64], 3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for x_b, y_b in tr_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_b), y_b)
        loss.backward()
        optimizer.step()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                pred = model(x_b.to(device)).argmax(1)
                correct += (pred == y_b.to(device)).sum().item()
        print(f'Epoch {epoch+1}: val acc={100*correct/len(val_ds):.2f}%')
```

### 18.2 CNN for Image Classification (CIFAR-10)

```python
import torchvision, torchvision.transforms as T

transform_train = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = T.Compose([
    T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
test_set  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            ResBlock(128), ResBlock(128)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            ResBlock(256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = SmallResNet().to(device)
compiled = torch.compile(model)  # PyTorch 2.0 compile for speedup

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1,
                                           steps_per_epoch=len(train_loader), epochs=100)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler    = torch.cuda.amp.GradScaler()

best_acc = 0.0
for epoch in range(100):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = criterion(model(inputs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            pred = model(inputs.to(device)).argmax(1)
            correct += (pred == labels.to(device)).sum().item()
    acc = 100.0 * correct / len(test_set)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_cnn.pth')
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/100  Test Acc: {acc:.2f}%  Best: {best_acc:.2f}%')
```

### 18.3 LSTM for Text Classification

```python
class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout  = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        packed   = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(),
                                                      batch_first=True, enforce_sorted=False)
        output, (hn, _) = self.lstm(packed)
        # Concatenate forward and backward final hidden states
        hn = torch.cat([hn[-2], hn[-1]], dim=1)  # (batch, hidden*2)
        return self.fc(self.dropout(hn))
```

### 18.4 Transformer from Scratch

```python
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.scale   = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn   = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, V), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k      = d_model // num_heads
        self.h        = num_heads
        self.W_q      = nn.Linear(d_model, d_model)
        self.W_k      = nn.Linear(d_model, d_model)
        self.W_v      = nn.Linear(d_model, d_model)
        self.W_o      = nn.Linear(d_model, d_model)
        self.attention= ScaledDotProductAttention(self.d_k, dropout)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        def proj_and_split(x, W):
            return W(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
        Q, K, V = proj_and_split(Q, self.W_q), proj_and_split(K, self.W_k), proj_and_split(V, self.W_v)
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.W_o(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN (more stable)
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = nn.Embedding(max_len, d_model)
        self.layers    = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_final  = nn.LayerNorm(d_model)
        self.classifier= nn.Linear(d_model, num_classes)
        self.dropout   = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=d_model ** -0.5)

    def forward(self, tokens, mask=None):
        B, L = tokens.shape
        pos  = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x    = self.dropout(self.embedding(tokens) + self.pos_enc(pos))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x[:, 0])  # [CLS] token representation
        return self.classifier(x)

# Example usage
model = TransformerClassifier(
    vocab_size=30000, d_model=256, num_heads=8, d_ff=1024,
    num_layers=4, num_classes=5, max_len=128
)
tokens = torch.randint(1, 30000, (8, 128))
logits = model(tokens)
print(logits.shape)  # (8, 5)
```

---

## Common Pitfalls and Debugging

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Forgetting `model.train()` / `model.eval()`** | Dropout/BatchNorm behave incorrectly | Call `model.train()` before training, `model.eval()` before eval |
| **In-place ops on leaf tensors with `requires_grad`** | RuntimeError | Use `with torch.no_grad():` or avoid in-place on leaf params |
| **Gradient not zeroed** | Gradients accumulate across batches | Call `optimizer.zero_grad(set_to_none=True)` each step |
| **Mixing devices** | RuntimeError (CPU vs CUDA) | Ensure model and data on same device; use `.to(device)` |
| **`state_dict` vs full model save** | Version/code coupling when loading | Prefer `torch.save(model.state_dict(), ...)` for portability |
| **DataLoader `num_workers=0` on Windows** | Multiprocessing spawn issues | Use `num_workers=0` or protect `if __name__ == '__main__'` |
| **torch.compile cold start** | Slow first batch | Warm-up with a few dummy batches before benchmarking |
| **DDP and `DistributedSampler`** | Uneven batches, hangs | Call `sampler.set_epoch(epoch)` each epoch |

---

## Production Deployment Notes

- **Export**: ONNX (cross-framework), TorchScript (C++ lib), `torch.compile` (inductor).
- **Serving**: TorchServe, Triton Inference Server, or custom FastAPI/Flask with loaded model.
- **Quantization**: `torch.quantization` (PTQ), `torch.ao.quantization`; INT8 for 2–4× smaller/faster.
- **Mobile**: CoreML (iOS), TFLite via ONNX. Use smaller models (MobileNet, EfficientNet-Lite).
- **Best practice**: Save `state_dict` + config; version checkpoints; profile before optimizing.

---

## Resources and Further Reading

| Resource | Link |
|---|---|
| Official Docs | pytorch.org/docs |
| Tutorials | pytorch.org/tutorials |
| torch.compile guide | pytorch.org/tutorials/recipes/torch_compile_inductor.html |
| torch.func (functorch) | pytorch.org/docs/stable/func |
| PyTorch Examples | github.com/pytorch/examples |
| PyTorch Lightning | lightning.ai |
| timm (image models) | timm.fast.ai |
| Hugging Face Transformers | huggingface.co |
| TorchServe | pytorch.org/serve |

**Key Takeaways:**
1. Tensors are the fundamental data structure — understand dtypes, devices, and memory layout
2. Autograd builds a dynamic computation graph — understand `requires_grad`, `backward()`, `no_grad()`
3. `nn.Module` is the building block — compose modules, register parameters/buffers, use hooks
4. The training loop: forward → loss → backward → clip → step → zero_grad
5. For performance: mixed precision AMP, DDP for multi-GPU, `torch.compile` for 2x+ speedup
6. Always save `state_dict`, not the full model
7. Use PyTorch Lightning to remove boilerplate in production training
