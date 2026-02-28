# Model Compression: Complete Guide

## Table of Contents
1. [Introduction to Model Compression](#introduction-to-model-compression)
2. [Quantization](#quantization)
3. [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
4. [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
5. [Pruning](#pruning)
6. [Knowledge Distillation](#knowledge-distillation)
7. [Low-Rank Factorization](#low-rank-factorization)
8. [Neural Architecture Search for Efficiency](#neural-architecture-search-for-efficiency)
9. [Practical Examples](#practical-examples)
10. [Advanced Topics](#advanced-topics)
11. [Best Practices](#best-practices)

---

## Introduction to Model Compression

**Model compression** reduces the size and computational cost of neural networks while preserving accuracy. Essential for deployment on edge devices, mobile, and cost-effective inference.

### Why Compress?

| Motivation | Benefit |
|------------|---------|
| **Edge deployment** | Run on phones, IoT devices |
| **Latency** | Faster inference for real-time apps |
| **Cost** | Lower GPU/cloud compute costs |
| **Memory** | Fit larger models in VRAM |
| **Energy** | Reduced power consumption |

### Compression Techniques Overview

| Technique | Size Reduction | Accuracy Impact | Training Required |
|-----------|----------------|-----------------|-------------------|
| **Quantization** | 2–4× | Low (with calibration) | PTQ: No; QAT: Yes |
| **Pruning** | 2–10× | Moderate | Usually yes |
| **Distillation** | 2–5× | Low | Yes (teacher needed) |
| **Low-rank** | 2–4× | Moderate | Yes |

---

## Quantization

Quantization maps floating-point weights/activations to lower-bit representations (e.g., FP32 → INT8).

### Benefits

- **4× smaller** model (32-bit → 8-bit)
- **2–4× faster** on hardware with INT8 support (GPUs, TPUs)
- **Lower memory bandwidth** for inference

### Quantization Types

```python
# FP32 (full precision): 32 bits per value
# FP16 (half precision): 16 bits, ~2× smaller
# INT8: 8 bits, ~4× smaller
# INT4: 4 bits, ~8× smaller (more accuracy loss)

# Dynamic range: INT8 can represent -128 to 127
# Scale and zero-point: real_value = scale * (quantized - zero_point)
```

### Scale and Zero-Point

For symmetric quantization (INT8):
- `scale = max(abs(weights)) / 127`
- `quantized = round(weights / scale)`
- `dequantized = quantized * scale`

For asymmetric (with zero-point for ReLU):
- `scale = (max - min) / 255`
- `zero_point = round(-min / scale)`
- `quantized = round(weights / scale) + zero_point`

---

## Post-Training Quantization (PTQ)

PTQ quantizes a trained model **without** retraining. Use calibration data to estimate activation ranges.

### Dynamic Quantization (Activations on-the-fly)

```python
import torch
import torch.quantization

# Dynamic: weights quantized, activations quantized at runtime
model = MyModel()
model.eval()

# Apply dynamic quantization (good for LSTM, Transformer)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

# Inference
output = quantized_model(input)
```

### Static Quantization (Calibration)

```python
# Static: both weights and activations quantized with calibration
model.eval()

# 1. Fuse Conv-Bn-ReLU
model_fused = torch.quantization.fuse_modules(
    model,
    [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']]
)

# 2. Set config
model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# 3. Prepare
model_prepared = torch.quantization.prepare(model_fused, inplace=False)

# 4. Calibrate with representative data
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

# 5. Convert to quantized
model_quantized = torch.quantization.convert(model_prepared, inplace=False)
```

### Per-Channel vs Per-Tensor

```python
# Per-tensor: one scale for entire tensor
# Per-channel: one scale per output channel (better for Conv)
qconfig = torch.quantization.get_default_qconfig('fbgemm')
# Per-channel is default for Conv2d
```

---

## Quantization-Aware Training (QAT)

QAT simulates quantization during training so the model learns to compensate.

### QAT with PyTorch

```python
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model_fused, inplace=False)

# Train with fake quantization
optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-4)
for epoch in range(5):
    for batch, labels in train_loader:
        optimizer.zero_grad()
        output = model_prepared(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Convert to real quantization
model_prepared.eval()
model_quantized = torch.quantization.convert(model_prepared, inplace=False)
```

### LLM Quantization: GPTQ, AWQ, GGUF

```python
# GPTQ: Post-training quantization for LLMs
# pip install auto-gptq

from transformers import AutoModelForCausalLM
from auto_gptq import BaseQuantizeConfig
from auto_gptq import AutoGPTQForCausalLM

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config
)

# Quantize with calibration data
model.quantize(calib_data, batch_size=1)

# Save quantized
model.save_quantized("./llama-7b-4bit")

# AWQ: Activation-aware weight quantization
# pip install autoawq
# Similar API, often better accuracy for LLMs
```

### QLoRA: Quantized LoRA

```python
# 4-bit base model + LoRA adapters
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
# Train LoRA adapters on 4-bit base
```

---

## Pruning

Pruning removes redundant weights (often small-magnitude) to create sparse models.

### Unstructured vs Structured

- **Unstructured**: Remove individual weights (high sparsity, needs sparse kernels)
- **Structured**: Remove entire channels/filters (easy to deploy, no special hardware)

### Magnitude-Based Pruning

```python
import torch.nn.utils.prune as prune

# Global unstructured pruning
model = MyModel()
parameters_to_prune = [
    (module, 'weight') for module in model.modules()
    if isinstance(module, torch.nn.Linear)
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3  # 30% of weights pruned
)

# Make pruning permanent
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)
```

### Structured Pruning (Channels)

```python
# Prune 30% of channels in Conv2d
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
```

### Iterative Pruning Schedule

```python
# Gradual pruning during training
from torch.nn.utils import prune

def apply_pruning_schedule(model, amount, epoch, total_epochs):
    """Increase pruning over time"""
    current_amount = amount * (epoch / total_epochs)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=current_amount)
```

### Torch Pruning (Production)

```python
import torch_pruning as tp

# Example: Prune 50% of channels in a CNN
model = resnet18()
example_input = torch.randn(1, 3, 224, 224)

pruner = tp.pruner.MetaPruner(
    model,
    example_input,
    importance=tp.importance.MagnitudeImportance(),
    pruning_ratio=0.5,
    ignored_layers=[model.fc]  # Don't prune classifier
)
pruner.step()
```

---

## Knowledge Distillation

Train a **student** to mimic a **teacher** (larger) model. Student learns soft labels.

### Soft Labels

Instead of hard labels [0, 0, 1, 0], use teacher's softmax output [0.1, 0.2, 0.6, 0.1] as targets.

### Distillation Loss

L = α * L_CE(student, hard_labels) + (1 - α) * T² * L_KL(softmax(student/T), softmax(teacher/T))

Temperature T softens distributions; higher T = smoother.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """Knowledge distillation loss"""
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * ce_loss + (1 - alpha) * kl_loss

# Training loop
teacher.eval()
for batch, labels in train_loader:
    with torch.no_grad():
        teacher_logits = teacher(batch)
    student_logits = student(batch)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

### Response vs Feature Distillation

```python
# Response: match output logits (above)
# Feature: match intermediate layer activations

def feature_distillation_loss(student_features, teacher_features):
    """Match intermediate representations"""
    return F.mse_loss(student_features, teacher_features)

# Use at multiple layers for better transfer
```

### DistilBERT-Style

```python
# BERT → DistilBERT: 6 layers instead of 12, same hidden size
# Trained with: MLM loss + distillation loss + cosine embedding loss
```

---

## Low-Rank Factorization

Replace weight matrix W (m×n) with W ≈ A·B where A is m×r, B is r×n, r ≪ min(m,n).

### SVD-Based Compression

```python
def svd_compress_linear(linear_layer, rank_ratio=0.5):
    """Compress Linear layer via SVD"""
    W = linear_layer.weight.data
    U, S, Vh = torch.linalg.svd(W)
    
    r = int(W.shape[0] * rank_ratio)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    
    A = U_r * S_r.sqrt()
    B = Vh_r * S_r.sqrt()
    
    # Replace with two smaller layers
    return nn.Sequential(
        nn.Linear(W.shape[1], r, bias=False),
        nn.Linear(r, W.shape[0], bias=linear_layer.bias is not None)
    )
```

### LoRA as Low-Rank Adaptation

LoRA adds ΔW = A·B; during inference, merge: W' = W + A·B.

```python
# Merge LoRA weights for deployment (no extra params at inference)
from peft import PeftModel
merged_model = model.merge_and_unload()
```

---

## Neural Architecture Search for Efficiency

### NAS for Mobile

- **MobileNet**: Depthwise separable convolutions
- **EfficientNet**: Compound scaling (depth, width, resolution)
- **NAS-discovered** architectures (e.g., MNasNet)

### Efficient Building Blocks

```python
# Depthwise separable convolution
# Standard: in_ch × out_ch × k × k params per position
# Depthwise: in_ch × 1 × k × k + in_ch × out_ch × 1 × 1
# ~8-9× fewer params for 3×3 conv

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size//2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

---

## Practical Examples

### Example 1: Full PTQ Pipeline for CNN

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

# Fuse
model_fused = torch.quantization.fuse_modules(
    model,
    [['conv1', 'bn1', 'relu1'],
     ['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu1'],
     # ... (all conv-bn-relu)
])

model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_prepared = torch.quantization.prepare(model_fused, inplace=False)

# Calibrate
for images, _ in calibration_loader:
    model_prepared(images)

model_quantized = torch.quantization.convert(model_prepared)
torch.save(model_quantized.state_dict(), 'resnet18_int8.pt')
```

### Example 2: Distillation for BERT → TinyBERT

```python
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import BertModel  # Teacher
# Student: 4 layers, 312 hidden

teacher = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
student = SmallBertForSequenceClassification(num_layers=4, hidden_size=312, num_labels=2)

for batch in train_loader:
    teacher_logits = teacher(**batch).logits
    student_logits = student(**batch).logits
    loss = distillation_loss(student_logits, teacher_logits, batch['labels'])
    loss.backward()
```

### Example 3: Pruning + Quantization

```python
# First prune, then quantize
model = load_model()
prune_model(model, amount=0.5)
model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
# Combined: ~8× smaller
```

---

## Advanced Topics

### Mixed Precision (FP16/BF16)

```python
# FP16 training with automatic loss scaling
scaler = torch.cuda.amp.GradScaler()
for batch in train_loader:
    with torch.cuda.amp.autocast():
        loss = model(batch)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Sparse Models (MoE, Mixtral)

Mixture of Experts: Only a subset of experts active per token. Effective capacity without proportional compute.

### ONNX and TensorRT

```python
# Export for optimized inference
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=14)
# TensorRT: further optimizations for NVIDIA GPUs
```

---

## Best Practices

1. **Start with PTQ**; use QAT only if accuracy drops
2. **Calibrate with representative data** for static quantization
3. **Prune gradually** during training
4. **Distill with soft labels** at higher temperature
5. **Validate on same metrics** as original model
6. **Profile** latency and memory before/after
7. **Use hardware-aware** quantization (different for CPU vs GPU)

---

## Summary

| Technique | When to Use | Typical Gain |
|-----------|-------------|--------------|
| PTQ | Fast deployment, minimal accuracy loss | 2–4× smaller, faster |
| QAT | Need best accuracy after quantization | Better than PTQ |
| Pruning | High sparsity, structured for deployment | 2–10× sparser |
| Distillation | Small student from large teacher | 2–5× smaller |
| LoRA/QLoRA | LLM fine-tuning on limited hardware | Train 7B on 1 GPU |
