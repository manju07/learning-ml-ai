# Vision Transformers (ViT): Complete Guide

## Table of Contents
1. [Introduction to Vision Transformers](#introduction-to-vision-transformers)
2. [From CNNs to Transformers](#from-cnns-to-transformers)
3. [ViT Architecture](#vit-architecture)
4. [Patch Embedding](#patch-embedding)
5. [Positional Encoding for Images](#positional-encoding-for-images)
6. [Transformer Encoder Block](#transformer-encoder-block)
7. [Pre-training Strategies](#pre-training-strategies)
8. [Fine-tuning ViT](#fine-tuning-vit)
9. [DeiT and Data-Efficient Training](#deit-and-data-efficient-training)
10. [Detection and Segmentation (DETR, Mask R-CNN + ViT)](#detection-and-segmentation)
11. [Practical Examples](#practical-examples)
12. [Advanced Topics](#advanced-topics)

---

## Introduction to Vision Transformers

**Vision Transformers (ViT)** apply the Transformer architecture—originally developed for NLP—to images. Instead of convolutions, ViT processes images as sequences of patches and uses self-attention to model global relationships.

### Why ViT?

| CNN | ViT |
|-----|-----|
| Local receptive fields, hierarchical | Global attention from layer 1 |
| Inductive bias (translation, locality) | Less inductive bias, data-hungry |
| Efficient for small datasets | Needs large-scale pre-training |
| Fixed kernel size | Flexible patch size |

### Key Breakthrough

- **ViT** (Dosovitskiy et al., 2021): First to show pure Transformer can match CNNs on ImageNet with sufficient pre-training
- **DeiT**: Data-efficient ViT with distillation
- **Swin Transformer**: Hierarchical, efficient for dense tasks

---

## From CNNs to Transformers

### CNN Limitation

CNNs process images with local convolutions. Receptive field grows with depth. Global context requires many layers.

### Transformer for Images

1. Split image into patches (e.g., 16×16)
2. Flatten each patch → sequence of "tokens"
3. Add positional embedding
4. Feed to standard Transformer encoder
5. Use [CLS] token or mean patch tokens for classification

```python
# Intuition: Each patch = "word", image = "sentence"
# Attention: patch i attends to all patches j (global)
```

---

## ViT Architecture

### High-Level Flow

```
Input Image (224×224×3)
    → Patch Embedding (14×14 patches = 196 tokens, each 768-dim)
    → Add [CLS] token + position embeddings
    → Transformer Encoder × 12 layers
    → [CLS] output → MLP head → Classification
```

### Model Variants

| Model | Patch Size | Layers | Hidden Dim | Params |
|-------|------------|--------|------------|--------|
| ViT-B/16 | 16×16 | 12 | 768 | 86M |
| ViT-L/16 | 16×16 | 24 | 1024 | 307M |
| ViT-H/14 | 14×14 | 32 | 1280 | 632M |

---

## Patch Embedding

### Process

1. Split image into non-overlapping patches: 224/16 = 14×14 patches
2. Flatten each patch: 16×16×3 = 768
3. Linear projection: 768 → 768 (embed_dim)

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# Example
pe = PatchEmbedding(224, 16, 3, 768)
x = torch.randn(2, 3, 224, 224)
out = pe(x)  # [2, 196, 768]
```

### Alternative: Linear Projection of Flattened Patches

```python
# Unfold patches, then linear
patches = x.unfold(2, 16, 16).unfold(3, 16, 16)  # [B, C, 14, 14, 16, 16]
patches = patches.contiguous().view(B, -1, 16*16*3)
embeddings = nn.Linear(16*16*3, embed_dim)(patches)
```

---

## Positional Encoding for Images

Images have 2D structure. Options:

1. **1D learned**: Treat sequence as 1D, learn position embeddings (ViT default)
2. **2D learned**: Separate row/col embeddings
3. **Sinusoidal**: Fixed, like original Transformer

```python
class PositionalEncoding2D(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # x: [B, N+1, D] (includes CLS)
        return x + self.pos_embed
```

### CLS Token

Prepend a learnable [CLS] token. Final [CLS] representation used for classification (like BERT).

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
# Forward: x = [cls_token, patch_1, ..., patch_N]
```

---

## Transformer Encoder Block

Same as standard Transformer:

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

---

## Pre-training Strategies

### Supervised (ViT Original)

- Dataset: ImageNet-21K, JFT-300M
- Loss: Cross-entropy on labels
- Requires large data

### Self-Supervised: MAE (Masked Autoencoder)

```python
# MAE: Mask 75% of patches, reconstruct pixels
# 1. Encode visible patches
# 2. Decoder reconstructs masked patches
# 3. Loss: MSE between reconstructed and original pixels
# No labels needed; learns strong representations
```

### Self-Supervised: DINO, MoCo v3

- Contrastive or self-distillation
- ViT as backbone

---

## Fine-tuning ViT

### Standard Fine-tuning

1. Load pre-trained ViT (e.g., on ImageNet)
2. Replace classification head for your num_classes
3. Fine-tune all layers (or freeze early layers)

```python
from transformers import ViTForImageClassification, ViTImageProcessor

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Change head for custom classes
model.classifier = nn.Linear(768, num_classes)

# Fine-tune
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

### Linear Probe

Freeze backbone, train only head. Fast, good for small datasets.

```python
for param in model.vit.parameters():
    param.requires_grad = False
# Only model.classifier is trained
```

---

## DeiT and Data-Efficient Training

**DeiT** (Data-efficient image Transformers) uses **knowledge distillation** to train ViT with less data.

### Distillation Token

Add a [DIST] token. It learns from teacher (e.g., CNN) soft labels.

```python
# Loss = alpha * CE(student, hard_labels) + (1-alpha) * KL(student, teacher)
# [DIST] token trained to match teacher output
```

### Training Tips (DeiT)

- Strong augmentation (RandAugment, random erasing)
- Repeated augmentation (same image, different augs, in same batch)
- Teacher: RegNet or ConvNet

---

## Detection and Segmentation

### DETR (Detection Transformer)

- **Backbone**: CNN (e.g., ResNet) extracts features
- **Transformer**: Encoder-decoder on flattened feature map
- **Object queries**: Learnable embeddings; each predicts one object (or none)
- **Bipartite matching**: Match predictions to ground truth for loss

```python
# DETR: No NMS, no anchors
# Queries attend to image features
# Output: (bbox, class) per query
```

### Segment Anything (SAM)

- ViT backbone (Huge)
- Prompt encoder (points, boxes, masks)
- Mask decoder

### U-Net with ViT Encoder

Replace CNN encoder with ViT; keep decoder for segmentation.

---

## Practical Examples

### Example 1: Image Classification with ViT

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

image = Image.open("cat.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()
print(f"Predicted: {model.config.id2label[predicted_class]}")
```

### Example 2: Custom ViT from Scratch (Simplified)

```python
import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token
```

### Example 3: Feature Extraction for Downstream Task

```python
# Extract patch embeddings before classification head
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.eval()

def get_features(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.vit(**inputs.pixel_values)
    # outputs.last_hidden_state: [1, 197, 768]
    return outputs.last_hidden_state[:, 0]  # CLS token [1, 768]
```

---

## Advanced Topics

### Swin Transformer

- **Hierarchical**: Like CNN, multi-scale
- **Shifted windows**: Attention within local windows, shift for cross-window connection
- **Efficient**: O(window²) per window vs O(N²) global

### PVT (Pyramid Vision Transformer)

- Progressive shrinking of sequence length (spatial reduction)
- Multi-scale feature maps for detection/segmentation

### Hybrid: CNN + ViT

- Early layers: CNN (local features, efficient)
- Late layers: Transformer (global context)
- Best of both

### Attention Visualization

```python
# Extract attention maps from ViT
# Which patches does [CLS] attend to?
attention_weights = model.vit.encoder.layer[-1].attention.attention.weights
# Reshape to 2D, overlay on image
```

### Compute Considerations

- ViT: O(N²) attention, N = num patches
- 224×224, patch 16 → 196 patches → manageable
- 512×512, patch 16 → 1024 patches → need windowed/local attention (Swin)

---

## Best Practices

1. **Pre-train or use pre-trained**: ViT needs large data; use ImageNet/ViT or MAE
2. **Patch size**: 16 balances accuracy and speed; 32 for faster, less accurate
3. **Resolution**: Fine-tune at higher resolution (e.g., 384) for better accuracy
4. **Augmentation**: Strong aug (RandAugment) crucial
5. **Learning rate**: Lower than CNN; warmup helpful

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Patches | Image → sequence of patches |
| [CLS] | Classification from first token |
| Attention | Global from first layer |
| Pre-training | Supervised (Imagenet) or self-supervised (MAE) |
| Fine-tuning | Full or linear probe |
| Alternatives | Swin (efficient), DETR (detection) |

**Libraries**: `transformers`, `timm`
