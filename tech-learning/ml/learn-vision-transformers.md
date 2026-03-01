# Vision Transformers: A Comprehensive Guide from Fundamentals to Cutting-Edge Research

## Table of Contents
1. [Motivation: Why Move Beyond CNNs?](#1-motivation-why-move-beyond-cnns)
2. [Vision Transformer (ViT): Full Architecture](#2-vision-transformer-vit-full-architecture)
3. [Mathematical Formulations](#3-mathematical-formulations)
4. [ViT Variants: DeiT, BEiT, DeiT-III](#4-vit-variants-deit-beit-deit-iii)
5. [Swin Transformer: Hierarchical Vision](#5-swin-transformer-hierarchical-vision)
6. [More Efficient Architectures: CvT, CoAtNet, MaxViT, EfficientViT, PVT](#6-more-efficient-architectures)
7. [CLIP and Contrastive Vision-Language Pretraining](#7-clip-and-contrastive-vision-language-pretraining)
8. [ALIGN, BLIP, BLIP-2, LLaVA](#8-align-blip-blip-2-llava)
9. [Segment Anything Model (SAM)](#9-segment-anything-model-sam)
10. [DINO and DINOv2: Self-Supervised ViT Training](#10-dino-and-dinov2-self-supervised-vit-training)
11. [Detection Transformers: DETR Family](#11-detection-transformers-detr-family)
12. [Grounding DINO and GroundedSAM](#12-grounding-dino-and-groundedsam)
13. [Masked Image Modeling: MAE, SimMIM, BEiT, data2vec](#13-masked-image-modeling)
14. [Video Transformers: ViViT, Video Swin, TimeSFormer](#14-video-transformers)
15. [Vision-Language Models (VLMs)](#15-vision-language-models-vlms)
16. [Pre-training vs Fine-tuning Strategies](#16-pre-training-vs-fine-tuning-strategies)
17. [Efficient ViTs: Pruning, Distillation, Token Reduction](#17-efficient-vits)
18. [Implementation: HuggingFace + timm](#18-implementation-huggingface--timm)
19. [Full Code: ViT from Scratch](#19-full-code-vit-from-scratch)
20. [Transfer Learning with Swin Transformer](#20-transfer-learning-with-swin-transformer)
21. [Best Practices and Research Insights](#21-best-practices-and-research-insights)

---

## 1. Motivation: Why Move Beyond CNNs?

### CNN Strengths and Limitations

Convolutional Neural Networks dominated computer vision for over a decade. CNNs exploit two powerful inductive biases:

1. **Translation equivariance**: If an object shifts in the image, feature maps shift correspondingly
2. **Locality**: Each neuron connects to a small receptive field, capturing local patterns

However, these same inductive biases become **limitations** in several scenarios:

#### The Receptive Field Problem

In a standard CNN, the effective receptive field grows linearly with depth. For a stack of 3×3 convolutions:
- After layer 1: 3×3 receptive field
- After layer k: (2k+1)×(2k+1) receptive field
- For 224×224 image to have full context: need ~112 layers

This means **long-range dependencies** (e.g., the relation between a face in the top-left and hands in the bottom-right) require very deep networks or dilated convolutions.

#### Fixed Spatial Processing

CNNs apply the same learned filters everywhere in the image. But different spatial locations may benefit from different processing depending on what's there — a concept called **adaptive computation**, which transformers handle naturally via attention.

#### Hard to Model Global Context

Tasks like image captioning, VQA, and scene understanding require models to reason about relationships between distant image regions. CNNs need special tricks (non-local blocks, squeeze-and-excitation) to capture global context, while Transformers do it naturally.

### Historical Context

```
LeNet (1998) → AlexNet (2012) → VGG (2014) → ResNet (2015) → EfficientNet (2019)
                                                                      ↓
                                                               ViT (2020) → Swin (2021) → ...
```

The key turning point: **Dosovitskiy et al. (2020)** showed that pure Transformers, when pre-trained on sufficient data (JFT-300M), can match or exceed CNN performance on ImageNet—without any convolutional operations.

### CNN vs ViT Comparison

| Property | CNN | Vision Transformer |
|----------|-----|-------------------|
| Receptive field | Local, grows with depth | Global from layer 1 |
| Inductive bias | Strong (locality, translation equivariance) | Weak (learns from data) |
| Data efficiency | Better on small datasets | Needs large-scale pre-training |
| Long-range dependencies | Requires deep stacks or non-local ops | Natural via attention |
| Scalability | Linear with layers | Scales well with data and model size |
| Positional encoding | Implicit (grid structure) | Explicit (learned or sinusoidal) |
| Feature hierarchy | Built-in (pooling) | Optional (Swin adds it) |
| Attention complexity | O(1) per conv | O(N²) for global, O(W²) for window |

---

## 2. Vision Transformer (ViT): Full Architecture

### High-Level Pipeline

```
Input Image [H, W, C]
    │
    ▼
Divide into N = (H/P × W/P) patches of size P×P×C
    │
    ▼  
Flatten each patch → vectors of size P²·C
    │
    ▼
Linear projection E ∈ ℝ^{P²C × D}  → N patch embeddings of dim D
    │
    ▼
Prepend learnable [CLS] token
    │
    ▼
Add 1D learned positional embeddings (N+1 positions)
    │
    ▼
Transformer Encoder × L layers (each: LN → MHSA → LN → MLP + residuals)
    │
    ▼
Extract [CLS] token output z_L^0  (or mean over patch tokens)
    │
    ▼
MLP Classification Head → class logits
```

### Model Variants

| Variant | Patch Size | Layers (L) | Hidden Dim (D) | MLP Dim | Heads | Params |
|---------|-----------|-----------|----------------|---------|-------|--------|
| ViT-Tiny | 16×16 | 12 | 192 | 768 | 3 | 6M |
| ViT-Small | 16×16 | 12 | 384 | 1536 | 6 | 22M |
| ViT-Base/16 | 16×16 | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large/16 | 16×16 | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge/14 | 14×14 | 32 | 1280 | 5120 | 16 | 632M |
| ViT-G/14 | 14×14 | 40 | 1408 | 6144 | 16 | ~1.8B |

---

## 3. Mathematical Formulations

### 3.1 Patch Embedding

Given an image **x** ∈ ℝ^{H×W×C}, with patch size P:

1. Reshape into sequence: **x** → **x_p** ∈ ℝ^{N × (P²·C)}, where N = HW/P²
2. Project with E ∈ ℝ^{(P²·C) × D}:

\[
\mathbf{z}_i = \mathbf{x}_p^i \mathbf{E}, \quad i = 1, \ldots, N
\]

The class token **x_class** ∈ ℝ^D is prepended:

\[
\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1\mathbf{E}; \ldots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}
\]

where **E_pos** ∈ ℝ^{(N+1)×D} is the positional embedding.

### 3.2 Multi-Head Self-Attention (MHSA)

For input **z** ∈ ℝ^{(N+1)×D}, with h heads:

Each head computes:
\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
\]

where:
- \(\mathbf{Q} = \mathbf{z}\mathbf{W}_Q^h\), \(\mathbf{K} = \mathbf{z}\mathbf{W}_K^h\), \(\mathbf{V} = \mathbf{z}\mathbf{W}_V^h\)
- \(\mathbf{W}_Q^h, \mathbf{W}_K^h, \mathbf{W}_V^h \in \mathbb{R}^{D \times d_k}\), \(d_k = D/h\)

Multi-head combines all heads:
\[
\text{MHSA}(\mathbf{z}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O
\]

**Computational complexity**: O((N+1)² · D) — quadratic in sequence length N.

### 3.3 MLP Block

The feed-forward network after attention:
\[
\text{MLP}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
\]

where W₁ ∈ ℝ^{D × D_ff}, W₂ ∈ ℝ^{D_ff × D}, and D_ff = 4D typically.

### 3.4 Full Layer Computation

With Pre-LN (Layer Norm before attention, which is more stable):
\[
\mathbf{z}^\prime_\ell = \text{MHSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
\]
\[
\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}^\prime_\ell)) + \mathbf{z}^\prime_\ell
\]

### 3.5 Positional Encoding

**1D Learned** (ViT default): 
\[
\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}, \quad \text{initialized with } \mathcal{N}(0, 0.02)
\]

**2D Sine/Cosine** (fixed):
\[
PE(pos_r, pos_c, 2i) = \sin\left(\frac{pos_r}{10000^{2i/D/2}}\right), \quad PE(pos_r, pos_c, 2i+1) = \cos(\cdot)
\]

**Relative Position Bias** (Swin): Add bias terms b_{ij} to attention scores based on relative positions of tokens i and j.

### 3.6 Classification

\[
\hat{y} = \text{softmax}(\text{MLP}_{head}(\text{LN}(\mathbf{z}_L^0)))
\]

where \(\mathbf{z}_L^0\) is the [CLS] token output from the last layer.

---

## 4. ViT Variants: DeiT, BEiT, DeiT-III

### 4.1 DeiT: Data-Efficient Image Transformers

**Paper**: "Training data-efficient image transformers & distillation through attention" (Touvron et al., 2021)

**Key Problem**: ViT required JFT-300M to work well; DeiT trained only on ImageNet (1.2M images).

**Solution**: Knowledge distillation from a CNN teacher (RegNetY-16GF).

**Distillation Token**: Add a third token [DIST] alongside [CLS]:

```
Tokens: [CLS] [DIST] patch_1 patch_2 ... patch_N
```

The [DIST] token is trained to match the teacher's output:

\[
\mathcal{L}_{total} = (1-\lambda)\mathcal{L}_{CE}(\hat{y}_{cls}, y) + \lambda \cdot \mathcal{L}_{KL}(\hat{y}_{dist}, y_{teacher})
\]

At inference, average predictions from [CLS] and [DIST] tokens.

**Training recipe** (critical for DeiT's success):
- RandAugment + CutMix + Mixup
- Stochastic depth (DropPath)
- Repeated augmentation
- 300 epochs with AdamW
- Cosine learning rate schedule

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, distillation=True):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distillation else None
        
        num_tokens = num_patches + (2 if distillation else 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, dropout=0.0,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes) if distillation else None
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.dist_token is not None:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        x = self.norm(self.encoder(x))
        
        if self.dist_token is not None:
            return self.head(x[:, 0]), self.head_dist(x[:, 1])
        return self.head(x[:, 0])
```

### 4.2 BEiT: BERT Pre-Training of Image Transformers

**Paper**: "BEiT: BERT Pre-Training of Image Transformers" (Bao et al., 2022)

**Key Idea**: Apply BERT-style masked language modeling (MLM) to images. The challenge: what are image "tokens"?

**Solution**: Use a discrete VAE (dVAE) to tokenize images into visual tokens, then predict those tokens for masked patches.

**Pre-training Pipeline**:
1. Tokenize image with dVAE: image → discrete visual tokens (vocabulary of 8192)
2. Mask 40% of patches randomly (blockwise masking preferred)
3. Encode visible patches with ViT
4. Predict original visual tokens for masked positions

\[
\mathcal{L}_{BEiT} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}} \log p(v_i | \mathbf{x}^{\mathcal{M}})\right]
\]

where M is the set of masked positions, v_i is the visual token, x^M is the corrupted image.

### 4.3 DeiT-III: Revisiting ViT Training

**Paper**: "DeiT III: Revenge of the ViT" (Touvron et al., 2022)

Key improvements over original DeiT:
- Simple supervised pre-training (no distillation needed at large scale)
- 3-Augment strategy (grayscale, solarization, Gaussian blur)
- Binary cross-entropy loss instead of cross-entropy
- LayerScale initialization (tiny diagonal scaling after each residual block)
- LAMB optimizer

**LayerScale** (important trick to stabilize training very deep ViTs):

```python
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x * self.gamma

# Usage in TransformerBlock:
class TransformerBlockWithLayerScale(nn.Module):
    def __init__(self, dim, num_heads, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ls1 = LayerScale(dim, init_values)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ls2 = LayerScale(dim, init_values)
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
```

---

## 5. Swin Transformer: Hierarchical Vision

**Paper**: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)

**Motivation**: Standard ViT processes fixed-size patches; no multi-scale features → bad for dense prediction (detection, segmentation). Also, O(N²) attention is slow for high-resolution.

### 5.1 Key Innovations

1. **Hierarchical feature maps**: Like CNN, produce 4 stages with decreasing resolution and increasing channels
2. **Window-based attention**: Attention within non-overlapping local windows (fixed size M×M)
3. **Shifted window attention**: Cross-window connections via alternating shifts

### 5.2 Architecture

| Stage | Resolution | Channels | Layers |
|-------|-----------|---------|--------|
| Patch Partition | H/4 × W/4 | 48 → C | 1 |
| Stage 1 | H/4 × W/4 | C | 2 |
| Stage 2 | H/8 × W/8 | 2C | 2 |
| Stage 3 | H/16 × W/16 | 4C | 6 |
| Stage 4 | H/32 × W/32 | 8C | 2 |

For Swin-B: C=128, so channels are 128→256→512→1024.

### 5.3 Window Self-Attention (W-MSA)

Partition the feature map into non-overlapping M×M windows. Compute attention within each window independently:

- Feature map: H×W×C → (H/M × W/M) windows, each M²×C
- Attention within each window: O(M²·M²) = O(M⁴) per window
- Total: O((H/M)(W/M) · M⁴) = O(HW · M²)
- Since M is fixed (e.g., 7), this is linear in image size: **O(HW)**

Compare to global attention: O((HW)²) — huge difference!

### 5.4 Shifted Window Attention (SW-MSA)

Window attention has no cross-window communication. **Shifted windows** fix this:

In even layers: standard windows starting at (0,0)
In odd layers: shift windows by (M/2, M/2) → windows straddle original boundaries

```
Layer ℓ (W-MSA):          Layer ℓ+1 (SW-MSA):
┌──┬──┐                   ┌──┬──┐
│  │  │                   │──┼──│ (shifted by M/2)
├──┼──┤    →              ├──┼──┤
│  │  │                   │  │  │
└──┴──┘                   └──┴──┘
```

For efficient computation of shifted windows, **cyclic shift** + **attention masking** is used:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """Partition feature map into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows  # [num_windows*B, window_size, window_size, C]

def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)
```

### 5.5 Patch Merging (Downsampling)

Between stages, patch merging reduces spatial resolution by 2× and doubles channels:

```python
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        
        # Downsample 2×: take non-overlapping 2×2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2C]
        return x
```

### 5.6 Swin Variants

| Model | Channels (C) | Layers | Params | ImageNet Top-1 |
|-------|-------------|--------|--------|----------------|
| Swin-T | 96 | 2-2-6-2 | 28M | 81.3% |
| Swin-S | 96 | 2-2-18-2 | 50M | 83.0% |
| Swin-B | 128 | 2-2-18-2 | 88M | 83.5% |
| Swin-L | 192 | 2-2-18-2 | 197M | 86.3% (ImageNet-22K) |
| Swin-V2-G | 192 | huge | ~3B | 90.17% |

---

## 6. More Efficient Architectures

### 6.1 PVT: Pyramid Vision Transformer

**Paper**: "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions" (Wang et al., 2021)

PVT produces multi-scale feature maps like CNN (FPN), suitable for dense prediction:

**Spatial Reduction Attention (SRA)**: Reduce key/value spatial size by factor R before attention:

\[
\text{SRA}(Q, K, V) = \text{Attention}(Q, W_K \cdot SR(K), W_V \cdot SR(V))
\]

where SR(·) is a convolutional layer that reduces spatial resolution by R.

This reduces complexity from O(N²) to O(N · N/R²).

### 6.2 CvT: Convolutional Vision Transformer

**Paper**: "CvT: Introducing Convolutions to Vision Transformers" (Wu et al., 2021)

CvT introduces convolutions into ViT via:
1. **Convolutional Token Embedding**: Overlapping patch convolution (better local context)
2. **Convolutional Projection**: Replace linear Q, K, V projections with depth-wise conv

```python
class ConvProjection(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride=stride, 
                                 padding=kernel_size//2, groups=dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N', C]
        return self.proj(x)
```

### 6.3 CoAtNet: Combining Convolution and Attention

**Paper**: "CoAtNet: Marrying Convolution and Attention for All Data Sizes" (Dai et al., 2021)

CoAtNet stacks convolution (MBConv from MobileNet) and relative attention blocks, finding the optimal ratio:

**Architecture**: C-C-T-T (Conv stages first, Transformer stages later)

The key insight: early layers benefit from convolution (local patterns + translation equivariance), while later layers benefit from attention (global context + flexibility).

**Relative attention** in CoAtNet:
\[
A(x_i, x_j) = \frac{\exp(x_i^T x_j / \sqrt{d} + w_{i-j})}{\sum_k \exp(x_i^T x_k / \sqrt{d} + w_{i-k})}
\]

where w_{i-j} is a learnable scalar based on relative position.

### 6.4 MaxViT: Multi-Axis Attention

**Paper**: "MaxViT: Multi-Axis Vision Transformer" (Tu et al., 2022)

MaxViT uses **block attention** (local) + **grid attention** (global) alternately, achieving global receptive field with linear complexity:

- **Block attention**: Attention within M×M local windows (local, O(M²N))
- **Grid attention**: Attention on a sparse M×M grid sampled from H×W (global, O(N·M²/N) = O(M²))

Total complexity: O(M²N) — linear!

```python
class MaxVitBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.block_attn = WindowAttention(dim, window_size, num_heads)
        self.grid_attn = GridAttention(dim, window_size, num_heads)
        self.mbconv = MBConv(dim)
    
    def forward(self, x):
        # MBConv (local conv processing)
        x = x + self.mbconv(x)
        # Local block attention
        x = x + self.block_attn(x)
        # Global grid attention  
        x = x + self.grid_attn(x)
        return x
```

### 6.5 EfficientViT

**Paper**: "EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention" (Liu et al., 2023)

EfficientViT uses **cascaded group attention (CGA)**: split feature into groups, each group attends at different scales:

```python
class CascadedGroupAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkvs = nn.ModuleList([
            nn.Linear(head_dim, head_dim * 3) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        chunks = x.chunk(self.num_heads, dim=-1)
        
        outputs = []
        prev = None
        for i, (qkv_proj, chunk) in enumerate(zip(self.qkvs, chunks)):
            # Each head also receives output from previous head (cascading)
            if prev is not None:
                chunk = chunk + prev
            qkv = qkv_proj(chunk)
            q, k, v = qkv.chunk(3, dim=-1)
            attn = F.softmax(q @ k.transpose(-2,-1) / (head_dim**0.5), dim=-1)
            out = attn @ v
            outputs.append(out)
            prev = out
        
        return torch.cat(outputs, dim=-1)
```

---

## 7. CLIP and Contrastive Vision-Language Pretraining

**Paper**: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

### 7.1 Architecture

CLIP has two encoders:
- **Image encoder**: ViT (ViT-B/32, ViT-L/14) or ResNet
- **Text encoder**: Transformer (12 layers, 512 dim for B, 768 for L)

Both produce fixed-dimensional embeddings (e.g., 512-d) projected into a shared embedding space.

### 7.2 Contrastive Pre-training

Given a batch of (image, text) pairs {(I_i, T_i)}_{i=1}^N:

1. Encode: f_I = ImageEncoder(I), f_T = TextEncoder(T)
2. Normalize: i = f_I / ||f_I||, t = f_T / ||f_T||
3. Compute similarity matrix: S = (i · t^T) / τ (where τ is a learnable temperature)
4. Symmetric cross-entropy loss (contrastive):

\[
\mathcal{L} = -\frac{1}{2N}\left(\sum_i \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ij}/\tau}} + \sum_i \log \frac{e^{S_{ii}/\tau}}{\sum_j e^{S_{ji}/\tau}}\right)
\]

In-batch negatives: N=32768 pairs per batch → massive set of negatives.

### 7.3 Zero-Shot Classification

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Zero-shot: no task-specific training needed
image = preprocess(Image.open("dog.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a dog", "a cat", "a car", "a house"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)        # [1, 768]
    text_features = model.encode_text(text)            # [4, 768]
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = similarity[0]

labels = ["a dog", "a cat", "a car", "a house"]
for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.2%}")
```

### 7.4 CLIP for Dense Tasks

CLIP can be extended to dense prediction via:
- **CLIP Surgery**: Re-route attention for spatial features
- **MaskCLIP**: Extract dense CLIP features for segmentation
- **LSeg**: Language-driven semantic segmentation

### 7.5 Training Data

CLIP was trained on **WIT (WebImageText)**: 400M image-text pairs scraped from the internet. This massive scale is key to zero-shot generalization.

---

## 8. ALIGN, BLIP, BLIP-2, LLaVA

### 8.1 ALIGN

**Paper**: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision" (Jia et al., 2021, Google)

- Same contrastive approach as CLIP but trained on **1.8 billion** image-text pairs
- Uses **noisy** web data (no filtering → rely on scale)
- EfficientNet image encoder + BERT text encoder
- Showed that scale > data quality for contrastive pretraining

### 8.2 BLIP: Bootstrapping Language-Image Pre-training

**Paper**: "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation" (Li et al., 2022)

BLIP uses a **unified encoder-decoder** model with 3 objectives:
1. **ITC** (Image-Text Contrastive): Like CLIP, pull together matching pairs
2. **ITM** (Image-Text Matching): Binary classification, does text match image?
3. **LM** (Language Modeling): Generate text from image (captioning)

**CapFilt** (Captioning + Filtering): Generate synthetic captions for web images, filter out noisy ones with a trained model → better training data.

### 8.3 BLIP-2: Querying Transformer

**Paper**: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al., 2023)

**Key Innovation**: Bridge frozen image encoder and frozen LLM with a lightweight **Q-Former** (Querying Transformer).

```
Frozen Image Encoder → Q-Former (32 learnable queries) → Frozen LLM
                              ↑
                    Cross-attention to image features
                    Self-attention among queries
```

Q-Former is trained in two stages:
1. **Stage 1**: Learn vision-language alignment (ITC + ITM + LM) with frozen image encoder
2. **Stage 2**: Connect to LLM for language generation

This enables visual reasoning with LLMs at minimal training cost (only Q-Former parameters).

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
                                                        torch_dtype=torch.float16)
model.to("cuda")

image = Image.open("image.jpg")
prompt = "Question: What is in this image? Answer:"
inputs = processor(image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=30)
print(processor.batch_decode(generated_ids, skip_special_tokens=True))
```

### 8.4 LLaVA: Large Language and Vision Assistant

**Paper**: "Visual Instruction Tuning" (Liu et al., 2023)

LLaVA connects a visual encoder (CLIP ViT-L/14) to a large language model (LLaMA/Vicuna) via a simple linear projection:

```
CLIP ViT-L/14 → Linear Projection → LLaMA / Vicuna
     ↑                                     ↑
  Image features                    Language model
                    (concatenate)
```

**Training**: Two-stage instruction tuning
1. **Stage 1**: Pre-align features — freeze LLM, train only projection layer on image-text pairs
2. **Stage 2**: End-to-end fine-tuning on visual instruction data (GPT-4 generated Q&A)

**LLaVA-1.5** improvements:
- MLP connector (2-layer) instead of linear
- Higher resolution (336×336)
- More diverse instruction data

---

## 9. Segment Anything Model (SAM)

**Paper**: "Segment Anything" (Kirillov et al., 2023, Meta)

### 9.1 Architecture

SAM has three components:

**1. Image Encoder** (ViT-H/16, MAE pre-trained):
- Takes 1024×1024 image
- Outputs 64×64 image embeddings (each = 256-dim)
- Runs once per image → cached for efficient multi-prompt inference

**2. Prompt Encoder**:
- **Points** (foreground/background): positional encoding + learned type embedding
- **Boxes**: two corners as embeddings
- **Masks**: convolve with 4× downsampled conv → 256-dim tokens
- **Text**: (not used in SAM v1; added in Grounded SAM)

**3. Mask Decoder**:
- 2-layer transformer decoder
- Output tokens → predict 3 masks (whole, part, subpart) + IoU scores
- Upsampling from 64×64 → 256×256 via transposed convolutions

```python
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamAutomaticMaskGenerator

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")

# Point-based prompting
predictor = SamPredictor(sam)
predictor.set_image(image)  # Encode image once

# Predict masks from point prompt
masks, scores, logits = predictor.predict(
    point_coords=[[500, 375]],   # [x, y] in image coords
    point_labels=[1],             # 1 = foreground, 0 = background
    multimask_output=True         # Return 3 masks
)
# masks: [3, H, W], scores: [3] IoU predictions

# Box prompting
masks, _, _ = predictor.predict(
    box=np.array([100, 100, 400, 400]),  # [x1, y1, x2, y2]
    multimask_output=False
)

# Automatic mask generation (everything mode)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)  # List of dicts with 'segmentation', 'area', etc.
```

### 9.2 Prompt Engineering for SAM

SAM's power lies in its **prompt flexibility**:

```python
# Multiple prompts: combine points and boxes
masks, _, _ = predictor.predict(
    point_coords=np.array([[500, 375], [600, 400]]),
    point_labels=np.array([1, 0]),   # One foreground, one background
    box=np.array([200, 200, 700, 500]),
    multimask_output=False
)

# Iterative refinement: use previous mask as prompt
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    mask_input=logits[np.argmax(scores)][None],  # Feed best previous mask
    multimask_output=False
)
```

### 9.3 SAM 2: Segment Anything in Images and Videos

SAM 2 (2024) extends SAM to video:
- **Memory attention**: Attend to frames from memory bank
- **Memory encoder**: Encode past frame predictions
- **Memory bank**: Store up to 6 recent frames + conditioning frames
- Real-time tracking across video frames

---

## 10. DINO and DINOv2: Self-Supervised ViT Training

### 10.1 DINO: Self-DIstillation with NO labels

**Paper**: "Emerging Properties in Self-Supervised Vision Transformers" (Caron et al., 2021)

**Key Insight**: ViT self-supervised with DINO produces attention maps that look like object segmentation — without any segmentation labels!

**Framework**: Teacher-student with momentum update:
- **Student**: Updated by gradient
- **Teacher**: EMA (exponential moving average) of student
- Same architecture for both

**Multi-crop strategy**:
- 2 global crops (224×224) — both go to student AND teacher
- 8 local crops (96×96) — only go to student
- Student learns from teacher's predictions on global views

**Loss**: Cross-entropy between student and teacher softmax outputs:

\[
\mathcal{L}_{DINO} = -\sum_x \sum_{y \neq x} P_t(x) \log P_s(y)
\]

where x is global crop, y is any crop, P_t/P_s are teacher/student probability distributions.

**Collapse prevention** (critical!):
- **Centering**: Subtract running mean from teacher output (prevents mode collapse)
- **Sharpening**: Low temperature for teacher (τ_t=0.04), higher for student (τ_s=0.1)

```python
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.ncrops = ncrops
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_output, teacher_output):
        student_out = student_output / self.student_temp
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()  # No gradient through teacher
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # Skip same view
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)
```

### 10.2 DINOv2: Curated Data + iBOT

**Paper**: "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)

**Improvements over DINO**:

1. **Curated data**: LVD-142M — 142M high-quality images from curated sources + filtered web data. Uses clustering + deduplication. Data quality > raw web scale for SSL.

2. **iBOT (image BERT pre-training with Online Tokenizer)**: Combined with DINO — adds masked image modeling objective alongside contrastive:
   - Mask random patches of student input
   - Student predicts masked patches using teacher as online tokenizer
   - Aligns patch-level features with global features; improves dense prediction tasks

3. **KoLeo regularizer** (Kernel Loretz Embedding Optimization): \( \mathcal{L}_{KoLeo} = -\log \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{\|z_i - z_j\|_2} \) — pushes embeddings apart in hypersphere, preventing collapse without needing negatives. Helps in low-contrast regions.

4. **Register tokens**: Extra non-patch tokens added per layer. Fixes artifact attention patterns where some patch tokens attend only to a few locations. DINOv2 found that very deep ViTs develop "background" tokens; registers absorb this, improving foreground feature quality.

5. **Short high-resolution fine-tuning**: Train at 518×518 for better spatial understanding; critical for dense tasks like segmentation.

**DINOv2 ViT-g/14** achieves SOTA on:
- Linear probe: 86.5% ImageNet
- Depth estimation (without finetuning)
- Semantic segmentation (with simple linear head)

```python
import torch
# Using DINOv2 from HuggingFace
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Last hidden states: [B, 1+N, D] (CLS + N patches)
patch_features = outputs.last_hidden_state[:, 1:, :]  # [B, N, 1024]
cls_feature = outputs.last_hidden_state[:, 0, :]      # [B, 1024]
```

---

## 11. Detection Transformers: DETR Family

### 11.1 DETR: Detection Transformer

**Paper**: "End-to-End Object Detection with Transformers" (Carion et al., 2020)

**Key innovations**:
1. No anchors, no NMS (Non-Maximum Suppression) → end-to-end detection
2. N learnable object queries → N predictions
3. Bipartite matching loss (Hungarian algorithm)

**Architecture**:
```
Image → ResNet backbone → Flatten + Pos Encoding → Transformer Encoder
                                                          ↓
N Object Queries → Transformer Decoder (cross-attend to encoder) → FFN → (class, bbox) × N
```

**Bipartite Matching Loss**:
For N predictions and M ground truth boxes (M < N), find optimal 1-1 matching σ:

\[
\hat{\sigma} = \arg\min_\sigma \sum_{i=1}^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})
\]

\[
\mathcal{L}_{match} = -\mathbb{1}[c_i \neq \varnothing] \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}[c_i \neq \varnothing] \mathcal{L}_{box}(b_i, \hat{b}_{\sigma(i)})
\]

Then compute final loss only on matched pairs. No duplicates → no NMS needed!

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process predictions
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with score {score:.3f}")
    print(f"  Box: {[round(c, 2) for c in box.tolist()]}")
```

### 11.2 Deformable DETR

**Paper**: "Deformable DETR: Deformable Transformers for End-to-End Object Detection" (Zhu et al., 2021)

**Problem with DETR**: 
- Very slow to converge (500 epochs vs RCNN's 36)
- Attention is uniform — every position attends to all positions

**Solution**: **Deformable attention** — each query attends to only K (4) learned offsets:

\[
\text{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^M W_m \sum_{k=1}^K A_{mqk} \cdot W_m' x(p_q + \Delta p_{mqk})
\]

where M=8 heads, K=4 sampling points per head, A_{mqk} are attention weights, Δp_{mqk} are learned offsets.

This reduces complexity from O(H²W²) to O(HWK) — massive speedup.

Also uses **multi-scale features** from FPN-like backbone.

### 11.3 DAB-DETR and DN-DETR

**DAB-DETR** ("DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR", 2022):
- Reformulate object queries as dynamic anchor boxes (x, y, w, h)
- Positional queries: anchor box → positional embedding
- Anchor updates during decoding (refine position layer by layer)

**DN-DETR** ("DN-DETR: Accelerate DETR Training by Introducing Query DeNoising", 2022):
- Add **denoising training**: create noisy versions of ground truth boxes as extra queries
- Train to reconstruct original boxes from noisy queries
- Helps DETR learn matching faster (500 → 50 epochs!)

### 11.4 DINO-DETR

**Paper**: "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection" (Zhang et al., 2023)

DINO-DETR (not to be confused with DINO self-supervised learning) combines:
- Dynamic anchor boxes (DAB-DETR)
- Denoising training (DN-DETR)
- **Contrastive denoising**: negative DN queries (noisy boxes from different objects)
- **Mixed query selection**: Initialize content queries from encoder output

Achieves 63.3 AP on COCO with 48M params — SOTA single-model at the time.

---

## 12. Grounding DINO and GroundedSAM

### 12.1 Grounding DINO

**Paper**: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (Liu et al., 2023)

Grounding DINO extends DINO-DETR to **open-vocabulary detection** — detect any object described in natural language.

**Architecture**:
- DINO-DETR backbone + Text backbone (BERT)
- Feature Enhancer: bidirectional cross-attention between image and text features
- Language-guided query selection: anchor queries selected based on text relevance
- Cross-modality decoder: queries cross-attend to both image and text

```python
from groundingdino.util.inference import load_model, load_image, predict

model = load_model("groundingdino_swint_ogc.pth", "GroundingDINO_SwinT_OGC.py")
image_source, image = load_image("image.jpg")

# Detect anything described in text
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="dog . cat . car",  # Period-separated categories
    box_threshold=0.3,
    text_threshold=0.25
)
```

### 12.2 GroundedSAM

Combines Grounding DINO (open-vocabulary detection) with SAM (precise segmentation):

```python
# Step 1: Detect with Grounding DINO → bounding boxes
# Step 2: Use boxes as prompts for SAM → precise masks

boxes, _, phrases = predict(grounding_model, image, caption="person . car")

# Convert boxes to SAM format and predict masks
predictor.set_image(image_np)
boxes_tensor = torch.tensor(boxes).to(device)

masks, scores, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=boxes_tensor,
    multimask_output=False,
)
# masks: [N_detections, 1, H, W]
```

---

## 13. Masked Image Modeling

### 13.1 MAE: Masked Autoencoders

**Paper**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)

**Key design choices**:
- Mask **75%** of patches (much higher than BERT's 15%)
- Encoder only processes **visible** patches (25%) → efficient!
- Lightweight decoder reconstructs masked patches
- Loss: MSE in pixel space on masked patches only

**Why high masking ratio?**:
- Images have heavy spatial redundancy (adjacent pixels are similar)
- 15% masking is too easy — model can copy neighbors via local interpolation
- 75% forces semantic understanding; the model must infer object parts, structure, and context
- Ablation: 50% masking gives weaker features; 90% is too hard and hurts performance
- Unlike language, image semantics are spatially distributed — high masking creates a harder pretext task

**Why asymmetric encoder-decoder?**:
- Encoder (ViT-L) processes only 25% of patches → 4× faster than full sequence
- Decoder operates on a different token set (visible + mask tokens) — lightweight 8-layer MLP suffices
- Small decoder only used during pre-training; discarded for downstream tasks
- Pre-training compute: ~3× faster than supervised ViT training
- The bottleneck (encoder) learns compressed representations; decoder merely reconstructs

**MAE vs BEiT vs SimMIM**: MAE uses raw pixels (no discrete tokenizer); BEiT uses dVAE tokens (semantic targets); SimMIM uses Swin + simple linear decoder. MAE's simplicity often wins on efficiency and downstream transfer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_embed_dim=1024, encoder_depth=24, encoder_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        
        # Encoder
        self.patch_embed = nn.Conv2d(in_channels, encoder_embed_dim, 
                                     patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_embed_dim),
                                      requires_grad=False)  # Fixed sinusoidal
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(encoder_embed_dim, encoder_heads) 
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)  # Reconstruct patches
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embeddings with sin/cos
        pos_embed = self._get_sincos_pos_embed(self.pos_embed.shape[-1], 
                                                int(self.pos_embed.shape[-2]**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def _get_sincos_pos_embed(self, embed_dim, grid_size):
        import numpy as np
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])
        
        omega = np.arange(embed_dim // 4, dtype=np.float32) / (embed_dim // 4)
        omega = 1. / 10000 ** omega
        
        pos_h = grid[0].reshape(-1)[:, np.newaxis] * omega[np.newaxis, :]
        pos_w = grid[1].reshape(-1)[:, np.newaxis] * omega[np.newaxis, :]
        emb = np.concatenate([np.sin(pos_h), np.cos(pos_h), np.sin(pos_w), np.cos(pos_w)], axis=1)
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)  # CLS + patches
    
    def random_masking(self, x, mask_ratio):
        """Randomly mask patches, return visible subset and indices."""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)  # Uniform noise
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def patchify(self, x):
        """Convert image to patches."""
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.reshape(B, C, H//P, P, W//P, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H//P)*(W//P), C*P*P)
        return x
    
    def forward_encoder(self, x, mask_ratio):
        # Patch embed: [B, C, H, W] → [B, N, D]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:, :]  # Add pos embed (no CLS yet)
        
        # Mask
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Prepend CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Encode visible patches
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        # Append mask tokens to restore full sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No CLS
        x_ = torch.gather(x_, dim=1, 
                          index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Prepend CLS
        
        x = x + self.decoder_pos_embed
        
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        pred = pred[:, 1:, :]  # Remove CLS token
        return pred
    
    def forward(self, x, mask_ratio=None):
        mask_ratio = mask_ratio or self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        
        target = self.patchify(x)
        # Normalize target per patch (optional but improves results)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
        
        loss = (pred - target) ** 2
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()  # Only on masked patches
        return loss, pred, mask
```

### 13.2 SimMIM

**Paper**: "SimMIM: A Simple Framework for Masked Image Modeling" (Xie et al., 2022)

Simpler than MAE:
- Use **Swin Transformer** backbone (hierarchical)
- Mask patches with learnable mask token (replaced in input)
- Decoder: simple 1-layer linear projection
- Reconstruct **raw pixel values** (no tokenizer like BEiT)

```python
class SimMIM(nn.Module):
    def __init__(self, encoder, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder  # Swin Transformer
        self.mask_ratio = mask_ratio
        
        encoder_stride = 32  # Feature map stride
        dim = encoder.num_features[-1]
        
        # Simple linear decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, encoder_stride**2 * 3, 1),  # Patch-to-pixel
            nn.PixelShuffle(encoder_stride)             # Rearrange to image
        )
        self.mask_token = nn.Parameter(torch.zeros(1, encoder.embed_dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Create mask: randomly zero out patch tokens
        num_patches = (H // 4) * (W // 4)
        mask = torch.zeros(B, num_patches, device=x.device)
        num_mask = int(num_patches * self.mask_ratio)
        for i in range(B):
            idx = torch.randperm(num_patches)[:num_mask]
            mask[i, idx] = 1
        
        features = self.encoder(x, mask)  # Encoder handles masking internally
        pred = self.decoder(features)
        
        # Loss: L1 on masked pixels
        loss = F.l1_loss(pred, x, reduction='none')
        mask_upsampled = mask.reshape(B, H//4, W//4).repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
        loss = (loss * mask_upsampled.unsqueeze(1)).sum() / (mask_upsampled.sum() * C)
        return loss
```

### 13.3 data2vec: Unified Masked Prediction

**Paper**: "data2vec: A General Framework for Self-supervised Learning in Speech, Vision, and Language" (Baevski et al., 2022)

**Key difference from MAE/BEiT**: Target is not pixels or discrete tokens but **contextualized representations from a teacher network**:

\[
\mathcal{L} = -\frac{1}{|M|}\sum_{t \in M} \left\|\hat{y}_t - y_t\right\|^2_\beta
\]

where y_t = mean of top-K teacher layers' output at position t, and ‖·‖_β is smooth-L1 loss.

This makes data2vec **modality-agnostic** — same framework for vision, speech, and text, just change the encoder and masking strategy.

---

## 14. Video Transformers

### 14.1 ViViT: Video Vision Transformer

**Paper**: "ViViT: A Video Vision Transformer" (Arnab et al., 2021)

Extends ViT to video by treating video as a sequence of frame patches.

**Tokenization**: Extract spatio-temporal tubes of size t×h×w (e.g., 2×16×16):
- Tubelet embedding: single 3D conv covering multiple frames
- Or factorized: embed frames independently, then temporal attention

**Model variants** (most to least expensive):
1. **Model 1 (Spatiotemporal)**: Full attention over all tokens (T·H·W tokens)
2. **Model 2 (Factorized Encoder)**: Separate spatial ViT + temporal transformer
3. **Model 3 (Factorized Self-Attention)**: Spatial then temporal attention in each layer

```python
class ViViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_frames=8,
                 num_classes=400, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        self.num_frames = num_frames
        num_spatial_patches = (image_size // patch_size) ** 2
        
        # Tubelet embedding (2 frames × 16×16 spatial)
        self.tubelet_embed = nn.Conv3d(
            3, embed_dim, 
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size)
        )
        
        T = num_frames // 2
        N = num_spatial_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + T * N, embed_dim))
        
        # Factorized encoder: spatial blocks + temporal blocks
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        x = self.tubelet_embed(x)  # [B, D, T//2, H//P, W//P]
        T_, H_, W_ = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # [B, T'*N, D]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Spatial attention (reshape to treat each frame independently)
        N = H_ * W_
        x_spatial = x[:, 1:].reshape(B * T_, N, -1)  # [B*T', N, D]
        for block in self.spatial_blocks:
            x_spatial = block(x_spatial)
        x_spatial = x_spatial.reshape(B, T_, N, -1)
        
        # Temporal attention (treat each patch position independently)
        x_temporal = x_spatial.permute(0, 2, 1, 3).reshape(B * N, T_, -1)  # [B*N, T', D]
        for block in self.temporal_blocks:
            x_temporal = block(x_temporal)
        x_temporal = x_temporal.reshape(B, N, T_, -1).mean(dim=1)  # [B, T', D]
        
        # Pool and classify
        x_out = self.norm(x_temporal.mean(dim=1))
        return self.head(x_out)
```

### 14.2 Video Swin Transformer

**Paper**: "Video Swin Transformer" (Liu et al., 2022)

Extends Swin to 3D by computing attention in 3D spatio-temporal windows:
- 3D window size: (D×M×M) = (8×7×7) tokens
- Shifted windows in space AND time
- Temporal modeling at low cost

### 14.3 TimeSFormer: Divided Space-Time Attention

**Paper**: "Is Space-Time Attention All You Need for Video Understanding?" (Bertasius et al., 2021)

TimeSFormer factorizes space-time attention:
1. **Temporal attention**: Each patch attends to same spatial position across all frames
2. **Spatial attention**: Each frame patch attends to all patches in same frame

This is efficient: O(T·N + N·T) instead of O((TN)²).

```
For patch at (t, h, w):
  Temporal attn: attend to (t', h, w) for all t' (same position, different frames)
  Spatial attn: attend to (t, h', w') for all h', w' (all positions, same frame)
```

---

## 15. Vision-Language Models (VLMs)

### 15.1 Architecture Taxonomy

| Architecture | Image Encoder | Language Model | Bridge | Examples |
|-------------|---------------|----------------|--------|---------|
| Contrastive | ViT | Text Transformer | None (shared space) | CLIP, ALIGN |
| Cross-attention | ViT | Decoder | Cross-attention layers | Flamingo |
| Q-Former | ViT (frozen) | LLM (frozen) | Q-Former | BLIP-2 |
| Linear projection | ViT (frozen) | LLM | MLP | LLaVA |
| Full fusion | ViT + LLM | Jointly trained | Dense | PaLI |

### 15.2 Flamingo: Few-Shot VLM

**Architecture**: Interleave pretrained LM (Chinchilla) with cross-attention layers that attend to visual tokens. Freeze both LM and visual encoder; only train cross-attention "bridges".

### 15.3 GPT-4V, Gemini: Multimodal LLMs

Modern trend: Train large multimodal models end-to-end or with large-scale instruction tuning:
- **GPT-4V**: Proprietary; ViT backbone + GPT-4 style training
- **Gemini**: Natively multimodal — trained on image+text+audio together
- **Claude 3**: Constitutional AI + multimodal instruction tuning

### 15.4 VLM Evaluation

| Task | Metric | Benchmark |
|------|--------|-----------|
| VQA | Accuracy | VQA v2, OK-VQA |
| Image captioning | CIDEr, BLEU | COCO Captions |
| Visual reasoning | Accuracy | GQA, NLVR2 |
| Zero-shot retrieval | R@1 | MSCOCO, Flickr30K |
| OCR / doc understanding | Accuracy | TextVQA, DocVQA |

---

## 16. Pre-training vs Fine-tuning Strategies

### 16.1 Pre-training Objectives Summary

| Method | Signal | Data Needed | Typical Accuracy (ViT-B, ImageNet) |
|--------|--------|-------------|-----------------------------------|
| Supervised (JFT-300M) | Labels | 300M labeled | 86.9% |
| DINO | Self-distillation | 1M unlabeled | 81.1% (linear probe) |
| MAE | Pixel reconstruction | 1M unlabeled | 83.1% (fine-tuned) |
| BEiT | Token prediction | 1M unlabeled | 83.2% (fine-tuned) |
| DINOv2 | Distillation + iBOT | 142M curated | 86.5% (linear probe) |

### 16.2 Fine-tuning Strategies

**Full fine-tuning**: Train all parameters on downstream task.
- Best performance, requires sufficient data
- Use lower LR for pre-trained layers (lr_decay per layer)

```python
# Layer-wise LR decay (critical for ViT fine-tuning)
def build_optimizer_with_layer_decay(model, lr, layer_decay=0.75, weight_decay=0.05):
    num_layers = model.depth + 1  # +1 for embedding
    
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine layer ID
        if 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
            layer_id = 0
        elif 'blocks' in name:
            layer_id = int(name.split('blocks.')[1].split('.')[0]) + 1
        else:
            layer_id = num_layers
        
        param_groups.append({
            'params': [param],
            'lr': lr * layer_scales[layer_id],
            'weight_decay': weight_decay if param.dim() >= 2 else 0.0
        })
    
    return torch.optim.AdamW(param_groups)
```

**Linear probe**: Freeze backbone, train only head. Fast, good baseline.

**Adapter fine-tuning**: Add small adapter modules; only train adapters.

```python
class AdapterBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)  # Initialize to near-identity
    
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
```

**LoRA**: Low-rank adaptation of attention weights.

```python
class LoRAAttention(nn.Module):
    def __init__(self, original_attn, rank=4, alpha=16):
        super().__init__()
        self.original_attn = original_attn
        dim = original_attn.embed_dim
        
        self.lora_A_q = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.lora_B_q = nn.Parameter(torch.zeros(rank, dim))
        self.scale = alpha / rank
        
        # Freeze original parameters
        for p in original_attn.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        # Original attention
        out, _ = self.original_attn(x, x, x)
        
        # LoRA delta for Q
        lora_delta = (x @ self.lora_A_q @ self.lora_B_q) * self.scale
        return out + lora_delta
```

---

## 17. Efficient ViTs

### 17.1 Pruning

**Attention Head Pruning**: Not all heads are equally important; prune low-importance heads.

```python
def compute_head_importance(model, val_loader, device):
    """Taylor expansion-based head importance scoring."""
    model.eval()
    head_importance = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            head_importance[name] = torch.zeros(module.num_heads)
    
    for batch in val_loader:
        x, y = batch
        out = model(x.to(device))
        loss = F.cross_entropy(out, y.to(device))
        loss.backward()
        
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Gradient × weight as importance score
                head_importance[name] += module.out_proj.weight.grad.abs().mean(dim=0)
    
    return head_importance
```

**Token Pruning**: Progressively remove uninformative tokens during forward pass.

### 17.2 Knowledge Distillation

Train a smaller student ViT to mimic a larger teacher:

```python
class ViTDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft label distillation loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=-1),
            F.softmax(teacher_logits / self.T, dim=-1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

### 17.3 Token Reduction: EViT

**EViT** (Efficient Vision Transformers): Identify and fuse redundant tokens at each layer.

At each layer, compute attention score for each token relative to [CLS]:
- High attention → informative → keep
- Low attention → uninformative → fuse with neighbor (weighted average)

```python
class EViTBlock(nn.Module):
    def __init__(self, dim, num_heads, keep_ratio=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.keep_ratio = keep_ratio
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, N, D = x.shape
        x_norm = self.norm1(x)
        
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Identify informative tokens via CLS attention
        cls_attn = attn_weights[:, :, 0, 1:]  # [B, heads, N-1] (CLS attends to patches)
        cls_attn = cls_attn.mean(dim=1)       # [B, N-1]
        
        num_keep = int((N-1) * self.keep_ratio)
        topk_idx = cls_attn.topk(num_keep, dim=-1).indices  # [B, num_keep]
        
        # Keep top tokens, fuse rest into a single "attentive" token
        kept_tokens = torch.gather(x[:, 1:], dim=1, 
                                   index=topk_idx.unsqueeze(-1).expand(-1, -1, D))
        
        # Fuse discarded tokens: weighted average by CLS attention
        discard_mask = torch.ones(B, N-1, device=x.device, dtype=torch.bool)
        for b in range(B):
            discard_mask[b][topk_idx[b]] = False
        
        # Concatenate: CLS + kept + fused
        x = torch.cat([x[:, :1], kept_tokens], dim=1)
        x = x + self.mlp(self.norm2(x))
        return x
```

### 17.4 Sparse Attention

**Local Attention** (like Swin): O(N·W) instead of O(N²)

**Linformer**: Project K, V to lower dimension: O(N·k) where k << N

**Performer**: Random feature approximation of softmax attention: O(N·d) where d is feature dimension

### 17.5 Quantization

```python
import torch.quantization

# Dynamic quantization (easiest)
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Post-training static quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Calibrate with representative data
torch.quantization.convert(model, inplace=True)
```

---

## 18. Implementation: HuggingFace + timm

### 18.1 Using timm

```python
import timm
import torch
from torch import nn

# List available models
print(timm.list_models('vit*', pretrained=True)[:10])
print(timm.list_models('swin*', pretrained=True)[:10])

# Load pretrained ViT
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Custom number of classes
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)

# Feature extraction (remove classifier)
model_feat = timm.create_model('vit_base_patch16_224', pretrained=True, 
                                num_classes=0, global_pool='')
x = torch.randn(2, 3, 224, 224)
feats = model_feat(x)  # [2, 197, 768] — all tokens including CLS

# Swin Transformer
swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)

# EfficientViT
efficientvit = timm.create_model('efficientvit_b1', pretrained=True)

# Get model info
config = timm.models.model_factory.get_model_config('vit_base_patch16_224')
print(config)

# Data transforms from model
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
```

### 18.2 Using HuggingFace Transformers

```python
from transformers import (
    ViTForImageClassification, ViTImageProcessor,
    SwinForImageClassification, AutoImageProcessor,
    DetrForObjectDetection, DetrImageProcessor
)
from PIL import Image
import torch

# ViT Classification
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
pred = outputs.logits.argmax(-1)
print(model.config.id2label[pred.item()])

# Swin Transformer
swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
swin_model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

# ViT feature extraction
from transformers import ViTModel
vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
outputs = vit(**inputs)
cls_embed = outputs.last_hidden_state[:, 0]  # [B, 768]
patch_embeds = outputs.last_hidden_state[:, 1:]  # [B, 196, 768]

# CLIP
from transformers import CLIPModel, CLIPProcessor
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
inputs = clip_processor(text=["a cat", "a dog"], images=image, return_tensors="pt")
outputs = clip(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)

# DETR Object Detection
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
inputs = detr_processor(images=image, return_tensors="pt")
outputs = detr_model(**inputs)
results = detr_processor.post_process_object_detection(
    outputs, target_sizes=[image.size[::-1]], threshold=0.9
)
```

---

## 19. Full Code: ViT from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# ===========================
# Building Blocks
# ===========================

class PatchEmbedding(nn.Module):
    """Convert image to sequence of patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Equivalent to: flatten patch → linear; implemented as Conv2d for efficiency
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size
        
        x = self.proj(x)        # [B, D, H/P, W/P]
        x = x.flatten(2)        # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)             # Each: [B, heads, N, head_dim]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Aggregate values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn  # Return attn for visualization


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 4
        out_dim = out_dim or in_dim
        
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, 
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, proj_drop)
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_dim, dropout=proj_drop)
        
        # Stochastic depth (DropPath)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights


class StochasticDepth(nn.Module):
    """Drop entire residual branch with probability p."""
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        survival = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.bernoulli(torch.full(shape, survival, device=x.device))
        return x / survival * noise


# ===========================
# Full ViT Model
# ===========================

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0,
                 drop_path_rate=0.1, num_prefix_tokens=1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable tokens and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + num_prefix_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth decay rule (linear from 0 to drop_path_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, 
                           attn_dropout, dropout, dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        ) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def interpolate_pos_embed(self, x, pos_embed):
        """Interpolate positional embeddings for different resolutions."""
        N = x.shape[1] - 1  # Subtract CLS token
        N_orig = pos_embed.shape[1] - 1
        if N == N_orig:
            return pos_embed
        
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        
        orig_size = int(N_orig ** 0.5)
        new_size = int(N ** 0.5)
        
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), 
                                  mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, patch_pos.shape[1])
        
        return torch.cat([cls_pos, patch_pos], dim=1)
    
    def forward_features(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        
        # Add positional embeddings
        pos_embed = self.interpolate_pos_embed(x, self.pos_embed)
        x = self.pos_drop(x + pos_embed)
        
        # Transformer encoder
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)
        
        x = self.norm(x)
        return x, attention_maps
    
    def forward(self, x):
        features, _ = self.forward_features(x)
        cls_output = features[:, 0]  # CLS token
        return self.head(cls_output)
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization."""
        _, attention_maps = self.forward_features(x)
        return attention_maps


# ===========================
# Training
# ===========================

def train_vit(num_epochs=100, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transforms (CIFAR-10 as example)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # ViT-Small for CIFAR-10 (32×32 image, patch size 4)
    model = VisionTransformer(
        img_size=32, patch_size=4, in_channels=3,
        num_classes=10, embed_dim=384, depth=7, num_heads=6,
        mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer: AdamW with cosine LR schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixup augmentation
    def mixup_data(x, y, alpha=0.2):
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[idx]
        y_a, y_b = y, y[idx]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(pred, y_a, y_b, lam):
        return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)
    
    best_acc = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Mixup
            x_mixed, y_a, y_b, lam = mixup_data(x, y)
            
            optimizer.zero_grad()
            logits = model(x_mixed)
            loss = mixup_criterion(logits, y_a, y_b, lam)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_vit.pth')
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Acc: {acc:.2f}% | Best: {best_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return model


# ===========================
# Attention Visualization
# ===========================

def visualize_attention(model, image_tensor, device='cpu'):
    """Visualize self-attention maps from last layer."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        x = image_tensor.unsqueeze(0).to(device)
        _, attention_maps = model.forward_features(x)
    
    # Get last layer attention, mean over heads
    attn = attention_maps[-1][0]  # [heads, N+1, N+1]
    attn = attn.mean(0)            # [N+1, N+1]
    
    # CLS token's attention to patches
    cls_attn = attn[0, 1:]  # [N]
    num_patches = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()
    
    # Upsample to image size
    from PIL import Image as PILImage
    attn_img = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (image_tensor.shape[-1], image_tensor.shape[-2]), PILImage.BILINEAR
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Original image
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map overlay
    axes[1].imshow(img_np)
    axes[1].imshow(np.array(attn_img), alpha=0.6, cmap='hot')
    axes[1].set_title('CLS Attention Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()
```

---

## 20. Transfer Learning with Swin Transformer

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

def swin_transfer_learning(data_dir, num_classes, epochs=30, batch_size=32, lr=1e-4):
    """Fine-tune Swin Transformer on custom dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained Swin-Base
    model = timm.create_model(
        'swin_base_patch4_window7_224', 
        pretrained=True, 
        num_classes=num_classes
    )
    
    # Check data config for model
    data_config = timm.data.resolve_model_data_config(model)
    
    # Transforms
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = model.to(device)
    
    # Layer-wise learning rate decay
    no_decay = ['bias', 'norm']
    layer_decay = 0.75
    num_layers = 12  # Swin-Base layers
    
    param_groups = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue
        
        # Determine layer number for decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = 0.05
        
        if 'layers.0' in name:
            layer_scale = layer_decay ** (num_layers - 1)
        elif 'layers.1' in name:
            layer_scale = layer_decay ** (num_layers - 4)
        elif 'layers.2' in name:
            layer_scale = layer_decay ** (num_layers - 10)
        elif 'layers.3' in name:
            layer_scale = layer_decay ** (num_layers - 12)
        else:
            layer_scale = 1.0
        
        param_groups.append({
            'params': [param],
            'lr': lr * layer_scale,
            'weight_decay': wd
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Warmup + cosine schedule
    def get_lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision
    
    best_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'acc': acc}, 
                       'best_swin.pth')
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    return model


# ==========================
# Linear Probe Evaluation
# ==========================

def linear_probe_evaluation(backbone, train_loader, val_loader, embed_dim, num_classes, epochs=30):
    """Evaluate representation quality with linear probe."""
    device = next(backbone.parameters()).device
    
    # Freeze backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Extract features
    def extract_features(loader):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                f = backbone(x)  # [B, D]
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats), torch.cat(labels)
    
    print("Extracting features...")
    train_feats, train_labels = extract_features(train_loader)
    val_feats, val_labels = extract_features(val_loader)
    
    # Train linear classifier
    classifier = nn.Linear(embed_dim, num_classes)
    optimizer = torch.optim.LBFGS(classifier.parameters(), lr=0.1, max_iter=100)
    
    train_feats_norm = F.normalize(train_feats, dim=-1)  # L2-normalize
    val_feats_norm = F.normalize(val_feats, dim=-1)
    
    def closure():
        optimizer.zero_grad()
        logits = classifier(train_feats_norm)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    # Evaluate
    with torch.no_grad():
        val_logits = classifier(val_feats_norm)
        acc = (val_logits.argmax(1) == val_labels).float().mean().item()
    
    print(f"Linear Probe Accuracy: {acc*100:.2f}%")
    return acc
```

---

## 21. Best Practices and Research Insights

### 21.1 Training Recipes

| Component | Recommendation |
|-----------|---------------|
| Optimizer | AdamW (weight_decay=0.05) |
| LR schedule | Warmup (5 epochs) + Cosine decay |
| Learning rate | 1e-3 for scratch, 1e-4 for fine-tuning |
| Batch size | 1024+ for pretraining, 256 for finetuning |
| Drop path | 0.1-0.3 (higher for larger models) |
| Weight decay | 0.05 (backbone), 0 (norm/bias) |
| Gradient clip | 1.0 |
| Mixed precision | FP16/BF16 for training |
| Resolution | Train at 224, fine-tune at 384 or 518 |
| Augmentation | RandAugment + CutMix + Mixup + Random Erasing |
| Label smoothing | 0.1 |
| Layer-wise LR decay | 0.65-0.85 (lower = more decay) |

### 21.2 Common Pitfalls

1. **Training from scratch on small data**: ViT needs 1M+ images without strong regularization. Use DeiT recipe or SSL pre-training.

2. **Using wrong positional encoding for fine-tuning resolution**: Always interpolate position embeddings when changing resolution (e.g., 224→384). Missing interpolation causes performance collapse.

3. **Not freezing during linear probe**: Must freeze backbone completely; even one trainable layer invalidates linear probe results and inflates reported representation quality.

4. **Skipping warmup**: ViT training is unstable without LR warmup, especially for large models. Use 5–10% of steps for linear warmup.

5. **No gradient clipping**: Can cause training instability; clip at 1.0. Essential for contrastive methods (CLIP, DINO).

6. **Wrong attention mask for padding**: In variable-length batches, proper attention masking is critical. Padding tokens must not attend to or be attended by real tokens.

7. **Patch size vs resolution mismatch**: Fine-tuning at 384 with patch_size=16 → 24×24 patches. If positional embeddings were trained for 14×14, interpolation may distort spatial structure. Prefer models trained at target resolution or with 2D relative position encodings.

8. **Oversizing the decoder in MAE**: A heavy decoder during pre-training can cause the encoder to learn shallow features. Keep decoder 8× smaller than encoder.

9. **Ignoring label smoothing for fine-tuning**: ViT fine-tuning benefits from label_smoothing=0.1; hard targets can cause overfitting on small datasets.

10. **Batch size too small for contrastive pre-training**: CLIP/DINO need large batches (4K–32K) for sufficient negatives. Use gradient accumulation or memory bank if GPU memory is limited.

### 21.3 Research Directions (2024-2025)

- **Diffusion ViTs**: DiT (Diffusion Transformer) for image generation
- **Mamba-Vision**: Hybrid SSM + Attention for vision
- **FlashAttention**: IO-aware exact attention, 2-4× speedup
- **Native resolution ViTs**: Process images at arbitrary resolution without re-training
- **Token merging (ToMe)**: Merge similar tokens for 2× speedup without accuracy loss
- **Register tokens**: Add extra tokens to prevent artifact attention patterns (DINOv2 finding)

### 21.4 Model Selection Guide

| Task | Recommended Model | Why |
|------|------------------|-----|
| ImageNet classification | DINOv2 ViT-L/14 | Best linear probe accuracy |
| Object detection | Swin-L + DINO-DETR | Hierarchical features + best AP |
| Semantic segmentation | Mask2Former + Swin | State-of-the-art on ADE20K |
| Video understanding | Video Swin-B | Efficient spatio-temporal modeling |
| Zero-shot classification | CLIP ViT-L/14 | Best CLIP variant |
| VQA / Image captioning | BLIP-2 / LLaVA-1.5 | Efficient VLMs |
| Open-vocabulary detection | Grounding DINO | Open vocabulary |
| Instance segmentation | SAM + Grounding DINO | Any object, any prompt |
| Efficient inference | EfficientViT | Best accuracy/speed tradeoff |

---

## Summary Table

| Model | Year | Key Innovation | Best Use Case |
|-------|------|---------------|---------------|
| ViT | 2020 | Image patches as tokens | Large-scale classification |
| DeiT | 2021 | Knowledge distillation | Data-efficient ViT training |
| Swin | 2021 | Shifted window attention | Dense prediction |
| MAE | 2021 | 75% masking, pixel reconstruction | SSL pre-training |
| DINO | 2021 | Self-distillation, no labels | Feature learning |
| BEiT | 2022 | BERT for images (discrete tokens) | Masked image modeling |
| CLIP | 2021 | Contrastive image-text | Zero-shot classification |
| SAM | 2023 | Promptable segmentation | Universal segmentation |
| DINOv2 | 2023 | Curated data + iBOT | Best SSL features |
| DINO-DETR | 2023 | DN + DAB + contrastive | SOTA detection |
| LLaVA | 2023 | Visual instruction tuning | Multimodal chat |
| BLIP-2 | 2023 | Q-Former bridge to LLM | Efficient VLM |

**Key Libraries**: `transformers`, `timm`, `segment-anything`, `groundingdino`, `open_clip`, `mmsegmentation`, `mmdetection`

---

## References

| Paper | Year | Key Contribution |
|-------|------|------------------|
| An Image is Worth 16x16 Words (ViT) | Dosovitskiy et al., 2020 | Patch-based pure Transformer for images |
| Training data-efficient image transformers (DeiT) | Touvron et al., 2021 | Distillation, ImageNet-only training |
| BEiT: BERT Pre-Training of Image Transformers | Bao et al., 2022 | Masked image modeling with dVAE |
| DeiT III: Revenge of the ViT | Touvron et al., 2022 | Simple supervised recipe, LayerScale |
| Swin Transformer | Liu et al., 2021 | Hierarchical, shifted-window attention |
| Masked Autoencoders Are Scalable Vision Learners (MAE) | He et al., 2021 | 75% masking, asymmetric encoder-decoder |
| Emerging Properties in Self-Supervised ViTs (DINO) | Caron et al., 2021 | Self-distillation, object discovery |
| DINOv2: Learning Robust Visual Features | Oquab et al., 2023 | Curated data, iBOT, registers |
| Learning Transferable Visual Models (CLIP) | Radford et al., 2021 | Contrastive image-text pretraining |
| Segment Anything (SAM) | Kirillov et al., 2023 | Promptable segmentation |
| End-to-End Object Detection with Transformers (DETR) | Carion et al., 2020 | Bipartite matching, no NMS |
| BLIP-2: Bootstrapping with Frozen Encoders | Li et al., 2023 | Q-Former bridge to LLM |
| LLaVA: Visual Instruction Tuning | Liu et al., 2023 | Simple linear projection to LLM |
