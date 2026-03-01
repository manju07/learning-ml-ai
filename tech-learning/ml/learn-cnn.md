# Convolutional Neural Networks: Comprehensive Guide from Fundamentals to Advanced

## Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
2. [The Convolution Operation — Full Mathematics](#the-convolution-operation--full-mathematics)
3. [Padding, Stride, Dilation, and Output Size](#padding-stride-dilation-and-output-size)
4. [Feature Maps and Receptive Field](#feature-maps-and-receptive-field)
5. [Pooling Layers](#pooling-layers)
6. [1D, 2D, and 3D Convolutions](#1d-2d-and-3d-convolutions)
7. [Depthwise Separable Convolutions](#depthwise-separable-convolutions)
8. [Transposed Convolution (Deconvolution)](#transposed-convolution-deconvolution)
9. [CNN Architectures — In Depth](#cnn-architectures--in-depth)
10. [Feature Pyramid Networks (FPN)](#feature-pyramid-networks-fpn)
11. [Object Detection](#object-detection)
12. [Semantic Segmentation](#semantic-segmentation)
13. [Instance Segmentation: Mask R-CNN](#instance-segmentation-mask-r-cnn)
14. [Common Pitfalls and Debugging](#common-pitfalls-and-debugging)
15. [Production Deployment Notes](#production-deployment-notes)
16. [Image Classification Pipeline](#image-classification-pipeline)
17. [Full PyTorch Code Examples](#full-pytorch-code-examples)

---

## Introduction and Motivation

### Why Not Fully Connected Networks for Images?

Consider a modest 256×256 RGB image:
- Pixel count: 256 × 256 × 3 = **196,608 input values**
- A single fully connected layer of size 1024 requires: 196,608 × 1024 = **201 million parameters**
- This is expensive, prone to overfitting, and ignores spatial structure

### CNN Advantages

**1. Parameter Sharing**: A single 3×3 filter is applied across the entire image. A filter that detects a horizontal edge works everywhere — not just at position (10, 20).

**2. Local Connectivity**: Each neuron connects only to a small local region (receptive field), capturing local patterns (edges, textures).

**3. Translation Equivariance**: If an object moves in the input, the feature map shifts correspondingly.
\[
f(T(x)) = T(f(x)) \quad \text{(equivariance)}
\]

**4. Hierarchical Feature Learning**: 
- Early layers: edges, corners, blobs
- Middle layers: textures, patterns, parts
- Late layers: objects, semantic concepts

---

## The Convolution Operation — Full Mathematics

### Discrete 2D Cross-Correlation (What CNNs Actually Compute)

Despite being called "convolution," CNNs compute *cross-correlation* (convolution without flipping the kernel):

\[
(I \star K)(i, j) = \sum_{m=0}^{k_H - 1} \sum_{n=0}^{k_W - 1} I(i + m,\, j + n) \cdot K(m, n)
\]

Where:
- \( I \in \mathbb{R}^{H \times W} \): input feature map
- \( K \in \mathbb{R}^{k_H \times k_W} \): kernel / filter
- \( (i, j) \): output position

### Multi-Channel Convolution

For an input with \( C_{in} \) channels and \( C_{out} \) output channels (filters):

\[
\text{Output}[:, i, j] = \sum_{c=1}^{C_{in}} \sum_{m=0}^{k_H-1} \sum_{n=0}^{k_W-1} \text{Input}[c, i \cdot s + m,\, j \cdot s + n] \cdot K[c, m, n] + b
\]

Each of the \( C_{out} \) filters produces one output channel. Total parameters:

\[
\text{params} = C_{out} \times (C_{in} \times k_H \times k_W + 1)
\]

### Manual Convolution Implementation

```python
import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_naive(input_map: np.ndarray, kernel: np.ndarray,
                  stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Naive 2D convolution (cross-correlation) for a single-channel input.
    input_map: (H, W)
    kernel: (kH, kW)
    """
    H, W = input_map.shape
    kH, kW = kernel.shape

    if padding > 0:
        input_map = np.pad(input_map, padding, mode="constant", constant_values=0)
        H_pad, W_pad = input_map.shape
    else:
        H_pad, W_pad = H, W

    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            patch = input_map[i*stride:i*stride+kH, j*stride:j*stride+kW]
            output[i, j] = np.sum(patch * kernel)

    return output


# ── Classic filter examples ──
image = np.random.randn(8, 8)

# Sobel edge detectors
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
sobel_y = sobel_x.T

# Gaussian blur (5x5)
def gaussian_kernel(size=5, sigma=1.0):
    k = size // 2
    x = np.arange(-k, k+1)
    g = np.exp(-x**2 / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()

gauss = gaussian_kernel(5, 1.0)

edge_x = conv2d_naive(image, sobel_x, padding=1)
edge_y = conv2d_naive(image, sobel_y, padding=1)
blurred = conv2d_naive(image, gauss, padding=2)
magnitude = np.sqrt(edge_x**2 + edge_y**2)

print(f"Input:   {image.shape}")
print(f"Edge X:  {edge_x.shape}")
print(f"Blurred: {blurred.shape}")


# ── Efficient implementation with im2col ──
def im2col(input_map, kH, kW, stride=1, padding=0):
    """Convert input to column matrix for efficient batch convolution."""
    if padding > 0:
        input_map = np.pad(input_map, ((0,0),(0,0),(padding,padding),(padding,padding)))
    N, C, H, W = input_map.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    cols = []
    for r in range(kH):
        for c in range(kW):
            col = input_map[:, :, r:r+out_H*stride:stride, c:c+out_W*stride:stride]
            cols.append(col.reshape(N, C, -1))
    return np.concatenate(cols, axis=1)  # (N, C*kH*kW, out_H*out_W)
```

---

## Padding, Stride, Dilation, and Output Size

### Output Size Formula

\[
H_{out} = \left\lfloor \frac{H_{in} + 2p - d(k-1) - 1}{s} \right\rfloor + 1
\]

Where:
- \( H_{in} \): input height
- \( p \): padding
- \( d \): dilation rate
- \( k \): kernel size
- \( s \): stride

### Padding Modes

**`valid` (no padding)**: Output is smaller than input. \( H_{out} = H_{in} - k + 1 \) (stride=1)

**`same` (zero padding)**: Output has same spatial size as input (when stride=1). Required padding: \( p = (k-1)/2 \)

**Reflection padding**: Pads with mirrored values. Better for images (avoids border artifacts).

### Stride

Controls how many pixels the filter moves at each step. Stride-2 convolutions (used in modern architectures) halve spatial dimensions without pooling, acting as learned downsampling.

\[
H_{out} = \lfloor H_{in} / s \rfloor \quad \text{(with appropriate padding)}
\]

### Dilation (Atrous Convolution)

Inserts \( (d-1) \) zeros between kernel elements, expanding the receptive field **without** increasing parameters:

\[
(I \star_d K)(i,j) = \sum_m \sum_n I(i + d \cdot m, j + d \cdot n) \cdot K(m, n)
\]

Effective kernel size: \( k_{\text{eff}} = d(k-1) + 1 \)

**Used in**: DeepLab (semantic segmentation), WaveNet (audio generation)

```python
import torch
import torch.nn as nn

# Output size computation
def conv_output_size(H_in, k, stride=1, padding=0, dilation=1):
    return (H_in + 2*padding - dilation*(k-1) - 1) // stride + 1

# Examples
print(conv_output_size(224, k=3, stride=1, padding=1))   # 224 (same)
print(conv_output_size(224, k=3, stride=2, padding=1))   # 112 (halved)
print(conv_output_size(28, k=5, stride=1, padding=0))    # 24 (valid)
print(conv_output_size(64, k=3, stride=1, padding=2, dilation=2))  # 64 (dilated same)

# Compare regular vs dilated convolution
x = torch.randn(1, 1, 64, 64)
regular = nn.Conv2d(1, 1, kernel_size=3, padding=1)
dilated_d2 = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2)  # same output size
dilated_d4 = nn.Conv2d(1, 1, kernel_size=3, dilation=4, padding=4)

print("\nDilation comparison (all same spatial output):")
for d, layer in [(1, regular), (2, dilated_d2), (4, dilated_d4)]:
    out = layer(x)
    eff_rf = d * (3 - 1) + 1
    print(f"  Dilation={d}: output={out.shape}, effective RF = {eff_rf}x{eff_rf}")
```

---

## Feature Maps and Receptive Field

### Receptive Field

The receptive field (RF) of a neuron is the region in the original input image that influences its output value.

**Single conv layer** (k=3, stride=1): RF = 3×3  
**Stacked layers**: RF grows with depth

For a stack of \( L \) convolutional layers each with kernel size \( k \) and stride \( s \):
\[
\text{RF}_L = 1 + \sum_{l=1}^{L} (k_l - 1) \cdot \prod_{i=1}^{l-1} s_i
\]

For equal \( k=3, s=1 \): RF after \( L \) layers = \( 2L + 1 \)

**Key insight**: VGG showed that two 3×3 conv layers have the same RF as one 5×5 layer, but with fewer parameters and one more non-linearity:
- Two 3×3: \( 2 \times (3 \times 3 \times C^2) = 18C^2 \) params  
- One 5×5: \( 5 \times 5 \times C^2 = 25C^2 \) params

```python
def compute_receptive_field(layers: list) -> list:
    """
    Compute receptive field at each layer.
    layers: list of (kernel_size, stride, dilation) tuples
    """
    rf = 1
    stride_so_far = 1
    rfs = [1]
    for k, s, d in layers:
        k_eff = d * (k - 1) + 1
        rf += (k_eff - 1) * stride_so_far
        stride_so_far *= s
        rfs.append(rf)
    return rfs

# VGG-like stack of 3x3 convs
layers_vgg = [(3,1,1)] * 10
print("VGG 3x3 receptive fields:", compute_receptive_field(layers_vgg))

# ResNet with stride-2 layers
layers_resnet = [(3,1,1), (3,1,1), (3,2,1), (3,1,1), (3,2,1), (3,1,1)]
print("ResNet receptive fields:", compute_receptive_field(layers_resnet))

# Dilated stack (DeepLab-style)
layers_dilated = [(3,1,1), (3,1,2), (3,1,4), (3,1,8)]
print("Dilated conv receptive fields:", compute_receptive_field(layers_dilated))
```

### Visualizing Feature Maps

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def visualize_feature_maps(model, image_tensor, layer_name, n_maps=16):
    """Hook into a layer to visualize its feature maps."""
    feature_maps = {}

    def hook_fn(module, inp, output):
        feature_maps[layer_name] = output.detach()

    # Find and register hook
    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break

    with torch.no_grad():
        model(image_tensor.unsqueeze(0))

    handle.remove()

    fmaps = feature_maps[layer_name][0]  # (C, H, W)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        if i < min(n_maps, fmaps.size(0)):
            ax.imshow(fmaps[i].cpu().numpy(), cmap="viridis")
        ax.axis("off")
    plt.suptitle(f"Feature Maps: {layer_name}", fontsize=14)
    plt.tight_layout()
    plt.show()
    return fmaps
```

---

## Pooling Layers

### Max Pooling

Selects the maximum value in each pooling window. Provides translation invariance (small shifts in input don't change max output).

\[
\text{MaxPool}(i, j) = \max_{m,n \in R(i,j)} x(m, n)
\]

Gradient: passes gradient only to the element that achieved the maximum.

### Average Pooling

Computes the mean of the window. Preserves more spatial information but is less robust to translation.

\[
\text{AvgPool}(i, j) = \frac{1}{|R|} \sum_{m,n \in R(i,j)} x(m, n)
\]

### Global Average Pooling (GAP)

Reduces each feature map to a single value by averaging over all spatial positions:
\[
\text{GAP}(c) = \frac{1}{H \times W} \sum_{i,j} x_c(i, j)
\]

**Benefits over flatten + FC**:
- Far fewer parameters (no weights between feature maps and classes)
- Enforces correspondence between feature maps and output categories
- More robust to input size changes
- Acts as regularizer

```python
import torch
import torch.nn as nn

# Max vs Average vs Global Average Pooling
x = torch.tensor([[[[1., 2., 3., 4.],
                     [5., 6., 7., 8.],
                     [9., 10., 11., 12.],
                     [13., 14., 15., 16.]]]])

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
gap = nn.AdaptiveAvgPool2d(1)

print("Max Pool 2x2:")
print(max_pool(x).squeeze())  # [[6, 8], [14, 16]]

print("Avg Pool 2x2:")
print(avg_pool(x).squeeze())  # [[3.5, 5.5], [11.5, 13.5]]

print("Global Average Pooling:")
print(gap(x).squeeze())  # 8.5 (mean of all values)

# Stochastic pooling: randomly sample from multinomial distribution
def stochastic_pool2d(x, kernel_size=2):
    """Stochastic pooling: sample proportionally to activations."""
    B, C, H, W = x.shape
    out_H, out_W = H // kernel_size, W // kernel_size
    out = torch.zeros(B, C, out_H, out_W)
    for i in range(out_H):
        for j in range(out_W):
            patch = x[:, :, i*kernel_size:(i+1)*kernel_size,
                        j*kernel_size:(j+1)*kernel_size].reshape(B, C, -1)
            probs = torch.softmax(patch, dim=-1)
            indices = torch.multinomial(probs.reshape(B*C, -1), 1).reshape(B, C)
            out[:, :, i, j] = patch.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return out
```

---

## 1D, 2D, and 3D Convolutions

### 1D Convolutions

For sequential data (audio, time series, text with 1D position):

\[
\text{Output}[c_{out}, t] = \sum_{c_{in}} \sum_{k} \text{Input}[c_{in}, t \cdot s + k] \cdot W[c_{out}, c_{in}, k]
\]

```python
import torch
import torch.nn as nn

# 1D Conv for time series classification
class TCN(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions."""
    def __init__(self, input_size, n_channels, kernel_size=3, n_layers=6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation  # causal padding
            layers.extend([
                nn.utils.weight_norm(nn.Conv1d(
                    input_size if i == 0 else n_channels,
                    n_channels, kernel_size,
                    padding=padding, dilation=dilation
                )),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.network(x)[:, :, -1]  # take last time step (causal)
        return self.fc(out)

# 1D conv for text / NLP
text_cnn = nn.Sequential(
    nn.Embedding(10000, 128),   # vocab_size=10000, embed_dim=128
    # permute to (batch, embed, seq)
    nn.Conv1d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv1d(256, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AdaptiveMaxPool1d(1),    # Global max pooling over time
)

# Demo
x_1d = torch.randn(8, 128, 50)  # batch=8, features=128, seq=50
conv1d = nn.Conv1d(128, 64, kernel_size=3, padding=1)
print("1D Conv output:", conv1d(x_1d).shape)  # (8, 64, 50)
```

### 2D Convolutions (Standard)

Used for images. Input: (N, C, H, W), Kernel: (C_out, C_in, kH, kW)

### 3D Convolutions

For volumetric data (video, medical 3D scans):

```python
# 3D Conv for video understanding
x_3d = torch.randn(2, 3, 16, 112, 112)  # (batch, channels, T, H, W)
conv3d = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1, bias=False)
print("3D Conv output:", conv3d(x_3d).shape)  # (2, 64, 16, 112, 112)

# (2+1)D Convolution (separating spatial and temporal)
class SpatioTemporalConv(nn.Module):
    """Factorized (2+1)D conv = 2D spatial + 1D temporal."""
    def __init__(self, in_ch, out_ch, k_t=3, k_s=3):
        super().__init__()
        mid = (in_ch * out_ch * k_t * k_s * k_s) // (k_s * k_s * in_ch + k_t * out_ch)
        self.spatial = nn.Conv3d(in_ch, mid, (1, k_s, k_s), padding=(0, k_s//2, k_s//2))
        self.temporal = nn.Conv3d(mid, out_ch, (k_t, 1, 1), padding=(k_t//2, 0, 0))
        self.bn1 = nn.BatchNorm3d(mid)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.temporal(self.relu(self.bn1(self.spatial(x))))))
```

---

## Depthwise Separable Convolutions

### Standard vs Depthwise Separable

A standard \( k \times k \) conv with \( C_{in} \to C_{out} \) channels costs:
\[
\text{Standard ops} = k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W
\]

**Depthwise separable** splits this into:
1. **Depthwise conv**: \( k \times k \) conv applied to each channel independently → \( k^2 \cdot C_{in} \) parameters
2. **Pointwise (1×1) conv**: Linear combination across channels → \( C_{in} \cdot C_{out} \)

\[
\text{DSC ops} = k^2 \cdot C_{in} \cdot H \cdot W + C_{in} \cdot C_{out} \cdot H \cdot W
\]

**Reduction ratio**:
\[
\frac{\text{DSC}}{\text{Standard}} = \frac{1}{C_{out}} + \frac{1}{k^2} \approx \frac{1}{k^2}
\]

For k=3: ~**8–9× fewer operations**. Used in MobileNet, Xception, EfficientNet.

```python
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                      padding=1, groups=in_channels, bias=False),  # groups=C_in
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Comparison
standard = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
)
dsc = DepthwiseSeparableConv(64, 128)

print(f"Standard conv params: {count_parameters(standard):,}")
print(f"DSC params:           {count_parameters(dsc):,}")
print(f"Reduction: {count_parameters(standard)/count_parameters(dsc):.1f}x")
```

---

## Transposed Convolution (Deconvolution)

Transposed convolutions (also called fractionally strided convolutions) **increase** spatial resolution — the opposite of a regular convolution. Used in decoders (U-Net, GANs, segmentation).

### How It Works

A transposed conv with stride \( s \) inserts \( (s-1) \) zeros between input values, then applies a regular convolution:

\[
H_{out} = (H_{in} - 1) \cdot s - 2p + k
\]

**Note**: Transposed conv is NOT the inverse of conv (different learned weights), but it IS the gradient of a regular conv w.r.t. its input.

```python
import torch
import torch.nn as nn

# Encode then decode (autoencoder-like)
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # (1,28,28) → (32,14,14)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (32,14,14) → (64,7,7)
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (64,7,7) → (32,14,14)
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # (32,14,14) → (1,28,28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

x = torch.randn(4, 1, 28, 28)
ae = ConvAutoencoder()
out = ae(x)
print(f"Input: {x.shape} → Encoded: {ae.encoder(x).shape} → Decoded: {out.shape}")

# Alternative: bilinear upsampling + conv (no checkerboard artifacts)
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))
```

---

## CNN Architectures — In Depth

### 1. LeNet-5 (LeCun et al., 1998)

The original CNN. Designed for handwritten digit recognition.
- Tanh/sigmoid activations (pre-ReLU era)
- Average pooling
- ~60K parameters

```python
import torch.nn as nn

class LeNet5(nn.Module):
    """Original LeNet-5 for 32x32 grayscale input."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),           # 32→28, 6 maps
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                        # 28→14
            nn.Conv2d(6, 16, kernel_size=5),           # 14→10, 16 maps
            nn.Tanh(),
            nn.AvgPool2d(2, 2),                        # 10→5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### 2. AlexNet (Krizhevsky et al., 2012)

Won ImageNet 2012 by a large margin. Key innovations: ReLU, dropout, data augmentation, GPU training.

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),   # 227→55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                           # 55→27
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                           # 27→13
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),                           # 13→6
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### 3. VGGNet (Simonyan & Zisserman, 2014)

Key insight: Replace large filters with stacked 3×3 filters. Very regular, deep architecture.
- VGG-16: 13 conv + 3 FC layers, ~138M parameters
- VGG-19: 16 conv + 3 FC layers

```python
VGG_CONFIGS = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(self, config_name="VGG16", num_classes=1000, in_channels=3):
        super().__init__()
        cfg = VGG_CONFIGS[config_name]
        layers = []
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### 4. GoogLeNet / Inception (Szegedy et al., 2014)

Introduced the **Inception module**: multi-scale feature extraction at each layer using parallel branches. Uses 1×1 convolutions to reduce channels (bottleneck) before expensive 3×3/5×5 convs.

```python
class InceptionModule(nn.Module):
    """Inception v1 module with 4 parallel branches."""
    def __init__(self, in_ch, f_1x1, f_3x3_r, f_3x3, f_5x5_r, f_5x5, f_pool):
        super().__init__()
        def conv_bn_relu(in_c, out_c, k, **kwargs):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, padding=k//2, bias=False, **kwargs),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            )

        self.branch1 = conv_bn_relu(in_ch, f_1x1, 1)

        self.branch2 = nn.Sequential(
            conv_bn_relu(in_ch, f_3x3_r, 1),
            conv_bn_relu(f_3x3_r, f_3x3, 3),
        )
        self.branch3 = nn.Sequential(
            conv_bn_relu(in_ch, f_5x5_r, 1),
            conv_bn_relu(f_5x5_r, f_5x5, 5),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_bn_relu(in_ch, f_pool, 1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# Inception v3 uses factorized convolutions: 5x5 → two 3x3, 1xN + Nx1
class FactorizedInception(nn.Module):
    def __init__(self, in_ch, out_ch, n=7):
        super().__init__()
        def cbn(ic, oc, k, p):
            return nn.Sequential(
                nn.Conv2d(ic, oc, k, padding=p, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True)
            )
        mid = out_ch // 4
        self.branch1 = nn.Sequential(
            cbn(in_ch, mid, (1, n), (0, n//2)),
            cbn(mid, mid, (n, 1), (n//2, 0)),
        )
        self.branch2 = cbn(in_ch, out_ch - mid, 1, 0)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], dim=1)
```

### 5. ResNet (He et al., 2015–2016)

Residual networks solved the **degradation problem** (deeper networks performing worse than shallower ones). The key: skip connections allow gradients to flow directly to early layers.

**Variants**: ResNet-18, 34, 50, 101, 152, 200, ResNeXt, Wide-ResNet, SE-ResNet

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes),
        )

    def _make_layer(self, block, out_ch, n_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
        layers = [block(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch * block.expansion
        for _ in range(1, n_blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(x)

def resnet18(num_classes=1000): return ResNet(BasicBlock, [2,2,2,2], num_classes)
def resnet50(num_classes=1000): return ResNet(Bottleneck, [3,4,6,3], num_classes)
def resnet101(num_classes=1000): return ResNet(Bottleneck, [3,4,23,3], num_classes)

# Test
model = resnet50(num_classes=1000)
x = torch.randn(2, 3, 224, 224)
print("ResNet-50 output:", model(x).shape)  # (2, 1000)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")  # ~25M
```

### 6. DenseNet (Huang et al., 2017)

Every layer receives input from all previous layers (dense connections). Alleviates vanishing gradients, encourages feature reuse.

```python
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bn_size=4):
        super().__init__()
        inter_ch = bn_size * growth_rate
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch), nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_ch, growth_rate):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_ch + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x): return self.block(x)
```

### 7. EfficientNet (Tan & Le, 2019)

Systematically scales all dimensions (depth, width, resolution) using a **compound scaling coefficient** \( \phi \):

\[
\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi
\]

Subject to: \( \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \), \( \alpha, \beta, \gamma \geq 1 \)

Uses **MBConv** blocks (mobile inverted bottlenecks from MobileNetV2) + **Squeeze-and-Excitation** for channel attention.

```python
class SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        reduced = max(1, in_ch // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, reduced), nn.SiLU(),
            nn.Linear(reduced, in_ch), nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (used in EfficientNet, MobileNetV2)."""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6, se_ratio=0.25):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_skip = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch), nn.SiLU()]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),  # depthwise
            nn.BatchNorm2d(mid_ch), nn.SiLU(),
            SqueezeExcitation(mid_ch, reduction=int(1/se_ratio)),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_skip else out
```

### 8. MobileNet (Howard et al., 2017–2019)

Designed for on-device inference. Uses depthwise separable convolutions (V1), inverted residuals + linear bottlenecks (V2), and hard-swish + SE (V3).

### 9. Deformable Convolutions (Dai et al., 2017)

Standard convolutions use fixed rectangular receptive fields. **Deformable convolutions** learn spatial *offsets* for each sampling location, allowing the receptive field to adapt to object geometry (e.g., elongated limbs, irregular shapes). Each output position predicts 2×K×K offsets, then samples from *deformed* input locations via bilinear interpolation.

**Use cases**: Object detection (DCN for COCO), segmentation, pose estimation. Use `torchvision.ops.deform_conv2d` or mmcv implementations.

**Compound scaling table** (EfficientNet B0→B7): B0 (224, 5.3M) → B4 (380, 19M) → B7 (600, 66M). Scaling resolution alone hits diminishing returns; combining width + depth + resolution yields better accuracy per FLOP.

---

## Feature Pyramid Networks (FPN)

FPN (Lin et al., 2017) builds a multi-scale feature hierarchy with rich semantics at all scales, essential for detecting objects of different sizes.

```
Bottom-up: C2→C3→C4→C5  (ResNet backbone, stride 4,8,16,32)
Top-down:  P5→P4→P3→P2  (lateral connections + upsampling)
```

**Lateral connection**: element-wise addition of top-down and bottom-up features (after 1×1 conv to match channels).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Takes C3, C4, C5 from backbone, outputs P2-P6.
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        self.extra_p6 = nn.Conv2d(in_channels_list[-1], out_channels, 3, stride=2, padding=1)
        self.extra_p7 = nn.Sequential(nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1))

    def forward(self, features):
        """features: list of [C3, C4, C5] from backbone"""
        # Lateral connections
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], scale_factor=2.0, mode="nearest"
            )

        # Output feature maps P3, P4, P5
        out = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # P6, P7 (for FCOS / RetinaNet)
        p6 = self.extra_p6(features[-1])
        p7 = self.extra_p7(p6)
        out.extend([p6, p7])
        return out  # [P3, P4, P5, P6, P7]

# Test with mock backbone outputs
c3 = torch.randn(2, 256, 56, 56)
c4 = torch.randn(2, 512, 28, 28)
c5 = torch.randn(2, 1024, 14, 14)
fpn = FPN([256, 512, 1024], out_channels=256)
pyramid = fpn([c3, c4, c5])
for i, p in enumerate(pyramid):
    print(f"P{i+3}: {p.shape}")
```

---

## Object Detection

### R-CNN Family

**R-CNN** (2014): Region proposals → CNN features → SVM classification. Slow (one forward pass per proposal).

**Fast R-CNN** (2015): ROI Pooling extracts features from a shared feature map. Much faster.

**Faster R-CNN** (2015): Introduces **Region Proposal Network (RPN)** — learns to propose regions from feature maps. End-to-end trainable.

```python
class RPN(nn.Module):
    """Region Proposal Network."""
    def __init__(self, in_channels=256, mid_channels=256, n_anchors=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(mid_channels, n_anchors, 1)     # objectness
        self.bbox_pred = nn.Conv2d(mid_channels, n_anchors * 4, 1)  # (dx,dy,dw,dh)

    def forward(self, x):
        feat = self.conv(x)
        return self.cls_logits(feat), self.bbox_pred(feat)


class ROIPool(nn.Module):
    """ROI Pooling: extracts fixed-size (7x7) features from arbitrary-size ROIs."""
    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size

    def forward(self, feature_map, rois):
        """
        rois: (N, 5) where each row is [batch_idx, x1, y1, x2, y2]
        Feature map: (B, C, H, W)
        """
        outputs = []
        for roi in rois:
            b, x1, y1, x2, y2 = roi.int()
            region = feature_map[b:b+1, :, y1:y2+1, x1:x2+1]
            pooled = nn.functional.adaptive_max_pool2d(region, self.output_size)
            outputs.append(pooled)
        return torch.cat(outputs)
```

### YOLO (You Only Look Once)

YOLOv1 divided the image into S×S grid cells. Each cell predicts B bounding boxes + class probabilities simultaneously — **one pass, no proposals**.

```python
class YOLOv1Head(nn.Module):
    """YOLO v1 detection head."""
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        # Each cell predicts B*(5) + C values
        # 5 = (x, y, w, h, confidence)
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


# YOLOv3+ uses anchor boxes and multi-scale detection at 3 scales
# Modern: YOLOv5/v8 use CSP backbone, PANet neck, decoupled heads
```

### SSD (Single Shot Detector)

Like YOLO but predicts at multiple feature map scales directly from feature pyramid. Each location on each scale predicts k anchor boxes.

### DETR (Detection Transformer — Facebook, 2020)

Formulates detection as a **set prediction problem**. No NMS required.
- CNN backbone extracts features
- Transformer encoder-decoder processes features + learned object queries
- Each query (N=100) predicts one object via bipartite matching loss

---

## Semantic Segmentation

### FCN (Fully Convolutional Network, 2015)

Replace FC layers with 1×1 convolutions. Add skip connections for fine detail.

### DeepLab (2015–2018)

Uses **Atrous Spatial Pyramid Pooling (ASPP)** — apply dilated convolutions with multiple rates in parallel:

```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()

        # 1x1 conv
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        ))
        # Dilated convs
        for rate in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU()
            ))
        # Global average pooling
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5)
        )

    def forward(self, x):
        H, W = x.shape[2:]
        parts = [conv(x) for conv in self.convs]
        gap = F.interpolate(self.gap(x), size=(H, W), mode="bilinear", align_corners=False)
        parts.append(gap)
        return self.project(torch.cat(parts, dim=1))
```

### U-Net (Ronneberger et al., 2015)

Encoder-decoder with **skip connections** at each resolution. Originally for biomedical segmentation.

```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))  # after concat

        self.final = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)

# Test
unet = UNet(in_channels=3, n_classes=2)
x = torch.randn(2, 3, 256, 256)
print("U-Net output:", unet(x).shape)  # (2, 2, 256, 256)
```

---

## Instance Segmentation: Mask R-CNN

Mask R-CNN (He et al., 2017) extends Faster R-CNN by adding a third head that predicts a **binary mask** for each proposed object.

Key contribution: **ROIAlign** — replaces ROIPool's quantization with bilinear interpolation, crucial for pixel-accurate masks.

```python
class MaskHead(nn.Module):
    """Predicts a binary mask for each ROI."""
    def __init__(self, in_channels=256, n_classes=80, hidden=256):
        super().__init__()
        self.convs = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else hidden, hidden, 3, padding=1),
                            nn.ReLU(inplace=True))
              for i in range(4)]
        )
        self.deconv = nn.ConvTranspose2d(hidden, hidden, 2, stride=2)  # upsample 14→28
        self.mask_logits = nn.Conv2d(hidden, n_classes, 1)

    def forward(self, roi_features):
        x = self.convs(roi_features)    # (N, 256, 14, 14)
        x = F.relu(self.deconv(x))      # (N, 256, 28, 28)
        return self.mask_logits(x)       # (N, n_classes, 28, 28)
```

---

## Image Classification Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast

# ── Data Pipeline ──
def get_dataloaders(data_dir: str, batch_size: int = 64, image_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size * 2, shuffle=False,
                               num_workers=4, pin_memory=True)
    return train_loader, val_loader, len(train_ds.classes)
```

---

## Full PyTorch Code Examples

### Custom CNN for CIFAR-10

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class CIFARNet(nn.Module):
    """Compact CNN for CIFAR-10 with residual blocks."""

    def __init__(self, num_classes=10):
        super().__init__()

        def conv_bn_relu(in_c, out_c, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.stem = conv_bn_relu(3, 64)

        # Residual-like blocks with increasing channels
        self.block1 = nn.Sequential(
            conv_bn_relu(64, 64), conv_bn_relu(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2)  # 32→16

        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128), conv_bn_relu(128, 128)
        )
        self.shortcut2 = nn.Conv2d(64, 128, 1)  # adjust channels for skip
        self.pool2 = nn.MaxPool2d(2)  # 16→8

        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256), conv_bn_relu(256, 256)
        )
        self.shortcut3 = nn.Conv2d(128, 256, 1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)                                    # (B,64,32,32)
        x = self.block1(x) + x                             # skip connection
        x = self.pool1(x)                                   # (B,64,16,16)
        x = self.block2(x) + self.shortcut2(x)             # (B,128,16,16)
        x = self.pool2(x)                                   # (B,128,8,8)
        x = self.block3(x) + self.shortcut3(x)             # (B,256,8,8)
        return self.head(x)


def train_cifar10(epochs=30, lr=1e-3, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(2, 9),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("./data", train=False, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size*2, shuffle=False)

    model = CIFARNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*10,
        steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.2
    )
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, n = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

        model.eval()
        test_correct, test_n = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_correct += (model(x).argmax(1) == y).sum().item()
                test_n += x.size(0)

        print(f"Epoch {epoch:3d}: "
              f"Train {train_correct/n:.4f} | Test {test_correct/test_n:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")

    return model

# model = train_cifar10()  # Uncomment to train


### Transfer Learning with ResNet-50

def fine_tune_resnet(num_classes: int, data_dir: str, epochs: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all except last 2 residual groups + head
    for name, param in model.named_parameters():
        freeze = not any(layer in name for layer in ["layer3", "layer4", "fc"])
        param.requires_grad = not freeze

    # Replace head
    model.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    # Differential LR: backbone vs head
    backbone_params = [p for n, p in model.named_parameters()
                       if "fc" not in n and p.requires_grad]
    head_params = list(model.fc.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)

    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=32)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += x.size(0)
        acc = correct / total
        print(f"Epoch {epoch:2d}: Val Acc = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_acc:.4f}")
    return model
```

---

## Architecture Comparison

| Architecture | Year | Top-1 (ImageNet) | Params | Key Innovation |
|---|---|---|---|---|
| LeNet-5 | 1998 | — | 60K | First practical CNN |
| AlexNet | 2012 | 63.3% | 61M | ReLU, dropout, GPU |
| VGG-16 | 2014 | 74.4% | 138M | Deep 3×3 stacks |
| GoogLeNet | 2014 | 74.8% | 6.8M | Inception modules |
| ResNet-50 | 2015 | 76.1% | 25M | Residual connections |
| DenseNet-121 | 2017 | 74.9% | 8M | Dense connections |
| EfficientNet-B0 | 2019 | 77.1% | 5.3M | Compound scaling |
| EfficientNet-B7 | 2019 | 84.4% | 66M | Compound scaling |
| ConvNeXt-B | 2022 | 85.8% | 89M | Modernized CNN |
| ViT-L/16 | 2020 | 87.8% | 307M | Vision Transformer |

---

## Resources

- **CS231n**: cs231n.stanford.edu (Stanford CNN Course)
- **PyTorch Vision**: github.com/pytorch/vision
- **Papers with Code** (Detection): paperswithcode.com/task/object-detection
- **COCO Detection Benchmark**: cocodataset.org
- **Detectron2**: github.com/facebookresearch/detectron2
- **MMDetection**: github.com/open-mmlab/mmdetection
- **Hugging Face Transformers**: huggingface.co/models (Vision models)
