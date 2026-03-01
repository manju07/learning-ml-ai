# Self-Supervised Learning: Complete Guide

## Table of Contents
1. [Introduction to Self-Supervised Learning](#introduction-to-self-supervised-learning)
2. [Contrastive Learning](#contrastive-learning)
3. [SimCLR](#simclr)
4. [MoCo (Momentum Contrast)](#moco-momentum-contrast)
5. [BYOL, SimSiam, and Barlow Twins](#byol-simsiam-and-barlow-twins)
6. [DINO](#dino)
7. [Masked Autoencoders (MAE)](#masked-autoencoders-mae)
8. [Self-Supervised NLP (BERT, etc.)](#self-supervised-nlp-bert-etc)
9. [Practical Examples](#practical-examples)
10. [Advanced Topics](#advanced-topics)
11. [Best Practices](#best-practices)
12. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
13. [Performance Benchmarks](#performance-benchmarks)
14. [Further Reading and References](#further-reading-and-references)

---

## Introduction to Self-Supervised Learning

**Self-supervised learning (SSL)** learns representations from unlabeled data by defining a **pretext task** where the target is derived from the input itself. No human labels are required—the "supervision" comes from the structure of the data (e.g., two augmented views of the same image should have similar representations). Real-world analogy: children learn visual concepts before they can name them; SSL mimics this by exploiting structure (spatial, temporal, causal) inherent in data.

### Why Self-Supervised?

| Labeled Data | Unlabeled Data |
|---------------|----------------|
| Expensive to collect | Abundant (images, text, video) |
| Limited scale | Massive scale |
| Supervised: ceiling at data size | SSL: scale with data |

### Pretext Tasks Overview

| Domain | Pretext Task | Example |
|--------|--------------|---------|
| **Vision** | Contrastive | SimCLR, MoCo |
| **Vision** | Reconstruction | MAE, BEiT |
| **Vision** | Clustering | SwAV, DINO |
| **NLP** | Masked LM | BERT |
| **NLP** | Next sentence | BERT |
| **Audio** | CPC, wav2vec | Contrastive predictive coding |

### SSL Pipeline

```
Unlabeled Data → Pretext Task → Encoder → Representations
                                    ↓
                    Downstream Task (classification, detection)
                    (linear probe or fine-tuning)
```

---

## Contrastive Learning

**Key idea**: Pull **positive pairs** (different views of same sample) together in representation space, push **negative pairs** (different samples) apart. The loss encourages the model to be invariant to augmentations while remaining discriminative across samples.

### Positive and Negative Pairs

- **Positives**: Different augmentations of the same sample (e.g., cropped, color-jittered views of one image)
- **Negatives**: Other samples in the batch, or a queue/memory bank of past samples

### InfoNCE (Noise Contrastive Estimation) Loss

For query \(q\) and positive key \(k^+\), with negatives \(\{k_i\}\):

\[
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\sum_i \exp(\text{sim}(q, k_i)/\tau)}
\]

where \(\text{sim}(a,b) = a^\top b / (\|a\| \|b\|)\) (cosine similarity) and \(\tau\) is the **temperature**. Lower \(\tau\) sharpens the distribution (harder negatives matter more). This is a form of NCE that maximizes a lower bound on mutual information \(I(q; k^+)\).

```python
import torch
import torch.nn.functional as F

def info_nce_loss(query, key, temperature=0.07):
    """
    query: [B, D] - anchor embeddings
    key: [B, D] - positive embeddings (same sample, different view)
    In-batch negatives: other samples in batch are negatives
    """
    B = query.shape[0]
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    
    logits = (query @ key.T) / temperature  # [B, B]
    labels = torch.arange(B, device=query.device)  # Diagonal = positives
    
    return F.cross_entropy(logits, labels)
```

### Why Large Batch Size?

More samples → more negatives per query → better contrastive signal. SimCLR uses 4K–8K batch size (or memory bank to simulate).

---

## SimCLR

**SimCLR** (Simple Framework for Contrastive Learning of Visual Representations, Chen et al., 2020).

### Pipeline

1. Take image x
2. Apply two random augmentations → x_i, x_j
3. Encode: h_i = f(x_i), h_j = f(x_j)
4. Project: z_i = g(h_i), z_j = g(h_j)
5. Contrastive loss: (z_i, z_j) positive, (z_i, z_k) negative for k≠j

### Augmentations (Critical)

```python
import torchvision.transforms as T

def get_simclr_augmentations(image_size=224):
    return T.Compose([
        T.RandomResizedCrop(image_size),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

### SimCLR Implementation

```python
import torch
import torch.nn as nn

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder  # ResNet without classifier
        dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, projection_dim)
        )
    
    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        return z1, z2

def simclr_loss(z1, z2, temperature=0.5):
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / temperature
    # Mask self-similarity
    mask = torch.eye(2*B, device=z.device).bool()
    sim = sim.masked_fill(mask, float('-inf'))
    # Positives: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)])
    return F.cross_entropy(sim, labels)
```

---

## MoCo (Momentum Contrast)

**MoCo** (He et al., 2020) uses a **momentum encoder** and **queue** to have many negatives without large batch size.

### Key Components

- **Query encoder**: Updated by gradient
- **Key encoder**: Momentum update of query encoder
- **Queue**: FIFO of key representations (e.g., 65K keys)

### Momentum Update

\[
\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q
\]

Typically \(m = 0.999\). The key encoder changes slowly, providing stable targets for the query branch. The queue stores keys from many past batches, yielding thousands of negatives without a large batch.

### MoCo v2 / v3

- v2: Add MLP projection, stronger augmentations
- v3: ViT backbone, no queue (simpler)

```python
import copy

class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=65536, m=0.999):
        super().__init__()
        self.K = K
        self.m = m
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        
        encoder_dim = 2048  # ResNet-50; adjust for your encoder
        self.proj_q = nn.Linear(encoder_dim, dim)
        self.proj_k = nn.Linear(encoder_dim, dim)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr+keys.shape[0]] = keys.T
        ptr = (ptr + keys.shape[0]) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q, x_k):
        q = self.proj_q(self.encoder_q(x_q))
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = self.proj_k(self.encoder_k(x_k))
            k = F.normalize(k, dim=1)
        
        l_pos = (q * k).sum(dim=1, keepdim=True)
        l_neg = q @ self.queue.clone().detach()
        logits = torch.cat([l_pos, l_neg], dim=1) / 0.07
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        self._dequeue_and_enqueue(k)
        return F.cross_entropy(logits, labels)
```

---

## BYOL, SimSiam, and Barlow Twins

These methods achieve strong results **without explicit negatives**, avoiding the need for large batches or queues.

### BYOL (Bootstrap Your Own Latent)

**BYOL** (Grill et al., 2020) uses two branches: **online** (updated by gradient) and **target** (momentum update of online). The online branch has a **predictor** \(h\) that predicts the target representation from the online representation. Loss: MSE between predicted and target.

\[
\mathcal{L} = \|h(z_\text{online}) - z_\text{target}\|^2
\]

**Why it doesn't collapse**: The target is a slowly moving average; the predictor must work for many different inputs, preventing trivial solutions. Symmetric loss (swap online/target views) is often used.

```python
# BYOL: no negatives, predictor + momentum target
def byol_loss(z_online, z_target, predictor):
    pred = predictor(z_online)
    pred = F.normalize(pred, dim=1)
    z_target = F.normalize(z_target, dim=1)
    return 2 - 2 * (pred * z_target).sum(dim=1).mean()  # MSE of normalized
```

### SimSiam

**SimSiam** (Chen & He, 2021) simplifies BYOL: **no momentum encoder, no negatives**. Same architecture for both views; a **predictor** on one side predicts the other. Stop-gradient on the target branch prevents collapse.

\[
\mathcal{L} = -\frac{1}{2}\big( \langle h(z_1), \text{sg}(z_2) \rangle + \langle h(z_2), \text{sg}(z_1) \rangle \big)
\]

where \(\text{sg}\) = stop-gradient. Collapse is avoided by stop-gradient (the predictor cannot force both to a constant) and the asymmetry of the predictor.

```python
def simsiam_loss(z1, z2, predictor, temperature=0.5):
    """SimSiam: stop-gradient prevents collapse."""
    p1, p2 = predictor(z1), predictor(z2)
    p1, p2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
    z1, z2 = F.normalize(z1, dim=1).detach(), F.normalize(z2, dim=1).detach()
    return -(p1 * z2).sum(dim=1).mean() - (p2 * z1).sum(dim=1).mean()
```

### Barlow Twins

**Barlow Twins** (Zbontar et al., 2021) decorrelates the dimensions of the representations. Let \(Z^a, Z^b \in \mathbb{R}^{B \times D}\) be embeddings of two views. Compute the **cross-correlation matrix** \(C \in \mathbb{R}^{D \times D}\):

\[
C_{ij} = \frac{\sum_b z_{b,i}^a z_{b,j}^b}{\sqrt{\sum_b (z_{b,i}^a)^2} \sqrt{\sum_b (z_{b,j}^b)^2}}
\]

**Loss**: Make \(C\) close to the identity—diagonal elements 1 (invariance), off-diagonal 0 (redundancy reduction).

\[
\mathcal{L} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2
\]

No negatives, no queue, no momentum. Very simple and effective.

```python
def barlow_twins_loss(z1, z2, lambd=5e-3):
    """Barlow Twins: cross-correlation matrix → identity."""
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)
    C = (z1.T @ z2) / B  # [D, D]
    on_diag = ((1 - torch.diag(C)) ** 2).sum()
    off_diag = ((C * (1 - torch.eye(D, device=C.device))) ** 2).sum()
    return on_diag + lambd * off_diag
```

---

## DINO

**DINO** (Self-Distillation with No Labels, Caron et al., 2021) uses **knowledge distillation** without labels: teacher and student share the same architecture; teacher is EMA of student. Loss: cross-entropy between student and teacher softmax outputs over a vocabulary of *cluster centers* (in practice, the model's own output dimensions).

**Centering**: Teacher outputs are centered (running mean subtracted) to avoid collapse to a single mode. **Sharpening**: Low temperature on teacher, higher on student.

\[
\mathcal{L} = - \sum_i P_t(x_i) \log P_s(x_i)
\]

where \(P_t\) = teacher softmax (centered, sharp), \(P_s\) = student softmax.

---

## Masked Autoencoders (MAE)

**MAE** (He et al., 2021): Mask 75% of image patches, reconstruct pixels.

### Pipeline

1. Split image into patches
2. Mask 75% (keep 25% visible)
3. Encode visible patches with ViT
4. Add mask tokens
5. Decoder reconstructs masked patches
6. Loss: MSE on masked patches only

### Why 75% Masking?

High masking ratio (75%) forces **semantic understanding**—the model cannot simply copy neighboring patches; it must infer content from context. Lower ratios (e.g., 50%) allow more "easy" copying and weaker representations. MAE encoder sees only visible patches (no mask tokens), reducing compute.

### MAE Implementation Sketch

```python
class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def random_masking(self, x):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        x = patchify(x)
        x_masked, mask, ids_restore = self.random_masking(x)
        latent = self.encoder(x_masked)
        # Decoder: add mask tokens, reconstruct
        mask_tokens = self.mask_token.expand(latent.shape[0], ids_restore.shape[1] - latent.shape[1], -1)
        latent_full = torch.cat([latent, mask_tokens], dim=1)
        latent_full = torch.gather(latent_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, latent.shape[-1]))
        pred = self.decoder(latent_full)
        loss = ((pred - x) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
        return loss
```

### Using MAE (Hugging Face)

```python
from transformers import AutoModel, AutoImageProcessor

model = AutoModel.from_pretrained("facebook/vit-mae-base")
processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

# Forward returns loss when training
outputs = model(**processor(images=image, return_tensors="pt"))
# For representation: use outputs.last_hidden_state
```

---

## Self-Supervised NLP (BERT, etc.)

### Masked Language Modeling (MLM)

- Mask 15% of tokens
- Predict masked tokens
- 80% replace with [MASK], 10% random, 10% unchanged

### Next Sentence Prediction (NSP)

- Binary: is B the next sentence after A?
- Largely deprecated (RoBERTa dropped it)

### BART, T5

- BART: Denoising (mask, permute, delete)
- T5: Span corruption ("3 4 5" → "3 <X> 5" with "4" as target)

```python
# BERT MLM training
from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("The capital of [MASK] is Paris.", return_tensors="pt")
labels = inputs["input_ids"].clone()
labels[labels != tokenizer.mask_token_id] = -100  # Only loss on mask

outputs = model(**inputs, labels=labels)
loss = outputs.loss
```

---

## Practical Examples

### Example 1: SimCLR Training Loop

```python
def train_simclr(model, train_loader, augment, optimizer, epoch):
    model.train()
    for batch in train_loader:
        x1 = augment(batch)
        x2 = augment(batch)
        z1, z2 = model(x1, x2)
        loss = simclr_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 2: Linear Probing After SSL

```python
# Freeze encoder, train linear classifier
encoder = load_pretrained_mae()  # or SimCLR, MoCo
for p in encoder.parameters():
    p.requires_grad = False

classifier = nn.Linear(encoder_dim, num_classes)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for batch, labels in train_loader:
    with torch.no_grad():
        features = encoder(batch)
    logits = classifier(features)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
```

### Example 3: Fine-Tuning SSL Model

```python
# Add classification head, fine-tune all
model = MAE.from_pretrained("facebook/vit-mae-base")
model.head = nn.Linear(768, num_classes)
# Or replace decoder with head
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
# Train on labeled data
```

---

## Advanced Topics

### Clustering-Based: SwAV

**SwAV** (Caron et al., 2020): Soft clustering with **prototypes** (learnable cluster centroids). For two views, assign each to prototypes; **swap** the assignments—predict the cluster assignment of view 2 from the representation of view 1, and vice versa. No pairwise negatives; scales with number of prototypes (e.g., 3K).

### Data2Vec

**Data2Vec** (Baevski et al., 2022): Unified framework for vision, speech, NLP. **Teacher** builds contextualized target from full input; **student** sees masked input and predicts teacher representation. One algorithm across modalities.

### V-JEPA (Yann LeCun)

**V-JEPA**: Joint embedding predictive architecture. Predict in **representation space** (not pixel space)—predict masked patch representations from context. Aligns with LeCun's "world model" perspective for SSL.

---

## Best Practices

1. **Augmentations matter**: Strong but realistic augmentations for contrastive (SimCLR-style: crop, color, blur)
2. **Batch size**: Larger is better for SimCLR (4K+); MoCo/SimSiam/Barlow avoid this need
3. **Temperature**: 0.07–0.1 common; lower = sharper, harder negatives
4. **Projection head**: Use for contrastive; drop for downstream (use encoder output)
5. **Evaluation**: Linear probe on frozen features + full fine-tuning
6. **Pretrain long**: SSL benefits from many epochs (300–800 for ImageNet)

---

## Common Pitfalls and Troubleshooting

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **SimCLR collapse** | All embeddings similar, loss → 0 | Increase batch size; lower temperature |
| **SimSiam/BYOL collapse** | Embeddings constant | Ensure stop-gradient; check predictor |
| **Low linear probe** | Representations not discriminative | Stronger aug; train longer; try MoCo/MAE |
| **MAE poor on small data** | Overfitting | Reduce model size; fewer epochs |
| **Temperature too high** | Weak contrastive signal | Lower τ (e.g., 0.05–0.07) |
| **Augmentations too weak** | Trivial invariance | Add color jitter, blur, stronger crop |
| **OOM on SimCLR** | Large batch doesn't fit | Use MoCo, SimSiam, or gradient accumulation |

---

## Performance Benchmarks

Linear probe accuracy on ImageNet (Top-1, 224×224, ~100 epochs):

| Method | ResNet-50 | ViT-Base |
|--------|-----------|----------|
| **Supervised** | 76.5% | 79.0% |
| **SimCLR v2** | 69.3% | - |
| **MoCo v3** | - | 72.8% |
| **SimSiam** | 69.8% | - |
| **Barlow Twins** | 69.7% | - |
| **MAE** | - | 68.0% |
| **DINO** | 75.3% | 77.4% |

*Approximate; full fine-tuning often closes the gap. DINO excels for ViT.*

---

## Summary

| Method | Key Idea | Best For |
|--------|----------|----------|
| SimCLR | Contrastive, in-batch negatives | Vision, large batch |
| MoCo | Queue + momentum encoder | Vision, small batch |
| BYOL | Predictor + momentum target, no neg | Vision, stable |
| SimSiam | Stop-gradient predictor, no neg | Vision, simple |
| Barlow Twins | Cross-corr → identity | Vision, simple |
| MAE | Mask and reconstruct pixels | Vision, ViT |
| DINO | Self-distillation, no labels | Vision, ViT |
| BERT | Masked language model | NLP |

**When to use**: When you have lots of unlabeled data and want strong representations for downstream tasks.

---

## Further Reading and References

### Foundational Papers

- Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). ICML.
- He et al. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning* (MoCo). CVPR.
- Grill et al. (2020). *Bootstrap Your Own Latent* (BYOL). NeurIPS.
- Chen & He (2021). *Exploring Simple Siamese Representation Learning* (SimSiam). CVPR.
- Zbontar et al. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*. ICML.
- Caron et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV.
- He et al. (2022). *Masked Autoencoders Are Scalable Vision Learners* (MAE). CVPR.

### Surveys

- Liu et al. (2021). *Self-Supervised Learning: Generative or Contrastive*. IEEE TKDE.
