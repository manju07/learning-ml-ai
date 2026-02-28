# Self-Supervised Learning: Complete Guide

## Table of Contents
1. [Introduction to Self-Supervised Learning](#introduction-to-self-supervised-learning)
2. [Contrastive Learning](#contrastive-learning)
3. [SimCLR](#simclr)
4. [MoCo (Momentum Contrast)](#moco-momentum-contrast)
5. [BYOL and DINO](#byol-and-dino)
6. [Masked Autoencoders (MAE)](#masked-autoencoders-mae)
7. [Self-Supervised NLP (BERT, etc.)](#self-supervised-nlp-bert-etc)
8. [Practical Examples](#practical-examples)
9. [Advanced Topics](#advanced-topics)
10. [Best Practices](#best-practices)

---

## Introduction to Self-Supervised Learning

**Self-supervised learning (SSL)** learns representations from unlabeled data by defining a pretext task where the target is derived from the input itself. No human labels required—the "supervision" comes from the data structure.

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

**Key idea**: Pull positive pairs together, push negative pairs apart in representation space.

### Positive and Negative Pairs

- **Positives**: Different views of same sample (augmentations)
- **Negatives**: Different samples (or other views)

### InfoNCE (Noise Contrastive Estimation) Loss

L = -log( exp(sim(q, k+)/τ) / Σ_i exp(sim(q, k_i)/τ) )

- q: query (anchor)
- k+: positive key
- k_i: all keys (including negatives)
- τ: temperature

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

θ_k = m * θ_k + (1 - m) * θ_q

Typically m = 0.999.

### MoCo v2 / v3

- v2: Add MLP projection, stronger augmentations
- v3: ViT backbone, no queue (simpler)

```python
class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=65536, m=0.999):
        super().__init__()
        self.K = K
        self.m = m
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        
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

## BYOL and DINO

### BYOL (Bootstrap Your Own Latent)

- No negatives; uses predictor + momentum target
- Predictor: predict target from online
- Target: momentum encoder
- Loss: MSE between predicted and target

### DINO (Self-Distillation with No Labels)

- Teacher and student with same architecture
- Teacher: EMA of student
- Cross-entropy between student and teacher softmax outputs
- Centering for teacher to avoid collapse

```python
# DINO: Knowledge distillation without labels
# Teacher softmax is target for student
# Avoids collapse via centering and sharpening
```

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

High masking ratio forces semantic understanding; cannot copy neighbors.

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

- Soft clustering with prototypes
- Swap assignment: predict cluster of view from the other view
- No pairwise negatives needed

### Data2Vec

- Unified framework for vision, speech, NLP
- Predict contextualized target from masked input

### V-JEPA (Yann LeCun)

- Joint embedding predictive architecture
- Predict in representation space, not pixel space

---

## Best Practices

1. **Augmentations matter**: Strong but realistic augmentations for contrastive
2. **Batch size**: Larger is better for SimCLR; MoCo uses queue to simulate
3. **Temperature**: 0.07–0.1 common; lower = sharper
4. **Projection head**: Use for contrastive; drop for downstream
5. **Evaluation**: Linear probe on frozen features + full fine-tuning
6. **Pretrain long**: SSL benefits from many epochs

---

## Summary

| Method | Key Idea | Best For |
|--------|----------|----------|
| SimCLR | Contrastive, in-batch negatives | Vision, large batch |
| MoCo | Queue + momentum encoder | Vision, small batch |
| MAE | Mask and reconstruct pixels | Vision, ViT |
| DINO | Self-distillation, no labels | Vision, ViT |
| BERT | Masked language model | NLP |

**When to use**: When you have lots of unlabeled data and want strong representations for downstream tasks.
