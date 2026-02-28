# Anomaly Detection: Complete Guide

## Table of Contents
1. [Introduction to Anomaly Detection](#introduction-to-anomaly-detection)
2. [Statistical Methods](#statistical-methods)
3. [Isolation Forest](#isolation-forest)
4. [One-Class SVM](#one-class-svm)
5. [Autoencoder-Based Methods](#autoencoder-based-methods)
6. [Deep SVDD](#deep-svdd)
7. [GAN-Based Anomaly Detection](#gan-based-anomaly-detection)
8. [Time Series Anomaly Detection](#time-series-anomaly-detection)
9. [Practical Examples](#practical-examples)
10. [Evaluation and Best Practices](#evaluation-and-best-practices)

---

## Introduction to Anomaly Detection

**Anomaly detection** identifies data points that deviate from "normal" behavior. Used in fraud detection, intrusion detection, manufacturing defects, and health monitoring.

### Key Concepts

- **Anomaly**: Outlier, novelty, exception
- **Normal**: Inlier, typical behavior
- **Training**: Usually on normal data only (unsupervised)

### Paradigms

| Paradigm | Training Data | Use Case |
|----------|---------------|----------|
| **Unsupervised** | Unlabeled (assume mostly normal) | General |
| **Semi-supervised** | Labeled normal only | Clear normal definition |
| **Supervised** | Labeled normal + anomaly | Rare; when labels available |

### Challenges

- **Imbalance**: Anomalies rare (0.1–5%)
- **Definition**: "Normal" can be multi-modal
- **Concept drift**: Normal changes over time
- **Adversarial**: Attackers adapt

---

## Statistical Methods

### Z-Score

For univariate: z = (x - μ) / σ. Flag |z| > 3.

```python
import numpy as np

def zscore_anomaly(x, threshold=3):
    mean, std = np.mean(x), np.std(x)
    z = np.abs((x - mean) / (std + 1e-8))
    return z > threshold

# Example
data = np.random.randn(1000)
data[50] = 10  # Anomaly
flags = zscore_anomaly(data)
print(f"Anomalies: {np.where(flags)[0]}")
```

### Modified Z-Score (Median)

Robust to outliers in the statistics:

```python
def modified_zscore(x, threshold=3.5):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    mad = np.where(mad == 0, 1e-8, mad)
    modified_z = 0.6745 * (x - median) / mad
    return np.abs(modified_z) > threshold
```

### IQR (Interquartile Range)

```python
def iqr_anomaly(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (x < lower) | (x > upper)
```

### Multivariate: Mahalanobis Distance

```python
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

def mahalanobis_anomaly(X, threshold=3):
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    cov_inv = inv(cov + 1e-6 * np.eye(cov.shape[0]))
    dists = np.array([mahalanobis(x, mean, cov_inv) for x in X])
    return dists > threshold
```

---

## Isolation Forest

**Isolation Forest** (Liu et al.): Anomalies are "easier to isolate" — few splits needed. Build random trees; path length indicates anomaly score.

### Idea

- Normal points: need more splits to isolate
- Anomalies: isolated quickly
- Score ∝ 2^{-E(path_length) / c(n)}

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Fit on "normal" data (or mix with few anomalies)
X = np.random.randn(1000, 5)
X[50:52] = [[10, 10, 10, 10, 10], [-8, -8, -8, -8, -8]]  # Anomalies

clf = IsolationForest(
    contamination=0.02,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100
)
clf.fit(X)
predictions = clf.predict(X)  # -1 = anomaly, 1 = normal
scores = clf.decision_function(X)  # Lower = more anomalous

anomaly_indices = np.where(predictions == -1)[0]
print(f"Detected anomalies: {anomaly_indices}")
```

### Tuning

- **contamination**: Fraction of anomalies (or "auto")
- **n_estimators**: More trees = more stable
- **max_samples**: Subsampling for efficiency

---

## One-Class SVM

**One-Class SVM** learns a boundary around normal data in kernel space. Points outside = anomaly.

```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1  # Upper bound on anomaly fraction, lower bound on support vectors
)
clf.fit(X_train)
predictions = clf.predict(X_test)  # -1 = anomaly
scores = -clf.decision_function(X_test)  # Higher = more anomalous
```

### Limitations

- Doesn't scale to large data
- Sensitive to gamma, nu

---

## Autoencoder-Based Methods

Train autoencoder on normal data. **Reconstruction error** high for anomalies.

### Basic Autoencoder

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def recon_loss(self, x):
        recon = self.forward(x)
        return nn.functional.mse_loss(recon, x, reduction='none').mean(dim=1)

# Training (normal data only)
model = Autoencoder(input_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    for batch in normal_loader:
        loss = model.recon_loss(batch).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Inference: threshold on reconstruction error
with torch.no_grad():
    errors = model.recon_loss(X_test)
anomaly_flags = errors > np.percentile(errors, 95)
```

### Variational Autoencoder (VAE)

Use reconstruction + KL; or use **latent** as feature and detect in latent space.

### Donut (Time Series)

VAE for time series with missing data; handles seasonality.

---

## Deep SVDD

**Deep SVDD** (Ruff et al.): Map data to latent space; minimize volume of hypersphere containing normal data. Anomalies fall outside.

### Objective

min Σ ||φ(x_i) - c||² + λ||W||²

c = center (fixed or learned).

```python
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, rep_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, rep_dim)
        )
        self.center = nn.Parameter(torch.zeros(rep_dim))
    
    def forward(self, x):
        return self.encoder(x)
    
    def loss(self, x):
        z = self.forward(x)
        return torch.mean(torch.sum((z - self.center) ** 2, dim=1))
    
    def score(self, x):
        with torch.no_grad():
            z = self.forward(x)
        return torch.sum((z - self.center) ** 2, dim=1)
```

---

## GAN-Based Anomaly Detection

Train GAN on normal data. At test: **anomaly score** = generator's ability to produce similar sample, or discriminator confidence.

### AnoGAN

- Train GAN on normal data
- For test x: find z such that G(z) ≈ x
- Score = ||x - G(z)|| + λ * D(G(z))

### f-AnoGAN

- Train GAN + encoder E
- Map x → E(x) = z
- Score = reconstruction + discriminator loss

---

## Time Series Anomaly Detection

### Prophet

Detect anomalies as points with large residuals from trend + seasonality.

```python
from prophet import Prophet
import pandas as pd

df = pd.DataFrame({'ds': dates, 'y': values})
model = Prophet(interval_width=0.95)
model.fit(df)
forecast = model.predict(df)
df['residual'] = df['y'] - forecast['yhat']
# Anomaly if |residual| > k * std
```

### LSTM Autoencoder

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
    
    def forward(self, x):
        enc, _ = self.encoder(x)
        dec, _ = self.decoder(enc)
        return dec
    
    def recon_loss(self, x):
        recon = self.forward(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=(1,2))
```

### Matrix Profile (STAMP, STOMP)

Efficient motif discovery; anomalies = low similarity to rest of series.

---

## Practical Examples

### Example 1: Credit Card Fraud

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Assume X_train is mostly normal
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X_scaled)
scores = clf.decision_function(scaler.transform(X_test))
# Lower score = more anomalous
```

### Example 2: Image Anomaly (MVTec)

```python
# Use patch-based autoencoder or pretrained features
from torchvision import models
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])

def extract_features(images):
    with torch.no_grad():
        feats = model(images).squeeze()
    return feats

# Train autoencoder on normal image features
# High recon error = anomaly
```

### Example 3: Multivariate Time Series

```python
# LSTM-VAE or Transformer
# Reconstruction error per timestep
# Or: predict next step, large prediction error = anomaly
```

---

## Evaluation and Best Practices

### Metrics

- **Precision, Recall, F1** (if labels available)
- **AUC-ROC, AUC-PR** (scores vs labels)
- **FP rate at given recall**

### Best Practices

1. **Normal data only** for training when possible
2. **Scale features** (StandardScaler) for distance-based methods
3. **Tune contamination** or use validation set
4. **Ensemble** multiple detectors for robustness
5. **Adapt** to concept drift (retrain, sliding window)
6. **Domain**: Use appropriate method (statistical vs deep)

---

## Summary

| Method | Best For | Scalability |
|--------|----------|-------------|
| Statistical | Univariate, tabular | High |
| Isolation Forest | Tabular, mixed | High |
| One-Class SVM | Small, tabular | Low |
| Autoencoder | High-dim, images | Medium |
| Deep SVDD | High-dim | Medium |
| LSTM/VAE | Time series | Medium |

**Libraries**: `sklearn`, `pyod`, `alibi-detect`, `prophet`
