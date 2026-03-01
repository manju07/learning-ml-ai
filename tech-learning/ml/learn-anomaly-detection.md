# Anomaly Detection: Comprehensive Guide

## Table of Contents
1. [Introduction and Taxonomy](#introduction-and-taxonomy)
2. [Statistical Methods](#statistical-methods)
3. [Density-Based Methods: LOF](#density-based-methods-lof)
4. [Isolation Forest](#isolation-forest)
5. [One-Class SVM and SVDD](#one-class-svm-and-svdd)
6. [Autoencoders for Anomaly Detection](#autoencoders-for-anomaly-detection)
7. [Variational Autoencoders (VAE)](#variational-autoencoders-vae)
8. [Deep SVDD](#deep-svdd)
9. [DAGMM: Deep Autoencoding Gaussian Mixture Model](#dagmm)
10. [Time Series Anomaly Detection](#time-series-anomaly-detection)
11. [LSTM Autoencoders for Time Series](#lstm-autoencoders-for-time-series)
12. [Multivariate Time Series Methods](#multivariate-time-series-methods)
13. [Streaming Anomaly Detection](#streaming-anomaly-detection)
14. [PyOD: Python Outlier Detection Library](#pyod)
15. [Evaluation Metrics](#evaluation-metrics)
16. [Handling Imbalanced Datasets](#handling-imbalanced-datasets)
17. [Semi-Supervised and Unsupervised Approaches](#semi-supervised-and-unsupervised-approaches)
18. [Real-World Applications](#real-world-applications)
19. [Complete Code Examples](#complete-code-examples)

---

## Introduction and Taxonomy

### What Is Anomaly Detection?

Anomaly detection (also called outlier detection, novelty detection, or fault detection) is the process of identifying data instances that deviate significantly from an established norm. Applications span fraud detection, intrusion detection, manufacturing quality control, medical monitoring, and autonomous system safety.

### Types of Anomalies

**1. Point Anomalies**
A single data point is anomalous with respect to the rest of the data. This is the most common type.

*Example*: A credit card transaction of $50,000 when typical transactions are under $500.

**2. Contextual Anomalies (Conditional Anomalies)**
A data point is anomalous in a specific context but not globally.

*Example*: A temperature of 35°C is normal in July (summer) but anomalous in January (winter). The value 35°C by itself is not unusual — the temporal context makes it anomalous.

**3. Collective Anomalies**
A collection of data points is anomalous as a group, even though each individual point may not be.

*Example*: A sequence of ECG readings that individually look normal but collectively represent an arrhythmia pattern. Or a sudden 10-second spike in network traffic.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Visualize the three types
np.random.seed(42)
n = 300
time = np.arange(n)

# Point anomaly: single spike
data_point = np.random.randn(n) * 0.5 + np.sin(time * 0.1)
data_point[150] = 8.0  # Point anomaly

# Contextual anomaly: seasonally inappropriate value
seasonal = np.sin(time * 0.05) + np.random.randn(n) * 0.2
seasonal[200] = seasonal.mean()  # Contextually anomalous in that season

# Collective anomaly: burst pattern
data_collective = np.random.randn(n) * 0.3
data_collective[120:135] = np.random.randn(15) * 3.0  # Anomalous burst

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for ax, data, title, anomaly_idx in zip(
    axes,
    [data_point, seasonal, data_collective],
    ['Point Anomaly (single extreme value)',
     'Contextual Anomaly (unusual for time period)',
     'Collective Anomaly (anomalous burst)'],
    [[150], [200], list(range(120, 135))]
):
    ax.plot(data, color='steelblue', alpha=0.7, label='Data')
    ax.scatter(anomaly_idx, data[anomaly_idx], color='red', s=60, zorder=5, label='Anomaly')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.show()
```

### Detection Paradigms

| Paradigm | Training Data | Assumption | Use Case |
|----------|---------------|------------|----------|
| **Unsupervised** | Unlabeled (mix of normal + anomalies) | Anomalies are rare | Most common; general use |
| **Semi-supervised** | Only labeled normal examples | Clear definition of normal | When normal behavior is well-defined |
| **Supervised** | Labeled normal + anomaly examples | Labeled anomalies available | Rare; fraud with historical cases |

### Challenges

- **Class imbalance**: Anomalies are typically 0.01–5% of data
- **Unknown anomaly types**: New anomaly patterns emerge continuously
- **High dimensionality**: Curse of dimensionality degrades distance-based methods
- **Concept drift**: "Normal" behavior evolves over time
- **Evaluation difficulty**: Ground truth labels often absent
- **Adversarial anomalies**: Attackers mimic normal behavior

---

## Statistical Methods

### Univariate: Z-Score

Assumes data is normally distributed. A point is anomalous if its standardized distance from the mean exceeds a threshold:

\[
z_i = \frac{x_i - \mu}{\sigma}, \quad \text{flag if } |z_i| > k
\]

Typical threshold: \(k = 3\) (covers 99.7% of normal distribution).

```python
import numpy as np
import pandas as pd
from scipy import stats

def zscore_detection(data, threshold=3.0):
    """
    Z-score anomaly detection for univariate data.
    
    Args:
        data: 1D array
        threshold: Z-score threshold (default 3.0 → 99.7% normal)
    
    Returns:
        anomaly_mask: boolean array, True where anomalous
        z_scores: array of z-scores
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z_scores = np.abs((data - mean) / (std + 1e-10))
    return z_scores > threshold, z_scores

# Example
np.random.seed(42)
normal_data = np.random.randn(1000) * 5 + 20
normal_data[50] = 60    # Point anomaly
normal_data[200] = -20  # Point anomaly

anomalies, z_scores = zscore_detection(normal_data, threshold=3.0)
print(f"Z-score: {anomalies.sum()} anomalies detected")
print(f"Anomaly indices: {np.where(anomalies)[0]}")
```

### Robust Z-Score: Modified Z-Score (Median Absolute Deviation)

The standard Z-score is sensitive to outliers in the statistics themselves (since outliers inflate \(\mu\) and \(\sigma\)). The modified Z-score uses the **median** and **MAD** (Median Absolute Deviation):

\[
M_i = \frac{0.6745 (x_i - \tilde{x})}{\text{MAD}}, \quad \text{MAD} = \text{median}(|x_i - \tilde{x}|)
\]

The constant 0.6745 makes MAD comparable to the standard deviation for normal distributions.

```python
def modified_zscore(data, threshold=3.5):
    """
    Modified Z-score using median and MAD — robust to outliers in parameters.
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:  # All values identical or very similar
        mad = np.mean(np.abs(data - median)) + 1e-10
    
    modified_z = 0.6745 * (data - median) / mad
    return np.abs(modified_z) > threshold, modified_z

# More robust than standard Z-score
anomalies_mz, mz_scores = modified_zscore(normal_data)
print(f"Modified Z-score: {anomalies_mz.sum()} anomalies detected")
```

### IQR (Interquartile Range) Method

A non-parametric, distribution-free method using quartiles:

\[
\text{IQR} = Q_3 - Q_1, \quad \text{anomaly if } x < Q_1 - k \cdot \text{IQR} \text{ or } x > Q_3 + k \cdot \text{IQR}
\]

Standard: \(k = 1.5\) (Tukey fences). For extreme outliers: \(k = 3\).

```python
def iqr_detection(data, k=1.5):
    """
    Tukey IQR-based anomaly detection. Distribution-free and robust.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    
    anomalies = (data < lower_fence) | (data > upper_fence)
    return anomalies, lower_fence, upper_fence

anomalies_iqr, lo, hi = iqr_detection(normal_data, k=1.5)
print(f"IQR: {anomalies_iqr.sum()} anomalies detected, range: [{lo:.2f}, {hi:.2f}]")
```

### Grubbs' Test

Detects exactly one outlier per test in normally distributed data. The test statistic:

\[
G = \frac{\max|x_i - \bar{x}|}{s}
\]

The critical value comes from the t-distribution. Iteratively apply (Grubbs' iterative test) to detect multiple outliers.

```python
def grubbs_test(data, alpha=0.05):
    """
    Grubbs' test for a single outlier in normally distributed univariate data.
    Returns the outlier index and whether it's significant.
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    deviations = np.abs(data - mean)
    G_stat = np.max(deviations) / std
    outlier_idx = np.argmax(deviations)
    
    # Critical value from t-distribution
    t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
    
    is_outlier = G_stat > G_crit
    
    return {
        'outlier_index': outlier_idx,
        'outlier_value': data[outlier_idx],
        'G_statistic': G_stat,
        'G_critical': G_crit,
        'is_outlier': is_outlier
    }

result = grubbs_test(normal_data[:100])
print(f"Grubbs test: {result}")
```

### Multivariate: Mahalanobis Distance

For multivariate data, the Mahalanobis distance accounts for feature correlations and different scales:

\[
D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
\]

Under multivariate normality, \(D_M^2 \sim \chi^2(p)\) where \(p\) is the number of features. Flag points where \(D_M^2 > \chi^2_{p, 0.975}\).

```python
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, LinAlgError
from sklearn.covariance import MinCovDet, EmpiricalCovariance

def mahalanobis_detection(X, robust=True, threshold_pct=97.5):
    """
    Mahalanobis distance anomaly detection.
    
    Args:
        X: [n_samples, n_features] array
        robust: use Minimum Covariance Determinant (robust to outliers)
        threshold_pct: chi-squared percentile for threshold
    
    Returns:
        anomaly_mask, distances
    """
    p = X.shape[1]
    
    if robust:
        # MCD: robust estimation of mean and covariance
        mcd = MinCovDet(support_fraction=0.75, random_state=42)
        mcd.fit(X)
        mean = mcd.location_
        cov_inv = inv(mcd.covariance_ + 1e-6 * np.eye(p))
    else:
        mean = X.mean(axis=0)
        cov = np.cov(X.T) + 1e-6 * np.eye(p)
        cov_inv = inv(cov)
    
    # Compute Mahalanobis distance for each point
    diffs = X - mean
    dist_sq = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
    distances = np.sqrt(np.maximum(dist_sq, 0))
    
    # Chi-squared threshold
    threshold = np.sqrt(stats.chi2.ppf(threshold_pct / 100, df=p))
    
    return distances > threshold, distances

# Example with 2D data
np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 500)
X_anomalies = np.array([[4, -3], [-3, 4], [5, 5]])
X_all = np.vstack([X_normal, X_anomalies])

anomaly_mask, distances = mahalanobis_detection(X_all, robust=True)
print(f"Mahalanobis: {anomaly_mask.sum()} anomalies detected")
print(f"Anomaly indices: {np.where(anomaly_mask)[0]}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_all[~anomaly_mask, 0], X_all[~anomaly_mask, 1],
           alpha=0.3, color='steelblue', label='Normal')
ax.scatter(X_all[anomaly_mask, 0], X_all[anomaly_mask, 1],
           color='red', s=80, zorder=5, label='Anomaly')
ax.set_title('Mahalanobis Distance Anomaly Detection')
ax.legend()
plt.show()
```

---

## Density-Based Methods: LOF

### Local Outlier Factor (LOF)

LOF (Breunig et al., 2000) is a density-based anomaly score that compares the local density of a point to its neighbors. Points with substantially lower density than their neighbors are outliers.

**Step 1: k-distance and k-neighborhood**

The k-distance of point \(p\) is the distance to its k-th nearest neighbor. The k-neighborhood \(N_k(p)\) contains all points within this distance.

**Step 2: Reachability Distance**

\[
\text{reach-dist}_k(p, o) = \max\left\{k\text{-dist}(o),\; d(p, o)\right\}
\]

This smooths the distance function — nearby points use their neighbor's k-distance, reducing noise.

**Step 3: Local Reachability Density (LRD)**

\[
\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}
\]

High LRD = point is in a dense region (short reachability distances).

**Step 4: LOF Score**

\[
\text{LOF}_k(p) = \frac{\frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \text{lrd}_k(o)}{\text{lrd}_k(p)}
\]

- LOF ≈ 1: similar density to neighbors → normal
- LOF >> 1: much lower density than neighbors → outlier

```python
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create data with clusters and outliers
X_cluster1 = np.random.randn(200, 2) * 0.5 + [0, 0]
X_cluster2 = np.random.randn(100, 2) * 0.5 + [4, 4]
X_outliers = np.array([[6, 0], [-2, 6], [2, 8], [7, 7], [-1, -3]])
X_lof = np.vstack([X_cluster1, X_cluster2, X_outliers])

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,         # k: larger k = smoother, more global comparison
    contamination=0.05,     # Expected fraction of anomalies
    metric='euclidean',
    novelty=False           # False: unsupervised (fit_predict); True: semi-supervised (fit then predict new)
)

y_pred_lof = lof.fit_predict(X_lof)  # -1 = outlier, 1 = inlier
lof_scores = -lof.negative_outlier_factor_  # Higher = more anomalous

print(f"LOF detected {(y_pred_lof == -1).sum()} outliers")
print(f"Anomaly score range: [{lof_scores.min():.2f}, {lof_scores.max():.2f}]")

# Visualize with anomaly score as marker size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary prediction
axes[0].scatter(X_lof[y_pred_lof == 1, 0], X_lof[y_pred_lof == 1, 1],
                color='steelblue', alpha=0.5, label='Normal')
axes[0].scatter(X_lof[y_pred_lof == -1, 0], X_lof[y_pred_lof == -1, 1],
                color='red', s=100, zorder=5, label='Outlier')
axes[0].set_title('LOF Anomaly Detection')
axes[0].legend()

# Anomaly score as circle size
scatter = axes[1].scatter(X_lof[:, 0], X_lof[:, 1],
                           c=lof_scores, cmap='RdYlBu_r', s=50)
plt.colorbar(scatter, ax=axes[1], label='LOF Score (higher = more anomalous)')
axes[1].set_title('LOF Anomaly Scores')

plt.tight_layout()
plt.show()

# LOF for novelty detection (predict on new data)
lof_novelty = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof_novelty.fit(X_cluster1)  # Train only on cluster 1

# Predict on new points
new_points = np.array([[0.5, 0.2], [5, 5], [-5, -5]])
predictions = lof_novelty.predict(new_points)
print(f"Novelty predictions: {predictions}")  # -1 = novel/anomaly
```

**Choosing k for LOF:**
- Small k (5-10): sensitive to micro-clusters, more local
- Large k (20-50): more global comparison, misses local clusters
- Rule of thumb: k ≈ √n for medium datasets

---

## Isolation Forest

### Algorithm and Intuition

Isolation Forest (Liu et al., 2008) exploits the fact that anomalies are "few and different" — they are easier to **isolate** from the rest of the data.

**Key insight**: Randomly select a feature and a random split point. Repeat recursively. Anomalies will be isolated in fewer steps (shorter path length) because they are in sparse regions.

**The anomaly score** is based on the average path length across all trees:

\[
s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}
\]

where:
- \(E[h(x)]\) = average path length (isolation depth) across trees
- \(c(n) = 2H(n-1) - \frac{2(n-1)}{n}\) normalizes by the average path for a random BST of \(n\) samples
- \(H(i) = \ln(i) + \text{Euler constant} \approx \ln(i) + 0.5772\)

Score interpretation:
- \(s \approx 1\): anomaly (isolated quickly)
- \(s \approx 0.5\): indeterminate
- \(s \ll 0.5\): normal (requires many splits to isolate)

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# ── Basic Isolation Forest ────────────────────────────────────────────────────
np.random.seed(42)
X_normal = make_blobs(n_samples=400, centers=[[0,0],[4,4]], cluster_std=0.8, random_state=42)[0]
X_anom = np.random.uniform(-6, 10, (20, 2))  # Uniform random = anomalies
X_if = np.vstack([X_normal, X_anom])

iforest = IsolationForest(
    n_estimators=200,          # More trees = more stable scores
    max_samples=256,           # Subsample size per tree (paper recommends 256)
    contamination=0.05,        # Estimated fraction of anomalies
    max_features=1.0,          # Fraction of features to consider per split
    bootstrap=False,
    n_jobs=-1,
    random_state=42
)
iforest.fit(X_if)

# Predictions and scores
y_pred_if = iforest.predict(X_if)          # 1 = normal, -1 = anomaly
scores_if = -iforest.decision_function(X_if)  # Higher = more anomalous (negate for intuition)
anomaly_score_raw = iforest.score_samples(X_if)  # Raw anomaly score s(x,n)

print(f"Isolation Forest: {(y_pred_if == -1).sum()} anomalies detected")
print(f"Score range: [{anomaly_score_raw.min():.3f}, {anomaly_score_raw.max():.3f}]")

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-8, 12, 100), np.linspace(-8, 12, 100))
Z = iforest.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision boundary
axes[0].contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                  colors=['#FFCCCC', '#CCDDFF'], alpha=0.5)
axes[0].contour(xx, yy, Z, levels=[0], colors='red', linewidths=1.5)
axes[0].scatter(X_if[y_pred_if == 1, 0], X_if[y_pred_if == 1, 1],
                color='steelblue', alpha=0.5, s=20, label='Normal')
axes[0].scatter(X_if[y_pred_if == -1, 0], X_if[y_pred_if == -1, 1],
                color='red', s=60, zorder=5, label='Anomaly')
axes[0].set_title('Isolation Forest Decision Boundary')
axes[0].legend()

# Anomaly score distribution
axes[1].hist(scores_if[y_pred_if == 1], bins=30, alpha=0.6, color='steelblue', label='Normal')
axes[1].hist(scores_if[y_pred_if == -1], bins=10, alpha=0.6, color='red', label='Anomaly')
axes[1].set_xlabel('Anomaly Score (higher = more anomalous)')
axes[1].set_title('Anomaly Score Distribution')
axes[1].legend()

plt.tight_layout()
plt.show()

# ── High-dimensional use (Isolation Forest excels here) ──────────────────────
from sklearn.datasets import load_breast_cancer

X_cancer = load_breast_cancer().data
iforest_hd = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
iforest_hd.fit(X_cancer)

anomaly_mask_cancer = iforest_hd.predict(X_cancer) == -1
print(f"\nHigh-dim (30 features): {anomaly_mask_cancer.sum()} anomalies in {len(X_cancer)} samples")
```

### Extended Isolation Forest

Extended Isolation Forest (EIF) addresses a known limitation: standard IF uses axis-aligned cuts, which creates "ghost" anomalies in regions near the data that happen to be in sparse coordinate-wise directions. EIF uses random hyperplanes instead.

```python
# pip install eif
# import eif
# eif_model = eif.iForest(X_if, ntrees=200, sample_size=256, ExtensionLevel=1)
# scores_eif = eif_model.compute_paths(X_in=X_if)
# Anomalies have higher scores
```

---

## One-Class SVM and SVDD

### One-Class SVM

One-Class SVM (Schölkopf et al., 1999) maps data to a high-dimensional feature space (via kernel) and finds a **hyperplane** that separates the data from the origin, maximizing the margin. Data projected on the "origin side" = anomaly.

**Objective**:
\[
\min_{w, \xi, \rho} \frac{1}{2} \|w\|^2 - \rho + \frac{1}{\nu n} \sum_{i=1}^n \xi_i
\]
subject to: \(w \cdot \phi(x_i) \geq \rho - \xi_i, \quad \xi_i \geq 0\)

where \(\nu \in (0, 1]\) is an upper bound on the fraction of outliers and lower bound on the support vectors.

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X_train_svm = np.random.randn(300, 2) * 0.8  # Normal data only
X_test_mix = np.vstack([
    np.random.randn(100, 2) * 0.8,            # Normal
    np.array([[4, 4], [-4, -4], [4, -4]])     # Anomalies
])

# MUST standardize for SVM
scaler_svm = StandardScaler()
X_train_scaled = scaler_svm.fit_transform(X_train_svm)
X_test_scaled = scaler_svm.transform(X_test_mix)

ocsvm = OneClassSVM(
    kernel='rbf',
    gamma='scale',   # 1 / (n_features * X.var())
    nu=0.05          # Upper bound on anomaly fraction; lower bound on SVs
)
ocsvm.fit(X_train_scaled)

y_pred_ocsvm = ocsvm.predict(X_test_scaled)
scores_ocsvm = -ocsvm.decision_function(X_test_scaled)  # Higher = more anomalous

print(f"One-Class SVM: {(y_pred_ocsvm == -1).sum()} anomalies in test set")

# ── Effect of gamma and nu ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, gamma in zip(axes[0], ['scale', 0.1, 1.0]):
    for ax2, nu in zip([ax], [0.05]):
        m = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        m.fit(X_train_scaled)
        xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
        Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf],
                     colors=['#FFCCCC', '#CCDDFF'], alpha=0.5)
        ax.contour(xx, yy, Z, levels=[0], colors='red')
        ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                   alpha=0.4, s=10, color='steelblue')
        ax.set_title(f'gamma={gamma}')

plt.suptitle('One-Class SVM: Effect of Gamma (nu=0.05)')
plt.tight_layout()
plt.show()
```

### SVDD: Support Vector Data Description

SVDD (Tax & Duin, 2004) finds the **minimum enclosing hypersphere** around the data in feature space. The radius \(R\) and center \(a\) minimize:

\[
\min_{R, a, \xi} R^2 + C \sum_{i=1}^n \xi_i
\]
subject to: \(\|\phi(x_i) - a\|^2 \leq R^2 + \xi_i, \quad \xi_i \geq 0\)

Points outside the sphere are anomalies. When using the RBF kernel, SVDD and One-Class SVM are equivalent.

```python
# SVDD implementation sketch
class SVDD:
    """Simplified SVDD using sklearn's One-Class SVM (equivalent with RBF kernel)"""
    def __init__(self, C=0.1, gamma='scale'):
        self.ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=1-C)
    
    def fit(self, X):
        self.ocsvm.fit(X)
        return self
    
    def predict(self, X):
        return self.ocsvm.predict(X)
    
    def distance_to_boundary(self, X):
        """Positive = inside sphere (normal), Negative = outside (anomaly)"""
        return self.ocsvm.decision_function(X)
```

---

## Autoencoders for Anomaly Detection

### Core Principle

Train an autoencoder exclusively on **normal data**. The autoencoder learns a compact representation of normalcy. Anomalous inputs, not seen during training, cannot be reconstructed accurately — yielding high **reconstruction error**.

**Anomaly score**:
\[
s(x) = \|x - \hat{x}\|_2^2 = \|x - \text{Decoder}(\text{Encoder}(x))\|_2^2
\]

**Threshold**: Set at a high percentile (e.g., 95th–99th) of reconstruction errors on the training/validation set.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for anomaly detection.
    Adding noise during training improves robustness and anomaly separation.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], latent_dim=16,
                 noise_factor=0.1, dropout=0.3):
        super().__init__()
        self.noise_factor = noise_factor
        
        # Encoder
        enc_layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            enc_layers.extend([nn.Linear(prev_dim, hdim), nn.BatchNorm1d(hdim),
                               nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hdim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_layers = []
        prev_dim = latent_dim
        for hdim in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev_dim, hdim), nn.BatchNorm1d(hdim),
                               nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hdim
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # Add noise during training (denoising)
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            x_noisy = x + noise
        else:
            x_noisy = x
        
        z = self.encode(x_noisy)
        x_recon = self.decode(z)
        return x_recon
    
    def reconstruction_error(self, x):
        """Per-sample reconstruction error (anomaly score)"""
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
        # Mean squared error per sample
        return F.mse_loss(x_recon, x, reduction='none').mean(dim=1)


def train_autoencoder(model, train_loader, val_loader=None, epochs=100, lr=1e-3,
                       weight_decay=1e-5, patience=15, device='cpu'):
    """Train autoencoder with early stopping"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_recon = model(batch_x)
            loss = F.mse_loss(x_recon, batch_x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, in val_loader:
                    batch_x = batch_x.to(device)
                    x_recon = model(batch_x)
                    val_loss += F.mse_loss(x_recon, batch_x).item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '/tmp/best_ae.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    model.load_state_dict(torch.load('/tmp/best_ae.pt'))
                    break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}"
                  + (f", val_loss={avg_val_loss:.6f}" if val_loader else ""))
    
    return train_losses, val_losses


def evaluate_autoencoder(model, X_test, y_test, threshold_percentile=95, device='cpu'):
    """
    Evaluate autoencoder for anomaly detection.
    
    Args:
        model: trained autoencoder
        X_test: tensor [n, features]
        y_test: ground truth labels (1 = anomaly)
        threshold_percentile: percentile of train scores to use as threshold
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        scores = model.reconstruction_error(X_test.to(device)).cpu().numpy()
    
    # ROC-AUC
    auroc = roc_auc_score(y_test, scores)
    
    # Precision-Recall AUC
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    auprc = auc(recall, precision)
    
    # Find optimal F1 threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    best_f1 = f1_scores.max()
    
    print(f"AUROC:  {auroc:.4f}")
    print(f"AUPRC:  {auprc:.4f}")
    print(f"Best F1: {best_f1:.4f} at threshold {best_threshold:.6f}")
    
    return scores, auroc, auprc, best_f1


# ── Complete example with synthetic data ─────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# Simulate normal sensor data with 20 features
n_normal = 2000
n_anomaly = 100
n_features = 20

# Normal: samples from a low-dim manifold
latent_normal = np.random.randn(n_normal, 3)
projection = np.random.randn(3, n_features)
X_normal_ae = latent_normal @ projection + np.random.randn(n_normal, n_features) * 0.5

# Anomaly: off-manifold points
X_anomaly_ae = np.random.randn(n_anomaly, n_features) * 3

# Labels
y_test_ae = np.array([0] * n_normal + [1] * n_anomaly)
X_all_ae = np.vstack([X_normal_ae, X_anomaly_ae])

# Standardize
scaler_ae = StandardScaler()
X_normal_scaled = scaler_ae.fit_transform(X_normal_ae)
X_all_scaled = scaler_ae.transform(X_all_ae)

# Split normal data for training (no anomalies during training)
X_train_ae = torch.FloatTensor(X_normal_scaled[:1500])
X_val_ae = torch.FloatTensor(X_normal_scaled[1500:])
X_test_tensor = torch.FloatTensor(X_all_scaled)

train_ds = TensorDataset(X_train_ae)
val_ds = TensorDataset(X_val_ae)
train_loader_ae = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader_ae = DataLoader(val_ds, batch_size=64)

# Train
ae = DenoisingAutoencoder(
    input_dim=n_features, hidden_dims=[64, 32], latent_dim=8,
    noise_factor=0.1, dropout=0.2
)
train_losses, val_losses = train_autoencoder(
    ae, train_loader_ae, val_loader_ae, epochs=200, lr=1e-3, patience=20
)

# Evaluate
scores_ae, auroc_ae, auprc_ae, f1_ae = evaluate_autoencoder(
    ae, X_test_tensor, y_test_ae
)


# ── Contractive Autoencoder (alternative) ────────────────────────────────────
# Adds Jacobian penalty: encourages smoothness, anomalies often have large gradients
class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=16, lam=1e-4):
        super().__init__()
        self.lam = lam
        enc = []; prev = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []; prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def contractive_loss(self, x, x_recon, z):
        mse = F.mse_loss(x_recon, x)
        # Jacobian penalty: sum of squared gradients of z w.r.t. x
        if x.requires_grad:
            jacobian = torch.autograd.grad(z.sum(), x, create_graph=True)[0]
            penalty = (jacobian ** 2).sum()
        else:
            penalty = torch.tensor(0.0)
        return mse + self.lam * penalty
```

---

## Variational Autoencoders (VAE)

VAEs provide a **probabilistic** anomaly score that combines reconstruction error with the KL divergence of the latent distribution from the prior.

**ELBO (Evidence Lower Bound)**:
\[
\mathcal{L}(x) = \underbrace{E_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}}
\]

**Anomaly score options:**
1. Reconstruction probability: \(-\log p_\theta(x|z)\)
2. ELBO score: \(-\mathcal{L}(x)\)
3. Combined: reconstruction + KL divergence

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
        super().__init__()
        
        # Encoder: q_phi(z|x)
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: p_theta(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder_fc(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic at test time
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def vae_loss(self, x, x_recon, mu, logvar, beta=1.0):
        """
        ELBO loss: reconstruction + beta * KL divergence
        beta > 1: disentangled VAE (β-VAE)
        """
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(dim=1)
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return (recon_loss + beta * kl_loss).mean(), recon_loss.mean(), kl_loss.mean()
    
    def anomaly_score(self, x, n_samples=10):
        """
        Monte Carlo estimate of anomaly score.
        Average -ELBO over multiple latent samples for stability.
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            scores = []
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                x_recon = self.decode(z)
                recon = F.mse_loss(x_recon, x, reduction='none').sum(dim=1)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                scores.append(recon + kl)
        return torch.stack(scores).mean(dim=0)
```

---

## Deep SVDD

Deep SVDD (Ruff et al., 2018) maps normal data to a compact hypersphere in a learned latent space. At test time, points far from the sphere center are anomalous.

**Objective**:
\[
\min_{W} \frac{1}{n} \sum_{i=1}^n \|\phi(x_i; W) - c\|^2 + \frac{\lambda}{2} \|W\|_F^2
\]

where \(c\) is a fixed center (initialized as the mean of initial network outputs), and \(\phi(\cdot; W)\) is a neural network.

**Soft-boundary variant** adds slack variables to handle more extreme distributions:
\[
\min_{R, W} R^2 + \frac{1}{\nu n} \sum_{i=1}^n \max(0, \|\phi(x_i; W) - c\|^2 - R^2) + \frac{\lambda}{2} \|W\|_F^2
\]

```python
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], rep_dim=32):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            # No bias in Deep SVDD (to avoid hypersphere collapse to constant mappings)
            layers.extend([
                nn.Linear(prev_dim, hdim, bias=False),
                nn.BatchNorm1d(hdim),
                nn.LeakyReLU(0.1)
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, rep_dim, bias=False))
        
        self.network = nn.Sequential(*layers)
        self.rep_dim = rep_dim
        self.center = None
    
    def forward(self, x):
        return self.network(x)
    
    def initialize_center(self, data_loader, device='cpu', eps=0.1):
        """Initialize center c as the mean of initial network outputs"""
        self.eval()
        all_outputs = []
        with torch.no_grad():
            for batch_x, in data_loader:
                outputs = self(batch_x.to(device))
                all_outputs.append(outputs)
        
        center = torch.cat(all_outputs, dim=0).mean(dim=0)
        
        # Avoid collapse: if any component is too close to zero, shift it
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center >= 0)] = eps
        
        self.center = center.detach()
        return center
    
    def svdd_loss(self, x):
        """One-class Deep SVDD loss: minimize distances to center"""
        outputs = self(x)
        distances = torch.sum((outputs - self.center) ** 2, dim=1)
        return distances.mean(), distances
    
    def anomaly_score(self, x):
        """Distance from center = anomaly score"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
        return torch.sum((outputs - self.center) ** 2, dim=1)
```

---

## DAGMM

### Deep Autoencoding Gaussian Mixture Model

DAGMM (Zong et al., 2018) jointly trains an autoencoder and a Gaussian Mixture Model (GMM) in the latent space, enabling probabilistic anomaly scoring.

**Architecture:**
1. Autoencoder encodes \(x\) → latent \(z\), computes reconstruction features (reconstruction error components)
2. Estimation network maps \([z; e(x, \hat{x})]\) → mixture membership probabilities
3. GMM models the joint distribution; samples with low likelihood are anomalies

**Anomaly score**: negative log-likelihood under the GMM:
\[
s(x) = -\log p(z_x) = -\log \sum_{k=1}^K \phi_k \cdot \mathcal{N}(z_x; \mu_k, \Sigma_k)
\]

---

## Time Series Anomaly Detection

### Statistical: SPOT (Streaming Peaks-Over-Threshold)

SPOT uses **Extreme Value Theory** to set adaptive thresholds from the tail distribution of time series data. It's particularly suited for streaming data with unknown distributions.

```python
import numpy as np
from scipy.stats import genpareto

class SPOT:
    """
    SPOT: Statistical anomaly detection for streaming time series.
    Uses Peaks-Over-Threshold (POT) from Extreme Value Theory.
    """
    def __init__(self, q=1e-4, n_init=1000, level=0.02):
        """
        q: anomaly threshold risk level
        n_init: initial calibration window size
        level: level for local extremes selection
        """
        self.q = q
        self.n_init = n_init
        self.level = level
        self.thresholds = []
    
    def _grimshaw(self, peaks, epsilon=1e-8, n_points=10):
        """Fit Generalized Pareto Distribution via Grimshaw's method"""
        peaks = np.array(peaks)
        Ymin = peaks.min() - epsilon
        Ymax = peaks.max()
        Ymean = peaks.mean()
        
        a = -1.0 / Ymax
        
        def W(Y):
            return np.mean(np.log(1 - a * Y))
        
        def jac_W(Y):
            return np.mean(Y / (1 - a * Y) ** 2)
        
        # Simplified: use scipy's MLE directly
        try:
            params = genpareto.fit(peaks, floc=0)
            return params[0], params[2]  # shape (xi), scale (sigma)
        except:
            return 0.1, Ymean
    
    def _z_from_gp(self, xi, sigma, n_excess, n_total, q):
        """Compute threshold z from GPD parameters"""
        if xi != 0:
            return sigma / xi * ((q * n_total / n_excess) ** (-xi) - 1)
        else:
            return sigma * np.log(q * n_total / n_excess)
    
    def fit(self, data):
        """Initialize SPOT with calibration data"""
        self.init_data = data[:self.n_init]
        t = np.percentile(self.init_data, (1 - self.level) * 100)
        peaks = self.init_data[self.init_data > t] - t
        
        self.xi, self.sigma = self._grimshaw(peaks)
        n_excess = len(peaks)
        n_total = len(self.init_data)
        
        self.threshold = t + self._z_from_gp(self.xi, self.sigma, n_excess, n_total, self.q)
        return self
    
    def stream_detect(self, new_value):
        """Process a new streaming value, return True if anomaly"""
        return new_value > self.threshold


# SPOT example
np.random.seed(42)
ts = np.random.randn(2000)
ts[1500:1510] += 8  # Inject anomalies

spot = SPOT(q=1e-4, n_init=1000, level=0.02)
spot.fit(ts)

print(f"SPOT threshold: {spot.threshold:.3f}")
anomalies_spot = [i for i, val in enumerate(ts[1000:]) if spot.stream_detect(val)]
print(f"SPOT anomalies detected at indices: {[i+1000 for i in anomalies_spot[:10]]}")
```

### Prophet-Based Anomaly Detection

```python
from prophet import Prophet
import pandas as pd
import numpy as np

def prophet_anomaly(df, interval_width=0.99, changepoint_prior_scale=0.01):
    """
    Detect anomalies as points outside Prophet's prediction interval.
    
    Args:
        df: DataFrame with columns ['ds', 'y'] (date, value)
        interval_width: width of uncertainty interval (higher = fewer anomalies)
    
    Returns:
        DataFrame with anomaly flags and residuals
    """
    model = Prophet(
        interval_width=interval_width,
        changepoint_prior_scale=changepoint_prior_scale,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(df)
    
    forecast = model.predict(df)
    
    result = df.copy()
    result['yhat'] = forecast['yhat']
    result['yhat_lower'] = forecast['yhat_lower']
    result['yhat_upper'] = forecast['yhat_upper']
    result['residual'] = result['y'] - result['yhat']
    result['anomaly'] = (result['y'] > result['yhat_upper']) | (result['y'] < result['yhat_lower'])
    
    return result, model


# Synthetic time series with trend, seasonality, and anomalies
dates = pd.date_range('2023-01-01', periods=365, freq='D')
values = (np.sin(np.arange(365) * 2 * np.pi / 365) * 10 +  # Yearly seasonality
          np.arange(365) * 0.02 +                           # Trend
          np.random.randn(365) * 1.5)                       # Noise
values[100] += 15  # Anomaly 1: spike
values[200:205] -= 10  # Anomaly 2: dip

df_ts = pd.DataFrame({'ds': dates, 'y': values})
result_ts, prophet_model = prophet_anomaly(df_ts)

print(f"Prophet detected {result_ts['anomaly'].sum()} anomalies")
print(result_ts[result_ts['anomaly']][['ds', 'y', 'yhat', 'residual']])
```

---

## LSTM Autoencoders for Time Series

LSTM autoencoders capture temporal dependencies for sequential anomaly detection. The reconstruction error over a window indicates anomalous segments.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for multivariate time series anomaly detection.
    Encoder compresses the sequence; Decoder reconstructs it.
    High reconstruction error → anomaly.
    """
    def __init__(self, n_features, seq_len, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_features = n_features
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder LSTM
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)
    
    def encode(self, x):
        # x: [batch, seq_len, n_features]
        _, (hidden, _) = self.encoder_lstm(x)
        # Use last layer's hidden state
        z = self.encoder_fc(hidden[-1])  # [batch, latent_dim]
        return z
    
    def decode(self, z):
        # Expand latent to sequence
        h = self.decoder_fc(z)  # [batch, hidden_dim]
        h_seq = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch, seq_len, hidden_dim]
        
        out, _ = self.decoder_lstm(h_seq)
        recon = self.output_layer(out)  # [batch, seq_len, n_features]
        return recon
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
    
    def reconstruction_error(self, x):
        """Per-sequence anomaly score"""
        self.eval()
        with torch.no_grad():
            recon = self(x)
        # Mean squared error over time steps and features
        return F.mse_loss(recon, x, reduction='none').mean(dim=(1, 2))
    
    def timestep_error(self, x):
        """Per-timestep error (for pinpointing anomaly location)"""
        self.eval()
        with torch.no_grad():
            recon = self(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=2)  # [batch, seq_len]


def create_sequences(data, seq_len, stride=1):
    """Create sliding window sequences from time series"""
    sequences = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


# ── Training example ──────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

n_samples = 2000
n_feat = 5
seq_len = 30

# Generate multivariate normal time series
data_ts = np.column_stack([
    np.cumsum(np.random.randn(n_samples) * 0.1),         # Random walk
    np.sin(np.arange(n_samples) * 0.1),                   # Sinusoidal
    np.random.randn(n_samples),                            # White noise
    np.cumsum(np.random.randn(n_samples) * 0.05) * 0.5,  # Slow drift
    np.where(np.arange(n_samples) % 100 < 10, 1.0, 0.0)  # Periodic pulses
])

# Inject anomalies
anomaly_start = 1700
data_ts[anomaly_start:anomaly_start+50, :] += np.random.randn(50, n_feat) * 3

# Standardize
from sklearn.preprocessing import StandardScaler
scaler_ts = StandardScaler()
data_ts_scaled = scaler_ts.fit_transform(data_ts)

# Create sequences (training: only normal data)
sequences = create_sequences(data_ts_scaled[:1500], seq_len=seq_len, stride=1)
X_seqs = torch.FloatTensor(sequences)
train_ds_lstm = TensorDataset(X_seqs)
train_loader_lstm = DataLoader(train_ds_lstm, batch_size=32, shuffle=True)

# Initialize model
lstm_ae = LSTMAutoencoder(n_features=n_feat, seq_len=seq_len,
                           hidden_dim=64, latent_dim=16, num_layers=2)
optimizer_lstm = torch.optim.Adam(lstm_ae.parameters(), lr=1e-3)

# Train
for epoch in range(50):
    lstm_ae.train()
    total_loss = 0
    for (batch_x,) in train_loader_lstm:
        optimizer_lstm.zero_grad()
        recon = lstm_ae(batch_x)
        loss = F.mse_loss(recon, batch_x)
        loss.backward()
        optimizer_lstm.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(train_loader_lstm):.6f}")

# Evaluate on all data
all_sequences = create_sequences(data_ts_scaled, seq_len=seq_len, stride=1)
X_all_seqs = torch.FloatTensor(all_sequences)
errors = lstm_ae.reconstruction_error(X_all_seqs).numpy()

# Threshold at 95th percentile of training errors
train_errors = lstm_ae.reconstruction_error(X_seqs).numpy()
threshold_lstm = np.percentile(train_errors, 95)

anomaly_windows = errors > threshold_lstm
print(f"LSTM AE: {anomaly_windows.sum()} anomalous windows detected")
print(f"Threshold: {threshold_lstm:.6f}")
```

---

## Multivariate Time Series Methods

### MSCRED (Multi-Scale Convolutional Recurrent Encoder-Decoder)

Captures spatial correlations between sensors via signature matrices and temporal dynamics via ConvLSTM.

### USAD (UnSupervised Anomaly Detection)

USAD uses two autoencoders in an adversarial training scheme:
- AE1 trains normally
- AE2 trains to reconstruct AE1's output, amplifying anomaly signals
- Anomaly score: combines both reconstruction errors to balance detection sensitivity

```python
class USAD(nn.Module):
    """
    USAD: Two-autoencoder system for amplified anomaly detection.
    """
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder 1: standard reconstruction
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        # Decoder 2: reconstructs from encoder output of AE1's output
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        w1 = self.decoder1(z)
        w2 = self.decoder2(self.encoder(w1))
        return z, w1, w2
    
    def usad_loss(self, x, z, w1, w2, epoch, n_epochs=50):
        """
        Epoch-adaptive loss:
        - Early: favor AE1 reconstruction quality
        - Later: favor AE2 sensitivity (amplifies anomalies)
        """
        n = epoch
        N = n_epochs
        
        # AE1 loss: minimize reconstruction of x
        loss1 = F.mse_loss(w1, x)
        
        # AE2 loss: minimize reconstruction but with AE1 adversarial term
        loss2 = (1/n) * F.mse_loss(w2, x) - (1/n) * F.mse_loss(w2, w1)
        
        return (1/n) * loss1 + (1 - 1/n) * loss2
    
    def anomaly_score(self, x, alpha=0.5, beta=0.5):
        """
        Anomaly score combining both AE errors.
        High score = anomaly.
        """
        self.eval()
        with torch.no_grad():
            _, w1, w2 = self(x)
        score1 = F.mse_loss(w1, x, reduction='none').mean(dim=1)
        score2 = F.mse_loss(w2, x, reduction='none').mean(dim=1)
        return alpha * score1 + beta * score2
```

---

## Streaming Anomaly Detection

### ADWIN (ADaptive WINdowing)

ADWIN maintains an adaptive window of recent data and detects drift by comparing statistics of sub-windows. When drift is detected, it trims the oldest data.

```python
# pip install river
from river import anomaly, drift, stream, preprocessing

# ── River: Online Machine Learning for Anomaly Detection ─────────────────────
# HalfSpaceTrees: streaming version of Isolation Forest
hst = anomaly.HalfSpaceTrees(
    n_trees=25,
    height=15,
    window_size=250,
    seed=42
)

# Example with streaming data
np.random.seed(42)
n_stream = 1000
data_stream = np.random.randn(n_stream)
data_stream[500:510] = 8.0  # Anomaly injection

scores_stream = []
for i, x in enumerate(data_stream):
    # Score comes before learning (so we can score new points)
    score = hst.score_one({'x': x})
    hst.learn_one({'x': x})
    scores_stream.append(score)

scores_stream = np.array(scores_stream)
threshold_stream = np.percentile(scores_stream, 99)
anomalies_stream = np.where(scores_stream > threshold_stream)[0]
print(f"ADWIN/HST streaming anomalies: {anomalies_stream[:15]}")

# ADWIN for concept drift detection
adwin = drift.ADWIN(delta=0.002)
drift_detected = []
for i, x in enumerate(data_stream):
    adwin.update(x)
    if adwin.drift_detected:
        drift_detected.append(i)
        print(f"Drift detected at index {i}")

# ── River One-Class Gaussian ──────────────────────────────────────────────────
gaussian_detector = anomaly.GaussianScorer(window_size=200, n_std=3.0)
for val in data_stream[:200]:
    gaussian_detector.learn_one({'x': val})

# Score new points
new_score = gaussian_detector.score_one({'x': 9.0})  # Should be high
print(f"Gaussian score for extreme value: {new_score:.4f}")
```

---

## PyOD: Python Outlier Detection Library

PyOD (Zhao et al., 2019) provides a unified API for 40+ anomaly detection algorithms.

```python
# pip install pyod
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE as PyOD_VAE
from pyod.models.deep_svdd import DeepSVDD as PyOD_DeepSVDD
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from pyod.utils.example import visualize

# Generate benchmark data
X_train_pyod, X_test_pyod, y_train_pyod, y_test_pyod = generate_data(
    n_train=400, n_test=100,
    n_features=2,
    contamination=0.1,
    random_state=42
)

# ── Evaluate multiple detectors ──────────────────────────────────────────────
detectors = {
    'Isolation Forest': IForest(contamination=0.1, random_state=42, n_jobs=-1),
    'LOF': LOF(n_neighbors=20, contamination=0.1),
    'One-Class SVM': OCSVM(kernel='rbf', nu=0.1),
    'HBOS': HBOS(n_bins=10, contamination=0.1),       # Histogram-based
    'KNN': KNN(n_neighbors=5, contamination=0.1),      # k-NN distance
    'CBLOF': CBLOF(n_clusters=8, contamination=0.1, random_state=42),  # Cluster-based
    'COPOD': COPOD(contamination=0.1),                 # Copula-based
    'ECOD': ECOD(contamination=0.1),                   # Empirical CDF
}

results = {}
for name, detector in detectors.items():
    detector.fit(X_train_pyod)
    
    y_pred = detector.predict(X_test_pyod)
    y_scores = detector.decision_function(X_test_pyod)
    
    from sklearn.metrics import roc_auc_score, f1_score
    auroc = roc_auc_score(y_test_pyod, y_scores)
    f1 = f1_score(y_test_pyod, y_pred)
    
    results[name] = {'AUROC': auroc, 'F1': f1}
    print(f"{name:20s}: AUROC={auroc:.3f}, F1={f1:.3f}")

# ── Combining detectors (ensemble) ──────────────────────────────────────────
from pyod.models.combination import average, maximization, majority_vote
from pyod.utils.utility import standardizer

# Normalize scores to [0,1] for combination
test_scores_list = []
for detector in detectors.values():
    scores = detector.decision_function(X_test_pyod)
    test_scores_list.append(scores)

train_scores_list = [d.decision_scores_ for d in detectors.values()]
test_scores_norm, train_scores_norm = standardizer(test_scores_list, train_scores_list)

avg_scores = average(np.array(test_scores_norm))
max_scores = maximization(np.array(test_scores_norm))

print(f"\nEnsemble Average AUROC: {roc_auc_score(y_test_pyod, avg_scores):.3f}")
print(f"Ensemble Maximum AUROC: {roc_auc_score(y_test_pyod, max_scores):.3f}")
```

---

## Evaluation Metrics

### Why Standard Accuracy Fails

With 99% normal data and 1% anomalies, a model predicting "everything normal" achieves 99% accuracy but detects zero anomalies. Proper evaluation requires anomaly-aware metrics.

```python
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

def comprehensive_evaluation(y_true, y_scores, plot=True):
    """
    Complete evaluation for anomaly detection.
    
    Args:
        y_true: binary labels (1 = anomaly)
        y_scores: continuous anomaly scores (higher = more anomalous)
        plot: whether to plot ROC and PR curves
    """
    # ── AUROC ────────────────────────────────────────────────────────────────
    auroc = roc_auc_score(y_true, y_scores)
    
    # ── AUPRC (more informative for imbalanced data) ──────────────────────────
    auprc = average_precision_score(y_true, y_scores)
    
    # ── Precision@K ──────────────────────────────────────────────────────────
    k = int(y_true.sum())  # Set K = number of true anomalies
    top_k_idx = np.argsort(y_scores)[-k:]
    precision_at_k = y_true[top_k_idx].mean()
    
    # ── F1 at optimal threshold ───────────────────────────────────────────────
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    # ── At best threshold ─────────────────────────────────────────────────────
    y_pred_best = (y_scores >= best_threshold).astype(int)
    precision_best = precision_score(y_true, y_pred_best, zero_division=0)
    recall_best = recall_score(y_true, y_pred_best)
    
    print(f"{'='*50}")
    print(f"AUROC:            {auroc:.4f}")
    print(f"AUPRC:            {auprc:.4f}")
    print(f"Precision@K:      {precision_at_k:.4f}  (K={k})")
    print(f"Best F1:          {best_f1:.4f}")
    print(f"Best Precision:   {precision_best:.4f}")
    print(f"Best Recall:      {recall_best:.4f}")
    print(f"Best Threshold:   {best_threshold:.6f}")
    print(f"{'='*50}")
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {auroc:.3f})')
        axes[0].plot([0,1], [0,1], 'k--', alpha=0.5)
        axes[0].fill_between(fpr, tpr, alpha=0.1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        
        # PR Curve
        axes[1].plot(recall, precision, color='darkorange', lw=2, label=f'PR (AUPRC = {auprc:.3f})')
        axes[1].axhline(y=y_true.mean(), color='navy', linestyle='--', alpha=0.7,
                         label=f'Baseline (prevalence={y_true.mean():.3f})')
        axes[1].scatter([recall_best], [precision_best], color='red', s=100, zorder=5,
                         label=f'Best F1={best_f1:.3f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'auroc': auroc, 'auprc': auprc, 'precision_at_k': precision_at_k,
        'best_f1': best_f1, 'best_threshold': best_threshold
    }
```

---

## Handling Imbalanced Datasets

Anomaly detection inherently involves severe class imbalance. Strategies:

```python
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import numpy as np

# ── When you have labels ──────────────────────────────────────────────────────

# 1. Adjust class weights in supervised components
rf_weighted = RandomForestClassifier(
    class_weight='balanced',    # Weight = n_samples / (n_classes * class_count)
    n_estimators=200, random_state=42
)

# 2. SMOTE: generate synthetic minority samples
# smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Create up to 10% anomalies
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Adjust detection threshold toward higher recall
# (Prefer catching anomalies over false alarms in security contexts)
def threshold_by_recall(y_true, scores, target_recall=0.9):
    """Set threshold to achieve target recall"""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # Find threshold where recall >= target_recall
    valid = np.where(recall >= target_recall)[0]
    if len(valid) == 0:
        return thresholds[0]
    return thresholds[valid[-1]]

# 4. Ensemble: combine multiple detectors (reduces FPR)
def anomaly_ensemble_score(X, detectors):
    """Average normalized scores from multiple detectors"""
    scores_all = []
    for det in detectors:
        s = det.decision_function(X)
        # Normalize to [0,1]
        s = (s - s.min()) / (s.max() - s.min() + 1e-10)
        scores_all.append(s)
    return np.mean(scores_all, axis=0)
```

---

## Semi-Supervised and Unsupervised Approaches

```python
# ── Semi-supervised: novelty detection ───────────────────────────────────────
# Train on known normal; detect novel points at test time

# One-Class SVM (novelty=True in LOF, or One-Class SVM)
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

# EllipticEnvelope: fits a Gaussian to normal data, flags points outside ellipse
# Best when normal data is approximately Gaussian
ee = EllipticEnvelope(contamination=0.05, support_fraction=0.9)
# ee.fit(X_train_normal)  # Only normal training data
# scores = -ee.decision_function(X_test)

# ── Label Propagation for semi-supervised refinement ─────────────────────────
from sklearn.semi_supervised import LabelPropagation

def semi_supervised_anomaly(X_all, y_partial, k_neighbors=10):
    """
    Semi-supervised: propagate known labels to unlabeled points.
    y_partial: -1 = unknown, 0 = known normal, 1 = known anomaly
    """
    lp = LabelPropagation(kernel='knn', n_neighbors=k_neighbors)
    lp.fit(X_all, y_partial)
    return lp.predict(X_all), lp.predict_proba(X_all)[:, 1]
```

---

## Real-World Applications

### Application 1: Credit Card Fraud Detection

```python
"""
Credit card fraud detection pipeline:
- Highly imbalanced (0.1-0.5% fraud rate)
- Need high recall at acceptable precision
- Interpretable alerts for human review
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def fraud_detection_pipeline(df, label_col='Class', contamination=0.002):
    """
    Two-stage fraud detection:
    1. Isolation Forest for unsupervised pre-screening
    2. Calibrated supervised model for final score
    """
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stage 1: Isolation Forest
    iforest = IsolationForest(contamination=contamination, n_jobs=-1, random_state=42)
    iforest.fit(X_scaled[y == 0])  # Train on normal only if available
    if_scores = -iforest.score_samples(X_scaled)
    
    # Stage 2: Supervised model with class weights
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    rf_fraud = RandomForestClassifier(
        n_estimators=200,
        class_weight={0: 1, 1: 100},   # Heavy weight on fraud class
        max_depth=10,
        random_state=42, n_jobs=-1
    )
    rf_fraud.fit(X_train, y_train)
    y_prob_fraud = rf_fraud.predict_proba(X_test)[:, 1]
    
    # Use IF score to adjust threshold
    # High IF score + high RF probability = high confidence fraud
    return rf_fraud, scaler, y_prob_fraud, y_test

# Example usage (replace with actual fraud dataset):
# df = pd.read_csv('creditcard.csv')
# model, scaler, probs, labels = fraud_detection_pipeline(df)
# metrics = comprehensive_evaluation(labels, probs)
```

### Application 2: Network Intrusion Detection

```python
"""
Network intrusion detection using autoencoders.
Normal = benign traffic patterns; Anomaly = intrusion signatures.
"""
def network_ids_pipeline(X_normal_traffic, X_test_traffic, y_test_labels,
                           contamination=0.05):
    """
    Network IDS using autoencoder trained on normal traffic only.
    """
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal_traffic)
    X_test_scaled = scaler.transform(X_test_traffic)
    
    n_features = X_normal_traffic.shape[1]
    
    # Train autoencoder
    ae = DenoisingAutoencoder(
        input_dim=n_features,
        hidden_dims=[64, 32],
        latent_dim=8,
        noise_factor=0.05
    )
    
    X_t = torch.FloatTensor(X_normal_scaled)
    train_ds = TensorDataset(X_t)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    train_autoencoder(ae, loader, epochs=100, lr=1e-3)
    
    # Score test traffic
    X_test_t = torch.FloatTensor(X_test_scaled)
    scores = ae.reconstruction_error(X_test_t).numpy()
    
    # Set threshold at 99th percentile of normal traffic scores
    normal_scores = ae.reconstruction_error(X_t).numpy()
    threshold = np.percentile(normal_scores, 99)
    
    print(f"\nNetwork IDS Results:")
    print(f"Threshold (99th pct of normal): {threshold:.6f}")
    metrics = comprehensive_evaluation(y_test_labels, scores)
    
    return ae, scaler, scores, threshold, metrics
```

### Application 3: Manufacturing Defect Detection

```python
"""
Manufacturing quality control: detect defective products from sensor readings.
Semi-supervised: many normal products, few/no labeled defects.
"""
def manufacturing_qc(X_production, X_holdout=None, y_holdout=None,
                      window_size=100, contamination=0.01):
    """
    Real-time quality control with streaming detection.
    """
    # Train detectors on first batch (assumed normal)
    X_train_prod = X_production[:window_size]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_prod)
    
    # Multiple detectors for reliability
    detectors_qc = {
        'iforest': IsolationForest(contamination=contamination, random_state=42),
        'ocsvm': OneClassSVM(kernel='rbf', nu=contamination * 2),
        'lof': LocalOutlierFactor(n_neighbors=15, novelty=True)
    }
    
    for det in detectors_qc.values():
        det.fit(X_train_scaled)
    
    # Stream production data
    alerts = []
    for i in range(window_size, len(X_production)):
        x_new = scaler.transform(X_production[i:i+1])
        
        votes = []
        for name, det in detectors_qc.items():
            pred = det.predict(x_new)[0]
            votes.append(1 if pred == -1 else 0)
        
        # Majority vote anomaly
        if sum(votes) >= 2:  # 2+ detectors agree
            alerts.append(i)
    
    print(f"Manufacturing QC alerts: {len(alerts)} potential defects")
    return alerts, detectors_qc, scaler
```

---

## Complete Code Examples

### Complete Anomaly Detection Benchmark

```python
"""
Full benchmark comparing multiple anomaly detection methods on a synthetic dataset
with ground-truth labels for evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

# ── Data generation ───────────────────────────────────────────────────────────
np.random.seed(42)
n_normal = 1000
n_anomaly = 50
n_features = 10

# Normal: from 3 clusters in 10D
X_normal_b, _ = make_blobs(n_samples=n_normal, n_features=n_features,
                             centers=3, cluster_std=0.8, random_state=42)

# Anomalies: uniform random in wider range
X_anom_b = np.random.uniform(-8, 8, (n_anomaly, n_features))

X_all_b = np.vstack([X_normal_b, X_anom_b])
y_true_b = np.array([0] * n_normal + [1] * n_anomaly)

# Standardize
scaler_b = StandardScaler()
X_all_scaled_b = scaler_b.fit_transform(X_all_b)
X_normal_scaled_b = X_all_scaled_b[:n_normal]

# ── Benchmark ─────────────────────────────────────────────────────────────────
benchmark_results = {}

# 1. Isolation Forest
iforest_b = IsolationForest(contamination=n_anomaly/(n_normal+n_anomaly), random_state=42)
iforest_b.fit(X_all_scaled_b)
scores_if = -iforest_b.decision_function(X_all_scaled_b)
benchmark_results['Isolation Forest'] = scores_if

# 2. LOF
lof_b = LocalOutlierFactor(n_neighbors=20, contamination=n_anomaly/(n_normal+n_anomaly))
lof_b.fit_predict(X_all_scaled_b)
scores_lof = -lof_b.negative_outlier_factor_
benchmark_results['LOF'] = scores_lof

# 3. One-Class SVM (trained on normal only)
ocsvm_b = OneClassSVM(kernel='rbf', nu=0.05)
ocsvm_b.fit(X_normal_scaled_b)
scores_ocsvm = -ocsvm_b.decision_function(X_all_scaled_b)
benchmark_results['One-Class SVM'] = scores_ocsvm

# 4. Autoencoder (trained on normal only)
X_train_t_b = torch.FloatTensor(X_normal_scaled_b)
ae_b = DenoisingAutoencoder(input_dim=n_features, hidden_dims=[32, 16], latent_dim=4,
                             noise_factor=0.1, dropout=0.1)
ds_b = TensorDataset(X_train_t_b)
dl_b = DataLoader(ds_b, batch_size=64, shuffle=True)

opt_b = torch.optim.Adam(ae_b.parameters(), lr=1e-3)
for epoch in range(100):
    ae_b.train()
    for (bx,) in dl_b:
        opt_b.zero_grad()
        F.mse_loss(ae_b(bx), bx).backward()
        opt_b.step()

X_all_t_b = torch.FloatTensor(X_all_scaled_b)
scores_ae = ae_b.reconstruction_error(X_all_t_b).numpy()
benchmark_results['Autoencoder'] = scores_ae

# ── Compare results ───────────────────────────────────────────────────────────
print(f"\n{'Method':<20} {'AUROC':>8} {'AUPRC':>8}")
print('-' * 40)
for name, scores in benchmark_results.items():
    auroc = roc_auc_score(y_true_b, scores)
    auprc = average_precision_score(y_true_b, scores)
    print(f"{name:<20} {auroc:>8.4f} {auprc:>8.4f}")

# ── Plot score distributions ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, scores) in zip(axes.flatten(), benchmark_results.items()):
    # Normalize for comparison
    s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    ax.hist(s_norm[y_true_b == 0], bins=40, alpha=0.6, color='steelblue', density=True, label='Normal')
    ax.hist(s_norm[y_true_b == 1], bins=20, alpha=0.6, color='red', density=True, label='Anomaly')
    ax.set_title(name)
    ax.set_xlabel('Normalized Score')
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Anomaly Score Distributions by Method')
plt.tight_layout()
plt.show()
```

---

## Pitfalls and Common Mistakes

### 1. Training on Contaminated Data

**Problem**: Training unsupervised methods (LOF, Isolation Forest, autoencoders) on data that includes anomalies. Anomalies become part of the "normal" model.

**Fix**: Use semi-supervised training (fit only on verified normal data when available), or use robust methods (e.g., robust covariance for Mahalanobis). Validate that contamination assumption matches your data.

### 2. Threshold Selection Without Validation

**Problem**: Setting anomaly threshold arbitrarily (e.g., top 5%) without considering business cost of false positives vs. false negatives.

**Fix**: Use precision-recall curves, set threshold by target recall (e.g., catch 90% of anomalies), or use cost-sensitive thresholds. Recalibrate periodically.

### 3. Ignoring Feature Scaling

**Problem**: Distance-based methods (LOF, Isolation Forest, Mahalanobis, One-Class SVM) are sensitive to feature scales. Unscaled data biases toward high-variance features.

**Fix**: Always standardize (zero mean, unit variance) or robustly scale (median, MAD) before fitting.

### 4. Representation Collapse in Deep SVDD

**Problem**: Network maps all inputs to the same point; training loss goes to zero but the model is useless.

**Fix**: Fix center \(c\); use LeakyReLU; avoid bias in final layer; pretrain with autoencoder then fine-tune; check that latent variance is not near zero.

### 5. Autoencoder Reconstructs Anomalies Well

**Problem**: Some anomalies (e.g., blurred images, minor corruptions) can have low reconstruction error if they share structure with normal data.

**Fix**: Use Denoising AE (noise forces learning robust features), combine with latent norm (e.g., \(\|z\|^2\)) as additional score, or use Deep SVDD which does not rely on reconstruction.

### 6. High-Dimensional Curse

**Problem**: LOF, k-NN, and distance-based methods degrade in high dimensions (distances become less discriminative).

**Fix**: Use Isolation Forest (robust to high dimensions), autoencoders (learn low-dim manifold), or dimensionality reduction (PCA, UMAP) before distance-based methods.

### 7. Concept Drift

**Problem**: "Normal" behavior changes over time (e.g., seasonal patterns, policy changes). Static models become outdated.

**Fix**: Retrain periodically, use online/streaming methods (River, ADWIN), or detect drift and trigger retraining.

---

## Summary Table

| Method | Type | Scalability | Best For | Key Parameters |
|--------|------|-------------|----------|----------------|
| Z-score / IQR | Statistical | Very High | Univariate, simple data | threshold |
| Mahalanobis | Statistical | High | Multivariate Gaussian | covariance estimator |
| LOF | Density | Medium | Local clusters, varying density | n_neighbors |
| Isolation Forest | Tree-based | High | High-dimensional, mixed | n_estimators, contamination |
| One-Class SVM | Kernel | Low | Small, clean normal data | kernel, nu, gamma |
| Autoencoder | Deep Learning | High | High-dim, image, tabular | architecture, latent_dim |
| VAE | Deep Learning | High | Probabilistic score | beta, latent_dim |
| Deep SVDD | Deep Learning | High | Compact normal manifold | rep_dim |
| LSTM AE | Sequential | Medium | Time series | seq_len, hidden_dim |
| USAD | Sequential | Medium | Robust time series | latent_dim |
| Prophet | Statistical | Medium | Univariate TS with seasonality | interval_width |
| SPOT | Statistical | Very High | Streaming, EVT-based | q, level |

**Recommended Libraries:**
- `scikit-learn`: IF, LOF, OCSVM, EllipticEnvelope
- `PyOD`: 40+ algorithms with unified API
- `River`: Online/streaming anomaly detection
- `alibi-detect`: Deep learning methods + drift detection
- `PyTorch`: Custom AE, VAE, LSTM-AE, Deep SVDD

---

## References

- Breunig, M. M., et al. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD.
- Liu, F. T., et al. (2008). *Isolation Forest*. ICDM.
- Schölkopf, B., et al. (1999). *Support Vector Method for Novelty Detection*. NeurIPS.
- Tax, D. M. J., & Duin, R. P. W. (2004). *Support Vector Data Description*. Machine Learning.
- Ruff, L., et al. (2018). *Deep One-Class Classification*. ICML.
- Zong, B., et al. (2018). *Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection*. ICLR.
- Zhao, Y., et al. (2019). *PyOD: A Python Toolbox for Scalable Outlier Detection*. JMLR.
- Siffer, A., et al. (2017). *Anomaly Detection in Streams with Extreme Value Theory*. KDD (SPOT).
