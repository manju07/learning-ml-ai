# Machine Learning: Comprehensive Guide from Fundamentals to Advanced

## Table of Contents
1. [ML Paradigms](#1-ml-paradigms)
2. [Bias-Variance Tradeoff](#2-bias-variance-tradeoff)
3. [Overfitting, Underfitting & Regularization](#3-overfitting-underfitting--regularization)
4. [Feature Engineering](#4-feature-engineering)
5. [Supervised Learning Algorithms](#5-supervised-learning-algorithms)
6. [Ensemble Methods & Boosting](#6-ensemble-methods--boosting)
7. [Clustering (Unsupervised)](#7-clustering-unsupervised)
8. [Dimensionality Reduction](#8-dimensionality-reduction)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Cross-Validation Strategies](#10-cross-validation-strategies)
11. [Pipelines & Production Patterns](#11-pipelines--production-patterns)
12. [Hyperparameter Tuning](#12-hyperparameter-tuning)

---

## 1. ML Paradigms

Machine Learning algorithms learn patterns from data. Different paradigms differ in **what signal is used to learn**.

### 1.1 Supervised Learning

The algorithm learns a mapping \( f: X \to Y \) from labeled pairs \((x_i, y_i)\).

**Loss minimization:**
\[
\hat{\theta} = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n L(f_\theta(x_i), y_i)
\]

| Task | Output Type | Examples |
|------|-------------|---------|
| Classification | Discrete label | Spam detection, image recognition |
| Regression | Continuous value | House prices, stock forecasts |
| Structured prediction | Sequence/graph | NLP parsing, protein folding |

### 1.2 Unsupervised Learning

No labels — learn the structure of \( P(X) \).

- **Clustering**: Partition data into groups (K-Means, DBSCAN, GMM)
- **Density estimation**: Model the data distribution
- **Dimensionality reduction**: Find compact representation (PCA, UMAP)
- **Generative models**: Learn to generate new samples (VAE, GAN)

### 1.3 Semi-Supervised Learning

Uses a small labeled set \( \mathcal{L} \) plus a large unlabeled set \( \mathcal{U} \).

**Objective:**
\[
\mathcal{L}_{total} = \mathcal{L}_{supervised}(\mathcal{L}) + \lambda \cdot \mathcal{L}_{unsupervised}(\mathcal{U})
\]

Common techniques:
- **Label propagation**: Spread labels through a graph of similar samples
- **Pseudo-labeling**: Train on labeled data, use model to generate labels for unlabeled data, then retrain
- **Consistency regularization**: Model output should be similar under input perturbations

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Mask most labels (only 10% labeled)
rng = np.random.RandomState(42)
random_unlabeled = rng.rand(y.shape[0]) < 0.9
y_partial = y.copy()
y_partial[random_unlabeled] = -1  # -1 means unlabeled

lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
lp.fit(X, y_partial)

# Predict on all data
y_pred = lp.predict(X)
labeled_mask = y_partial != -1
acc_labeled = np.mean(y_pred[labeled_mask] == y[labeled_mask])
print(f"Accuracy on labeled: {acc_labeled:.4f}")
```

### 1.4 Self-Supervised Learning

Creates **pretext tasks** from unlabeled data to learn useful representations. No human annotation needed.

**Examples:**
- **Contrastive learning** (SimCLR, MoCo): Pull together different views of same image, push apart different images
- **BERT** (NLP): Masked language modeling — predict masked words
- **GPT**: Autoregressive prediction — predict next token

Contrastive loss (NT-Xent):
\[
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}
\]

```python
import numpy as np

def cosine_similarity(z1, z2):
    return np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))

def nt_xent_loss(z, tau=0.5):
    """NT-Xent contrastive loss. z has shape (2N, d), pairs are (z[i], z[i+N])."""
    N = len(z) // 2
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    sim = z @ z.T  # (2N, 2N) similarity matrix
    sim /= tau

    # Mask out self-similarity
    mask = np.eye(2 * N, dtype=bool)
    sim[mask] = -1e9

    labels = np.concatenate([np.arange(N, 2 * N), np.arange(N)])
    loss = 0
    for i in range(2 * N):
        exp_sim = np.exp(sim[i])
        loss -= sim[i, labels[i]] - np.log(exp_sim.sum())
    return loss / (2 * N)
```

### 1.5 Reinforcement Learning

An **agent** interacts with an **environment**, receiving **rewards** for actions.

**Markov Decision Process (MDP):**
- State space \( \mathcal{S} \)
- Action space \( \mathcal{A} \)
- Transition function \( P(s' | s, a) \)
- Reward function \( R(s, a) \)
- Discount factor \( \gamma \in [0, 1) \)

**Goal:** Learn policy \( \pi(a|s) \) that maximizes expected cumulative reward:
\[
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\]

**Q-Learning update:**
\[
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
\]

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

# Simple GridWorld simulation
agent = QLearningAgent(n_states=16, n_actions=4)
# Training loop would go here...
print("Q-table shape:", agent.Q.shape)
```

### 1.6 Comparison Table

| Paradigm | Data Required | Feedback Signal | Typical Use Case |
|----------|--------------|-----------------|-----------------|
| Supervised | Labeled (X, y) | Ground truth labels | Classification, regression |
| Unsupervised | Unlabeled X | None | Clustering, compression |
| Semi-supervised | Mostly unlabeled | Few labels | Low-label settings |
| Self-supervised | Unlabeled X | Pretext task | Pre-training large models |
| Reinforcement | Environment | Reward signal | Games, robotics, control |

---

## 2. Bias-Variance Tradeoff

### 2.1 Decomposition

For regression, the expected squared error decomposes as:
\[
\mathbb{E}[(f(x) - \hat{f}(x))^2] = \underbrace{\text{Bias}^2[\hat{f}(x)]}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{model sensitivity}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
\]

Where:
- **Bias** = \( \mathbb{E}[\hat{f}(x)] - f(x) \) — how far the average prediction is from truth
- **Variance** = \( \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] \) — how much predictions vary across datasets
- **\( \sigma^2 \)** = noise inherent in data; cannot be reduced

```
Model Complexity →
Low                    High
|----|----|----|----|----|----|
Underfitting          Overfitting
High Bias             High Variance
Low Variance          Low Bias
```

### 2.2 Mathematical Derivation

For a prediction \( \hat{f}(x) \) trained on a dataset \( \mathcal{D} \):

\[
\mathbb{E}_\mathcal{D}[(y - \hat{f}(x))^2]
= \mathbb{E}[(y - f(x))^2] + \mathbb{E}[(f(x) - \mathbb{E}[\hat{f}])^2] + \mathbb{E}[(\mathbb{E}[\hat{f}] - \hat{f})^2]
= \sigma^2 + \text{Bias}^2 + \text{Variance}
\]

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def true_function(x):
    return np.sin(x) + 0.1 * x

def generate_data(n=30, noise=0.3, seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0, 2 * np.pi, n)
    y = true_function(x) + rng.normal(0, noise, n)
    return x.reshape(-1, 1), y

def bias_variance_demo(degrees, n_datasets=50, n_samples=30):
    x_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    y_true = true_function(x_test.flatten())

    results = {}
    for degree in degrees:
        preds = []
        for seed in range(n_datasets):
            X_train, y_train = generate_data(n=n_samples, seed=seed)
            model = Pipeline([
                ('poly', PolynomialFeatures(degree)),
                ('lr', LinearRegression())
            ])
            model.fit(X_train, y_train)
            preds.append(model.predict(x_test))

        preds = np.array(preds)  # (n_datasets, n_test)
        mean_pred = preds.mean(axis=0)
        bias_sq = ((mean_pred - y_true) ** 2).mean()
        variance = preds.var(axis=0).mean()

        results[degree] = {'bias_sq': bias_sq, 'variance': variance,
                           'total': bias_sq + variance}
        print(f"Degree {degree:2d}: Bias²={bias_sq:.4f}, Var={variance:.4f}, "
              f"Total={bias_sq + variance:.4f}")

    return results

results = bias_variance_demo([1, 3, 5, 10, 15])
```

### 2.3 Visual Intuition

```
                    High Variance
                        ^
                        |
    (dartboard analogy) |
Low Bias ───────────────+──────────── High Bias
                        |
                        |
                    Low Variance

Best model: Low Bias AND Low Variance (center of dartboard, clustered)
High Bias, Low Variance: All darts cluster away from center
Low Bias, High Variance: Darts scattered around center
High Bias, High Variance: Worst — scattered and off-center
```

---

## 3. Overfitting, Underfitting & Regularization

### 3.1 Overfitting vs Underfitting

| Condition | Training Error | Validation Error | Remedy |
|-----------|---------------|-----------------|--------|
| Underfitting | High | High | More complex model, better features |
| Overfitting | Low | High | Regularization, more data, simpler model |
| Good fit | Low | Low ≈ Train | — |

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 100))
y = 2 * np.sin(X) + 0.5 * X + np.random.normal(0, 0.8, 100)
X = X.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

for deg in [1, 3, 10, 20]:
    pipe = Pipeline([('poly', PolynomialFeatures(deg)), ('lr', LinearRegression())])
    pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    val_mse = mean_squared_error(y_val, pipe.predict(X_val))
    status = "OK" if abs(train_mse - val_mse) < 1 else ("OVERFIT" if val_mse > 2 * train_mse else "UNDERFIT")
    print(f"Degree {deg:2d}: Train MSE={train_mse:.4f}, Val MSE={val_mse:.4f}  [{status}]")
```

### 3.2 L1 Regularization (Lasso)

Adds the **L1 norm** of weights as penalty:
\[
\mathcal{L}_{Lasso} = \underbrace{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}_{\text{MSE}} + \lambda \sum_{j=1}^p |w_j|
\]

**Key properties:**
- Drives some weights **exactly to zero** → automatic feature selection
- Corresponds to a **Laplace prior** in Bayesian view: \( P(w) \propto e^{-\lambda |w|} \)
- Non-differentiable at 0; requires subgradient or coordinate descent

```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y, coef_true = make_regression(n_samples=200, n_features=50, n_informative=10,
                                   noise=10, coef=True, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso — sparse solution
lasso = Lasso(alpha=0.5, max_iter=10000)
lasso.fit(X_scaled, y)
n_zero = np.sum(lasso.coef_ == 0)
print(f"Lasso: {n_zero}/{len(lasso.coef_)} coefficients are exactly zero")
```

### 3.3 L2 Regularization (Ridge)

Adds the **squared L2 norm** as penalty:
\[
\mathcal{L}_{Ridge} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p w_j^2
\]

**Closed-form solution:**
\[
\hat{w}_{Ridge} = (X^TX + \lambda I)^{-1} X^T y
\]

**Key properties:**
- Shrinks all weights toward zero, but **rarely to exactly zero**
- Corresponds to a **Gaussian prior**: \( P(w) \propto e^{-\lambda w^2/2} \)
- \( \lambda I \) makes \( X^TX \) invertible even when singular — numerical stability!

```python
# Ridge — dense but small weights
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
n_zero_ridge = np.sum(np.abs(ridge.coef_) < 1e-4)
print(f"Ridge: {n_zero_ridge}/{len(ridge.coef_)} near-zero coefficients")
print(f"Ridge weight L2 norm: {np.linalg.norm(ridge.coef_):.4f}")
print(f"Lasso weight L2 norm: {np.linalg.norm(lasso.coef_):.4f}")
```

### 3.4 ElasticNet (L1 + L2 Combined)

\[
\mathcal{L}_{EN} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_j |w_j| + \lambda_2 \sum_j w_j^2
\]

Or equivalently with `l1_ratio = r`:
\[
\mathcal{L}_{EN} = \frac{1}{n}\text{MSE} + \lambda \left[ r \|w\|_1 + \frac{(1-r)}{2} \|w\|_2^2 \right]
\]

Best of both worlds: sparsity from L1, grouping effect from L2.

```python
# ElasticNet — combines sparsity and grouping
enet = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000)
enet.fit(X_scaled, y)
n_zero_en = np.sum(enet.coef_ == 0)
print(f"ElasticNet: {n_zero_en}/{len(enet.coef_)} coefficients are zero")
```

### 3.5 Regularization Paths

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, ridge_regression

alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y, eps=1e-4, n_alphas=100)

plt.figure(figsize=(10, 6))
plt.semilogx(alphas_lasso, coefs_lasso.T)
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficients')
plt.title('Lasso Regularization Path')
plt.axvline(x=0.5, color='r', linestyle='--', label='Chosen alpha')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
```

### 3.6 Dropout (Neural Network Regularization)

During training, randomly zero out neurons with probability \( p \):
\[
\tilde{h}_j = h_j \cdot \text{Bernoulli}(1-p)
\]
During inference, multiply weights by \( (1-p) \) to maintain expected activations.

---

## 4. Feature Engineering

### 4.1 Handling Missing Values

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create data with missing values
np.random.seed(42)
df = pd.DataFrame({
    'age': [25, np.nan, 35, 28, np.nan, 45],
    'salary': [50000, 80000, np.nan, 60000, 75000, np.nan],
    'score': [8.5, 7.0, 9.0, np.nan, 8.0, 7.5]
})

# Strategy 1: Simple imputation
simple = SimpleImputer(strategy='median')
df_median = pd.DataFrame(simple.fit_transform(df), columns=df.columns)

# Strategy 2: KNN imputation
knn_imp = KNNImputer(n_neighbors=2)
df_knn = pd.DataFrame(knn_imp.fit_transform(df), columns=df.columns)

# Strategy 3: MICE (Multiple Imputation by Chained Equations)
mice = IterativeImputer(max_iter=10, random_state=42)
df_mice = pd.DataFrame(mice.fit_transform(df), columns=df.columns)

# Strategy 4: Add missingness indicator
df['age_missing'] = df['age'].isna().astype(int)
df['salary_missing'] = df['salary'].isna().astype(int)

print("Simple imputation:\n", df_median)
print("\nKNN imputation:\n", df_knn)
```

### 4.2 Feature Scaling

| Method | Formula | Best for |
|--------|---------|---------|
| Standardization | \( z = \frac{x - \mu}{\sigma} \) | Gaussian-like data, SVM, NN |
| Min-Max | \( z = \frac{x - x_{min}}{x_{max} - x_{min}} \) | Bounded data, KNN |
| Robust | \( z = \frac{x - Q_2}{Q_3 - Q_1} \) | Data with outliers |
| Log transform | \( z = \log(1 + x) \) | Right-skewed data |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
import numpy as np

# Data with outliers
X = np.array([[1], [2], [3], [4], [5], [100]])  # Outlier at 100

scaler_std = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_robust = RobustScaler()
scaler_power = PowerTransformer(method='yeo-johnson')

print("Original:      ", X.flatten())
print("Standardized:  ", scaler_std.fit_transform(X).flatten().round(3))
print("Min-Max:       ", scaler_minmax.fit_transform(X).flatten().round(3))
print("Robust:        ", scaler_robust.fit_transform(X).flatten().round(3))
print("Power:         ", scaler_power.fit_transform(X).flatten().round(3))
```

### 4.3 Categorical Encoding

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce  # pip install category_encoders

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'size': ['small', 'medium', 'large', 'medium', 'small'],
    'target': [1, 0, 1, 0, 1]
})

# --- Ordinal Encoding (for ordered categories) ---
ord_enc = OrdinalEncoder(categories=[['small', 'medium', 'large']])
df['size_ord'] = ord_enc.fit_transform(df[['size']])

# --- One-Hot Encoding (for nominal categories) ---
ohe = OneHotEncoder(sparse_output=False, drop='first')
color_ohe = ohe.fit_transform(df[['color']])
color_cols = ohe.get_feature_names_out(['color'])
df_ohe = pd.concat([df, pd.DataFrame(color_ohe, columns=color_cols)], axis=1)
print("One-hot encoded:\n", df_ohe[['color'] + list(color_cols)].head())

# --- Target Encoding (mean of target per category) ---
# Requires category_encoders library
# te = ce.TargetEncoder(cols=['color'])
# df['color_target'] = te.fit_transform(df['color'], df['target'])

# --- Binary Encoding (efficient for high-cardinality) ---
# be = ce.BinaryEncoder(cols=['color'])
# df_binary = be.fit_transform(df)
```

### 4.4 Feature Selection

#### Filter Methods (model-independent)

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    chi2, VarianceThreshold
)
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.1)
X_var = var_thresh.fit_transform(X)
print(f"After variance threshold: {X_var.shape[1]} features (from {X.shape[1]})")

# F-test (ANOVA) — linear correlation
selector_f = SelectKBest(f_classif, k=10)
X_f = selector_f.fit_transform(X, y)
f_scores = selector_f.scores_
top_f = X_f.shape[1]

# Mutual Information — non-linear
selector_mi = SelectKBest(mutual_info_classif, k=10)
X_mi = selector_mi.fit_transform(X, y)
mi_scores = selector_mi.scores_

print(f"Top 10 features by F-test selected")
print(f"Top 10 features by MI selected")
```

#### Wrapper Methods (model-dependent)

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=10)
rfe.fit(X, y)
selected_mask = rfe.support_
print(f"RFE selected features: {selected_mask.sum()}")

# RFECV — automatically selects optimal number of features via CV
rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=50, random_state=42),
              step=1, cv=5, scoring='accuracy', n_jobs=-1)
rfecv.fit(X, y)
print(f"RFECV optimal feature count: {rfecv.n_features_}")
```

#### Embedded Methods (learned during training)

```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# L1-based selection (Lasso)
lasso_cv = LassoCV(cv=5, max_iter=5000, random_state=42)
lasso_cv.fit(X, y)
selector_lasso = SelectFromModel(lasso_cv, prefit=True)
X_lasso = selector_lasso.transform(X)
print(f"Lasso selected: {X_lasso.shape[1]} features")

# Tree-based importance selection
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
selector_rf = SelectFromModel(rf, prefit=True, threshold='median')
X_rf_selected = selector_rf.transform(X)
print(f"RF importance selected: {X_rf_selected.shape[1]} features")
```

### 4.5 Advanced Feature Engineering

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X[:, :5])
print(f"Polynomial features shape: {X_poly.shape}")

# Date/time features
dates = pd.to_datetime(['2024-01-15', '2024-03-22', '2024-07-08'])
df_time = pd.DataFrame({'date': dates})
df_time['year'] = df_time['date'].dt.year
df_time['month'] = df_time['date'].dt.month
df_time['dayofweek'] = df_time['date'].dt.dayofweek
df_time['quarter'] = df_time['date'].dt.quarter
df_time['is_weekend'] = df_time['dayofweek'].isin([5, 6]).astype(int)

# Cyclic encoding for periodic features (month, hour, etc.)
df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)

print(df_time)
```

---

## 5. Supervised Learning Algorithms

### 5.1 Linear Regression

**Model:** \( \hat{y} = w^T x + b \)

**Objective (OLS):**
\[
\hat{w} = \arg\min_w \|y - Xw\|^2 = (X^TX)^{-1}X^Ty
\]

**Assumptions (BLUE — Gauss-Markov):**
1. Linearity: \( y = Xw + \epsilon \)
2. No multicollinearity
3. Homoscedasticity: \( \text{Var}(\epsilon) = \sigma^2 I \)
4. Independence of errors
5. Zero-mean errors: \( \mathbb{E}[\epsilon] = 0 \)

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

np.random.seed(42)
n = 500
X = np.column_stack([
    np.random.randn(n),
    np.random.randn(n) * 2,
    np.random.randn(n) + 1
])
true_w = np.array([3.0, -1.5, 2.0])
y = X @ true_w + 5 + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# OLS Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"True weights:      {true_w}")
print(f"Learned weights:   {lr.coef_.round(4)}")
print(f"Intercept: {lr.intercept_:.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# Assumptions check
residuals = y_test - y_pred
print(f"\nResidual mean (should ≈ 0): {residuals.mean():.6f}")
print(f"Residual std: {residuals.std():.4f}")
```

### 5.2 Logistic Regression

**Model:**
\[
P(y=1 | x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
\]

**Loss (Binary Cross-Entropy / Negative Log-Likelihood):**
\[
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]
\]

**Gradient:**
\[
\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} X^T (\hat{p} - y)
\]

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, average_precision_score)
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                           n_redundant=5, class_sep=0.8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multiple solvers and penalties
models = {
    'L2 (liblinear)': LogisticRegression(penalty='l2', C=1.0, solver='liblinear'),
    'L1 (liblinear)': LogisticRegression(penalty='l1', C=1.0, solver='liblinear'),
    'ElasticNet (saga)': LogisticRegression(penalty='elasticnet', C=1.0,
                                            solver='saga', l1_ratio=0.5, max_iter=1000),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    n_nonzero = np.sum(np.abs(model.coef_[0]) > 1e-6)
    print(f"{name}: AUC={auc:.4f}, Non-zero coefs={n_nonzero}/{X.shape[1]}")

# Multi-class (one-vs-rest vs multinomial)
from sklearn.datasets import load_iris
X_iris, y_iris = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
lr_multi.fit(X_tr, y_tr)
print(f"\nMulti-class accuracy: {lr_multi.score(X_te, y_te):.4f}")
print(classification_report(y_te, lr_multi.predict(X_te)))
```

### 5.3 Decision Trees

**Splitting criteria:**

*Gini Impurity* (classification):
\[
G = 1 - \sum_{k=1}^K p_k^2
\]

*Entropy* (classification):
\[
H = -\sum_{k=1}^K p_k \log_2 p_k
\]

*Variance reduction* (regression):
\[
\Delta = \text{Var}(parent) - \frac{n_L}{n}\text{Var}(left) - \frac{n_R}{n}\text{Var}(right)
\]

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gini vs Entropy
for criterion in ['gini', 'entropy']:
    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)
    print(f"Criterion={criterion}: Accuracy={dt.score(X_test, y_test):.4f}, "
          f"Tree depth={dt.get_depth()}, Leaves={dt.get_n_leaves()}")

# Visualize tree rules
dt_best = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_best.fit(X_train, y_train)
print("\nDecision Tree Rules:")
print(export_text(dt_best, feature_names=feature_names))

# Feature importance
print("\nFeature importances:")
for fname, imp in zip(feature_names, dt_best.feature_importances_):
    print(f"  {fname}: {imp:.4f}")
```

### 5.4 K-Nearest Neighbors (KNN)

**Prediction:** Given query point \( x \), find \( k \) nearest neighbors \( \mathcal{N}(x) \):
- *Classification:* Majority vote: \( \hat{y} = \text{mode}\{y_i : i \in \mathcal{N}(x)\} \)
- *Regression:* Average: \( \hat{y} = \frac{1}{k}\sum_{i \in \mathcal{N}(x)} y_i \)

**Distance metrics:**
- Euclidean: \( d(a,b) = \|a - b\|_2 \)
- Manhattan: \( d(a,b) = \|a - b\|_1 \)
- Minkowski: \( d(a,b) = \|a - b\|_p \)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN is sensitive to scale — must standardize!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Grid search over k and distance metric
param_grid = {
    'n_neighbors': [3, 5, 7, 11, 15, 21],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

gs = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train_sc, y_train)
print(f"Best KNN params: {gs.best_params_}")
print(f"Best CV accuracy: {gs.best_score_:.4f}")
print(f"Test accuracy: {gs.score(X_test_sc, y_test):.4f}")
```

### 5.5 Support Vector Machines (SVM)

**Hard-margin SVM** (linearly separable):
\[
\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 \; \forall i
\]

**Soft-margin SVM** (with slack variables \( \xi_i \)):
\[
\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0
\]

**Kernel trick:** Replace \( x_i^T x_j \) with \( K(x_i, x_j) = \phi(x_i)^T \phi(x_j) \)

| Kernel | Formula | Use Case |
|--------|---------|---------|
| Linear | \( x^T z \) | Linearly separable |
| Polynomial | \( (\gamma x^T z + r)^d \) | Polynomial boundaries |
| RBF/Gaussian | \( e^{-\gamma\|x-z\|^2} \) | General non-linear |
| Sigmoid | \( \tanh(\gamma x^T z + r) \) | Neural-network-like |

```python
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM pipeline (scaling is crucial!)
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])
svm_pipe.fit(X_train, y_train)
print(f"SVM (RBF) accuracy: {svm_pipe.score(X_test, y_test):.4f}")

# Hyperparameter search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'svm__C': loguniform(0.01, 100),
    'svm__gamma': loguniform(0.001, 10),
    'svm__kernel': ['rbf', 'poly']
}
rs = RandomizedSearchCV(svm_pipe, param_dist, n_iter=20, cv=5, scoring='accuracy',
                        n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print(f"Best SVM params: {rs.best_params_}")
print(f"Best CV accuracy: {rs.best_score_:.4f}")
```

### 5.6 Naive Bayes

**Bayes rule with conditional independence assumption:**
\[
P(y | x_1, \ldots, x_p) \propto P(y) \prod_{j=1}^p P(x_j | y)
\]

**Variants:**

| Variant | Likelihood Model | Best for |
|---------|-----------------|---------|
| GaussianNB | Gaussian \( P(x_j\|y) = \mathcal{N}(\mu_{jy}, \sigma_{jy}^2) \) | Continuous features |
| MultinomialNB | Multinomial | Word counts (text) |
| BernoulliNB | Bernoulli | Binary features |
| ComplementNB | Complement of class | Imbalanced text |

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Gaussian NB for continuous features
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Gaussian NB accuracy: {gnb.score(X_test, y_test):.4f}")

# Multinomial NB for text
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics']
news_train = fetch_20newsgroups(subset='train', categories=categories)
news_test = fetch_20newsgroups(subset='test', categories=categories)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_news_train = tfidf.fit_transform(news_train.data)
X_news_test = tfidf.transform(news_test.data)

mnb = MultinomialNB(alpha=0.1)  # Laplace smoothing
mnb.fit(X_news_train, news_train.target)
print(f"Multinomial NB (text) accuracy: {mnb.score(X_news_test, news_test.target):.4f}")
```

---

## 6. Ensemble Methods & Boosting

### 6.1 Random Forest

**Algorithm:**
1. Draw \( B \) bootstrap samples from training data
2. For each bootstrap sample, fit a decision tree:
   - At each split, consider only \( m = \sqrt{p} \) (classification) or \( m = p/3 \) (regression) random features
3. Aggregate: majority vote (classification) or mean (regression)

**Why it works:**
- Each tree has high variance (deep, unpruned) but low bias
- Averaging \( B \) correlated trees reduces variance by factor \( \approx 1/B \cdot (1 + (B-1)\rho) \) where \( \rho \) is pairwise correlation
- Feature randomness decorrelates trees → larger variance reduction

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import numpy as np

X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,        # Fully grown trees
    max_features='sqrt',   # sqrt(p) features per split
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,        # Out-of-bag error estimate
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"Test accuracy:     {rf.score(X_test, y_test):.4f}")
print(f"OOB accuracy:      {rf.oob_score_:.4f}")

# Feature importance (impurity-based)
imp_impurity = rf.feature_importances_

# Permutation importance (more reliable, model-agnostic)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42)

print(f"\nTop 5 features by impurity importance:")
top5 = np.argsort(imp_impurity)[-5:][::-1]
for i in top5:
    print(f"  Feature {i}: {imp_impurity[i]:.4f}")
```

### 6.2 Gradient Boosting Theory

**General form:** Add trees sequentially to minimize loss.

Given current ensemble \( F_m(x) \), fit tree \( h_m \) to **pseudo-residuals** (negative gradient of loss):
\[
r_i^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_m}
\]

**Update:**
\[
F_{m+1}(x) = F_m(x) + \eta \cdot h_m(x)
\]

For squared loss: \( r_i = y_i - F_m(x_i) \) (literal residuals)
For log-loss: \( r_i = y_i - \sigma(F_m(x_i)) \)

### 6.3 XGBoost

Adds **second-order Taylor expansion** of loss:
\[
\mathcal{L}^{(m)} \approx \sum_{i=1}^n \left[ g_i h_m(x_i) + \frac{1}{2} h_i h_m(x_i)^2 \right] + \Omega(h_m)
\]

Where \( g_i = \partial_{\hat{y}} L(y_i, \hat{y}) \), \( h_i = \partial^2_{\hat{y}} L(y_i, \hat{y}) \), and:
\[
\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2
\]

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=5000, n_features=30, n_informative=15,
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with early stopping
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,          # Row subsampling
    colsample_bytree=0.8,   # Column subsampling per tree
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    gamma=0.1,              # Minimum loss reduction for split
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42
)

eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

print(f"XGBoost test accuracy: {xgb_model.score(X_test, y_test):.4f}")
print(f"Best iteration: {xgb_model.best_iteration}")
```

### 6.4 LightGBM

Key advantages over XGBoost:
- **GOSS** (Gradient-based One-Side Sampling): Keep high-gradient samples, randomly sample low-gradient
- **EFB** (Exclusive Feature Bundling): Bundle mutually exclusive features
- **Leaf-wise** growth instead of level-wise → deeper trees faster

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,          # Controls complexity (not max_depth)
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)
print(f"LightGBM test accuracy: {lgb_model.score(X_test, y_test):.4f}")
```

### 6.5 CatBoost

Specialized for **categorical features** — uses ordered boosting to avoid target leakage.

```python
from catboost import CatBoostClassifier, Pool

# Example with categorical features
import pandas as pd
X_cat = pd.DataFrame({
    'age': np.random.randint(18, 70, 1000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston'], 1000),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 1000),
    'income': np.random.uniform(20000, 200000, 1000)
})
y_cat = np.random.randint(0, 2, 1000)

cat_features = ['city', 'education']

cb_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    eval_metric='Accuracy',
    verbose=False,
    random_seed=42
)

X_tr, X_te = X_cat[:800], X_cat[800:]
y_tr, y_te = y_cat[:800], y_cat[800:]

train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
test_pool = Pool(X_te, y_te, cat_features=cat_features)

cb_model.fit(train_pool, eval_set=test_pool)
print(f"CatBoost accuracy: {cb_model.score(X_te, y_te):.4f}")
```

### 6.6 Comparison Table: Boosting Libraries

| Property | XGBoost | LightGBM | CatBoost |
|----------|---------|---------|---------|
| Tree growth | Level-wise | Leaf-wise | Symmetric |
| Categorical | Manual encoding | Native | Native (ordered) |
| Speed | Fast | Fastest | Medium |
| Memory | Medium | Low | Medium |
| Overfit risk | Medium | Higher (leaf-wise) | Lower |
| GPU support | Yes | Yes | Yes |
| Best for | Tabular data | Large datasets | Categorical features |

---

## 7. Clustering (Unsupervised)

### 7.1 K-Means

**Objective:** Minimize within-cluster sum of squares (WCSS/Inertia):
\[
J = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2
\]

**Algorithm (Lloyd's):**
1. Initialize centroids \( \mu_1, \ldots, \mu_K \) (random or K-Means++)
2. **E-step**: Assign each point to nearest centroid: \( c_i = \arg\min_k \|x_i - \mu_k\|^2 \)
3. **M-step**: Update centroids: \( \mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i \)
4. Repeat until convergence

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=1000, centers=4, n_features=2,
                        cluster_std=1.0, random_state=42)

# Elbow method + Silhouette for optimal K
inertias, silhouettes = [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

best_k = K_range[np.argmax(silhouettes)]
print(f"Best K by silhouette: {best_k}")

# Fit with best K
km_best = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
labels = km_best.fit_predict(X)

print(f"Silhouette Score:          {silhouette_score(X, labels):.4f}")
print(f"Davies-Bouldin Index:      {davies_bouldin_score(X, labels):.4f}  (lower = better)")
print(f"Calinski-Harabasz Index:   {calinski_harabasz_score(X, labels):.4f}  (higher = better)")
```

### 7.2 DBSCAN

**Density-based**: Finds arbitrarily shaped clusters; automatically identifies outliers.

**Parameters:**
- `eps` (\( \epsilon \)): Neighborhood radius
- `min_samples`: Minimum points to form a dense region

**Point types:**
- **Core point**: Has \( \geq \) `min_samples` points within \( \epsilon \)
- **Border point**: Within \( \epsilon \) of a core point, but fewer than `min_samples` neighbors
- **Noise point**: Not core or border — labeled -1

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# K-Means fails on non-convex clusters
km_moons = KMeans(n_clusters=2, random_state=42)
km_labels = km_moons.fit_predict(X_moons)

# DBSCAN succeeds
db = DBSCAN(eps=0.3, min_samples=10)
db_labels = db.fit_predict(X_moons)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
print(f"DBSCAN silhouette: {silhouette_score(X_moons[db_labels != -1], db_labels[db_labels != -1]):.4f}")
```

### 7.3 Hierarchical Clustering (Agglomerative)

**Algorithm:**
1. Start: each point is its own cluster
2. Merge the two closest clusters
3. Repeat until one cluster remains
4. Cut dendrogram at desired level

**Linkage methods:**

| Linkage | Distance formula | Properties |
|---------|-----------------|-----------|
| Single | \( \min_{i \in C_1, j \in C_2} d(i,j) \) | Chaining effect |
| Complete | \( \max_{i \in C_1, j \in C_2} d(i,j) \) | Compact clusters |
| Average | \( \frac{1}{\|C_1\|\|C_2\|}\sum d(i,j) \) | Compromise |
| Ward | Minimizes within-cluster variance | Usually best |

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X_small, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Dendrogram
Z = linkage(X_small, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_rotation=90)
plt.title('Dendrogram (Ward linkage)')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.axhline(y=5, color='r', linestyle='--', label='Cut here')
plt.legend()

# Fit with chosen n_clusters
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_small)
print(f"Agglomerative silhouette: {silhouette_score(X_small, labels_agg):.4f}")
```

### 7.4 Gaussian Mixture Models (GMM)

Models data as a mixture of \( K \) Gaussians:
\[
P(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
\]

Learned via **EM algorithm**:
- **E-step**: Compute responsibilities \( r_{ik} = P(z=k | x_i) \)
- **M-step**: Update \( \pi_k, \mu_k, \Sigma_k \) using weighted MLE

Advantages over K-Means:
- Soft assignments (probabilities, not hard labels)
- Models cluster shapes (not just spherical)
- Provides density estimates

```python
from sklearn.mixture import GaussianMixture

# GMM — soft probabilistic clustering
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',    # 'full', 'tied', 'diag', 'spherical'
    n_init=5,
    max_iter=200,
    random_state=42
)
gmm.fit(X)
labels_gmm = gmm.predict(X)
probs = gmm.predict_proba(X)  # Soft assignments

print(f"GMM weights (π_k):  {gmm.weights_.round(3)}")
print(f"GMM means:          {gmm.means_.round(3)}")
print(f"GMM BIC:            {gmm.bic(X):.2f}")
print(f"GMM AIC:            {gmm.aic(X):.2f}")

# Model selection by BIC
bic_scores = []
for k in range(1, 10):
    g = GaussianMixture(n_components=k, random_state=42).fit(X)
    bic_scores.append(g.bic(X))
best_k_bic = np.argmin(bic_scores) + 1
print(f"\nBest K by BIC: {best_k_bic}")
```

---

## 8. Dimensionality Reduction

### 8.1 Principal Component Analysis (PCA)

**Goal:** Find orthogonal directions of maximum variance.

**Steps:**
1. Center data: \( \tilde{X} = X - \bar{x} \)
2. Compute covariance: \( C = \frac{1}{n-1}\tilde{X}^T\tilde{X} \)
3. Eigendecompose: \( C = V \Lambda V^T \)
4. Project: \( Z = \tilde{X} V_k \) (top \( k \) eigenvectors)

**Equivalent via SVD:** \( \tilde{X} = U\Sigma V^T \), principal components = columns of \( V \).

**Variance explained:** \( \frac{\lambda_j}{\sum_i \lambda_i} \)

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import numpy as np

X_digits, y_digits = load_digits(return_X_y=True)

# Determine components needed for 95% variance
pca_full = PCA().fit(X_digits)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"Components for 95% variance: {n_95} (out of {X_digits.shape[1]})")

# Reduce dimensions
pca = PCA(n_components=n_95, random_state=42)
X_pca = pca.fit_transform(X_digits)
X_reconstructed = pca.inverse_transform(X_pca)

reconstruction_error = np.mean((X_digits - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5].round(4)}")

# PCA for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_digits)
print(f"2D PCA variance explained: {pca_2d.explained_variance_ratio_.sum():.4f}")
```

### 8.2 t-SNE

**Goal:** Preserve local neighborhood structure in low-dimensional embedding.

**Method:** Minimize KL divergence between pairwise similarities in high-D (\( P_{ij} \)) and low-D (\( Q_{ij} \)):
\[
\text{KL}(P \| Q) = \sum_{i \neq j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}
\]

Where \( P_{ij} \) uses Gaussian kernel (high-D) and \( Q_{ij} \) uses Student-t kernel (low-D, handles crowding problem).

**Hyperparameters:**
- `perplexity`: Effective number of neighbors (5–50 typical; try multiple)
- `learning_rate`: Step size (10–1000; sklearn default=200)
- `n_iter`: Usually 1000+

```python
from sklearn.manifold import TSNE

# t-SNE for visualization (not for feature engineering!)
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    n_iter=1000,
    init='pca',          # Better than random init
    random_state=42,
    n_jobs=-1
)
X_tsne = tsne.fit_transform(X_digits)
print(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")

# IMPORTANT: t-SNE distances between clusters are NOT meaningful!
# Only local structure (neighbor relationships) is preserved.
# Do NOT use t-SNE for preprocessing — only visualization.
```

### 8.3 UMAP

**Advantages over t-SNE:**
- Preserves **global** structure better
- Much faster (especially for large datasets)
- Can be used for preprocessing (not just visualization)
- Supports supervised/semi-supervised dimensionality reduction

```python
import umap  # pip install umap-learn

# Unsupervised UMAP
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,      # Controls local vs global structure
    min_dist=0.1,        # Minimum distance between points in embedding
    metric='euclidean',
    random_state=42
)
X_umap = reducer.fit_transform(X_digits)

# Supervised UMAP (uses labels to guide separation)
reducer_sup = umap.UMAP(n_components=2, random_state=42)
X_umap_sup = reducer_sup.fit_transform(X_digits, y=y_digits)

print("UMAP embedding computed.")
print(f"Shape: {X_umap.shape}")
```

### 8.4 Linear Discriminant Analysis (LDA)

**Goal:** Find directions that maximize **between-class** variance relative to **within-class** variance.

\[
J(W) = \frac{|W^T S_B W|}{|W^T S_W W|}
\]

Where \( S_B \) = between-class scatter, \( S_W \) = within-class scatter.

**Key difference from PCA:** LDA is **supervised** — uses class labels.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA — supervised dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_digits, y_digits)

print(f"LDA explained variance ratio: {lda.explained_variance_ratio_.round(4)}")
print(f"Max LDA components = n_classes - 1 = {len(np.unique(y_digits)) - 1}")

# LDA as classifier
lda_clf = LinearDiscriminantAnalysis()
cv_scores = cross_val_score(lda_clf, X_digits, y_digits, cv=5, scoring='accuracy')
print(f"LDA classifier CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 8.5 Comparison: Dimensionality Reduction Methods

| Method | Type | Global Structure | Speed | Interpretable | Best Use |
|--------|------|-----------------|-------|---------------|---------|
| PCA | Linear, unsup | Yes | Fast | Yes | Preprocessing, decorrelation |
| LDA | Linear, sup | Yes | Fast | Yes | Classification preprocessing |
| t-SNE | Non-linear, unsup | No | Slow | No | Visualization only |
| UMAP | Non-linear, unsup/sup | Partial | Medium | No | Visualization + preprocessing |
| Autoencoder | Non-linear, unsup | Depends | Slow (train) | No | Complex data |

---

## 9. Evaluation Metrics

### 9.1 Classification Metrics

**Confusion Matrix:**

```
                  Predicted
                  Negative  Positive
Actual Negative |   TN    |   FP   |
Actual Positive |   FN    |   TP   |
```

**Core metrics:**
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]
\[
\text{Precision} = \frac{TP}{TP + FP} \quad \text{(of predicted positives, how many are correct)}
\]
\[
\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} \quad \text{(of actual positives, how many did we catch)}
\]
\[
\text{Specificity} = \frac{TN}{TN + FP} \quad \text{(true negative rate)}
\]
\[
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
\]
\[
F_\beta = \frac{(1+\beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
\]

(\( \beta > 1 \): recall more important; \( \beta < 1 \): precision more important)

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
)

# Fit a model
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Threshold-dependent metrics
print("=== Threshold-dependent (at 0.5) ===")
print(f"Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:   {precision_score(y_test, y_pred):.4f}")
print(f"Recall:      {recall_score(y_test, y_pred):.4f}")
print(f"F1:          {f1_score(y_test, y_pred):.4f}")
print(f"MCC:         {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")

# Threshold-independent metrics
print("\n=== Threshold-independent ===")
print(f"ROC-AUC:     {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC:      {average_precision_score(y_test, y_proba):.4f}")
print(f"Log Loss:    {log_loss(y_test, y_proba):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba):.4f}")

print("\n=== Full Report ===")
print(classification_report(y_test, y_pred))
```

### 9.2 ROC and PR Curves

```python
# ROC Curve
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Optimal threshold by Youden's J statistic
J = tpr - fpr
optimal_idx = np.argmax(J)
optimal_threshold_roc = thresholds_roc[optimal_idx]
print(f"Optimal ROC threshold: {optimal_threshold_roc:.4f}")

# Precision-Recall Curve (better for imbalanced data)
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

# Optimal threshold by F1
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
optimal_idx_pr = np.argmax(f1_scores)
optimal_threshold_pr = thresholds_pr[optimal_idx_pr]
print(f"Optimal PR threshold: {optimal_threshold_pr:.4f}")
```

### 9.3 Regression Metrics

\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
\]
\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
\]
\[
\text{RMSE} = \sqrt{\text{MSE}}
\]
\[
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
\]
\[
\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              mean_absolute_percentage_error,
                              median_absolute_error, explained_variance_score)

# Generate regression predictions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

X_ca, y_ca = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_ca, y_ca, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                                 random_state=42)
gbr.fit(X_tr, y_tr)
y_pred_reg = gbr.predict(X_te)

mae = mean_absolute_error(y_te, y_pred_reg)
mse = mean_squared_error(y_te, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_te, y_pred_reg)
mape = mean_absolute_percentage_error(y_te, y_pred_reg)
med_ae = median_absolute_error(y_te, y_pred_reg)

print(f"MAE:    {mae:.4f}")
print(f"MSE:    {mse:.4f}")
print(f"RMSE:   {rmse:.4f}")
print(f"R²:     {r2:.4f}")
print(f"MAPE:   {mape:.4f} ({mape*100:.2f}%)")
print(f"MedAE:  {med_ae:.4f}")
```

### 9.4 Clustering Metrics

```python
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score,
                              adjusted_rand_score, normalized_mutual_info_score,
                              adjusted_mutual_info_score, fowlkes_mallows_score,
                              homogeneity_completeness_v_measure)

X_cl, y_true_cl = make_blobs(n_samples=500, centers=4, random_state=42)
km = KMeans(n_clusters=4, random_state=42)
labels_pred = km.fit_predict(X_cl)

# Internal metrics (no ground truth needed)
print("=== Internal Metrics ===")
print(f"Silhouette Score:          {silhouette_score(X_cl, labels_pred):.4f}  [-1,1] higher=better")
print(f"Davies-Bouldin Index:      {davies_bouldin_score(X_cl, labels_pred):.4f}  lower=better")
print(f"Calinski-Harabasz Index:   {calinski_harabasz_score(X_cl, labels_pred):.4f}  higher=better")

# External metrics (ground truth available)
print("\n=== External Metrics ===")
print(f"Adjusted Rand Index:       {adjusted_rand_score(y_true_cl, labels_pred):.4f}  [-1,1] higher=better")
print(f"Normalized MI:             {normalized_mutual_info_score(y_true_cl, labels_pred):.4f}  [0,1]")
print(f"Adjusted MI:               {adjusted_mutual_info_score(y_true_cl, labels_pred):.4f}")
print(f"Fowlkes-Mallows:           {fowlkes_mallows_score(y_true_cl, labels_pred):.4f}  [0,1]")
h, c, v = homogeneity_completeness_v_measure(y_true_cl, labels_pred)
print(f"Homogeneity:               {h:.4f}")
print(f"Completeness:              {c:.4f}")
print(f"V-Measure:                 {v:.4f}")
```

---

## 10. Cross-Validation Strategies

### 10.1 K-Fold and Variants

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    RepeatedStratifiedKFold, LeaveOneOut, ShuffleSplit,
    cross_validate, learning_curve
)
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Stratified K-Fold (preserves class proportions — use for classification)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Repeated Stratified (more robust estimate)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Compare
model = RandomForestClassifier(n_estimators=50, random_state=42)

for name, cv in [('KFold', kf), ('StratifiedKFold', skf), ('RepeatedSKF', rskf)]:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 10.2 Time Series Cross-Validation

```python
# Time series: NEVER use random splits!
# Future data must not be used to predict past.
tscv = TimeSeriesSplit(n_splits=5, gap=10)  # gap avoids leakage

# Visual representation:
# Fold 1: Train [1..100]         Test [101..120]
# Fold 2: Train [1..120]         Test [121..140]
# Fold 3: Train [1..140]         Test [141..160]
# Fold 4: Train [1..160]         Test [161..180]
# Fold 5: Train [1..180]         Test [181..200]

import pandas as pd

n = 300
ts_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=n, freq='D'),
    'x': np.random.randn(n),
    'y': np.cumsum(np.random.randn(n))  # Simulated time series
})
ts_data = ts_data.set_index('date')

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(ts_data)):
    print(f"Fold {fold+1}: Train={len(train_idx)} Test={len(test_idx)}")
```

### 10.3 Group K-Fold (Preventing Data Leakage)

```python
# When samples are NOT independent (e.g., multiple readings per patient)
groups = np.repeat(np.arange(100), 10)  # 100 patients, 10 readings each
gkf = GroupKFold(n_splits=5)

# Ensures no patient appears in both train and test
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    print(f"Fold {fold+1}: Train groups={len(train_groups)}, "
          f"Test groups={len(test_groups)}, Overlap={len(overlap)}")
```

### 10.4 Learning Curves

```python
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

print("Learning curve (last 3 points):")
for i in [-3, -2, -1]:
    print(f"  n_train={train_sizes[i]}: train={train_mean[i]:.4f}, val={val_mean[i]:.4f}")

gap = train_mean[-1] - val_mean[-1]
if gap > 0.05:
    print("High gap → overfitting (needs more data or regularization)")
elif val_mean[-1] < 0.8:
    print("Low score → underfitting (needs more complex model)")
else:
    print("Good fit!")
```

---

## 11. Pipelines & Production Patterns

### 11.1 sklearn Pipeline

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# Realistic dataset with mixed types
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.randint(18, 65, n).astype(float),
    'income': np.random.exponential(50000, n),
    'credit_score': np.random.randint(300, 850, n).astype(float),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),
    'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
    'default': np.random.randint(0, 2, n)
})

# Inject missingness
df.loc[np.random.choice(n, 50, replace=False), 'age'] = np.nan
df.loc[np.random.choice(n, 80, replace=False), 'income'] = np.nan

X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing per column type
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['city', 'education']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
], remainder='drop')

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

full_pipeline.fit(X_train, y_train)
print(f"Pipeline accuracy: {full_pipeline.score(X_test, y_test):.4f}")

# Pipeline is serializable and prevents leakage
import joblib
joblib.dump(full_pipeline, '/tmp/credit_pipeline.joblib')
loaded_pipeline = joblib.load('/tmp/credit_pipeline.joblib')
print(f"Loaded pipeline accuracy: {loaded_pipeline.score(X_test, y_test):.4f}")
```

### 11.2 Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """Log-transform positive numeric features."""
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self  # Nothing to learn

    def transform(self, X):
        return np.log(np.array(X) + self.offset)

class FeatureInteractions(BaseEstimator, TransformerMixin):
    """Create pairwise feature interactions."""
    def __init__(self, feature_pairs=None):
        self.feature_pairs = feature_pairs

    def fit(self, X, y=None):
        if self.feature_pairs is None:
            n = X.shape[1]
            self.feature_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        return self

    def transform(self, X):
        X = np.array(X)
        interactions = [X[:, i] * X[:, j] for i, j in self.feature_pairs]
        return np.column_stack([X] + interactions)

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features else [f'x{i}' for i in range(self.n_features_in_)]
        inter = [f'{base[i]}*{base[j]}' for i, j in self.feature_pairs]
        return base + inter

# Use in pipeline
from sklearn.linear_model import LogisticRegression as LR

custom_pipe = Pipeline([
    ('interactions', FeatureInteractions()),
    ('scaler', StandardScaler()),
    ('clf', LR(max_iter=500, C=0.1))
])

X_simple, y_simple = make_classification(n_samples=500, n_features=5, random_state=42)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_simple, y_simple, test_size=0.2)
custom_pipe.fit(X_tr2, y_tr2)
print(f"Custom pipeline accuracy: {custom_pipe.score(X_te2, y_te2):.4f}")
```

---

## 12. Hyperparameter Tuning

### 12.1 Grid Search vs Random Search vs Bayesian

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
import time

X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)

# Grid Search — exhaustive, exponential in parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
}

t0 = time.time()
gs = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)
print(f"GridSearch: best={gs.best_score_:.4f}, time={time.time()-t0:.1f}s, "
      f"fits={len(gs.cv_results_['mean_test_score'])}")

# Random Search — usually better for same budget
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
}

t0 = time.time()
rs = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=5, scoring='accuracy',
                        n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)
print(f"RandomSearch: best={rs.best_score_:.4f}, time={time.time()-t0:.1f}s, fits=50")
```

### 12.2 Bayesian Optimization with Optuna

```python
# pip install optuna
import optuna
from sklearn.ensemble import GradientBoostingClassifier
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy',
                            n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30, show_progress_bar=False)

print(f"Optuna best CV accuracy: {study.best_value:.4f}")
print(f"Optuna best params: {study.best_params}")

# Train with best params
best_model = GradientBoostingClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
print(f"Test accuracy: {best_model.score(X_test, y_test):.4f}")
```

### 12.3 Nested Cross-Validation (Unbiased Estimation)

```python
from sklearn.model_selection import cross_val_score, KFold

# WRONG: using same CV for tuning and evaluation (optimistic bias)
# CORRECT: inner CV for tuning, outer CV for evaluation

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

model = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}

# Nested CV
outer_scores = []
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    inner_gs = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy')
    inner_gs.fit(X_tr, y_tr)
    outer_scores.append(inner_gs.score(X_te, y_te))

print(f"Nested CV accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
print("This is an unbiased estimate of generalization performance.")
```

---

## Summary: Choosing the Right Algorithm

```
Problem Type?
├── Regression
│   ├── Linear relationship → Linear Regression (Ridge/Lasso)
│   ├── Non-linear → Gradient Boosting, Random Forest, SVR
│   └── Very large dataset → LightGBM, SGDRegressor
├── Classification
│   ├── Interpretability needed → Logistic Regression, Decision Tree
│   ├── High accuracy, tabular → XGBoost, LightGBM, CatBoost
│   ├── Text data → Naive Bayes, Logistic Regression + TF-IDF
│   ├── Small dataset → SVM, KNN
│   └── Imbalanced → adjust class_weight, use PR-AUC metric
└── Clustering
    ├── Know K → K-Means (convex), GMM (soft)
    ├── Arbitrary shapes → DBSCAN
    ├── Hierarchy needed → Agglomerative
    └── Large scale → MiniBatchKMeans

Rule of thumb: Always start simple (linear models), then add complexity.
```

| Algorithm | Complexity | Interpretable | Handles Non-linear | Handles Missing | Feature Scale |
|-----------|-----------|---------------|-------------------|-----------------|---------------|
| Linear Reg | Low | Yes | No | No | Required |
| Logistic Reg | Low | Yes | No | No | Required |
| Decision Tree | Medium | Yes | Yes | No | Not required |
| Random Forest | High | Partially | Yes | No | Not required |
| XGBoost | High | Partially | Yes | Yes (native) | Not required |
| SVM | Medium | No | Yes (kernel) | No | Required |
| KNN | Low | No | Yes | No | Required |
| Naive Bayes | Low | Partially | No | No | Not required |
