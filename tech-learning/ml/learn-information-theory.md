# Information Theory for Machine Learning and Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Information and Entropy](#information-and-entropy)
3. [Joint and Conditional Entropy](#joint-and-conditional-entropy)
4. [Mutual Information](#mutual-information)
5. [Kullback-Leibler Divergence](#kullback-leibler-divergence)
6. [Cross-Entropy](#cross-entropy)
7. [Maximum Entropy Principle](#maximum-entropy-principle)
8. [Applications in ML/DL](#applications-in-mldl)
9. [Practical Examples](#practical-examples)

---

## Introduction

Information theory provides a mathematical framework for quantifying information, uncertainty, and communication. It's fundamental to machine learning and deep learning, especially for:
- **Loss Functions**: Cross-entropy loss for classification
- **Regularization**: KL divergence for Bayesian methods
- **Feature Selection**: Mutual information for feature importance
- **Model Compression**: Information-theoretic bounds
- **Uncertainty Quantification**: Entropy-based measures

### Key Concepts
- **Entropy**: Measure of uncertainty/information
- **Mutual Information**: Dependence between variables
- **KL Divergence**: Difference between distributions
- **Cross-Entropy**: Expected code length

---

## Information and Entropy

### Information Content
The information content of an event with probability p is:
I(x) = -log₂(p)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def information_content(p):
    """Information content: I(x) = -log₂(p)"""
    return -np.log2(p + 1e-10)  # Add small epsilon to avoid log(0)

# Example: Information content for different probabilities
p_values = np.linspace(0.01, 1, 100)
info = information_content(p_values)

plt.plot(p_values, info)
plt.xlabel('Probability p')
plt.ylabel('Information Content I(x)')
plt.title('Information Content: I(x) = -log₂(p)')
plt.grid(True)
plt.show()

# Rare events have high information content
print(f"Information of rare event (p=0.01): {information_content(0.01):.4f} bits")
print(f"Information of common event (p=0.9): {information_content(0.9):.4f} bits")
```

### Shannon Entropy
Entropy measures the average information content (uncertainty) of a random variable:

H(X) = -Σ p(x) * log₂(p(x))

```python
def shannon_entropy(probabilities):
    """Compute Shannon entropy: H(X) = -Σ p(x) * log₂(p(x))"""
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]  # Remove zeros
    return -np.sum(probabilities * np.log2(probabilities))

# Example: Fair coin
p_fair = [0.5, 0.5]
entropy_fair = shannon_entropy(p_fair)
print(f"Entropy of fair coin: {entropy_fair:.4f} bits")

# Biased coin
p_biased = [0.9, 0.1]
entropy_biased = shannon_entropy(p_biased)
print(f"Entropy of biased coin: {entropy_biased:.4f} bits")

# Maximum entropy occurs when distribution is uniform
p_uniform = [1/6] * 6  # Fair die
entropy_uniform = shannon_entropy(p_uniform)
print(f"Entropy of fair die: {entropy_uniform:.4f} bits")
print(f"Maximum entropy (log₂(6)): {np.log2(6):.4f} bits")

# Visualize entropy for binary distribution
p_heads = np.linspace(0.01, 0.99, 100)
entropies = [shannon_entropy([p, 1-p]) for p in p_heads]

plt.plot(p_heads, entropies)
plt.axvline(x=0.5, color='r', linestyle='--', label='Maximum entropy')
plt.xlabel('P(Heads)')
plt.ylabel('Entropy H(X)')
plt.title('Entropy of Binary Distribution')
plt.legend()
plt.grid(True)
plt.show()
```

### Properties of Entropy
```python
# 1. Non-negative: H(X) ≥ 0
# 2. Maximum when uniform: H(X) ≤ log₂(n)
# 3. Zero when deterministic: H(X) = 0 if p(x) = 1 for some x

# Demonstrate properties
n = 10
uniform_dist = np.ones(n) / n
deterministic_dist = np.zeros(n)
deterministic_dist[0] = 1.0

entropy_uniform = shannon_entropy(uniform_dist)
entropy_deterministic = shannon_entropy(deterministic_dist)
max_entropy = np.log2(n)

print(f"Uniform distribution entropy: {entropy_uniform:.4f}")
print(f"Deterministic distribution entropy: {entropy_deterministic:.4f}")
print(f"Maximum possible entropy: {max_entropy:.4f}")
print(f"Uniform achieves maximum: {np.isclose(entropy_uniform, max_entropy)}")
```

---

## Joint and Conditional Entropy

### Joint Entropy
H(X, Y) = -Σ Σ p(x, y) * log₂(p(x, y))

```python
def joint_entropy(joint_prob):
    """Compute joint entropy H(X, Y)"""
    joint_prob = np.array(joint_prob)
    joint_prob = joint_prob[joint_prob > 0]
    return -np.sum(joint_prob * np.log2(joint_prob))

# Example: Two dependent variables
# X: First die, Y: Second die
joint_prob_dice = np.ones((6, 6)) / 36  # Independent, uniform
H_XY = joint_entropy(joint_prob_dice.flatten())
print(f"Joint entropy H(X, Y): {H_XY:.4f} bits")
print(f"Sum of individual entropies: {2 * np.log2(6):.4f} bits")
print(f"Equal (independent): {np.isclose(H_XY, 2 * np.log2(6))}")
```

### Conditional Entropy
H(Y|X) = -Σ Σ p(x, y) * log₂(p(y|x))

```python
def conditional_entropy(joint_prob):
    """Compute conditional entropy H(Y|X)"""
    joint_prob = np.array(joint_prob)
    marginal_X = np.sum(joint_prob, axis=1, keepdims=True)
    cond_prob = joint_prob / (marginal_X + 1e-10)
    
    # H(Y|X) = -Σ Σ p(x, y) * log₂(p(y|x))
    cond_prob_flat = cond_prob[cond_prob > 0]
    joint_prob_flat = joint_prob[joint_prob > 0]
    
    # Reconstruct conditional probabilities
    H_Y_given_X = 0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                p_y_given_x = cond_prob[i, j]
                H_Y_given_X -= joint_prob[i, j] * np.log2(p_y_given_x + 1e-10)
    
    return H_Y_given_X

# Example: Y = X (perfectly dependent)
joint_prob_dependent = np.eye(6) / 6  # Y always equals X
H_Y_given_X_dependent = conditional_entropy(joint_prob_dependent)
print(f"Conditional entropy H(Y|X) when Y=X: {H_Y_given_X_dependent:.4f} bits")
print(f"Should be 0 (no uncertainty): {H_Y_given_X_dependent < 0.01}")

# Chain rule: H(X, Y) = H(X) + H(Y|X)
H_X = np.log2(6)
H_Y_given_X = conditional_entropy(joint_prob_dice)
H_XY_chain = H_X + H_Y_given_X
print(f"\nChain rule verification:")
print(f"H(X) + H(Y|X) = {H_XY_chain:.4f}")
print(f"H(X, Y) = {H_XY:.4f}")
print(f"Match: {np.isclose(H_XY_chain, H_XY)}")
```

---

## Mutual Information

### Definition
Mutual information measures the amount of information one variable contains about another:

I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)

```python
def mutual_information(joint_prob):
    """Compute mutual information I(X; Y)"""
    joint_prob = np.array(joint_prob)
    
    # Marginal distributions
    marginal_X = np.sum(joint_prob, axis=1)
    marginal_Y = np.sum(joint_prob, axis=0)
    
    # Individual entropies
    H_X = shannon_entropy(marginal_X)
    H_Y = shannon_entropy(marginal_Y)
    
    # Joint entropy
    H_XY = joint_entropy(joint_prob.flatten())
    
    # Mutual information
    I_XY = H_X + H_Y - H_XY
    
    return I_XY

# Example 1: Independent variables
joint_independent = np.ones((6, 6)) / 36
I_independent = mutual_information(joint_independent)
print(f"Mutual information (independent): {I_independent:.6f} bits")
print(f"Should be 0: {I_independent < 0.01}")

# Example 2: Perfectly dependent (Y = X)
joint_dependent = np.eye(6) / 6
I_dependent = mutual_information(joint_dependent)
H_X = np.log2(6)
print(f"\nMutual information (Y=X): {I_dependent:.4f} bits")
print(f"Should equal H(X): {H_X:.4f} bits")
print(f"Match: {np.isclose(I_dependent, H_X)}")

# Example 3: Partially dependent
joint_partial = np.array([
    [0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.05, 0.05],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])
joint_partial = joint_partial / np.sum(joint_partial)  # Normalize
I_partial = mutual_information(joint_partial)
print(f"\nMutual information (partially dependent): {I_partial:.4f} bits")
```

### Properties of Mutual Information
```python
# 1. Non-negative: I(X; Y) ≥ 0
# 2. Symmetric: I(X; Y) = I(Y; X)
# 3. Zero if independent: I(X; Y) = 0 iff X and Y independent
# 4. Upper bound: I(X; Y) ≤ min(H(X), H(Y))

# Demonstrate properties
H_X = shannon_entropy(np.sum(joint_partial, axis=1))
H_Y = shannon_entropy(np.sum(joint_partial, axis=0))
I_XY = mutual_information(joint_partial)

print(f"I(X; Y) = {I_XY:.4f}")
print(f"min(H(X), H(Y)) = {min(H_X, H_Y):.4f}")
print(f"I(X; Y) ≤ min(H(X), H(Y)): {I_XY <= min(H_X, H_Y)}")
```

---

## Kullback-Leibler Divergence

### Definition
KL divergence measures how different one probability distribution is from another:

D_KL(P || Q) = Σ P(x) * log₂(P(x) / Q(x))

```python
def kl_divergence(P, Q):
    """Compute KL divergence D_KL(P || Q)"""
    P = np.array(P)
    Q = np.array(Q)
    
    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Avoid division by zero
    mask = P > 0
    P = P[mask]
    Q = Q[mask]
    
    return np.sum(P * np.log2(P / (Q + 1e-10)))

# Example: Two different distributions
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.33, 0.33, 0.34])

D_KL_PQ = kl_divergence(P, Q)
D_KL_QP = kl_divergence(Q, P)

print(f"D_KL(P || Q) = {D_KL_PQ:.4f} bits")
print(f"D_KL(Q || P) = {D_KL_QP:.4f} bits")
print(f"KL divergence is NOT symmetric: {not np.isclose(D_KL_PQ, D_KL_QP)}")

# KL divergence is zero when distributions are identical
D_KL_same = kl_divergence(P, P)
print(f"\nD_KL(P || P) = {D_KL_same:.6f} bits (should be 0)")
```

### Properties
```python
# 1. Non-negative: D_KL(P || Q) ≥ 0
# 2. Zero if identical: D_KL(P || Q) = 0 iff P = Q
# 3. Not symmetric: D_KL(P || Q) ≠ D_KL(Q || P) in general
# 4. Not a metric (doesn't satisfy triangle inequality)

# Visualize KL divergence
P_fixed = np.array([0.5, 0.3, 0.2])
Q_range = np.linspace(0.01, 0.99, 100)
kl_values = []

for q1 in Q_range:
    q2 = (1 - q1) / 2
    q3 = (1 - q1) / 2
    Q = np.array([q1, q2, q3])
    kl_values.append(kl_divergence(P_fixed, Q))

plt.plot(Q_range, kl_values)
plt.xlabel('Q[0]')
plt.ylabel('D_KL(P || Q)')
plt.title('KL Divergence as Q Varies')
plt.grid(True)
plt.show()
```

### Applications in ML
```python
# KL divergence is used in:
# 1. Variational Autoencoders (VAE)
# 2. Bayesian neural networks
# 3. Regularization (KL penalty)

def vae_loss(reconstruction_loss, mu, logvar):
    """VAE loss includes KL divergence term"""
    # KL divergence between q(z|x) and p(z) = N(0, I)
    kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    return reconstruction_loss + kl_loss

# Example
recon_loss = 10.0
mu = np.array([0.1, -0.2, 0.3])
logvar = np.array([-1.0, -0.5, -1.5])

total_loss = vae_loss(recon_loss, mu, logvar)
print(f"Reconstruction loss: {recon_loss:.4f}")
print(f"KL loss: {total_loss - recon_loss:.4f}")
print(f"Total VAE loss: {total_loss:.4f}")
```

---

## Cross-Entropy

### Definition
Cross-entropy measures the average code length when using distribution Q to encode events from distribution P:

H(P, Q) = -Σ P(x) * log₂(Q(x))

```python
def cross_entropy(P, Q):
    """Compute cross-entropy H(P, Q)"""
    P = np.array(P)
    Q = np.array(Q)
    
    # Normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Avoid log(0)
    mask = P > 0
    P = P[mask]
    Q = Q[mask]
    
    return -np.sum(P * np.log2(Q + 1e-10))

# Relationship: H(P, Q) = H(P) + D_KL(P || Q)
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.33, 0.33, 0.34])

H_P = shannon_entropy(P)
D_KL_PQ = kl_divergence(P, Q)
H_PQ = cross_entropy(P, Q)

print(f"H(P) = {H_P:.4f} bits")
print(f"D_KL(P || Q) = {D_KL_PQ:.4f} bits")
print(f"H(P, Q) = {H_PQ:.4f} bits")
print(f"H(P) + D_KL(P || Q) = {H_P + D_KL_PQ:.4f} bits")
print(f"Match: {np.isclose(H_PQ, H_P + D_KL_PQ)}")
```

### Cross-Entropy Loss in Classification
```python
# Cross-entropy loss is the standard loss for classification
def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss for classification"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# Example: Multi-class classification
# True distribution: [0, 0, 1, 0] (class 2)
# Predicted distribution: [0.1, 0.2, 0.6, 0.1]

y_true = np.array([0, 0, 1, 0])
y_pred_good = np.array([0.1, 0.1, 0.7, 0.1])
y_pred_bad = np.array([0.7, 0.1, 0.1, 0.1])

loss_good = cross_entropy_loss(y_pred_good, y_true)
loss_bad = cross_entropy_loss(y_pred_bad, y_true)

print(f"Loss (good prediction): {loss_good:.4f}")
print(f"Loss (bad prediction): {loss_bad:.4f}")
print(f"Lower loss = better prediction: {loss_good < loss_bad}")
```

---

## Maximum Entropy Principle

The maximum entropy principle states that, given constraints, the probability distribution with maximum entropy is the most unbiased.

```python
from scipy.optimize import minimize

def max_entropy_distribution(constraints, n_states=10):
    """Find maximum entropy distribution given constraints"""
    # Constraints: E[f_i(X)] = c_i
    
    def objective(p):
        """Negative entropy (to minimize)"""
        p = p[p > 0]
        return np.sum(p * np.log2(p + 1e-10))
    
    def constraint_sum(p):
        """Probabilities must sum to 1"""
        return np.sum(p) - 1
    
    # Initial guess: uniform distribution
    p0 = np.ones(n_states) / n_states
    
    # Constraints
    constraints_list = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Add custom constraints
    for constraint_func, value in constraints:
        constraints_list.append({
            'type': 'eq',
            'fun': lambda p, f=constraint_func, v=value: np.sum(p * f(np.arange(n_states))) - v
        })
    
    # Bounds: probabilities must be non-negative
    bounds = [(0, 1)] * n_states
    
    result = minimize(objective, p0, method='SLSQP',
                     bounds=bounds, constraints=constraints_list)
    
    return result.x

# Example: Maximum entropy with mean constraint
# Constraint: E[X] = 5
n_states = 10
constraints = [
    (lambda x: x, 5.0)  # Mean constraint
]

p_max_ent = max_entropy_distribution(constraints, n_states)

plt.bar(range(n_states), p_max_ent)
plt.xlabel('State')
plt.ylabel('Probability')
plt.title('Maximum Entropy Distribution (E[X] = 5)')
plt.show()

entropy_max = shannon_entropy(p_max_ent)
entropy_uniform = np.log2(n_states)
print(f"Maximum entropy (with constraint): {entropy_max:.4f}")
print(f"Uniform entropy (no constraint): {entropy_uniform:.4f}")
```

---

## Applications in ML/DL

### 1. Feature Selection with Mutual Information
```python
def mutual_information_feature_selection(X, y, n_features=5):
    """Select features using mutual information"""
    n_samples, n_total_features = X.shape
    selected_features = []
    remaining_features = list(range(n_total_features))
    
    for _ in range(n_features):
        best_feature = None
        best_mi = -1
        
        for feature_idx in remaining_features:
            # Compute mutual information between feature and target
            feature_values = X[:, feature_idx]
            
            # Discretize for MI computation (simplified)
            feature_bins = np.digitize(feature_values, bins=10)
            target_bins = np.digitize(y, bins=10) if y.dtype == float else y
            
            # Compute joint probability
            joint_prob = np.zeros((10, 10))
            for i, j in zip(feature_bins, target_bins):
                if i < 10 and j < 10:
                    joint_prob[i, j] += 1
            joint_prob /= np.sum(joint_prob)
            
            mi = mutual_information(joint_prob)
            
            if mi > best_mi:
                best_mi = mi
                best_feature = feature_idx
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
    
    return selected_features

# Example usage
np.random.seed(42)
X = np.random.randn(1000, 20)
y = X[:, 0] + X[:, 1] + 0.1 * np.random.randn(1000)  # y depends on first 2 features

selected = mutual_information_feature_selection(X, y, n_features=5)
print(f"Selected features: {selected}")
print(f"First 2 features should be selected: {0 in selected and 1 in selected}")
```

### 2. Model Compression
```python
def information_bottleneck(X, Y, compression_ratio=0.5):
    """Information bottleneck: compress X while preserving information about Y"""
    # Simplified version: use PCA-like compression
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=int(X.shape[1] * compression_ratio))
    X_compressed = pca.fit_transform(X)
    
    # Measure information preserved
    # I(X_compressed; Y) / I(X; Y)
    
    return X_compressed, pca
```

### 3. Uncertainty Quantification
```python
def prediction_entropy(y_pred_probs):
    """Compute entropy of predictions as uncertainty measure"""
    # y_pred_probs: (n_samples, n_classes)
    entropies = []
    for probs in y_pred_probs:
        ent = shannon_entropy(probs)
        entropies.append(ent)
    return np.array(entropies)

# Example: High entropy = uncertain, Low entropy = confident
y_pred_confident = np.array([[0.1, 0.1, 0.8]])  # Confident prediction
y_pred_uncertain = np.array([[0.33, 0.33, 0.34]])  # Uncertain prediction

ent_confident = prediction_entropy(y_pred_confident)
ent_uncertain = prediction_entropy(y_pred_uncertain)

print(f"Entropy (confident): {ent_confident[0]:.4f}")
print(f"Entropy (uncertain): {ent_uncertain[0]:.4f}")
print(f"Lower entropy = more confident: {ent_confident[0] < ent_uncertain[0]}")
```

---

## Practical Examples

### Example 1: Information Gain for Decision Trees
```python
def information_gain(y_parent, y_left, y_right):
    """Compute information gain for decision tree split"""
    # Parent entropy
    parent_probs = np.bincount(y_parent) / len(y_parent)
    H_parent = shannon_entropy(parent_probs)
    
    # Weighted child entropy
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    left_probs = np.bincount(y_left) / n_left if n_left > 0 else [1]
    right_probs = np.bincount(y_right) / n_right if n_right > 0 else [1]
    
    H_left = shannon_entropy(left_probs)
    H_right = shannon_entropy(right_probs)
    
    H_children = (n_left / n_total) * H_left + (n_right / n_total) * H_right
    
    # Information gain
    IG = H_parent - H_children
    return IG

# Example
y_parent = np.array([0, 0, 0, 1, 1, 1])
y_left = np.array([0, 0, 0])  # Pure left
y_right = np.array([1, 1, 1])  # Pure right

ig = information_gain(y_parent, y_left, y_right)
print(f"Information gain: {ig:.4f} bits")
print(f"Perfect split (IG = H(parent)): {np.isclose(ig, shannon_entropy([0.5, 0.5]))}")
```

### Example 2: Variational Inference with KL Divergence
```python
def variational_inference_loss(data, q_params, p_params):
    """Variational inference loss: reconstruction + KL divergence"""
    # q_params: parameters of approximate posterior q(z|x)
    # p_params: parameters of prior p(z)
    
    mu_q, logvar_q = q_params
    mu_p, logvar_p = p_params
    
    # KL divergence: D_KL(q(z|x) || p(z))
    kl_loss = 0.5 * np.sum(
        logvar_p - logvar_q - 1 + 
        (np.exp(logvar_q) + (mu_q - mu_p)**2) / np.exp(logvar_p)
    )
    
    # Reconstruction loss (simplified)
    recon_loss = np.mean((data - mu_q)**2)
    
    return recon_loss + kl_loss

# Example
data = np.random.randn(100)
q_params = (np.random.randn(100), np.ones(100) * -1)
p_params = (np.zeros(100), np.ones(100) * -2)

loss = variational_inference_loss(data, q_params, p_params)
print(f"Variational loss: {loss:.4f}")
```

---

## Key Takeaways

1. **Entropy**: Measures uncertainty/information content
2. **Mutual Information**: Quantifies dependence between variables
3. **KL Divergence**: Measures difference between distributions
4. **Cross-Entropy**: Standard loss for classification
5. **Maximum Entropy**: Principle for unbiased distributions
6. **Applications**: Feature selection, model compression, uncertainty quantification

---

## Additional Resources

- **Elements of Information Theory (Cover & Thomas)**: Comprehensive textbook
- **Information Theory, Inference, and Learning Algorithms (MacKay)**: ML perspective
- **Deep Learning Book (Goodfellow)**: Information theory in DL context

---

**Master information theory to understand uncertainty and information in ML/DL!**

