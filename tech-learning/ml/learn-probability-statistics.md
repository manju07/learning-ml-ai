# Probability and Statistics for Machine Learning and Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Probability Fundamentals](#probability-fundamentals)
3. [Random Variables](#random-variables)
4. [Probability Distributions](#distributions)
5. [Joint and Conditional Probability](#joint-conditional)
6. [Bayes' Theorem](#bayes)
7. [Expectation and Variance](#expectation-variance)
8. [Covariance and Correlation](#covariance)
9. [Central Limit Theorem](#clt)
10. [Hypothesis Testing](#hypothesis-testing)
11. [Maximum Likelihood Estimation](#mle)
12. [Bayesian Inference](#bayesian-inference)
13. [Applications in ML/DL](#applications)
14. [Practical Examples](#examples)

---

## Introduction {#introduction}

Probability and statistics form the theoretical foundation for machine learning and deep learning. They provide:
- **Uncertainty Quantification**: Understanding model confidence
- **Statistical Learning Theory**: Theoretical guarantees
- **Bayesian Methods**: Incorporating prior knowledge
- **Hypothesis Testing**: Model validation and comparison
- **Estimation Theory**: Parameter estimation from data

### Why Probability & Statistics Matter in ML/DL
- **Loss Functions**: Often based on probability (cross-entropy, log-likelihood)
- **Regularization**: Bayesian interpretation (L2 = Gaussian prior)
- **Uncertainty**: Quantifying prediction confidence
- **Model Selection**: Using statistical tests
- **Data Analysis**: Understanding data distributions

---

## Probability Fundamentals {#probability-fundamentals}

### Sample Space and Events
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, bernoulli, binomial, poisson, exponential

# Sample space: All possible outcomes
# Event: Subset of sample space

# Example: Rolling a die
sample_space = {1, 2, 3, 4, 5, 6}
event_even = {2, 4, 6}
event_odd = {1, 3, 5}

print(f"Sample space: {sample_space}")
print(f"Event (even): {event_even}")
print(f"Probability of even: {len(event_even) / len(sample_space)}")
```

### Axioms of Probability
1. P(A) ≥ 0 for any event A
2. P(S) = 1 (sample space)
3. P(A ∪ B) = P(A) + P(B) if A and B are disjoint

```python
def probability_union(A, B):
    """P(A ∪ B) = P(A) + P(B) - P(A ∩ B)"""
    intersection = len(A & B)
    return (len(A) + len(B) - intersection) / len(sample_space)

# Example
A = {1, 2, 3}
B = {3, 4, 5}
print(f"P(A ∪ B) = {probability_union(A, B)}")
```

---

## Random Variables {#random-variables}

### Discrete Random Variables
```python
# Discrete: Takes countable values
# Example: Number of heads in 10 coin flips

# Bernoulli: Single trial (success/failure)
p = 0.5
bernoulli_rv = bernoulli(p)
print(f"P(X=1) = {bernoulli_rv.pmf(1):.4f}")
print(f"P(X=0) = {bernoulli_rv.pmf(0):.4f}")

# Visualize
x = [0, 1]
pmf = [bernoulli_rv.pmf(0), bernoulli_rv.pmf(1)]
plt.bar(x, pmf)
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.title('Bernoulli Distribution (p=0.5)')
plt.xticks(x)
plt.show()
```

### Continuous Random Variables
```python
# Continuous: Takes uncountable values
# Example: Height of a person

# Normal distribution
mu, sigma = 0, 1
normal_rv = norm(mu, sigma)

x = np.linspace(-4, 4, 100)
pdf = normal_rv.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Normal Distribution (μ=0, σ=1)')
plt.grid(True)
plt.show()
```

---

## Probability Distributions {#distributions}

### Discrete Distributions

#### Bernoulli Distribution
```python
# Single trial: P(X=1) = p, P(X=0) = 1-p
p = 0.7
bernoulli_dist = bernoulli(p)

x = np.array([0, 1])
pmf = bernoulli_dist.pmf(x)

plt.bar(x, pmf)
plt.xlabel('x')
plt.ylabel('P(X=x)')
plt.title(f'Bernoulli Distribution (p={p})')
plt.xticks(x)
plt.show()

# Mean and variance
print(f"Mean: {bernoulli_dist.mean():.4f}")
print(f"Variance: {bernoulli_dist.var():.4f}")
```

#### Binomial Distribution
```python
# n independent Bernoulli trials
n, p = 10, 0.5
binomial_dist = binomial(n, p)

x = np.arange(0, n+1)
pmf = binomial_dist.pmf(x)

plt.bar(x, pmf)
plt.xlabel('Number of successes')
plt.ylabel('P(X=k)')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.show()

# Mean: np, Variance: np(1-p)
print(f"Mean: {binomial_dist.mean():.4f}")
print(f"Variance: {binomial_dist.var():.4f}")
```

#### Poisson Distribution
```python
# Number of events in fixed interval
lambda_param = 3
poisson_dist = poisson(lambda_param)

x = np.arange(0, 15)
pmf = poisson_dist.pmf(x)

plt.bar(x, pmf)
plt.xlabel('Number of events')
plt.ylabel('P(X=k)')
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.show()

# Mean = Variance = λ
print(f"Mean: {poisson_dist.mean():.4f}")
print(f"Variance: {poisson_dist.var():.4f}")
```

### Continuous Distributions

#### Normal (Gaussian) Distribution
```python
# Most important distribution in statistics
mu, sigma = 0, 1
normal_dist = norm(mu, sigma)

x = np.linspace(-4, 4, 100)
pdf = normal_dist.pdf(x)
cdf = normal_dist.cdf(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(x, pdf)
ax1.set_title('PDF: Normal Distribution')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.grid(True)

ax2.plot(x, cdf)
ax2.set_title('CDF: Normal Distribution')
ax2.set_xlabel('x')
ax2.set_ylabel('F(x)')
ax2.grid(True)
plt.show()

# Properties
print(f"Mean: {normal_dist.mean():.4f}")
print(f"Variance: {normal_dist.var():.4f}")
print(f"68% within 1σ: {normal_dist.cdf(1) - normal_dist.cdf(-1):.4f}")
print(f"95% within 2σ: {normal_dist.cdf(2) - normal_dist.cdf(-2):.4f}")
```

#### Uniform Distribution
```python
from scipy.stats import uniform

a, b = 0, 1
uniform_dist = uniform(a, b - a)

x = np.linspace(-0.5, 1.5, 100)
pdf = uniform_dist.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Uniform Distribution [a={a}, b={b}]')
plt.grid(True)
plt.show()
```

#### Exponential Distribution
```python
# Time between events in Poisson process
lambda_param = 2
exponential_dist = exponential(scale=1/lambda_param)

x = np.linspace(0, 5, 100)
pdf = exponential_dist.pdf(x)

plt.plot(x, pdf)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Exponential Distribution (λ={lambda_param})')
plt.grid(True)
plt.show()
```

#### Multivariate Normal Distribution
```python
from scipy.stats import multivariate_normal

# 2D Gaussian
mean = [0, 0]
cov = [[1, 0.5],
       [0.5, 1]]

mv_normal = multivariate_normal(mean, cov)

# Generate samples
samples = mv_normal.rvs(1000)

# Visualize
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Samples from 2D Gaussian')
plt.grid(True)
plt.show()

# Contour plot of PDF
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = mv_normal.pdf(pos)

plt.contourf(X, Y, Z, levels=20)
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('PDF: 2D Gaussian')
plt.show()
```

---

## Joint and Conditional Probability {#joint-conditional}

### Joint Probability
```python
# P(A and B) = P(A ∩ B)

# Example: Two dice
# P(sum = 7 and first die = 3)
# = P(first die = 3) * P(second die = 4 | first die = 3)
# = (1/6) * (1/6) = 1/36

# Create joint probability table
dice1 = np.arange(1, 7)
dice2 = np.arange(1, 7)
joint_prob = np.zeros((6, 6))

for i in dice1:
    for j in dice2:
        joint_prob[i-1, j-1] = 1/36

plt.imshow(joint_prob, cmap='viridis')
plt.colorbar()
plt.xlabel('Dice 2')
plt.ylabel('Dice 1')
plt.title('Joint Probability: Two Dice')
plt.show()
```

### Conditional Probability
```python
# P(A|B) = P(A ∩ B) / P(B)

# Example: P(sum = 7 | first die = 3)
# = P(sum = 7 and first die = 3) / P(first die = 3)
# = (1/36) / (1/6) = 1/6

def conditional_probability(A_given_B, B):
    """P(A|B) = P(A ∩ B) / P(B)"""
    return A_given_B / B

# Example
P_sum7_and_first3 = 1/36
P_first3 = 1/6
P_sum7_given_first3 = conditional_probability(P_sum7_and_first3, P_first3)
print(f"P(sum=7 | first=3) = {P_sum7_given_first3:.4f}")
```

### Independence
```python
# Events A and B are independent if P(A|B) = P(A)
# Equivalently: P(A ∩ B) = P(A) * P(B)

# Example: Two fair coin flips
P_A = 0.5  # First coin is heads
P_B = 0.5  # Second coin is heads
P_A_and_B = 0.25  # Both heads

is_independent = np.isclose(P_A_and_B, P_A * P_B)
print(f"Independent: {is_independent}")
```

---

## Bayes' Theorem {#bayes}

### Formula
P(A|B) = P(B|A) * P(A) / P(B)

```python
def bayes_theorem(P_B_given_A, P_A, P_B):
    """Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)"""
    return P_B_given_A * P_A / P_B

# Medical test example
# P(disease) = 0.01 (1% prevalence)
# P(positive | disease) = 0.99 (99% sensitivity)
# P(positive | no disease) = 0.05 (5% false positive rate)

P_disease = 0.01
P_positive_given_disease = 0.99
P_positive_given_no_disease = 0.05

# P(positive) = P(positive|disease)*P(disease) + P(positive|no disease)*P(no disease)
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_no_disease * (1 - P_disease))

# P(disease | positive) using Bayes' theorem
P_disease_given_positive = bayes_theorem(
    P_positive_given_disease, P_disease, P_positive
)

print(f"P(disease | positive) = {P_disease_given_positive:.4f}")
print(f"Even with 99% accurate test, only {P_disease_given_positive*100:.2f}% chance of disease!")
```

### Bayesian Update
```python
def bayesian_update(prior, likelihood, evidence):
    """Update prior belief with new evidence"""
    posterior = (likelihood * prior) / evidence
    return posterior

# Coin flip example: Is coin fair?
# Prior: P(fair) = 0.5
# Evidence: 3 heads in 3 flips
# Likelihood: P(3 heads | fair) = 0.5^3 = 0.125

prior_fair = 0.5
likelihood_3heads_given_fair = 0.5**3
likelihood_3heads_given_unfair = 1.0  # Assume unfair always heads

# Total probability of 3 heads
P_3heads = (likelihood_3heads_given_fair * prior_fair + 
            likelihood_3heads_given_unfair * (1 - prior_fair))

# Posterior
posterior_fair = bayesian_update(prior_fair, likelihood_3heads_given_fair, P_3heads)
print(f"Prior P(fair): {prior_fair:.4f}")
print(f"Posterior P(fair | 3 heads): {posterior_fair:.4f}")
```

---

## Expectation and Variance {#expectation-variance}

### Expectation (Mean)
```python
# E[X] = Σ x * P(X=x) for discrete
# E[X] = ∫ x * f(x) dx for continuous

# Discrete example
x = np.array([1, 2, 3, 4, 5, 6])
p = np.array([1/6] * 6)  # Fair die
expectation = np.sum(x * p)
print(f"E[X] = {expectation:.4f}")

# Continuous example: E[X] for X ~ N(μ, σ²) = μ
mu, sigma = 5, 2
normal_dist = norm(mu, sigma)
expectation_continuous = normal_dist.mean()
print(f"E[X] for N({mu}, {sigma}²) = {expectation_continuous:.4f}")

# Properties
# E[aX + b] = aE[X] + b
# E[X + Y] = E[X] + E[Y]
```

### Variance
```python
# Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²

# Discrete example
variance = np.sum((x - expectation)**2 * p)
print(f"Var(X) = {variance:.4f}")

# Or: Var(X) = E[X²] - (E[X])²
E_X_squared = np.sum(x**2 * p)
variance_alt = E_X_squared - expectation**2
print(f"Var(X) (alternative) = {variance_alt:.4f}")

# Continuous example
variance_continuous = normal_dist.var()
print(f"Var(X) for N({mu}, {sigma}²) = {variance_continuous:.4f}")

# Properties
# Var(aX + b) = a²Var(X)
# Var(X + Y) = Var(X) + Var(Y) if X, Y independent
```

### Standard Deviation
```python
std_dev = np.sqrt(variance)
print(f"Std Dev = {std_dev:.4f}")
```

---

## Covariance and Correlation {#covariance}

### Covariance
```python
# Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]

# Generate correlated data
np.random.seed(42)
n = 1000
X = np.random.randn(n)
Y = 0.7 * X + 0.3 * np.random.randn(n)  # Correlated with X

# Compute covariance
cov_XY = np.cov(X, Y)[0, 1]
print(f"Cov(X, Y) = {cov_XY:.4f}")

# Visualize
plt.scatter(X, Y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Correlated Data (Cov = {cov_XY:.4f})')
plt.grid(True)
plt.show()
```

### Correlation
```python
# ρ(X, Y) = Cov(X, Y) / (σ_X * σ_Y)
# Range: [-1, 1]

correlation = np.corrcoef(X, Y)[0, 1]
print(f"Correlation = {correlation:.4f}")

# Or compute manually
std_X = np.std(X)
std_Y = np.std(Y)
correlation_manual = cov_XY / (std_X * std_Y)
print(f"Correlation (manual) = {correlation_manual:.4f}")
```

### Correlation Matrix
```python
# For multiple variables
data = np.column_stack([X, Y, np.random.randn(n)])
corr_matrix = np.corrcoef(data.T)

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()
```

---

## Central Limit Theorem {#clt}

The CLT states that the sum of independent random variables approaches a normal distribution.

```python
# Demonstrate CLT
n_samples = 10000
n_trials = 1000

# Sample from uniform distribution
uniform_samples = np.random.uniform(0, 1, (n_trials, n_samples))

# Compute means
sample_means = np.mean(uniform_samples, axis=1)

# Plot distribution of sample means
plt.hist(sample_means, bins=50, density=True, alpha=0.7, label='Sample Means')

# Overlay normal distribution
mu_clt = 0.5  # Mean of uniform[0,1]
sigma_clt = np.sqrt(1/12 / n_samples)  # Std of mean
x = np.linspace(0.4, 0.6, 100)
plt.plot(x, norm(mu_clt, sigma_clt).pdf(x), 'r-', linewidth=2, label='Normal Approximation')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Central Limit Theorem Demonstration')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Hypothesis Testing {#hypothesis-testing}

### T-Test
```python
from scipy.stats import ttest_1samp, ttest_ind

# One-sample t-test: Is mean significantly different from μ₀?
data = np.random.normal(5.2, 1.0, 100)  # True mean = 5.2
mu0 = 5.0

t_stat, p_value = ttest_1samp(data, mu0)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at α=0.05: {p_value < 0.05}")

# Two-sample t-test: Are means of two groups different?
group1 = np.random.normal(5.0, 1.0, 100)
group2 = np.random.normal(5.5, 1.0, 100)

t_stat, p_value = ttest_ind(group1, group2)
print(f"\nTwo-sample t-test:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

### Chi-Square Test
```python
from scipy.stats import chi2_contingency

# Test independence in contingency table
observed = np.array([[10, 20, 30],
                     [6, 9, 5]])

chi2, p_value, dof, expected = chi2_contingency(observed)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
```

---

## Maximum Likelihood Estimation {#mle}

### Concept
Find parameters that maximize the likelihood of observing the data.

```python
def log_likelihood_normal(data, mu, sigma):
    """Log-likelihood for normal distribution"""
    n = len(data)
    return -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)

# Generate data from N(5, 2²)
true_mu, true_sigma = 5, 2
data = np.random.normal(true_mu, true_sigma, 1000)

# MLE estimates
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # MLE uses n, not n-1

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mu_mle:.4f}, σ={sigma_mle:.4f}")

# Visualize likelihood surface
mu_range = np.linspace(4, 6, 50)
sigma_range = np.linspace(1, 3, 50)
Mu, Sigma = np.meshgrid(mu_range, sigma_range)

log_lik = np.zeros_like(Mu)
for i in range(len(mu_range)):
    for j in range(len(sigma_range)):
        log_lik[j, i] = log_likelihood_normal(data, mu_range[i], sigma_range[j])

plt.contourf(Mu, Sigma, log_lik, levels=20)
plt.colorbar()
plt.plot(mu_mle, sigma_mle, 'r*', markersize=15, label='MLE')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Log-Likelihood Surface')
plt.legend()
plt.show()
```

---

## Bayesian Inference {#bayesian-inference}

### Prior, Likelihood, Posterior
```python
# Bayesian parameter estimation
# Prior: P(θ)
# Likelihood: P(data | θ)
# Posterior: P(θ | data) ∝ P(data | θ) * P(θ)

# Example: Estimating probability of heads
# Prior: Beta(α, β) - conjugate prior for binomial
from scipy.stats import beta

# Prior: Beta(2, 2) - slightly favors fair coin
alpha_prior, beta_prior = 2, 2
prior_dist = beta(alpha_prior, beta_prior)

# Data: 7 heads out of 10 flips
n_heads, n_tails = 7, 3

# Posterior: Beta(α + n_heads, β + n_tails)
alpha_posterior = alpha_prior + n_heads
beta_posterior = beta_prior + n_tails
posterior_dist = beta(alpha_posterior, beta_posterior)

# Visualize
theta = np.linspace(0, 1, 100)
plt.plot(theta, prior_dist.pdf(theta), label='Prior', linewidth=2)
plt.plot(theta, posterior_dist.pdf(theta), label='Posterior', linewidth=2)
plt.axvline(n_heads / (n_heads + n_tails), color='r', linestyle='--', 
            label='MLE estimate')
plt.xlabel('θ (probability of heads)')
plt.ylabel('Density')
plt.title('Bayesian Update: Coin Flip')
plt.legend()
plt.grid(True)
plt.show()

# Posterior mean
posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
print(f"Posterior mean: {posterior_mean:.4f}")
print(f"MLE estimate: {n_heads / (n_heads + n_tails):.4f}")
```

---

## Applications in ML/DL {#applications}

### 1. Loss Functions as Negative Log-Likelihood
```python
# Cross-entropy loss = negative log-likelihood for classification
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# MSE loss = negative log-likelihood for regression (assuming Gaussian noise)
def mse_loss(y_pred, y_true, sigma=1.0):
    n = len(y_true)
    return n/2 * np.log(2 * np.pi * sigma**2) + np.sum((y_pred - y_true)**2) / (2 * sigma**2)
```

### 2. Regularization as Prior
```python
# L2 regularization = Gaussian prior
# L1 regularization = Laplace prior

# L2: P(θ) ∝ exp(-λ||θ||²/2) - Gaussian prior
# L1: P(θ) ∝ exp(-λ||θ||₁) - Laplace prior

def l2_regularization(theta, lambda_reg):
    """L2 regularization term"""
    return lambda_reg * np.sum(theta**2) / 2

def l1_regularization(theta, lambda_reg):
    """L1 regularization term"""
    return lambda_reg * np.sum(np.abs(theta))
```

### 3. Uncertainty Quantification
```python
# Bayesian neural networks provide uncertainty estimates
def predict_with_uncertainty(model_samples, X):
    """Predict with uncertainty using ensemble"""
    predictions = []
    for model in model_samples:
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return mean_pred, std_pred

# Example: 95% confidence interval
mean, std = predict_with_uncertainty([], [])  # Placeholder
confidence_interval = [mean - 1.96*std, mean + 1.96*std]
```

### 4. A/B Testing
```python
# Statistical test for comparing two models/strategies
def ab_test(group_a_successes, group_a_total, 
            group_b_successes, group_b_total):
    """Compare conversion rates"""
    p_a = group_a_successes / group_a_total
    p_b = group_b_successes / group_b_total
    
    # Z-test for proportions
    p_pooled = (group_a_successes + group_b_successes) / (group_a_total + group_b_total)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/group_a_total + 1/group_b_total))
    z = (p_b - p_a) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return {
        'p_a': p_a,
        'p_b': p_b,
        'z_score': z,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Example
result = ab_test(100, 1000, 120, 1000)
print(f"Group A conversion: {result['p_a']:.4f}")
print(f"Group B conversion: {result['p_b']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
```

---

## Practical Examples {#examples}

### Example 1: Naive Bayes Classifier
```python
class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_params = {}
    
    def fit(self, X, y):
        """Train Naive Bayes classifier"""
        classes = np.unique(y)
        n_samples = len(y)
        
        for c in classes:
            # Prior: P(class)
            self.class_priors[c] = np.sum(y == c) / n_samples
            
            # Likelihood parameters: P(feature | class)
            X_c = X[y == c]
            self.feature_params[c] = {
                'mean': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0) + 1e-6  # Avoid division by zero
            }
    
    def predict(self, X):
        """Predict using Bayes' theorem"""
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.class_priors:
                # P(class | features) ∝ P(features | class) * P(class)
                likelihood = np.prod(norm.pdf(x, 
                    self.feature_params[c]['mean'],
                    self.feature_params[c]['std']))
                posteriors[c] = likelihood * self.class_priors[c]
            
            # Predict class with highest posterior
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42)
nb = NaiveBayes()
nb.fit(X, y)
y_pred = nb.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Naive Bayes Accuracy: {accuracy:.4f}")
```

### Example 2: Confidence Intervals
```python
def confidence_interval(data, confidence=0.95):
    """Compute confidence interval for mean"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # t-distribution for small samples, normal for large
    if n < 30:
        from scipy.stats import t
        t_critical = t.ppf((1 + confidence) / 2, n - 1)
        margin = t_critical * se
    else:
        z_critical = norm.ppf((1 + confidence) / 2)
        margin = z_critical * se
    
    return mean - margin, mean + margin

# Example
data = np.random.normal(5, 2, 100)
ci_lower, ci_upper = confidence_interval(data)
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"True mean: 5.0")
```

---

## Key Takeaways

1. **Probability**: Foundation for understanding uncertainty
2. **Distributions**: Model data and noise
3. **Bayes' Theorem**: Update beliefs with evidence
4. **Expectation/Variance**: Summarize distributions
5. **MLE**: Estimate parameters from data
6. **Hypothesis Testing**: Validate models and compare
7. **Bayesian Inference**: Incorporate prior knowledge

---

## Additional Resources

- **Introduction to Probability (Blitzstein)**: Comprehensive textbook
- **Pattern Recognition and Machine Learning (Bishop)**: ML perspective
- **Bayesian Data Analysis (Gelman)**: Bayesian methods
- **Statistical Rethinking (McElreath)**: Modern Bayesian approach

---

**Master probability and statistics to build robust ML/DL models!**

