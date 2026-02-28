# Causal Inference for Machine Learning: Complete Guide

## Table of Contents
1. [Introduction to Causal Inference](#introduction-to-causal-inference)
2. [Correlation vs Causation](#correlation-vs-causation)
3. [Causal Graphical Models](#causal-graphical-models)
4. [Potential Outcomes Framework](#potential-outcomes-framework)
5. [Randomized Experiments (A/B Testing)](#randomized-experiments-ab-testing)
6. [Observational Data: Confounding](#observational-data-confounding)
7. [Causal Identification](#causal-identification)
8. [Estimation Methods](#estimation-methods)
9. [Machine Learning for Causal Inference](#machine-learning-for-causal-inference)
10. [Uplift Modeling](#uplift-modeling)
11. [Practical Examples](#practical-examples)
12. [Advanced Topics](#advanced-topics)

---

## Introduction to Causal Inference

**Causal Inference** answers "what would happen if we intervened?"—not just "what is associated?" Predicting outcomes under intervention is essential for decisions: treatment effects, policy, recommendations.

### Why Causal Inference Matters in ML

| Scenario | Prediction (ML) | Causal (What we need) |
|----------|-----------------|------------------------|
| Drug efficacy | Who gets better? | Does the drug cause improvement? |
| Recommendation | Who will click? | Will showing this item cause a purchase? |
| Pricing | What's the revenue? | How does price change demand? |
| Churn | Who will leave? | Would this intervention retain them? |

### Key Concepts

- **Treatment (T)**: The intervention (drug, ad, feature)
- **Outcome (Y)**: What we care about (recovery, click, revenue)
- **Confounders (Z)**: Variables affecting both T and Y
- **Effect**: Causal impact of T on Y

---

## Correlation vs Causation

### The Fundamental Problem

**Correlation** = P(Y|T) ≠ P(Y)  
**Causation** = P(Y|do(T)) ≠ P(Y)

**do(T)** denotes intervention (setting T), different from observing T.

### Simpson's Paradox

Averages can reverse when conditioning on a confounder:

```python
import pandas as pd

# Example: Drug effectiveness by gender
# Without conditioning: Drug appears harmful
# With conditioning: Drug helps both groups

data = pd.DataFrame({
    'gender': ['M']*100 + ['F']*100 + ['M']*100 + ['F']*100,
    'treatment': ['drug']*100 + ['drug']*100 + ['placebo']*100 + ['placebo']*100,
    'recovered': [90, 20, 30, 80] + [10, 80, 70, 20]  # Simplified counts
})

# Overall: Drug recovery rate vs Placebo
# Conditioning on gender: Different story
print("Simpson's Paradox: Aggregation can hide or reverse effects")
```

### Example: Ice Cream and Drownings

- **Correlation**: More ice cream sales ↔ more drownings
- **Confounder**: Summer (hot weather)
- **Causation**: Ice cream doesn't cause drownings; summer causes both

---

## Causal Graphical Models

### Directed Acyclic Graphs (DAGs)

Nodes = variables, edges = direct causal influence. No cycles.

```python
# Example DAG: T -> Y, Z -> T, Z -> Y
# Z confounds the T-Y relationship

"""
    Z (confounder, e.g., severity)
   / \
  v   v
  T   Y
(treatment) (outcome)

Observing T tells us about Z, which affects Y.
Intervening on T breaks the Z->T edge.
"""

# Tools: networkx, causalnex, dowhy
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([('Z', 'T'), ('Z', 'Y'), ('T', 'Y')])
# Z = confounder, T = treatment, Y = outcome
```

### d-separation

**d-separation** determines conditional independence in a DAG.

- **Chain**: A → B → C ⇒ A ⊥ C | B
- **Fork**: A ← B → C ⇒ A ⊥ C | B
- **Collider**: A → B ← C ⇒ A ⊥ C (but NOT A ⊥ C | B!)

```python
# Collider bias: Conditioning on a common effect creates spurious association
# Example: Talent -> Success <- Beauty
# Talent and Beauty are independent, but Talent ⊥⊥ Beauty | Success is FALSE
# (Among successful people, talent and beauty may be negatively correlated)
```

### Backdoor Criterion

To estimate causal effect of T on Y, we need to **block all backdoor paths** (paths from T to Y that go backward through T's ancestors).

**Adjustment set**: Variables that block all backdoor paths when conditioned on.

```python
# If Z is the only confounder: adjust for Z
# Causal effect identified by: E[Y|T=1,Z] - E[Y|T=0,Z] averaged over Z
```

---

## Potential Outcomes Framework

### Setup

For each unit i and binary treatment T ∈ {0,1}:

- **Y_i(1)**: Outcome if unit i receives treatment
- **Y_i(0)**: Outcome if unit i does not receive treatment

**Individual treatment effect**: ITE_i = Y_i(1) - Y_i(0)

**Fundamental problem**: We only observe one of Y_i(1), Y_i(0) for each unit.

### Average Treatment Effect (ATE)

ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]

### Average Treatment Effect on the Treated (ATT)

ATT = E[Y(1) - Y(0) | T=1]

### Conditional Average Treatment Effect (CATE)

CATE(x) = E[Y(1) - Y(0) | X=x]

Useful for **personalization**: which users benefit most from treatment?

```python
import numpy as np

def simulate_potential_outcomes(n=1000):
    """Simulate binary treatment, continuous outcome"""
    X = np.random.randn(n, 5)  # Covariates
    T = np.random.binomial(1, 0.5, n)  # Random treatment
    
    # Potential outcomes (we never observe both)
    Y0 = 2 + X[:, 0] + 0.5*X[:, 1] + np.random.randn(n) * 0.5
    Y1 = 3 + X[:, 0] + 0.5*X[:, 1] + np.random.randn(n) * 0.5  # Effect = 1 on average
    
    # Observed outcome
    Y = T * Y1 + (1 - T) * Y0
    return X, T, Y, Y0, Y1

X, T, Y, Y0, Y1 = simulate_potential_outcomes()
true_ate = np.mean(Y1 - Y0)
print(f"True ATE: {true_ate:.4f}")
# Naive comparison (biased if confounding): E[Y|T=1] - E[Y|T=0]
naive = Y[T==1].mean() - Y[T==0].mean()
print(f"Naive difference: {naive:.4f}")
```

---

## Randomized Experiments (A/B Testing)

**Gold standard**: Random assignment of T ensures no confounding.

### Why Randomization Works

- E[Y(1)|T=1] = E[Y(1)] and E[Y(0)|T=0] = E[Y(0)] (in expectation)
- So E[Y|T=1] - E[Y|T=0] = ATE

### A/B Test Analysis

```python
from scipy import stats

def ab_test_analysis(control_outcomes, treatment_outcomes):
    """Two-sample t-test for ATE"""
    t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)
    ate = np.mean(treatment_outcomes) - np.mean(control_outcomes)
    se = np.sqrt(np.var(treatment_outcomes)/len(treatment_outcomes) + 
                 np.var(control_outcomes)/len(control_outcomes))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se
    return {
        'ATE': ate,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper)
    }

# Example
control = np.random.normal(10, 2, 1000)
treatment = np.random.normal(10.5, 2, 1000)  # True effect = 0.5
results = ab_test_analysis(control, treatment)
print(f"ATE: {results['ATE']:.4f}, p-value: {results['p_value']:.4f}")
```

### Power and Sample Size

```python
def required_sample_size(alpha=0.05, power=0.8, effect_size=0.2, sigma=1):
    """Sample size per group for two-sided t-test"""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    n = 2 * (sigma ** 2) * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

n = required_sample_size(effect_size=0.5, sigma=2)
print(f"Required sample per group: {n}")
```

---

## Observational Data: Confounding

Without randomization, **confounders** bias the naive comparison.

### Confounding Diagram

```
     Z (confounder)
    / \
   v   v
   T   Y

Naive: E[Y|T=1] - E[Y|T=0] ≠ ATE
Z causes both T and Y, so T is associated with Y even without causation.
```

### Identification Assumptions

1. **SUTVA**: No interference, one version of treatment
2. **Ignorability**: (Y(0), Y(1)) ⊥ T | X (conditional on X, treatment is as-good-as random)
3. **Collapsibility** or specific parametric assumptions

---

## Causal Identification

### Backdoor Adjustment

If Z satisfies backdoor criterion for (T, Y):

ATE = E_Z [ E[Y|T=1,Z] - E[Y|T=0,Z] ]

```python
def backdoor_ate(df, treatment, outcome, covariates):
    """Estimate ATE via backdoor adjustment"""
    ate = 0
    n = len(df)
    for _, row in df.iterrows():
        # Stratify by covariates, then average
        pass
    # Simpler: use regression
    import statsmodels.formula.api as smf
    formula = f"{outcome} ~ {treatment} + " + " + ".join(covariates)
    model = smf.ols(formula, data=df).fit()
    return model.params[treatment]
```

### Inverse Probability Weighting (IPW)

Weight units by 1/P(T=t|X) so that weighted population is "balanced" in X.

ATE = E[ (T·Y)/e(X) - ((1-T)·Y)/(1-e(X)) ]

where e(X) = P(T=1|X) is the **propensity score**.

```python
from sklearn.linear_model import LogisticRegression

def ipw_ate(X, T, Y):
    """Inverse probability weighting for ATE"""
    ps_model = LogisticRegression().fit(X, T)
    e = ps_model.predict_proba(X)[:, 1]  # P(T=1|X)
    e = np.clip(e, 0.05, 0.95)  # Stabilize
    w1 = T / e
    w0 = (1 - T) / (1 - e)
    return np.mean(w1 * Y) - np.mean(w0 * Y)
```

### Doubly Robust Estimation

Combine outcome regression and IPW. Consistent if either is correct.

```python
def doubly_robust_ate(X, T, Y, mu1_hat, mu0_hat, e_hat):
    """
    mu1_hat, mu0_hat: E[Y|X,T=1], E[Y|X,T=0]
    e_hat: P(T=1|X)
    """
    term1 = T * (Y - mu1_hat) / e_hat + mu1_hat
    term0 = (1 - T) * (Y - mu0_hat) / (1 - e_hat) + mu0_hat
    return np.mean(term1 - term0)
```

---

## Estimation Methods

### Propensity Score Matching

Match each treated unit with similar control unit(s) by propensity score.

```python
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(X, T, Y, k=5):
    """Match on propensity score"""
    ps_model = LogisticRegression().fit(X, T)
    e = ps_model.predict_proba(X)[:, 1]
    
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    e_treated = e[treated_idx].reshape(-1, 1)
    e_control = e[control_idx].reshape(-1, 1)
    
    nn = NearestNeighbors(n_neighbors=k).fit(e_control)
    distances, indices = nn.kneighbors(e_treated)
    
    ate = 0
    for i, idx in enumerate(treated_idx):
        matched_controls = control_idx[indices[i]]
        ate += Y[idx] - Y[matched_controls].mean()
    ate /= len(treated_idx)
    return ate
```

### Regression Discontinuity (RD)

When treatment is determined by a cutoff: T = 1{X ≥ c}. Compare just above vs just below cutoff.

### Instrumental Variables (IV)

When unobserved confounder exists, use variable Z (instrument) that:
- Affects T
- Affects Y only through T
- Independent of confounders

---

## Machine Learning for Causal Inference

### Causal ML Goals

1. **Heterogeneous treatment effects**: CATE(x) = E[Y(1)-Y(0)|X=x]
2. **Off-policy evaluation**: Estimate reward of new policy from logged data
3. **Causal discovery**: Learn DAG from data

### Meta-Learners

**T-Learner**: Separate models for μ_1(x)=E[Y|X,T=1], μ_0(x)=E[Y|X,T=0]. CATE(x) = μ_1(x) - μ_0(x).

```python
from sklearn.ensemble import RandomForestRegressor

class TLearner:
    def __init__(self):
        self.model_1 = RandomForestRegressor()
        self.model_0 = RandomForestRegressor()
    
    def fit(self, X, T, Y):
        self.model_1.fit(X[T==1], Y[T==1])
        self.model_0.fit(X[T==0], Y[T==0])
    
    def predict_cate(self, X):
        return self.model_1.predict(X) - self.model_0.predict(X)
```

**X-Learner**: Uses control to predict treatment outcomes (and vice versa) for propensity weighting.

**S-Learner**: Single model μ(x,t). CATE(x) = μ(x,1) - μ(x,0).

```python
class SLearner:
    def __init__(self):
        self.model = RandomForestRegressor()
    
    def fit(self, X, T, Y):
        X_with_t = np.hstack([X, T.reshape(-1, 1)])
        self.model.fit(X_with_t, Y)
    
    def predict_cate(self, X):
        n = len(X)
        return (self.model.predict(np.hstack([X, np.ones((n,1))])) -
                self.model.predict(np.hstack([X, np.zeros((n,1))])))
```

### Causal Forest (GRF)

```python
# Generalized Random Forests - R package grf
# Python: econml (Microsoft)
# pip install econml

from econml.grf import CausalForest

cf = CausalForest(n_estimators=1000, max_depth=5)
cf.fit(Y, T, X=X)
cate = cf.effect(X_test)
```

### Double Machine Learning (DML)

1. Predict Y from X (residualize)
2. Predict T from X (residualize)
3. Regress Y_resid on T_resid → causal effect

```python
# EconML DML
from econml.dml import CausalForestDML

dml = CausalForestDML(
    model_y=RandomForestRegressor(),
    model_t=RandomForestRegressor(),
    n_estimators=1000
)
dml.fit(Y, T, X=X)
cate = dml.effect(X_test)
```

---

## Uplift Modeling

**Uplift** = P(Y=1|T=1,X) - P(Y=1|T=0,X) = net effect of treatment for segment X.

### Use Cases

- **Targeting**: Only treat users with positive uplift
- **Marketing**: Avoid treating those who would buy anyway (persuadables)
- **Clinical**: Treat those who benefit

### Uplift Trees

```python
# Criterion: maximize variance in uplift across splits
# Packages: upliftpy, causalml

from causalml.inference.tree import UpliftTreeClassifier

uplift_model = UpliftTreeClassifier(max_depth=5)
uplift_model.fit(X, treatment=T, y=Y)
uplift = uplift_model.predict(X_test, treatment=1) - uplift_model.predict(X_test, treatment=0)
```

### Qini Curve

Cumulative uplift when targeting by predicted uplift (descending).

```python
def qini_curve(y, treatment, uplift_pred):
    """Qini: cumulative uplift vs random"""
    order = np.argsort(-uplift_pred)
    n = len(order)
    cum_treated = np.cumsum(treatment[order])
    cum_y_treated = np.cumsum(y[order] * treatment[order])
    cum_y_control = np.cumsum(y[order] * (1 - treatment[order]))
    # Normalize
    cum_uplift = cum_y_treated / (cum_treated + 1e-10) - cum_y_control / (np.arange(1, n+1) - cum_treated + 1e-10)
    return np.cumsum(cum_uplift)
```

---

## Practical Examples

### Example: Campaign ROI

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Simulated campaign data
np.random.seed(42)
n = 5000
X = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.exponential(50, n),
    'past_purchases': np.random.poisson(5, n)
})
T = (X['income'] > 50).astype(int)  # "Targeted" (confounded by income)
Y = (0.1 + 0.02*T + 0.01*X['income']/50 + np.random.randn(n)*0.1 > 0.5).astype(int)

# Naive: E[Y|T=1] - E[Y|T=0] overstates (targeted are richer)
naive = Y[T==1].mean() - Y[T==0].mean()
print(f"Naive lift: {naive:.4f}")

# CATE: who benefits?
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.2, random_state=42
)

t_learner = TLearner()
t_learner.fit(X_train.values, T_train.values, Y_train.values)
cate = t_learner.predict_cate(X_test.values)
print(f"Mean CATE: {cate.mean():.4f}")
print(f"Fraction positive uplift: {(cate > 0).mean():.2%}")
```

### Example: DoWhy

```python
# pip install dowhy

from dowhy import CausalModel

df = pd.DataFrame({'Z': X['income'], 'T': T, 'Y': Y})

model = CausalModel(
    data=df,
    treatment='T',
    outcome='Y',
    common_causes=['Z']
)

identified = model.identify_effect()
estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
print(f"Estimated ATE: {estimate.value}")
```

---

## Advanced Topics

### Mediation Analysis

Decompose effect into direct (T→Y) and indirect (T→M→Y).

### Time-Varying Confounding

G-methods, marginal structural models, inverse probability of treatment weighting (IPTW) over time.

### Causal Discovery

Learn DAG from data: PC algorithm, GES, NOTEARS (continuous optimization).

```python
# pip install causalnex
# NOTEARS: differentiable DAG learning
```

---

## Best Practices

1. **Draw the DAG** before analysis
2. **Sensitivity analysis**: How robust to unmeasured confounding?
3. **Use multiple estimators** (IPW, outcome regression, doubly robust) and compare
4. **Validate** on RCT if possible
5. **Report confidence intervals** and assumptions

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Correlation ≠ Causation | Confounding, do-calculus |
| RCT | Gold standard, no confounding |
| Observational | Backdoor, IPW, doubly robust |
| CATE | Personalization, uplift |
| ML for causal | Meta-learners, causal forest, DML |

**Libraries**: `econml`, `causalml`, `dowhy`, `DoWhy`, `causalnex`
