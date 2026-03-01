# Causal Inference for Machine Learning: Comprehensive Guide

## Table of Contents
1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Correlation vs Causation](#2-correlation-vs-causation)
3. [Potential Outcomes Framework (Rubin Causal Model)](#3-potential-outcomes-framework-rubin-causal-model)
4. [Structural Causal Models and DAGs](#4-structural-causal-models-and-dags)
5. [Pearl's Do-Calculus](#5-pearls-do-calculus)
6. [Randomized Controlled Trials](#6-randomized-controlled-trials)
7. [Observational Study Methods](#7-observational-study-methods)
8. [Propensity Score Methods](#8-propensity-score-methods)
9. [Doubly Robust Estimators](#9-doubly-robust-estimators)
10. [Difference-in-Differences](#10-difference-in-differences)
11. [Regression Discontinuity Design](#11-regression-discontinuity-design)
12. [Instrumental Variables](#12-instrumental-variables)
13. [Synthetic Control Method](#13-synthetic-control-method)
14. [Heterogeneous Treatment Effects and CATE](#14-heterogeneous-treatment-effects-and-cate)
15. [Meta-Learners for CATE](#15-meta-learners-for-cate)
16. [Causal Forests and Double ML](#16-causal-forests-and-double-ml)
17. [Causal Discovery](#17-causal-discovery)
18. [Uplift Modeling](#18-uplift-modeling)
19. [Causal Inference Libraries](#19-causal-inference-libraries)
20. [Full End-to-End Code Examples](#20-full-end-to-end-code-examples)

---

## 1. Introduction and Motivation

Causal inference is the science of determining **cause-and-effect relationships** from data. Standard machine learning is primarily predictive — it answers "what will happen?" — but causal inference answers "what would happen **if** we intervened?"

### Why Predictive ML Is Insufficient for Decision-Making

| Question | Predictive ML | Causal Inference |
|----------|--------------|-----------------|
| Who will churn? | ✓ (predicts) | ✓ |
| Will this retention offer prevent churn? | ✗ | ✓ (intervention) |
| Should we raise prices? | ✗ | ✓ (counterfactual) |
| Does drug X cause recovery? | ✗ | ✓ (causal effect) |
| Why did sales drop? | Partial | ✓ (attribution) |

### Core Concepts Terminology

- **Treatment (T)**: The intervention or exposure (drug dose, ad, feature flag)
- **Outcome (Y)**: The variable we care about (recovery, revenue, clicks)
- **Covariates (X)**: Pre-treatment features (age, demographics, history)
- **Confounder (Z)**: A variable that causally affects both T and Y — creates spurious correlations
- **Collider**: A variable that is caused by both T and Y — conditioning on it creates bias
- **Mediator (M)**: A variable on the causal path T → M → Y
- **Effect**: The causal impact of T on Y, represented as P(Y | do(T)) vs P(Y)

---

## 2. Correlation vs Causation

### The Fundamental Distinction

**Statistical association**: \( P(Y|T=1) \neq P(Y|T=0) \)

**Causal effect** (Pearl's do-notation): \( P(Y \mid do(T=1)) \neq P(Y \mid do(T=0)) \)

The **do-operator** represents an intervention — physically setting T to a value by breaking its natural causal mechanism. Observing T=1 may tell us about background variables (e.g., sicker patients get treatment), but intervening on T forces T=1 regardless of background.

### Simpson's Paradox

A trend in subgroups can reverse when groups are combined. This is not just a statistical curiosity — it arises from confounding and has led to incorrect medical conclusions.

**Classic example** — UC Berkeley admissions (1973):
- Overall: Men had higher admission rate than women
- By department: Women had equal or higher admission rates
- **Explanation**: Women applied to more competitive departments (the confounder)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate Simpson's Paradox: Drug effectiveness by disease severity
np.random.seed(42)
n = 2000

# Severity (confounder): 0=mild, 1=severe
severity = np.random.binomial(1, 0.5, n)

# Treatment assignment confounded: severe patients less likely to receive drug
# (maybe drug has side effects, so doctors avoid for severe cases)
p_treat = np.where(severity == 0, 0.7, 0.3)
treatment = np.random.binomial(1, p_treat, n)

# Outcome: recovery
# Drug helps (causal effect = +0.2), but severity hurts (-0.4)
p_recover = 0.5 + 0.2 * treatment - 0.4 * severity
p_recover = np.clip(p_recover, 0, 1)
recovery = np.random.binomial(1, p_recover, n)

df = pd.DataFrame({
    'severity': severity,
    'treatment': treatment,
    'recovery': recovery
})

# Naive comparison: drug appears harmful!
print("=== Simpson's Paradox Demo ===")
naive = df.groupby('treatment')['recovery'].mean()
print(f"Naive: Recovery with drug={naive[1]:.3f}, without drug={naive[0]:.3f}")
print(f"Naive ATE = {naive[1] - naive[0]:.3f}  <-- WRONG, appears harmful")

# Stratified: drug helps in both groups
for sev in [0, 1]:
    subdf = df[df['severity'] == sev]
    rate = subdf.groupby('treatment')['recovery'].mean()
    print(f"Severity={sev}: With drug={rate.get(1,0):.3f}, Without={rate.get(0,0):.3f}, Effect={rate.get(1,0)-rate.get(0,0):.3f}")

# Correct causal estimate (adjust for severity)
from sklearn.linear_model import LogisticRegression
X = df[['treatment', 'severity']].values
y = df['recovery'].values
# Adjusted estimate
from statsmodels.formula.api import logit
model = logit('recovery ~ treatment + severity', data=df).fit(disp=0)
print(f"\nAdjusted treatment effect: {model.params['treatment']:.3f} (log-odds)")
```

### Confounding vs Other Sources of Bias

| Bias Type | Cause | Example |
|-----------|-------|---------|
| **Confounding** | Common cause of T and Y | Severity drives both treatment and recovery |
| **Selection bias** | Selection on T or Y | Survivors studied, not general pop |
| **Measurement bias** | Mismeasured variables | Self-reported diet data |
| **Collider bias** | Conditioning on collider | Studying only hospitalized patients |
| **Reverse causation** | Y causes T, not T→Y | Health causes exercise, not just vice versa |

### Ice Cream and Drownings — Classic Confounder

```
Ice cream sales → [+] Drownings?
       ↑                  ↑
       Summer (heat)  ─────┘
```

Both ice cream and drownings increase in summer (hot weather). Controlling for temperature eliminates the correlation. This is textbook confounding.

---

## 3. Potential Outcomes Framework (Rubin Causal Model)

Developed by Donald Rubin (1974), the **potential outcomes** (or counterfactual) framework is one of two dominant frameworks in causal inference.

### Notation and Setup

For each unit \(i\) with binary treatment \(T_i \in \{0,1\}\):

- \(Y_i(1)\): Potential outcome **if** unit i receives treatment
- \(Y_i(0)\): Potential outcome **if** unit i does not receive treatment

The **Individual Treatment Effect (ITE)**:

\[
\tau_i = Y_i(1) - Y_i(0)
\]

**Fundamental Problem of Causal Inference**: We only ever observe ONE of the two potential outcomes for each unit. The other is the **counterfactual** — what would have happened under the alternative treatment.

### Causal Estimands

**Average Treatment Effect (ATE)**:
\[
\text{ATE} = \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y_i(1)] - \mathbb{E}[Y_i(0)]
\]

**Average Treatment Effect on the Treated (ATT)**:
\[
\text{ATT} = \mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 1]
\]
ATT answers: "What was the effect of treatment on those who actually received it?" — useful when treatment is voluntary.

**Average Treatment Effect on the Control (ATC)**:
\[
\text{ATC} = \mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 0]
\]
ATC answers: "What would happen to untreated units if they were treated?" — useful for policy targeting.

**Conditional Average Treatment Effect (CATE)**:
\[
\tau(x) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x]
\]
CATE allows personalization: who benefits most from treatment?

```python
import numpy as np

def simulate_rubin_model(n=5000, confounded=True):
    """
    Simulate potential outcomes with optional confounding.
    True ATE = 1.0, CATE varies by age.
    """
    np.random.seed(42)
    
    # Covariates
    age = np.random.uniform(20, 60, n)
    income = np.random.exponential(50000, n)
    
    # Heterogeneous individual effects: older users benefit more
    true_ite = 0.5 + 0.01 * (age - 40)   # avg=0.5
    true_ate = true_ite.mean()
    
    # Potential outcomes
    Y0 = 2 + 0.01 * age + 0.00001 * income + np.random.randn(n) * 0.5
    Y1 = Y0 + true_ite
    
    if confounded:
        # Higher-income users more likely to get treatment
        p_treat = 1 / (1 + np.exp(-(0.00001 * income - 0.5)))
    else:
        p_treat = np.full(n, 0.5)  # RCT
    
    T = np.random.binomial(1, p_treat, n)
    
    # Observed outcome (only one potential outcome per unit)
    Y_obs = T * Y1 + (1 - T) * Y0
    
    return {
        'age': age, 'income': income, 'T': T,
        'Y_obs': Y_obs, 'Y0': Y0, 'Y1': Y1,
        'true_ite': true_ite, 'true_ate': true_ate
    }

data = simulate_rubin_model(confounded=True)
print(f"True ATE: {data['true_ate']:.4f}")
print(f"Naive estimate (biased): {data['Y_obs'][data['T']==1].mean() - data['Y_obs'][data['T']==0].mean():.4f}")
print(f"True CATE range: [{data['true_ite'].min():.3f}, {data['true_ite'].max():.3f}]")
```

### SUTVA — Stable Unit Treatment Value Assumption

SUTVA is a foundational assumption:

1. **No interference**: Unit i's outcome depends only on i's own treatment, not other units' treatments
   - Violated in networks (if your friend gets vaccinated, your risk changes)
   - Violated in markets (one person's price affects others through supply/demand)

2. **No hidden versions of treatment**: There is only one version of each treatment level
   - Violated if "drug X" means different doses

When SUTVA fails, we need spillover/interference models.

### Identification Assumptions for Observational Studies

To identify ATE without randomization, we need:

1. **Ignorability** (Unconfoundedness): \( \{Y(0), Y(1)\} \perp T \mid X \)
   - Given X, treatment assignment is "as good as random"
   - This is **untestable** from observed data

2. **Overlap** (Positivity): \( 0 < P(T=1 \mid X=x) < 1 \) for all x in the support
   - Every type of unit has some chance of receiving either treatment
   - Violated if some subgroups always/never treated

3. **SUTVA** (see above)

---

## 4. Structural Causal Models and DAGs

Judea Pearl's **Structural Causal Model (SCM)** framework uses directed acyclic graphs (DAGs) to represent causal structure.

### Directed Acyclic Graphs (DAGs)

A DAG is a set of nodes (variables) and directed edges (direct causal relationships) with no directed cycles.

```python
import networkx as nx
import matplotlib.pyplot as plt

def draw_dag(edges, title="DAG"):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=1500, font_size=12, arrows=True,
            arrowsize=20, edge_color='gray', width=2)
    plt.title(title)
    plt.show()

# Classic confounded DAG: Z → T, Z → Y, T → Y
confounded_dag = [('Z', 'T'), ('Z', 'Y'), ('T', 'Y')]
draw_dag(confounded_dag, "Confounded: Z confounds T→Y")

# Mediation DAG: T → M → Y, T → Y
mediation_dag = [('T', 'M'), ('M', 'Y'), ('T', 'Y')]
draw_dag(mediation_dag, "Mediation: T affects Y directly and through M")

# Collider DAG: T → C ← Y
collider_dag = [('T', 'C'), ('Y', 'C')]
draw_dag(collider_dag, "Collider: T and Y both cause C")
```

### Structural Equations

An SCM is a set of structural equations, one per variable:

\[
X_i := f_i(\text{Parents}(X_i), U_i)
\]

where \(U_i\) is an exogenous noise term. Example:

```
Z := U_Z                    (Z is exogenous)
T := σ(α·Z + U_T)          (T depends on Z)  
Y := β·T + γ·Z + U_Y       (Y depends on T and Z)
```

```python
# Simulate a structural causal model
def scm_simulate(n=1000, beta=2.0, gamma=1.0, alpha=1.5):
    """
    SCM: Z → T, Z → Y, T → Y
    Y := beta*T + gamma*Z + noise
    """
    np.random.seed(42)
    U_Z = np.random.randn(n)
    U_T = np.random.randn(n) * 0.5
    U_Y = np.random.randn(n) * 0.5
    
    Z = U_Z  # Exogenous confounder
    T_logit = alpha * Z + U_T
    T = (T_logit > 0).astype(float)  # Binary treatment
    Y = beta * T + gamma * Z + U_Y
    
    return Z, T, Y, beta  # beta is the true causal effect

Z, T, Y, true_effect = scm_simulate()
print(f"True causal effect: {true_effect}")
print(f"Naive (biased): {Y[T==1].mean() - Y[T==0].mean():.3f}")
```

### d-Separation: Reading Conditional Independences from DAGs

**d-separation** is an algorithm to determine if two variables are conditionally independent given a set of observed variables, by examining the graph structure.

**Three types of connections:**

1. **Chain**: \( A \to B \to C \)
   - A and C are d-separated given B: \( A \perp C \mid B \)
   - Information flows through B but is blocked when B is conditioned on

2. **Fork** (common cause): \( A \leftarrow B \rightarrow C \)
   - A and C are d-separated given B: \( A \perp C \mid B \)
   - B is a confounder; conditioning blocks the spurious association

3. **Collider**: \( A \rightarrow B \leftarrow C \)
   - A and C are d-separated (NOT given B): \( A \perp C \)
   - **Conditioning on B OPENS a path!** Explains collider bias

```python
# Collider bias demonstration
n = 10000
# T and Y are independent (no causal relationship)
T = np.random.randn(n)
Y = np.random.randn(n)
# Collider: C is caused by both T and Y
C = (T + Y + np.random.randn(n) * 0.5 > 1).astype(bool)

corr_overall = np.corrcoef(T, Y)[0,1]
corr_given_C = np.corrcoef(T[C], Y[C])[0,1]

print(f"Correlation T,Y (overall): {corr_overall:.4f}")
print(f"Correlation T,Y (conditioning on C=True): {corr_given_C:.4f}")
print("→ Conditioning on the collider CREATES spurious correlation!")
```

---

## 5. Pearl's Do-Calculus

The **do-operator** \(do(T=t)\) represents an intervention: physically setting T to value t by removing all arrows into T in the DAG (cutting T from its causes).

### Observing vs Intervening

\[
P(Y \mid T=1) \neq P(Y \mid do(T=1))
\]

Observing T=1 in a confounded system gives us P(Y|T=1) which includes the confounder's contribution. Intervening sets T=1 while keeping everything else natural.

### The Backdoor Criterion

A set of variables **Z satisfies the backdoor criterion** for the causal effect of T on Y if:

1. No variable in Z is a descendant of T
2. Z blocks every "backdoor path" from T to Y (paths that start with an arrow into T)

If Z satisfies the backdoor criterion:
\[
P(Y \mid do(T=t)) = \sum_z P(Y \mid T=t, Z=z) \cdot P(Z=z)
\]

```python
# Backdoor adjustment implementation
def backdoor_adjustment(df, treatment_col, outcome_col, adjustment_set):
    """
    Estimate causal effect via backdoor adjustment (standardization).
    Works for continuous outcome with regression outcome model.
    """
    from sklearn.linear_model import LinearRegression
    
    # Fit outcome model E[Y | T, Z]
    features = [treatment_col] + list(adjustment_set)
    X = df[features].values
    y = df[outcome_col].values
    
    model = LinearRegression().fit(X, y)
    
    # Standardize: E_Z[ E[Y|T=1,Z] - E[Y|T=0,Z] ]
    n = len(df)
    X_treat1 = df[features].copy()
    X_treat0 = df[features].copy()
    X_treat1[treatment_col] = 1
    X_treat0[treatment_col] = 0
    
    Y1_pred = model.predict(X_treat1.values)
    Y0_pred = model.predict(X_treat0.values)
    
    ate = (Y1_pred - Y0_pred).mean()
    return ate

# Example usage
df = pd.DataFrame({'Z': Z, 'T': T, 'Y': Y})
adj_ate = backdoor_adjustment(df, 'T', 'Y', ['Z'])
print(f"Backdoor-adjusted ATE: {adj_ate:.3f}")
```

### The Frontdoor Criterion

When all confounders are unobserved but there is a mediator M on the T→Y path (and T→M has no unmeasured confounding, and M→Y has no unmeasured confounding):

\[
P(Y \mid do(T=t)) = \sum_m \left[\sum_{t'} P(Y \mid T=t', M=m) P(T=t')\right] P(M=m \mid T=t)
\]

This is remarkable: it allows identification of the causal effect even with unmeasured confounders!

### Do-Calculus Rules (Pearl)

Three rules that together allow manipulation of interventional distributions:

1. **Rule 1** (Insertion/deletion of observations): 
   \( P(y \mid do(x), z, w) = P(y \mid do(x), w) \) if \( Y \perp_d Z \mid X, W \) in \(G_{\overline{X}}\)

2. **Rule 2** (Action/observation exchange):
   \( P(y \mid do(x), do(z), w) = P(y \mid do(x), z, w) \) if \( Y \perp_d Z \mid X, W \) in \(G_{\overline{X}\underline{Z}}\)

3. **Rule 3** (Insertion/deletion of actions):
   \( P(y \mid do(x), do(z), w) = P(y \mid do(x), w) \) if \( Y \perp_d Z \mid X, W \) in \(G_{\overline{X}, \overline{Z(W)}}\)

---

## 6. Randomized Controlled Trials

RCTs are the **gold standard** for causal inference because randomization ensures:

\[
\{Y(0), Y(1)\} \perp T
\]

This means: \(\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0] = \text{ATE}\)

### Why Randomization Eliminates Confounding

When T is assigned randomly:
- Treated and control groups are balanced on ALL covariates (observed and unobserved) in expectation
- There are no backdoor paths (no common causes of T and Y)
- The causal graph becomes: \(T \to Y\) with no arrows into T

### Complete RCT Analysis

```python
from scipy import stats
import numpy as np

class RCTAnalysis:
    """Complete RCT analysis toolkit."""
    
    def __init__(self, Y_control, Y_treatment, alpha=0.05):
        self.Y_c = np.array(Y_control)
        self.Y_t = np.array(Y_treatment)
        self.alpha = alpha
        self.n_c = len(self.Y_c)
        self.n_t = len(self.Y_t)
    
    def ate(self):
        return self.Y_t.mean() - self.Y_c.mean()
    
    def standard_error(self):
        return np.sqrt(self.Y_t.var(ddof=1)/self.n_t + 
                       self.Y_c.var(ddof=1)/self.n_c)
    
    def confidence_interval(self):
        z = stats.norm.ppf(1 - self.alpha/2)
        se = self.standard_error()
        ate = self.ate()
        return ate - z*se, ate + z*se
    
    def p_value(self):
        t_stat = self.ate() / self.standard_error()
        return 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    def welch_ttest(self):
        return stats.ttest_ind(self.Y_t, self.Y_c, equal_var=False)
    
    def mann_whitney_u(self):
        """Non-parametric test (when normality fails)."""
        return stats.mannwhitneyu(self.Y_t, self.Y_c, alternative='two-sided')
    
    def cohens_d(self):
        """Standardized effect size."""
        pooled_std = np.sqrt((self.Y_t.var(ddof=1) + self.Y_c.var(ddof=1)) / 2)
        return self.ate() / pooled_std
    
    def minimum_detectable_effect(self, power=0.8):
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(power)
        pooled_std = np.sqrt((self.Y_t.var(ddof=1) + self.Y_c.var(ddof=1)) / 2)
        n = (self.n_t + self.n_c) / 2
        return (z_alpha + z_beta) * pooled_std * np.sqrt(2/n)
    
    def report(self):
        ate = self.ate()
        ci = self.confidence_interval()
        p = self.p_value()
        d = self.cohens_d()
        print(f"=== RCT Results ===")
        print(f"Control mean: {self.Y_c.mean():.4f} ± {self.Y_c.std():.4f}")
        print(f"Treatment mean: {self.Y_t.mean():.4f} ± {self.Y_t.std():.4f}")
        print(f"ATE: {ate:.4f}")
        print(f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
        print(f"p-value: {p:.4f} {'(significant)' if p < self.alpha else '(not significant)'}")
        print(f"Cohen's d: {d:.4f}")

# Example
np.random.seed(42)
control = np.random.normal(10, 2, 500)
treatment = np.random.normal(10.8, 2.2, 500)  # True ATE = 0.8

rct = RCTAnalysis(control, treatment)
rct.report()
```

### Sample Size Calculation

```python
def rct_sample_size(effect_size, sigma, alpha=0.05, power=0.8, two_sided=True):
    """
    Calculate required sample size per group for two-sample t-test.
    
    Parameters:
    -----------
    effect_size: absolute difference in means (delta)
    sigma: common standard deviation
    alpha: Type I error rate
    power: 1 - Type II error rate
    
    Returns: n per group
    """
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/(2 if two_sided else 1))
    z_beta = norm.ppf(power)
    
    n = 2 * (sigma ** 2) * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# Example: detect 5% conversion lift (baseline 10%, sigma~0.3)
n = rct_sample_size(effect_size=0.05, sigma=0.3, alpha=0.05, power=0.8)
print(f"Required sample per group: {n}")
print(f"Total participants needed: {2*n}")
```

### Common A/B Testing Pitfalls

1. **Peeking**: Stopping when p < 0.05 without pre-registering → inflated false positive rate
2. **Multiple comparisons**: Testing many metrics → use Bonferroni or FDR correction
3. **Network effects**: SUTVA violated if users interact (use cluster randomization)
4. **Novelty effects**: Treatment boost is temporary; users revert to baseline
5. **Survivorship bias**: Only analyze users who complete the experiment

---

## 7. Observational Study Methods

When randomization is impossible (ethical, practical), we work with observational data and must be explicit about assumptions.

### Regression Adjustment

The simplest approach: include confounders in a regression model.

\[
Y = \alpha + \tau T + \beta Z + \epsilon
\]

Under ignorability, \(\hat{\tau}\) estimates the ATE. **Problems**: model misspecification, nonlinear effects, interaction terms.

```python
import statsmodels.api as sm

def regression_adjustment_ate(df, outcome, treatment, covariates):
    """OLS regression adjustment for ATE."""
    X = sm.add_constant(df[[treatment] + covariates])
    model = sm.OLS(df[outcome], X).fit()
    print(model.summary())
    return model.params[treatment], model.conf_int().loc[treatment]

# Standardization (more robust to extrapolation)
def standardization_ate(df, outcome, treatment, covariates):
    """
    Standardization / G-computation.
    Fit flexible outcome model, then marginalize over covariate distribution.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    X_train = df[[treatment] + covariates].values
    y_train = df[outcome].values
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    model.fit(X_train, y_train)
    
    # Predict potential outcomes under T=1 and T=0 for all units
    df_t1 = df.copy(); df_t1[treatment] = 1
    df_t0 = df.copy(); df_t0[treatment] = 0
    
    Y1_pred = model.predict(df_t1[[treatment] + covariates].values)
    Y0_pred = model.predict(df_t0[[treatment] + covariates].values)
    
    ate = (Y1_pred - Y0_pred).mean()
    return ate, Y1_pred - Y0_pred  # returns CATE estimates too
```

---

## 8. Propensity Score Methods

The **propensity score** \(e(x) = P(T=1 \mid X=x)\) is the probability of receiving treatment given covariates.

**Key theorem** (Rosenbaum & Rubin, 1983): If ignorability holds conditional on X, it also holds conditional on e(X):
\[
\{Y(0), Y(1)\} \perp T \mid e(X)
\]

This collapses a high-dimensional covariate adjustment into a 1D problem!

### Propensity Score Estimation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def estimate_propensity_scores(X, T, method='logistic', calibrate=True):
    """
    Estimate propensity scores P(T=1|X).
    
    Parameters:
    -----------
    X: covariates (n, p)
    T: binary treatment (n,)
    method: 'logistic', 'gbm', or 'rf'
    calibrate: apply Platt scaling for better calibration
    """
    if method == 'logistic':
        base_model = LogisticRegression(C=1.0, max_iter=1000)
    elif method == 'gbm':
        base_model = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    elif method == 'rf':
        base_model = RandomForestClassifier(n_estimators=200, min_samples_leaf=10)
    
    if calibrate and method != 'logistic':
        model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
    else:
        model = base_model
    
    model.fit(X, T)
    ps = model.predict_proba(X)[:, 1]
    return ps

def check_overlap(ps, T, trim=0.01):
    """Check positivity assumption and trim extreme propensity scores."""
    import matplotlib.pyplot as plt
    
    # Visual check
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(ps[T==0], bins=50, alpha=0.5, label='Control', density=True)
    axes[0].hist(ps[T==1], bins=50, alpha=0.5, label='Treated', density=True)
    axes[0].set_xlabel('Propensity Score')
    axes[0].set_title('Propensity Score Distribution')
    axes[0].legend()
    
    # Trim extreme values
    trimmed_mask = (ps > trim) & (ps < 1 - trim)
    trimming_rate = 1 - trimmed_mask.mean()
    print(f"Trimming {trimming_rate:.1%} of units with extreme PS")
    
    return ps, trimmed_mask
```

### Propensity Score Matching (PSM)

Match each treated unit to one or more control units with similar propensity scores.

```python
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

class PropensityScoreMatching:
    """
    Propensity Score Matching with multiple matching strategies.
    """
    
    def __init__(self, method='nearest', k=1, caliper=None):
        """
        Parameters:
        -----------
        method: 'nearest', 'radius', or 'kernel'
        k: number of control matches per treated unit
        caliper: max allowed PS distance (None = no limit)
        """
        self.method = method
        self.k = k
        self.caliper = caliper
    
    def fit(self, X, T, Y):
        self.X = X
        self.T = T
        self.Y = Y
        return self
    
    def estimate_att(self, ps):
        """Estimate ATT via matching."""
        treated_idx = np.where(self.T == 1)[0]
        control_idx = np.where(self.T == 0)[0]
        
        ps_treated = ps[treated_idx].reshape(-1, 1)
        ps_control = ps[control_idx].reshape(-1, 1)
        
        nn = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn.fit(ps_control)
        
        distances, indices = nn.kneighbors(ps_treated)
        
        ite_estimates = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if self.caliper is not None and dist.min() > self.caliper:
                continue  # Skip if no close match
            treated_y = self.Y[treated_idx[i]]
            matched_y = self.Y[control_idx[idx]].mean()
            ite_estimates.append(treated_y - matched_y)
        
        att = np.mean(ite_estimates)
        se = np.std(ite_estimates) / np.sqrt(len(ite_estimates))
        return {
            'ATT': att,
            'SE': se,
            'CI_95': (att - 1.96*se, att + 1.96*se),
            'n_matched': len(ite_estimates)
        }
    
    def balance_check(self, ps, covariates):
        """Check covariate balance before and after matching."""
        treated_idx = np.where(self.T == 1)[0]
        control_idx = np.where(self.T == 0)[0]
        
        # Standardized mean differences
        def smd(x1, x2):
            pooled_std = np.sqrt((x1.var() + x2.var()) / 2)
            return abs(x1.mean() - x2.mean()) / (pooled_std + 1e-8)
        
        print("Covariate balance (|SMD| < 0.1 is good):")
        print(f"{'Covariate':<20} {'Before':<12} {'After':<12}")
        for j, cov_name in enumerate(covariates):
            before = smd(self.X[treated_idx, j], self.X[control_idx, j])
            # After matching (simplified: use PS weights)
            w = np.where(self.T == 1, 1, ps / (1 - ps + 1e-8))
            after = smd(self.X[treated_idx, j], 
                       np.average(self.X[control_idx, j:j+1], 
                                  weights=w[control_idx], axis=0))
            print(f"{cov_name:<20} {before:<12.4f} {after:<12.4f}")
```

### Inverse Probability Weighting (IPW)

Re-weight units so that each stratum of confounders is equally represented in treatment and control groups.

\[
\hat{\text{ATE}}_{\text{IPW}} = \frac{1}{n} \sum_{i=1}^n \left[\frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i) Y_i}{1-e(X_i)}\right]
\]

For ATT, use different weights:
\[
\hat{\text{ATT}}_{\text{IPW}} = \frac{\sum_i T_i Y_i}{\sum_i T_i} - \frac{\sum_i \frac{e(X_i)}{1-e(X_i)} (1-T_i) Y_i}{\sum_i \frac{e(X_i)}{1-e(X_i)} (1-T_i)}
\]

```python
def ipw_ate(T, Y, ps, stabilize=True, trim_quantile=0.01):
    """
    Inverse Probability Weighting for ATE.
    
    Parameters:
    -----------
    T: treatment indicator
    Y: outcome
    ps: propensity scores P(T=1|X)
    stabilize: use stabilized weights (recommended)
    trim_quantile: trim extreme weights
    """
    # Trim extreme propensity scores
    ps_trimmed = np.clip(ps, trim_quantile, 1 - trim_quantile)
    
    if stabilize:
        # Stabilized weights: multiply by marginal P(T)
        p_t = T.mean()
        w1 = (T * p_t) / ps_trimmed
        w0 = ((1 - T) * (1 - p_t)) / (1 - ps_trimmed)
    else:
        w1 = T / ps_trimmed
        w0 = (1 - T) / (1 - ps_trimmed)
    
    # Trim extreme weights
    w_max = np.percentile(np.concatenate([w1[T==1], w0[T==0]]), 99)
    w1 = np.minimum(w1, w_max)
    w0 = np.minimum(w0, w_max)
    
    ate = np.mean(w1 * Y) - np.mean(w0 * Y)
    
    # Bootstrap SE
    n = len(T)
    boot_ates = []
    for _ in range(200):
        idx = np.random.choice(n, n, replace=True)
        w1b = T[idx] / np.clip(ps_trimmed[idx], 0.01, 0.99)
        w0b = (1 - T[idx]) / (1 - np.clip(ps_trimmed[idx], 0.01, 0.99))
        boot_ates.append(np.mean(w1b * Y[idx]) - np.mean(w0b * Y[idx]))
    
    se = np.std(boot_ates)
    return {
        'ATE': ate,
        'SE': se,
        'CI_95': (ate - 1.96*se, ate + 1.96*se)
    }
```

---

## 9. Doubly Robust Estimators

**Augmented Inverse Probability Weighting (AIPW)** combines outcome modeling with propensity score weighting. It is **doubly robust**: consistent if EITHER the outcome model OR the propensity score model is correctly specified.

### AIPW Estimator

\[
\hat{\tau}_{\text{AIPW}} = \frac{1}{n}\sum_{i=1}^n \left[\underbrace{\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)}_{\text{outcome model}} + \underbrace{\frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}}_{\text{bias correction}}\right]
\]

where \(\hat{\mu}_t(x) = \hat{E}[Y \mid T=t, X=x]\).

```python
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def aipw_ate(X, T, Y, n_folds=5):
    """
    AIPW (Doubly Robust) ATE estimator using cross-fitting.
    Cross-fitting avoids overfitting in the nuisance models.
    """
    n = len(T)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)
    e_hat = np.zeros(n)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        T_tr, T_val = T[train_idx], T[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        
        # Fit propensity score model
        ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        ps_model.fit(X_tr, T_tr)
        e_hat[val_idx] = ps_model.predict_proba(X_val)[:, 1]
        
        # Fit outcome models
        mu1_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        mu0_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        
        mu1_model.fit(X_tr[T_tr==1], Y_tr[T_tr==1])
        mu0_model.fit(X_tr[T_tr==0], Y_tr[T_tr==0])
        
        mu1_hat[val_idx] = mu1_model.predict(X_val)
        mu0_hat[val_idx] = mu0_model.predict(X_val)
    
    # Clip propensity scores for stability
    e_hat = np.clip(e_hat, 0.05, 0.95)
    
    # AIPW formula
    psi = (mu1_hat - mu0_hat 
           + T * (Y - mu1_hat) / e_hat 
           - (1 - T) * (Y - mu0_hat) / (1 - e_hat))
    
    ate = psi.mean()
    se = psi.std() / np.sqrt(n)
    
    return {
        'ATE': ate,
        'SE': se,
        'CI_95': (ate - 1.96*se, ate + 1.96*se),
        'IF_values': psi  # influence function values for inference
    }
```

---

## 10. Difference-in-Differences

**DiD** exploits panel data (repeated measurements) to control for time-invariant confounding.

### Setup and Parallel Trends Assumption

| Group | Pre-treatment | Post-treatment |
|-------|--------------|----------------|
| Treatment | \(Y_{T,\text{pre}}\) | \(Y_{T,\text{post}}\) |
| Control | \(Y_{C,\text{pre}}\) | \(Y_{C,\text{post}}\) |

**DiD estimator**:
\[
\hat{\tau}_{\text{DiD}} = (Y_{T,\text{post}} - Y_{T,\text{pre}}) - (Y_{C,\text{post}} - Y_{C,\text{pre}})
\]

**Parallel trends assumption**: In the absence of treatment, both groups would have had the same trend over time. This is the critical identifying assumption.

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def did_analysis(df, outcome, treatment_group_col, post_col, 
                  cluster_col=None, covariates=None):
    """
    Difference-in-Differences analysis.
    
    Parameters:
    -----------
    df: panel dataframe
    outcome: outcome variable name
    treatment_group_col: 1 if unit in treatment group (not time-varying)
    post_col: 1 if post-treatment period
    cluster_col: column for clustered standard errors
    covariates: list of control variables
    
    Model: Y = α + β₁·Treated + β₂·Post + τ·(Treated×Post) + Xγ + ε
    τ is the DiD estimate.
    """
    # Create interaction term
    df = df.copy()
    df['did'] = df[treatment_group_col] * df[post_col]
    
    formula = f"{outcome} ~ {treatment_group_col} + {post_col} + did"
    if covariates:
        formula += " + " + " + ".join(covariates)
    
    if cluster_col:
        model = smf.ols(formula, data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df[cluster_col]}
        )
    else:
        model = smf.ols(formula, data=df).fit()
    
    did_coef = model.params['did']
    did_se = model.bse['did']
    did_pval = model.pvalues['did']
    did_ci = model.conf_int().loc['did']
    
    print(f"DiD Estimate (ATT): {did_coef:.4f}")
    print(f"Std Error: {did_se:.4f}")
    print(f"95% CI: ({did_ci[0]:.4f}, {did_ci[1]:.4f})")
    print(f"p-value: {did_pval:.4f}")
    
    return model, did_coef

# Simulate DiD example: job training program
np.random.seed(42)
n_units = 200
n_treated = 100

# True ATT = 500 (training increases wages by $500)
true_att = 500

pre_wages_control = np.random.normal(30000, 5000, n_units - n_treated)
pre_wages_treated = np.random.normal(28000, 5000, n_treated)  # Slightly lower baseline

# Parallel trends: both grow by $2000 without treatment
post_wages_control = pre_wages_control + 2000 + np.random.randn(n_units - n_treated) * 1000
post_wages_treated = pre_wages_treated + 2000 + true_att + np.random.randn(n_treated) * 1000

panel_df = pd.DataFrame({
    'unit_id': list(range(n_units)) * 2,
    'wage': np.concatenate([pre_wages_control, pre_wages_treated, 
                             post_wages_control, post_wages_treated]),
    'treated': np.concatenate([[0]*100 + [1]*100] * 2),
    'post': [0]*n_units + [1]*n_units,
    'period': ['pre']*n_units + ['post']*n_units
})

model, att = did_analysis(panel_df, 'wage', 'treated', 'post', cluster_col='unit_id')
print(f"\nTrue ATT: {true_att}, Estimated: {att:.2f}")
```

### Testing Parallel Trends

```python
def parallel_trends_test(df, outcome, treatment_group_col, 
                          time_col, treatment_period):
    """
    Test parallel trends using pre-treatment periods.
    Fit event-study regression with leads and lags.
    """
    # Only use pre-treatment data
    pre_df = df[df[time_col] < treatment_period].copy()
    
    # Create time dummies relative to treatment
    periods = sorted(pre_df[time_col].unique())
    base_period = periods[-1]  # Last pre-treatment period as base
    
    for t in periods[:-1]:
        pre_df[f'treated_t{t}'] = (pre_df[treatment_group_col] == 1) & (pre_df[time_col] == t)
    
    time_vars = [f'treated_t{t}' for t in periods[:-1]]
    formula = f"{outcome} ~ " + " + ".join(time_vars)
    
    model = smf.ols(formula, data=pre_df).fit()
    
    # Test joint significance of pre-treatment interactions
    from scipy.stats import chi2
    print("Pre-treatment trend test (should all be ~0 for parallel trends):")
    print(model.params[time_vars])
    print(f"\nJoint p-value: {model.f_pvalue:.4f}")
    
    return model
```

### Staggered DiD and Recent Advances

Classic DiD with staggered treatment timing (units treated at different times) has issues with heterogeneous treatment effects. Recent approaches:

- **Callaway & Sant'Anna (2021)**: Group-time average treatment effects
- **Sun & Abraham (2021)**: Interaction-weighted estimator
- **Roth et al. (2023)**: Sensitivity analysis for pre-trend violations

---

## 11. Regression Discontinuity Design

**RDD** exploits a threshold rule that determines treatment assignment. Units just above the cutoff are treated; units just below are not. Near the cutoff, assignment is "as good as random."

### Sharp RDD

\[
T_i = \mathbf{1}[X_i \geq c]
\]

The causal effect at the cutoff:
\[
\tau_{\text{RD}} = \lim_{x \downarrow c} E[Y \mid X=x] - \lim_{x \uparrow c} E[Y \mid X=x]
\]

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def sharp_rdd(X, Y, cutoff, bandwidth=None, kernel='triangular'):
    """
    Sharp RDD estimator using local linear regression.
    
    Parameters:
    -----------
    X: running variable (score/forcing variable)
    Y: outcome
    cutoff: threshold value
    bandwidth: window around cutoff (if None, use all data)
    kernel: 'triangular', 'uniform', or 'epanechnikov'
    """
    # Center running variable at cutoff
    X_c = X - cutoff
    T = (X_c >= 0).astype(float)
    
    # Apply bandwidth restriction
    if bandwidth is not None:
        in_band = np.abs(X_c) <= bandwidth
        X_c = X_c[in_band]
        T = T[in_band]
        Y = Y[in_band]
    
    # Kernel weights
    h = bandwidth if bandwidth else np.std(X_c)
    if kernel == 'triangular':
        weights = np.maximum(0, 1 - np.abs(X_c) / h)
    elif kernel == 'epanechnikov':
        weights = np.maximum(0, 3/4 * (1 - (X_c/h)**2))
    else:
        weights = np.ones(len(X_c))
    
    # Local linear regression: allow different slopes on each side
    # Features: [1, X_c, T, X_c*T]
    features = np.column_stack([np.ones(len(X_c)), X_c, T, X_c * T])
    
    # Weighted least squares
    W = np.diag(weights)
    XtWX = features.T @ W @ features
    XtWY = features.T @ W @ Y
    
    try:
        coefs = np.linalg.solve(XtWX, XtWY)
        tau_rdd = coefs[2]  # Coefficient on T = jump at cutoff
    except np.linalg.LinAlgError:
        print("Singular matrix; using np.linalg.lstsq")
        coefs, _, _, _ = np.linalg.lstsq(XtWX, XtWY, rcond=None)
        tau_rdd = coefs[2]
    
    return tau_rdd, coefs

# Simulate RDD: class size rule (Angrist & Lavy)
# Students in classes > 40 get a teaching aide; outcome = test score
np.random.seed(42)
n = 1000
class_size = np.random.uniform(20, 60, n)
cutoff = 40
teaching_aide = (class_size >= cutoff).astype(float)

# True effect = 5 points improvement
true_rdd = 5
test_score = (50 + 0.3 * class_size - true_rdd * teaching_aide 
              + np.random.randn(n) * 8)

tau, coefs = sharp_rdd(class_size, test_score, cutoff=cutoff, bandwidth=10)
print(f"True RDD effect: {true_rdd}")
print(f"Estimated RDD effect: {tau:.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
control = class_size < cutoff
treated = class_size >= cutoff

ax.scatter(class_size[control], test_score[control], alpha=0.3, color='blue', label='Control')
ax.scatter(class_size[treated], test_score[treated], alpha=0.3, color='red', label='Treated')
ax.axvline(x=cutoff, color='black', linestyle='--', label='Cutoff')
ax.set_xlabel('Class Size'); ax.set_ylabel('Test Score')
ax.set_title('Regression Discontinuity Design')
ax.legend()
plt.tight_layout()
plt.show()
```

### Fuzzy RDD

Treatment probability jumps at the cutoff but is not deterministic:
\[
P(T=1 \mid X \geq c) > P(T=1 \mid X < c)
\]

This becomes an IV problem: use the cutoff as an instrument for actual treatment.

```python
def fuzzy_rdd(X, T, Y, cutoff, bandwidth=5):
    """
    Fuzzy RDD: Local Wald estimator = reduced form / first stage.
    """
    X_c = X - cutoff
    in_band = np.abs(X_c) <= bandwidth
    
    Z = (X_c[in_band] >= 0).astype(float)  # Indicator above cutoff
    T_band = T[in_band]
    Y_band = Y[in_band]
    
    # First stage: Z → T
    # Reduced form: Z → Y
    first_stage = Y_band[Z==1].mean() - Y_band[Z==0].mean()  # Actually this is reduced form
    # Proper: regression
    import statsmodels.api as sm
    
    # First stage regression
    X_fs = sm.add_constant(np.column_stack([Z, X_c[in_band], Z * X_c[in_band]]))
    fs_model = sm.OLS(T_band, X_fs).fit()
    first_stage = fs_model.params[1]  # Coefficient on Z
    
    # Reduced form regression  
    rf_model = sm.OLS(Y_band, X_fs).fit()
    reduced_form = rf_model.params[1]
    
    # Local Wald = Reduced Form / First Stage
    tau_fuzzy = reduced_form / first_stage
    print(f"Fuzzy RDD: First stage={first_stage:.3f}, Reduced form={reduced_form:.3f}")
    print(f"LATE at cutoff: {tau_fuzzy:.3f}")
    
    return tau_fuzzy
```

### RDD Validity Checks

```python
def rdd_validity_checks(X, cutoff):
    """
    Check RDD assumptions:
    1. McCrary density test: No bunching at cutoff (no manipulation)
    2. Covariate continuity: Pre-treatment covariates continuous at cutoff
    """
    X_c = X - cutoff
    
    # Density test: check for discontinuity in density at cutoff
    # (People manipulating their score to just above cutoff)
    from scipy.stats import gaussian_kde
    
    bandwidth = 0.5 * np.std(X_c) * len(X_c)**(-0.2)  # Silverman rule
    
    kde = gaussian_kde(X_c, bw_method=bandwidth)
    x_eval = np.linspace(-3*np.std(X_c), 3*np.std(X_c), 1000)
    density = kde(x_eval)
    
    # Compare density just left and right of cutoff
    left_density = kde([-0.001])[0]
    right_density = kde([0.001])[0]
    
    print(f"Density left of cutoff: {left_density:.4f}")
    print(f"Density right of cutoff: {right_density:.4f}")
    print(f"Ratio: {right_density/left_density:.4f} (should be ~1)")
    
    plt.figure(figsize=(8, 4))
    plt.plot(x_eval, density, 'b-', linewidth=2)
    plt.axvline(0, color='red', linestyle='--', label='Cutoff')
    plt.xlabel('Running variable (centered)')
    plt.ylabel('Density')
    plt.title('McCrary Density Test')
    plt.legend()
    plt.show()
```

---

## 12. Instrumental Variables

**IV** handles unmeasured confounders using a variable Z (the instrument) that:
1. **Relevance**: Z affects T (first-stage relationship)
2. **Exclusion restriction**: Z affects Y only through T (no direct effect)
3. **Independence**: Z is independent of unmeasured confounders

### Two-Stage Least Squares (2SLS)

**Stage 1**: Regress T on Z (and covariates X):
\[
\hat{T} = \hat{\pi}_0 + \hat{\pi}_1 Z + \hat{\pi}_2 X
\]

**Stage 2**: Regress Y on \(\hat{T}\) (and covariates X):
\[
Y = \alpha + \tau \hat{T} + \gamma X + \epsilon
\]

The 2SLS estimand is the **Local Average Treatment Effect (LATE)**: effect for "compliers" — units whose treatment changes with the instrument.

```python
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

def two_stage_least_squares(Y, T, Z, X=None):
    """
    2SLS Instrumental Variables estimation.
    
    Parameters:
    -----------
    Y: outcome (n,)
    T: endogenous treatment (n,)
    Z: instrument (n,) or (n, k) for k instruments
    X: exogenous covariates (n, p), optional
    
    Returns: LATE estimate, SE, F-statistic for weak instrument test
    """
    n = len(Y)
    Z = np.atleast_2d(Z).T if Z.ndim == 1 else Z
    
    # Build regressor matrices
    if X is not None:
        W = np.column_stack([np.ones(n), X])  # Exogenous controls
        Z_full = np.column_stack([np.ones(n), Z, X])  # Instruments + controls
    else:
        W = np.ones((n, 1))
        Z_full = np.column_stack([np.ones(n), Z])
    
    # First stage: T ~ Z_full
    T_col = T.reshape(-1, 1)
    first_stage = sm.OLS(T, Z_full).fit()
    T_hat = first_stage.fittedvalues
    
    # First stage F-statistic (weak instrument test)
    # Rule of thumb: F > 10 means instrument is not weak
    f_stat = first_stage.fvalue
    print(f"First stage F-statistic: {f_stat:.2f}")
    if f_stat < 10:
        print("⚠ Weak instrument! F < 10. LATE estimates may be unreliable.")
    
    # Second stage: Y ~ T_hat + X
    if X is not None:
        second_stage_X = np.column_stack([np.ones(n), T_hat, X])
    else:
        second_stage_X = np.column_stack([np.ones(n), T_hat])
    
    second_stage = sm.OLS(Y, second_stage_X).fit()
    late = second_stage.params[1]
    
    # Proper 2SLS SE (not just OLS SE from second stage)
    # Use statsmodels IV2SLS for correct inference
    try:
        from linearmodels.iv import IV2SLS
        endog = Y
        exog = sm.add_constant(X) if X is not None else np.ones((n, 1))
        instruments = sm.add_constant(Z) if X is None else np.column_stack([sm.add_constant(Z), X])
        
        # Note: linearmodels syntax
        iv_model = IV2SLS(endog, exog, T_col, instruments).fit(cov_type='robust')
        print(iv_model.summary)
        return iv_model
    except ImportError:
        print("Install linearmodels for proper 2SLS SEs: pip install linearmodels")
        return {'LATE': late, 'F_stat': f_stat}

# Simulate IV: Drug compliance as instrument for blood pressure
np.random.seed(42)
n = 2000

# Unmeasured confounder: health habits
health_habits = np.random.randn(n)  # unobserved

# Instrument: random encouragement letter to take drug (RCT-style)
Z = np.random.binomial(1, 0.5, n)  # Random assignment

# Treatment: actual drug use (partial compliance, confounded by habits)
T_prob = 0.2 + 0.5 * Z + 0.2 * health_habits
T = np.random.binomial(1, np.clip(T_prob, 0, 1), n)

# Outcome: blood pressure
# True LATE (for compliers) = -10
Y = 140 - 10 * T - 5 * health_habits + np.random.randn(n) * 5

result = two_stage_least_squares(Y, T, Z)
```

### Weak Instruments

If the first-stage F-statistic is low, 2SLS is biased. Solutions:
- **LIML** (Limited Information Maximum Likelihood): more robust than 2SLS with weak instruments
- **Anderson-Rubin test**: Robust inference even with weak instruments
- **Jackknife IV (JIVE)**: Bias correction

---

## 13. Synthetic Control Method

The **Synthetic Control** method (Abadie et al., 2010) constructs a weighted combination of control units that best matches the pre-treatment trajectory of the treated unit.

```python
import numpy as np
from scipy.optimize import minimize

def synthetic_control(Y_treated, Y_donors, pre_periods):
    """
    Estimate synthetic control weights.
    
    Parameters:
    -----------
    Y_treated: (T_total,) outcome for treated unit
    Y_donors: (T_total, N_donors) outcomes for donor pool
    pre_periods: indices of pre-treatment periods
    
    Returns: weights w (N_donors,), synthetic control trajectory
    """
    Y_pre_treated = Y_treated[pre_periods]
    Y_pre_donors = Y_donors[pre_periods, :]
    n_donors = Y_donors.shape[1]
    
    def objective(w):
        """Sum of squared differences in pre-treatment period."""
        synth = Y_pre_donors @ w
        return np.sum((Y_pre_treated - synth) ** 2)
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    
    # Initial guess: uniform weights
    w0 = np.ones(n_donors) / n_donors
    
    result = minimize(objective, w0, method='SLSQP',
                     constraints=constraints, bounds=bounds,
                     options={'maxiter': 1000})
    
    weights = result.x
    synth_control = Y_donors @ weights
    
    # Effect: treated - synthetic in post-treatment
    post_periods = [t for t in range(len(Y_treated)) if t not in pre_periods]
    gaps = Y_treated[post_periods] - synth_control[post_periods]
    
    return weights, synth_control, gaps

# Example: California cigarette tax (Abadie, Diamond, Hainmueller 2010)
np.random.seed(42)
T = 30  # Time periods
pre = list(range(20))  # First 20 pre-treatment
post = list(range(20, 30))

# Simulate treated (California) and 10 donor states
n_donors = 10
Y_donors = np.random.randn(T, n_donors) * 5 + np.linspace(100, 80, T)[:, None]

# Treated unit follows donors pre-treatment, then drops
Y_treated = np.zeros(T)
for t in range(T):
    Y_treated[t] = Y_donors[t, :3].mean() + np.random.randn() * 2
Y_treated[20:] -= 15  # Policy reduces cigarette sales by 15

weights, synth, gaps = synthetic_control(Y_treated, Y_donors, pre)
print(f"Synthetic control weights: {weights.round(3)}")
print(f"Average post-treatment effect: {gaps.mean():.2f}")
print(f"True effect: -15")
```

---

## 14. Heterogeneous Treatment Effects and CATE

In practice, treatment effects vary across individuals. CATE quantifies this heterogeneity:

\[
\tau(x) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x]
\]

### Motivation

- **Precision medicine**: Treat only patients who benefit (CATE > 0)
- **Policy targeting**: Allocate limited interventions where most impactful
- **Marketing**: Focus spend on "persuadables" (positive uplift)
- **Fairness**: Check if treatment effects differ across demographic groups

### Evaluation of CATE Methods

Since we never observe both potential outcomes, evaluation is indirect:

1. **PEHE** (Precision in Estimation of Heterogeneous Effects): \(\sqrt{\mathbb{E}[(\hat{\tau}(x) - \tau(x))^2]}\) — only in simulations
2. **Qini/AUUC curve**: Rank by predicted CATE, check cumulative gain
3. **RATE** (Rank-Weighted Average Treatment Effect): Formal statistical test
4. **Calibration**: Does \(\hat{\tau}(x)\) for subgroup equal observed effect in subgroup?

---

## 15. Meta-Learners for CATE

Meta-learners use any ML model as a base learner to estimate CATE.

### S-Learner (Single Learner)

Fit one model with T as a feature. CATE = difference in predictions with T=1 vs T=0.

**Risk**: T may be regularized toward zero if it is weakly associated.

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import numpy as np

class SLearner:
    """Single model for CATE estimation."""
    
    def __init__(self, base_model=None):
        self.model = base_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05
        )
    
    def fit(self, X, T, Y):
        X_T = np.column_stack([X, T])
        self.model.fit(X_T, Y)
        return self
    
    def predict(self, X):
        n = len(X)
        X_t1 = np.column_stack([X, np.ones(n)])
        X_t0 = np.column_stack([X, np.zeros(n)])
        return self.model.predict(X_t1) - self.model.predict(X_t0)

class TLearner:
    """Two separate models for treated and control groups."""
    
    def __init__(self, base_model=None):
        self.model1 = (base_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=3))
        self.model0 = (base_model or GradientBoostingRegressor(
            n_estimators=200, max_depth=3))
    
    def fit(self, X, T, Y):
        self.model1.fit(X[T==1], Y[T==1])
        self.model0.fit(X[T==0], Y[T==0])
        return self
    
    def predict(self, X):
        return self.model1.predict(X) - self.model0.predict(X)

class XLearner:
    """
    X-Learner (Künzel et al., 2019): 
    Uses imputed treatment effects to correct T-learner estimates.
    Better with imbalanced treatment groups.
    """
    
    def __init__(self, base_model=None, propensity_model=None):
        self.mu1 = GradientBoostingRegressor(n_estimators=200, max_depth=3)
        self.mu0 = GradientBoostingRegressor(n_estimators=200, max_depth=3)
        self.tau1 = GradientBoostingRegressor(n_estimators=200, max_depth=3)
        self.tau0 = GradientBoostingRegressor(n_estimators=200, max_depth=3)
        from sklearn.linear_model import LogisticRegression
        self.ps_model = propensity_model or LogisticRegression(max_iter=1000)
    
    def fit(self, X, T, Y):
        # Stage 1: Estimate outcome models
        self.mu1.fit(X[T==1], Y[T==1])
        self.mu0.fit(X[T==0], Y[T==0])
        
        # Stage 2: Imputed treatment effects
        D1 = Y[T==1] - self.mu0.predict(X[T==1])  # ITE for treated
        D0 = self.mu1.predict(X[T==0]) - Y[T==0]  # ITE for control
        
        # Stage 3: Model imputed effects
        self.tau1.fit(X[T==1], D1)
        self.tau0.fit(X[T==0], D0)
        
        # Propensity score for combining
        self.ps_model.fit(X, T)
        return self
    
    def predict(self, X):
        e = self.ps_model.predict_proba(X)[:, 1]
        tau1_pred = self.tau1.predict(X)
        tau0_pred = self.tau0.predict(X)
        # Propensity-weighted combination
        return e * tau0_pred + (1 - e) * tau1_pred

class RLearner:
    """
    R-Learner (Nie & Wager, 2021): 
    Based on Robinson decomposition. Handles confounding well.
    τ(x) = argmin_τ E[(Y - m(X) - τ(X)(T - e(X)))²]
    """
    
    def __init__(self, outcome_model=None, ps_model=None, tau_model=None):
        self.m_model = outcome_model or GradientBoostingRegressor(n_estimators=200)
        self.e_model = ps_model or GradientBoostingRegressor(n_estimators=200)
        self.tau_model = tau_model or GradientBoostingRegressor(n_estimators=200)
    
    def fit(self, X, T, Y, n_folds=5):
        from sklearn.model_selection import KFold
        
        n = len(Y)
        m_hat = np.zeros(n)
        e_hat = np.zeros(n)
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Cross-fit nuisance models
        for train_idx, val_idx in kf.split(X):
            self.m_model.fit(X[train_idx], Y[train_idx])
            self.e_model.fit(X[train_idx], T[train_idx])
            m_hat[val_idx] = self.m_model.predict(X[val_idx])
            e_hat[val_idx] = self.e_model.predict(X[val_idx])
        
        # Residuals
        Y_res = Y - m_hat
        T_res = T - np.clip(e_hat, 0.05, 0.95)
        
        # Pseudo-outcome: weighted regression
        pseudo_outcome = Y_res / T_res
        weights = T_res ** 2
        
        # Fit CATE model
        self.tau_model.fit(X, pseudo_outcome, sample_weight=weights)
        return self
    
    def predict(self, X):
        return self.tau_model.predict(X)

# Comprehensive comparison
def compare_metalearners(n=5000):
    """Compare meta-learners on synthetic data with known CATE."""
    np.random.seed(42)
    
    X = np.random.randn(n, 5)
    # True CATE: heterogeneous
    true_cate = 2 + X[:, 0] * 1.5 - X[:, 1] * 0.5
    
    # Propensity score (confounding)
    ps = 1 / (1 + np.exp(-X[:, 0]))
    T = np.random.binomial(1, ps, n)
    
    # Outcomes
    Y0 = X[:, 0] + X[:, 1] ** 2 + np.random.randn(n)
    Y1 = Y0 + true_cate
    Y = T * Y1 + (1 - T) * Y0
    
    # Split
    n_train = int(0.8 * n)
    X_tr, X_te = X[:n_train], X[n_train:]
    T_tr, T_te = T[:n_train], T[n_train:]
    Y_tr, Y_te = Y[:n_train], Y[n_train:]
    true_cate_te = true_cate[n_train:]
    
    learners = {
        'S-Learner': SLearner(),
        'T-Learner': TLearner(),
        'X-Learner': XLearner(),
        'R-Learner': RLearner()
    }
    
    print("=== Meta-Learner CATE Comparison ===")
    print(f"{'Method':<15} {'PEHE':<12} {'Corr':<12}")
    print("-" * 40)
    
    for name, learner in learners.items():
        learner.fit(X_tr, T_tr, Y_tr)
        cate_hat = learner.predict(X_te)
        pehe = np.sqrt(np.mean((cate_hat - true_cate_te)**2))
        corr = np.corrcoef(cate_hat, true_cate_te)[0,1]
        print(f"{name:<15} {pehe:<12.4f} {corr:<12.4f}")

compare_metalearners()
```

---

## 16. Causal Forests and Double ML

### Generalized Random Forests (GRF) / Causal Forest

Wager & Athey (2018) developed causal forests: an adaptation of random forests that estimates CATE by using honesty (separate data for splitting and estimation) and targeting variance of treatment effects rather than outcomes.

```python
# pip install econml
from econml.grf import CausalForest
from econml.dml import CausalForestDML, LinearDML

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

def causal_forest_example(n=2000):
    np.random.seed(42)
    
    X = np.random.randn(n, 5)
    ps = 1 / (1 + np.exp(-0.5 * X[:, 0]))
    T = np.random.binomial(1, ps, n)
    true_cate = 1 + X[:, 0] * 2 - X[:, 1]
    Y = true_cate * T + X[:, 0] + X[:, 2] ** 2 + np.random.randn(n) * 0.5
    
    n_train = int(0.8 * n)
    X_tr, X_te = X[:n_train], X[n_train:]
    T_tr, T_te = T[:n_train], T[n_train:]
    Y_tr, Y_te = Y[:n_train], Y[n_train:]
    
    # Method 1: Pure Causal Forest
    cf = CausalForest(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=5,
        max_samples=0.5,
        inference=True  # Enables confidence intervals
    )
    cf.fit(X_tr, T_tr, Y_tr)
    
    cate_cf, lb_cf, ub_cf = cf.predict(X_te, interval=True, alpha=0.1)
    
    # Method 2: Causal Forest DML (residualize first)
    cfdml = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
        n_estimators=1000,
        cv=5,
        inference=True
    )
    cfdml.fit(Y_tr, T_tr, X=X_tr)
    
    cate_dml = cfdml.effect(X_te)
    ci_dml = cfdml.effect_interval(X_te, alpha=0.1)
    
    true_cate_te = true_cate[n_train:]
    
    pehe_cf = np.sqrt(np.mean((cate_cf - true_cate_te)**2))
    pehe_dml = np.sqrt(np.mean((cate_dml - true_cate_te)**2))
    
    print(f"Causal Forest PEHE: {pehe_cf:.4f}")
    print(f"CausalForest DML PEHE: {pehe_dml:.4f}")
    
    # ATE with confidence intervals
    ate, ate_ci = cfdml.ate(X_te), cfdml.ate_interval(X_te, alpha=0.05)
    print(f"ATE: {ate:.4f}, 95% CI: ({ate_ci[0]:.4f}, {ate_ci[1]:.4f})")
    
    return cf, cfdml

### Double Machine Learning (DML)

Double/debiased ML (Chernozhukov et al., 2018) is a general framework for **orthogonalization**: it removes the effect of confounders X on both treatment T and outcome Y using ML, then estimates the causal effect from the residuals. The method is **root-n consistent** and **approximately unbiased** even when the nuisance models (m, e) are fitted with flexible ML (random forests, neural nets) — as long as they converge at rate \(o(n^{-1/4})\).

**Partially linear model**:
\[
Y = \tau \cdot T + m(X) + \varepsilon, \quad T = e(X) + \eta
\]

where \(m(X) = E[Y|X]\), \(e(X) = E[T|X]\) (propensity score for binary T), and \(\varepsilon, \eta\) are orthogonal to X.

**Key insight — Orthogonalization**: If we knew \(m(X)\) and \(e(X)\), we could compute residuals \(\tilde{Y} = Y - m(X)\) and \(\tilde{T} = T - e(X)\). The causal effect \(\tau\) would satisfy \(E[\tilde{Y} - \tau \tilde{T}] = 0\), giving:
\[
\hat{\tau} = \frac{E[\tilde{Y} \cdot \tilde{T}]}{E[\tilde{T}^2]}
\]

**Cross-fitting** (splitting data, fitting nuisance on one fold and predicting on another) avoids overfitting bias from using the same sample for both nuisance estimation and final regression. This "double" use of data (hence "double ML") ensures the bias from regularized ML is negligible.

```python
from econml.dml import LinearDML, NonParamDML

def double_ml_example(X, T, Y):
    """
    Double Machine Learning for partially linear model.
    """
    # LinearDML: linear CATE in features
    linear_dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=200, max_depth=3),
        model_t=GradientBoostingClassifier(n_estimators=200, max_depth=3),
        cv=5,
        linear_first_stages=False
    )
    linear_dml.fit(Y, T, X=X)
    
    print("=== Linear DML ===")
    print(linear_dml.summary(feat_names=[f"X{i}" for i in range(X.shape[1])]))
    
    # CATE for a specific subgroup
    high_x0 = X[X[:, 0] > 1]
    if len(high_x0) > 0:
        cate_high = linear_dml.effect(high_x0).mean()
        print(f"\nCATÉ for X₀ > 1: {cate_high:.4f}")
    
    return linear_dml
```

---

## 17. Causal Discovery

Causal discovery algorithms learn DAG structure from observational data — a challenging task because many DAGs imply the same conditional independences (Markov equivalence). Assumptions (e.g., faithfulness, non-Gaussianity, linearity) allow identification.

### Conceptual Overview

| Algorithm | Assumptions | Output | Complexity |
|-----------|-------------|--------|------------|
| **PC** | Faithfulness, causal sufficiency | Markov equiv. class (CPDAG) | Exponential in max degree |
| **GES** | Score-based (BIC) | Greedy DAG search | \(O(n^2)\) per step |
| **LiNGAM** | Linear, non-Gaussian errors | Unique DAG | Polynomial |
| **NOTEARS** | Linear (or additive) | DAG via continuous opt. | \(O(n^3)\) |

**Faithfulness**: The true DAG implies certain conditional independences; faithfulness says the reverse — no extra independences beyond those implied by the graph. Violations occur with parameter cancellations (e.g., two paths with equal magnitude cancel).

**Causal sufficiency**: No unmeasured common causes. Violations require FCI (Fast Causal Inference), which outputs a PAG (Partial Ancestral Graph) with bidirected edges for latent confounders.

### PC Algorithm (Constraint-Based)

Uses conditional independence tests to determine edges and orientations. Starts with a complete graph; removes edges for which independence holds given some subset of neighbors. Orients colliders (A→C←B) when conditioning on C creates dependence.

```python
# pip install causal-learn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

def pc_algorithm_example():
    """Run PC algorithm on simulated data."""
    np.random.seed(42)
    n = 500
    
    # True DAG: X1 → X2 → X4, X1 → X3 → X4, X2 → X3
    X1 = np.random.randn(n)
    X2 = 0.8 * X1 + np.random.randn(n) * 0.5
    X3 = 0.6 * X1 + 0.5 * X2 + np.random.randn(n) * 0.5
    X4 = 0.7 * X2 + 0.9 * X3 + np.random.randn(n) * 0.5
    
    data = np.column_stack([X1, X2, X3, X4])
    
    # Run PC algorithm
    cg = pc(data, alpha=0.05, indep_test=fisherz)
    print("PC Algorithm Graph:")
    print(cg.G)  # Shows adjacency matrix
    return cg
```

### LiNGAM (Linear Non-Gaussian Acyclic Model)

Shimizu et al. (2006): Uses non-Gaussianity of errors to identify full causal DAG.

```python
# pip install lingam
import lingam

def lingam_example(n=1000):
    """Run LiNGAM causal discovery."""
    np.random.seed(42)
    
    # True DAG: X0 → X1 → X2
    X0 = np.random.laplace(0, 1, n)  # Non-Gaussian errors
    X1 = 0.9 * X0 + np.random.laplace(0, 0.5, n)
    X2 = -0.7 * X1 + np.random.laplace(0, 0.5, n)
    
    data = np.column_stack([X0, X1, X2])
    
    model = lingam.DirectLiNGAM()
    model.fit(data)
    
    print("Adjacency Matrix (causal_order):")
    print(model.adjacency_matrix_)
    print(f"Causal order: {model.causal_order_}")
    
    return model
```

### NOTEARS (Differentiable DAG Learning)

Zheng et al. (2018): Reformulates DAG learning as continuous optimization.

\[
\min_{W} \frac{1}{n}||X - XW^T||_F^2 + \lambda||W||_1
\quad \text{s.t.} \quad h(W) = 0
\]

where \(h(W) = \text{tr}(e^{W \circ W}) - d = 0\) is a smooth DAG constraint.

```python
# pip install causalnex
from causalnex.structure import DAGRegressor

def notears_example(data):
    """Run NOTEARS for DAG discovery."""
    from causalnex.structure.notears import from_numpy
    
    # data: (n, d) numpy array
    sm = from_numpy(data, tabu_edges=None, w_threshold=0.8)
    print("Learned DAG edges:")
    for u, v, w in sm.edges.data('weight'):
        print(f"  {u} → {v}: weight = {w:.3f}")
    return sm
```

---

## 18. Uplift Modeling

**Uplift** = CATE for binary outcomes. The goal is to identify "persuadables" — units that would take the desired action ONLY because of the treatment.

### Four Customer Types

| Response without T | Response with T | Type |
|-------------------|----------------|------|
| Buy | Buy | **Sure thing** (treat wasteful) |
| Don't Buy | Buy | **Persuadable** (treat!) |
| Buy | Don't Buy | **Sleeping dog** (treatment harmful) |
| Don't Buy | Don't Buy | **Lost cause** (treat wasteful) |

Uplift = P(Buy\|T=1, X) - P(Buy\|T=0, X) → target only when Uplift > 0

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class UpliftTwoModel:
    """Two-model uplift estimator (T-learner for binary outcomes)."""
    
    def __init__(self):
        self.model_t1 = GradientBoostingClassifier(n_estimators=200, max_depth=3)
        self.model_t0 = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    
    def fit(self, X, T, Y):
        self.model_t1.fit(X[T==1], Y[T==1])
        self.model_t0.fit(X[T==0], Y[T==0])
        return self
    
    def predict(self, X):
        p1 = self.model_t1.predict_proba(X)[:, 1]
        p0 = self.model_t0.predict_proba(X)[:, 1]
        return p1 - p0  # Uplift

def qini_curve(Y, T, uplift_pred):
    """
    Qini curve: measures cumulative uplift vs random targeting.
    Higher AUC (area under Qini) = better uplift model.
    """
    df = pd.DataFrame({'y': Y, 't': T, 'uplift': uplift_pred})
    df = df.sort_values('uplift', ascending=False).reset_index(drop=True)
    
    n = len(df)
    n_treated_total = df['t'].sum()
    n_control_total = (1 - df['t']).sum()
    
    # Cumulative uplift
    qini_values = []
    for k in range(1, n + 1):
        top_k = df.iloc[:k]
        n_t = top_k['t'].sum()
        n_c = k - n_t
        
        if n_t > 0 and n_c > 0:
            qini = (top_k[top_k['t']==1]['y'].sum() / n_t - 
                    top_k[top_k['t']==0]['y'].sum() / n_c) * n_t / n_treated_total
        else:
            qini = 0
        qini_values.append(qini)
    
    return np.array(qini_values)

# Simulate uplift modeling for marketing
np.random.seed(42)
n = 5000

age = np.random.uniform(20, 60, n)
income = np.random.exponential(50, n)
recency = np.random.exponential(30, n)

# True uplift varies by segment
true_uplift = (0.1 * (age > 35) + 0.15 * (income > 60) - 0.05 * (recency > 60))

T = np.random.binomial(1, 0.5, n)  # RCT
p_buy_ctrl = 0.1 + 0.001 * age + 0.0005 * income
p_buy_trt = np.clip(p_buy_ctrl + true_uplift, 0, 1)
Y = np.where(T == 1, 
             np.random.binomial(1, p_buy_trt, n),
             np.random.binomial(1, p_buy_ctrl, n))

X = np.column_stack([age, income, recency])

X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
    X, T, Y, test_size=0.3, random_state=42
)

# Fit uplift model
uplift_model = UpliftTwoModel()
uplift_model.fit(X_tr, T_tr, Y_tr)
uplift_pred = uplift_model.predict(X_te)

# Qini curve
qini = qini_curve(Y_te, T_te, uplift_pred)
print(f"Qini AUC: {qini.cumsum()[-1] / len(qini):.4f}")
print(f"Mean predicted uplift: {uplift_pred.mean():.4f}")
print(f"Fraction with positive uplift: {(uplift_pred > 0).mean():.2%}")
```

---

## 19. Causal Inference Libraries

### DoWhy

End-to-end causal inference with explicit modeling of assumptions.

```python
# pip install dowhy
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

def dowhy_example(df, treatment, outcome, common_causes, instruments=None):
    """
    DoWhy four-step causal inference workflow:
    1. Model: encode causal assumptions as DAG
    2. Identify: find identification strategy
    3. Estimate: compute causal effect
    4. Refute: test robustness of assumptions
    """
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        instruments=instruments
    )
    
    print("=== Step 1: Causal Graph ===")
    model.view_model()
    
    print("\n=== Step 2: Identification ===")
    identified_estimand = model.identify_effect()
    print(identified_estimand)
    
    print("\n=== Step 3: Estimation ===")
    # Try multiple methods
    methods = [
        ("backdoor.linear_regression", {}),
        ("backdoor.propensity_score_weighting", {}),
        ("backdoor.propensity_score_matching", {}),
    ]
    
    estimates = {}
    for method, kwargs in methods:
        try:
            est = model.estimate_effect(
                identified_estimand, 
                method_name=method,
                **kwargs
            )
            estimates[method] = est.value
            print(f"{method}: {est.value:.4f}")
        except Exception as e:
            print(f"{method}: failed - {e}")
    
    print("\n=== Step 4: Refutation Tests ===")
    # Random common cause: adding random confounder shouldn't change estimate
    refute_random = model.refute_estimate(
        identified_estimand, 
        list(estimates.values())[0],
        method_name="random_common_cause"
    )
    print(refute_random)
    
    # Placebo treatment: replacing treatment with random should give ~0 effect
    refute_placebo = model.refute_estimate(
        identified_estimand,
        list(estimates.values())[0],
        method_name="placebo_treatment_refuter"
    )
    print(refute_placebo)
    
    return estimates
```

### EconML

Microsoft's library for heterogeneous treatment effects.

```python
# pip install econml
from econml.dml import CausalForestDML, LinearDML
from econml.metalearners import XLearner, SLearner, TLearner
from econml.dr import DRLearner

def econml_comprehensive_example(X, T, Y):
    """Comprehensive EconML usage."""
    from sklearn.ensemble import GradientBoostingRegressor as GBR, GradientBoostingClassifier as GBC
    
    n_train = int(0.8 * len(X))
    X_tr, X_te = X[:n_train], X[n_train:]
    T_tr, T_te = T[:n_train], T[n_train:]
    Y_tr, Y_te = Y[:n_train], Y[n_train:]
    
    # Doubly Robust Learner
    dr = DRLearner(
        model_propensity=GBC(n_estimators=100, max_depth=3),
        model_regression=GBR(n_estimators=100, max_depth=3),
        model_final=GBR(n_estimators=100, max_depth=3),
        cv=5
    )
    dr.fit(Y_tr, T_tr, X=X_tr)
    
    # X-Learner
    xl = XLearner(models=[GBR(n_estimators=100)] * 2)
    xl.fit(Y_tr, T_tr, X=X_tr)
    
    # Predictions
    cate_dr = dr.effect(X_te)
    cate_xl = xl.effect(X_te)
    
    print(f"DR CATE: mean={cate_dr.mean():.4f}")
    print(f"XL CATE: mean={cate_xl.mean():.4f}")
    
    # Confidence intervals from DR
    ci_dr = dr.effect_interval(X_te, alpha=0.05)
    print(f"DR 95% CI for mean CATE: ({cate_dr.mean():.4f} ± {(ci_dr[1] - ci_dr[0]).mean()/4:.4f})")
    
    # Subgroup analysis
    # Find which features moderate the treatment effect
    dr.coef_  # For LinearDML variant
    
    return dr, xl
```

### CausalML (Uber)

```python
# pip install causalml
from causalml.inference.meta import LRSRegressor, XGBTRegressor
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.propensity import ElasticNetPropensityModel

def causalml_example(X, T, Y, X_test):
    """CausalML uplift and CATE estimation."""
    
    # Propensity score
    ps_model = ElasticNetPropensityModel()
    ps = ps_model.fit_predict(pd.DataFrame(X), pd.Series(T))
    
    # X-learner with XGBoost
    learner = XGBTRegressor(ate_alpha=0.05)
    learner.fit(X=X, treatment=T, y=Y, p=ps)
    cate = learner.predict(X=X_test)
    print(f"CATE (XGBoost XLearner): mean={cate.mean():.4f}")
    
    # Uplift forest
    uplift_rf = UpliftRandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=50,
        evaluationFunction='KL',
        control_name='control'
    )
    
    # Format for CausalML: treatment must include control label
    treatment_labels = np.where(T == 1, 'treatment', 'control')
    uplift_rf.fit(X=X, treatment=treatment_labels, y=Y)
    
    uplift_preds = uplift_rf.predict(X_test)
    print(f"Uplift Forest predictions: mean={uplift_preds.mean():.4f}")
    
    return cate, uplift_preds
```

---

## 20. Full End-to-End Code Examples

### Example 1: Difference-in-Differences for Policy Evaluation

```python
"""
Evaluating the effect of a minimum wage increase using DiD.
Scenario: State A raises minimum wage in Q3 2023; State B (control) does not.
Outcome: Employment rate.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def simulate_did_policy(n_units=500, n_periods=8, treatment_period=5, true_att=-0.03):
    """
    Simulate DiD panel data.
    true_att: minimum wage increases → 3% employment decrease
    """
    np.random.seed(42)
    
    units = range(n_units)
    periods = range(n_periods)
    
    # Assign treatment group
    treated = np.random.binomial(1, 0.5, n_units)
    
    # Fixed unit effects (size, industry)
    unit_fe = np.random.randn(n_units) * 0.05
    
    # Time effects (business cycle)
    time_fe = np.cumsum(np.random.randn(n_periods) * 0.01)
    
    records = []
    for i in units:
        for t in periods:
            post = int(t >= treatment_period)
            treatment = treated[i] * post
            
            emp_rate = (0.95 
                       + unit_fe[i] 
                       + time_fe[t] 
                       + true_att * treatment 
                       + np.random.randn() * 0.01)
            
            records.append({
                'unit': i,
                'period': t,
                'treated_group': treated[i],
                'post': post,
                'treatment': treatment,
                'employment': emp_rate
            })
    
    return pd.DataFrame(records)

# Run analysis
df = simulate_did_policy()

# Two-way fixed effects DiD (most robust)
model = smf.ols(
    'employment ~ treatment + C(unit) + C(period)', 
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['unit']})

print(f"True ATT: -0.03")
print(f"Estimated ATT: {model.params['treatment']:.4f}")
print(f"SE: {model.bse['treatment']:.4f}")
print(f"95% CI: ({model.conf_int().loc['treatment', 0]:.4f}, "
      f"{model.conf_int().loc['treatment', 1]:.4f})")

# Parallel trends visualization
pre_df = df[df['post'] == 0]
trend = pre_df.groupby(['period', 'treated_group'])['employment'].mean().unstack()

plt.figure(figsize=(10, 5))
plt.plot(trend.index, trend[0], 'b-o', label='Control group')
plt.plot(trend.index, trend[1], 'r-s', label='Treatment group')
plt.axvline(x=4.5, color='black', linestyle='--', label='Policy start')
plt.xlabel('Period')
plt.ylabel('Employment Rate')
plt.title('Difference-in-Differences: Parallel Trends Check')
plt.legend()
plt.tight_layout()
plt.show()
```

### Example 2: Propensity Score Matching for Observational Study

```python
"""
Evaluating effect of an online tutoring program on test scores.
Confounding: Students with higher baseline performance more likely to enroll.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def tutoring_program_study(n=3000):
    np.random.seed(42)
    
    # Student characteristics
    baseline_score = np.random.normal(70, 15, n)
    study_hours = np.random.exponential(10, n)
    income = np.random.exponential(60, n)  # household income ($K)
    
    # Confounded: higher-baseline students self-select into tutoring
    p_enroll = 1 / (1 + np.exp(-(0.03 * baseline_score + 0.02 * study_hours - 2)))
    enrolled = np.random.binomial(1, p_enroll, n)
    
    # True causal effect of tutoring: +8 points, more for lower-baseline
    true_effect = 8 - 0.05 * (baseline_score - 70)
    
    # Post-test score
    Y0 = baseline_score + 0.5 * study_hours + np.random.randn(n) * 10
    Y1 = Y0 + true_effect
    Y_obs = enrolled * Y1 + (1 - enrolled) * Y0
    
    true_att = true_effect[enrolled == 1].mean()
    
    df = pd.DataFrame({
        'baseline_score': baseline_score,
        'study_hours': study_hours,
        'income': income,
        'enrolled': enrolled,
        'final_score': Y_obs
    })
    
    print(f"Enrollment rate: {enrolled.mean():.2%}")
    print(f"True ATT: {true_att:.4f}")
    print(f"Naive estimate: {Y_obs[enrolled==1].mean() - Y_obs[enrolled==0].mean():.4f}")
    
    # Estimate propensity scores
    covariates = ['baseline_score', 'study_hours', 'income']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[covariates].values)
    T = df['enrolled'].values
    Y = df['final_score'].values
    
    # Fit logistic regression for PS
    ps_model = LogisticRegression(C=1.0, max_iter=1000)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    
    print(f"\nPS range: [{ps.min():.3f}, {ps.max():.3f}]")
    
    # Check overlap
    ps_treated = ps[T == 1]
    ps_control = ps[T == 0]
    overlap = (ps_treated.min() < ps_control.max() and 
               ps_control.min() < ps_treated.max())
    print(f"Overlap: {overlap}")
    
    # 1-to-1 nearest neighbor matching without replacement
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(ps[control_idx].reshape(-1, 1))
    dists, match_idx = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    matched_control = control_idx[match_idx.flatten()]
    
    # Caliper: discard matches with PS distance > 0.1
    caliper = 0.1 * ps.std()
    good_matches = dists.flatten() < caliper
    
    treated_matched = treated_idx[good_matches]
    control_matched = matched_control[good_matches]
    
    att_psm = (Y[treated_matched] - Y[control_matched]).mean()
    att_psm_se = (Y[treated_matched] - Y[control_matched]).std() / np.sqrt(good_matches.sum())
    
    print(f"\nPSM ATT: {att_psm:.4f}")
    print(f"PSM SE: {att_psm_se:.4f}")
    print(f"95% CI: ({att_psm - 1.96*att_psm_se:.4f}, {att_psm + 1.96*att_psm_se:.4f})")
    print(f"Matched sample size: {good_matches.sum()} treated, {good_matches.sum()} control")
    print(f"True ATT: {true_att:.4f}")
    
    # Covariate balance check
    print("\nCovariate Balance (|SMD|):")
    for cov in covariates:
        vals = df[cov].values
        smd_before = abs(vals[T==1].mean() - vals[T==0].mean()) / vals.std()
        smd_after = abs(vals[treated_matched].mean() - vals[control_matched].mean()) / vals.std()
        print(f"  {cov:<20}: Before={smd_before:.3f}, After={smd_after:.3f}")
    
    return df, ps, att_psm

df, ps, att = tutoring_program_study()
```

### Example 3: Causal Forest for Personalized Marketing

```python
"""
Identify which customers have the highest uplift from email campaign.
Use Causal Forest (EconML) to estimate CATE and rank customers for targeting.
"""

import numpy as np
import pandas as pd
from econml.grf import CausalForest
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def marketing_causal_forest(n=10000):
    np.random.seed(42)
    
    # Customer features
    age = np.random.uniform(18, 70, n)
    tenure = np.random.exponential(24, n)  # months
    monthly_spend = np.random.exponential(200, n)
    n_purchases = np.random.poisson(5, n)
    recency = np.random.exponential(30, n)  # days since last purchase
    
    X = np.column_stack([age, tenure, monthly_spend, n_purchases, recency])
    feature_names = ['age', 'tenure', 'monthly_spend', 'n_purchases', 'recency']
    
    # True heterogeneous CATE
    # Young customers with high spend benefit most from email
    true_cate = (0.05 
                 + 0.002 * (30 - np.minimum(age, 30))  # younger is better
                 + 0.0005 * monthly_spend 
                 - 0.001 * recency)
    true_cate = np.clip(true_cate, -0.05, 0.3)
    
    # Confounded treatment: high-spenders more likely targeted historically
    p_email = 1 / (1 + np.exp(-(0.003 * monthly_spend - 0.6)))
    T = np.random.binomial(1, p_email, n)
    
    # Binary outcome: purchase in next 30 days
    p_purchase_ctrl = 1 / (1 + np.exp(-(0.002 * monthly_spend - 0.003 * recency - 1)))
    p_purchase_trt = np.clip(p_purchase_ctrl + true_cate, 0, 1)
    Y = np.where(T == 1, 
                 np.random.binomial(1, p_purchase_trt, n),
                 np.random.binomial(1, p_purchase_ctrl, n)).astype(float)
    
    X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
        X, T, Y, test_size=0.3, random_state=42
    )
    true_cate_te = true_cate[int(0.7*n):]
    
    # Fit Causal Forest DML
    cfdml = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3),
        n_estimators=500,
        max_depth=5,
        cv=5,
        random_state=42,
        inference=True
    )
    cfdml.fit(Y_tr, T_tr, X=X_tr)
    
    # Predict CATE with confidence intervals
    cate_hat = cfdml.effect(X_te)
    lb, ub = cfdml.effect_interval(X_te, alpha=0.05)
    
    # Evaluate
    pehe = np.sqrt(np.mean((cate_hat - true_cate_te)**2))
    corr = np.corrcoef(cate_hat, true_cate_te)[0, 1]
    
    print(f"Causal Forest PEHE: {pehe:.4f}")
    print(f"Correlation with true CATE: {corr:.4f}")
    print(f"Mean CATE: {cate_hat.mean():.4f} (true: {true_cate_te.mean():.4f})")
    print(f"Fraction positive CATE: {(cate_hat > 0).mean():.2%}")
    
    # Marketing targeting strategy
    results_df = pd.DataFrame(X_te, columns=feature_names)
    results_df['predicted_uplift'] = cate_hat
    results_df['uplift_lb'] = lb
    results_df['uplift_ub'] = ub
    results_df['actual_treatment'] = T_te
    results_df['purchased'] = Y_te
    results_df['true_cate'] = true_cate_te
    
    # Top N targeting
    top_20pct = results_df.nlargest(int(0.2 * len(results_df)), 'predicted_uplift')
    print(f"\nTop 20% by predicted uplift:")
    print(f"  Mean predicted uplift: {top_20pct['predicted_uplift'].mean():.4f}")
    print(f"  Mean true CATE: {top_20pct['true_cate'].mean():.4f}")
    print(f"  Mean age: {top_20pct['age'].mean():.1f}")
    print(f"  Mean monthly spend: ${top_20pct['monthly_spend'].mean():.1f}")
    
    # Feature importance for treatment effect heterogeneity
    feat_importance = cfdml.feature_importances_
    print("\nFeature importance for CATE heterogeneity:")
    for feat, imp in sorted(zip(feature_names, feat_importance), 
                            key=lambda x: -x[1]):
        print(f"  {feat:<20}: {imp:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(true_cate_te, cate_hat, alpha=0.1, s=5)
    axes[0].plot([true_cate_te.min(), true_cate_te.max()],
                 [true_cate_te.min(), true_cate_te.max()], 'r--')
    axes[0].set_xlabel('True CATE')
    axes[0].set_ylabel('Predicted CATE')
    axes[0].set_title(f'Causal Forest CATE Predictions (ρ={corr:.3f})')
    
    axes[1].hist(cate_hat, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='Zero effect')
    axes[1].set_xlabel('Predicted CATE')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Predicted Treatment Effects')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return cfdml, results_df

model, results = marketing_causal_forest()
```

---

## Summary and Best Practices

### Choosing the Right Method

| Situation | Recommended Method |
|-----------|-------------------|
| RCT available | Simple t-test / regression |
| Observational, measured confounders | IPW, AIPW, regression adjustment |
| Before/after with control group | Difference-in-Differences |
| Threshold-based treatment | Regression Discontinuity |
| Unobserved confounding + valid instrument | Instrumental Variables |
| Few treatment units | Synthetic Control |
| Want heterogeneous effects | Causal Forest, Meta-Learners |
| Want to find persuadables | Uplift Modeling |
| Want to learn causal structure | PC, LiNGAM, NOTEARS |

### The Causal Inference Workflow

1. **Draw the DAG**: Before any analysis, write down your causal assumptions
2. **Identify the estimand**: ATE, ATT, LATE, or CATE?
3. **Choose identification strategy**: What assumptions are you willing to make?
4. **Check overlap**: Is there sufficient propensity score support?
5. **Estimate with multiple methods**: If results agree, more credible
6. **Sensitivity analysis**: How much unmeasured confounding would flip your conclusion?
7. **Report assumptions explicitly**: All causal claims rest on untestable assumptions

### Key Libraries Reference

| Library | Strength | Install |
|---------|----------|---------|
| **DoWhy** | Explicit causal model + refutation tests | `pip install dowhy` |
| **EconML** | Heterogeneous effects (CausalForest, DML) | `pip install econml` |
| **CausalML** | Uplift modeling + meta-learners | `pip install causalml` |
| **causal-learn** | Causal discovery (PC, FCI, GES) | `pip install causal-learn` |
| **lingam** | LiNGAM causal discovery | `pip install lingam` |
| **causalnex** | NOTEARS + Bayesian networks | `pip install causalnex` |
| **linearmodels** | IV/2SLS with correct SE | `pip install linearmodels` |
| **rpy2 + grf** | Generalized Random Forests | `pip install rpy2` |

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Chernozhukov, V., et al. (2018). *Double/Debiased Machine Learning for Treatment and Causal Parameters*. Econometrica.
- Wager, S., & Athey, S. (2018). *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*. JASA.
- Spirtes, P., et al. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press (PC algorithm).
- Shimizu, S., et al. (2006). *A Linear Non-Gaussian Acyclic Model for Causal Discovery*. JMLR.
- Zheng, X., et al. (2018). *DAGs with NO TEARS: Continuous Optimization for Structure Learning*. NeurIPS.

### Common Pitfalls

1. **Selecting on colliders**: Conditioning on post-treatment variables creates bias. Example: studying only hospitalized patients (hospitalization is caused by both severity and treatment).

2. **Weak overlap**: Extrapolating outside common support. When some covariate values appear only in treated or only in control, weights explode and estimates are unstable. Check propensity score histograms; trim or use overlap weights.

3. **Model misspecification**: Nonlinear confounding not captured by OLS. Use flexible ML (GBM, RF) for outcome and propensity in AIPW/DML rather than linear models.

4. **Ignoring SUTVA**: Network effects, spillovers. If treating one unit affects others' outcomes, standard methods underestimate or overestimate effects. Consider cluster randomization or spillover models.

5. **Fishing for significance**: Testing multiple endpoints without correction. Pre-register analysis plan; use Bonferroni or FDR for multiple comparisons.

6. **Conflating ATE and ATT**: These answer different questions. ATT = effect on the treated (relevant for scaling up an existing program). ATE = effect if everyone were treated (for new interventions). Weight accordingly.

7. **Not doing sensitivity analysis**: Every causal claim needs robustness checks. E-value, Rosenbaum bounds, or tilting analysis quantify how much unmeasured confounding would need to exist to nullify the result.

8. **Causal discovery overreach**: Learned DAGs are only as valid as the assumptions (faithfulness, no latent confounders). Validate with domain experts; use as hypothesis generation, not definitive structure.
