# Model Interpretability & Explainability: Comprehensive Guide

## Table of Contents
1. [Why Interpretability Matters](#why-interpretability-matters)
2. [Taxonomy of Interpretability Methods](#taxonomy-of-interpretability-methods)
3. [Inherently Interpretable Models](#inherently-interpretable-models)
4. [Feature Importance Methods](#feature-importance-methods)
5. [Partial Dependence Plots (PDP)](#partial-dependence-plots-pdp)
6. [Individual Conditional Expectation (ICE)](#individual-conditional-expectation-ice)
7. [Accumulated Local Effects (ALE)](#accumulated-local-effects-ale)
8. [SHAP: SHapley Additive exPlanations](#shap-shapley-additive-explanations)
9. [LIME: Local Interpretable Model-Agnostic Explanations](#lime)
10. [Integrated Gradients](#integrated-gradients)
11. [Grad-CAM and Vision Attribution Methods](#grad-cam-and-vision-attribution)
12. [Attention Visualization for Transformers](#attention-visualization)
13. [Anchors: Rule-Based Explanations](#anchors)
14. [Counterfactual Explanations](#counterfactual-explanations)
15. [Concept-Based Explanations: TCAV](#concept-based-explanations-tcav)
16. [Fairness in ML](#fairness-in-ml)
17. [Fairness Toolkits](#fairness-toolkits)
18. [Complete Practical Examples](#complete-practical-examples)
19. [Best Practices](#best-practices)

---

## Why Interpretability Matters

### The Case for Explainable AI

Model interpretability — the ability to understand why a model makes a given prediction — has moved from academic curiosity to regulatory necessity. Several forces drive this:

#### 1. Trust and Adoption
Users and organizations will not deploy systems they do not understand. A doctor will not follow a treatment recommendation from a "black box" with no reasoning. A loan officer needs to justify rejections. Interpretability builds the **calibrated trust** required for real-world deployment.

#### 2. Debugging and Model Improvement
Models learn spurious correlations. A famous example: a COVID-19 X-ray classifier that learned to detect the hospital scanner brand (since different hospitals had different patient populations) rather than the disease. Without interpretability tools, this bug is invisible on aggregate accuracy metrics. SHAP and saliency maps can expose such failures.

#### 3. Regulatory Compliance

**GDPR (EU, 2018)** — Article 22 grants individuals the "right to explanation" for automated decisions. This effectively mandates explainability for credit scoring, hiring, and other consequential AI systems in Europe.

**EU AI Act (2024)** — Classifies high-risk AI systems (healthcare, law enforcement, credit) and mandates transparency, human oversight, and audit trails.

**Fair Credit Reporting Act (FCRA, USA)** — Requires adverse action notices explaining credit denials.

**Equal Credit Opportunity Act (ECOA, USA)** — Mandates specific reasons for adverse actions.

#### 4. Fairness and Bias Detection
Protected attributes (race, gender, religion) may be encoded in proxy features. Interpretability tools reveal which features drive decisions, enabling auditing for disparate impact.

#### 5. Scientific Discovery
In drug discovery and genomics, the model's learned relationships are themselves the research output. Understanding which molecular features predict binding affinity advances the science.

---

## Taxonomy of Interpretability Methods

### Dimension 1: Scope — Global vs. Local

| Scope | Question Answered | Example Method |
|-------|------------------|----------------|
| **Global** | How does the model work overall? | Feature importance, PDP |
| **Local** | Why did the model predict X for this instance? | SHAP waterfall, LIME, Grad-CAM |

### Dimension 2: Model Relationship — Intrinsic vs. Post-hoc

| Type | Description | Example |
|------|-------------|---------|
| **Intrinsic** | Model is itself interpretable | Linear regression, decision tree, GAM |
| **Post-hoc** | Separate analysis after training | SHAP, LIME, Integrated Gradients |

### Dimension 3: Model Scope — Agnostic vs. Specific

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| **Model-agnostic** | Works with any model | Flexible, consistent API | May be approximate |
| **Model-specific** | Exploits model structure | Exact, efficient | Limited applicability |

### Choosing the Right Method

```
Is the model inherently interpretable?
  Yes → Use model directly (coefficients, tree structure)
  No → Is the question global or local?
    Global → PDP, ALE, permutation importance, global SHAP
    Local → SHAP waterfall/force, LIME, Integrated Gradients
         → Is the model a neural network for images?
              Yes → Grad-CAM, Integrated Gradients, RISE
         → Is the model a transformer?
              Yes → Attention visualization, SHAP for text
```

---

## Inherently Interpretable Models

### Linear Regression — Coefficient Interpretation

Linear regression is the gold standard for interpretability. The model is:

\[
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
\]

Each coefficient \(\beta_j\) represents: *holding all other features constant, a one-unit increase in \(x_j\) changes the prediction by \(\beta_j\) units*.

**Critical prerequisite**: features must be standardized for coefficients to be comparable.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
feature_names = housing.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MUST standardize for coefficient comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Coefficient analysis
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("Feature Coefficients (standardized):")
print(coef_df.to_string(index=False))

# Interpretation: largest |coefficient| = most influential feature
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['steelblue' if c > 0 else 'tomato' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Standardized Coefficient')
ax.set_title('Linear Regression Coefficients\n(Blue=positive effect, Red=negative effect)')
plt.tight_layout()
plt.show()

# Confidence intervals via bootstrap
from sklearn.utils import resample

def bootstrap_coefs(X, y, model_class, n_bootstraps=500):
    coefs = []
    for _ in range(n_bootstraps):
        X_bs, y_bs = resample(X, y)
        m = model_class()
        m.fit(X_bs, y_bs)
        coefs.append(m.coef_)
    return np.array(coefs)

# Note: coefficients ± 2*std give ~95% confidence interval
```

### Logistic Regression — Odds Ratio Interpretation

For classification, logistic regression coefficients represent log-odds:

\[
\log\frac{P(Y=1|x)}{P(Y=0|x)} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
\]

The **odds ratio** for feature \(j\) is \(e^{\beta_j}\): a multiplicative factor on the odds per unit increase in \(x_j\).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_scaled, y)

odds_ratios = pd.DataFrame({
    'feature': cancer.feature_names,
    'log_odds': log_reg.coef_[0],
    'odds_ratio': np.exp(log_reg.coef_[0]),
    'direction': ['↑ risk' if c > 0 else '↓ risk' for c in log_reg.coef_[0]]
}).sort_values('odds_ratio', ascending=False)

print(odds_ratios.to_string(index=False))
# Odds ratio > 1: feature increases probability of class 1
# Odds ratio < 1: feature decreases probability of class 1
```

### Decision Trees — Visualization and Path Explanations

Decision trees are fully transparent — each prediction follows a deterministic path of binary rules.

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# Text representation
rules = export_text(tree, feature_names=list(iris.feature_names))
print(rules)

# Visual representation
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    tree,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax
)
plt.title('Decision Tree - Fully Interpretable Structure')
plt.show()

# Explain a single prediction by tracing its path
def explain_prediction_path(tree, X_sample, feature_names):
    """Trace the decision path for one sample"""
    node_indicator = tree.decision_path(X_sample.reshape(1, -1))
    leaf_id = tree.apply(X_sample.reshape(1, -1))
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    
    node_ids = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    print(f"\nDecision path for sample:")
    for node_id in node_ids:
        if leaf_id[0] == node_id:
            print(f"  → LEAF: Predicted class {tree.tree_.value[node_id].argmax()}")
        else:
            fname = feature_names[feature[node_id]]
            thresh = threshold[node_id]
            val = X_sample[feature[node_id]]
            direction = "<=" if val <= thresh else ">"
            print(f"  Node {node_id}: {fname} = {val:.3f} {direction} {thresh:.3f}")

explain_prediction_path(tree, X.iloc[0].values, list(iris.feature_names))
```

### Generalized Additive Models (GAMs)

GAMs extend linear models while maintaining interpretability via additive shape functions:

\[
g(E[Y]) = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)
\]

Each \(f_j\) is a non-linear smooth function, but the model remains additive — no feature interactions. Each feature's effect can be plotted independently.

```python
# pip install interpret
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Global explanation: shape functions for each feature
ebm_global = ebm.explain_global()
show(ebm_global)  # Interactive HTML visualization

# Local explanation: contribution of each feature for one prediction
ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
show(ebm_local)
```

---

## Feature Importance Methods

### Impurity-Based Feature Importance (Tree Models)

For Random Forests and Gradient Boosted Trees, feature importance is computed as the total reduction in node impurity (Gini or MSE) weighted by node probability:

\[
\text{Importance}(x_j) = \sum_{t: \text{split on } x_j} p(t) \cdot \Delta I(t)
\]

where \(p(t) = n_t / N\) is the fraction of samples reaching node \(t\), and \(\Delta I(t)\) is the impurity decrease.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target
feature_names = list(cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_,
    'std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
}).sort_values('importance', ascending=False)

print("Top 10 Features by Impurity Importance:")
print(importance_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 8))
top_n = importance_df.head(15)
ax.barh(top_n['feature'][::-1], top_n['importance'][::-1],
        xerr=top_n['std'][::-1], capsize=3, color='steelblue', alpha=0.8)
ax.set_xlabel('Mean Decrease in Impurity')
ax.set_title('Random Forest Feature Importance (with std across trees)')
plt.tight_layout()
plt.show()
```

**Known Pitfalls of Impurity-Based Importance:**
- **High-cardinality bias**: Continuous features and features with many unique values are favored even if irrelevant (more possible split points = more chances to appear important)
- **Correlated features**: Importance is split among correlated features, underestimating each
- **Computed on training data**: May reflect overfitting

### Permutation Feature Importance

A more reliable approach: for each feature, randomly shuffle its values across the test set and measure the performance drop. Large drop = important feature.

\[
\text{PFI}(x_j) = L(f, D) - L(f, D_{\text{perm}_j})
\]

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance on TEST set (critical!)
perm_result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,          # Repeat permutation 30 times for stability
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'      # Use metric appropriate to the task
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_result.importances_mean,
    'importance_std': perm_result.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nTop 10 Features by Permutation Importance:")
print(perm_df.head(10).to_string(index=False))

# Compare impurity vs permutation importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_imp = importance_df.head(10)
top_perm = perm_df.head(10)

axes[0].barh(top_imp['feature'][::-1], top_imp['importance'][::-1], color='steelblue')
axes[0].set_title('Impurity Importance (training data bias)')

axes[1].barh(top_perm['feature'][::-1], top_perm['importance_mean'][::-1],
             xerr=top_perm['importance_std'][::-1], capsize=3, color='darkorange')
axes[1].set_title('Permutation Importance (test set, unbiased)')

plt.tight_layout()
plt.show()
```

**When features are correlated**, permutation importance can be misleading because shuffling one feature may break its correlation with another, artificially inflating importance. Use **conditional permutation importance** or SHAP in this case.

---

## Partial Dependence Plots (PDP)

PDP shows the **marginal effect** of one or two features on the predicted outcome, averaging over all other features. The partial dependence function for feature \(x_S\) is:

\[
\hat{f}_{x_S}(x_S) = E_{x_C}\left[\hat{f}(x_S, x_C)\right] = \int \hat{f}(x_S, x_C) \, dP(x_C)
\]

Estimated empirically by averaging predictions over the data:

\[
\hat{f}_{x_S}(x_S) = \frac{1}{n} \sum_{i=1}^{n} \hat{f}(x_S, x_C^{(i)})
\]

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Single feature PDPs for top features
fig, ax = plt.subplots(figsize=(14, 6))
PartialDependenceDisplay.from_estimator(
    rf, X_train,
    features=[0, 1, 2, 3],          # Feature indices
    feature_names=feature_names,
    grid_resolution=50,              # Number of grid points
    percentiles=(0.05, 0.95),        # Clip extreme values
    n_jobs=-1,
    ax=ax
)
plt.suptitle('Partial Dependence Plots — Marginal Effect of Each Feature')
plt.tight_layout()
plt.show()

# 2D Interaction PDP: shows joint effect of two features
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    rf, X_train,
    features=[(0, 1)],              # Tuple = interaction
    feature_names=feature_names,
    grid_resolution=25,
    ax=ax
)
plt.suptitle(f'Interaction PDP: {feature_names[0]} × {feature_names[1]}')
plt.tight_layout()
plt.show()
```

**PDP Assumptions and Limitations:**
- Assumes features are **independent** — averaged predictions may include unrealistic feature combinations
- Shows average effect, hiding **heterogeneous** effects (some individuals may react oppositely)
- For classification, shows predicted probability, not log-odds

---

## Individual Conditional Expectation (ICE)

ICE plots are per-instance PDPs — instead of averaging, show a separate line for each data point. They reveal **heterogeneity** in the feature effect across different individuals.

\[
\hat{f}^{(i)}_{x_j}(x_j) = \hat{f}(x_j, x_C^{(i)})
\]

**Centered ICE (c-ICE)** removes the intercept by subtracting the prediction at a reference value \(x_j^0\):

\[
\hat{f}^{(i)}_{x_j, \text{centered}}(x_j) = \hat{f}^{(i)}_{x_j}(x_j) - \hat{f}^{(i)}_{x_j}(x_j^0)
\]

This makes it easy to see if effects differ across instances (heterogeneous = lines diverge).

```python
# ICE plots: individual lines per data point
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Standard ICE
PartialDependenceDisplay.from_estimator(
    rf, X_train,
    features=[0],
    kind='individual',              # 'individual' = ICE
    feature_names=feature_names,
    subsample=200,                  # Sample 200 lines for clarity
    alpha=0.05,                     # Semi-transparent
    ax=axes[0]
)
axes[0].set_title('ICE Plot (Individual Lines)')

# Centered ICE
PartialDependenceDisplay.from_estimator(
    rf, X_train,
    features=[0],
    kind='both',                    # 'both' = ICE + PDP average
    centered=True,                  # Center at x=min
    feature_names=feature_names,
    subsample=200,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title('Centered ICE (Heterogeneity Visible)')

plt.tight_layout()
plt.show()
```

**Reading ICE plots:**
- Lines that are **parallel**: homogeneous effect — PDP tells the whole story
- Lines that **cross or diverge**: heterogeneous effect — PDP hides important sub-group differences
- Diverging c-ICE lines suggest **interactions** with other features

---

## Accumulated Local Effects (ALE)

ALE fixes the major flaw of PDP: **correlated features**. Instead of marginalizing over the full distribution (which can produce unrealistic combinations), ALE computes the effect by looking at small local windows along the feature's distribution.

For feature \(x_j\) at value \(z\), the uncentered ALE is:

\[
\tilde{f}_{x_j}^{\text{ALE}}(z) = \int_{z_{\min}}^{z} E\left[\frac{\partial \hat{f}(x)}{\partial x_j} \bigg| x_j = t\right] dt
\]

In practice, computed over bins:

\[
\hat{f}_{x_j}^{\text{ALE}}(z_k) = \sum_{l=1}^{k} \frac{1}{n_l} \sum_{i: x_j^{(i)} \in (z_{l-1}, z_l]} \left[\hat{f}(z_l, x_C^{(i)}) - \hat{f}(z_{l-1}, x_C^{(i)})\right]
\]

```python
# pip install alibi
from alibi.explainers import ALE, plot_ale

# ALE for tabular model
ale = ALE(rf.predict, feature_names=feature_names, target_names=['probability'])
ale_exp = ale.explain(X_train.values)

# Plot ALE for all features
plot_ale(ale_exp, n_cols=4, fig_kw={'figsize': (20, 12)})
plt.suptitle('Accumulated Local Effects (ALE) — Unbiased with Correlated Features')
plt.tight_layout()
plt.show()

# ALE vs PDP comparison: they diverge when features are correlated
# ALE is preferred in practice
```

**When to use ALE vs PDP:**
- **Uncorrelated features**: PDP and ALE give similar results; PDP is more interpretable
- **Correlated features**: Always use ALE — PDP produces spurious effects from unrealistic combinations
- **ALE** is faster to compute than PDP for large datasets

---

## SHAP: SHapley Additive exPlanations

SHAP (Lundberg & Lee, 2017) is the most principled approach to feature attribution. It is rooted in **cooperative game theory**.

### Cooperative Game Theory Foundation

Consider a coalition game where features are "players" and the model prediction is the "payout". The **Shapley value** of player \(j\) is the average marginal contribution of feature \(j\) across all possible coalitions:

\[
\phi_j(f, x) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \left[\hat{f}(x_{S \cup \{j\}}) - \hat{f}(x_S)\right]
\]

where \(\mathcal{F}\) is the set of all features, \(S\) ranges over all subsets not containing \(j\), and \(\hat{f}(x_S)\) means the model prediction using only features in \(S\) (all others marginalized out).

**Key properties** (uniquely satisfied by Shapley values):
1. **Efficiency**: \(\sum_j \phi_j = \hat{f}(x) - E[\hat{f}]\) — attributions sum to the prediction gap from baseline
2. **Symmetry**: Features with equal contributions get equal attributions
3. **Dummy**: A feature that contributes nothing gets attribution 0
4. **Additivity**: SHAP values add up correctly for model ensembles

### SHAP as an Additive Feature Attribution

SHAP defines the **SHAP explanation model** as:

\[
g(z') = \phi_0 + \sum_{j=1}^{M} \phi_j z_j'
\]

where \(z' \in \{0,1\}^M\) indicates feature presence, and \(\phi_j\) are the SHAP values. This is a local linear model that satisfies the Shapley axioms.

### TreeSHAP: Exact Computation for Tree Models

For tree-based models (Random Forest, XGBoost, LightGBM), TreeSHAP computes exact Shapley values in **polynomial time** — \(O(TLD^2)\) where \(T\) = number of trees, \(L\) = max leaves, \(D\) = max depth — compared to \(O(2^M)\) for brute force.

The key insight: for trees, the expectation over subsets of features can be computed exactly using the tree's internal structure, tracking which samples flow through each node for each subset.

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target
feature_names = list(cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# TreeSHAP — exact and efficient
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [class0, class1]
# Use class 1 (malignant)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

print(f"Base value (mean prediction): {base_value:.4f}")
print(f"SHAP values shape: {sv.shape}")  # [n_samples, n_features]

# Verify efficiency property: SHAP values sum to prediction - base_value
pred = rf.predict_proba(X_test[:1])[:, 1]
shap_sum = sv[0].sum() + base_value
print(f"Prediction: {pred[0]:.4f}, SHAP sum: {shap_sum:.4f}")  # Should match

# ──────────────────────────────────────────────
# SHAP SUMMARY PLOT (Global + Local combined)
# ──────────────────────────────────────────────
shap.summary_plot(
    sv, X_test,
    feature_names=feature_names,
    plot_type='dot',          # 'dot' = beeswarm, 'bar' = mean |SHAP|
    max_display=15,
    show=True
)
# Reading: y-axis = features (sorted by mean |SHAP|)
# x-axis = SHAP value (direction of effect)
# Color = feature value (red=high, blue=low)

# ──────────────────────────────────────────────
# SHAP BAR PLOT (Global Importance)
# ──────────────────────────────────────────────
shap.summary_plot(
    sv, X_test,
    feature_names=feature_names,
    plot_type='bar',
    show=True
)

# ──────────────────────────────────────────────
# SHAP WATERFALL PLOT (Single Prediction)
# ──────────────────────────────────────────────
idx = 5  # Explain 5th test instance
explanation = shap.Explanation(
    values=sv[idx],
    base_values=base_value,
    data=X_test.iloc[idx].values,
    feature_names=feature_names
)
shap.plots.waterfall(explanation)
# Shows: how each feature pushes prediction from base_value to final prediction

# ──────────────────────────────────────────────
# SHAP FORCE PLOT (Single Prediction — Interactive)
# ──────────────────────────────────────────────
force = shap.force_plot(
    base_value,
    sv[idx],
    X_test.iloc[idx],
    feature_names=feature_names,
    matplotlib=True
)
plt.show()

# ──────────────────────────────────────────────
# SHAP DEPENDENCE PLOT
# ──────────────────────────────────────────────
# Shows SHAP value vs feature value for one feature
# Color by a second feature to reveal interactions
shap.dependence_plot(
    'worst radius',
    sv,
    X_test,
    feature_names=feature_names,
    interaction_index='mean concave points',  # Auto-detect interaction partner
    show=True
)

# ──────────────────────────────────────────────
# SHAP HEATMAP (Multiple Instances)
# ──────────────────────────────────────────────
shap.plots.heatmap(
    shap.Explanation(values=sv[:100], base_values=base_value,
                     data=X_test.iloc[:100].values, feature_names=feature_names)
)
```

### KernelSHAP: Model-Agnostic SHAP

KernelSHAP is a model-agnostic approximation of SHAP values. It uses a **weighted linear regression** on randomly masked feature subsets to estimate Shapley values. The SHAP kernel weighting assigns weights to each coalition \(S\):

\[
\pi_x(S) = \frac{(M-1)}{\binom{M}{|S|} |S| (M - |S|)}
\]

This weighting ensures the linear model minimizes the distance to the Shapley values.

```python
# KernelSHAP for any black-box model
# Use a background dataset (representative summary)
background = shap.sample(X_train, 100)  # Or kmeans: shap.kmeans(X_train, 10)

# Works with any predict function
def predict_fn(X):
    return rf.predict_proba(X)[:, 1]

kernel_explainer = shap.KernelExplainer(predict_fn, background)

# Note: KernelSHAP is slow — use small test sets
shap_values_kernel = kernel_explainer.shap_values(X_test[:20], nsamples=500)

shap.summary_plot(shap_values_kernel, X_test[:20], feature_names=feature_names)
```

### DeepSHAP and GradientSHAP for Neural Networks

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Build a simple neural network
class Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

# Prepare tensors
X_train_t = torch.FloatTensor(X_train.values)
y_train_t = torch.LongTensor(y_train.values)
X_test_t = torch.FloatTensor(X_test.values)

net = Net(X_train.shape[1])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(100):
    net.train()
    optimizer.zero_grad()
    out = net(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

net.eval()

# DeepSHAP — backpropagation-based, uses DeepLIFT rules
background = X_train_t[:100]
deep_explainer = shap.DeepExplainer(net, background)
shap_values_deep = deep_explainer.shap_values(X_test_t[:50])

shap.summary_plot(shap_values_deep[1], X_test[:50], feature_names=feature_names)

# GradientSHAP — combines Integrated Gradients with SHAP
gradient_explainer = shap.GradientExplainer(net, background)
shap_values_grad = gradient_explainer.shap_values(X_test_t[:50])
shap.summary_plot(shap_values_grad[1], X_test[:50], feature_names=feature_names)
```

### SHAP for XGBoost and LightGBM

```python
import xgboost as xgb
import lightgbm as lgb

# XGBoost — TreeSHAP natively integrated
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

explainer_xgb = shap.TreeExplainer(xgb_model)
sv_xgb = explainer_xgb.shap_values(X_test)

# SHAP interaction values (pairwise feature interactions)
shap_interaction = explainer_xgb.shap_interaction_values(X_test[:100])
# shap_interaction[i, j, k] = interaction between features j and k for sample i

# Plot interaction matrix
mean_abs_interaction = np.abs(shap_interaction).mean(axis=0)
plt.figure(figsize=(12, 10))
plt.imshow(mean_abs_interaction, cmap='Blues')
plt.colorbar(label='Mean |SHAP Interaction Value|')
plt.xticks(range(len(feature_names)), feature_names, rotation=90)
plt.yticks(range(len(feature_names)), feature_names)
plt.title('SHAP Feature Interaction Values')
plt.tight_layout()
plt.show()
```

---

## LIME: Local Interpretable Model-Agnostic Explanations

LIME (Ribeiro et al., 2016) explains individual predictions by approximating the model locally with a simple interpretable model (usually sparse linear regression).

### Algorithm

1. **Sample** \(z' \in \{0,1\}^M\) — presence/absence masks for features
2. **Recover** original feature values for each mask: \(z = h_x(z')\)
3. **Predict** using the black-box model: \(\hat{f}(z)\)
4. **Weight** samples by proximity to the instance: \(\pi_x(z) = \exp(-D(x,z)^2 / \sigma^2)\)
5. **Fit** a sparse linear model on the weighted samples:
   \[
   \xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)
   \]

The learned coefficients of the linear model are the LIME attributions.

```python
# pip install lime
from lime import lime_tabular, lime_text, lime_image
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# ──────────────────────────────────────────────
# LIME for Tabular Data
# ──────────────────────────────────────────────
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Benign', 'Malignant'],
    mode='classification',
    discretize_continuous=True,      # Discretize continuous features for interpretability
    discretizer='quartile',
    kernel_width=0.75,               # Controls neighborhood size
    random_state=42
)

# Explain a single instance
instance = X_test.iloc[10].values
lime_exp = lime_explainer.explain_instance(
    instance,
    rf.predict_proba,
    num_features=10,                 # Show top 10 features
    num_samples=5000,                # More samples = more stable
    labels=[1]                       # Explain probability of class 1
)

# Display as list of (feature condition, weight) pairs
print("\nLIME Explanation for Instance 10:")
for feat_cond, weight in lime_exp.as_list(label=1):
    print(f"  {feat_cond:40s}: {weight:+.4f}")

# Visualize
lime_exp.show_in_notebook(show_table=True, show_all=False)

# As matplotlib figure
fig = lime_exp.as_pyplot_figure(label=1)
plt.title('LIME Local Explanation')
plt.tight_layout()
plt.show()

# Verify local fidelity
print(f"\nBlack-box prediction: {rf.predict_proba(instance.reshape(1,-1))[:,1][0]:.4f}")
print(f"LIME local model score: {lime_exp.local_pred[1]:.4f}")
print(f"Intercept: {lime_exp.intercept[1]:.4f}")

# ──────────────────────────────────────────────
# LIME for Text Classification
# ──────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR
from lime.lime_text import LimeTextExplainer

# Build a text classifier pipeline
texts = ["This movie is fantastic", "Terrible waste of time", "Great film!", "Boring and slow"]
labels = [1, 0, 1, 0]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LR())
])
pipeline.fit(texts, labels)

text_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])
text_exp = text_explainer.explain_instance(
    "This is a fantastic film",
    pipeline.predict_proba,
    num_features=6,
    num_samples=2000
)
text_exp.show_in_notebook()

# ──────────────────────────────────────────────
# LIME for Image Classification
# ──────────────────────────────────────────────
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import torch
import torchvision

# Load a pretrained model
model_img = torchvision.models.resnet50(pretrained=True)
model_img.eval()

def predict_image(images):
    """Batch prediction function for LIME"""
    # images: numpy array [N, H, W, C], values in [0,1]
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch = torch.stack([transform(img.astype(np.float32)) for img in images])
    with torch.no_grad():
        probs = torch.softmax(model_img(batch), dim=1).numpy()
    return probs

image_explainer = LimeImageExplainer()
# image_exp = image_explainer.explain_instance(
#     image_array,            # [H, W, 3] numpy array, values in [0,1]
#     predict_image,
#     top_labels=5,
#     hide_color=0,           # Color for masked superpixels
#     num_samples=1000,
#     segmentation_fn=None    # Default: quickshift segmentation
# )
# 
# temp, mask = image_exp.get_image_and_mask(
#     image_exp.top_labels[0],
#     positive_only=True,
#     num_features=5,
#     hide_rest=False
# )
# plt.imshow(mark_boundaries(temp, mask))
```

**LIME Limitations:**
- **Instability**: Different runs can produce different explanations (random sampling)
- **Linear approximation**: Non-linear local regions are poorly captured
- **Neighborhood definition**: Hard to define meaningful neighborhood for tabular data
- **Fidelity vs interpretability**: More complex local models improve fidelity but reduce interpretability
- **Correlated features**: Explanations can be misleading when features are correlated

---

## Integrated Gradients

Integrated Gradients (IG, Sundararajan et al., 2017) is a gradient-based attribution method for differentiable models. It satisfies the **completeness axiom**: attributions sum to the difference between the model output at the input and at a reference baseline. Unlike simple gradients (which can be noisy and saturate for ReLU), IG averages gradients along a path from a neutral baseline to the input, capturing the *cumulative* contribution of each feature.

### Conceptual Intuition

**Why not plain gradients?** For a ReLU network, many neurons have zero gradient; the gradient at the input may be zero even when the input clearly matters (the "saturation" problem). IG fixes this by integrating the gradient along a path—the path from baseline to input. Each point on the path gets a gradient; the integral aggregates how the model's sensitivity to feature \(x_i\) evolves as we interpolate.

**Baseline choice**: The baseline represents "absence" of the feature. Common choices: (1) **zeros** for images/positive features, (2) **mean of training data**, (3) **blurred input**. The attribution is sensitive to baseline—ensure it is a meaningful reference (e.g., black image for "what added" explanations).

### Mathematical Formulation

The integrated gradient for feature \(x_i\) along the straight-line path from baseline \(x'\) to input \(x\):

\[
\text{IntegratedGrad}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha
\]

The integral is approximated by the **Riemann sum** with \(m\) steps:

\[
\text{IntegratedGrad}_i(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i}
\]

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def integrated_gradients(model, input_tensor, baseline=None, steps=300, target_class=None):
    """
    Compute Integrated Gradients attribution.
    
    Args:
        model: PyTorch model
        input_tensor: [1, ...] input tensor
        baseline: reference input (zeros by default)
        steps: number of integration steps
        target_class: class index to explain (None = argmax)
    
    Returns:
        attributions: same shape as input_tensor
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    # Interpolate inputs along the path
    alphas = torch.linspace(0, 1, steps + 1).view(-1, *([1] * (input_tensor.dim() - 1)))
    interpolated = baseline + alphas * (input_tensor - baseline)  # [steps+1, ...]
    interpolated.requires_grad_(True)
    
    # Forward pass for all interpolated inputs
    outputs = model(interpolated)
    
    if target_class is None:
        target_class = outputs[-1].argmax().item()
    
    # Backward pass
    score = outputs[:, target_class].sum()
    model.zero_grad()
    score.backward()
    
    grads = interpolated.grad.detach()  # [steps+1, ...]
    
    # Trapezoidal rule for integration
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2
    
    # Multiply by (input - baseline)
    attributions = (input_tensor - baseline) * avg_grads
    
    return attributions, target_class

# For tabular data
net.eval()
instance = X_test_t[0:1]
baseline = torch.zeros_like(instance)  # Or mean of training data

attrs, predicted_class = integrated_gradients(net, instance, baseline, steps=300)
attrs_np = attrs.numpy().squeeze()

# Visualize
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['steelblue' if a > 0 else 'tomato' for a in attrs_np]
ax.bar(range(len(feature_names)), attrs_np, color=colors)
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Attribution (Integrated Gradient)')
ax.set_title(f'Integrated Gradients — Predicted Class: {predicted_class}')
plt.tight_layout()
plt.show()

# Verify completeness: sum of attributions ≈ f(x) - f(baseline)
net.eval()
with torch.no_grad():
    f_x = net(instance).softmax(dim=1)[0, predicted_class].item()
    f_baseline = net(baseline).softmax(dim=1)[0, predicted_class].item()

print(f"Completeness check:")
print(f"  f(x) - f(baseline) = {f_x - f_baseline:.4f}")
print(f"  Sum of attributions = {attrs_np.sum():.4f}")
```

### Captum Library for PyTorch Attribution

```python
# pip install captum
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    DeepLift,
    Saliency,
    InputXGradient,
    LayerGradCam,
    NoiseTunnel
)

ig = IntegratedGradients(net)
attributions_ig = ig.attribute(
    instance,
    baseline,
    target=predicted_class,
    n_steps=300,
    return_convergence_delta=True
)

# Noise tunnel: average over noisy inputs for smoother attributions
nt = NoiseTunnel(ig)
attributions_smooth = nt.attribute(
    instance,
    nt_type='smoothgrad',    # or 'smoothgrad_sq', 'vargrad'
    nt_samples=50,
    stdevs=0.1,
    target=predicted_class
)
```

---

## Grad-CAM and Vision Attribution

### Grad-CAM: Gradient-weighted Class Activation Mapping

Grad-CAM (Selvaraju et al., 2017) produces class-discriminative localization maps for CNNs. The key idea: use the gradient of the class score with respect to feature maps of the last convolutional layer.

**Step 1**: Compute gradient of class score w.r.t. last conv layer feature maps:
\[
\alpha_k^c = \underbrace{\frac{1}{Z} \sum_i \sum_j}_{\text{global avg pool}} \frac{\partial y^c}{\partial A_{ij}^k}
\]

**Step 2**: Weighted combination + ReLU:
\[
L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)
\]

```python
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    """Grad-CAM implementation for PyTorch CNN models"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Pool gradients over spatial dimensions
        pooled_grads = self.gradients.mean(dim=[2, 3])  # [1, C]
        
        # Weight activation maps
        activations = self.activations[0]  # [C, H, W]
        for k in range(activations.shape[0]):
            activations[k] *= pooled_grads[0, k]
        
        # Compute heatmap
        heatmap = activations.mean(dim=0)  # [H, W]
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.numpy(), target_class, output
    
    def overlay_on_image(self, image_np, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        superimposed = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
        return superimposed

# Usage with ResNet
model_vis = models.resnet50(pretrained=True)
model_vis.eval()

# Target the last conv layer
grad_cam = GradCAM(model_vis, model_vis.layer4[-1].conv3)

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# image = Image.open("path/to/image.jpg")
# image_tensor = transform(image).unsqueeze(0)
# heatmap, pred_class, logits = grad_cam.generate_cam(image_tensor)
# overlaid = grad_cam.overlay_on_image(np.array(image.resize((224,224))), heatmap)
# 
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(image.resize((224,224)))
# axes[0].set_title('Original Image')
# axes[1].imshow(heatmap, cmap='jet')
# axes[1].set_title('Grad-CAM Heatmap')
# axes[2].imshow(overlaid)
# axes[2].set_title(f'Overlay (Class: {pred_class})')
# plt.tight_layout()
# plt.show()
```

### Guided Backpropagation and Grad-CAM++

```python
from captum.attr import GuidedBackprop, GuidedGradCam, LayerGradCam

# Guided Backpropagation: modifies backward pass to only propagate positive gradients
gbp = GuidedBackprop(model_vis)
# attrs_gbp = gbp.attribute(image_tensor, target=pred_class)

# Guided Grad-CAM: element-wise product of Guided Backprop and Grad-CAM
ggcam = GuidedGradCam(model_vis, model_vis.layer4[-1])
# attrs_ggcam = ggcam.attribute(image_tensor, target=pred_class)

# Layer-wise Relevance Propagation (LRP) via Captum
from captum.attr import LRP
lrp = LRP(net)
```

### RISE: Randomized Input Sampling for Explanation

RISE generates explanations by masking random regions and observing how predictions change — applicable to any black-box model including non-differentiable ones.

```python
def RISE(predict_fn, image, n_masks=2000, mask_size=8, p=0.5):
    """
    RISE: Randomized Input Sampling for Explanation
    
    Args:
        predict_fn: function(batch) -> class probabilities
        image: numpy [H, W, C]
        n_masks: number of random masks
        mask_size: size of upsampled mask grid
    """
    H, W, C = image.shape
    
    # Generate random binary masks at low resolution, then upsample
    masks = np.random.binomial(1, p, size=(n_masks, mask_size, mask_size))
    masks_upsampled = np.array([
        cv2.resize(m.astype(float), (W, H), interpolation=cv2.INTER_LINEAR)
        for m in masks
    ])  # [n_masks, H, W]
    
    # Apply masks to image
    masked_images = image[np.newaxis] * masks_upsampled[..., np.newaxis]  # [n_masks, H, W, C]
    
    # Predict on masked images
    probs = predict_fn(masked_images)  # [n_masks, n_classes]
    
    # RISE saliency = weighted sum of masks
    saliency = np.zeros((H, W, probs.shape[1]))
    for c in range(probs.shape[1]):
        saliency[:, :, c] = (masks_upsampled * probs[:, c:c+1, np.newaxis]).mean(axis=0)
    
    return saliency / p  # Normalize
```

---

## Attention Visualization for Transformers

Transformer attention weights show which tokens attend to which. **Attention Rollout** (Abnar & Zuidema, 2020) improves on raw attention by propagating relevance through layers and incorporating residual connections, yielding a more faithful input attribution than single-layer attention.

### Raw Attention vs. Attention Rollout

**Raw attention**: A single layer's attention matrix (e.g., last layer, head 0) shows where each token attends. **Limitation**: Attention is not the same as importance—high attention can be to punctuation or to tokens that don't affect the output. Jain & Wallace (2019) showed attention weights can be manipulated without changing predictions.

**Attention Rollout**: Aggregates attention across all layers, accounting for residual connections. At each layer, the effective attention is \((A + I)/2\) (or a proper residual formulation), then multiplied layer-to-layer. The result is a single matrix indicating how much each input token influences the output. More faithful than raw attention for "which tokens mattered."

**Gradient × Input**: For stronger causal attribution, combine with gradient-based methods (e.g., gradient × embedding) rather than relying on attention alone.

```python
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(text, model_name='bert-base-uncased', layer=-1, head=0):
    """Visualize BERT attention weights for a text input"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # attentions: tuple of [1, num_heads, seq_len, seq_len] per layer
    attentions = outputs.attentions
    
    # Get specific layer and head
    attn_matrix = attentions[layer][0, head].numpy()  # [seq_len, seq_len]
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(tokens)), max(6, len(tokens) * 0.8)))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        vmin=0, vmax=attn_matrix.max(),
        ax=ax
    )
    ax.set_title(f'Attention Weights — Layer {layer}, Head {head}')
    ax.set_xlabel('Key (token being attended to)')
    ax.set_ylabel('Query (token doing the attending)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return attn_matrix, tokens

# attn, tokens = visualize_attention("The patient was given aspirin for headache relief")

def attention_rollout(attentions, head_fusion='mean', discard_ratio=0.0):
    """
    Attention Rollout (Abnar & Zuidema, 2020): propagate attention through all layers.
    Incorporates residual connections for more faithful input attribution.

    Args:
        attentions: list of [batch, n_heads, seq_len, seq_len] per layer
        head_fusion: 'mean' (average heads) or 'max'
        discard_ratio: drop lowest-attention values for sparsity (0 = no drop)

    Returns:
        rollout: [seq_len, seq_len] matrix of token-to-token relevance
    """
    # Start with identity (each token attends to itself)
    result = torch.eye(attentions[0].shape[-1], device=attentions[0].device)

    for layer_attn in attentions:
        # Fuse heads
        if head_fusion == 'mean':
            attn = layer_attn[0].mean(dim=0)  # [T, T]
        else:
            attn = layer_attn[0].max(dim=0)[0]

        # Incorporate residual: (I + A) / 2 then renormalize
        attn = attn + torch.eye(attn.shape[-1], device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Optional: sparsify by discarding low values
        if discard_ratio > 0:
            flat = attn.flatten()
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                thresh = flat.kthvalue(k).values
                attn = torch.where(attn >= thresh, attn, torch.zeros_like(attn))
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        result = attn @ result

    return result.detach().cpu().numpy()

# Apply rollout: get token importances (sum over columns = how much each input token influenced output)
# rollout_matrix = attention_rollout(outputs.attentions)
# token_importance = rollout_matrix[-1]  # Last token's attention to all inputs (for CLS/decoder)
# Or: token_importance = rollout_matrix.sum(axis=0)  # Total influence of each token

# BertViz for interactive attention visualization
# pip install bertviz
# from bertviz import head_view, model_view
# head_view(attentions, tokens)  # Interactive multi-head visualization
# model_view(attentions, tokens)  # All layers and heads overview
```

**Limitations of attention as explanation:**
- Attention is not the same as "importance" — high attention to a token doesn't mean it caused the prediction
- Attention can be manipulated without changing predictions (Jain & Wallace, 2019)
- Use attention in combination with gradient methods for more reliable insights

---

## Anchors: Rule-Based Explanations

Anchors (Ribeiro et al., 2018) find **sufficient conditions** — minimal subsets of feature conditions that "anchor" a prediction with high precision.

An anchor \(A\) for prediction \(\hat{f}(x)\) satisfies:
\[
E[1(\hat{f}(z) = \hat{f}(x)) \mid z \in \mathcal{D}(A)] \geq \tau
\]

where \(\tau\) is a precision threshold (typically 0.95), and \(\mathcal{D}(A)\) is the perturbation distribution satisfying anchor \(A\).

```python
# pip install alibi
from alibi.explainers import AnchorTabular
import numpy as np

anchor_explainer = AnchorTabular(
    predictor=rf.predict,
    feature_names=feature_names,
    categorical_names={}  # Dict mapping feature index to category names
)
anchor_explainer.fit(X_train.values, disc_perc=(25, 50, 75))

# Explain a single prediction
instance = X_test.iloc[5].values
explanation = anchor_explainer.explain(
    instance,
    threshold=0.95,          # Minimum precision of the rule
    delta=0.1,               # Confidence in precision estimate
    tau=0.15,                # Tolerance for stopping
    beam_size=4
)

print(f"\nAnchor Explanation:")
print(f"Prediction: {rf.predict(instance.reshape(1,-1))[0]}")
print(f"Anchor rule: {' AND '.join(explanation.anchor)}")
print(f"Precision: {explanation.precision:.3f}")
print(f"Coverage: {explanation.coverage:.3f}")
# Coverage: fraction of data that satisfies the anchor
# High coverage = more general rule
# High precision = rule reliably predicts this class
```

---

## Counterfactual Explanations

Counterfactuals answer: **"What would need to change to get a different prediction?"** This is highly actionable — "If your credit score were 650 instead of 610, you would have been approved."

A counterfactual \(x'\) is found by:
\[
x' = \arg\min_{x'} \left[ \lambda \cdot \text{loss}(f(x'), y_{\text{target}}) + d(x, x') \right]
\]

where \(d(x, x')\) measures proximity to the original instance and \(y_{\text{target}}\) is the desired prediction.

```python
# pip install dice-ml
import dice_ml
from dice_ml import Dice
import pandas as pd

# DiCE: Diverse Counterfactual Explanations
d = dice_ml.Data(
    dataframe=pd.concat([X_train, y_train], axis=1),
    continuous_features=feature_names,
    outcome_name=y_train.name
)

m = dice_ml.Model(model=rf, backend='sklearn')
exp = Dice(d, m, method='random')

# Generate diverse counterfactuals
query_instance = X_test.iloc[0:1]
dice_exp = exp.generate_counterfactuals(
    query_instance,
    total_CFs=5,                  # Number of diverse CFs
    desired_class='opposite',     # Flip the prediction
    permitted_range=None,         # Optionally restrict feature ranges
    features_to_vary='all'        # Or list specific features
)
dice_exp.visualize_as_dataframe()

# Alibi counterfactuals (gradient-based for differentiable models)
from alibi.explainers import Counterfactual

# For neural networks — differentiable optimization
# cf = Counterfactual(net_predict, shape=(1, n_features), target_proba=1.0,
#                     target_class='other', max_iter=1000, lam_init=1e-1,
#                     max_lam_steps=10, tol=0.05)
# cf_exp = cf.explain(instance.reshape(1,-1))
```

---

## Concept-Based Explanations: TCAV

TCAV (Testing with Concept Activation Vectors, Kim et al., 2018) provides **global** explanations in terms of human-defined **concepts** (e.g., "striped texture", "roundness") rather than individual features.

**Method:**
1. Define a concept with positive examples (images containing the concept) and negative examples
2. Train a linear classifier to separate concept activations from non-concept activations in an intermediate layer → this gives the **Concept Activation Vector (CAV)**
3. Compute **TCAV score**: fraction of inputs for which the directional derivative of the class prediction in the direction of the CAV is positive

\[
\text{TCAV}_{Q, C, l, k} = \frac{\left| \{x \in X_k : S_{C,k,l}(x) > 0\} \right|}{|X_k|}
\]

```python
# TCAV conceptual example (simplified)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import numpy as np

def compute_cav(concept_acts, non_concept_acts):
    """
    Compute Concept Activation Vector.
    
    Args:
        concept_acts: [n_concept, layer_dim] activations for concept examples
        non_concept_acts: [n_non, layer_dim] activations for non-concept examples
    
    Returns:
        cav: unit vector orthogonal to decision boundary
    """
    X = np.vstack([concept_acts, non_concept_acts])
    y = np.array([1] * len(concept_acts) + [0] * len(non_concept_acts))
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    cav = clf.coef_[0]  # Normal to decision boundary
    cav = cav / np.linalg.norm(cav)  # Normalize
    
    return cav, clf.score(X, y)  # Also return CAV accuracy

def tcav_score(model_fn, inputs, cav, target_class):
    """
    Compute TCAV score: fraction of inputs where gradient aligns with CAV.
    """
    inputs_tensor = torch.FloatTensor(inputs)
    inputs_tensor.requires_grad_(True)
    
    output = model_fn(inputs_tensor)
    score = output[:, target_class].sum()
    score.backward()
    
    gradients = inputs_tensor.grad.detach().numpy()  # [N, layer_dim]
    
    # Directional derivative in the direction of CAV
    directional_derivs = (gradients * cav).sum(axis=1)  # [N]
    
    tcav = (directional_derivs > 0).mean()
    return tcav

# Usage:
# concept_examples = images_containing_stripes
# concept_acts = extract_layer_activations(model, concept_examples, layer='layer3')
# random_acts = extract_layer_activations(model, random_images, layer='layer3')
# cav, cav_acc = compute_cav(concept_acts, random_acts)
# score = tcav_score(model, test_images_class_k, cav, target_class=k)
# print(f"TCAV score for 'striped' concept: {score:.3f}")
# Score > 0.5 means the concept positively influences predictions
```

---

## Fairness in ML

### Protected Attributes and Proxy Variables

Even when protected attributes (race, gender, age) are excluded from model inputs, they can be encoded in **proxy features** (zip code ≈ race, name ≈ gender). Fairness analysis must consider this.

### Key Fairness Metrics

#### Group Fairness Metrics

**Demographic Parity (Statistical Parity)**:
\[
P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1)
\]
The positive prediction rate is equal across groups. Also called "disparate impact" — its violation ratio is the most common legal criterion.

**Equalized Odds**:
\[
P(\hat{Y}=1 | A=0, Y=y) = P(\hat{Y}=1 | A=1, Y=y), \quad y \in \{0, 1\}
\]
Both TPR and FPR are equal across groups. More nuanced than demographic parity — demands equal treatment for equally qualified individuals.

**Equal Opportunity** (relaxed equalized odds):
\[
P(\hat{Y}=1 | A=0, Y=1) = P(\hat{Y}=1 | A=1, Y=1)
\]
Only TPR (recall) must be equal — important when false negatives are costly (e.g., loan approval for creditworthy applicants).

**Calibration / Predictive Parity**:
\[
P(Y=1 | \hat{p}=p, A=0) = P(Y=1 | \hat{p}=p, A=1)
\]
Predicted probabilities have the same meaning across groups. Required for risk scores to be fair.

**Impossibility Theorem**: Calibration, equalized odds, and demographic parity cannot all be satisfied simultaneously (except in degenerate cases). Practitioners must choose which metric to optimize.

```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def fairness_metrics(y_true, y_pred, y_score, protected_attr):
    """Compute comprehensive fairness metrics"""
    groups = np.unique(protected_attr)
    results = {}
    
    for group in groups:
        mask = protected_attr == group
        y_t = y_true[mask]
        y_p = y_pred[mask]
        y_s = y_score[mask]
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        n = len(y_t)
        
        results[f'group_{group}'] = {
            'n': n,
            'positive_rate': y_p.mean(),             # For demographic parity
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # For equal opportunity
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # For equalized odds
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'accuracy': (tp + tn) / n
        }
    
    # Compute gaps between groups
    groups_list = list(groups)
    if len(groups_list) == 2:
        g0, g1 = f'group_{groups_list[0]}', f'group_{groups_list[1]}'
        results['demographic_parity_gap'] = abs(
            results[g0]['positive_rate'] - results[g1]['positive_rate']
        )
        results['equal_opportunity_gap'] = abs(
            results[g0]['tpr'] - results[g1]['tpr']
        )
        results['equalized_odds_gap'] = max(
            abs(results[g0]['tpr'] - results[g1]['tpr']),
            abs(results[g0]['fpr'] - results[g1]['fpr'])
        )
    
    return results

# Example usage
# protected = data['gender'].values  # 0 = female, 1 = male
# metrics = fairness_metrics(y_test, y_pred, y_prob, protected)
# print(pd.DataFrame(metrics).T)
```

### Individual Fairness

Individual fairness requires similar individuals to receive similar predictions:
\[
d_{\hat{Y}}(\hat{f}(x), \hat{f}(x')) \leq L \cdot d_X(x, x')
\]

where \(d_X\) is a task-specific similarity metric on inputs and \(d_{\hat{Y}}\) is a distance on predictions.

---

## Fairness Toolkits

### Fairlearn

Microsoft's toolkit for assessing and improving fairness.

```python
# pip install fairlearn
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference,
    MetricFrame
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

import pandas as pd

# Assess fairness
sensitive_features = X_test['gender'] if 'gender' in X_test.columns else np.random.randint(0, 2, len(y_test))

y_pred = rf.predict(X_test)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)

print(f"Demographic Parity Difference: {dp_diff:.4f} (0 = fair)")
print(f"Equalized Odds Difference: {eo_diff:.4f} (0 = fair)")

# MetricFrame: comprehensive disaggregated metrics
metrics = {
    'accuracy': lambda y, yp: (y == yp).mean(),
    'precision': lambda y, yp: (yp[yp==1] == y[yp==1]).mean() if (yp==1).any() else 0,
    'recall': lambda y, yp: (yp[y==1] == y[y==1]).mean() if (y==1).any() else 0,
    'selection_rate': lambda y, yp: yp.mean()
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("\nDisaggregated Metrics:")
print(mf.by_group)
print(f"\nMax group difference in accuracy: {mf.difference(method='between_groups')['accuracy']:.4f}")

# Fairness improvement via post-processing
# ThresholdOptimizer: adjust decision threshold per group
fair_model = ThresholdOptimizer(
    estimator=rf,
    constraints='equalized_odds',
    objective='balanced_accuracy_score',
    predict_method='predict_proba'
)
fair_model.fit(X_train, y_train, sensitive_features=sensitive_features[:len(y_train)])
y_pred_fair = fair_model.predict(X_test, sensitive_features=sensitive_features)

# Fairness improvement via in-processing (constrained optimization)
exp_grad = ExponentiatedGradient(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    constraints=EqualizedOdds(),
    eps=0.01          # Maximum allowed constraint violation
)
exp_grad.fit(X_train, y_train, sensitive_features=sensitive_features[:len(y_train)])
```

### AI Fairness 360 (IBM)

```python
# pip install aif360
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing

# Wrap data in AIF360 format
df_train = pd.concat([X_train.assign(label=y_train.values,
                                      gender=sensitive_features[:len(y_train)])], axis=1)

aif_dataset = BinaryLabelDataset(
    df=df_train,
    label_names=['label'],
    protected_attribute_names=['gender'],
    favorable_label=1,
    unfavorable_label=0
)

privileged_groups = [{'gender': 1}]
unprivileged_groups = [{'gender': 0}]

# Measure disparate impact
metric = BinaryLabelDatasetMetric(
    aif_dataset,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

print(f"Disparate Impact: {metric.disparate_impact():.3f}")  # 1.0 = fair
print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.3f}")  # 0 = fair

# Pre-processing: Reweighing
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
aif_reweighed = RW.fit_transform(aif_dataset)
print(f"\nAfter reweighing:")
print(f"  Instance weights range: {aif_reweighed.instance_weights.min():.3f} - {aif_reweighed.instance_weights.max():.3f}")
```

---

## Complete Practical Examples

### Example 1: Full Interpretability Pipeline on Real Data

```python
"""
Complete interpretability analysis pipeline using scikit-learn, SHAP, and LIME
Dataset: California Housing (regression) and Breast Cancer (classification)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

# ── Data preparation ──────────────────────────────────────────────────────────
cancer = load_breast_cancer(as_frame=True)
X, y = cancer.data, cancer.target
feature_names = list(cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model training ────────────────────────────────────────────────────────────
model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_samples_leaf=5, random_state=42
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# ── Step 1: Impurity-Based Feature Importance ─────────────────────────────────
imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='steelblue')
ax.set_xlabel('Feature Importance (impurity)')
ax.set_title('Step 1: Gradient Boosting Feature Importance')
plt.tight_layout()
plt.show()

# ── Step 2: Permutation Importance ───────────────────────────────────────────
perm = permutation_importance(model, X_test, y_test, n_repeats=20,
                               random_state=42, scoring='roc_auc', n_jobs=-1)
perm_df = pd.DataFrame({
    'feature': feature_names,
    'mean': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(perm_df['feature'][::-1], perm_df['mean'][::-1],
        xerr=perm_df['std'][::-1], capsize=3, color='darkorange')
ax.set_xlabel('Decrease in ROC-AUC')
ax.set_title('Step 2: Permutation Importance (more reliable)')
plt.tight_layout()
plt.show()

# ── Step 3: SHAP Global Analysis ─────────────────────────────────────────────
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Beeswarm plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  plot_type='dot', max_display=15, show=True)

# Bar plot
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  plot_type='bar', max_display=15, show=True)

# ── Step 4: SHAP Local Explanations ──────────────────────────────────────────
# Find a correctly predicted malignant case
correct_malignant = np.where((y_test == 1) & (y_pred == 1))[0]
idx = correct_malignant[0]

print(f"\nExplaining instance {idx}:")
print(f"  True label: {'Malignant' if y_test.iloc[idx] == 1 else 'Benign'}")
print(f"  Predicted prob: {y_prob[idx]:.4f}")

# Waterfall plot
explanation = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx].values,
    feature_names=feature_names
)
shap.plots.waterfall(explanation, max_display=12)

# Find a misclassified instance
misclassified = np.where(y_pred != y_test)[0]
if len(misclassified) > 0:
    idx_wrong = misclassified[0]
    print(f"\nExplaining misclassified instance {idx_wrong}:")
    print(f"  True: {'Malignant' if y_test.iloc[idx_wrong] == 1 else 'Benign'}, "
          f"Predicted: {'Malignant' if y_pred[idx_wrong] == 1 else 'Benign'}")
    
    explanation_wrong = shap.Explanation(
        values=shap_values[idx_wrong],
        base_values=explainer.expected_value,
        data=X_test.iloc[idx_wrong].values,
        feature_names=feature_names
    )
    shap.plots.waterfall(explanation_wrong, max_display=12)

# ── Step 5: SHAP Dependence Plot ─────────────────────────────────────────────
# Top 2 most important features by mean |SHAP|
top_features = pd.Series(np.abs(shap_values).mean(axis=0),
                          index=feature_names).nlargest(2).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, feat in enumerate(top_features):
    shap.dependence_plot(feat, shap_values, X_test, feature_names=feature_names,
                          ax=axes[i], show=False)
plt.suptitle('SHAP Dependence Plots (Interactions Colored Automatically)')
plt.tight_layout()
plt.show()

# ── Step 6: PDP and ICE ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1],
    kind='both', centered=True,
    feature_names=feature_names,
    subsample=100, alpha=0.1,
    ax=axes[:2].flatten()[:2]
)
plt.suptitle('Step 6: ICE + PDP (Centered)')
plt.tight_layout()
plt.show()

# ── Step 7: LIME Local Explanation ───────────────────────────────────────────
lime_exp_tabular = LimeTabularExplainer(
    X_train.values, feature_names=feature_names,
    class_names=['Benign', 'Malignant'],
    mode='classification', discretize_continuous=True, random_state=42
)

lime_explanation = lime_exp_tabular.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10, num_samples=5000, labels=[1]
)

print("\nLIME Explanation:")
for feat, weight in sorted(lime_explanation.as_list(label=1), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:45s}: {weight:+.4f}")
```

### Example 2: Model Debugging with Interpretability

```python
"""
Use interpretability tools to debug a model that has learned spurious correlations
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# Simulate a dataset where 'hospital_id' is spuriously correlated with outcome
np.random.seed(42)
n = 1000

# True signal: age and severity determine recovery
age = np.random.randint(20, 80, n)
severity = np.random.uniform(0, 10, n)
hospital_id = np.random.randint(0, 5, n)

# Hospital 0 has better resources → lower threshold for labeling as "recovered"
# But this is a spurious correlation — hospital assignment is not causal
hospital_bias = (hospital_id == 0).astype(float) * 0.3

true_recovery = (severity < 5) & (age < 60)
y = (true_recovery | (np.random.random(n) < hospital_bias)).astype(int)

X = pd.DataFrame({
    'age': age,
    'severity': severity,
    'hospital_id': hospital_id,
    'temperature': np.random.uniform(36, 39, n),  # Random noise
    'blood_pressure': np.random.uniform(80, 140, n)  # Random noise
})

# Train model
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

model_debug = RandomForestClassifier(n_estimators=200, random_state=42)
model_debug.fit(X_train, y_train)

# SHAP analysis reveals the spurious feature
explainer_debug = shap.TreeExplainer(model_debug)
sv_debug = explainer_debug.shap_values(X_test)

print("Model learned features (should only use age and severity):")
mean_abs_shap = pd.Series(
    np.abs(sv_debug[1]).mean(axis=0),
    index=X.columns
).sort_values(ascending=False)
print(mean_abs_shap)

# Reveal: if hospital_id has high SHAP importance, model learned spurious correlation
# Fix: remove hospital_id from features and retrain
X_train_fixed = X_train.drop(columns=['hospital_id'])
X_test_fixed = X_test.drop(columns=['hospital_id'])

model_fixed = RandomForestClassifier(n_estimators=200, random_state=42)
model_fixed.fit(X_train_fixed, y_train)

explainer_fixed = shap.TreeExplainer(model_fixed)
sv_fixed = explainer_fixed.shap_values(X_test_fixed)

print("\nFixed model SHAP values (hospital_id removed):")
mean_abs_shap_fixed = pd.Series(
    np.abs(sv_fixed[1]).mean(axis=0),
    index=X_train_fixed.columns
).sort_values(ascending=False)
print(mean_abs_shap_fixed)
```

---

## Pitfalls and Common Mistakes

### 1. Treating Attention as Ground-Truth Importance

**Problem**: Assuming high attention weight = causal importance. Attention can be manipulated without changing predictions (Jain & Wallace, 2019).

**Fix**: Use attention for qualitative inspection only. For formal attribution, use gradient-based methods (IG, gradient × input) or attention rollout with validation.

### 2. Baseline Choice in Integrated Gradients

**Problem**: Zero baseline for tabular data with negative or mixed-sign features is meaningless; attributions can be misleading.

**Fix**: Use training-data mean, or a domain-specific neutral reference (e.g., blurred image for vision).

### 3. PDP with Correlated Features

**Problem**: PDP marginalizes over other features, creating unrealistic combinations (e.g., 5-year-old with 30 years of work experience).

**Fix**: Use ALE instead, which respects feature correlations.

### 4. Impurity Importance for Feature Selection

**Problem**: Impurity-based importance (RF, GBM) favors high-cardinality and continuous features; biased and can misrank.

**Fix**: Use permutation importance on the test set, or SHAP.

### 5. LIME Instability

**Problem**: LIME uses random sampling; explanations vary across runs; can be sensitive to `num_samples` and `kernel_width`.

**Fix**: Run multiple times; prefer SHAP for stability when feasible.

### 6. Over-Trusting Post-hoc Explanations

**Problem**: Explanations are approximate (LIME) or model-dependent (SHAP). They describe the model, not necessarily reality.

**Fix**: Validate with domain experts; compare multiple methods; use inherently interpretable models when accuracy permits.

### 7. Ignoring Fairness in High-Stakes Applications

**Problem**: Reporting only overall accuracy; disparate impact on protected groups goes unnoticed.

**Fix**: Always disaggregate metrics by protected attributes; use Fairlearn/AIF360; conduct fairness audits before deployment.

---

## Best Practices

### 1. Match the Method to the Question

| Question | Recommended Method |
|----------|-------------------|
| "Which features matter overall?" | Permutation importance, SHAP bar |
| "How does feature X affect predictions?" | PDP, ALE, SHAP dependence |
| "Why did the model predict X for this instance?" | SHAP waterfall, LIME, Integrated Gradients |
| "What's the minimal rule that ensures this prediction?" | Anchors |
| "What would change the prediction?" | DiCE counterfactuals |
| "Is the model fair across groups?" | Fairlearn MetricFrame |
| "Is this CNN looking at the right region?" | Grad-CAM |
| "Which concept does the model use?" | TCAV |

### 2. Prefer Permutation Importance Over Impurity Importance
Impurity importance is biased toward high-cardinality features. Always validate with permutation importance on the **test set**.

### 3. Use ALE Instead of PDP When Features Are Correlated
PDP can create unrealistic feature combinations. ALE is more reliable in practice.

### 4. Validate SHAP/LIME with Domain Knowledge
Confirm that top features make sense to domain experts. Unexpected top features signal data leakage or spurious correlations.

### 5. Beware of LIME Instability
LIME results vary across runs. Run multiple times and check consistency. Use SHAP for more stable explanations.

### 6. Layer Explanations — Simple to Complex
Start with global feature importance → drill into PDPs for top features → use SHAP for specific predictions → investigate outliers and errors.

### 7. Document Explanations
Keep records of model explanations for compliance and auditing. Track how explanations change across model versions.

### 8. Report Fairness Metrics Alongside Accuracy
Never report only overall accuracy. Always disaggregate performance by protected groups. Address identified disparities before deployment.

### 9. Consider Explanation Faithfulness
The explanation should accurately describe the model's reasoning, not just be plausible. Verify using faithfulness metrics (e.g., sufficiency: does removing top-k features degrade performance?).

---

## Resources and Further Reading

| Resource | Description |
|----------|-------------|
| **Interpretable Machine Learning** (Molnar) | Free book covering all methods: christophm.github.io/interpretable-ml-book |
| **SHAP documentation** | github.com/slundberg/shap |
| **Captum** | PyTorch attribution library: captum.ai |
| **Alibi** | Python library for explanations and fairness |
| **Fairlearn** | Microsoft's fairness toolkit |
| **AI Fairness 360** | IBM's comprehensive fairness library |
| **Original SHAP paper** | Lundberg & Lee (2017) — NIPS |
| **Original LIME paper** | Ribeiro et al. (2016) — KDD |
| **Integrated Gradients** | Sundararajan et al. (2017) — ICML |
| **Grad-CAM** | Selvaraju et al. (2017) — ICCV |
| **Anchors** | Ribeiro et al. (2018) — AAAI |
| **DiCE** | Mothilal et al. (2020) |
| **TCAV** | Kim et al. (2018) — ICML |
| **Fairness impossibility** | Chouldechova (2017) |
| **Attention Rollout** | Abnar & Zuidema (2020) — ACL |
| **Attention ≠ Explanation** | Jain & Wallace (2019) — ACL |

---

## Summary

Model interpretability is not a single tool but a **methodological framework**. The right approach depends on the model type, the question being asked, the audience, and regulatory requirements.

**Key takeaways:**

1. **Inherently interpretable models** (linear, GAM, shallow trees) should be the first choice when accuracy permits — they're transparent by design
2. **SHAP** is the most principled method: it satisfies key axioms and handles global and local questions. Use TreeSHAP for tree models, KernelSHAP for any model
3. **LIME** is fast for local explanations but can be unstable — validate results
4. **PDP/ICE** visualize marginal effects; use **ALE** when features are correlated
5. **Grad-CAM** and **Integrated Gradients** are the tools of choice for neural network vision tasks
6. **Fairness is non-negotiable** in high-stakes domains — use Fairlearn or AIF360 to assess and mitigate bias before deployment
7. **Explanation ≠ debugging tool alone** — interpretability is also a communication tool for stakeholders, auditors, and regulators
