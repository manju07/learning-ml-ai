# Ensemble Methods: Complete Guide

## Table of Contents
1. [Introduction to Ensemble Methods](#introduction-to-ensemble-methods)
2. [Bagging](#bagging)
3. [Boosting](#boosting)
4. [Stacking](#stacking)
5. [Voting Classifiers](#voting-classifiers)
6. [XGBoost](#xgboost)
7. [LightGBM](#lightgbm)
8. [CatBoost](#catboost)
9. [Advanced Ensemble Techniques](#advanced-ensemble-techniques)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)
12. [Common Pitfalls](#common-pitfalls)
13. [Complete Runnable Example](#complete-runnable-example)

---

## Introduction to Ensemble Methods

Ensemble methods combine multiple models to improve predictive performance. They often outperform individual models.

### Why Ensembles Work

**Variance reduction (Bagging):** For \( B \) base learners with identical bias and pairwise correlation \( \rho \), the variance of the average is:
\[
\text{Var}(\bar{f}) = \frac{1}{B}\left(1 + (B-1)\rho\right) \sigma^2
\]
When trees are decorrelated (e.g., Random Forest's feature subsampling), \( \rho \) is small and variance shrinks roughly by \( 1/B \).

**Bias reduction (Boosting):** Sequentially adding weak learners (high bias, low variance) that fit residuals reduces the ensemble's bias. The final model is a sum of corrections.

**Wisdom of crowds:** If base errors are uncorrelated, average error decreases as \( 1/\sqrt{B} \). Even partially correlated errors yield gains when individual accuracies exceed 50%.

### Diversity–Accuracy Tradeoff

Ensembles benefit when base models are:
- **Accurate**: Individually better than random
- **Diverse**: Make different mistakes

The **diversity–accuracy tradeoff**: Very similar models (e.g., many shallow trees with same hyperparameters) have low diversity → limited gain from averaging. Very different models (e.g., random vs. expert) may include weak ones → accuracy suffers. **Optimal ensembles** balance strong base learners with sufficient diversity (different algorithms, feature subsets, or training data).

### Types of Ensembles

1. **Bagging**: Train models in parallel on different subsets
2. **Boosting**: Train models sequentially, each correcting previous
3. **Stacking**: Train meta-model on base model predictions
4. **Voting**: Combine predictions by voting

---

## Bagging

### Conceptual Overview

**Bootstrap Aggregating (Bagging)** trains \( B \) models on bootstrap samples (sample with replacement, same size as original). Predictions are averaged (regression) or majority-voted (classification). Each bootstrap sample leaves out ~37% of data (out-of-bag), useful for internal validation. **Key**: Base models should have low bias and high variance (e.g., deep unpruned trees); bagging reduces variance without increasing bias.

### Bootstrap Aggregating

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Base estimator
base_estimator = DecisionTreeClassifier(max_depth=10)

# Bagging classifier
bagging_clf = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=100,
    max_samples=0.8,  # 80% of data for each bootstrap
    max_features=0.8,  # 80% of features
    bootstrap=True,    # Sample with replacement
    bootstrap_features=False,
    random_state=42
)

# Train
bagging_clf.fit(X_train, y_train)

# Predict
predictions = bagging_clf.predict(X_test)
probabilities = bagging_clf.predict_proba(X_test)

# Evaluate
scores = cross_val_score(bagging_clf, X_train, y_train, cv=5)
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Random Forest (bagging of decision trees)
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # sqrt(n_features) for each tree
    bootstrap=True,
    random_state=42,
    n_jobs=-1  # Use all cores
)

rf_clf.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Predictions
predictions = rf_clf.predict(X_test)
probabilities = rf_clf.predict_proba(X_test)
```

### Extra Trees (Extremely Randomized Trees)

```python
from sklearn.ensemble import ExtraTreesClassifier

# Extra Trees - more randomization
et_clf = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

et_clf.fit(X_train, y_train)
```

---

## Boosting

### Conceptual Overview

**Boosting** trains models **sequentially**: each new model focuses on examples the previous ensemble got wrong. Weights or residual targets emphasize hard examples. **AdaBoost** reweights samples by \( w_i \exp(-\alpha y_i h(x_i)) \) where \( \alpha \) is the learner weight; misclassified samples get higher weight. **Gradient Boosting** fits each new learner to the negative gradient of the loss — a generic formulation that works for any differentiable loss.

### AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Stumps
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=42
)

ada_clf.fit(X_train, y_train)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,  # Stochastic gradient boosting
    random_state=42
)

gb_clf.fit(X_train, y_train)

# Staged predictions (for early stopping)
for i, y_pred in enumerate(gb_clf.staged_predict(X_test)):
    if i % 10 == 0:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Iteration {i}: Accuracy = {accuracy:.4f}")
```

---

## XGBoost

### Installation and Basic Usage

```python
import xgboost as xgb

# XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)

# Feature importance
xgb.plot_importance(xgb_clf, max_num_features=10)

# Predictions
predictions = xgb_clf.predict(X_test)
probabilities = xgb_clf.predict_proba(X_test)
```

### XGBoost with DMatrix

```python
# DMatrix format (more efficient)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# Parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss'
}

# Train
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    verbose_eval=10
)

# Predict
predictions = model.predict(dtest)
```

### XGBoost Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_clf = xgb.XGBClassifier(random_state=42)

grid_search = GridSearchCV(
    xgb_clf,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## LightGBM

### Basic Usage

```python
import lightgbm as lgb

# LightGBM classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
)

# Feature importance
lgb.plot_importance(lgb_clf, max_num_features=10)

# Predictions
predictions = lgb_clf.predict(X_test)
```

### LightGBM Dataset Format

```python
# Create datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8
}

# Train
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(10)]
)

# Predict
predictions = model.predict(X_test, num_iteration=model.best_iteration)
```

---

## CatBoost

### Basic Usage

```python
from catboost import CatBoostClassifier, CatBoostRegressor

# CatBoost (handles categorical features automatically)
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False
)

# Fit with categorical features
cat_clf.fit(
    X_train, y_train,
    cat_features=categorical_indices,  # Indices of categorical columns
    eval_set=(X_val, y_val),
    early_stopping_rounds=10
)

# Predictions
predictions = cat_clf.predict(X_test)
probabilities = cat_clf.predict_proba(X_test)
```

---

## Stacking

### Stacking with Meta-Learner: Conceptual Overview

**Stacking** (stacked generalization) uses a **meta-learner** to combine base model predictions. The key insight: base models' *predictions* become features for a second-level model.

**Steps:**
1. Split training data (or use CV) to generate **out-of-fold predictions** — each base model predicts on folds it wasn't trained on.
2. Stack these predictions into a **meta-feature matrix** \( Z \in \mathbb{R}^{n \times M} \) where \( M \) is the number of base models.
3. Train the **meta-learner** on \( (Z, y) \) to learn optimal weights/combinations.
4. At test time: base models predict → meta-learner combines.

**Why CV for meta-features?** Using in-sample predictions would leak information (base models saw the targets). Out-of-fold predictions simulate genuine "test" inputs for the meta-learner.

**Meta-learner choices:**
- **Logistic Regression**: Simple, interpretable weights per base model; good default.
- **Ridge Regression**: For regression stacking; prevents overfitting.
- **Light models**: Meta-learner should be simpler than base models to avoid overfitting the meta-features.

### Manual Stacking

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
import numpy as np

# Base estimators — diverse models for better stacking
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42, kernel='rbf')),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Meta-learner: Logistic Regression learns optimal blend of base predictions
meta_learner = LogisticRegression(C=1.0, max_iter=1000)

# StackingClassifier uses cv=5 to generate OOF meta-features internally
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,  # Critical: prevents target leakage into meta-features
    stack_method='predict_proba',  # Use probabilities (richer than hard labels)
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
predictions = stacking_clf.predict(X_test)

# Inspect meta-learner weights (which base model contributes most)
print("Meta-learner coefficients:", stacking_clf.final_estimator_.coef_)
```

### Advanced Stacking: Multi-Level and Custom Meta-Features

```python
# Multi-level stacking: level-1 outputs become inputs to level-2
level1_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
]

level2_estimator = LogisticRegression(C=0.1)  # Regularization for meta-level

level1_clf = StackingClassifier(
    estimators=level1_estimators,
    final_estimator=level2_estimator,
    cv=5
)

# Optional: Add original features to meta-features (passthrough)
stacking_with_features = StackingClassifier(
    estimators=level1_estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True  # Concatenate X with predictions for meta-learner
)
stacking_with_features.fit(X_train, y_train)
```

---

## Voting Classifiers

### Hard Voting

```python
from sklearn.ensemble import VotingClassifier

# Hard voting (majority class)
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('svm', SVC()),
        ('lr', LogisticRegression())
    ],
    voting='hard'  # Use class predictions
)

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
```

### Soft Voting

```python
# Soft voting (average probabilities)
voting_clf_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', xgb.XGBClassifier()),
        ('lgb', lgb.LGBMClassifier())
    ],
    voting='soft',  # Use probabilities
    weights=[2, 1, 1]  # Weight each model
)

voting_clf_soft.fit(X_train, y_train)
probabilities = voting_clf_soft.predict_proba(X_test)
```

---

## Advanced Ensemble Techniques

### Bayesian Model Averaging (BMA)

**BMA** combines models by weighting each according to its posterior probability given the data:
\[
P(y|x, D) = \sum_{m=1}^{M} P(y|x, m) \, P(m|D)
\]
where \( P(m|D) \propto P(D|m) P(m) \) is the posterior probability of model \( m \). BMA naturally balances model fit (likelihood) with complexity (prior).

**Practical approximation:** Use BIC or AIC to estimate \( P(D|m) \), then softmax to get weights:
\[
w_m = \frac{e^{-\text{BIC}_m/2}}{\sum_j e^{-\text{BIC}_j/2}}
\]

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def bma_weights(models, X_val, y_val):
    """Compute BMA weights from validation log-likelihood (or BIC proxy)."""
    scores = []
    for m in models:
        pred_proba = m.predict_proba(X_val)[:, 1]
        pred_proba = np.clip(pred_proba, 1e-10, 1 - 1e-10)
        # Negative log-likelihood (lower = better)
        nll = -np.mean(y_val * np.log(pred_proba) + (1 - y_val) * np.log(1 - pred_proba))
        scores.append(-nll)  # Higher = better
    scores = np.array(scores)
    # Softmax to get weights
    exp_scores = np.exp(scores - scores.max())
    return exp_scores / exp_scores.sum()

# Example: BMA over 3 models
models_bma = [
    LogisticRegression(max_iter=1000).fit(X_train, y_train),
    GaussianNB().fit(X_train, y_train),
    DecisionTreeClassifier(max_depth=5).fit(X_train, y_train),
]
w = bma_weights(models_bma, X_val, y_val)
print("BMA weights:", w)

# Weighted average of predicted probabilities
y_proba_bma = sum(w[i] * models_bma[i].predict_proba(X_test)[:, 1] for i in range(len(models_bma)))
```

### Blending

```python
def blend_models(models, X_train, y_train, X_val, y_val, X_test):
    """Blend multiple models"""
    # Get predictions from each model on validation set
    val_predictions = []
    for model in models:
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)
        val_predictions.append(pred)
    
    # Train meta-model on validation predictions
    val_features = np.hstack(val_predictions)
    meta_model = LogisticRegression()
    meta_model.fit(val_features, y_val)
    
    # Get test predictions
    test_predictions = []
    for model in models:
        pred = model.predict_proba(X_test)
        test_predictions.append(pred)
    
    # Blend
    test_features = np.hstack(test_predictions)
    final_predictions = meta_model.predict_proba(test_features)
    
    return final_predictions
```

### Measuring and Encouraging Diversity

```python
def prediction_diversity(predictions_list):
    """Average pairwise disagreement: higher = more diverse."""
    n_models = len(predictions_list)
    total_disagree = 0
    count = 0
    for i in range(n_models):
        for j in range(i + 1, n_models):
            total_disagree += np.mean(predictions_list[i] != predictions_list[j])
            count += 1
    return total_disagree / count if count > 0 else 0

# Example: compare homogeneous vs diverse ensemble
preds_rf = [RandomForestClassifier(n_estimators=50, random_state=i).fit(X_train, y_train).predict(X_val)
            for i in range(5)]  # 5 similar RFs
preds_mixed = [
    RandomForestClassifier(n_estimators=50).fit(X_train, y_train).predict(X_val),
    xgb.XGBClassifier(n_estimators=50).fit(X_train, y_train).predict(X_val),
    LogisticRegression().fit(X_train, y_train).predict(X_val),
]
print("Diversity (5 RFs):", prediction_diversity(preds_rf))
print("Diversity (RF+XGB+LR):", prediction_diversity(preds_mixed))  # Typically higher
```

### Ensemble Diversity

```python
def create_diverse_ensemble():
    """Create ensemble with diverse models"""
    models = [
        # Tree-based
        RandomForestClassifier(n_estimators=100),
        xgb.XGBClassifier(n_estimators=100),
        lgb.LGBMClassifier(n_estimators=100),
        
        # Linear
        LogisticRegression(),
        SGDClassifier(),
        
        # Non-linear
        SVC(probability=True),
        KNeighborsClassifier(),
        
        # Neural
        MLPClassifier(hidden_layer_sizes=(100, 50))
    ]
    
    return models
```

---

## Practical Examples

### Example 1: Complete Ensemble Pipeline

```python
from sklearn.model_selection import cross_val_score

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': xgb.XGBClassifier(n_estimators=100),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100),
    'CatBoost': CatBoostClassifier(iterations=100, verbose=False)
}

# Evaluate each model
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}: {results[name]['mean']:.4f} (+/- {results[name]['std']*2:.4f})")

# Create ensemble from best models
best_models = [
    ('rf', models['Random Forest']),
    ('xgb', models['XGBoost']),
    ('lgb', models['LightGBM'])
]

ensemble = VotingClassifier(estimators=best_models, voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate ensemble
ensemble_score = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nEnsemble: {ensemble_score.mean():.4f} (+/- {ensemble_score.std()*2:.4f})")
```

### Example 2: Stacking with Feature Engineering

```python
# Create different feature sets
X_train_poly = PolynomialFeatures(degree=2).fit_transform(X_train)
X_train_scaled = StandardScaler().fit_transform(X_train)

# Models on different feature sets
base_models = [
    ('rf_original', RandomForestClassifier(), X_train),
    ('rf_poly', RandomForestClassifier(), X_train_poly),
    ('svm_scaled', SVC(probability=True), X_train_scaled),
    ('xgb_original', xgb.XGBClassifier(), X_train)
]

# Get predictions for stacking
meta_features = []
for name, model, X in base_models:
    model.fit(X, y_train)
    pred = model.predict_proba(X_val)
    meta_features.append(pred)

meta_X = np.hstack(meta_features)
meta_model = LogisticRegression()
meta_model.fit(meta_X, y_val)
```

---

## Best Practices

1. **Diversity**: Use diverse models (different algorithms)
2. **Quality**: Ensure base models are reasonably good
3. **Tuning**: Tune individual models before ensembling
4. **Validation**: Use proper cross-validation
5. **Weighting**: Consider model performance when weighting
6. **Computational Cost**: Balance performance vs. cost
7. **Interpretability**: Ensembles are less interpretable

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Identical base models** | No diversity → little gain over single model | Vary algorithms, hyperparameters, or feature subsets |
| **Stacking without CV** | Meta-learner trained on in-sample predictions → severe overfitting | Always use `cv=k` in StackingClassifier |
| **Overfitting meta-learner** | Complex meta-model memorizes base predictions | Use simple meta-learner (e.g., LogisticRegression, Ridge) |
| **Including weak models** | One bad model can drag ensemble down | Filter by minimum CV score; use weighted voting |
| **Data leakage in blending** | Training meta-model on same fold used for base training | Use held-out validation set for meta-features |
| **Too many base models** | Diminishing returns; increased compute and overfitting risk | 3–7 diverse, strong models usually sufficient |
| **Ignoring calibration** | Ensemble probabilities may be poorly calibrated | Apply CalibratedClassifierCV on top of ensemble |

---

## Complete Runnable Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, VotingClassifier
)
import xgboost as xgb
import lightgbm as lgb
import numpy as np

X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Diverse base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, eval_metric='logloss')),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=6, verbose=-1))
]

# Individual CV scores
for name, model in base_models:
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='roc_auc')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Soft voting ensemble
voting = VotingClassifier(estimators=base_models, voting='soft')
voting.fit(X_train_s, y_train)
print(f"Voting ROC-AUC: {cross_val_score(voting, X_train_s, y_train, cv=5, scoring='roc_auc').mean():.4f}")

# Stacking with meta-learner
from sklearn.linear_model import LogisticRegression
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5,
    stack_method='predict_proba'
)
stacking.fit(X_train_s, y_train)
y_proba_stack = stacking.predict_proba(X_test_s)[:, 1]
print(f"Stacking Test ROC-AUC: {roc_auc_score(y_test, y_proba_stack):.4f}")
```

---

## Resources and References

- **XGBoost**: xgboost.readthedocs.io
- **LightGBM**: lightgbm.readthedocs.io
- **CatBoost**: catboost.ai
- **Papers & Books**:
  - Breiman, L. *Random Forests* (2001). Machine Learning.
  - Chen & Guestrin. *XGBoost: A Scalable Tree Boosting System* (2016). KDD.
  - Ke et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (2017). NIPS.
  - Wolpert, D. *Stacked Generalization* (1992). Neural Networks.
  - Ho, T.K. *The Random Subspace Method for Constructing Decision Forests* (1998). IEEE TPAMI.
  - Raftery et al. *Bayesian Model Averaging for Linear Regression Models* (1997). JASA.

---

## Conclusion

Ensemble methods significantly improve model performance. Key takeaways:

1. **Start with Bagging**: Random Forest is a great baseline
2. **Try Boosting**: XGBoost, LightGBM often perform best
3. **Use Stacking**: For maximum performance
4. **Ensure Diversity**: Different models capture different patterns
5. **Validate Properly**: Use cross-validation

Remember: Ensembles often win competitions and improve production models!

