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

---

## Introduction to Ensemble Methods

Ensemble methods combine multiple models to improve predictive performance. They often outperform individual models.

### Why Ensembles Work

- **Reduces Variance**: Averaging reduces overfitting
- **Reduces Bias**: Different models capture different patterns
- **Improves Robustness**: Less sensitive to noise
- **Better Generalization**: Works well on unseen data

### Types of Ensembles

1. **Bagging**: Train models in parallel on different subsets
2. **Boosting**: Train models sequentially, each correcting previous
3. **Stacking**: Train meta-model on base model predictions
4. **Voting**: Combine predictions by voting

---

## Bagging

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

### Manual Stacking

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Base estimators
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=5,  # Cross-validation for meta-features
    stack_method='predict_proba',  # Use probabilities
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)
predictions = stacking_clf.predict(X_test)
```

### Advanced Stacking

```python
# Multi-level stacking
level1_estimators = [
    ('rf', RandomForestClassifier()),
    ('xgb', xgb.XGBClassifier()),
    ('lgb', lgb.LGBMClassifier())
]

level2_estimator = LogisticRegression()

# First level
level1_clf = StackingClassifier(
    estimators=level1_estimators,
    final_estimator=level2_estimator,
    cv=5
)

# Second level (stack on top)
final_clf = StackingClassifier(
    estimators=[('level1', level1_clf)],
    final_estimator=LogisticRegression(),
    cv=5
)
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

## Resources

- **XGBoost**: xgboost.readthedocs.io
- **LightGBM**: lightgbm.readthedocs.io
- **CatBoost**: catboost.ai
- **Papers**: 
  - Random Forest (2001)
  - XGBoost (2016)
  - LightGBM (2017)

---

## Conclusion

Ensemble methods significantly improve model performance. Key takeaways:

1. **Start with Bagging**: Random Forest is a great baseline
2. **Try Boosting**: XGBoost, LightGBM often perform best
3. **Use Stacking**: For maximum performance
4. **Ensure Diversity**: Different models capture different patterns
5. **Validate Properly**: Use cross-validation

Remember: Ensembles often win competitions and improve production models!

