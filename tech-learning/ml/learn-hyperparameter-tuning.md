# Hyperparameter Tuning: Complete Guide

## Table of Contents
1. [Introduction to Hyperparameter Tuning](#introduction-to-hyperparameter-tuning)
2. [Grid Search](#grid-search)
3. [Random Search](#random-search)
4. [Bayesian Optimization](#bayesian-optimization)
5. [Hyperband and ASHA](#hyperband-and-asha)
6. [Optuna Framework](#optuna-framework)
7. [Hyperopt Framework](#hyperopt-framework)
8. [Ray Tune](#ray-tune)
9. [Automated ML (AutoML)](#automated-ml-automl)
10. [Early Stopping Strategies](#early-stopping-strategies)
11. [Pitfalls and Failure Modes](#pitfalls-and-failure-modes)
12. [Benchmarks](#benchmarks)
13. [Practical Examples](#practical-examples)
14. [Best Practices](#best-practices)

---

## Introduction to Hyperparameter Tuning

Hyperparameter tuning optimizes model performance by finding the best hyperparameter values. Unlike **model parameters** (learned during training, e.g., weights), **hyperparameters** are set before training and control the learning process. The search space is often high-dimensional and non-convex; evaluations are costly (minutes to hours per trial).

### Types of Hyperparameters

- **Learning Rate**: Step size in optimization
- **Batch Size**: Number of samples per update
- **Number of Layers**: Architecture depth
- **Regularization**: Dropout, L1/L2 coefficients
- **Tree Depth**: For tree-based models
- **Number of Estimators**: For ensemble methods

### Tuning Strategies

1. **Manual**: Try values manually
2. **Grid Search**: Exhaustive search
3. **Random Search**: Random sampling
4. **Bayesian Optimization**: Smart search
5. **Evolutionary**: Genetic algorithms
6. **AutoML**: Automated frameworks

---

## Grid Search

**Grid search** exhaustively evaluates every combination in the parameter grid. Simple and parallelizable. **Curse of dimensionality**: With d parameters and k values each, you need k^d evaluations. For d=5, k=5 → 3,125 trials. Often infeasible for expensive models. Use for 2–3 important parameters with coarse grids.

### Basic Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
```

### Grid Search with Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Parameter grid
param_grid = {
    'scaler__with_mean': [True, False],
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01],
    'classifier__kernel': ['rbf', 'poly', 'sigmoid']
}

# Grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

---

## Random Search

**Random search** samples hyperparameters from specified distributions. Bergstra & Bengio (2012) showed it often **outperforms grid search** with the same budget: random search explores the space more effectively; some parameters matter more than others, and random search doesn't waste budget on irrelevant dimensions. Use as a baseline before Bayesian methods.

### Basic Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # Number of iterations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

### Random Search with Continuous Distributions

```python
from scipy.stats import loguniform

# Continuous distributions
param_distributions = {
    'learning_rate': loguniform(1e-4, 1e-1),
    'alpha': uniform(0.01, 0.1),
    'lambda': uniform(0.1, 1.0)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5
)
```

---

## Bayesian Optimization

**Bayesian optimization (BO)** builds a probabilistic surrogate model (typically a **Gaussian Process**, GP) of the objective from past evaluations, then uses an **acquisition function** to decide where to evaluate next. It balances **exploration** (uncertain regions) and **exploitation** (promising regions), making it sample-efficient for expensive black-box optimization.

**Key components**:
1. **Surrogate**: GP models f(x) ~ N(μ(x), σ²(x))
2. **Acquisition**: EI (Expected Improvement), UCB (Upper Confidence Bound), or PI (Probability of Improvement)
3. **Optimizer**: Maximize acquisition to get next hyperparameter config

**When to use**: Expensive evaluations (e.g., training deep nets), limited budget, continuous/categorical mixed spaces. **When to avoid**: Very cheap evaluations (grid/random may suffice), very high dimensions (GP scales poorly).

### Using scikit-optimize

```bash
pip install scikit-optimize
```

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score

# Define search space
dimensions = [
    Integer(50, 300, name='n_estimators'),
    Integer(5, 20, name='max_depth'),
    Real(0.01, 1.0, name='min_samples_split'),
    Categorical(['sqrt', 'log2'], name='max_features')
]

# Objective function
@use_named_args(dimensions=dimensions)
def objective(**params):
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return -scores.mean()  # Minimize negative accuracy

# Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=dimensions,
    n_calls=50,
    random_state=42,
    verbose=True
)

# Best parameters
best_params = {
    'n_estimators': result.x[0],
    'max_depth': result.x[1],
    'min_samples_split': result.x[2],
    'max_features': result.x[3]
}

print(f"Best parameters: {best_params}")
print(f"Best score: {-result.fun:.4f}")
```

---

## Hyperband and ASHA

**Hyperband** (Li et al., 2017) addresses the **bandit-based early stopping** problem: many hyperparameter configs are poor; we want to stop them early and allocate budget to promising ones. It extends **Successive Halving** (SH): run n configs for k epochs, keep the best n/η, run them for η·k epochs, repeat.

**Idea**: Allocate total budget B across different "brackets." Each bracket runs Successive Halving with different (n, k) trade-offs: many configs few epochs vs few configs many epochs. Hyperband runs all brackets in parallel and returns the best config from any bracket.

**ASHA** (Asynchronous Successive Halving Algorithm) is an asynchronous variant used in Ray Tune: trials can be promoted or stopped early without waiting for the full bracket. **Much more efficient** for distributed tuning.

```python
from ray import tune
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def train_fn(config):
    """Training function: must support early stopping"""
    model = build_model(config)
    for epoch in range(config.get('max_epochs', 100)):
        train_epoch(model)
        val_acc = evaluate(model)
        # Report intermediate metric for early stopping
        tune.report(mean_accuracy=val_acc, training_iteration=epoch + 1)

# Hyperband: stops poor configs early
hyperband = HyperBandScheduler(
    time_attr='training_iteration',
    max_t=100,           # max iterations per config
    grace_period=10,     # min iterations before stopping
    reduction_factor=3,  # keep 1/3 of configs each rung
    metric='mean_accuracy',
    mode='max'
)

# ASHA: asynchronous, better for distributed
asha = ASHAScheduler(
    time_attr='training_iteration',
    max_t=100,
    grace_period=10,
    reduction_factor=3,
    metric='mean_accuracy',
    mode='max'
)

# Run with ASHA
analysis = tune.run(
    train_fn,
    config={
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': tune.choice([32, 64, 128]),
        'max_epochs': 100
    },
    scheduler=asha,
    num_samples=50,
    metric='mean_accuracy',
    mode='max'
)
```

**When to use**: Training that supports early stopping (epochs), expensive per-epoch cost, many trials. **Not for**: Single-run models (e.g., one CV fit), very cheap evaluations.

---

## Optuna Framework

### Installation

```bash
pip install optuna
```

### Basic Optuna

```python
import optuna

def objective(trial):
    """Objective function for Optuna"""
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    
    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # Evaluate
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Create study
study = optuna.create_study(direction='maximize', study_name='rf_optimization')

# Optimize
study.optimize(objective, n_trials=100, timeout=3600)

# Best parameters
print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.4f}")

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### Optuna with Pruning

```python
import optuna
from optuna.pruners import MedianPruner

def objective_with_pruning(trial):
    """Objective with early stopping"""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # Use pruning callback
    scores = []
    for fold in range(5):
        # Train and evaluate fold
        score = evaluate_fold(model, fold)
        scores.append(score)
        
        # Report intermediate value for pruning
        trial.report(score, fold)
        
        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(scores)

# Study with pruner
study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner(n_startup_trials=5)
)

study.optimize(objective_with_pruning, n_trials=100)
```

### Optuna for Deep Learning

```python
def objective_deep_learning(trial):
    """Optuna for neural networks"""
    # Architecture
    n_layers = trial.suggest_int('n_layers', 2, 5)
    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int(f'n_units_l{i}', 32, 256)
        layers.append(n_units)
    
    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Build model
    model = build_model(layers, dropout_rate)
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5)],
        verbose=0
    )
    
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective_deep_learning, n_trials=50)
```

---

## Hyperopt Framework

### Installation

```bash
pip install hyperopt
```

### Basic Hyperopt

```python
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Define search space
space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300]),
    'max_depth': hp.randint('max_depth', 5, 20),
    'min_samples_split': hp.uniform('min_samples_split', 0.01, 1.0),
    'max_features': hp.choice('max_features', ['sqrt', 'log2'])
}

# Objective function
def objective(params):
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split'] * 100),
        max_features=params['max_features'],
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return {'loss': -scores.mean(), 'status': STATUS_OK}

# Optimize
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

print(f"Best parameters: {best}")
```

---

## Ray Tune

### Installation

```bash
pip install ray[tune]
```

### Basic Ray Tune

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_model(config):
    """Training function for Ray Tune"""
    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Report to Ray Tune
    tune.report(mean_accuracy=scores.mean())

# Define search space
config = {
    'n_estimators': tune.choice([50, 100, 200, 300]),
    'max_depth': tune.randint(5, 20)
}

# Scheduler
scheduler = ASHAScheduler(metric='mean_accuracy', mode='max')

# Run tuning
analysis = tune.run(
    train_model,
    config=config,
    num_samples=100,
    scheduler=scheduler,
    metric='mean_accuracy',
    mode='max'
)

# Best config
best_config = analysis.get_best_config('mean_accuracy', 'max')
print(f"Best config: {best_config}")
```

---

## Automated ML (AutoML)

### Auto-sklearn

```bash
pip install auto-sklearn
```

```python
import autosklearn.classification

# AutoML classifier
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 1 hour
    per_run_time_limit=300,  # 5 minutes per model
    memory_limit=4096,  # 4GB
    ensemble_size=1,
    initial_configurations_via_metalearning=25
)

# Fit
automl.fit(X_train, y_train)

# Predict
predictions = automl.predict(X_test)

# Get models
print(automl.show_models())
```

### TPOT

```bash
pip install tpot
```

```python
from tpot import TPOTClassifier

# TPOT AutoML
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=5,
    random_state=42,
    verbosity=2,
    n_jobs=-1
)

tpot.fit(X_train, y_train)

# Export best pipeline
tpot.export('tpot_pipeline.py')

# Predict
predictions = tpot.predict(X_test)
```

---

## Early Stopping Strategies

### Learning Curve Analysis

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv=5):
    """Plot learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', label='Training')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, 'o-', label='Validation')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

plot_learning_curve(model, X_train, y_train)
```

---

## Pitfalls and Failure Modes

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Overfitting to validation** | Tuning too long overfits to validation set | Use nested CV; hold out final test set; limit n_trials |
| **Search space** | Too narrow misses good regions; too wide wastes budget | Start broad, refine; use log-scale for learning rate |
| **Noise** | CV scores are noisy; one good run may be luck | Use multiple seeds; report mean ± std; increase cv folds |
| **Correlation of params** | Some params interact (e.g., lr × batch_size) | Use conditional search spaces; consider joint tuning |
| **Budget exhaustion** | Bayesian/HPO stops before finding optimum | Set realistic timeout; use early stopping for expensive trials |
| **Reproducibility** | Different runs give different "best" params | Set random seeds; fix n_trials; log all trials |
| **Metric mismatch** | Optimizing accuracy when business cares about AUC | Align metric with deployment objective |
| **Cold start (BO)** | GP needs a few points; first trials are random | Use random initialization (n_initial points); Latin hypercube |

---

## Benchmarks

| Benchmark | Domain | Task | Notes |
|-----------|--------|------|-------|
| **HPO-B** | ML | Classification | Standardized HPO benchmarks (XGBoost, MLP, etc.) |
| **NAS-Bench-101/201** | NAS | Architecture search | Tabular benchmarks for quick comparison |
| **PD1** | Chemical | Molecular design | Black-box optimization |
| **MLPerf** | DL | End-to-end | Includes HPO for training |

**Typical metrics**: Best validation score, regret (gap to oracle), wall-clock time to reach target, sample efficiency (score vs. # evaluations).

---

## Practical Examples

### Example 1: Complete Tuning Pipeline

```python
def tune_model(X_train, y_train, model_type='random_forest'):
    """Complete hyperparameter tuning pipeline"""
    
    if model_type == 'random_forest':
        space = {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300]),
            'max_depth': hp.randint('max_depth', 5, 20),
            'min_samples_split': hp.randint('min_samples_split', 2, 20)
        }
        
        def objective(params):
            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=5)
            return -scores.mean()
    
    elif model_type == 'xgboost':
        space = {
            'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
            'max_depth': hp.randint('max_depth', 3, 10),
            'learning_rate': hp.loguniform('learning_rate', -5, -1)
        }
        
        def objective(params):
            model = xgb.XGBClassifier(**params, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=5)
            return -scores.mean()
    
    # Optimize
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    
    return best, trials

# Use
best_params, trials = tune_model(X_train, y_train, 'xgboost')
```

### Example 2: Multi-Model Tuning

```python
def tune_multiple_models(X_train, y_train):
    """Tune multiple models and compare"""
    
    models = {
        'random_forest': RandomForestClassifier(),
        'xgboost': xgb.XGBClassifier(),
        'lightgbm': lgb.LGBMClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        # Define search space based on model
        if name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15]
            }
        elif name == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:  # lightgbm
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        # Grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    # Compare
    for name, result in results.items():
        print(f"{name}: {result['best_score']:.4f}")
    
    return results
```

---

## Best Practices

1. **Start Simple**: Begin with grid/random search
2. **Use Cross-Validation**: Avoid overfitting to validation set
3. **Set Budget**: Limit time/compute resources
4. **Use Pruning**: Early stopping for expensive evaluations
5. **Parallelize**: Use multiple cores/GPUs
6. **Document**: Track all experiments
7. **Validate**: Test best model on hold-out set
8. **Consider Trade-offs**: Accuracy vs. complexity

---

## Resources

**Frameworks**:
- **Optuna**: optuna.org — Modern, pruning, visualization
- **Hyperopt**: github.com/hyperopt/hyperopt — TPE, MongoDB for distributed
- **Ray Tune**: docs.ray.io — Distributed, ASHA, Hyperband
- **scikit-optimize**: scikit-optimize.github.io — GP-based BO
- **Auto-sklearn**: automl.github.io/auto-sklearn
- **TPOT**: epistasislab.github.io/tpot

**Papers**:
- Bergstra & Bengio (2012) — Random search for hyperparameters
- Snoek et al. (2012) — Bayesian optimization with GP
- Li et al. (2017) — Hyperband
- Li et al. (2020) — ASHA (Asynchronous Successive Halving)

---

## Conclusion

Hyperparameter tuning significantly improves model performance. Key takeaways:

1. **Start with Grid/Random**: Simple and effective
2. **Use Bayesian**: For expensive evaluations
3. **Use Optuna**: Great framework with pruning
4. **Set Budgets**: Time and compute limits
5. **Validate**: Always validate on hold-out set

Remember: Good hyperparameters can make the difference between good and great models!

### Summary of Enhancements

- **Bayesian optimization**: Surrogate model, acquisition functions, when to use/avoid
- **Hyperband & ASHA**: Successive Halving, early stopping for expensive training, code example
- **Pitfalls**: Validation overfitting, search space, noise, reproducibility, metric mismatch
- **Benchmarks**: HPO-B, NAS-Bench, MLPerf
- **References**: Key papers (Hyperband, ASHA, GP-BO)

