# Hyperparameter Tuning: Complete Guide

## Table of Contents
1. [Introduction to Hyperparameter Tuning](#introduction)
2. [Grid Search](#grid-search)
3. [Random Search](#random-search)
4. [Bayesian Optimization](#bayesian)
5. [Optuna Framework](#optuna)
6. [Hyperopt Framework](#hyperopt)
7. [Ray Tune](#ray-tune)
8. [Automated ML (AutoML)](#automl)
9. [Early Stopping Strategies](#early-stopping)
10. [Practical Examples](#examples)
11. [Best Practices](#best-practices)

---

## Introduction to Hyperparameter Tuning {#introduction}

Hyperparameter tuning optimizes model performance by finding the best hyperparameter values. Unlike model parameters (learned during training), hyperparameters are set before training.

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

## Grid Search {#grid-search}

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

## Random Search {#random-search}

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

## Bayesian Optimization {#bayesian}

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

## Optuna Framework {#optuna}

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

## Hyperopt Framework {#hyperopt}

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

## Ray Tune {#ray-tune}

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

## Automated ML (AutoML) {#automl}

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

## Early Stopping Strategies {#early-stopping}

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

## Practical Examples {#examples}

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

## Best Practices {#best-practices}

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

- **Optuna**: optuna.org
- **Hyperopt**: github.com/hyperopt/hyperopt
- **Ray Tune**: docs.ray.io/en/latest/tune
- **Auto-sklearn**: automl.github.io/auto-sklearn

---

## Conclusion

Hyperparameter tuning significantly improves model performance. Key takeaways:

1. **Start with Grid/Random**: Simple and effective
2. **Use Bayesian**: For expensive evaluations
3. **Use Optuna**: Great framework with pruning
4. **Set Budgets**: Time and compute limits
5. **Validate**: Always validate on hold-out set

Remember: Good hyperparameters can make the difference between good and great models!

