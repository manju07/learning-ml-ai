# MLOps: Machine Learning Operations Guide

## Table of Contents
1. [Introduction to MLOps](#introduction-to-mlops)
2. [MLOps Lifecycle](#mlops-lifecycle)
3. [Version Control](#version-control)
4. [CI/CD for ML](#cicd-for-ml)
5. [Feature Stores](#feature-stores)
6. [Model Registry](#model-registry)
7. [Model Monitoring](#model-monitoring)
8. [Model Deployment](#model-deployment)
9. [Infrastructure as Code](#infrastructure-as-code)
10. [Experiment Tracking](#experiment-tracking)
11. [Pitfalls and Anti-Patterns](#pitfalls-and-anti-patterns)
12. [Benchmarks and Maturity](#benchmarks-and-maturity)
13. [Best Practices](#best-practices)
14. [References](#references)

---

## Introduction to MLOps

MLOps (Machine Learning Operations) is the practice of deploying and maintaining ML models in production reliably and efficiently.

### Key Principles
- **Automation**: Automate ML workflows (training, deployment, retraining)
- **Reproducibility**: Ensure reproducible experiments (code, data, environment)
- **Monitoring**: Track model performance and data quality over time
- **Scalability**: Handle production workloads and multi-model deployments
- **Collaboration**: Enable team collaboration across data scientists and engineers

### Why MLOps Differs from Software

Unlike traditional software, ML systems have **two feedback loops**: (1) data scientists iterating on models, and (2) production data influencing model behavior. Models **degrade** when:
- **Data drift**: Input distribution changes (e.g., user behavior shifts)
- **Concept drift**: Relationship between features and target changes (e.g., fraud patterns evolve)
- **Upstream changes**: Feature pipelines or data sources change silently

### MLOps vs DevOps
- **DevOps**: Software development and operations
- **MLOps**: ML model development, deployment, and operations
- **Key Difference**: ML models degrade over time (data drift, concept drift)

---

## MLOps Lifecycle

### Stages

1. **Data Collection**: Gather and store data
2. **Data Validation**: Ensure data quality
3. **Feature Engineering**: Create features
4. **Model Training**: Train models
5. **Model Validation**: Evaluate models
6. **Model Deployment**: Deploy to production
7. **Monitoring**: Monitor performance
8. **Retraining**: Update models

---

## Version Control

### DVC (Data Version Control)

```python
# Install: pip install dvc

# Initialize DVC
# dvc init

# Track data
# dvc add data/train.csv
# git add data/train.csv.dvc

# Track models
# dvc add models/model.pkl
# git add models/model.pkl.dvc

# Push to remote
# dvc push
```

### Git LFS for Large Files

```bash
# Install Git LFS
# git lfs install

# Track large files
# git lfs track "*.pkl"
# git lfs track "*.h5"
# git add .gitattributes
```

---

## CI/CD for ML

### GitHub Actions Example

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/
      - name: Run linting
        run: |
          flake8 src/
  
  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Train model
        run: |
          python train.py
      - name: Upload model
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/
```

### Model Validation

```python
import mlflow
from sklearn.metrics import accuracy_score

def validate_model(model, X_test, y_test, threshold=0.8):
    """Validate model before deployment"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    if accuracy < threshold:
        raise ValueError(f"Model accuracy {accuracy} below threshold {threshold}")
    
    return accuracy

# In CI/CD pipeline
with mlflow.start_run():
    model = train_model()
    accuracy = validate_model(model, X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
```

---

## Feature Stores

A **feature store** centralizes feature computation, storage, and serving for training and inference. It solves **training–serving skew**: features used offline often differ from those at inference due to different code paths, latency requirements, or data availability.

### Offline vs Online Stores

| Store | Use Case | Latency | Example |
|-------|----------|---------|---------|
| **Offline** | Training, batch inference | Minutes–hours | S3, BigQuery, Snowflake |
| **Online** | Real-time inference | Milliseconds | Redis, DynamoDB, dedicated stores |

### Feast (Open Source Feature Store)

```python
# pip install feast

from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64
from datetime import timedelta

# Define entity and feature view
user = Entity(name="user_id", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_purchase_amount", dtype=Float32),
        Field(name="purchase_count_7d", dtype=Int64),
    ],
    source=...,  # Batch or stream source
)

# Training: get historical features
store = FeatureStore(repo_path=".")
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with user_id, event_timestamp
    features=["user_features:avg_purchase_amount", "user_features:purchase_count_7d"]
).to_df()

# Inference: get online features
online_features = store.get_online_features(
    features=["user_features:avg_purchase_amount", "user_features:purchase_count_7d"],
    entity_rows=[{"user_id": 12345}]
).to_dict()
```

### Key Concepts

- **Point-in-time correctness**: Features at training must reflect values available at prediction time (no future leakage).
- **Feature versioning**: Track schema and computation logic; backfill when definitions change.
- **Transform consistency**: Same transforms for training and serving (e.g., normalization, encoding).

### Feature Store Tools

| Tool | Type | Best For |
|------|------|----------|
| **Feast** | Open source | Flexibility, cloud-agnostic |
| **Tecton** | Managed | Enterprise, real-time |
| **Databricks Feature Store** | Managed | Databricks users |
| **SageMaker Feature Store** | Managed | AWS ecosystem |

---

## Model Registry

A **model registry** stores, versions, and manages model artifacts for deployment. It provides governance, lineage, and stage transitions (None → Staging → Production → Archived).

### Registry Concepts

- **Model name**: Logical grouping (e.g., `churn_predictor`)
- **Version**: Immutable snapshot with metadata
- **Stage**: Lifecycle state (Staging, Production, Archived)
- **Aliases**: Human-readable tags (`champion`, `challenger`, `v2.1`)

### MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

# Register model from run
run_id = "abc123"
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "ChurnPredictor")

# Transition to staging
client = MlflowClient()
client.transition_model_version_stage(
    name="ChurnPredictor",
    version=1,
    stage="Staging"
)

# Add model signature (input/output schema)
from mlflow.models import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
mlflow.log_model(model, "model", signature=signature)

# Load by alias
model = mlflow.pyfunc.load_model(model_uri="models:/ChurnPredictor@champion")

# Compare versions
client.search_model_versions("name='ChurnPredictor'")
```

### Registry Best Practices

- **Immutable versions**: Never overwrite; create new versions for retrains
- **Metadata**: Log training config, metrics, data version, Git commit
- **Approval workflow**: Require approval before promoting to Production
- **Rollback**: Keep previous Production version available for quick rollback

---

## Model Monitoring

### Data Drift Detection

```python
from evidently import ColumnDriftMetric, Dashboard
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

# Compare reference and current data
data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference_data, current_data)
data_drift_dashboard.save('reports/data_drift.html')
```

### Model Performance Monitoring

```python
import mlflow
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency')

def predict_with_monitoring(model, data):
    """Make prediction with monitoring"""
    start_time = time.time()
    prediction = model.predict(data)
    latency = time.time() - start_time
    
    prediction_counter.inc()
    prediction_latency.observe(latency)
    
    # Log to MLflow
    mlflow.log_metric("prediction_latency", latency)
    
    return prediction
```

### Alerting

```python
def check_model_performance(accuracy, threshold=0.8):
    """Check if model performance degraded"""
    if accuracy < threshold:
        send_alert(f"Model accuracy dropped to {accuracy}")
        trigger_retraining()
```

---

## Model Deployment

### Docker Container

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: ml-model:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### A/B Testing

```python
def route_request(features):
    """Route request to model A or B"""
    if random.random() < 0.5:
        return model_a.predict(features), 'A'
    else:
        return model_b.predict(features), 'B'

# Track which model was used
results = []
for features in test_data:
    prediction, model_version = route_request(features)
    results.append({
        'prediction': prediction,
        'model': model_version,
        'features': features
    })
```

---

## Infrastructure as Code

### Terraform Example

```hcl
# main.tf
resource "aws_sagemaker_model" "ml_model" {
  name               = "my-ml-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image = "${var.ecr_repository_url}:latest"
  }
}

resource "aws_sagemaker_endpoint_configuration" "endpoint_config" {
  name = "ml-endpoint-config"

  production_variants {
    variant_name           = "variant-1"
    model_name             = aws_sagemaker_model.ml_model.name
    initial_instance_count = 1
    instance_type          = "ml.t2.medium"
  }
}

resource "aws_sagemaker_endpoint" "endpoint" {
  name                 = "ml-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.endpoint_config.name
}
```

---

## Experiment Tracking

### MLflow

```python
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("n_estimators", 100)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("plots/confusion_matrix.png")
```

### Weights & Biases

```python
import wandb

# Initialize
wandb.init(project="my-project")

# Log hyperparameters
wandb.config.learning_rate = 0.01
wandb.config.batch_size = 32

# Train
for epoch in range(epochs):
    loss = train_step()
    wandb.log({"loss": loss, "epoch": epoch})

# Log model
wandb.log_model("model", model)
```

---

## Pitfalls and Anti-Patterns

### 1. Training–Serving Skew

**Problem**: Features or preprocessing differ between training and inference.

```python
# BAD: Different code paths
# Training: sklearn StandardScaler fit on train
# Serving: forgot to apply scaler or used different fit

# GOOD: Single source of truth (feature store, shared transform code)
def get_features(user_id):
    return feature_store.get_online_features(user_id)  # Same computation
```

### 2. Data Leakage

**Problem**: Future or holdout data leaks into training.

- **Temporal leakage**: Using features from after prediction time
- **Target leakage**: Including information derived from the target
- **Cross-validation leakage**: Validation folds not properly separated

### 3. Silent Failures

**Problem**: Model or pipeline fails without alerting.

- **Stale models**: No retraining when data drifts
- **Broken pipelines**: Upstream feature jobs fail silently
- **Wrong version**: Deployed model doesn't match registry

### 4. Ignoring Model Degradation

- Set **performance thresholds** and alert when exceeded
- Schedule **periodic evaluation** on holdout data
- Define **retraining triggers**: accuracy drop, data volume, time

### 5. Over-Engineering Early

- Start with simple pipelines; add complexity when needed
- Prefer managed services (SageMaker, Vertex AI) for initial deployments

---

## Benchmarks and Maturity

### MLOps Maturity Levels

| Level | Characteristics |
|-------|------------------|
| **0** | Manual, ad-hoc scripts, no tracking |
| **1** | Experiment tracking, manual deployment |
| **2** | CI/CD, automated deployment, basic monitoring |
| **3** | Feature store, automated retraining, full governance |
| **4** | Auto-scaling, A/B tests, full automation |

### Tool Comparison (Simplified)

| Need | Options |
|------|---------|
| Experiment tracking | MLflow, Weights & Biases, Neptune |
| Model registry | MLflow, SageMaker, Vertex Model Registry |
| Feature store | Feast, Tecton, Databricks |
| Orchestration | Airflow, Kubeflow, Prefect |
| Deployment | SageMaker, Seldon, KServe, BentoML |
| Monitoring | Evidently, WhyLabs, custom Prometheus |

---

## Best Practices

1. **Version Everything**: Code, data, models, configs
2. **Automate Testing**: Unit tests, integration tests
3. **Monitor Continuously**: Performance, data drift
4. **Document**: Document decisions and experiments
5. **Security**: Secure model endpoints and data
6. **Scalability**: Design for scale from start
7. **Rollback Plan**: Ability to rollback models
8. **Governance**: Model approval process

---

## Tools

- **MLflow**: Experiment tracking and model registry
- **Kubeflow**: Kubernetes ML workflows
- **Airflow**: Workflow orchestration
- **DVC**: Data version control
- **Evidently**: Data and model monitoring
- **Seldon**: Model deployment platform
- **Terraform**: Infrastructure as code

---

## References

- **Google**: [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — Best practices for ML in production
- **Sculley et al.** (2015): *Hidden Technical Debt in Machine Learning Systems* — Pitfalls in ML systems
- **Feast**: [Feature Store Documentation](https://docs.feast.dev/) — Open source feature store
- **MLflow**: [MLflow Docs](https://mlflow.org/docs/latest/index.html) — Experiment tracking and registry
- **Databricks**: [MLOps on Databricks](https://www.databricks.com/glossary/mlops) — End-to-end MLOps guide

---

## Conclusion

MLOps ensures ML models are deployed and maintained effectively. Key takeaways:

1. **Automate**: Automate ML workflows (train, deploy, retrain)
2. **Monitor**: Track model performance, data drift, and pipeline health
3. **Version**: Version code, data, models, and features
4. **Test**: Test models, transforms, and pipelines before deployment
5. **Avoid skew**: Use feature stores to keep training and serving consistent

Remember: MLOps is about making ML production-ready and maintainable!

