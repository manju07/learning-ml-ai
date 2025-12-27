# MLOps: Machine Learning Operations Guide

## Table of Contents
1. [Introduction to MLOps](#introduction)
2. [MLOps Lifecycle](#lifecycle)
3. [Version Control](#version-control)
4. [CI/CD for ML](#cicd)
5. [Model Monitoring](#monitoring)
6. [Model Deployment](#deployment)
7. [Infrastructure as Code](#iac)
8. [Experiment Tracking](#experiment-tracking)
9. [Model Registry](#model-registry)
10. [Best Practices](#best-practices)

---

## Introduction to MLOps {#introduction}

MLOps (Machine Learning Operations) is the practice of deploying and maintaining ML models in production reliably and efficiently.

### Key Principles
- **Automation**: Automate ML workflows
- **Reproducibility**: Ensure reproducible experiments
- **Monitoring**: Track model performance
- **Scalability**: Handle production workloads
- **Collaboration**: Enable team collaboration

### MLOps vs DevOps
- **DevOps**: Software development and operations
- **MLOps**: ML model development, deployment, and operations
- **Key Difference**: ML models degrade over time (data drift, concept drift)

---

## MLOps Lifecycle {#lifecycle}

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

## Version Control {#version-control}

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

## CI/CD for ML {#cicd}

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

## Model Monitoring {#monitoring}

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

## Model Deployment {#deployment}

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

## Infrastructure as Code {#iac}

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

## Experiment Tracking {#experiment-tracking}

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

## Model Registry {#model-registry}

### MLflow Model Registry

```python
# Register model
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "MyModel")

# Transition to staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Staging"
)

# Load model from registry
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/MyModel/Staging"
)
```

---

## Best Practices {#best-practices}

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

## Conclusion

MLOps ensures ML models are deployed and maintained effectively. Key takeaways:

1. **Automate**: Automate ML workflows
2. **Monitor**: Track model and data quality
3. **Version**: Version code, data, and models
4. **Test**: Test models before deployment
5. **Document**: Document everything

Remember: MLOps is about making ML production-ready and maintainable!

