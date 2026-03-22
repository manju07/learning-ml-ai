# ML/AI Systems Architecture: From MLOps to LLM Production

## Table of Contents
1. [Introduction](#1-introduction)
2. [ML System Architecture Fundamentals](#2-ml-system-architecture-fundamentals)
3. [Feature Stores](#3-feature-stores)
4. [Model Training Infrastructure](#4-model-training-infrastructure)
5. [Model Serving & Inference](#5-model-serving--inference)
6. [MLOps & CI/CD for ML](#6-mlops--cicd-for-ml)
7. [LLM Systems Architecture](#7-llm-systems-architecture)
8. [RAG Architecture](#8-rag-architecture)
9. [Vector Databases](#9-vector-databases)
10. [Observability for ML Systems](#10-observability-for-ml-systems)
11. [Practical Examples](#11-practical-examples)

---

## 1. Introduction

**ML Systems Architecture** is the discipline of designing production systems that reliably train, evaluate, deploy, and monitor machine learning models at scale. Unlike traditional software, ML systems have unique challenges: data dependencies, non-determinism, model decay, and feedback loops.

### 1.1 Why ML Systems Are Different

```
Traditional Software vs ML Systems:

TRADITIONAL SOFTWARE         ML SYSTEMS
───────────────────────      ──────────────────────────────
Logic is explicit code       Logic is learned from data
Bugs are in code             Bugs can be in data or model
Tests verify behavior        Tests verify metrics (accuracy, AUC)
Deploy once, stable          Models degrade over time (drift)
Version: code only           Version: code + data + model + params
Failure: crash / exception   Failure: silent (bad predictions)
Debug: stack trace           Debug: feature importance, data slices
```

### 1.2 The Hidden Technical Debt in ML Systems

From the famous [Google paper](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — only a small fraction of ML code is the model:

```
ML System Components:

                    ┌─────────────┐
                    │  ML Code    │  ← What everyone focuses on
                    │  (5-10%)    │
                    └─────────────┘
┌──────────────────────────────────────────────────────────────┐
│                  Everything Else (90-95%)                    │
├──────────────┬───────────────┬──────────────┬───────────────┤
│  Data        │  Feature      │  Model       │  Serving      │
│  Collection  │  Engineering  │  Training    │  Infra        │
│  Validation  │  Store        │  Evaluation  │  Monitoring   │
│  Versioning  │  Pipelines    │  Versioning  │  Feedback     │
└──────────────┴───────────────┴──────────────┴───────────────┘

Sources of technical debt unique to ML:
  ─ Entanglement: changing one feature affects all model outputs
  ─ Undeclared consumers: models silently coupled to upstream data
  ─ Pipeline jungles: spaghetti of glue code
  ─ Dead experimental code paths left in production
  ─ Feedback loops: model output influences future training data
```

### 1.3 ML System Maturity Levels

```
Level 0 — MANUAL
  Training and deployment are manual processes.
  Scientist trains model in Jupyter → IT deploys → forgotten.
  No monitoring. No reproducibility.

Level 1 — ML PIPELINE AUTOMATION
  Automated training pipelines.
  New data triggers retraining.
  Still manual deployment.

Level 2 — CI/CD FOR ML (MLOps)
  Automated training + validation + deployment.
  Model registry with lineage.
  Monitoring + drift detection.
  A/B testing for model updates.

Level 3 — PLATFORM SCALE
  Self-service ML platform.
  Feature store shared across teams.
  Automated retraining on drift.
  Shadow mode + canary deployments.
```

---

## 2. ML System Architecture Fundamentals

### 2.1 End-to-End ML System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  Raw Data → Data Lake (Bronze) → Feature Store → Training Sets  │
│  Sources: events, databases, third-party, labels                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      TRAINING LAYER                             │
│  Experiment Tracking (MLflow) → Training Pipeline (Kubeflow)    │
│  Distributed Training → Model Evaluation → Model Registry       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       SERVING LAYER                             │
│  Real-time Inference (REST/gRPC) │ Batch Inference (Spark)      │
│  Model Server (TorchServe/TFX)   │ A/B Testing Framework        │
│  Shadow Mode │ Canary Deployment │ Feature Retrieval            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    MONITORING LAYER                             │
│  Prediction Monitoring │ Data Drift │ Model Performance         │
│  Feature Drift │ Label Drift │ Business Metric Correlation      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Training vs Serving Skew

One of the most common ML production failures:

```
Training/Serving Skew:

TRAINING TIME                      SERVING TIME
─────────────────────              ─────────────────────
Feature: "days since signup"       Feature: computed differently
  computed from batch SQL join        computed from Redis lookup
  
Age bucketing logic in Python      Age bucketing in Java service
  [0-18] → "youth"                   [0-18] → "youth"
  [18-65] → "adult"   ✅            [18-65] → "adult"  ✅
  [65+] → "senior"                   [65-100] → "senior" ❌ BUG
  
Null handling:                     Null handling:
  fillna(0)                          returns -1

Result: Model trained on one distribution,
        served on a slightly different distribution.
        Silent accuracy degradation.

Prevention:
  ─ Feature Store (single definition for train + serve)
  ─ Transform code shared between training and serving
  ─ Shadow mode: compare training-path vs serving-path features
```

### 2.3 ML System Design Decisions

| Decision | Options | Guidance |
|----------|---------|---------|
| Real-time vs batch inference | REST API vs batch job | < 100ms required → real-time; bulk/daily → batch |
| Feature freshness | Real-time (stream) vs precomputed | User activity features → real-time; demographic → batch |
| Model update frequency | On-schedule vs on-drift | Stable domains → weekly; dynamic (ads, fraud) → daily/hourly |
| A/B test vs canary | Traffic split vs gradual rollout | New model type → A/B; incremental improvement → canary |
| Online vs offline labels | Real-time feedback vs delayed labels | Fraud: delayed (chargebacks); CTR: near-real-time |

---

## 3. Feature Stores

### 3.1 What is a Feature Store

A **Feature Store** is a centralized repository for storing, computing, and serving ML features — eliminating training/serving skew and enabling feature reuse across teams.

```
Feature Store Architecture:

OFFLINE (Training)           ONLINE (Serving)
────────────────────         ──────────────────────
┌─────────────────┐          ┌─────────────────────┐
│  Batch Pipeline │          │  Stream Pipeline     │
│  (Spark / dbt)  │          │  (Flink / Kafka SS)  │
└────────┬────────┘          └──────────┬───────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  Offline Store  │          │   Online Store       │
│ (S3 + Parquet / │          │  (Redis / DynamoDB / │
│  Delta Lake)    │          │   Cassandra)         │
└────────┬────────┘          └──────────┬───────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
              ┌─────────▼──────────┐
              │  Feature Store API  │
              │  point-in-time join │
              │  feature retrieval  │
              └─────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
  ┌──────▼──────┐               ┌──────▼──────┐
  │  ML Training│               │  Model      │
  │  (historical│               │  Serving    │
  │   features) │               │  (real-time)│
  └─────────────┘               └─────────────┘
```

### 3.2 Point-in-Time Correctness

Critical for preventing data leakage in training:

```python
# Point-in-time correct feature retrieval
# For each training example, retrieve feature values
# as they existed AT THE TIME of the label event.
# (Not the current values — that would leak future info)

from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Entity dataframe: user_id + event_timestamp (label time)
entity_df = pd.DataFrame({
    "user_id": ["u1", "u2", "u3"],
    "event_timestamp": [
        datetime(2024, 1, 15),   # Churn date for u1
        datetime(2024, 1, 20),   # Churn date for u2
        datetime(2024, 2, 1),    # Churn date for u3
    ]
})

# Feast retrieves feature values AS OF each event_timestamp
# u1 gets features from before 2024-01-15 only
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_activity:login_count_30d",
        "user_activity:pages_viewed_7d",
        "user_profile:account_age_days",
        "payment_history:failed_payments_90d",
    ]
).to_df()

# Serving: get latest feature values for real-time prediction
online_features = store.get_online_features(
    features=["user_activity:login_count_30d"],
    entity_rows=[{"user_id": "u42"}]
).to_dict()
```

### 3.3 Feature Store: Feast vs Tecton vs Hopsworks

| Feature | Feast (OSS) | Tecton (SaaS) | Hopsworks (OSS) |
|---------|------------|---------------|-----------------|
| **Hosting** | Self-managed | Fully managed | Self-managed / cloud |
| **Online store** | Redis, DynamoDB | Managed | RonDB (MySQL cluster) |
| **Offline store** | S3/BigQuery/Snowflake | Managed | S3/HDFS |
| **Streaming features** | Spark Streaming, Flink | Built-in | Built-in |
| **Feature monitoring** | Basic | Advanced | Advanced |
| **Python SDK** | ✅ | ✅ | ✅ |
| **Best for** | Teams wanting OSS control | Enterprise, managed | Integrated ML platform |

---

## 4. Model Training Infrastructure

### 4.1 Experiment Tracking with MLflow

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("fraud-detection-v3")

with mlflow.start_run(run_name="xgboost-baseline"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "feature_version": "v2.3",
        "training_data_date": "2024-03-01",
    })

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Log metrics
    mlflow.log_metrics({
        "train_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        "val_auc": roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]),
        "val_precision_at_90recall": precision_at_recall(y_val, preds, 0.90),
        "val_false_positive_rate": false_positive_rate(y_val, preds),
    })

    # Log model with input/output schema
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name="fraud-detector",
    )

    # Log feature importance artifact
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("data_profile.html")
```

### 4.2 Distributed Training

```python
# Distributed training with PyTorch DDP on Kubernetes

# training/train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train():
    dist.init_process_group("nccl")  # GPU communication backend
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = MyLargeModel().cuda(rank)
    model = DDP(model, device_ids=[rank])

    # Each process handles a partition of the data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
```

```yaml
# Kubernetes training job (using Kubeflow PyTorchJob)
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: fraud-model-training-v3
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: pytorch
              image: company/ml-training:latest
              resources:
                limits:
                  nvidia.com/gpu: 1
                  memory: 32Gi
    Worker:
      replicas: 7            # 8 total GPUs: 1 master + 7 workers
      template:
        spec:
          containers:
            - name: pytorch
              image: company/ml-training:latest
              resources:
                limits:
                  nvidia.com/gpu: 1
```

### 4.3 Model Evaluation Framework

```python
# Comprehensive model evaluation before promotion

class ModelEvaluator:
    def evaluate(self, model, test_data: pd.DataFrame) -> EvaluationReport:
        report = EvaluationReport()

        # 1. Overall performance
        report.add("auc_roc", roc_auc_score(y_true, y_pred))
        report.add("avg_precision", average_precision_score(y_true, y_pred))

        # 2. Calibration (confidence = actual probability?)
        report.add("brier_score", brier_score_loss(y_true, y_pred))
        report.add_plot("calibration_curve", plot_calibration(y_true, y_pred))

        # 3. Fairness analysis (critical for production)
        for group in ["age_group", "gender", "region"]:
            for subgroup in test_data[group].unique():
                subset = test_data[test_data[group] == subgroup]
                report.add(
                    f"auc_{group}_{subgroup}",
                    roc_auc_score(subset.label, model.predict(subset))
                )

        # 4. Performance vs champion model (must beat by threshold)
        champion_auc = self.get_champion_metric("auc_roc")
        improvement = (report["auc_roc"] - champion_auc) / champion_auc
        report.add("improvement_vs_champion", improvement)

        # 5. Latency under load
        latencies = self.benchmark_inference(model, n=1000)
        report.add("p50_latency_ms", np.percentile(latencies, 50))
        report.add("p99_latency_ms", np.percentile(latencies, 99))

        return report

    def meets_promotion_criteria(self, report: EvaluationReport) -> bool:
        return (
            report["auc_roc"] >= 0.85 and
            report["improvement_vs_champion"] >= 0.01 and
            report["p99_latency_ms"] <= 50 and
            report["brier_score"] <= 0.15 and
            all(  # No group AUC below 0.80 (fairness gate)
                v >= 0.80 for k, v in report.items() if k.startswith("auc_")
            )
        )
```

---

## 5. Model Serving & Inference

### 5.1 Serving Architecture Patterns

```
Real-time Serving Stack:

Client → API Gateway → Model Serving Layer → Feature Store
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              ┌─────▼──┐  ┌──▼───┐  ┌──▼────┐
              │ Model  │  │Model │  │Shadow │
              │  v1.2  │  │ v1.3 │  │ v2.0  │
              │(90%    │  │(10%  │  │(0%    │
              │traffic)│  │canary│  │shadow)│
              └────────┘  └──────┘  └───────┘
                    │
              ┌─────▼──────────────────┐
              │  Prediction Logger     │
              │  (for monitoring +     │
              │   future retraining)   │
              └────────────────────────┘

Model Server options:
  TorchServe       → PyTorch models, dynamic batching
  TensorFlow Serve → TF/Keras models, gRPC
  Triton Inference → Multi-framework, GPU optimization
  BentoML          → Python-first, flexible
  Ray Serve        → Distributed serving, Python
  vLLM             → LLM-optimized, PagedAttention
```

### 5.2 Inference Optimization

```
Inference Optimization Techniques:

QUANTIZATION
  FP32 → INT8 (4x smaller, 2-4x faster, ~1% accuracy loss)
  
  import torch.quantization
  model_quantized = torch.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
  )

BATCHING
  Dynamic batching: wait up to 10ms to collect requests,
  then process as single batch for GPU efficiency
  
  max_batch_size: 64
  max_batch_delay_microseconds: 10000
  
  Throughput: 100 RPS single → 2000 RPS batched (20x)

CACHING
  Cache predictions for identical inputs
  (effective for repeated lookups, e.g., product recommendations)
  
  cache_key = hash(feature_vector)
  if redis.exists(cache_key):
      return redis.get(cache_key)

HARDWARE ACCELERATION
  GPU: 10-100x faster than CPU for neural networks
  GPU sharding: split large models across multiple GPUs
  
  Tensor parallelism: split matrix multiply across GPUs
  Pipeline parallelism: different layers on different GPUs

KNOWLEDGE DISTILLATION
  Train small "student" model to mimic large "teacher" model
  Student: 10x smaller, 5x faster, ~3% accuracy drop
  Used by: DistilBERT (40% smaller, 60% faster than BERT)
```

### 5.3 Online vs Batch vs Streaming Inference

```
Inference Modes:

ONLINE (Real-time)
  ─ Request/response API call
  ─ Latency: < 100ms
  ─ Use cases: fraud detection, recommendations, search ranking
  ─ Infrastructure: REST API + model server + feature store

BATCH
  ─ Process large dataset offline
  ─ Latency: hours
  ─ Use cases: churn scoring, risk scoring, email targeting
  ─ Infrastructure: Spark job, scheduled Airflow DAG
  ─ Write predictions to DB for downstream consumption

STREAMING
  ─ Inference on event stream
  ─ Latency: seconds
  ─ Use cases: anomaly detection, real-time personalization
  ─ Infrastructure: Flink / Kafka Streams consumer
```

---

## 6. MLOps & CI/CD for ML

### 6.1 ML Pipeline Automation

```
MLOps CI/CD Pipeline:

Code Change / Data Change / Schedule
              │
    ┌─────────▼──────────┐
    │   Data Validation   │ ─── fail → alert data team
    │   (Great Expectations│
    │   / TFX Validate)   │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Feature Pipeline   │ ─── compute features from raw data
    │  (Spark / dbt)      │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   Model Training    │ ─── distributed if large model
    │   (Kubeflow /       │
    │    SageMaker)       │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Model Evaluation   │ ─── fail → block promotion, alert
    │  (perf + fairness   │
    │   + latency gates)  │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Model Registry     │ ─── version, lineage, metadata
    │  (MLflow / W&B)     │
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Shadow Deployment  │ ─── 0% traffic, log predictions
    └─────────┬──────────┘
              │ (manual approval)
    ┌─────────▼──────────┐
    │  Canary (10%)       │ ─── monitor business metrics
    └─────────┬──────────┘
              │ (auto if metrics OK)
    ┌─────────▼──────────┐
    │  Full Rollout       │
    └────────────────────┘
```

### 6.2 Model Registry

```python
# MLflow Model Registry workflow

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from training run
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name="fraud-detector"
)
version = result.version  # e.g., "42"

# Add metadata
client.update_model_version(
    name="fraud-detector",
    version=version,
    description="XGBoost v3 trained on 2024-Q1 data. AUC=0.924. Fairness-checked."
)
client.set_model_version_tag(
    name="fraud-detector", version=version,
    key="training_data_cutoff", value="2024-03-15"
)
client.set_model_version_tag(
    name="fraud-detector", version=version,
    key="approved_by", value="ml-review-board"
)

# Promote through stages
client.transition_model_version_stage(
    name="fraud-detector", version=version, stage="Staging"
)
# After validation...
client.transition_model_version_stage(
    name="fraud-detector", version=version, stage="Production"
)
```

---

## 7. LLM Systems Architecture

### 7.1 LLM Production Architecture

```
LLM Production System:

┌──────────────────────────────────────────────────────────────┐
│                      Client Layer                            │
│   Web / Mobile / API Consumers                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                  LLM Gateway / Router                        │
│  ─ Authentication / rate limiting                            │
│  ─ Model routing (Claude / GPT-4 / Llama based on task)      │
│  ─ PII detection & redaction (before sending to LLM)         │
│  ─ Prompt injection detection                                │
│  ─ Response caching (semantic cache)                         │
│  ─ Cost tracking (token counting per user/team)              │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
   ┌─────▼─────┐ ┌─────▼─────┐ ┌────▼──────┐
   │ Hosted LLM│ │Self-hosted│ │Fine-tuned │
   │(Anthropic │ │  (vLLM +  │ │  model    │
   │  / OpenAI)│ │  Llama 3) │ │  (LoRA)   │
   └───────────┘ └───────────┘ └───────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                  Supporting Services                         │
│  Vector DB (embeddings) │ Prompt templates │ Tool registry   │
│  Conversation memory    │ Guardrails       │ Eval framework  │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 LLM Inference Optimization

```
LLM Inference Challenges & Solutions:

CHALLENGE: Latency
  Token generation is sequential (autoregressive)
  100-token response at 30 tokens/sec = 3.3 seconds

SOLUTIONS:
  Speculative decoding:
    ─ Small "draft" model generates candidate tokens fast
    ─ Large model verifies in parallel
    ─ 2-3x speedup for long generations

  KV Cache:
    ─ Reuse key/value attention matrices across requests
    ─ Critical for chat with long context
    ─ vLLM's PagedAttention: GPU memory like OS virtual memory
    ─ 2-4x throughput improvement

  Batching (continuous / iteration-level):
    ─ Group requests of similar length
    ─ Add new requests without waiting for batch to complete
    ─ vLLM, TGI (Text Generation Inference) implement this

  Quantization:
    ─ FP16 → INT4 (GGUF / AWQ / GPTQ)
    ─ Run Llama 3 70B on 2x A100 instead of 4x
    ─ ~5% perplexity increase, 2-4x memory reduction

CHALLENGE: Cost (token costs dominate at scale)
SOLUTIONS:
  Prompt caching (Anthropic, OpenAI):
    ─ Cache prefix of long system prompts
    ─ 90% cost reduction on cached tokens

  Semantic caching:
    ─ Cache LLM responses for similar queries
    ─ "What's the weather?" ≈ "Tell me today's weather"
    ─ GPTCache, Redis with embedding similarity

  Model routing:
    ─ Simple queries → small cheap model (GPT-4o-mini)
    ─ Complex reasoning → large capable model (Claude 3.5)
    ─ RouteLLM, LiteLLM for routing
```

### 7.3 Fine-Tuning Architecture

```
Fine-Tuning Strategies:

FULL FINE-TUNING
  Update all weights. Expensive. Catastrophic forgetting risk.
  Use: Domain-specific tasks with large proprietary dataset.
  Cost: High (need same GPU as pretraining)

LoRA (Low-Rank Adaptation)
  Freeze base weights. Learn small rank-decomposition matrices.
  Parameters: 0.1-1% of full model. Same quality for most tasks.
  Use: Task-specific adapters, cost-effective customization.
  
  W_new = W_base + α * (A × B)   [A: d×r, B: r×d, r << d]

QLORA
  LoRA + 4-bit quantized base model.
  Fine-tune 65B model on single 48GB GPU.
  Minimal quality loss vs full LoRA.

RAFT (Retrieval-Augmented Fine-Tuning)
  Fine-tune on (question, retrieved context, answer) triples.
  Model learns to use retrieved context effectively.
  Better than RAG alone for specialized domains.

Adapter Pattern (production):
  ┌─────────────────────────┐
  │    Base Model (frozen)   │
  │    Llama 3 70B           │
  └──────────────┬──────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
  ┌──▼──┐    ┌──▼──┐    ┌──▼──┐
  │LoRA │    │LoRA │    │LoRA │
  │Legal│    │ Med.│    │Code │
  └─────┘    └─────┘    └─────┘
  Load adapters dynamically per request based on task type
```

---

## 8. RAG Architecture

### 8.1 RAG System Design

**Retrieval-Augmented Generation (RAG)** grounds LLM responses in retrieved documents, reducing hallucinations and enabling knowledge updates without retraining.

```
RAG Pipeline:

INDEXING (offline)
  Documents → Chunking → Embedding Model → Vector Store
  
  ┌──────────┐   ┌────────────┐   ┌───────────────┐
  │   PDFs   │   │  Chunking  │   │   Embedding   │
  │   Docs   ├──►│(512 tokens │──►│  Model        │
  │   HTML   │   │ with 50    │   │(text-embed-3) │
  │  Tables  │   │ overlap)   │   └───────┬───────┘
  └──────────┘   └────────────┘           │
                                          ▼
                                  ┌───────────────┐
                                  │  Vector Store  │
                                  │  (Pinecone /   │
                                  │  pgvector)     │
                                  └───────────────┘

RETRIEVAL (online, per query)
  User Query → Embed Query → Vector Search → Rerank → Augment Prompt

  query_embedding = embed("How do I reset my password?")
  candidates = vector_db.search(query_embedding, top_k=20)
  reranked = cross_encoder.rerank(query, candidates, top_k=5)
  
  prompt = f"""
  Answer using only the context below:
  
  Context:
  {format_chunks(reranked)}
  
  Question: {user_query}
  """
  
  response = llm.generate(prompt)
```

### 8.2 Advanced RAG Techniques

```
Advanced RAG Strategies:

HYBRID SEARCH (keyword + semantic)
  ─ BM25 (keyword/TF-IDF) + vector search in parallel
  ─ Reciprocal rank fusion (RRF) to merge results
  ─ 15-20% better recall than pure vector search
  
  sparse_results  = bm25_search(query, top_k=20)
  dense_results   = vector_search(embed(query), top_k=20)
  merged_results  = reciprocal_rank_fusion(sparse_results, dense_results)

HyDE (Hypothetical Document Embedding)
  ─ LLM generates hypothetical answer to query
  ─ Embed the hypothesis (not the query)
  ─ Retrieve docs similar to what a good answer looks like
  ─ Better for sparse or unusual queries

RAPTOR (Recursive Abstractive Processing)
  ─ Build tree of summaries from document chunks
  ─ Bottom: raw chunks. Top: high-level summaries
  ─ Search all levels, combine results
  ─ Better for long-document Q&A

SELF-RAG
  ─ LLM decides whether to retrieve (not always needed)
  ─ LLM critiques its own response, retrieves more if needed
  ─ Better precision/recall trade-off

PARENT-CHILD CHUNKING
  ─ Index small chunks (high precision retrieval)
  ─ Return parent (larger) chunk to LLM (more context)
  ─ e.g., retrieve 128-token child, return 512-token parent
```

### 8.3 RAG Evaluation

```python
# RAG evaluation with RAGAS framework
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Is answer grounded in retrieved docs?
    answer_relevancy,    # Is answer relevant to the question?
    context_precision,   # Are retrieved docs relevant?
    context_recall,      # Are all relevant docs retrieved?
)

result = evaluate(
    dataset=eval_dataset,  # question, context, answer, ground_truth
    metrics=[
        faithfulness,       # Hallucination check (0-1, higher=better)
        answer_relevancy,   # Off-topic check (0-1)
        context_precision,  # Retrieval quality (0-1)
        context_recall,     # Retrieval completeness (0-1)
    ]
)

# Production: track these metrics in Grafana alongside latency/cost
# Alert if faithfulness drops below 0.8 (hallucination spike)
```

---

## 9. Vector Databases

### 9.1 Vector Database Comparison

| Database | Hosting | Filtering | Scale | Best For |
|----------|---------|-----------|-------|---------|
| **Pinecone** | SaaS only | Good | Very large | Production RAG, managed |
| **Weaviate** | OSS + SaaS | Excellent | Large | Hybrid search, GraphQL API |
| **Qdrant** | OSS + SaaS | Excellent | Large | High-performance, Rust-based |
| **Chroma** | OSS | Basic | Small-medium | Development, prototyping |
| **pgvector** | PostgreSQL ext. | Excellent | Medium | Existing Postgres users |
| **Milvus** | OSS | Good | Very large | Enterprise, Kubernetes |
| **Redis (VSS)** | OSS + SaaS | Good | Medium | Low-latency, existing Redis |

### 9.2 Vector Indexing Algorithms

```
HNSW (Hierarchical Navigable Small World) — most popular:

  ─ Graph-based approximate nearest neighbor search
  ─ Multiple layers: top layers are coarse, bottom is fine
  ─ O(log n) query time vs O(n) brute force
  ─ Tuning parameters:
    M (connections per node): 16-64, higher = better recall + more memory
    ef_construction: 100-500, higher = better index quality + slower build
    ef (search): 50-200, higher = better recall + slower query

IVF (Inverted File Index) — good for very large:
  ─ Cluster vectors into nlist clusters at build time
  ─ At query time, search only nprobe nearest clusters
  ─ Good memory efficiency at billion+ scale
  ─ Used in FAISS (Facebook AI Similarity Search)

ANNOY (Approximate Nearest Neighbors Oh Yeah) — Spotify's:
  ─ Multiple random projection trees
  ─ Good for static indices (no updates)
  ─ Low memory footprint

Tradeoffs:
  Recall  vs  Speed  vs  Memory
  ─────────────────────────────
  HNSW:  high recall, fast query, high memory
  IVF:   tunable recall, fast, memory efficient
  Flat:  100% recall (exact), slow, memory efficient
```

### 9.3 Embeddings Architecture

```python
# Embedding pipeline design for production

class EmbeddingPipeline:
    def __init__(self):
        # Separate models for different content types
        self.text_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.code_model = CodeEmbeddings(model="voyage-code-2")
        self.image_model = CLIPEmbeddings()

    def embed_document(self, doc: Document) -> list[EmbeddedChunk]:
        chunks = self.chunk(doc)
        embedded = []
        for chunk in chunks:
            embedding = self.get_model(chunk.type).embed(chunk.text)
            embedded.append(EmbeddedChunk(
                id=f"{doc.id}-{chunk.index}",
                text=chunk.text,
                embedding=embedding,
                metadata={
                    "doc_id": doc.id,
                    "source": doc.source,
                    "created_at": doc.created_at,
                    "content_type": chunk.type,
                    # Filterable metadata (stored in vector DB)
                    "tenant_id": doc.tenant_id,
                    "doc_type": doc.doc_type,
                    "language": doc.language,
                }
            ))
        return embedded

    def chunk(self, doc: Document) -> list[Chunk]:
        # Semantic chunking: split at meaningful boundaries
        # not arbitrary token counts
        return semantic_splitter.split(
            doc.content,
            min_chunk_size=100,
            max_chunk_size=512,
            overlap=50
        )
```

---

## 10. Observability for ML Systems

### 10.1 ML Monitoring Dimensions

```
ML Monitoring Pyramid:

BUSINESS METRICS (lagging, most important)
  ─ Revenue impact of model
  ─ User engagement change
  ─ Business KPI delta vs baseline
  ─ Alert: metric drops > 5% week-over-week

MODEL PERFORMANCE (leading, need labels)
  ─ AUC, precision, recall, F1
  ─ Requires ground truth labels (may lag hours/days)
  ─ Alert: AUC drops below threshold

PREDICTION DISTRIBUTION (immediate)
  ─ Score distribution shift (histogram comparison)
  ─ Prediction volume anomalies
  ─ Class distribution changes
  ─ Alert: KL divergence > threshold

FEATURE DRIFT (immediate, no labels needed)
  ─ Statistical tests: PSI, KS test, Jensen-Shannon
  ─ Null rate changes
  ─ Value range violations
  ─ Alert: PSI > 0.2 for any feature

INFRASTRUCTURE (always on)
  ─ Inference latency (P50, P99)
  ─ Throughput (requests/sec)
  ─ GPU utilization, memory
  ─ Error rates
```

### 10.2 Drift Detection

```python
# Production drift detection

import numpy as np
from scipy import stats

class DriftDetector:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        self.psi_threshold = 0.2      # Significant drift
        self.ks_p_value_threshold = 0.05

    def calculate_psi(self, feature: str, production: pd.Series) -> float:
        """Population Stability Index — standard in finance/ML monitoring"""
        ref = self.reference[feature].dropna()
        prod = production.dropna()

        # Bin based on reference distribution
        bins = np.percentile(ref, np.arange(0, 110, 10))
        bins = np.unique(bins)

        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref)
        prod_pct = np.histogram(prod, bins=bins)[0] / len(prod)

        # Avoid log(0)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        prod_pct = np.where(prod_pct == 0, 0.0001, prod_pct)

        psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))

        # PSI < 0.1: no drift, 0.1-0.2: slight drift, > 0.2: significant
        return psi

    def kolmogorov_smirnov_test(self, feature: str, production: pd.Series):
        """Non-parametric test for distribution shift"""
        ref = self.reference[feature].dropna()
        statistic, p_value = stats.ks_2samp(ref, production.dropna())
        return {"statistic": statistic, "p_value": p_value,
                "drift_detected": p_value < self.ks_p_value_threshold}
```

### 10.3 Prediction Logging & Feedback Loops

```
Prediction Logging Architecture:

Request               Model              Logger             Storage
──────                ─────              ──────             ───────
input features  ──►  predict()  ──►  async log  ──►  Kafka topic
                        │                │                  │
                        │           prediction_id          │
                      output  ──►   + features   ──►  ────►│
                                    + timestamp            │
                                                    ┌──────▼──────┐
                                                    │  Data Lake  │
                                                    │ (for future │
                                                    │ retraining) │
                                                    └──────┬──────┘
                                                           │
                    Label arrives (hours/days later):      │
                                                    ┌──────▼──────┐
User action  ──►  Label service  ──►  Join with  ──►│  Labeled    │
(e.g., fraud                         prediction_id  │  Dataset    │
 confirmed)                                         └─────────────┘
                                                           │
                                                    Retraining trigger
```

---

## 11. Practical Examples

### 11.1 LLMOps Reference Stack

```yaml
# LLM Production Platform Stack (2024)

model_providers:
  external:
    - anthropic_claude          # Complex reasoning, long context
    - openai_gpt4o              # Vision, general purpose
    - cohere                    # Enterprise, RAG-optimized
  self_hosted:
    - llama_3_70b_on_vllm       # Privacy, cost at scale
    - mistral_on_triton         # Code tasks

llm_gateway:
  - litellm                     # Unified API + routing
  - openrouter                  # Cost-optimized routing

rag_infrastructure:
  chunking: langchain_text_splitters
  embedding_model: text-embedding-3-large
  vector_db: qdrant              # OSS, high performance
  reranking: cohere_rerank       # Cross-encoder reranking

prompt_management:
  - langfuse                    # Prompt versioning + A/B testing

evaluation:
  - ragas                       # RAG evaluation metrics
  - langfuse_evals              # LLM-as-judge scoring
  - promptfoo                   # Regression testing for prompts

observability:
  - langfuse                    # Traces, costs, latency per call
  - helicone                    # Token usage, cost tracking

guardrails:
  - guardrails_ai               # Output validation
  - presidio                    # PII detection/redaction
  - llm_guard                   # Prompt injection detection

fine_tuning:
  - axolotl                     # LoRA/QLoRA training
  - unsloth                     # 2x faster LoRA training
  - modal_labs                  # Serverless GPU training
```

### 11.2 ML Platform for Classic ML (non-LLM)

```yaml
# Classic ML Platform Stack

data_layer:
  feature_store: feast
  offline_store: delta_lake_on_s3
  online_store: redis_cluster
  label_store: postgresql

training:
  orchestration: kubeflow_pipelines
  distributed_training: pytorch_ddp
  hyperparameter_tuning: optuna
  experiment_tracking: mlflow

model_management:
  registry: mlflow_model_registry
  packaging: bentoml
  validation: great_expectations + custom_eval_framework

serving:
  real_time: bentoml_on_kubernetes
  batch: spark_on_emr
  a_b_testing: custom_gateway_with_weights
  shadow_mode: mirror_traffic_with_logger

monitoring:
  data_drift: evidently_ai
  model_performance: custom_grafana_dashboards
  prediction_logging: kafka + s3
  alerting: prometheus_alertmanager → pagerduty
  labeling_pipeline: label_studio + airflow
```

### 11.3 ML System Design Interview: Fraud Detection

```
Problem: Design a real-time fraud detection system
  Requirements: < 200ms P99, 100K TPS, 99.9% recall

ARCHITECTURE DECISIONS:

Data:
  ─ Features: transaction amount, merchant, velocity
    (# transactions in last 1h, 24h), user history
  ─ Feature freshness: velocity features must be real-time
  ─ Labels: from chargebacks (delayed ~30 days)

Feature Store:
  ─ Offline: batch features (user profile) in Delta Lake
  ─ Online: velocity counters in Redis (TTL-based)
  ─ Real-time: Flink job updating Redis on each transaction

Model:
  ─ XGBoost (fast inference, good feature importance)
  ─ Ensemble with neural net for card-not-present
  ─ Threshold: calibrated for 99.9% recall (minimize miss fraud)

Serving:
  ─ gRPC API (lower latency than REST for structured data)
  ─ Feature retrieval: Redis lookup in parallel (~5ms)
  ─ Inference: XGBoost in-process (~1ms)
  ─ Total P99 target: < 20ms (well within 200ms SLA)

Model updates:
  ─ Weekly retraining on new labeled data
  ─ Shadow mode: new model runs in parallel, predictions logged
  ─ Champion/challenger: 10% traffic to challenger
  ─ Auto-promote if challenger AUC > champion + 0.5%

Monitoring:
  ─ Business: fraud rate, false positive rate (merchant friction)
  ─ Model: score distribution, prediction volume
  ─ Features: null rates, value distributions
  ─ Feedback loop: chargeback data → label store → retraining
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **ML Technical Debt** | 90% of ML code is infra, not model — invest accordingly |
| **Training/Serving Skew** | Feature Store eliminates the #1 source of silent ML bugs |
| **Feature Store** | Single feature definition shared by training and serving — use Feast or Tecton |
| **MLOps** | Same CI/CD discipline as software, adapted for model+data+code versioning |
| **LLM Gateway** | Every production LLM deployment needs routing, caching, rate limiting, PII scrubbing |
| **RAG** | Hybrid search (BM25 + vector) + reranking beats pure vector search |
| **LoRA/QLoRA** | Fine-tune LLMs at 1% of full parameter cost — enough for most production tasks |
| **vLLM / PagedAttention** | Essential for LLM serving throughput — 5-20x improvement over naive serving |
| **Drift Detection** | PSI + KS tests on features; monitor prediction distribution before labels arrive |
| **Feedback Loops** | Log predictions + join with delayed labels → continuous retraining pipeline |
