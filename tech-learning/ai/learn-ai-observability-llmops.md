# AI Observability & LLM Operations: Complete Guide

## Table of Contents
1. [Introduction to LLM Ops](#introduction-to-llm-ops)
2. [Logging and Tracing](#logging-and-tracing)
3. [LangSmith Integration](#langsmith-integration)
4. [Weights & Biases Integration](#weights--biases-integration)
5. [LLM Metrics](#llm-metrics)
6. [Evaluation and Benchmarks](#evaluation-and-benchmarks)
7. [Guardrails and Safety](#guardrails-and-safety)
8. [Cost and Latency Optimization](#cost-and-latency-optimization)
9. [Drift and Quality Monitoring](#drift-and-quality-monitoring)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)
12. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
13. [Production Considerations](#production-considerations)
14. [References and Further Reading](#references-and-further-reading)

---

## Introduction to LLM Ops

**LLM Ops** extends MLOps to production LLM systems: observability, evaluation, safety, cost control, and continuous improvement. LLMs introduce unique challenges: non-determinism, prompt dependency, high token cost, and "soft" quality metrics (relevance, helpfulness) that are harder to automate than accuracy.

### Conceptual Foundation: Why LLM Ops Differs

Traditional ML deploys a fixed model with deterministic input→output. LLMs are **generative** and **contextual**: the same prompt can yield different outputs; small prompt changes can drastically alter behavior. Observability must capture the full chain (retrieval → prompt assembly → generation → post-processing) and link outputs to specific prompt versions and model configs.

### Key Differences from Traditional ML Ops

| Traditional ML | LLM Ops |
|----------------|---------|
| Fixed input → output | Prompt + context → variable output |
| Metric: accuracy, F1 | Metric: relevance, safety, latency, cost |
| Model versioning | Model + prompt versioning |
| Drift: input distribution | Drift: output quality, user behavior |
| Deterministic | Non-deterministic (temperature > 0) |
| Single model per task | Multi-model routing, fallbacks |

### LLM Ops Stack

```
Observability (tracing, metrics) → Evaluation (evals, benchmarks) 
    → Guardrails (safety, PII) → Cost/Latency optimization → Monitoring (drift)
```

---

## Logging and Tracing

### Request-Level Logging

```python
import uuid
import time
import hashlib

def logged_llm_call(llm, prompt, **kwargs):
    request_id = str(uuid.uuid4())
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]  # For dedup
    start = time.time()
    try:
        response = llm.generate(prompt, **kwargs)
        latency_ms = (time.time() - start) * 1000
        log_event({
            "request_id": request_id,
            "prompt_hash": prompt_hash,
            "prompt_tokens": count_tokens(prompt),
            "completion_tokens": count_tokens(response),
            "latency_ms": latency_ms,
            "model": kwargs.get("model", "default"),
            "status": "success",
        })
        return response
    except Exception as e:
        log_event({"request_id": request_id, "status": "error", "error": str(e)})
        raise
```

### Distributed Tracing (OpenTelemetry)

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("llm-app", "1.0")

def traced_llm_call(prompt):
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("prompt.length", len(prompt))
        try:
            result = llm.generate(prompt)
            span.set_attribute("response.length", len(result))
            span.set_attribute("model", "gpt-4")
            return result
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

### Structured Logging for RAG

```python
def logged_rag_query(query, retriever, llm):
    trace_id = str(uuid.uuid4())
    t0 = time.time()
    docs = retriever.get_relevant_documents(query)
    retrieve_ms = (time.time() - t0) * 1000
    log("retrieved_docs", trace_id=trace_id, count=len(docs),
        top_scores=[d.metadata.get("score") for d in docs[:3]])
    t0 = time.time()
    response = llm.generate(build_prompt(query, docs))
    generate_ms = (time.time() - t0) * 1000
    log("rag_response", trace_id=trace_id, retrieve_ms=retrieve_ms, generate_ms=generate_ms,
        prompt_tokens=count_tokens(build_prompt(query, docs)), completion_tokens=count_tokens(response))
    return response
```

---

## LangSmith Integration

**LangSmith** (LangChain) provides tracing, datasets, evaluators, and monitoring. Auto-instruments LangChain/LangGraph; works with custom chains via `@traceable`.

### Setup and Tracing

```python
# pip install langsmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-llm-project"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Runs auto-traced to LangSmith dashboard
llm = ChatOpenAI(model="gpt-4")
chain = ChatPromptTemplate.from_messages([("user", "{query}")]) | llm | StrOutputParser()
response = chain.invoke({"query": "What is the capital of France?"})
```

### Datasets and Evaluators

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()
# Create dataset: client.create_dataset(), client.create_examples()

def relevance_score(run, example):
    output = run.outputs.get("output", "")
    expected = example.outputs.get("expected", "")
    return {"score": compute_similarity(output, expected)}  # 0-1

results = evaluate(chain, data="my-eval-set", evaluators=[relevance_score], experiment_prefix="eval-v1")
```

### Production Monitoring

```python
# Add tags for filtering: env, version, feature
chain.invoke({"query": "..."}, config={"tags": ["prod", "v2.1"], "run_name": "prod-request"})
```

---

## Weights & Biases Integration

**Weights & Biases (W&B)** offers experiment tracking, prompt versioning, eval tables, and cost tracking.

### W&B for LLM Evals and Tables

```python
# pip install wandb
import wandb

run = wandb.init(project="llm-ops", job_type="eval")

# Eval table: inputs, outputs, scores
eval_table = wandb.Table(columns=["query", "expected", "output", "relevance", "latency_ms", "cost"])
for row in eval_results:
    eval_table.add_data(row["query"], row["expected"], row["output"],
                        row["relevance_score"], row["latency_ms"], row["cost"])
wandb.log({"eval_results": eval_table})
# Filter, sort, compare in W&B UI
```

### W&B Prompts and LangChain

```python
# Log prompt versions for A/B comparison
wandb.log({"prompt_v1": "You are helpful. Answer concisely.", "prompt_v2": "You are an expert..."})

# W&B callbacks with LangChain
from langchain.callbacks import wandb_tracer
chain.invoke({"query": "..."}, config={"callbacks": [wandb_tracer.WandbTracer()]})
```

---

## LLM Metrics

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Latency (p50, p95, p99)** | Response time | < 2s for chat |
| **Tokens/sec** | Throughput | Higher for batch |
| **Error rate** | % failed requests | < 0.1% |
| **Cost per request** | $ per 1K tokens | Track by model |
| **Cache hit rate** | Semantic cache | Reduce cost |
| **User satisfaction** | Thumbs up/down | Optional |

### Token Counting

Different models use different tokenizers; mismatched counts lead to wrong cost estimates.

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Cost calculation (update rates per provider pricing)
def cost_estimate(prompt_tokens, completion_tokens, model="gpt-4"):
    rates = {"gpt-4": (0.03, 0.06), "gpt-3.5-turbo": (0.0005, 0.0015)}
    in_rate, out_rate = rates.get(model, (0.01, 0.03))
    return (prompt_tokens/1000)*in_rate + (completion_tokens/1000)*out_rate
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

llm_requests = Counter("llm_requests_total", "Total LLM requests", ["model", "status"])
llm_latency = Histogram("llm_latency_seconds", "LLM latency", ["model"])
llm_tokens = Counter("llm_tokens_total", "Total tokens", ["model", "type"])  # type=prompt|completion

def instrumented_call(model, prompt, **kwargs):
    start = time.time()
    try:
        result = llm.generate(prompt, model=model, **kwargs)
        llm_requests.labels(model=model, status="success").inc()
        llm_latency.labels(model=model).observe(time.time() - start)
        llm_tokens.labels(model=model, type="prompt").inc(count_tokens(prompt))
        llm_tokens.labels(model=model, type="completion").inc(count_tokens(result))
        return result
    except:
        llm_requests.labels(model=model, status="error").inc()
        raise
```

---

## Evaluation and Benchmarks

### LLM Evals

Evaluate outputs on: correctness, relevance, safety, coherence.

### Custom Eval Pipeline

```python
def eval_llm(model_output, ground_truth, criteria=["correctness", "relevance"]):
    scores = {}
    for c in criteria:
        prompt = f"""
        Evaluate the model output on {c}.
        Model output: {model_output}
        Ground truth: {ground_truth}
        Score 1-5 with brief justification.
        """
        eval_result = llm.generate(prompt)
        scores[c] = parse_score(eval_result)
    return scores
```

### Benchmark Suites

| Benchmark | Focus | Use Case |
|-----------|-------|----------|
| **MMLU** | Broad knowledge | General capability |
| **HumanEval** | Code generation | Code models |
| **TruthfulQA** | Factuality | Truthfulness |
| **HELM** | Holistic evaluation | Multi-metric |
| **MT-Bench** | Chat quality | Chat models |
| **SWE-bench** | Real-world code fixes | Code agents |

### Regression Testing

```python
def regression_test(model, eval_set, baseline_scores):
    """Run eval set, compare to baseline"""
    new_scores = run_evals(model, eval_set)
    for metric in baseline_scores:
        if new_scores[metric] < baseline_scores[metric] * 0.95:
            alert(f"Regression: {metric} dropped from {baseline_scores[metric]} to {new_scores[metric]}")
```

---

## Guardrails and Safety

### Output Filters

```python
# Block PII in output
def filter_pii(text):
    import re
    # Simple: detect patterns
    patterns = [
        r'\b\d{16}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text)
    return text

# Use LLM to detect and redact
def llm_redact_pii(text):
    prompt = f"Redact PII (emails, SSN, credit cards) from: {text}. Return redacted text only."
    return llm.generate(prompt)
```

### Guardrails AI

```python
# pip install guardrails-ai
from guardrails import Guard
from guardrails.hub import ToxicContent, PII

guard = Guard().use(
    ToxicContent(),
    PII(pii_entities=["EMAIL", "PHONE_NUMBER"], redact=True)
)
validated_output = guard.validate(llm_output)
```

### Input Validation

```python
def validate_prompt(prompt, max_length=10000, blocked_patterns=None):
    if len(prompt) > max_length:
        raise ValueError("Prompt too long")
    for pattern in blocked_patterns or []:
        if re.search(pattern, prompt, re.I):
            raise ValueError("Blocked content in prompt")
    return True
```

### Jailbreak Detection

```python
def detect_jailbreak_attempt(prompt):
    # Heuristics + model-based
    suspicious = ["ignore previous", "pretend you are", "DAN mode", "no restrictions"]
    if any(s in prompt.lower() for s in suspicious):
        return True
    # Optional: small classifier fine-tuned on jailbreak examples
    return jailbreak_detector(prompt)
```

---

## Cost and Latency Optimization

### Caching

**Exact cache**: Hash prompt → hit only on identical prompts. **Semantic cache**: Embedding similarity → hit on paraphrases (e.g., GPTCache, LangChain's InMemoryCache with embeddings).

```python
# Semantic cache: similar prompts → reuse response (threshold cosine similarity)
def cached_llm_call(prompt, cache, embedding_model, similarity_threshold=0.95):
    emb = embedding_model.encode(prompt)
    cached = cache.find_nearest(emb, threshold=similarity_threshold)
    if cached:
        return cached["response"]
    result = llm.generate(prompt)
    cache.set(emb, {"response": result})
    return result
```

### Model Routing

```python
def route_request(prompt, intent):
    if intent == "simple_qa":
        return "gpt-3.5-turbo"  # Cheaper
    elif intent == "complex_reasoning":
        return "gpt-4"
    return "gpt-4-mini"
```

### Prompt Optimization

- Shorter system prompts
- Few-shot instead of long context when possible
- Output format constraints to reduce tokens

### Batching

```python
# Batch multiple requests for higher throughput
def batch_llm_calls(prompts, batch_size=10):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        # Use batch API if available
        batch_results = llm.generate_batch(batch)
        results.extend(batch_results)
    return results
```

---

## Drift and Quality Monitoring

### Output Quality Drift

```python
# Track distribution of output lengths, sentiment, topic
def monitor_output_quality(outputs, window=1000):
    recent = outputs[-window:]
    length_dist = [len(o) for o in recent]
    if np.mean(length_dist) < historical_mean * 0.7:
        alert("Output length dropped significantly")
```

### User Feedback Loop

```python
def collect_feedback(request_id, thumbs_up):
    store_feedback(request_id, thumbs_up)
    # Aggregate by prompt template, model version
    # Trigger retraining or prompt update if satisfaction drops
```

### A/B Testing Prompts

```python
def ab_test_prompts(prompt_a, prompt_b, traffic_split=0.5):
    for request in incoming:
        prompt = prompt_a if random() < traffic_split else prompt_b
        result = llm.generate(prompt)
        log_experiment(request, prompt_version="A" or "B", result, feedback)
```

---

## Practical Examples

### Example 1: Full Instrumentation

```python
def instrumented_rag_chain(query, retriever, llm):
    trace_id = str(uuid.uuid4())
    metrics = {"retrieve_ms": 0, "generate_ms": 0, "tokens": 0}
    
    t0 = time.time()
    docs = retriever.get_relevant_documents(query)
    metrics["retrieve_ms"] = (time.time() - t0) * 1000
    
    t0 = time.time()
    response = llm.generate(build_prompt(query, docs))
    metrics["generate_ms"] = (time.time() - t0) * 1000
    metrics["tokens"] = count_tokens(response)
    
    log_event({"trace_id": trace_id, "query": query[:100], **metrics})
    return response
```

### Example 2: Guardrails Pipeline

```python
def safe_llm_pipeline(prompt):
    if detect_jailbreak_attempt(prompt):
        return {"error": "Request blocked"}
    response = llm.generate(prompt)
    response = guardrails.validate(response)
    response = filter_pii(response)
    return response
```

### Example 3: Cost Dashboard Queries

```python
# Example: aggregate cost by day, model
# SELECT date, model, SUM(cost) FROM llm_logs GROUP BY date, model
```

---

## Best Practices

1. **Log everything**: request_id, prompt hash, tokens, latency, model, status
2. **Trace end-to-end**: from user request through RAG/agent to response
3. **Eval before deploy**: run benchmark on new prompts/models
4. **Guardrails**: PII, toxicity, jailbreak detection
5. **Cost alerts**: Set budgets, alert on spikes
6. **Version prompts**: Track prompt changes with model versions
7. **Human review**: Sample for high-stakes applications

---

## Common Pitfalls and Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| **Missing traces** | Env vars not set, wrong project | Set LANGCHAIN_* or OTLP endpoints; check exporter config |
| **High cost spikes** | Long context, no cache, wrong model | Enable semantic cache; route simple queries to cheaper models |
| **P99 latency spikes** | Cold start, rate limits, long prompts | Warm instances; implement backoff; truncate context |
| **Eval inconsistency** | LLM-as-judge variance | Use temperature=0; run multiple runs; use rubric-based evals |
| **Log volume explosion** | Logging full prompts | Log prompt hash + length; sample full prompts for debug |
| **Metrics mismatch** | Different tokenizers | Use tiktoken for OpenAI; provider-specific for others |

### Debugging Slow RAG

```python
# Isolate latency: retrieval vs generation
t0 = time.time()
docs = retriever.get_relevant_documents(query)
retrieve_ms = (time.time() - t0) * 1000
t0 = time.time()
response = llm.generate(build_prompt(query, docs))
generate_ms = (time.time() - t0) * 1000
# If retrieve_ms >> generate_ms: optimize embedding/index; if opposite: use smaller model or cache
```

---

## Production Considerations

- **Sampling**: Don't log 100% of prompts (PII, cost); sample 1–10% for debugging
- **Retention**: Define retention for traces (e.g., 7 days); archive to cold storage
- **Alerts**: p99 latency > 5s, error rate > 1%, cost > daily budget
- **Fallbacks**: On LLM failure, fall back to cached response or human handoff
- **Rate limits**: Respect provider limits; implement queue/backoff

---

## References and Further Reading

- **LangSmith**: [LangSmith docs](https://docs.smith.langchain.com/)
- **W&B LLM**: [Weights & Biases for LLMs](https://docs.wandb.ai/guides/llms)
- **OpenTelemetry**: [OTLP specification](https://opentelemetry.io/docs/specs/otlp/)
- **Helicone**: [Open source LLM observability](https://www.helicone.ai/)
- **Arize Phoenix**: [LLM evals and tracing](https://phoenix.arize.com/)

---

## Summary

| Area | Key Practice |
|------|--------------|
| **Logging** | Request ID, prompt hash, tokens, latency, errors |
| **Tracing** | LangSmith, W&B, OpenTelemetry |
| **Metrics** | Latency percentiles, cost, error rate |
| **Evals** | Custom + benchmarks (MMLU, HumanEval) |
| **Guardrails** | PII, toxicity, jailbreak, input validation |
| **Cost** | Caching, routing, batching |
| **Drift** | Output quality, user feedback |
| **A/B** | Test prompts and models |

**Tools**: LangSmith, Weights & Biases, Helicone, Arize Phoenix, OpenTelemetry, Guardrails AI
