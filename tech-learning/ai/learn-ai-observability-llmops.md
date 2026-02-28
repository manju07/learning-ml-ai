# AI Observability & LLM Operations: Complete Guide

## Table of Contents
1. [Introduction to LLM Ops](#introduction-to-llm-ops)
2. [Logging and Tracing](#logging-and-tracing)
3. [LLM Metrics](#llm-metrics)
4. [Evaluation and Benchmarks](#evaluation-and-benchmarks)
5. [Guardrails and Safety](#guardrails-and-safety)
6. [Cost and Latency Optimization](#cost-and-latency-optimization)
7. [Drift and Quality Monitoring](#drift-and-quality-monitoring)
8. [Practical Examples](#practical-examples)
9. [Best Practices](#best-practices)

---

## Introduction to LLM Ops

**LLM Ops** extends MLOps to production LLM systems: observability, evaluation, safety, cost control, and continuous improvement. LLMs introduce unique challenges: non-determinism, prompt dependency, and high token cost.

### Key Differences from Traditional ML Ops

| Traditional ML | LLM Ops |
|----------------|---------|
| Fixed input → output | Prompt + context → variable output |
| Metric: accuracy, F1 | Metric: relevance, safety, latency, cost |
| Model versioning | Model + prompt versioning |
| Drift: input distribution | Drift: output quality, user behavior |
| Deterministic | Non-deterministic (temperature > 0) |

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

def logged_llm_call(llm, prompt, **kwargs):
    request_id = str(uuid.uuid4())
    start = time.time()
    try:
        response = llm.generate(prompt, **kwargs)
        latency_ms = (time.time() - start) * 1000
        log_event({
            "request_id": request_id,
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
from opentelemetry.trace import Status

tracer = trace.get_tracer("llm-app", "1.0")

def traced_llm_call(prompt):
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attribute("prompt.length", len(prompt))
        result = llm.generate(prompt)
        span.set_attribute("response.length", len(result))
        span.set_attribute("model", "gpt-4")
        return result
```

### Structured Logging for RAG

```python
def logged_rag_query(query, retriever, llm):
    with trace_span("rag_query"):
        with trace_span("retrieve"):
            docs = retriever.get_relevant_documents(query)
            log("retrieved_docs", count=len(docs), top_scores=[d.metadata.get("score") for d in docs[:3]])
        with trace_span("generate"):
            response = llm.generate(build_prompt(query, docs))
        log("rag_response", prompt_tokens=..., completion_tokens=...)
        return response
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

```python
def count_tokens(text, model="gpt-4"):
    # OpenAI: tiktoken
    import tiktoken
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Cost calculation
def cost_estimate(prompt_tokens, completion_tokens, model="gpt-4"):
    # Pricing per 1K tokens (example)
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

```python
# Semantic cache: similar prompts → same cache key
def semantic_cache_key(prompt, embedding_model):
    emb = embedding_model.encode(prompt)
    return find_nearest(emb, cache_keys) or None

def cached_llm_call(prompt, cache):
    key = semantic_cache_key(prompt)
    if key and cache.has(key):
        return cache.get(key)
    result = llm.generate(prompt)
    cache.set(semantic_cache_key(prompt), result)
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

## Summary

| Area | Key Practice |
|------|--------------|
| **Logging** | Request ID, tokens, latency, errors |
| **Metrics** | Latency percentiles, cost, error rate |
| **Evals** | Custom + benchmarks (MMLU, HumanEval) |
| **Guardrails** | PII, toxicity, jailbreak, input validation |
| **Cost** | Caching, routing, batching |
| **Drift** | Output quality, user feedback |
| **A/B** | Test prompts and models |

**Tools**: LangSmith, Weights & Biases, Helicone, Arize Phoenix, OpenTelemetry, Guardrails AI
