# Observability and SRE: Guide for Architects

## Table of Contents
1. [Observability vs Monitoring](#1-observability-vs-monitoring)
2. [The Three Pillars: Metrics, Logs, Traces](#2-the-three-pillars-metrics-logs-traces)
3. [SLIs, SLOs, and Error Budgets](#3-slis-slos-and-error-budgets)
4. [Distributed Tracing](#4-distributed-tracing)
5. [Structured Logging](#5-structured-logging)
6. [Alerting and On-Call](#6-alerting-and-on-call)
7. [OpenTelemetry](#7-opentelemetry)
8. [Incident Response](#8-incident-response)
9. [Practical Examples](#9-practical-examples)

---

## 1. Observability vs Monitoring

### 1.1 Monitoring

Pre-defined dashboards and alerts. Answers: "Is X broken?"

### 1.2 Observability

Ability to explore system behavior from outputs. Answers: "Why is X broken?" — explore unknowns.

**Observability** = Metrics + Logs + Traces, with ability to correlate and explore.

---

## 2. The Three Pillars: Metrics, Logs, Traces

### 2.1 Metrics

Numeric measurements over time. Aggregated (count, sum, histogram).

| Type | Example |
|------|---------|
| **Counter** | Requests total, errors total |
| **Gauge** | Active connections, queue size |
| **Histogram** | Request duration (p50, p95, p99) |
| **Summary** | Similar to histogram, client-side quantiles |

### 2.2 Logs

Discrete events. Should be **structured** (JSON) for querying.

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "ERROR",
  "message": "Payment failed",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "usr_123",
  "order_id": "ord_456",
  "error": "card_declined"
}
```

### 2.3 Traces

Request flow across services. Span = one operation; trace = tree of spans.

```
Trace: Request from API Gateway
  Span: api-gateway (10ms)
    Span: order-service (8ms)
      Span: db query (5ms)
      Span: payment-service call (3ms)
    Span: user-service (2ms)
```

---

## 3. SLIs, SLOs, and Error Budgets

### 3.1 SLI (Service Level Indicator)

A measurable value. Examples:

- **Availability**: % of successful requests
- **Latency**: % of requests under 200ms
- **Throughput**: Requests per second
- **Error rate**: % of 5xx responses

### 3.2 SLO (Service Level Objective)

Target for SLI. Example: "99.9% of requests succeed."

### 3.3 SLA (Service Level Agreement)

Contract with consequences. SLO is internal; SLA is external.

### 3.4 Error Budget

If SLO is 99.9%, error budget = 0.1% failures allowed.

- **1000 requests/min** → ~1 failure/min allowed
- Exhausted budget → freeze feature releases, focus on reliability

### 3.5 Example: SLO Definitions

```
Availability SLO: 99.9% (3 nines)
  -> 43 minutes downtime/month allowed

Latency SLO: 95% of requests < 200ms
  -> 5% can be slower

Error rate SLO: 99.95% success
  -> 0.05% errors allowed
```

---

## 4. Distributed Tracing

### 4.1 OpenTelemetry / Jaeger / Zipkin

- **Trace ID**: Unique per request (propagated in headers)
- **Span ID**: Unique per operation
- **Parent span**: Links spans in hierarchy

### 4.2 W3C Trace Context

```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
tracestate: vendor=value
```

### 4.3 Instrumentation

- **Auto**: Framework middleware (e.g., FastAPI, gRPC)
- **Manual**: Start span for custom logic

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__, "1.0.0")

def process_order(order_id: str):
    with tracer.start_as_current_span("process_order") as span:
        span.set_attribute("order.id", order_id)
        try:
            result = do_work(order_id)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

---

## 5. Structured Logging

### 5.1 Format

JSON preferred for machine parsing. Include:

- `timestamp`
- `level` (DEBUG, INFO, WARN, ERROR)
- `message`
- `trace_id`, `span_id` (for correlation)
- Context: `user_id`, `request_id`, etc.

### 5.2 Example: structlog (Python)

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()
log.info("order_created", order_id="ord_123", user_id="usr_456", amount=99.99)
# {"event": "order_created", "order_id": "ord_123", "user_id": "usr_456", "amount": 99.99, "timestamp": "..."}
```

### 5.3 Log Levels

- **ERROR**: Failures requiring attention
- **WARN**: Recoverable issues
- **INFO**: Business events (order created, user logged in)
- **DEBUG**: Diagnostic (only in dev/staging)

---

## 6. Alerting and On-Call

### 6.1 Alert Design

- **Actionable**: Clear action to take
- **Few false positives**: Tune thresholds
- **Runbooks**: Document response steps

### 6.2 Alert Tiers

| Tier | Example | Response |
|------|---------|----------|
| **Critical** | Service down | Page immediately |
| **Warning** | Latency high | Investigate soon |
| **Info** | Approaching limit | Review in morning |

### 6.3 Prometheus Alert Example

```yaml
groups:
- name: api
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate on {{ $labels.service }}"
      runbook: "https://runbooks.example.com/high-error-rate"
```

### 6.4 Escalation

```
Alert fires -> Notify primary
  -> 5 min ack? Escalate to secondary
  -> 15 min ack? Escalate to manager
```

---

## 7. OpenTelemetry

### 7.1 Unified SDK

One SDK for traces, metrics, logs. Exporters to Jaeger, Prometheus, etc.

### 7.2 Setup (Python)

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server

# Traces
trace_provider = TracerProvider()
jaeger_exporter = JaegerExporter(agent_host_name="localhost", agent_port=6831)
trace_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(trace_provider)

# Metrics
reader = PrometheusMetricReader()
meter_provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(meter_provider)
start_http_server(port=9464, addr="0.0.0.0")  # Scrape endpoint
```

### 7.3 Auto-Instrumentation

```bash
opentelemetry-instrument --traces_exporter jaeger python app.py
```

---

## 8. Incident Response

### 8.1 Phases

1. **Detect**: Alert, user report
2. **Triage**: Severity, impact
3. **Mitigate**: Restore service (rollback, scale, fix)
4. **Resolve**: Root cause, postmortem
5. **Improve**: Action items, prevent recurrence

### 8.2 Postmortem Template

- **Summary**: What happened
- **Impact**: Users, duration, metrics
- **Timeline**: Events
- **Root cause**: 5 Whys
- **Action items**: With owners
- **Blameless**: Focus on process, not people

---

## 9. Practical Examples

### 9.1 Prometheus Metrics (Python)

```python
from prometheus_client import Counter, Histogram, start_http_server

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    return response

start_http_server(9090)  # Metrics endpoint
```

### 9.2 Correlation ID Middleware

```python
from uuid import uuid4

@app.middleware("http")
async def correlation_id_middleware(request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response
```

### 9.3 SLO-Based Alerting

```yaml
# 99.9% availability = 0.1% error budget
# Alert when error rate > 0.001 for 5 minutes
- alert: AvailabilitySLOBreach
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m])) 
    / sum(rate(http_requests_total[5m])) > 0.001
  for: 5m
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Observability** | Metrics + Logs + Traces, explorable |
| **SLI/SLO** | Measure, set targets, error budget |
| **Tracing** | Trace ID propagation, span hierarchy |
| **Logging** | Structured JSON, correlation IDs |
| **Alerting** | Actionable, runbooks, escalation |
| **OpenTelemetry** | Unified SDK, auto-instrumentation |

---

## Further Reading

- *Site Reliability Engineering* — Google SRE Book
- OpenTelemetry: https://opentelemetry.io/
- Prometheus: https://prometheus.io/docs/
