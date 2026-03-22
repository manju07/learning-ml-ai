# Resilience Engineering & Chaos Engineering

## Table of Contents
1. [Introduction](#1-introduction)
2. [Resilience Principles](#2-resilience-principles)
3. [Failure Modes & Analysis](#3-failure-modes--analysis)
4. [Chaos Engineering Fundamentals](#4-chaos-engineering-fundamentals)
5. [Chaos Experiment Design](#5-chaos-experiment-design)
6. [Chaos Tooling](#6-chaos-tooling)
7. [Game Days](#7-game-days)
8. [Blast Radius Management](#8-blast-radius-management)
9. [Building a Chaos Program](#9-building-a-chaos-program)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

**Resilience Engineering** is the practice of designing and operating systems that continue to function correctly in the presence of failures — and recover gracefully when they do fail.

**Chaos Engineering** is the discipline of *intentionally* introducing controlled failures to expose weaknesses before they cause outages in production.

### 1.1 The Case for Breaking Things Intentionally

```
Traditional approach:
  Build system → Write tests → Deploy → Hope it works
  Result: First real failure discovered during production outage

Chaos Engineering approach:
  Build system → Continuously inject failures → Learn + fix
  Result: Weaknesses discovered in controlled experiments,
          not during a 3AM on-call escalation

Netflix's observation:
  "If we don't test our system's resilience, our customers will"
```

### 1.2 Resilience vs Reliability vs Fault Tolerance

| Term | Definition | Focus |
|------|-----------|-------|
| **Reliability** | Performing intended functions for a given time | MTBF, uptime SLAs |
| **Fault Tolerance** | Continuing operation despite component failures | Redundancy, replication |
| **Resilience** | Adapting to and recovering from disruption | Elasticity, self-healing |
| **Chaos Engineering** | Science of discovering failure modes proactively | Controlled experiments |

---

## 2. Resilience Principles

### 2.1 Design for Failure

```
Resilience Design Principles:

1. ASSUME FAILURE ALWAYS HAPPENS
   ─ Servers fail, networks partition, disks corrupt
   ─ Design every component to fail safely
   ─ Never assume external calls will succeed

2. FAIL FAST
   ─ Detect failure immediately
   ─ Don't let slow failures cascade
   ─ Timeouts on every external call

3. FAIL SAFE
   ─ Degrade gracefully (show cached data vs crash)
   ─ Safe defaults over failure
   ─ User-visible error messages, not stack traces

4. ISOLATE FAILURES
   ─ Bulkhead pattern (separate thread pools)
   ─ Circuit breaker (stop calling failing services)
   ─ Timeouts + retries with jitter

5. RECOVER AUTOMATICALLY
   ─ Auto-restart failed pods (Kubernetes liveness probe)
   ─ Auto-failover to replica (RDS Multi-AZ)
   ─ Circuit breaker auto-closes after cooldown
```

### 2.2 Resilience Patterns (Reference)

These build on [05-system-design-patterns.md](./05-system-design-patterns.md):

```
Resilience Pattern Hierarchy:

PREVENTION
  ─ Load shedding      → drop low-priority traffic when overloaded
  ─ Rate limiting      → protect from traffic spikes
  ─ Bulkhead           → isolate failures to thread pool

DETECTION
  ─ Health checks      → liveness + readiness probes
  ─ Circuit breaker    → detect downstream degradation
  ─ Timeout            → detect slow calls

RECOVERY
  ─ Retry with backoff → transient failure recovery
  ─ Fallback           → cached response or default
  ─ Failover           → switch to healthy replica

LEARNING
  ─ Chaos experiments  → discover unknown failure modes
  ─ Game days          → practice incident response
  ─ Postmortems        → learn from real incidents
```

### 2.3 The Availability Math

```
Cascading failure math (serial dependencies):
  Service A: 99.9% × Service B: 99.9% × Service C: 99.9%
  = 99.7% availability (2.6 hours downtime/year)

Parallel (redundant) components:
  P(both fail) = (1 - 0.999) × (1 - 0.999) = 0.000001
  = 99.9999% availability

Lesson: Eliminate single points of failure via redundancy
        Build in isolation to contain failures
```

---

## 3. Failure Modes & Analysis

### 3.1 Failure Mode Taxonomy

```
Types of Failures:

INFRASTRUCTURE
  ─ Server crash / OOM kill
  ─ Disk full / I/O saturation
  ─ Network partition (split-brain)
  ─ AZ outage
  ─ Region outage

APPLICATION
  ─ Memory leak (slow degradation)
  ─ Thread pool exhaustion
  ─ Connection pool exhaustion
  ─ Deadlock
  ─ Cascading failure (thundering herd)

DEPENDENCY
  ─ Downstream service timeout
  ─ Database primary failure
  ─ External API rate limiting
  ─ DNS resolution failure
  ─ TLS certificate expiry

DATA
  ─ Corrupt data in database
  ─ Schema migration failure
  ─ Event schema mismatch
  ─ Kafka consumer lag spike

HUMAN
  ─ Bad deployment (config error)
  ─ Accidental deletion
  ─ Runaway query (table scan)
  ─ Secret rotation breaking auth
```

### 3.2 FMEA (Failure Mode and Effects Analysis)

A structured way to inventory and prioritize failure risks:

| Failure Mode | Probability (1-5) | Impact (1-5) | Detectability (1-5) | RPN | Mitigation |
|-------------|-------------------|--------------|---------------------|-----|------------|
| Primary DB fails | 2 | 5 | 2 | 20 | Multi-AZ + read replica |
| Downstream timeout | 4 | 3 | 3 | 36 | Circuit breaker + timeout |
| Memory leak | 3 | 4 | 4 | 48 | OOM alerts + auto-restart |
| Network partition | 2 | 4 | 3 | 24 | Retry + idempotency |
| Bad deployment | 3 | 5 | 2 | 30 | Canary + rollback |
| Cert expiry | 2 | 5 | 5 | 50 | Auto-renewal + alerts |

*RPN = Probability × Impact × Detectability (higher = more critical)*

### 3.3 Steady State Hypothesis

Before chaos experiments, define "normal":

```python
# Steady state metrics definition
STEADY_STATE = {
    # Business metrics (most important)
    "checkout_success_rate": "> 99.5%",
    "payment_processing_p99_latency": "< 2000ms",
    "active_sessions": "within 10% of baseline",

    # Technical metrics
    "http_5xx_rate": "< 0.1%",
    "database_connection_pool_usage": "< 70%",
    "kafka_consumer_lag": "< 10000 messages",
    "pod_restart_count_1h": "< 3 per service",

    # Infrastructure metrics
    "cpu_utilization": "< 70%",
    "memory_utilization": "< 80%",
    "disk_io_utilization": "< 60%",
}
```

---

## 4. Chaos Engineering Fundamentals

### 4.1 The Chaos Engineering Process

```
Chaos Experiment Lifecycle:

1. HYPOTHESIZE
   "When X happens, our system will Y
    because we have Z mitigation in place"
   
   Example: "When the payment service loses its
   database connection, checkout will degrade
   gracefully by returning cached pricing,
   because we have a Redis fallback"

2. DESIGN
   ─ Define steady state (what 'normal' looks like)
   ─ Choose failure injection method
   ─ Define abort conditions (when to stop)
   ─ Choose blast radius (start small)

3. EXECUTE (SAFELY)
   ─ Start in dev/staging
   ─ Gradually move to production
   ─ Have rollback ready
   ─ Monitor in real-time

4. ANALYZE
   ─ Did steady state hold?
   ─ If yes: confidence in resilience mechanism
   ─ If no: discovered a weakness → file bug → fix

5. FIX & AUTOMATE
   ─ Fix discovered weakness
   ─ Add experiment to CI/CD (continuous chaos)
   ─ Update runbooks based on learnings
```

### 4.2 Chaos Experiment Template

```yaml
# chaos-experiment.yaml
name: payment-service-db-failover
hypothesis: |
  When the payments database primary fails, the
  service will automatically failover to the read
  replica within 30 seconds, with checkout success
  rate staying above 95% during failover.

steady_state:
  - metric: checkout_success_rate
    condition: "> 99.5%"
    window: 5m
  - metric: payment_p99_latency_ms
    condition: "< 2000"
    window: 5m

method:
  - type: kill_database_primary
    target: payments-postgres-primary
    duration: 60s

abort_conditions:
  - metric: checkout_success_rate
    condition: "< 90%"  # Stop if it gets too bad
  - metric: payment_p99_latency_ms
    condition: "> 10000"

blast_radius:
  environment: staging
  traffic_percentage: 100  # All staging traffic
  estimated_users_affected: 0  # No prod users

rollback:
  - restore: payments-postgres-primary
  - verify: steady_state_metrics

owner: payments-team
last_run: 2024-01-15
result: FAILED  # → Discovered: failover took 90s, not 30s
ticket: INFRA-4521  # Multi-AZ failover optimization
```

---

## 5. Chaos Experiment Design

### 5.1 Chaos Experiment Categories

```
Experiment Categories:

RESOURCE PRESSURE
  ─ CPU stress (cpu_burn)         → test throttling behavior
  ─ Memory fill (memory_hog)      → test OOM handling
  ─ Disk fill (disk_fill)         → test disk-full handling
  ─ Network bandwidth limit       → test degraded network

NETWORK FAILURES
  ─ Latency injection             → test timeouts and retries
  ─ Packet loss (10-20%)         → test retry logic
  ─ Network partition             → test split-brain handling
  ─ DNS failure                   → test DNS resilience
  ─ TLS handshake failure         → test cert validation

PROCESS FAILURES
  ─ Kill process / pod            → test restart/redundancy
  ─ Kill all pods in deployment   → test k8s rescheduling
  ─ Kill leader pod               → test leader election
  ─ OOM kill container            → test OOM handling

DEPENDENCY FAILURES
  ─ Downstream service timeout    → test circuit breaker
  ─ Downstream returns errors     → test error handling
  ─ Database primary kill         → test failover
  ─ Cache unavailable             → test cache-miss behavior
  ─ Message queue unavailable     → test producer buffering

STATE CORRUPTION
  ─ Clock skew injection          → test time-dependent logic
  ─ Corrupt response body         → test deserialization errors
  ─ Return partial response       → test partial failure handling
```

### 5.2 Hypothesis-Driven Design Examples

```
Example 1: Circuit Breaker Validation
──────────────────────────────────────
Hypothesis: "When inventory service becomes slow (>2s),
  our product detail pages will still load in <500ms
  because the circuit breaker will open and serve cached
  inventory from Redis"

Experiment:
  ─ Inject: 3s latency into inventory service responses
  ─ Monitor: product_page_p99_latency
  ─ Expected: circuit opens within 5 requests, cache served

Weak hypothesis (bad): "System will handle failures"
Strong hypothesis (good): Specific failure + specific mitigation
  + measurable outcome

Example 2: Thundering Herd Prevention
──────────────────────────────────────
Hypothesis: "When we deploy and all 50 pods restart,
  the database will not be overwhelmed because connection
  pools are sized per-pod and startup is staggered via
  Kubernetes maxSurge/maxUnavailable"

Experiment:
  ─ Inject: Simultaneous restart of all pods
  ─ Monitor: DB connection count, query latency
  ─ Expected: Connection count stays < 200 (50 pods × 4 per pod)
```

---

## 6. Chaos Tooling

### 6.1 Open Source Chaos Tools

| Tool | Layer | Key Strength |
|------|-------|-------------|
| **Chaos Monkey** (Netflix) | AWS EC2 | Randomly terminates instances |
| **Chaos Mesh** | Kubernetes | K8s-native, rich experiment types |
| **Litmus Chaos** | Kubernetes | Large experiment library, GitOps |
| **Toxiproxy** (Shopify) | Network | Network failure simulation for testing |
| **tc (Linux)** | Network | Kernel-level network manipulation |
| **stress-ng** | CPU/Memory | Stress testing resources |
| **Pumba** | Docker | Container-level chaos |

### 6.2 Chaos Mesh — Kubernetes-Native Chaos

```yaml
# Chaos Mesh: Kill 50% of payment service pods
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: payments-pod-failure
  namespace: chaos-testing
spec:
  action: pod-failure
  mode: percentage
  value: "50"
  selector:
    namespaces:
      - payments
    labelSelectors:
      app: payments-service
  duration: "60s"
  scheduler:
    cron: "@every 12h"  # Run automatically every 12 hours
```

```yaml
# Chaos Mesh: Network latency on payments → database
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: payments-db-latency
spec:
  action: delay
  mode: all
  selector:
    namespaces:
      - payments
    labelSelectors:
      app: payments-service
  delay:
    latency: "500ms"
    correlation: "25"     # 25% correlation between packets
    jitter: "100ms"
  target:
    selector:
      namespaces:
        - databases
      labelSelectors:
        app: postgres
  direction: to
  duration: "120s"
```

### 6.3 Toxiproxy for Integration Testing

```go
// Toxiproxy: Test resilience in integration tests
// Simulates network conditions programmatically

import "github.com/Shopify/toxiproxy/v2/api/client"

func TestCircuitBreakerOpensOnTimeout(t *testing.T) {
    client := toxiproxy.NewClient("http://toxiproxy:8474")

    // Create proxy to database
    proxy, _ := client.CreateProxy("payments-db",
        "0.0.0.0:15432",       // listen
        "postgres:5432")        // upstream

    // Inject 3-second latency
    proxy.AddToxic("latency", "latency", "downstream", 1.0,
        toxiproxy.Attributes{"latency": 3000})

    // Run payment operations — circuit breaker should open
    results := runPaymentRequests(100)

    // Assert: after 5 failures, circuit opens, subsequent
    // calls fail fast (<10ms) not slow (>3000ms)
    slowCalls := countCallsAbove(results, 100*time.Millisecond)
    assert.Less(t, slowCalls, 10,
        "Circuit breaker should have opened after 5 failures")

    // Cleanup
    proxy.RemoveToxic("latency")
}
```

### 6.4 Gremlin (Commercial)

Gremlin provides a SaaS chaos engineering platform:

```
Gremlin Features:
  ─ Attack templates (CPU, memory, network, state)
  ─ Scenario builder (multi-step experiment)
  ─ Auto-halt on SLO breach
  ─ Reliability score tracking over time
  ─ Team collaboration and audit trail

Gremlin Scenario Example:
  "Simulate AZ outage for payments"
    Step 1: Kill all pods in us-east-1a (30s)
    Step 2: Inject 2x traffic to remaining pods (60s)
    Step 3: Restore pods in us-east-1a
    
  Success criteria:
    ─ Checkout success rate stays > 99%
    ─ No increase in P99 latency > 20%
    ─ Automatic recovery within 5 minutes
```

---

## 7. Game Days

### 7.1 What is a Game Day

A **Game Day** is a structured exercise where teams simulate a significant failure scenario to practice detection, communication, and response — discovering gaps in runbooks, monitoring, and on-call processes.

```
Game Day vs Chaos Experiment:

Chaos Experiment                    Game Day
────────────────                    ─────────────────
Automated, repeatable               Manual, collaborative exercise
Small, focused blast radius         Larger scenario, multiple teams
Run frequently (CI/CD)              Run quarterly
Tests resilience mechanisms         Tests human response + process
Developer runs solo                 Multiple teams participate
Minutes long                        Hours long
```

### 7.2 Game Day Structure

```
Game Day Agenda Template (4 hours):

PRE-GAME (1 week before)
  ─ Define scenario (keep secret from participants)
  ─ Prepare failure injection scripts
  ─ Brief observers/facilitators
  ─ Confirm rollback procedures ready
  ─ Notify stakeholders (not participants)

SETUP (30 min)
  ─ Introduce game day goals
  ─ Assign roles: IC, comms, on-call engineer
  ─ Confirm steady state (monitors all green)

INJECTION (2 hours)
  Phase 1 (0:00): Inject initial failure
                  Observe detection time
  Phase 2 (0:20): Escalate if not detected
                  Observe escalation process
  Phase 3 (0:40): Add second failure
                  Test multi-failure handling
  Phase 4 (1:20): Inject resolution hint
  Phase 5 (1:40): Restore normal state
                  Measure MTTR

DEBRIEF (1.5 hours)
  ─ Timeline review (what happened when)
  ─ What went well?
  ─ What was unclear or missing?
  ─ Action items: monitoring gaps, runbook updates, alerts
  ─ Blameless retrospective — systems, not people
```

### 7.3 Sample Game Day Scenarios

```
Scenario 1: "The Midnight Database Failover"
  Setup: Primary DB fails at 2AM
  Inject: Kill RDS primary
  Tests:
    ─ Does the alert fire within 2 minutes?
    ─ Does the on-call know which runbook to follow?
    ─ Does the app reconnect automatically?
    ─ Is the communication template ready?
    ─ How long until service restored?

Scenario 2: "The Cascading Cache Failure"
  Setup: Redis cluster goes down
  Inject: Kill Redis, then inject traffic spike
  Tests:
    ─ Does DB get overwhelmed without cache?
    ─ Does circuit breaker protect downstream?
    ─ How does the team detect cache miss storm?
    ─ Can they restore Redis before DB falls over?

Scenario 3: "The Bad Deployment Blast"
  Setup: New deployment with subtle bug (high CPU)
  Inject: Deploy a version with CPU-intensive code
  Tests:
    ─ How long until CPU alert fires?
    ─ Does team know how to rollback quickly?
    ─ Is the rollback procedure documented?
    ─ Does the feature flag kill switch work?

Scenario 4: "Dependency Removal" (DiRT — Disaster
           Recovery Test, Google's approach)
  Setup: External payment gateway unavailable
  Inject: Block traffic to payment provider
  Tests:
    ─ Does circuit breaker open?
    ─ Are errors surfaced clearly to users?
    ─ Can operations team switch to backup provider?
    ─ Are SLO burn rate alerts firing correctly?
```

---

## 8. Blast Radius Management

### 8.1 Progressive Chaos Confidence

Never start chaos in production. Build confidence gradually:

```
Chaos Maturity Progression:

Stage 1 — Development
  ─ Unit tests for failure scenarios
  ─ Toxiproxy in integration tests
  ─ Zero customer impact
  ─ Run: always (in CI)

Stage 2 — Staging
  ─ Full chaos experiment suite
  ─ Production-like load
  ─ Zero customer impact
  ─ Run: after every deploy

Stage 3 — Production (Dark Traffic)
  ─ Shadow traffic experiments
  ─ Non-user-facing paths only
  ─ Minimal risk
  ─ Run: weekly

Stage 4 — Production (Limited)
  ─ Single AZ experiments
  ─ 5-10% traffic affected max
  ─ Business hours only
  ─ Run: biweekly with approval

Stage 5 — Production (Full)
  ─ Netflix-style continuous chaos
  ─ Any service, any time
  ─ Full automation with auto-abort
  ─ Run: continuously (Chaos Monkey)
```

### 8.2 Safety Controls

```yaml
# Safety controls for every chaos experiment

safety:
  # Automatic abort conditions
  abort_if:
    - metric: checkout_success_rate
      drops_below: 95%
    - metric: error_rate_5xx
      rises_above: 5%
    - time_elapsed: 300s  # Auto-stop after 5 minutes

  # Pre-flight checks (don't run if...)
  skip_if:
    - active_incident: true           # Incident already in progress
    - recent_deployment: "< 30 min"  # Too soon after deploy
    - traffic_anomaly: true           # Unusual traffic pattern
    - business_hours_only: false      # After hours only

  # Rollback
  auto_rollback:
    enabled: true
    trigger: any_abort_condition

  # Notification
  notify:
    before: [oncall-team, stakeholders-channel]
    on_abort: [oncall-team, incident-channel]
    after: [oncall-team, chaos-team]
```

---

## 9. Building a Chaos Program

### 9.1 Chaos Engineering Maturity Model

```
Chaos Maturity Model:

Level 0 — REACTIVE
  ─ Failures discovered in production incidents
  ─ No proactive testing
  ─ "We'll fix it when it breaks"

Level 1 — AWARE
  ─ Understand the value of chaos engineering
  ─ Occasional manual fault injection
  ─ No formal process

Level 2 — MANUAL EXPERIMENTS
  ─ Structured game days (quarterly)
  ─ Documented experiment templates
  ─ Team trained in chaos tooling
  ─ All chaos in pre-prod only

Level 3 — AUTOMATED
  ─ Experiments in CI/CD pipeline
  ─ Chaos in production (limited)
  ─ Steady state monitoring
  ─ Blameless postmortems

Level 4 — CONTINUOUS
  ─ Chaos running continuously (a la Netflix)
  ─ Reliability scores tracked over time
  ─ Chaos as a first-class engineering practice
  ─ Chaos team or guild

Level 5 — OPTIMIZED
  ─ AI-driven hypothesis generation
  ─ Automatic fix recommendations
  ─ Predictive resilience analysis
```

### 9.2 Starting a Chaos Program (Step by Step)

```
Month 1 — Foundation:
  □ Inventory your most critical user journeys
  □ Define steady state metrics for top 3 journeys
  □ Install Chaos Mesh in staging cluster
  □ Run first experiment: kill 1 replica of a service

Month 2 — Expand Experiments:
  □ Run game day (DB failover scenario)
  □ Add 5 more experiment types (network, resource)
  □ Document all findings as tickets
  □ Fix top 3 discovered weaknesses

Month 3 — Automate:
  □ Add 2 experiments to deployment pipeline
  □ Set up auto-abort on SLO breach
  □ Present findings to engineering leadership
  □ Get approval for production experiments

Month 4+ — Production & Culture:
  □ Run first production experiment (dark traffic)
  □ Quarterly game days standard across all teams
  □ Chaos engineering in on-call training
  □ Track reliability score trend over time
```

### 9.3 Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|---------|
| **Chaos without observability** | Can't tell if steady state held | Set up monitoring before chaos |
| **Big bang experiments** | Entire production destroyed | Start with kill-one-pod, not kill-all |
| **Chaos theater** | Experiments cherry-picked to succeed | Let experiments run without bias |
| **No fix follow-through** | Weakness discovered, nothing done | Treat experiment failure as P2 bug |
| **Fear-driven chaos** | Team scared → only chaos in dev | Build trust incrementally with staging |
| **Missing abort conditions** | Chaos runs too long, real outage | Always define auto-abort conditions |

---

## 10. Practical Examples

### 10.1 Netflix Chaos Monkey

Netflix's original chaos tool terminates random EC2 instances during business hours:

```
Netflix Simian Army (evolved):
  Chaos Monkey        → Terminates random instances (always on)
  Chaos Gorilla       → Simulates entire AZ failure
  Chaos Kong          → Simulates entire region failure
  Latency Monkey      → Introduces artificial latency
  Conformity Monkey   → Checks instances meet best practices
  Doctor Monkey       → Monitors health checks
  Janitor Monkey      → Cleans up unused resources
  Security Monkey     → Audits security policies

Philosophy:
  "We need to be comfortable with failure in production
   because it's going to happen whether we like it or not.
   Better to have it happen on our terms." — Netflix
```

### 10.2 AWS Fault Injection Simulator (FIS)

```json
// AWS FIS Experiment Template
{
  "description": "Inject CPU stress on payment service ECS tasks",
  "targets": {
    "paymentTasks": {
      "resourceType": "aws:ecs:task",
      "resourceTags": {"service": "payments"},
      "filters": [{
        "path": "taskArn",
        "values": ["RUNNING"]
      }],
      "selectionMode": "PERCENT(25)"
    }
  },
  "actions": {
    "cpuStress": {
      "actionId": "aws:ssm:send-command",
      "parameters": {
        "documentArn": "arn:aws:ssm:::document/AWSFIS-Run-CPU-Stress",
        "documentParameters": "{\"DurationSeconds\": \"120\", \"CPU\": \"0\"}",
        "duration": "PT2M"
      },
      "targets": {"Tasks": "paymentTasks"}
    }
  },
  "stopConditions": [{
    "source": "aws:cloudwatch:alarm",
    "value": "arn:aws:cloudwatch:::alarm:PaymentCheckoutErrorRateHigh"
  }]
}
```

### 10.3 Continuous Chaos in CI/CD

```yaml
# .github/workflows/chaos.yaml
# Runs after every deployment to staging

name: Chaos Validation
on:
  deployment_status:
    environments: [staging]

jobs:
  chaos-experiments:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      - name: Wait for stable deployment
        run: sleep 120

      - name: Verify steady state
        run: |
          ./scripts/check-steady-state.sh \
            --metric checkout_success_rate --min 99.5 \
            --metric p99_latency_ms --max 2000

      - name: Run pod failure experiment
        run: |
          chaos-mesh apply \
            --experiment experiments/pod-failure-25pct.yaml \
            --duration 60s \
            --abort-on-metric checkout_success_rate:below:95

      - name: Run network latency experiment
        run: |
          chaos-mesh apply \
            --experiment experiments/downstream-latency-500ms.yaml \
            --duration 60s \
            --abort-on-metric p99_latency_ms:above:5000

      - name: Verify steady state restored
        run: |
          sleep 30
          ./scripts/check-steady-state.sh \
            --metric checkout_success_rate --min 99.5

      - name: Publish chaos report
        run: ./scripts/publish-chaos-report.sh
        if: always()
```

### 10.4 Measuring Resilience Progress

```
Resilience Score Dashboard (tracked over time):

Service: payments-service
Month: March 2024

Experiment Results:
  ┌──────────────────────────────────┬──────────┬──────────┐
  │ Experiment                       │ Result   │ MTTR     │
  ├──────────────────────────────────┼──────────┼──────────┤
  │ Kill 50% of pods                 │ ✅ PASS  │  45s     │
  │ DB primary failover              │ ✅ PASS  │  28s     │
  │ 500ms downstream latency         │ ✅ PASS  │  N/A     │
  │ Redis unavailable                │ ❌ FAIL  │  420s    │ ← bug filed
  │ Disk 90% full                    │ ✅ PASS  │  N/A     │
  │ 2x traffic spike                 │ ✅ PASS  │  N/A     │
  └──────────────────────────────────┴──────────┴──────────┘

Resilience Score: 5/6 (83%) ↑ from 67% last month

Trend: improving — Redis fallback was fixed in PYMT-4521
Next focus: Multi-region failover experiment
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Chaos Engineering** | Intentional controlled failure injection to discover weaknesses proactively |
| **Steady State Hypothesis** | Define what "normal" looks like *before* running experiments |
| **FMEA** | Systematically inventory and prioritize failure risks |
| **Chaos Mesh / Litmus** | Kubernetes-native chaos tooling — start here |
| **Game Days** | Quarterly exercises to test human response, not just automated resilience |
| **Blast Radius** | Start in dev, progress to staging, then limited production |
| **Safety Controls** | Auto-abort conditions prevent chaos from becoming real outages |
| **Continuous Chaos** | Embed experiments in CI/CD for every deployment to staging |
| **Blameless Culture** | Experiments that fail are successes — they revealed hidden weaknesses |
| **Resilience Score** | Track experiment pass rate over time to measure program progress |
