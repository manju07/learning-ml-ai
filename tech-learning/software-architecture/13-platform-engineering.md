# Platform Engineering & Developer Experience

## Table of Contents
1. [Introduction](#1-introduction)
2. [Internal Developer Platforms (IDP)](#2-internal-developer-platforms-idp)
3. [Platform as a Product](#3-platform-as-a-product)
4. [Golden Paths & Paved Roads](#4-golden-paths--paved-roads)
5. [Backstage & Service Catalogs](#5-backstage--service-catalogs)
6. [Self-Service Infrastructure](#6-self-service-infrastructure)
7. [Developer Experience (DevEx) Metrics](#7-developer-experience-devex-metrics)
8. [CI/CD Platform Architecture](#8-cicd-platform-architecture)
9. [Platform Team Topology](#9-platform-team-topology)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

**Platform Engineering** is the discipline of designing and building toolchains and workflows that enable self-service capabilities for software engineering organizations. The goal is to reduce cognitive load on application teams and accelerate software delivery.

### 1.1 Why Platform Engineering

```
Traditional DevOps Model (problematic):
┌──────────────────────────────────────┐
│         Application Teams            │
│  Write code + manage infra +         │
│  configure CI/CD + handle security   │
│  + manage observability + ...        │
│         COGNITIVE OVERLOAD           │
└──────────────────────────────────────┘

Platform Engineering Model:
┌──────────────────────────────────────┐
│         Application Teams            │
│    Focus: Write business logic       │
│    Self-service platform APIs        │
└──────────────────┬───────────────────┘
                   │ consume
┌──────────────────▼───────────────────┐
│         Platform Team                │
│  Provides: IDP, Golden Paths,        │
│  Service Catalog, Paved Roads        │
└──────────────────────────────────────┘
```

### 1.2 Key Problems Solved

| Problem | Traditional Approach | Platform Engineering |
|---------|---------------------|---------------------|
| New service onboarding | Weeks, manual tickets | Minutes, self-service |
| Infrastructure provisioning | Ops team bottleneck | Developer self-service |
| Security compliance | Post-deployment audit | Built into golden path |
| Observability setup | Per-team inconsistency | Automatic by default |
| Deployment pipeline | Each team builds own | Standardized templates |

### 1.3 DORA Metrics Connection

Platform engineering directly improves [DORA metrics](https://dora.dev):

| DORA Metric | Platform Engineering Impact |
|-------------|----------------------------|
| **Deployment Frequency** | Automated pipelines → deploy on every commit |
| **Lead Time for Changes** | Self-service → no ops tickets, no waiting |
| **Change Failure Rate** | Guardrails in golden paths → fewer bad deploys |
| **MTTR** | Standardized observability → faster diagnosis |

---

## 2. Internal Developer Platforms (IDP)

### 2.1 IDP Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Developer Portal                    │
│              (Backstage / Custom UI)                 │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
   ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌────▼────┐
   │Service│ │  CI/  │ │ Infra │ │Security │
   │Catalog│ │  CD   │ │  API  │ │ Portal  │
   └───┬───┘ └───┬───┘ └───┬───┘ └────┬────┘
       │          │          │          │
┌──────▼──────────▼──────────▼──────────▼──────────────┐
│              Platform Control Plane                   │
│   (Crossplane / Terraform / Pulumi / Custom)          │
└──────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │
   ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌────▼────┐
   │  k8s  │ │  AWS  │ │  GCP  │ │ Vault   │
   │Cluster│ │       │ │       │ │Secrets  │
   └───────┘ └───────┘ └───────┘ └─────────┘
```

### 2.2 Core IDP Capabilities

```
IDP Capabilities Map:
┌────────────────────────────────────────────┐
│ Compute         │ Databases                │
│ ─ Kubernetes    │ ─ Managed Postgres       │
│ ─ Serverless    │ ─ Redis clusters         │
│ ─ VMs           │ ─ MongoDB instances      │
├────────────────────────────────────────────┤
│ Networking      │ Observability            │
│ ─ Service mesh  │ ─ Metrics (Prometheus)   │
│ ─ Ingress       │ ─ Tracing (Jaeger)       │
│ ─ DNS           │ ─ Logging (ELK/Loki)     │
├────────────────────────────────────────────┤
│ Security        │ Deployment               │
│ ─ Secrets mgmt  │ ─ GitOps (ArgoCD/Flux)   │
│ ─ mTLS certs    │ ─ Feature flags          │
│ ─ RBAC          │ ─ Canary/blue-green      │
└────────────────────────────────────────────┘
```

### 2.3 IDP vs DevOps Toolchain

| Aspect | Raw DevOps Toolchain | IDP |
|--------|---------------------|-----|
| Interface | CLIs and config files | Developer portal + APIs |
| Onboarding | Learn 10+ tools | One unified UI |
| Compliance | Manual enforcement | Built-in guardrails |
| Customization | Unlimited, unguided | Guided with escape hatches |
| Maintenance | Per-team burden | Centralized platform team |

---

## 3. Platform as a Product

### 3.1 Product Thinking for Platforms

The platform team must treat internal developers as **customers**:

```
Product Management Applied to Platforms:
┌──────────────────────────────────────────┐
│  Understand Developers (User Research)   │
│  ─ Developer surveys (quarterly)         │
│  ─ Journey mapping (onboarding, deploy)  │
│  ─ Pain point interviews                 │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  Prioritize Platform Backlog             │
│  ─ Impact vs effort matrix               │
│  ─ Time-to-first-deploy reduction        │
│  ─ Security compliance automation        │
└──────────────────┬───────────────────────┘
                   │
┌──────────────────▼───────────────────────┐
│  Measure & Iterate                       │
│  ─ Adoption metrics                      │
│  ─ NPS for platform teams                │
│  ─ p95 time-to-deploy                    │
└──────────────────────────────────────────┘
```

### 3.2 Platform Roadmap Framework

```
Platform Maturity Levels:
Level 1 - REACTIVE
  ─ Scripts and wikis
  ─ Manual handoffs
  ─ No self-service

Level 2 - STANDARDIZED  
  ─ Shared Terraform modules
  ─ Common CI/CD templates
  ─ Documented runbooks

Level 3 - SELF-SERVICE
  ─ Developer portal
  ─ Infrastructure on demand
  ─ Automated compliance

Level 4 - PRODUCT
  ─ SLAs for platform services
  ─ Usage analytics
  ─ Developer feedback loop

Level 5 - INTELLIGENT
  ─ Cost optimization auto-suggestions
  ─ Predictive scaling recommendations
  ─ AI-assisted troubleshooting
```

### 3.3 Platform SLAs

Platforms must define SLAs like any product:

| Platform Service | Availability SLA | Response Time SLA |
|----------------|-----------------|-------------------|
| Service catalog API | 99.9% | < 200ms |
| Pipeline execution start | 99.5% | < 30s |
| Infra provisioning (k8s namespace) | 99% | < 5 min |
| Secret retrieval | 99.99% | < 100ms |
| Build artifact storage | 99.9% | < 500ms |

---

## 4. Golden Paths & Paved Roads

### 4.1 What is a Golden Path

A **Golden Path** (or Paved Road) is the recommended, opinionated, pre-built template for building and operating a service — embedding best practices for security, observability, deployment, and operations automatically.

```
Golden Path for a New Microservice:
Developer runs: platform new-service --name payments --type java-spring

Scaffolded output:
payments/
├── src/
│   └── main/java/...          # Service code scaffold
├── Dockerfile                  # Optimized multi-stage build
├── kubernetes/
│   ├── deployment.yaml         # Resource limits, health probes
│   ├── service.yaml
│   └── hpa.yaml               # Auto-scaling config
├── .github/workflows/
│   └── ci-cd.yaml             # Build, test, scan, deploy
├── observability/
│   ├── grafana-dashboard.json  # Pre-built dashboard
│   └── alerts.yaml            # Sensible alert rules
└── backstage-catalog.yaml      # Auto-registers in catalog
```

### 4.2 Golden Path Components

```
┌─────────────────────────────────────────────────────┐
│                   Golden Path Stack                  │
├──────────────┬──────────────┬────────────────────────┤
│   Runtime    │   CI/CD      │    Observability        │
│ ─ Container  │ ─ Lint/test  │ ─ Logs (structured)    │
│   base image │ ─ SAST scan  │ ─ Metrics (Prom)       │
│ ─ JVM flags  │ ─ Image scan │ ─ Traces (OTel)        │
│ ─ Health     │ ─ SBOM gen   │ ─ Dashboard template   │
│   endpoints  │ ─ GitOps PR  │ ─ On-call runbook      │
├──────────────┼──────────────┼────────────────────────┤
│   Security   │   Networking │    Infrastructure       │
│ ─ mTLS auto  │ ─ Mesh sidecar│ ─ Auto namespace      │
│ ─ RBAC       │ ─ Ingress    │ ─ Resource quotas      │
│ ─ Secrets    │ ─ Rate limit │ ─ Network policies     │
│   injection  │ ─ Retry/CB   │ ─ PodDisruptionBudget  │
└──────────────┴──────────────┴────────────────────────┘
```

### 4.3 Escape Hatches

Golden paths must have escape hatches — not prisons:

```yaml
# Example: Override golden path defaults in service config
platform:
  golden-path: java-spring-v3
  overrides:
    # Team needs custom JVM flags
    jvm-flags: "-Xmx4g -XX:+UseZGC"
    # Team uses different logging format
    log-format: custom
    # Team manages their own Grafana dashboard
    observability.grafana: self-managed
  
  # Must justify non-standard choices (for audit)
  justifications:
    jvm-flags: "High-memory NLP model loading - JIRA-12345"
    log-format: "Legacy log aggregation contract - approved by arch board"
```

---

## 5. Backstage & Service Catalogs

### 5.1 Backstage Architecture

[Backstage](https://backstage.io) is Spotify's open-source developer portal platform:

```
Backstage Architecture:
┌─────────────────────────────────────────────────────┐
│                  Backstage Frontend                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │ Software │  │   TechDocs│  │  Cost Dashboard  │  │
│   │ Catalog  │  │(Docs site)│  │   (FinOps)       │  │
│   └──────────┘  └──────────┘  └──────────────────┘  │
│   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│   │  CI/CD   │  │  Infra   │  │ Security Scores  │  │
│   │  Plugin  │  │  Plugin  │  │  Plugin          │  │
│   └──────────┘  └──────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                  Backstage Backend                   │
│   ─ Catalog service                                  │
│   ─ Plugin API aggregator                            │
│   ─ Auth providers (OIDC, SAML)                      │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
   ┌───▼───┐  ┌──▼───┐  ┌───▼──┐  ┌───▼────┐
   │GitHub │  │Jira  │  │ AWS  │  │PagerD. │
   │ API   │  │ API  │  │ API  │  │  API   │
   └───────┘  └──────┘  └──────┘  └────────┘
```

### 5.2 Catalog Entity YAML

```yaml
# backstage-catalog.yaml — lives in every service repo
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: payments-service
  description: Handles payment processing and reconciliation
  annotations:
    github.com/project-slug: org/payments-service
    pagerduty.com/integration-key: abc123
    grafana/dashboard-selector: "service=payments"
    jenkins.io/job-full-name: payments/main
  tags:
    - java
    - payments
    - critical-path
  links:
    - url: https://runbook.company.com/payments
      title: Runbook
spec:
  type: service
  lifecycle: production
  owner: team-payments
  system: payment-platform
  dependsOn:
    - component:user-service
    - resource:payments-postgres-db
  providesApis:
    - payments-api
```

### 5.3 Service Catalog Benefits

```
Service Catalog as Single Source of Truth:
┌──────────────────────────────────────────────┐
│  For Developers                              │
│  ─ Discover what APIs exist                  │
│  ─ Find service owners quickly               │
│  ─ Access runbooks and docs                  │
├──────────────────────────────────────────────┤
│  For On-Call Engineers                       │
│  ─ Understand service dependencies           │
│  ─ Find who to escalate to                   │
│  ─ Access dashboards directly                │
├──────────────────────────────────────────────┤
│  For Architects                              │
│  ─ Visualize system topology                 │
│  ─ Identify orphaned/deprecated services     │
│  ─ Track tech debt across org                │
├──────────────────────────────────────────────┤
│  For Managers                                │
│  ─ Team ownership clarity                    │
│  ─ Technology diversity metrics              │
│  ─ Security score trends                     │
└──────────────────────────────────────────────┘
```

---

## 6. Self-Service Infrastructure

### 6.1 Crossplane — Infrastructure as Code via K8s CRDs

[Crossplane](https://crossplane.io) extends Kubernetes to manage cloud infrastructure as native k8s resources:

```yaml
# Developer creates a database via kubectl — no AWS console needed
apiVersion: database.platform.company.com/v1alpha1
kind: PostgresDatabase
metadata:
  name: payments-db
  namespace: team-payments
spec:
  version: "15"
  size: medium           # Maps to RDS db.t3.large
  region: us-east-1
  backup:
    enabled: true
    retentionDays: 14
  allowedFrom:
    - payments-service
---
# Platform team defines what "medium" means via CompositeResourceDefinition
# Developer never touches AWS directly
```

```
Crossplane Request Flow:
Developer          K8s API          Crossplane         AWS
   │                  │                  │               │
   │ kubectl apply    │                  │               │
   ├─────────────────►│                  │               │
   │                  │ CRD created      │               │
   │                  ├─────────────────►│               │
   │                  │                  │ CreateDBInstance
   │                  │                  ├──────────────►│
   │                  │                  │ DB ARN        │
   │                  │                  │◄──────────────┤
   │                  │ Status: Ready    │               │
   │                  │◄─────────────────┤               │
   │ kubectl get      │                  │               │
   ├─────────────────►│                  │               │
   │ Ready / endpoint │                  │               │
   │◄─────────────────┤                  │               │
```

### 6.2 Environment-on-Demand

```yaml
# Developer creates a full ephemeral environment for a PR
# platform/environments/pr-1234.yaml
apiVersion: platform.company.com/v1
kind: Environment
metadata:
  name: pr-1234
spec:
  type: ephemeral
  ttl: 72h                    # Auto-deleted after 72 hours
  pullRequest: 1234
  services:
    - name: payments-service
      image: payments:pr-1234  # PR build image
    - name: user-service
      image: user-service:main # Stable dependency
  databases:
    - name: payments-db
      snapshot: staging-latest  # Pre-seeded with staging data
  ingress:
    hostname: pr-1234.preview.company.com
    auth: github-pr-members     # Only PR contributors can access
```

### 6.3 Infrastructure Vending Machine Pattern

```
Infra Vending Machine:

Developer Request                  Platform Response
─────────────────                  ─────────────────

"I need a message queue"    ──►    Kafka topic created
                                   + consumer group
                                   + monitoring alerts
                                   + runbook link

"I need object storage"     ──►    S3 bucket created
                                   + lifecycle policy
                                   + encryption enabled
                                   + access via service account

"I need a cache"            ──►    Redis cluster created
                                   + maxmemory-policy set
                                   + eviction alerts
                                   + connection string in Vault

All with:
  ✓ Tagging (team, cost-center, env)
  ✓ Backup policy
  ✓ Compliance controls
  ✓ Cost attribution
```

---

## 7. Developer Experience (DevEx) Metrics

### 7.1 SPACE Framework

The [SPACE framework](https://queue.acm.org/detail.cfm?id=3454124) measures developer productivity across 5 dimensions:

| Dimension | Description | Example Metrics |
|-----------|-------------|-----------------|
| **S**atisfaction | Developer happiness and fulfillment | NPS, survey scores |
| **P**erformance | Outcomes and quality of work | Deployment frequency, defect rate |
| **A**ctivity | Actions and output | PRs merged, builds run |
| **C**ommunication | Collaboration quality | PR review time, docs coverage |
| **E**fficiency | Minimal friction and interruptions | CI duration, time-to-deploy |

### 7.2 Key Platform Metrics

```python
# Platform Health Dashboard KPIs

metrics = {
    # Developer Productivity
    "time_to_first_commit": "< 30 min for new hire",
    "time_to_first_deploy": "< 2 hours for new service",
    "ci_pipeline_p95_duration": "< 10 min",
    "pr_merge_to_production_lead_time": "< 1 hour",

    # Platform Reliability
    "platform_availability": "> 99.9%",
    "self_service_success_rate": "> 95%",
    "support_ticket_volume": "trending down",

    # Adoption & Engagement
    "golden_path_adoption_rate": "> 80%",
    "service_catalog_coverage": "> 90% of services",
    "docs_coverage": "> 70% of APIs",

    # Business Impact
    "deployment_frequency": "multiple/day",
    "change_failure_rate": "< 5%",
    "mttr_minutes": "< 30 min",
}
```

### 7.3 Developer Friction Index

Regularly measure and eliminate friction points:

```
Developer Friction Audit:

ONBOARDING                              SCORE
─────────────────────────────────────────────
New laptop setup time          3h  ██░░░░ 4/10
Dev environment setup          45m ████░░ 7/10
First PR merged                2d  ██░░░░ 4/10

DAILY DEVELOPMENT
─────────────────────────────────────────────
Local build time               2m  ████░░ 7/10
CI pipeline wait time          12m ██░░░░ 4/10
PR review turnaround           4h  ███░░░ 6/10
Deploy to staging              auto ██████ 10/10

OPERATIONS
─────────────────────────────────────────────
Debug prod issue (trace→log)   15m ████░░ 7/10
Provision new infra (DB)       2d  ░░░░░░ 2/10  ← TARGET
Certificate rotation           manual ░░ 2/10  ← TARGET
```

---

## 8. CI/CD Platform Architecture

### 8.1 Unified Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CI/CD Platform                        │
├──────────────────────────────────────────────────────────┤
│  Trigger Layer                                          │
│  ─ PR opened → run tests + SAST                         │
│  ─ Merge to main → build + push to registry             │
│  ─ Tag created → promote to production                  │
├──────────────────────────────────────────────────────────┤
│  Build Pipeline (GitHub Actions / Tekton / Jenkins)     │
│                                                         │
│  Lint → Unit Test → Integration → Build → SAST → SBOM  │
│                                    │                    │
│                             Push to Registry            │
│                          (Harbor / ECR / GCR)           │
├──────────────────────────────────────────────────────────┤
│  Deploy Pipeline (GitOps — ArgoCD / Flux)               │
│                                                         │
│  dev ──► staging ──► canary (5%) ──► production (100%)  │
│          auto        with smoke     with approval gate  │
│                      tests          for critical svc    │
├──────────────────────────────────────────────────────────┤
│  Quality Gates                                          │
│  ─ Coverage > 80%              ─ No critical CVEs       │
│  ─ All tests pass              ─ Performance regression │
│  ─ SAST no new HIGH/CRITICAL     within 5% threshold    │
└─────────────────────────────────────────────────────────┘
```

### 8.2 GitOps with ArgoCD

```yaml
# ArgoCD Application (managed by platform team)
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: payments-service-production
  namespace: argocd
spec:
  project: payments-team
  source:
    repoURL: https://github.com/org/payments-service
    targetRevision: main
    path: kubernetes/overlays/production
  destination:
    server: https://prod-cluster.company.com
    namespace: payments
  syncPolicy:
    automated:
      prune: true        # Remove resources deleted from Git
      selfHeal: true     # Revert manual kubectl changes
    syncOptions:
      - CreateNamespace=true
  revisionHistoryLimit: 10
```

### 8.3 Progressive Delivery

```
Canary Deployment Strategy (using Argo Rollouts):

Traffic split over time:
  Step 1:  5% → canary    ─→  run smoke tests + 5 min bake time
  Step 2: 20% → canary    ─→  monitor error rate (abort if > 1%)
  Step 3: 50% → canary    ─→  monitor p99 latency (abort if degrades)
  Step 4: 100% → canary   ─→  promotion complete

Auto-abort conditions:
  error_rate > 1%                 → rollback immediately
  p99_latency increase > 20%      → rollback immediately
  custom_metric_breach            → rollback immediately

Analysis template (metrics from Prometheus):
  success_rate = sum(rate(http_requests_total{code=~"2.."}[5m]))
               / sum(rate(http_requests_total[5m]))
  threshold: > 0.99
```

---

## 9. Platform Team Topology

### 9.1 Team Topologies Alignment

Based on [Team Topologies](https://teamtopologies.com) by Skelton & Pais:

```
Team Topologies in Platform Engineering:

┌────────────────────────────────────────────────────────┐
│           Stream-Aligned Teams (Product Teams)         │
│   ─ Own their services end-to-end                      │
│   ─ Consume platform via self-service                  │
│   ─ Interact: X-as-a-Service                          │
└──────────────────────┬─────────────────────────────────┘
                       │ consume
┌──────────────────────▼─────────────────────────────────┐
│              Platform Team                             │
│   ─ Owns IDP, CI/CD platform, observability stack      │
│   ─ Reduces cognitive load for stream-aligned teams    │
│   ─ Interact: X-as-a-Service + collaboration           │
└──────────────────────┬─────────────────────────────────┘
                       │ enables
┌──────────────────────▼─────────────────────────────────┐
│           Enabling Teams (temporary)                   │
│   ─ Chaos engineering guild                            │
│   ─ Security champions                                 │
│   ─ Help teams adopt new platform capabilities         │
└────────────────────────────────────────────────────────┘
```

### 9.2 Platform Team Responsibilities

```
┌─────────────────────────────────────────────────────┐
│                Platform Team Owns                   │
├───────────────────────┬─────────────────────────────┤
│  YES (Platform Owns)  │   NO (Team Owns)            │
├───────────────────────┼─────────────────────────────┤
│ CI/CD templates       │ Business logic              │
│ Base container images │ Application config          │
│ k8s cluster           │ Feature flags               │
│ Observability stack   │ Domain-specific dashboards  │
│ Secret management     │ Service-level alerts        │
│ Service mesh config   │ Application databases       │
│ Network policies      │ Team-level Jira workflows   │
│ Security scanning     │ Load testing scenarios      │
└───────────────────────┴─────────────────────────────┘
```

### 9.3 Platform Engineering Anti-Patterns

| Anti-Pattern | Problem | Solution |
|-------------|---------|----------|
| **Platform as Gatekeeper** | Teams blocked on platform tickets | Self-service APIs, async review |
| **One-size-fits-all** | Golden path too rigid, teams avoid it | Escape hatches + customization layers |
| **No dogfooding** | Platform team doesn't use their own platform | Platform team deploys services on IDP |
| **Ignoring adoption metrics** | Build it and they won't come | Treat adoption as a KPI |
| **No SLA for platform** | Unreliable platform destroys trust | Define and publish platform SLAs |
| **Rebuilding everything** | NIH syndrome, ignore OSS | Curate OSS + thin internal layer |

---

## 10. Practical Examples

### 10.1 New Service Onboarding — Before vs After

```
BEFORE Platform Engineering:
Week 1: Request cloud account → wait for approval
Week 2: Manual VPC setup, security groups
Week 3: Configure CI/CD pipeline from scratch
Week 4: Set up monitoring, alerts, dashboards
Week 5: Security review, credentials setup
Week 6: First deployment to staging
Total: ~6 weeks, 50+ manual steps

AFTER Platform Engineering:
Day 1: Run `platform new-service --name my-service`
       → Git repo scaffolded
       → Kubernetes namespace created
       → CI/CD pipeline active
       → Monitoring + alerts pre-configured
       → Security baselines applied
Day 1: First deployment to staging
Total: ~2 hours, 1 command + ~5 config edits
```

### 10.2 Platform Tech Stack (Reference Implementation)

```yaml
# Reference Platform Stack

developer_portal: backstage          # Service catalog + docs
source_control: github               # Code + GitOps manifests

ci:
  engine: github-actions             # Build, test, scan
  image_registry: harbor             # Self-hosted OCI registry
  artifact_storage: nexus            # Maven/npm/Python packages
  sast: semgrep + snyk               # Security scanning
  sbom: syft                         # Software bill of materials

cd:
  engine: argocd                     # GitOps deployment
  progressive_delivery: argo-rollouts # Canary/blue-green
  config_management: kustomize       # k8s overlays

infrastructure:
  orchestration: kubernetes (EKS)
  infra_as_code: terraform + crossplane
  secrets: hashicorp-vault
  service_mesh: istio
  network_policy: cilium

observability:
  metrics: prometheus + thanos        # Long-term storage
  visualization: grafana
  logging: loki + promtail
  tracing: jaeger / tempo
  alerting: alertmanager + pagerduty
  synthetic: blackbox-exporter

platform_engineering_tools:
  ephemeral_envs: vcluster + telepresence
  cost_visibility: opencost
  policy_enforcement: open-policy-agent (OPA)
  developer_cli: custom platform CLI
```

### 10.3 Developer CLI Example

```bash
# The platform CLI — the developer's primary interface

# Scaffold a new service with golden path
$ platform new-service \
    --name payments \
    --type java-spring \
    --team payments-team \
    --tier critical

# Create ephemeral environment for PR
$ platform env create --pr 1234 --ttl 24h

# Provision infrastructure
$ platform infra create postgres \
    --name payments-db \
    --size medium \
    --env staging

# Check service health across environments
$ platform status payments-service

  Environment  Version    Health   Deploy Time    Error Rate
  ─────────────────────────────────────────────────────────
  dev          abc123     ✅ OK    2 hours ago    0.01%
  staging      abc123     ✅ OK    1 hour ago     0.02%
  production   xyz789     ✅ OK    2 days ago     0.05%

# Promote to production (triggers ArgoCD sync)
$ platform deploy payments-service --to production --version abc123
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **IDP** | Unify tooling into one self-service portal to reduce cognitive load |
| **Platform as Product** | Treat developers as customers; measure adoption and satisfaction |
| **Golden Paths** | Embed security, observability, and compliance by default — with escape hatches |
| **Backstage** | Open-source foundation for service catalogs and developer portals |
| **Crossplane** | Kubernetes-native infrastructure abstraction for self-service provisioning |
| **DevEx Metrics** | SPACE framework + DORA metrics to measure platform impact |
| **Team Topologies** | Platform team enables stream-aligned teams via X-as-a-Service |
| **Anti-Patterns** | Avoid gatekeeper model; build for adoption, not compliance |
