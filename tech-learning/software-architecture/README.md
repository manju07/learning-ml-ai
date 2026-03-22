# Software Architecture & Principal Engineer Tech Stack

Comprehensive documentation for **Software Architects** and **Principal Engineers** covering distributed systems, microservices, scalability, cloud infrastructure, and production-grade patterns.

## Table of Contents

| Document | Description |
|----------|-------------|
| [01-distributed-systems.md](./01-distributed-systems.md) | CAP theorem, consistency models, consensus (Raft/Paxos), distributed transactions |
| [02-microservices-architecture.md](./02-microservices-architecture.md) | Microservices patterns, service mesh, decomposition, deployment |
| [03-database-scalability.md](./03-database-scalability.md) | SQL vs NoSQL, sharding, replication, caching strategies |
| [04-event-driven-architecture.md](./04-event-driven-architecture.md) | Event sourcing, CQRS, message brokers, saga pattern |
| [05-system-design-patterns.md](./05-system-design-patterns.md) | Load balancing, rate limiting, circuit breaker, bulkhead |
| [06-api-design-comprehensive.md](./06-api-design-comprehensive.md) | REST, GraphQL, gRPC—design principles and trade-offs |
| [07-cloud-infrastructure-kubernetes.md](./07-cloud-infrastructure-kubernetes.md) | Kubernetes, containers, IaC, multi-region, serverless |
| [08-security-architecture.md](./08-security-architecture.md) | Zero Trust, OAuth2, mTLS, secrets management |
| [09-observability-sre.md](./09-observability-sre.md) | Metrics, tracing, logging, SRE, SLIs/SLOs |
| [10-domain-driven-design.md](./10-domain-driven-design.md) | DDD, bounded contexts, aggregates, tactical patterns |
| [11-performance-engineering.md](./11-performance-engineering.md) | JVM tuning, profiling, load testing, capacity planning |
| [12-big-data-streaming.md](./12-big-data-streaming.md) | Apache Pulsar, Apache Spark, stream processing, data pipelines |
| [13-platform-engineering.md](./13-platform-engineering.md) | Internal Developer Platforms, golden paths, Backstage, DevEx metrics |
| [14-data-mesh-architecture.md](./14-data-mesh-architecture.md) | Data mesh principles, data products, data contracts, lakehouse, medallion architecture |
| [15-resilience-chaos-engineering.md](./15-resilience-chaos-engineering.md) | Chaos engineering, game days, failure injection, blast radius, Chaos Mesh |
| [16-architectural-decision-making.md](./16-architectural-decision-making.md) | ADRs, tech radar, fitness functions, evolutionary architecture, tech debt |
| [17-ml-ai-systems-architecture.md](./17-ml-ai-systems-architecture.md) | MLOps, feature stores, model serving, LLM systems, RAG, vector databases, drift detection |
| [18-multi-tenancy-saas-architecture.md](./18-multi-tenancy-saas-architecture.md) | Silo/pool/bridge models, data isolation, tenant provisioning, RBAC, metering & billing |
| [19-cost-engineering-finops.md](./19-cost-engineering-finops.md) | FinOps framework, rightsizing, Savings Plans, Spot, unit economics, cost-aware design |

## Learning Path for Architects

### Foundation
1. **Distributed Systems** — Understand CAP, consistency, and failure modes
2. **Database Scalability** — Replication, sharding, and caching
3. **API Design** — REST, GraphQL, gRPC trade-offs

### Intermediate
4. **Microservices** — Decomposition, service mesh, resilience
5. **Event-Driven Architecture** — Async messaging, CQRS, saga
6. **System Design Patterns** — Load balancing, circuit breaker, rate limiting

### Advanced
7. **Cloud & Kubernetes** — Container orchestration, IaC, multi-region
8. **Security Architecture** — Zero Trust, OAuth2, mTLS
9. **Observability & SRE** — SLIs/SLOs, tracing, incident response
10. **Domain-Driven Design** — Strategic and tactical DDD

### Principal Engineer / Staff+
11. **Performance Engineering** — JVM tuning, profiling, load testing, capacity planning
12. **Big Data & Streaming** — Pulsar, Spark, stream processing patterns
13. **Platform Engineering** — IDP, golden paths, Backstage, developer experience
14. **Data Mesh** — Data products, data contracts, lakehouse, medallion architecture
15. **Resilience & Chaos Engineering** — Chaos experiments, game days, failure injection
16. **Architectural Decision-Making** — ADRs, tech radar, fitness functions, tech debt

### Emerging & Cross-Cutting
17. **ML/AI Systems Architecture** — MLOps, LLM systems, RAG, feature stores, model serving
18. **Multi-Tenancy & SaaS** — Tenant isolation, provisioning, billing, noisy neighbor
19. **Cost Engineering & FinOps** — Cloud cost optimization, unit economics, Savings Plans

## Key Concepts Overview

| Topic | Concepts |
|-------|----------|
| **Distributed Systems** | CAP, PACELC, eventual consistency, Raft, 2PC, Saga |
| **Microservices** | BFF, API Gateway, service discovery, circuit breaker |
| **Databases** | Read replicas, sharding, eventual consistency, caching |
| **Event-Driven** | Event sourcing, CQRS, Kafka, outbox pattern |
| **APIs** | Resource design, versioning, idempotency, backward compatibility |
| **Cloud** | K8s, Helm, Terraform, multi-AZ, serverless |
| **Security** | OAuth2/OIDC, JWT, mTLS, RBAC, secrets rotation |
| **Observability** | OpenTelemetry, Prometheus, Grafana, distributed tracing |
| **Platform Engineering** | IDP, Backstage, golden paths, Crossplane, DevEx metrics |
| **Data Mesh** | Data products, data contracts, Delta/Iceberg, medallion architecture |
| **Chaos Engineering** | Steady state, blast radius, Chaos Mesh, game days, fitness functions |
| **Architecture Decisions** | ADRs, tech radar, ATAM, evolutionary architecture, fitness functions |
| **ML/AI Systems** | Feature stores, MLOps, LLM serving, RAG, vLLM, drift detection |
| **Multi-Tenancy** | Silo/pool/bridge, RLS, schema-per-tenant, tenant provisioning, metering |
| **FinOps** | Rightsizing, Savings Plans, Spot, unit economics, cost attribution, Infracost |

---

*For ML/AI content, see the parent [tech-learning](../README.md) index.*
