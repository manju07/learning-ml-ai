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

---

*For ML/AI content, see the parent [tech-learning](../README.md) index.*
