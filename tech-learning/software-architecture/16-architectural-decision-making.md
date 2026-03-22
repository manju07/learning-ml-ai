# Architectural Decision-Making for Principal Engineers

## Table of Contents
1. [Introduction](#1-introduction)
2. [Architecture Decision Records (ADRs)](#2-architecture-decision-records-adrs)
3. [Architecture Review Process](#3-architecture-review-process)
4. [Tech Radar](#4-tech-radar)
5. [Evolutionary Architecture & Fitness Functions](#5-evolutionary-architecture--fitness-functions)
6. [Technical Debt Management](#6-technical-debt-management)
7. [Architecture Governance](#7-architecture-governance)
8. [Communicating Architecture](#8-communicating-architecture)
9. [Trade-off Analysis Frameworks](#9-trade-off-analysis-frameworks)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

The **Principal Engineer / Software Architect** role requires more than designing systems — it requires making and communicating *decisions* that outlast any single sprint, influencing how hundreds of engineers build software for years to come.

### 1.1 The Decision-Making Problem

```
Architecture Decisions Are Hard Because:

SHORT-TERM PRESSURE             LONG-TERM IMPACT
────────────────────            ────────────────────
"Ship the feature now"    vs.   "Wrong DB choice = 2-year migration"
"Copy what team X did"    vs.   "Their context is very different"
"This worked before"      vs.   "Context has fundamentally changed"
"The vendor says so"      vs.   "Vendor lock-in risk"

Common Failure Modes:
  ─ Decisions made informally, lost in Slack threads
  ─ Context not captured → future engineers don't know WHY
  ─ No review → different teams make conflicting choices
  ─ No retirement plan → technologies never sunset
  ─ Analysis paralysis → decision delayed, team blocked
```

### 1.2 Architect's Decision Spectrum

```
NOT all decisions need formal treatment:

HIGH FORMALITY (ADR + Architecture Review)
  ─ New database technology adoption
  ─ Cross-cutting infrastructure changes
  ─ New communication protocols (REST → gRPC)
  ─ Fundamental decomposition choices
  ─ Decisions that affect 3+ teams

MEDIUM FORMALITY (ADR, lightweight review)
  ─ New framework adoption within a domain
  ─ Significant performance architecture changes
  ─ New external vendor integration

LOW FORMALITY (documented in PR or ticket)
  ─ Library version upgrades
  ─ Internal implementation patterns
  ─ Single-service design choices
  ─ Reversible decisions with low blast radius
```

---

## 2. Architecture Decision Records (ADRs)

### 2.1 What is an ADR

An **Architecture Decision Record (ADR)** is a short document capturing an important architectural decision, its context, the options considered, and the rationale for the chosen approach.

> Coined by Michael Nygard. Template popularized by [adr.github.io](https://adr.github.io).

### 2.2 ADR Format (Nygard Template)

```markdown
# ADR-0042: Use Apache Kafka for Asynchronous Order Events

**Date**: 2024-03-15  
**Status**: Accepted  
**Deciders**: Principal Engineer (Payments), Arch Council  
**Context**: [link to design doc]

## Context

The order processing system currently uses synchronous REST calls
between services. As order volume scales to 50K/min, we are
experiencing:
- Checkout P99 latency > 3s (SLA: < 1s) due to downstream coupling
- Fragile cascading failures when inventory service is slow
- No audit trail of order state transitions

We need to decouple order processing from downstream services
(inventory, fulfillment, notifications).

## Decision Drivers
- Must support 100K events/min at peak
- Exactly-once delivery required for financial events
- Replay capability needed for audit and debugging
- Team has existing Kafka expertise (2 engineers certified)

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **Kafka** | High throughput, replay, exactly-once, mature | Ops complexity, ZooKeeper (pre-3.x) |
| **RabbitMQ** | Simple, good DX, low ops overhead | Limited replay, lower throughput |
| **AWS SQS/SNS** | Managed, low ops | No replay, vendor lock-in, limited ordering |
| **Pulsar** | Multi-tenancy, tiered storage | Smaller ecosystem, less team familiarity |

## Decision

**Use Apache Kafka** (Confluent Cloud managed).

Rationale:
- Replay capability is non-negotiable for financial audit
- Confluent Cloud eliminates ops overhead concern
- Exactly-once delivery semantics match financial requirements
- Team expertise reduces onboarding risk

## Consequences

**Positive:**
- Order service decoupled from downstream → P99 improves
- Full event replay for audit and debugging
- Foundation for future event-sourcing use cases

**Negative:**
- Confluent Cloud cost: ~$2K/month (approved by finance)
- All consumers must handle idempotency
- Eventual consistency: order status may lag 100-500ms

**Risks:**
- Consumer lag buildup during deployment windows → mitigation: consumer lag alerts
- Schema evolution complexity → mitigation: schema registry + backward compatibility policy

## Alternatives if This Fails

If Confluent Cloud pricing grows prohibitive (>$20K/month),
evaluate self-managed Kafka on Kubernetes or migrate to Pulsar.

## References
- [Kafka vs RabbitMQ Benchmark Results](https://...)
- [Order Event Schema v1](https://...)
- [ADR-0039: Event-Driven Architecture Strategy](./0039-event-driven-strategy.md)
```

### 2.3 ADR Status Lifecycle

```
ADR Statuses:

PROPOSED → ACCEPTED → DEPRECATED → SUPERSEDED
    │           │
    └──► REJECTED

PROPOSED:    ADR drafted, under discussion
ACCEPTED:    Decision made, implementation approved
DEPRECATED:  Technology/pattern still in use but discouraged
SUPERSEDED:  Replaced by newer ADR (link to successor)
REJECTED:    Considered but not adopted (still valuable to keep!)
```

### 2.4 ADR Repository Structure

```
project-root/
└── docs/
    └── architecture/
        └── decisions/
            ├── README.md           ← Index of all ADRs
            ├── 0001-record-architecture-decisions.md
            ├── 0002-use-postgresql-as-primary-database.md
            ├── 0003-adopt-kubernetes-for-orchestration.md
            ├── ...
            └── 0042-use-kafka-for-order-events.md

# Use adr-tools CLI to manage ADRs:
$ adr new "Use GraphQL for mobile API"    # Creates next numbered ADR
$ adr list                                # Lists all ADRs with status
$ adr link 0042 "is superseded by" 0051  # Link related ADRs
```

### 2.5 ADR Anti-Patterns

| Anti-Pattern | Problem | Fix |
|-------------|---------|-----|
| **Decision without context** | Future readers don't know WHY | Always include the problem statement |
| **Documenting after the fact** | ADR becomes archaeology | Write ADR before or during decision |
| **Too long** | Nobody reads it | Max 1-2 pages; link to full design doc |
| **No alternatives** | Looks like one option was considered | Document rejected options + why |
| **Never superseded** | ADRs become stale artifacts | Review ADRs when technology changes |
| **Just for architects** | Developers don't know decisions exist | ADRs in repo, linked from README |

---

## 3. Architecture Review Process

### 3.1 When to Request Architecture Review

```
Architecture Review Triggers (any of these requires review):

□ New data store technology (first time using Redis, Cassandra, etc.)
□ New communication protocol or API style
□ New external service dependency (vendor)
□ System decomposition (splitting or merging services)
□ Cross-team interface changes (API contracts)
□ Non-reversible infrastructure decisions
□ Estimated migration effort > 2 engineer-weeks
□ Security architecture changes
□ Compliance-related design choices (GDPR, PCI, HIPAA)
```

### 3.2 Architecture Review Board (ARB) vs RFC Process

```
Two models for architecture review:

ARB (Architecture Review Board):
  ─ Standing committee (principals + architects)
  ─ Formal submission + meeting
  ─ Good for: org-wide standards, cross-cutting decisions
  ─ Risk: becomes bottleneck if too many decisions routed here

RFC (Request for Comments) — async:
  ─ Author posts RFC document (like ADR but pre-decision)
  ─ Comment period: 1-2 weeks
  ─ Lazy consensus: no objections = accepted
  ─ Good for: team-level decisions, technical proposals
  ─ Risk: low participation → decisions made in vacuum

Recommendation: Use RFC for most decisions, ARB for org-wide ones
```

### 3.3 RFC Template

```markdown
# RFC-0089: Adopt OpenTelemetry as Standard Instrumentation

**Author**: @jane.smith (Principal Engineer, Platform)  
**Created**: 2024-03-01  
**Last Updated**: 2024-03-10  
**Status**: OPEN (comment period ends 2024-03-15)  
**Affects**: All engineering teams  

## Summary

Standardize on OpenTelemetry (OTel) SDK for metrics, traces, and
logs across all services, replacing our current fragmented approach
(custom metrics client + Zipkin + ELK per-team configs).

## Motivation

Currently:
- 3 different tracing libraries in use (Zipkin, Jaeger client, custom)
- No standard metrics naming convention → Grafana dashboards non-reusable
- Logs not correlated with traces → 30-min debug sessions for 5-min problems

## Proposed Solution
[Technical details...]

## Migration Plan
[Step-by-step migration...]

## Open Questions
1. Should we self-host OTel collector or use vendor (Datadog/Honeycomb)?
2. Do we need to maintain existing Zipkin data for compliance?

## Feedback Requested By
- Platform Team: Collector hosting decision
- Security Team: Data egress implications
- All teams: Migration timeline feasibility

## Discussion
[Comments added by reviewers in GitHub PR]
```

---

## 4. Tech Radar

### 4.1 What is a Tech Radar

The [ThoughtWorks Tech Radar](https://www.thoughtworks.com/radar) is a framework for tracking technology adoption across four rings:

```
Tech Radar Rings:

        ADOPT
    ────────────
      Technologies we have high confidence in;
      actively recommend for use.

        TRIAL
    ────────────
      Worth pursuing — try them on projects that
      can handle the risk. Important to understand
      and build capability.

        ASSESS
    ────────────
      Worth exploring to understand their impact
      on your organization. Worth investing research
      and prototype effort.

        HOLD
    ────────────
      Technologies we think are not (yet) worth
      using for new projects. Proceed with caution.
      May be superseded or have concerns.

Four Quadrants:
  ─ Techniques (patterns, practices)
  ─ Platforms (infra, cloud services)
  ─ Tools (software, frameworks)
  ─ Languages & Frameworks
```

### 4.2 Building Your Company Tech Radar

```
Example Internal Tech Radar (2024):

LANGUAGES & FRAMEWORKS
  ADOPT:  Java 21, Python 3.12, TypeScript, React
  TRIAL:  Kotlin, Go (for CLIs and infra tools)
  ASSESS: Rust (for performance-critical services)
  HOLD:   Java 8, Python 2, AngularJS, jQuery

PLATFORMS
  ADOPT:  AWS EKS, Terraform, PostgreSQL, Kafka, Redis
  TRIAL:  Apache Iceberg, Crossplane, ArgoCD
  ASSESS: WASM (edge computing), eBPF (observability)
  HOLD:   EC2 (prefer containers), MongoDB (prefer Postgres)

TOOLS
  ADOPT:  Docker, Helm, Prometheus+Grafana, OpenTelemetry
  TRIAL:  Backstage, Chaos Mesh, dbt
  ASSESS: Bazel (monorepo builds), Buf (Protobuf tooling)
  HOLD:   Jenkins (migrate to GitHub Actions), Ansible

TECHNIQUES
  ADOPT:  GitOps, Event sourcing, ADRs, Golden paths
  TRIAL:  Data Mesh, Platform Engineering, eBPF tracing
  ASSESS: AI-assisted code review, WebAssembly modules
  HOLD:   Shared databases between services, stored procedures
```

### 4.3 Tech Radar Governance

```
Tech Radar Update Process (quarterly):

1. Collect nominations (anyone can nominate a blip)
2. Tech leads review nominations async (1 week)
3. Tech Council meeting: debate and vote on placement
4. Publish updated radar with rationale
5. Communicate to engineering org via:
   ─ All-hands presentation (highlight changes)
   ─ Engineering newsletter
   ─ Update in developer portal (Backstage)
   ─ ADRs linking to radar for new adoptions

Ring Change Examples:
  Kafka: ASSESS (2021) → TRIAL (2022) → ADOPT (2023)
         Rationale: proven in 5 production services at scale

  MongoDB: ADOPT (2018) → HOLD (2022)
           Rationale: data integrity issues, Postgres covers use cases
```

---

## 5. Evolutionary Architecture & Fitness Functions

### 5.1 What is Evolutionary Architecture

Coined by Neal Ford, Rebecca Parsons & Patrick Kua in [Building Evolutionary Architectures](https://www.oreilly.com/library/view/building-evolutionary-architectures/9781491986356/), an evolutionary architecture supports **guided, incremental change as a first principle**.

```
Traditional Architecture:    Evolutionary Architecture:
───────────────────────      ───────────────────────────
Design upfront (big design)  Design for change (enable evolution)
Architecture as destination  Architecture as ongoing practice
Manual compliance checks     Automated fitness functions
Guard with governance        Guide with fitness functions
"Don't break the architecture" "Architecture must accommodate change"
```

### 5.2 Architectural Fitness Functions

A **fitness function** is any mechanism that performs an automated assessment of architectural characteristics:

```
Types of Fitness Functions:

CONTINUOUS (run in CI/CD every commit)
  ─ Cyclomatic complexity limits per module
  ─ No direct cross-domain database joins
  ─ All public APIs versioned
  ─ Test coverage > 80%
  ─ No new critical security vulnerabilities

TRIGGERED (run on schedule or specific events)
  ─ Performance baseline regression (weekly)
  ─ Dependency drift check (monthly)
  ─ Unused code detection (monthly)
  ─ Architecture documentation freshness (quarterly)

TEMPORAL (time-bound checks)
  ─ Deprecated library usage (must migrate by date)
  ─ ADR review due (older than 2 years)
  ─ Certificate expiry warning (90 days out)
```

### 5.3 Implementing Fitness Functions

```python
# Example: ArchUnit-style fitness function in Python
# Enforces no cross-domain imports

# fitness_functions/domain_isolation.py
import ast
import sys
from pathlib import Path

DOMAIN_MAPPING = {
    "payments": "src/payments/",
    "users":    "src/users/",
    "orders":   "src/orders/",
}

ALLOWED_SHARED = ["src/shared/", "src/platform/"]

def check_no_cross_domain_imports(source_dir: str) -> list[str]:
    violations = []
    for py_file in Path(source_dir).rglob("*.py"):
        domain = get_domain(py_file)
        if not domain:
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = get_module_path(node)
                if is_cross_domain_import(module, domain):
                    violations.append(
                        f"{py_file}:{node.lineno} — "
                        f"'{domain}' imports from '{get_domain(module)}'"
                    )
    return violations

if __name__ == "__main__":
    violations = check_no_cross_domain_imports("src/")
    if violations:
        print("ARCHITECTURE VIOLATION — Cross-domain imports:")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    print("✓ Domain isolation fitness function: PASS")
```

```java
// Java: ArchUnit fitness function
// Enforces layered architecture within a service

@AnalyzeClasses(packages = "com.company.payments")
class ArchitectureFitnessTest {

    @ArchTest
    static final ArchRule layeredArchitecture =
        layeredArchitecture()
            .consideringAllDependencies()
            .layer("API").definedBy("..api..")
            .layer("Application").definedBy("..application..")
            .layer("Domain").definedBy("..domain..")
            .layer("Infrastructure").definedBy("..infrastructure..")
            .whereLayer("API").mayOnlyBeAccessedByLayers("Application")
            .whereLayer("Domain").mayOnlyBeAccessedByLayers("Application", "Infrastructure")
            .whereLayer("Infrastructure").mayNotBeAccessedByAnyLayer();

    @ArchTest
    static final ArchRule noCyclesInPackages =
        slices().matching("com.company.payments.(*)..").should().beFreeOfCycles();

    @ArchTest
    static final ArchRule servicesAnnotated =
        classes().that().haveNameEndingWith("Service")
            .should().beAnnotatedWith(Service.class);
}
```

---

## 6. Technical Debt Management

### 6.1 Types of Technical Debt

```
Technical Debt Taxonomy (Fowler):

DELIBERATE / PRUDENT
  "We know we're cutting corners now, and we'll address it later"
  Example: Hard-coded config values to ship on deadline
  → OK if tracked and paid down within sprint/quarter

DELIBERATE / RECKLESS
  "We don't have time for design"
  Example: No error handling, no tests
  → Accumulates interest quickly, avoid this

INADVERTENT / PRUDENT
  "Now we know how we should have done it"
  Example: Discovery that our ORM approach causes N+1 problems
  → Normal learning; refactor when discovered

INADVERTENT / RECKLESS
  "What's layering?" (lack of knowledge)
  Example: Business logic in database stored procedures
  → Address via mentoring + code standards
```

### 6.2 Tech Debt Inventory

```
Tech Debt Classification Matrix:

HIGH IMPACT + LOW EFFORT (Fix Now)
  ─ Unhandled exceptions causing silent data corruption
  ─ N+1 queries on hot paths
  ─ Missing indexes on high-traffic queries

HIGH IMPACT + HIGH EFFORT (Plan Strategically)
  ─ Monolith decomposition
  ─ Legacy authentication migration (basic auth → OAuth2)
  ─ Database migration (MySQL → PostgreSQL)
  ─ Framework upgrade (Spring 2 → Spring 6)

LOW IMPACT + LOW EFFORT (Do When Convenient)
  ─ Rename confusing variables
  ─ Add missing tests for edge cases
  ─ Clean up dead code

LOW IMPACT + HIGH EFFORT (Question / Defer)
  ─ Perfect abstractions for rarely-touched code
  ─ Refactoring services with zero current issues
  ─ Migrating working infrastructure "for best practices"
```

### 6.3 Tech Debt Tracking and Budgeting

```
Tech Debt Budget Model:

Recommendation: Allocate 20% of engineering capacity to tech debt

Per-Sprint Allocation:
  80% → Feature work
  15% → Tech debt reduction
   5% → Exploratory / learning

Tech Debt Scoring (for prioritization):
  Score = (Business Risk × 3) + (Developer Pain × 2) + (Fix Effort × -1)

Example:
  Item: "Legacy auth (HTTP Basic) still used in 5 services"
  Business Risk: 5/5 (security vulnerability)
  Developer Pain: 2/5 (rarely touched)
  Fix Effort:     3/5 (medium complexity)
  Score = (5×3) + (2×2) + (3×-1) = 15 + 4 - 3 = 16 (HIGH priority)

  Item: "Inconsistent variable naming in reports module"
  Business Risk: 1/5 (no user impact)
  Developer Pain: 2/5 (minor confusion)
  Fix Effort:     1/5 (easy)
  Score = (1×3) + (2×2) + (1×-1) = 3 + 4 - 1 = 6 (LOW priority)
```

---

## 7. Architecture Governance

### 7.1 Governance Models

```
Governance Spectrum:

CENTRALIZED (Ivory Tower)              DECENTRALIZED (Anarchy)
──────────────────────────             ──────────────────────
All decisions go through               Each team does what they want
architecture board                     No consistency

Problems:                              Problems:
─ Bottleneck                           ─ Tech sprawl (20 databases)
─ Architects disconnected              ─ Incompatible systems
  from reality                         ─ No knowledge transfer
─ Teams work around governance         ─ Security gaps

BALANCED (Recommended: "Guardrails, not gates")
──────────────────────────────────────────────
─ Clear principles published (not rules)
─ Golden paths make the "right way" the easy way
─ Architecture reviews for cross-cutting decisions only
─ Teams self-govern within guardrails
─ Fitness functions enforce non-negotiables automatically
```

### 7.2 Architecture Principles

Good architecture principles guide decisions without prescribing solutions:

```
Example Architecture Principles:

1. DESIGN FOR FAILURE
   "Any component can and will fail. Design for graceful degradation."

2. PREFER MANAGED SERVICES
   "Unless we have a compelling reason, prefer managed cloud services
   to self-managed infrastructure."

3. DATA IS A FIRST-CLASS CITIZEN
   "All data has an owner, SLA, and documented schema.
   No anonymous data stores."

4. API FIRST
   "Design the API contract before implementation.
   Internal APIs are treated with the same rigor as public APIs."

5. OBSERVABILITY BY DEFAULT
   "Every service ships with metrics, structured logging, and tracing
   from day one — not as an afterthought."

6. SECURITY IN DEPTH
   "Defense in multiple layers. No security control is the last line."

7. MINIMIZE COORDINATION
   "Prefer loose coupling. Services that require synchronized releases
   indicate a decomposition problem."

8. EMBRACE BORING TECHNOLOGY
   "New technology for new problems only. Proven technology for
   solved problems. [See: Choose Boring Technology — Dan McKinley]"
```

### 7.3 Principal Engineer vs Architect Roles

```
Role Distinctions (organizations vary):

Software Architect           Principal Engineer
────────────────────         ──────────────────────
Often more breadth           Deep technical expertise
Less hands-on coding         Still writes code regularly
Org-wide responsibilities    Team/domain focused
Strategic direction          Tactical excellence + strategy
Documentation-heavy          Implementation-heavy
"What should we build?"      "How should we build it?"

Both roles share:
  ─ ADR authoring and review
  ─ Technical mentorship
  ─ Cross-team coordination
  ─ Tech radar contributions
  ─ Architecture review participation
```

---

## 8. Communicating Architecture

### 8.1 C4 Model for Architecture Diagrams

The [C4 Model](https://c4model.com) (Simon Brown) provides 4 levels of architecture diagrams:

```
C4 Diagram Levels:

Level 1: SYSTEM CONTEXT (audience: anyone)
  ─ Shows the system and its external users/dependencies
  ─ No technical detail
  ─ "What does this system do and who uses it?"

Level 2: CONTAINER (audience: technical leads)
  ─ Shows major technical components (web apps, databases, services)
  ─ Technology choices visible
  ─ "What runs, and how do they communicate?"

Level 3: COMPONENT (audience: developers)
  ─ Zooms into one container to show components inside
  ─ Maps to code packages/modules
  ─ "How is this service structured internally?"

Level 4: CODE (audience: developers, optional)
  ─ UML class diagrams of key components
  ─ Only for complex/critical areas
  ─ Usually auto-generated from code

Example Level 2 (Container Diagram):

  ┌─────────────────────────────────────────────────────────┐
  │              Payment Platform [System]                   │
  │                                                         │
  │  ┌──────────┐      ┌──────────┐      ┌──────────┐      │
  │  │  Web App │─────►│ Payment  │─────►│ Postgres │      │
  │  │ (React)  │      │   API    │      │  (RDS)   │      │
  │  │          │      │ (Spring) │      └──────────┘      │
  │  └──────────┘      │          │─────►┌──────────┐      │
  │                    │          │      │  Redis   │      │
  │                    └──────────┘      │  Cache   │      │
  │                         │            └──────────┘      │
  └─────────────────────────┼───────────────────────────────┘
                             │ HTTPS
                    ┌────────▼────────┐
                    │  Stripe API     │
                    │ [External]      │
                    └─────────────────┘
```

### 8.2 Documenting Architecture Decisions for Different Audiences

```
Same decision, different framings:

Decision: "Adopt Event-Driven Architecture for Orders"

For ENGINEERS (technical detail):
  "We are implementing event sourcing using Apache Kafka.
   Order state changes will be published as immutable events
   to the 'orders.events' topic. Consumers use the outbox pattern
   (transactional outbox table + Debezium) to ensure exactly-once
   delivery. Schema evolution follows Avro backward compatibility."

For ENGINEERING MANAGER (impact + timeline):
  "We're decoupling the order service from downstream dependencies
   by introducing async messaging. This will eliminate the 3-second
   P99 checkout latency caused by synchronous calls to inventory.
   Migration timeline: 8 weeks. Risk: medium (new pattern for team;
   mitigated by Kafka training in week 1)."

For VP/BUSINESS STAKEHOLDER (business outcome):
  "We're upgrading our order processing infrastructure to handle
   10x our current peak load, reduce checkout errors by ~40%, and
   enable new real-time features (instant order tracking, live
   inventory) that were previously too complex to build. This is
   foundational work for Q3 personalization features."
```

### 8.3 Architecture Decision Communication Checklist

```
When announcing an architectural decision:

□ Write the ADR (in repo, linked from wiki)
□ Brief affected team leads (1:1 or small group)
□ Post RFC or decision summary in #architecture Slack channel
□ Update the tech radar if applicable
□ Update the relevant architecture diagram (C4)
□ Add to developer portal / Backstage docs
□ Present in engineering all-hands if org-wide impact
□ Update onboarding docs so new engineers learn the pattern
```

---

## 9. Trade-off Analysis Frameworks

### 9.1 ATAM (Architecture Trade-off Analysis Method)

A structured approach to evaluating architectural alternatives:

```
ATAM Process (simplified for day-to-day use):

1. PRESENT THE ARCHITECTURE
   ─ Describe the proposed architecture to stakeholders
   ─ Identify key architectural approaches

2. IDENTIFY QUALITY ATTRIBUTE SCENARIOS
   ─ "System must handle 10K RPS with < 200ms P99"
   ─ "New developer can make a change in < 1 day"
   ─ "System recovers from DB failure in < 30s"

3. ANALYZE ARCHITECTURAL APPROACHES
   ─ For each approach, evaluate each quality attribute
   ─ Use +/- scoring

4. IDENTIFY SENSITIVITY POINTS
   ─ Components where small changes have large impact on quality
   ─ "Adding one more sync external call breaks latency SLA"

5. IDENTIFY TRADE-OFF POINTS
   ─ Where improving one quality attribute degrades another
   ─ "Caching improves latency but increases complexity and stale data risk"
```

### 9.2 Decision Matrix (for technology comparison)

```
Example: Choosing a message broker

Weighted Criteria Matrix:

Criterion           Weight  Kafka  RabbitMQ  SQS/SNS
───────────────────────────────────────────────────────
Throughput (req/s)    25%     5       3          3
Replay capability     20%     5       2          2
Managed service       20%     3       3          5
Team familiarity      15%     4       3          4
Exactly-once deliv.   10%     4       3          3
Total cost of own.    10%     3       4          4

Weighted Scores:
Kafka:    5×25 + 5×20 + 3×20 + 4×15 + 4×10 + 3×10 = 420
RabbitMQ: 3×25 + 2×20 + 3×20 + 3×15 + 3×10 + 4×10 = 295
SQS/SNS:  3×25 + 2×20 + 5×20 + 4×15 + 3×10 + 4×10 = 340

Winner: Kafka (also wins on most critical criteria: throughput + replay)
```

### 9.3 OODA Loop for Architectural Decisions

```
OODA Loop Applied to Architecture:

OBSERVE
  ─ Gather data: performance metrics, developer pain points
  ─ Monitor: incidents, DORA metrics, on-call load
  ─ Input: team feedback, post-mortems, tech radar trends

ORIENT
  ─ Frame the problem: what quality attribute is suffering?
  ─ Identify constraints: budget, team skills, timeline
  ─ Research: alternatives, industry patterns

DECIDE
  ─ Create shortlist of options
  ─ Score using decision matrix
  ─ Run RFC or ARB review
  ─ Document in ADR

ACT
  ─ Implement (start with proof of concept)
  ─ Measure against success criteria
  ─ Loop back to OBSERVE to validate decision
```

---

## 10. Practical Examples

### 10.1 Complete ADR Example: Database Choice

```markdown
# ADR-0015: Use PostgreSQL as Primary Relational Database

**Date**: 2022-08-10  
**Status**: Accepted  
**Supersedes**: ADR-0008 (MySQL adoption)

## Context

We adopted MySQL in 2019 (ADR-0008). Since then:
- 3 production incidents caused by MySQL replication lag
- Lack of JSONB support forcing awkward schema designs
- Team has grown and new hires universally prefer PostgreSQL
- Our cloud provider (AWS) offers superior RDS Postgres support
- PostGIS extension needed for upcoming geo-features

## Decision

**Migrate from MySQL to PostgreSQL** for all new services.
Existing MySQL services migrated on a rolling basis.

## Migration Strategy

Phase 1 (Q3 2022): All new services use Postgres only  
Phase 2 (Q4 2022 - Q1 2023): Migrate payments and users services  
Phase 3 (2023): Migrate remaining 8 MySQL services  

## Consequences

+ Advanced query capabilities (JSONB, CTEs, window functions)
+ Better replication with logical replication slots
+ PostGIS for geo-features
+ Unified RDS monitoring (one runbook)
- Migration effort: ~16 engineer-weeks total
- New ops runbooks needed (pg_dump vs mysqldump)
- Team training on Postgres-specific features

## References
- ADR-0008: MySQL Adoption (superseded)
- MySQL Incident Post-mortems: INC-0441, INC-0523, INC-0601
```

### 10.2 Fitness Function CI Integration

```yaml
# .github/workflows/architecture-fitness.yaml
name: Architecture Fitness Functions

on: [push, pull_request]

jobs:
  fitness-functions:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Domain isolation check
        run: python fitness_functions/domain_isolation.py

      - name: Cyclomatic complexity check
        run: |
          radon cc src/ --min C --show-complexity \
            --total-average --xml > complexity.xml
          python fitness_functions/check_complexity.py complexity.xml

      - name: Dependency drift check
        run: |
          pip install pip-audit
          pip-audit --vulnerability-service pypi \
            --fail-on-finding

      - name: Architecture test (ArchUnit)
        run: mvn test -Dtest=ArchitectureFitnessTest

      - name: API versioning check
        run: |
          # Check all new/changed API endpoints have version prefix
          python fitness_functions/api_versioning.py \
            --changed-files $(git diff --name-only HEAD~1)

      - name: No direct cross-service DB access
        run: |
          # Check no service imports another service's DB models
          python fitness_functions/no_cross_service_db.py
```

### 10.3 Architecture Review Checklist (for reviewers)

```
Architecture Review Checklist:

CORRECTNESS
□ Does the design solve the stated problem?
□ Are edge cases handled?
□ Is the failure behavior correct?

SCALABILITY & PERFORMANCE
□ What are the bottlenecks under load?
□ How does it behave at 10x expected traffic?
□ Are there N+1 queries or expensive operations on hot paths?

RESILIENCE
□ What happens when dependency X fails?
□ Are there single points of failure?
□ How is recovery handled?

OPERABILITY
□ How is this monitored? (metrics, alerts, dashboards)
□ How does an on-call engineer debug an incident?
□ How is it deployed? Rollback procedure?

SECURITY
□ Authentication and authorization model?
□ What data is sensitive? How is it protected?
□ Is there a threat model?

SIMPLICITY
□ Is this the simplest solution that could work?
□ Can a new team member understand this in an hour?
□ Are there unnecessary abstractions?

EVOLVABILITY
□ What assumptions does this design encode?
□ What would be hard to change 2 years from now?
□ Is this decision reversible? What would reversal cost?

COST
□ What is the steady-state infrastructure cost?
□ How does cost scale with traffic?
□ Are there more cost-effective alternatives?
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **ADR** | Capture the WHY behind decisions — future engineers will thank you |
| **RFC Process** | Async review via PRs is faster than standing ARB meetings |
| **Tech Radar** | Quarterly alignment on technology adoption across the org |
| **Fitness Functions** | Automate architectural compliance in CI/CD — not manual reviews |
| **Evolutionary Architecture** | Design for change, not perfection; guard characteristics automatically |
| **Tech Debt Budget** | Explicit 20% allocation; score debt by business risk × dev pain |
| **C4 Model** | Four-level diagrams matching audience: context → container → component → code |
| **ATAM / Decision Matrix** | Structured trade-off analysis prevents "gut feel" decisions |
| **Governance = Guardrails** | Make the right way the easy way — golden paths over gatekeeping |
| **Communication** | Same decision needs different framings for engineers, managers, and executives |
