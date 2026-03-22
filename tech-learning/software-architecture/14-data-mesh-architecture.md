# Data Mesh & Modern Data Architecture

## Table of Contents
1. [Introduction](#1-introduction)
2. [Data Mesh Principles](#2-data-mesh-principles)
3. [Data Products](#3-data-products)
4. [Data Contracts](#4-data-contracts)
5. [Data Platform Architecture](#5-data-platform-architecture)
6. [Lakehouse Architecture](#6-lakehouse-architecture)
7. [Data Governance & Catalog](#7-data-governance--catalog)
8. [Data Observability](#8-data-observability)
9. [Federated Computational Governance](#9-federated-computational-governance)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

### 1.1 The Problem with Centralized Data Architectures

Traditional centralized data platforms (monolithic data warehouses, data lakes) face the same problems that drove microservices adoption:

```
Centralized Data Lake Problems:

                    ┌─────────────────────┐
  All teams ──────► │   Central Data Lake  │ ◄── All queries
  (producers)       │   (Data Engineering  │     (consumers)
                    │    Team Owns All)    │
                    └─────────────────────┘

Results:
─ Data Engineering team becomes a bottleneck
─ Domain knowledge lost in translation (payments team
  knows payments data best, not data engineers)
─ Ungoverned swamp: inconsistent schemas, stale data
─ Months to add a new data source
─ Business teams distrust data quality
```

### 1.2 Evolution of Data Architectures

```
Generation 1: Data Warehouse (2000s)
  ─ Structured, schema-on-write
  ─ SQL analytics
  ─ Expensive, limited scale
  ─ ETL pipelines

Generation 2: Data Lake (2010s)
  ─ Raw, schema-on-read
  ─ Cheap storage (HDFS/S3)
  ─ Batch processing (Spark)
  ─ Became a "data swamp"

Generation 3: Modern Data Stack (2015-2020)
  ─ Cloud DWH (Snowflake, BigQuery, Redshift)
  ─ dbt for transformation
  ─ Fivetran for ingestion
  ─ Still centralized bottleneck

Generation 4: Data Mesh (2020+)
  ─ Decentralized ownership (domain teams)
  ─ Data as a product
  ─ Self-serve data platform
  ─ Federated governance

Generation 5: Lakehouse (parallel track)
  ─ ACID transactions on data lake
  ─ Schema enforcement + evolution
  ─ Unified batch + streaming
  ─ Open formats (Delta Lake, Iceberg, Hudi)
```

---

## 2. Data Mesh Principles

Data Mesh was coined by [Zhamak Dehghani](https://martinfowler.com/articles/data-mesh-principles.html). It has four core principles:

### 2.1 Principle 1: Domain Ownership

```
Data Mesh Domain Ownership:

Payments Domain Team
  ├── Owns: payments service (operational)
  └── Owns: payments data products (analytical)
       ─ daily_transactions
       ─ payment_success_rate
       ─ fraud_signals

User Domain Team
  ├── Owns: user service (operational)
  └── Owns: user data products
       ─ user_profile_snapshot
       ─ user_activity_events
       ─ churn_risk_features

NOT a central data team owning all analytics
```

### 2.2 Principle 2: Data as a Product

Data products are first-class citizens with:

| Quality | Description | Example |
|---------|-------------|---------|
| **Discoverable** | Findable in catalog | Searchable in data catalog |
| **Addressable** | Stable, unique address | `payments.prod.daily_transactions` |
| **Trustworthy** | SLAs, quality metrics | Freshness < 1hr, Completeness > 99% |
| **Self-describing** | Schema + docs embedded | Avro schema + README + ownership |
| **Interoperable** | Standard formats | Parquet, Avro, Delta Lake |
| **Secure** | Access controlled | Column-level masking, RBAC |

### 2.3 Principle 3: Self-Serve Data Platform

```
Self-Serve Data Platform enables domain teams:

Domain Team Needs        Platform Provides
──────────────────────   ────────────────────────────
Store tabular data   →   Managed Delta Lake tables
Stream events        →   Kafka topic + schema registry
Query data           →   Spark/Trino/BigQuery access
Build pipelines      →   Airflow / Prefect templates
Share data product   →   Data catalog registration
Access control       →   Attribute-based access control
Monitor quality      →   dbt tests + Great Expectations
```

### 2.4 Principle 4: Federated Computational Governance

```
Federated Governance Model:

         Global Policies (automated, enforced by platform)
         ─ PII classification required
         ─ Retention policies applied automatically
         ─ Data lineage tracked everywhere
                     │
         ┌───────────┴────────────┐
         │   Data Governance      │
         │   Council              │
         │  (1 rep per domain)    │
         └───────────┬────────────┘
                     │ sets standards
    ┌────────────────┼────────────────┐
    │                │                │
┌───▼───┐       ┌───▼───┐       ┌───▼───┐
│Domain │       │Domain │       │Domain │
│  A    │       │  B    │       │  C    │
│Governs│       │Governs│       │Governs│
│own    │       │own    │       │own    │
│data   │       │data   │       │data   │
└───────┘       └───────┘       └───────┘
```

---

## 3. Data Products

### 3.1 Data Product Types

```
┌─────────────────────────────────────────────────────────┐
│                  Data Product Types                     │
├─────────────────┬──────────────┬────────────────────────┤
│   Source-aligned │  Aggregate   │    Consumer-aligned    │
├─────────────────┼──────────────┼────────────────────────┤
│ Raw operational  │ Cross-domain │ Purpose-built for a    │
│ data exposed as  │ aggregations │ specific consumer or   │
│ a product        │ and joins    │ use case               │
├─────────────────┼──────────────┼────────────────────────┤
│ payments_events  │ customer_360 │ ml_churn_features      │
│ user_profiles    │ product_perf │ finance_revenue_report │
│ inventory_state  │ order_funnel │ marketing_cohorts      │
└─────────────────┴──────────────┴────────────────────────┘
```

### 3.2 Data Product Specification

```yaml
# data-product.yaml — lives in domain team's repo
name: payments.daily_transactions
version: "2.1"
domain: payments
owner:
  team: payments-data-engineers
  oncall: payments-data@company.com

description: |
  Daily aggregated payment transactions per merchant.
  Updated hourly. Primary source for finance reporting
  and merchant analytics.

output_ports:
  - type: batch_table
    format: delta
    location: s3://data-lake/payments/daily_transactions/
    schema_ref: schemas/daily_transactions_v2.avsc

  - type: streaming
    format: avro
    topic: payments.daily_transactions.v2
    schema_registry: https://schema-registry.company.com

  - type: api
    endpoint: https://data-api.company.com/payments/daily_transactions
    auth: oauth2

slas:
  freshness: "< 1 hour"
  completeness: "> 99.5%"
  availability: "> 99.9%"
  support_response: "< 4 hours"

quality_checks:
  - no_nulls: [transaction_id, merchant_id, amount, date]
  - row_count_daily: "> 100000"
  - amount_positive: "amount > 0"
  - referential_integrity: "merchant_id in merchants.merchant_id"

data_classification:
  pii: false
  financial: true
  retention_days: 2555    # 7 years (regulatory)

consumers:
  - finance-reporting-team
  - ml-fraud-team
  - merchant-analytics-team
```

### 3.3 Data Product Lifecycle

```
Data Product Lifecycle:

DISCOVER → BUILD → PUBLISH → CONSUME → EVOLVE → DEPRECATE

DISCOVER:
  ─ Identify need from consumer team
  ─ Check existing products in catalog
  ─ Define data product spec with SLAs

BUILD:
  ─ Implement pipeline (dbt/Spark)
  ─ Add quality tests
  ─ Document schema

PUBLISH:
  ─ Register in data catalog
  ─ Set access controls
  ─ Announce via data changelog

CONSUME:
  ─ Consumers subscribe (track in catalog)
  ─ Breaking change notifications

EVOLVE:
  ─ Add new columns (backward compatible)
  ─ Schema version bump for breaking changes
  ─ Maintain N-1 version support

DEPRECATE:
  ─ Migrate consumers to v2
  ─ 90-day deprecation notice
  ─ Sunset old version
```

---

## 4. Data Contracts

### 4.1 What is a Data Contract

A **Data Contract** is a formal agreement between a data producer and data consumers defining: schema, semantics, quality, freshness, and SLAs. It prevents "silent breaking changes."

```
Without Data Contracts:
  Producer changes column name →
  Consumer pipeline breaks silently →
  Finance report shows wrong numbers →
  Discovered 3 days later in board meeting

With Data Contracts:
  Producer proposes schema change →
  Contract validation fails in CI →
  PR blocked until consumers updated →
  Coordinated migration with notification
```

### 4.2 Data Contract Specification (Open Standard)

```yaml
# Following datacontract.com specification
dataContractSpecification: 0.9.2
id: urn:datacontract:payments:daily_transactions:v2
info:
  title: Payments Daily Transactions
  version: 2.0.0
  description: Daily aggregated payment transactions
  owner: payments-team
  contact:
    name: Payments Data Team
    email: payments-data@company.com

servers:
  production:
    type: s3
    location: s3://data-lake/payments/daily_transactions/
    format: parquet
    delimiter: none

terms:
  usage: |
    May be used for financial reporting, analytics, and ML.
    Cannot be redistributed externally.
  billing: free for internal use
  noticePeriod: 90 days

models:
  daily_transactions:
    description: One row per merchant per day
    fields:
      transaction_date:
        type: date
        required: true
        description: UTC date of transactions
      merchant_id:
        type: string
        required: true
        pattern: "^M[0-9]{8}$"
      transaction_count:
        type: integer
        minimum: 0
      gross_amount_usd:
        type: decimal
        precision: 18
        scale: 2
        minimum: 0
      currency_breakdown:
        type: object
        description: Amount split by currency code

quality:
  type: SodaCL
  specification:
    checks for daily_transactions:
      - row_count > 100000
      - missing_count(merchant_id) = 0
      - duplicate_count(transaction_date, merchant_id) = 0
      - freshness(transaction_date) < 2h

servicelevels:
  availability:
    description: Data available in S3
    percentage: 99.9%
  freshness:
    description: Data lag behind source
    threshold: 1h
  completeness:
    description: Merchant coverage
    percentage: 99.5%
```

### 4.3 Data Contract in CI/CD

```
Data Contract CI/CD Pipeline:

Producer PR:
  1. Edit dbt model or pipeline code
  2. CI runs: datacontract lint my-contract.yaml
  3. CI runs: datacontract test (against sample data)
  4. CI checks: are there any BREAKING changes?
     ─ Column removed → BREAKING (PR blocked)
     ─ Column renamed → BREAKING (PR blocked)
     ─ New required column → BREAKING (PR blocked)
     ─ New optional column → COMPATIBLE (PR allowed)
  5. If breaking: auto-notify all registered consumers
  6. Create migration branch + consumer update PRs

Consumer Protection:
  ─ Consumers register in contract
  ─ Schema registry enforces compatibility
  ─ Alerts on quality degradation (SLA breach)
```

---

## 5. Data Platform Architecture

### 5.1 Modern Data Platform Components

```
┌──────────────────────────────────────────────────────────────┐
│                    Data Platform                             │
├─────────────────┬────────────────┬────────────────┬──────────┤
│    INGESTION    │    STORAGE     │  PROCESSING    │ SERVING  │
│                 │                │                │          │
│ Fivetran/Airbyte│ Data Lake (S3) │ Apache Spark   │ BI Layer │
│ Kafka Connect   │ Delta/Iceberg  │ dbt (SQL xform)│ Superset │
│ Debezium (CDC)  │ Data Warehouse │ Apache Flink   │ Trino    │
│ custom ingestors│ (Snowflake)    │ Airflow/Prefect│ Data API │
├─────────────────┴────────────────┴────────────────┴──────────┤
│                   DATA CATALOG & GOVERNANCE                  │
│   Datahub / Apache Atlas / Amundsen / OpenMetadata           │
│   ─ Lineage   ─ Schema registry   ─ PII discovery            │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Lambda Architecture (Batch + Speed)

```
Lambda Architecture:
                    ┌─────────────┐
                    │   Source    │
                    │   Systems   │
                    └──────┬──────┘
                           │
               ┌───────────┴───────────┐
               │                       │
          ┌────▼────┐             ┌────▼────┐
          │  Batch  │             │ Speed   │
          │  Layer  │             │  Layer  │
          │(Spark)  │             │(Flink/  │
          │         │             │ Kafka)  │
          └────┬────┘             └────┬────┘
               │                       │
          ┌────▼────┐             ┌────▼────┐
          │ Batch   │             │  Real-  │
          │  Views  │             │ time    │
          │(complete│             │  Views  │
          │ history)│             │(recent) │
          └────┬────┘             └────┬────┘
               │                       │
               └───────────┬───────────┘
                       ┌───▼───┐
                       │Serving│
                       │ Layer │
                       │(merge)│
                       └───────┘

Pros: Accurate batch + low-latency real-time
Cons: Maintain two separate codebases
```

### 5.3 Kappa Architecture (Stream-Only)

```
Kappa Architecture:

All data treated as streams:
               ┌─────────────┐
               │   Source    │
               │   Systems   │
               └──────┬──────┘
                      │ all events
               ┌──────▼──────┐
               │   Kafka /   │
               │   Pulsar    │ ← permanent event log
               │ (long-term) │
               └──────┬──────┘
                      │
          ┌───────────┴───────────┐
          │     Stream Processor  │
          │    (Flink / Spark SS) │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    Serving Store      │
          │ (Delta Lake / Redis / │
          │  PostgreSQL)          │
          └───────────────────────┘

"Reprocessing" = replay from Kafka with new logic
Pros: Single codebase, simpler ops
Cons: Requires fast stream processing for all batch needs
```

---

## 6. Lakehouse Architecture

### 6.1 Lakehouse Concept

A **Lakehouse** combines the cost-effective storage of a data lake with the data management features of a data warehouse:

```
Data Warehouse vs Data Lake vs Lakehouse:

Data Warehouse         Data Lake              Lakehouse
──────────────         ─────────              ─────────
✅ ACID transactions   ❌ No transactions     ✅ ACID transactions
✅ Schema enforcement  ❌ Schema on read      ✅ Schema enforcement
✅ BI performance      ✅ Cheap storage        ✅ Cheap storage
❌ Expensive           ✅ Any data format      ✅ Any data format
❌ Limited ML support  ✅ ML/AI workloads      ✅ ML/AI workloads
❌ Siloed from ML      ❌ Poor BI performance  ✅ Good BI performance
```

### 6.2 Open Table Formats

Three major open table formats compete:

| Feature | Delta Lake | Apache Iceberg | Apache Hudi |
|---------|-----------|----------------|-------------|
| **Origin** | Databricks (2019) | Netflix (2018) | Uber (2016) |
| **ACID** | ✅ | ✅ | ✅ |
| **Time travel** | ✅ | ✅ | ✅ |
| **Schema evolution** | ✅ | ✅ | ✅ |
| **Partition evolution** | ❌ | ✅ (best-in-class) | Partial |
| **Row-level updates** | ✅ | ✅ | ✅ (optimized) |
| **Streaming CDC** | Good | Good | Excellent |
| **Engine support** | Spark, Trino, Flink | Spark, Trino, Flink, Hive | Spark, Flink |
| **Metadata scalability** | OK | Excellent | OK |
| **Best for** | Databricks shops | Multi-engine, large tables | CDC/upsert heavy |

### 6.3 Delta Lake Architecture

```
Delta Lake Storage Layout:
s3://data-lake/payments/transactions/
├── _delta_log/
│   ├── 00000000000000000000.json  ← commit 0: table creation
│   ├── 00000000000000000001.json  ← commit 1: insert batch
│   ├── 00000000000000000002.json  ← commit 2: update/delete
│   └── 00000000000000000010.checkpoint.parquet ← checkpoint
├── date=2024-01-01/
│   ├── part-00001.parquet
│   └── part-00002.parquet
└── date=2024-01-02/
    └── part-00001.parquet

Delta Log JSON (commit entry):
{
  "add": {
    "path": "date=2024-01-02/part-00001.parquet",
    "size": 1048576,
    "stats": {"numRecords": 50000, "minValues": {...}}
  },
  "commitInfo": {
    "timestamp": 1704153600000,
    "operation": "WRITE",
    "operationParameters": {"mode": "Append"}
  }
}
```

### 6.4 Medallion Architecture (Bronze/Silver/Gold)

```
Medallion Architecture:

BRONZE (Raw)          SILVER (Cleaned)       GOLD (Curated)
─────────────         ────────────────       ──────────────
Raw ingested data     Validated, deduplicated Business-ready
Append-only           Schema enforced        Aggregated
No transformation     PII masked             Joined datasets
Immutable             Null handling          Denormalized
Retained 7 years      Retained 3 years       Retained 2 years

s3://lake/bronze/  → s3://lake/silver/ → s3://lake/gold/

Example - Payments:
raw_payment_events → payments_cleansed → daily_merchant_revenue
  (from Kafka)        (deduped, typed)    (joined with merchants)

Failure Recovery:
  ─ Bronze never loses raw data (replay anytime)
  ─ Silver can be fully rebuilt from Bronze
  ─ Gold can be fully rebuilt from Silver
```

---

## 7. Data Governance & Catalog

### 7.1 Data Catalog Architecture (DataHub)

[DataHub](https://datahubproject.io) is LinkedIn's open-source data catalog:

```
DataHub Architecture:
┌─────────────────────────────────────────────────────┐
│                   DataHub UI                        │
│  Search │ Lineage │ Datasets │ Pipelines │ Glossary │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              DataHub Backend                        │
│  ─ Metadata graph (Neptune/Elasticsearch)           │
│  ─ Search index                                     │
│  ─ REST + GraphQL APIs                              │
└──────┬──────────┬──────────┬──────────┬─────────────┘
       │          │          │          │
   ┌───▼───┐  ┌──▼───┐  ┌───▼──┐  ┌───▼────┐
   │Spark  │  │ dbt  │  │Kafka │  │ Airflow│
   │Ingest.│  │Ingest│  │Ingest│  │ Ingest │
   └───────┘  └──────┘  └──────┘  └────────┘
```

### 7.2 Data Lineage

```
Column-Level Lineage Example:
                                            ┌──────────────┐
                                    ┌──────►│ finance_rpt  │
                                    │       │ gross_revenue│
┌──────────────┐   ┌─────────────┐  │       └──────────────┘
│payments_raw  │   │daily_txns   │  │
│ amount       ├──►│ gross_amount├──┤       ┌──────────────┐
│ currency     │   │ _usd        │  └──────►│ merchant_dash│
└──────────────┘   └─────────────┘          │ revenue_usd  │
                          ▲                 └──────────────┘
                          │
┌──────────────┐   ┌─────────────┐
│forex_rates   │   │ fx_convert()│
│ rate_usd     ├──►│ (dbt macro) │
└──────────────┘   └─────────────┘

Impact Analysis:
  Q: "What breaks if payments_raw.amount changes type?"
  A: Lineage graph shows: daily_txns → finance_rpt, merchant_dash
     → notify owners: finance-team, merchant-analytics-team
```

### 7.3 Data Classification & PII

```python
# Automated PII detection and classification
# Runs during catalog ingestion

PII_PATTERNS = {
    "email":      r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn":        r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "phone":      r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
}

CLASSIFICATION_LEVELS = {
    "PUBLIC":       "No restrictions, shareable externally",
    "INTERNAL":     "Internal use only",
    "CONFIDENTIAL": "Need-to-know basis",
    "PII":          "Personal data, GDPR/CCPA regulated",
    "SENSITIVE_PII": "SSN, financial, health data — strict controls",
}

# Column-level masking policy (enforced in Trino/Spark)
column_policies = {
    "users.email":        "MASK: show first 3 chars + ****",
    "users.phone":        "MASK: show last 4 digits only",
    "payments.card_last4": "MASK: show to payment-team only",
    "users.ssn":          "RESTRICT: data-security-team only",
}
```

---

## 8. Data Observability

### 8.1 Data Quality Dimensions

```
Five Pillars of Data Observability (Monte Carlo):

1. FRESHNESS
   ─ When was this table last updated?
   ─ Is the update frequency as expected?
   Alert: "payments.daily_txns not updated in 2 hours"

2. VOLUME
   ─ How many rows are in the table?
   ─ Is row count within historical norms?
   Alert: "daily_txns rows dropped 40% vs yesterday"

3. DISTRIBUTION
   ─ Are column value distributions normal?
   ─ Any unexpected nulls, zeros, outliers?
   Alert: "merchant_id null rate jumped from 0.1% to 5%"

4. SCHEMA
   ─ Did columns change (added/removed/type changed)?
   Alert: "Column 'amount_usd' type changed from DECIMAL to STRING"

5. LINEAGE
   ─ Which upstream tables/pipelines changed?
   ─ What downstream tables are affected?
   Alert: "forex_rates pipeline failed → impacts daily_txns"
```

### 8.2 dbt Tests for Data Quality

```yaml
# models/staging/stg_payments.yml
version: 2

models:
  - name: stg_payments
    description: Cleaned payment transactions
    columns:
      - name: payment_id
        tests:
          - not_null
          - unique
      - name: merchant_id
        tests:
          - not_null
          - relationships:
              to: ref('merchants')
              field: merchant_id
      - name: amount_usd
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              max_value: 1000000
      - name: status
        tests:
          - accepted_values:
              values: ['SUCCESS', 'FAILED', 'PENDING', 'REFUNDED']

    # Custom data freshness test
    freshness:
      warn_after: {count: 1, period: hour}
      error_after: {count: 3, period: hour}
    loaded_at_field: _extracted_at
```

### 8.3 Great Expectations for Pipeline Validation

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations as code
suite = context.add_expectation_suite("payments.daily_transactions")

suite.expect_column_to_exist("merchant_id")
suite.expect_column_values_to_not_be_null("merchant_id")
suite.expect_column_values_to_be_unique("transaction_id")
suite.expect_column_values_to_be_between(
    "amount_usd", min_value=0, max_value=1_000_000
)
suite.expect_table_row_count_to_be_between(
    min_value=100_000, max_value=10_000_000
)

# Run in Airflow DAG before promotion to Silver
validation_result = context.run_checkpoint("payments_checkpoint")
if not validation_result.success:
    # Block pipeline, alert data owner
    raise DataQualityException(validation_result.statistics)
```

---

## 9. Federated Computational Governance

### 9.1 Policy as Code (OPA for Data)

```rego
# Open Policy Agent policy for data access
# Evaluated at query time (Trino / Spark / data API)

package data.access

default allow = false

# Allow data owners to access their own domain's data
allow {
    input.user.team == "payments-team"
    startswith(input.resource.table, "payments.")
}

# Allow analysts to access Gold layer (non-PII)
allow {
    input.user.role == "analyst"
    input.resource.layer == "gold"
    not input.resource.contains_pii
}

# Restrict PII columns based on purpose
allow_column[column] {
    column := input.resource.columns[_]
    not is_pii_column(column)
}

allow_column[column] {
    column := input.resource.columns[_]
    is_pii_column(column)
    input.user.has_pii_certification == true
    input.request.purpose in ["fraud-investigation", "audit"]
}

is_pii_column(col) {
    pii_columns := {"email", "phone", "ssn", "date_of_birth"}
    pii_columns[col]
}
```

### 9.2 Data Mesh Governance Council Structure

```
Governance Council:
  ─ Meets bi-weekly
  ─ 1 representative per domain (rotating)
  ─ Platform engineering representative
  ─ Security/Compliance representative
  ─ Data leadership (Chair)

Council Responsibilities:
  1. Ratify global data standards (naming, classification)
  2. Resolve cross-domain data ownership disputes
  3. Approve new PII data capture
  4. Review and update data retention policies
  5. Measure data mesh adoption metrics

Council Does NOT:
  ✗ Approve individual data product designs
  ✗ Manage day-to-day data operations
  ✗ Review individual pipeline code
  (Those are domain team responsibilities)
```

---

## 10. Practical Examples

### 10.1 End-to-End Data Product Delivery

```
Scenario: Payments team needs to publish
"merchant_daily_revenue" for Finance and ML teams

Step 1 — Discovery (1 day):
  ─ Check catalog: does this product exist? → No
  ─ Interview consumers: Finance needs daily grain
  ─ Define SLAs: freshness < 1hr, completeness > 99.5%

Step 2 — Design (1 day):
  ─ Draft data-product.yaml and datacontract.yaml
  ─ Review with Finance and ML teams
  ─ Register as "DRAFT" in data catalog

Step 3 — Build (3 days):
  ─ Write dbt model: merchants + payments join
  ─ Add dbt tests (not_null, unique, referential)
  ─ Set up Airflow DAG with SLA monitoring
  ─ Add Great Expectations checkpoint

Step 4 — Publish (1 day):
  ─ Promote catalog status to "PUBLISHED"
  ─ Notify Finance and ML teams (via catalog subscription)
  ─ Set up Grafana dashboard for quality monitoring

Step 5 — Operate (ongoing):
  ─ Monitor freshness/completeness alerts
  ─ Respond to consumer questions via catalog
  ─ Publish changelog for schema changes
```

### 10.2 Data Mesh vs Centralized Data Team

```
Centralized Data Team (Anti-pattern for scale):

  Finance request: "We need merchant revenue by country"
  → Ticket to central data team (backlog: 3 weeks)
  → Data engineer interviews finance team (1 day)
  → Data engineer builds pipeline (3 days)
  → Finance team reviews, requests changes (2 days)
  → Goes back to backlog for fixes
  Total: ~5-6 weeks

Data Mesh (Domain team owns it):

  Finance needs merchant revenue by country:
  → Payments team already publishes merchant_revenue
  → Finance team creates derived product using self-serve platform
  → Payments team reviews data contract (1 hr review)
  → Finance's aggregated product published same day
  Total: 1-2 days
```

### 10.3 Reference Data Stack for Data Mesh

```yaml
# Data Mesh Platform Stack (2024 reference)

ingestion:
  batch: fivetran / airbyte          # SaaS connectors
  cdc:   debezium + kafka            # Real-time DB changes
  streaming: kafka connect

storage:
  lake: s3 / gcs / adls
  table_format: apache_iceberg       # Best multi-engine support
  warehouse: snowflake / bigquery    # For BI/SQL consumers

transformation:
  sql: dbt                           # SQL-based transforms
  python: pyspark / pandas           # Complex transforms
  streaming: apache_flink            # Real-time processing

orchestration:
  batch: apache_airflow / prefect
  event_driven: kafka_streams / flink

quality:
  testing: dbt_tests + great_expectations
  observability: monte_carlo / acceldata
  contracts: datacontract_cli

catalog_and_governance:
  catalog: datahub / openmetadata
  schema_registry: confluent_schema_registry
  lineage: datahub / openlineage
  access_control: apache_ranger / opa

serving:
  sql_query: trino / spark_sql
  bi: apache_superset / looker / tableau
  api: custom data api (fastapi)
  ml_features: feast (feature store)
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Data Mesh** | Decentralize data ownership to domain teams; treat data as a product |
| **Data Products** | Discoverable, trustworthy, self-describing assets with SLAs |
| **Data Contracts** | Formal producer-consumer agreements preventing silent breaking changes |
| **Lakehouse** | ACID + schema enforcement on cheap cloud storage via Delta/Iceberg/Hudi |
| **Medallion Architecture** | Bronze (raw) → Silver (clean) → Gold (curated) layers |
| **Data Catalog** | DataHub/OpenMetadata for discovery, lineage, and governance |
| **Data Observability** | Monitor freshness, volume, distribution, schema, lineage |
| **Policy as Code** | OPA-based automated access control enforcement at query time |
| **dbt** | SQL-first transformation layer with built-in testing and lineage |
