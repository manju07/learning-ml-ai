# Multi-Tenancy & SaaS Architecture

## Table of Contents
1. [Introduction](#1-introduction)
2. [Tenancy Models](#2-tenancy-models)
3. [Data Isolation Strategies](#3-data-isolation-strategies)
4. [Identity & Access in SaaS](#4-identity--access-in-saas)
5. [Tenant Onboarding & Provisioning](#5-tenant-onboarding--provisioning)
6. [Noisy Neighbor & Resource Management](#6-noisy-neighbor--resource-management)
7. [Metering & Billing Architecture](#7-metering--billing-architecture)
8. [SaaS Deployment Patterns](#8-saas-deployment-patterns)
9. [Customization & Configuration](#9-customization--configuration)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

**Multi-tenancy** is a software architecture where a single instance of software serves multiple tenants (customers). It is the foundational pattern of SaaS, enabling economics of scale while maintaining isolation between customers.

### 1.1 Why Multi-Tenancy Matters

```
Single-Tenant vs Multi-Tenant:

SINGLE-TENANT (one instance per customer)
  ┌────────┐  ┌────────┐  ┌────────┐
  │Tenant A│  │Tenant B│  │Tenant C│
  │  App   │  │  App   │  │  App   │
  │  DB    │  │  DB    │  │  DB    │
  └────────┘  └────────┘  └────────┘
  
  ✅ Perfect isolation
  ✅ Independent upgrade cycles
  ✅ Easy to customise per tenant
  ❌ 3x infra cost
  ❌ 3x operational overhead
  ❌ Slow onboarding (provision new stack)

MULTI-TENANT (shared infrastructure)
  ┌──────────────────────┐
  │   Shared App         │
  ├──────────────────────┤
  │ Tenant A │ B │ C ... │
  └──────────────────────┘
  
  ✅ Economies of scale (1/N infra cost per tenant)
  ✅ Single deployment → all tenants updated
  ✅ Instant tenant onboarding
  ❌ Isolation complexity
  ❌ Noisy neighbor risk
  ❌ Customization harder
```

### 1.2 Multi-Tenancy Design Axes

```
Design Decision Space:

COMPUTE     ─── Shared pods ─────────────────── Per-tenant pods
STORAGE     ─── Shared table ─── Shared DB ─── Per-tenant DB
NETWORK     ─── Shared VPC ─────────────────── Per-tenant VPC
CUSTOMIZATION ─ No customization ─── Config ─── Per-tenant code

"Pooled" (shared everything) vs "Silo" (isolated everything)
vs "Bridge" (hybrid)

Cost            ──────────────────────────►
Isolation       ◄──────────────────────────
Onboarding speed ──────────────────────────►
              Pooled        Bridge        Silo
```

---

## 2. Tenancy Models

### 2.1 Silo Model (Full Isolation)

```
Silo Model:

Each tenant gets their own dedicated stack.

┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Tenant A   │   │   Tenant B   │   │   Tenant C   │
│              │   │              │   │              │
│  App Servers │   │  App Servers │   │  App Servers │
│  Database    │   │  Database    │   │  Database    │
│  Cache       │   │  Cache       │   │  Cache       │
│  Network     │   │  Network     │   │  Network     │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                  │
        └───────────────────┴──────────────────┘
                            │
              Control Plane (single)
              ─ Tenant registry
              ─ Billing
              ─ SSO routing
              ─ Operations console

When to use Silo:
  ✅ Enterprise customers requiring dedicated infra
  ✅ Strict compliance (HIPAA, FedRAMP tenant isolation)
  ✅ Customers willing to pay premium for isolation
  ✅ Customers needing independent upgrade control
  ❌ Too expensive for SMB/startup customers
```

### 2.2 Pool Model (Shared Everything)

```
Pool Model:

All tenants share infrastructure. Isolation via application logic.

          ┌──────────────────────────────────┐
          │         Shared App Tier           │
          │                                  │
          │  tenant_id passed in every       │
          │  request → filter all queries    │
          └──────────────┬───────────────────┘
                         │
          ┌──────────────▼───────────────────┐
          │         Shared Database           │
          │                                  │
          │  users table:                    │
          │  | tenant_id | user_id | email|  │
          │  | acme      | u1      | ...  |  │
          │  | globex    | u2      | ...  |  │
          └──────────────────────────────────┘

Implementation with Row-Level Security (PostgreSQL):
  
  CREATE POLICY tenant_isolation ON users
    USING (tenant_id = current_setting('app.tenant_id'));
  
  ALTER TABLE users ENABLE ROW LEVEL SECURITY;
  
  -- Every query automatically filtered:
  SET app.tenant_id = 'acme';
  SELECT * FROM users;  -- Only returns acme users

When to use Pool:
  ✅ Many small tenants (thousands to millions)
  ✅ Cost sensitivity is primary driver
  ✅ Similar usage patterns across tenants
  ✅ Fast onboarding required (no provisioning)
  ❌ Cannot provide dedicated resource guarantees
  ❌ Harder to comply with strict data residency
```

### 2.3 Bridge Model (Hybrid / Tiered)

```
Bridge Model — Different tiers get different isolation:

SMB Plan (Pooled)         Business Plan           Enterprise Plan (Silo)
─────────────────         (Bridge)                ──────────────────────
Shared app               Shared app              Dedicated app
Shared DB                Shared app              Dedicated DB
                         Dedicated DB schema     Dedicated infra
                         (schema-per-tenant)     Dedicated VPC

Routing logic:
  ─ Control plane knows each tenant's tier
  ─ API Gateway routes to appropriate cluster/namespace
  ─ Tenant routing table: tenant_id → {tier, endpoint, db}

Tenant Routing Table:
  tenant_id  │ tier        │ app_endpoint               │ db_schema
  ───────────┼─────────────┼────────────────────────────┼──────────
  acme       │ enterprise  │ https://acme.app.company.com│ dedicated
  globex     │ business    │ https://api.app.company.com │ globex
  initech    │ smb         │ https://api.app.company.com │ public
```

---

## 3. Data Isolation Strategies

### 3.1 Database Isolation Options

```
Option 1: Shared Database, Shared Schema (Row-level)
  ─ tenant_id column on every table
  ─ Application or RLS filters data
  
  Pros: Simplest to manage, cheapest
  Cons: RLS bugs can leak data; schema changes affect all tenants
  Risk: HIGH if not implemented carefully

Option 2: Shared Database, Separate Schemas
  ─ Each tenant has their own PostgreSQL schema
  ─ Tables: acme.users, globex.users, etc.
  ─ Connection: SET search_path TO 'acme'
  
  Pros: Schema isolation, easy backup per tenant
  Cons: Schema migrations need to run N times
        PostgreSQL: ~10K schemas max before perf issues
  Risk: MEDIUM

Option 3: Separate Databases (same server)
  ─ Each tenant has their own database
  ─ Different connection string per tenant
  
  Pros: Strong isolation, easy restore/export per tenant
  Cons: Connection pooling complexity (PgBouncer per DB)
  Risk: LOW

Option 4: Separate Database Servers (Silo)
  ─ Completely separate RDS instances
  
  Pros: Full compute + storage isolation
  Cons: High cost, high ops overhead
  Risk: LOWEST
```

### 3.2 Schema Migration at Scale

```
Challenge: "How do I run a migration on 10,000 tenant schemas?"

Expand-Contract Migration Pattern:
  Phase 1 — EXPAND (backward compatible)
    ─ Add new column (nullable)
    ─ Both old and new code works
    ─ Migrate data in background batches
  
  Phase 2 — CONTRACT (after all tenants migrated)
    ─ Drop old column
    ─ Make new column NOT NULL
  
  Migration Runner for Multi-tenant:
  ┌─────────────────────────────────┐
  │  for each tenant in tenants:   │
  │    if tenant.schema_version     │
  │       < migration.version:     │
  │      try:                       │
  │        apply_migration(tenant)  │
  │        update_version(tenant)   │
  │      except Exception:          │
  │        log_and_skip(tenant)     │
  │        # Don't block other      │
  │        # tenant migrations      │
  └─────────────────────────────────┘
  
  Progress tracking:
  migrations table:
    tenant_id | migration_id | applied_at | status
```

### 3.3 Data Residency & Compliance

```
Data Residency Architecture (GDPR / data sovereignty):

Tenant metadata stored centrally:
  ─ Tenant: acme, region: eu-west-1, tier: enterprise

Data stored in tenant's designated region:
  Global Router → reads tenant region → routes to regional stack
  
  ┌─────────────────┐
  │  Global Control  │ ← tenant registry, billing, auth
  │  Plane (us-east) │
  └────────┬────────┘
           │ route by tenant region
    ┌──────┴──────┐
    │             │
┌───▼───┐     ┌───▼───┐
│US Data│     │EU Data│     (future: APAC, etc.)
│Plane  │     │Plane  │
│  App  │     │  App  │
│  DB   │     │  DB   │
└───────┘     └───────┘

Key design rules:
  ✓ EU tenant data NEVER leaves EU regions
  ✓ Audit logs show data locality compliance
  ✓ Tenant region is immutable once set
  ✓ Cross-region requests blocked at API Gateway
```

---

## 4. Identity & Access in SaaS

### 4.1 Tenant-Aware Authentication

```
SaaS Auth Flow:

User: alice@acme.com
  │
  ▼
Tenant Resolution (from email domain or subdomain):
  acme.app.company.com → tenant: acme
  OR
  email domain: acme.com → tenant: acme

  ▼
Identity Provider (per tenant or shared):
  ─ Shared IdP: all tenants use company SSO
  ─ Per-tenant IdP: enterprise customers bring their own IdP
    (acme uses Okta, globex uses Azure AD)
  ─ SAML 2.0 / OIDC federation

  ▼
JWT with tenant claims:
  {
    "sub": "user-uuid",
    "tenant_id": "acme",
    "tenant_tier": "enterprise",
    "roles": ["admin"],
    "permissions": ["read:reports", "write:users"],
    "iss": "https://auth.company.com",
    "exp": 1710000000
  }

  ▼
Every service validates:
  1. Token signature (asymmetric key from JWKS endpoint)
  2. tenant_id matches requested resource's tenant
  3. required permission in token claims
```

### 4.2 RBAC in Multi-Tenant SaaS

```
Multi-Tenant RBAC Model:

Global Roles (platform-level):
  ─ super_admin: platform operators (cross-tenant access)
  ─ support_agent: read-only cross-tenant (scoped)

Tenant-Scoped Roles (customer-defined):
  ─ owner: full access within tenant
  ─ admin: manage users + settings
  ─ member: standard access
  ─ viewer: read-only
  ─ custom_role: tenant defines their own permissions

Database model:
  tenants:       tenant_id, name, tier, settings
  users:         user_id, email, tenant_id
  roles:         role_id, tenant_id (NULL = global), name, permissions
  user_roles:    user_id, role_id, tenant_id

Permission check (in every API handler):
  def require_permission(permission: str):
      def decorator(fn):
          def wrapper(request):
              user = request.user
              tenant = request.tenant
              
              # Platform admin bypasses tenant checks
              if user.has_global_role("super_admin"):
                  return fn(request)
              
              # Verify user belongs to this tenant
              if user.tenant_id != tenant.id:
                  raise TenantMismatchError()
              
              # Check tenant-scoped permission
              if not user.has_permission(permission, tenant.id):
                  raise PermissionDeniedError(permission)
              
              return fn(request)
          return wrapper
      return decorator
```

### 4.3 Bring Your Own Identity (BYOI)

Enterprise customers often require SSO integration:

```
SAML 2.0 Integration per Tenant:

Admin Setup (done once per enterprise tenant):
  1. Tenant admin provides: IdP metadata XML (Okta/Azure AD/PingID)
  2. Platform provides: SP entity ID + ACS URL for this tenant
  3. Platform stores: cert, SSO URL, entity ID per tenant_id

Login Flow:
  User visits: acme.app.company.com/login
  ─ Platform detects tenant: acme
  ─ Checks: acme has SAML configured? → yes
  ─ Redirects to acme's Okta (SP-initiated SAML)
  ─ Okta authenticates user, returns SAML assertion
  ─ Platform validates assertion, creates session JWT
  ─ User is logged in as acme member

SCIM Provisioning (optional enterprise feature):
  ─ When IT admin adds user to "App Users" group in Okta:
    → Okta calls SCIM endpoint: POST /scim/v2/Users
    → User automatically created in platform (just-in-time)
  ─ When user is removed:
    → Okta calls: DELETE /scim/v2/Users/{id}
    → User deactivated in platform within minutes
```

---

## 5. Tenant Onboarding & Provisioning

### 5.1 Tenant Provisioning State Machine

```
Tenant Lifecycle:

SIGNUP ──► PROVISIONING ──► ACTIVE ──► SUSPENDED ──► CANCELLED
              │                │
              ▼                ▼
           FAILED         UPGRADING (tier change)

Provisioning Steps (automated):
  1. Create tenant record (tenant_id, slug, tier)
  2. Provision data store (create schema or DB)
  3. Run migrations on new tenant schema
  4. Seed default data (roles, settings, sample data)
  5. Create initial admin user + send invite email
  6. Configure billing (Stripe customer + subscription)
  7. Configure SSO defaults (if enterprise)
  8. Warm up caches
  9. Mark tenant ACTIVE
  10. Send welcome email + onboarding checklist

Target: < 60 seconds end-to-end for pooled tier
        < 5 minutes for silo tier (infra provisioning)
```

### 5.2 Tenant Provisioning Code Pattern

```python
# Tenant provisioning workflow (using Temporal or Airflow)

class TenantProvisioningWorkflow:

    async def provision(self, request: TenantProvisionRequest) -> Tenant:
        tenant_id = generate_tenant_id(request.slug)

        try:
            # 1. Create tenant in registry (idempotent)
            tenant = await self.registry.create_tenant(
                tenant_id=tenant_id,
                name=request.name,
                tier=request.tier,
                status=TenantStatus.PROVISIONING
            )

            # 2. Provision data store based on tier
            if request.tier == Tier.SMB:
                await self.db_provisioner.create_schema(tenant_id)
            elif request.tier == Tier.ENTERPRISE:
                await self.db_provisioner.create_database(tenant_id)
                await self.infra_provisioner.create_k8s_namespace(tenant_id)

            # 3. Run migrations (idempotent — safe to retry)
            await self.migrator.migrate_to_latest(tenant_id)

            # 4. Seed default data
            await self.seeder.seed_defaults(tenant_id)

            # 5. Create admin user
            admin_user = await self.user_service.create_user(
                tenant_id=tenant_id,
                email=request.admin_email,
                role="owner"
            )
            await self.email_service.send_welcome(admin_user)

            # 6. Configure billing
            stripe_customer = await self.billing.create_customer(tenant, request)
            await self.billing.create_subscription(stripe_customer, request.plan)

            # 7. Mark active
            await self.registry.update_status(tenant_id, TenantStatus.ACTIVE)

            await self.metrics.record_tenant_provisioned(request.tier)
            return tenant

        except Exception as e:
            await self.registry.update_status(tenant_id, TenantStatus.FAILED)
            await self.cleanup(tenant_id)  # Rollback what was created
            raise ProvisioningError(tenant_id, e)
```

---

## 6. Noisy Neighbor & Resource Management

### 6.1 The Noisy Neighbor Problem

```
Noisy Neighbor:

Normal state:
  Tenant A: 100 RPS (normal)
  Tenant B: 100 RPS (normal)
  Tenant C: 100 RPS (normal)
  Total: 300 RPS — shared infra handles fine

Noisy neighbor event:
  Tenant A: 10,000 RPS (runaway job / DDoS / viral growth)
  Tenant B: 100 RPS ← degraded! (starved of resources)
  Tenant C: 100 RPS ← degraded!

Result: B and C paying customers experience outage
        due to A's behavior — unacceptable in SaaS
```

### 6.2 Rate Limiting & Throttling

```
Multi-Tenant Rate Limiting:

Per-tenant limits (based on tier):

  Tier     │ Req/min  │ Concurrent │ Data bandwidth
  ─────────┼──────────┼────────────┼───────────────
  SMB      │  1,000   │     10     │   10 MB/req
  Business │ 10,000   │     50     │   50 MB/req
  Enterpr. │100,000   │    500     │  500 MB/req
  Custom   │    *     │      *     │       *

Implementation (token bucket per tenant):
  
  # Redis-backed per-tenant rate limiter
  def check_rate_limit(tenant_id: str, tier: Tier) -> bool:
      key = f"ratelimit:{tenant_id}:{current_minute()}"
      limit = TIER_LIMITS[tier].requests_per_minute

      current = redis.incr(key)
      if current == 1:
          redis.expire(key, 60)  # expire after 1 minute
      
      if current > limit:
          metrics.increment("rate_limit_exceeded", tenant=tenant_id)
          return False  # → return 429 Too Many Requests
      return True

Graduated throttling (better than hard cutoff):
  80% of limit → add 50ms delay (slow down, don't fail)
  100% of limit → return 429 with Retry-After header
  Sustained abuse → alert and consider suspension
```

### 6.3 Kubernetes Resource Quotas per Tenant

For silo/bridge models with per-tenant k8s namespaces:

```yaml
# ResourceQuota per tenant namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-quota
  namespace: tenant-acme
spec:
  hard:
    requests.cpu: "10"          # Total CPU requests across all pods
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    pods: "50"                  # Max number of pods
    services: "20"
    persistentvolumeclaims: "10"

---
# LimitRange: default + max per container
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-limits
  namespace: tenant-acme
spec:
  limits:
    - type: Container
      default:          # Applied if not specified
        cpu: "500m"
        memory: 512Mi
      max:              # Cannot exceed
        cpu: "4"
        memory: 8Gi
      min:
        cpu: "100m"
        memory: 128Mi
```

---

## 7. Metering & Billing Architecture

### 7.1 Usage Metering Architecture

```
Metering Pipeline:

Application Events          Metering Pipeline          Billing
──────────────────          ─────────────────          ───────

API requests          ──►  Event collector      ──►  Aggregator
File uploads               (Kafka topic)             (hourly)
Active users                                          │
API calls                                             ▼
Compute hours              Deduplication         Usage records
Storage bytes              (exactly-once)        (per tenant, per
Emails sent                     │                metric, per hour)
                                ▼                     │
                          Meter events                 ▼
                          table (append            Billing system
                          only)                   (Stripe / custom)
                                                        │
                                                        ▼
                                                    Invoice
                                                    generation
```

### 7.2 Metered Billing Model

```python
# Usage metering event schema
class MeterEvent:
    tenant_id: str
    metric_name: str         # "api_calls", "storage_bytes", "active_users"
    quantity: Decimal
    timestamp: datetime
    idempotency_key: str     # Prevent double-counting on retry

# Metering ingestion API (called by all services)
class MeteringService:
    def record(self, event: MeterEvent):
        # Idempotent write — safe to retry
        if self.already_processed(event.idempotency_key):
            return

        self.kafka_producer.produce(
            topic="metering.events",
            key=event.tenant_id,
            value=event.to_json()
        )

# Aggregation job (hourly, via Spark or Flink)
class UsageAggregator:
    def aggregate_hourly(self, hour: datetime):
        events = self.read_events(hour)
        
        usage = events.groupBy("tenant_id", "metric_name") \
                       .agg(sum("quantity").alias("total"))
        
        for row in usage:
            self.usage_db.upsert(
                tenant_id=row.tenant_id,
                metric=row.metric_name,
                period=hour,
                quantity=row.total
            )
            
            # Report to Stripe Metered Billing
            stripe.billing_meter_events.create(
                event_name=row.metric_name,
                payload={
                    "stripe_customer_id": self.get_stripe_id(row.tenant_id),
                    "value": str(row.total),
                }
            )
```

### 7.3 Pricing Models

```
SaaS Pricing Model Options:

FLAT RATE
  $99/month regardless of usage
  Predictable for customer, unpredictable for provider
  Good for: stable usage patterns

USAGE-BASED (Pay-as-you-go)
  $0.001 per API call
  Scales with customer value
  Good for: variable usage, large range of customer sizes

TIERED (Volume discounts)
  0-10K calls:  $0.001/call
  10K-100K:     $0.0008/call
  100K+:        $0.0005/call
  Good for: encouraging growth, standard SaaS

SEAT-BASED
  $30/user/month
  Simple to understand, easy to forecast
  Good for: collaboration tools, productivity software

HYBRID (most enterprise SaaS)
  Base fee: $500/month (includes 10K API calls)
  Overage:  $0.05 per 1K additional calls
  Good for: predictable base + usage flexibility

Per-Unit Economics (what to track):
  CAC: Cost to Acquire Customer
  LTV: Lifetime Value
  COGS per tenant: infra + support cost
  LTV/CAC ratio: should be > 3x for healthy SaaS
```

---

## 8. SaaS Deployment Patterns

### 8.1 Tenant-Aware Routing

```
Request routing for multi-tenant SaaS:

Subdomain-based routing:
  acme.app.company.com → tenant: acme
  globex.app.company.com → tenant: globex
  
  Wildcard SSL cert: *.app.company.com
  Nginx / API Gateway resolves tenant from Host header

Path-based routing:
  app.company.com/acme/api/... → tenant: acme
  Simpler cert management, but less white-labeling

Custom domain routing (enterprise):
  crm.acme.com → CNAME → acme.app.company.com
  Platform looks up: acme.app.company.com → tenant: acme

┌─────────────────────────────────────────────┐
│             Tenant Routing Middleware        │
│                                             │
│  1. Extract tenant from Host/path/JWT       │
│  2. Load tenant config (tier, features,     │
│     region, status)                         │
│  3. Inject tenant_id into request context   │
│  4. Check tenant status (active? suspended?)│
│  5. Route to appropriate backend            │
└─────────────────────────────────────────────┘
```

### 8.2 Feature Flags per Tenant

```python
# Tenant-aware feature flagging

class TenantFeatureFlags:
    """
    Features can be enabled:
    - Globally (all tenants)
    - By tier (enterprise only)
    - By specific tenant (beta testers)
    - By percentage rollout
    """

    def is_enabled(self, feature: str, tenant: Tenant) -> bool:
        flag = self.get_flag(feature)

        if not flag.enabled:
            return False

        # Specific tenant override (whitelist/blacklist)
        if tenant.id in flag.enabled_tenants:
            return True
        if tenant.id in flag.disabled_tenants:
            return False

        # Tier-based enablement
        if flag.enabled_tiers and tenant.tier not in flag.enabled_tiers:
            return False

        # Percentage rollout (consistent per tenant)
        if flag.rollout_percentage < 100:
            hash_value = int(hashlib.md5(
                f"{feature}:{tenant.id}".encode()
            ).hexdigest(), 16)
            if (hash_value % 100) >= flag.rollout_percentage:
                return False

        return True

# Usage in API handler
@require_permission("reports:export")
def export_report(request):
    if feature_flags.is_enabled("advanced_export", request.tenant):
        return export_advanced(request)
    return export_basic(request)
```

### 8.3 Blue-Green & Canary for Multi-Tenant

```
Deploying to multi-tenant SaaS safely:

Canary by tenant (better than % traffic split):
  Phase 1: Deploy to internal tenants only
    ─ company internal tenants always get new version first
    ─ "dogfood" deployment
    
  Phase 2: Deploy to opted-in beta tenants
    ─ Adventurous customers who signed up for early access
    ─ 5% of tenant base
    
  Phase 3: Deploy to SMB tier
    ─ Lower risk: smaller tenants, faster feedback
    
  Phase 4: Deploy to Business tier
  Phase 5: Deploy to Enterprise tier (last — most risk)
    ─ Enterprise customers often have change freeze windows
    ─ Notify in advance
    ─ Consider maintenance windows in their timezone

Emergency rollback (schema changes are the hardest):
  ─ Never ship non-backward-compatible schema changes
  ─ Expand-contract migration (always additive first)
  ─ Feature flag new behavior behind flag
  ─ Remove flag only after confident in rollout
```

---

## 9. Customization & Configuration

### 9.1 Tenant Configuration Model

```python
# Hierarchical tenant configuration

@dataclass
class TenantConfig:
    # Identity
    tenant_id: str
    slug: str          # used in subdomain
    name: str
    tier: Tier

    # Branding (white-labeling)
    logo_url: Optional[str] = None
    primary_color: str = "#0066CC"
    custom_domain: Optional[str] = None

    # Feature access
    features: Set[str] = field(default_factory=set)
    max_users: int = 10
    max_api_calls_per_month: int = 10_000

    # Integrations
    sso_config: Optional[SSOConfig] = None
    webhook_endpoints: List[str] = field(default_factory=list)
    allowed_ip_ranges: List[str] = field(default_factory=list)  # IP allowlist

    # Regional
    data_region: str = "us-east-1"
    timezone: str = "UTC"
    locale: str = "en-US"

    # Compliance
    data_retention_days: int = 90
    audit_log_enabled: bool = False
    mfa_required: bool = False

# Stored in PostgreSQL, cached in Redis per tenant
# TTL: 5 minutes (balance freshness vs DB load)
```

### 9.2 Extension Points Architecture

```
Tenant Customization Spectrum:

Level 1: Configuration
  ─ Toggle features on/off
  ─ Branding (logo, colors)
  ─ Notification preferences
  No code changes needed

Level 2: Templates & Workflows
  ─ Custom email templates
  ─ Custom approval workflows
  ─ Custom field definitions (dynamic forms)
  Low-code customization

Level 3: Webhooks & Integrations
  ─ Outbound webhooks on events
  ─ Zapier / Make integrations
  ─ Custom integrations via API
  No-code automation

Level 4: Custom Logic (advanced)
  ─ Custom validation rules
  ─ Custom transformation scripts
  ─ Plugin marketplace
  Risk: security sandbox needed (WASM / V8 isolate)

Security for custom code execution:
  ─ Run in isolated WASM sandbox (no system access)
  ─ CPU time limit: 500ms max per execution
  ─ Memory limit: 64MB
  ─ No network access (except pre-approved webhook URLs)
  ─ Audit log every execution
```

---

## 10. Practical Examples

### 10.1 SaaS Architecture Tier Matrix

```
Full SaaS Architecture Decision Matrix:

                    SMB Tier          Business          Enterprise
                    (Pooled)          (Bridge)          (Silo)
────────────────────────────────────────────────────────────────
Compute            Shared pods       Shared pods       Dedicated
Database           Shared + RLS      Schema per tenant  Dedicated DB
Deployments        All at once       All at once       Maintenance window
Rate limiting      1K req/min        10K req/min       Custom SLA
SSO                Username/pw       SAML optional     SAML + SCIM
Custom domain      No                No                Yes
SLA                99.5%             99.9%             99.99%
Support            Email/docs        Email priority    Dedicated CSM
Data residency     US only           US + EU           Any region
Audit logs         No                30 days           1 year
Price              $49/mo            $499/mo           $5K+/mo
```

### 10.2 Tenant-Aware API Design

```python
# FastAPI example: tenant-aware API

from fastapi import FastAPI, Depends, HTTPException
from starlette.requests import Request

app = FastAPI()

# Dependency: resolve tenant from request
async def get_tenant(request: Request) -> Tenant:
    # Strategy 1: from JWT claim
    token = extract_bearer_token(request)
    claims = verify_jwt(token)
    tenant_id = claims["tenant_id"]

    # Strategy 2: from subdomain (fallback)
    if not tenant_id:
        host = request.headers["host"]
        tenant_slug = host.split(".")[0]
        tenant_id = resolve_slug(tenant_slug)

    tenant = await tenant_cache.get(tenant_id)
    if not tenant or tenant.status != TenantStatus.ACTIVE:
        raise HTTPException(403, "Tenant not found or inactive")

    return tenant

# Tenant is automatically injected into every handler
@app.get("/api/reports")
async def list_reports(
    tenant: Tenant = Depends(get_tenant),
    user: User = Depends(get_current_user),
):
    # user.tenant_id is validated against tenant.id in get_current_user
    reports = await report_service.list(tenant_id=tenant.id)
    return reports

# Tenant middleware: inject into request state
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    # Rate limiting check
    tenant = await resolve_tenant_from_request(request)
    if tenant and not rate_limiter.allow(tenant.id, tenant.tier):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"error": "rate_limit_exceeded"}
        )
    response = await call_next(request)
    return response
```

### 10.3 Reference SaaS Architecture Stack

```yaml
# SaaS Platform Reference Stack

control_plane:
  tenant_registry: postgresql          # Source of truth for tenant config
  tenant_cache: redis                  # Cached tenant config (5 min TTL)
  provisioning_workflow: temporal      # Reliable async provisioning
  billing: stripe                      # Subscriptions + metered billing
  metering: kafka + spark              # Usage event aggregation

data_plane:
  compute: kubernetes_eks
  app_routing: nginx_ingress           # Subdomain → tenant routing
  rate_limiting: redis_token_bucket    # Per-tenant limits
  auth: auth0_or_custom               # JWT + SAML federation

data_isolation:
  smb_tier: postgresql_rls             # Row-level security
  business_tier: schema_per_tenant     # PostgreSQL schemas
  enterprise_tier: rds_per_tenant      # Dedicated instances

feature_flags: launchdarkly            # Tenant + tier targeting

observability:
  metrics: prometheus_with_tenant_label
  dashboards: grafana_per_tenant_filter
  alerting: per_tenant_slo_alerts
  audit_logs: cloudtrail + custom

customization:
  webhooks: outbound_webhook_service
  custom_domains: cloudfront + acm    # Automated cert provisioning
  white_label: tenant_theme_config
  sso_federation: auth0_saml_connections
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Tenancy Models** | Silo (full isolation) vs Pool (shared) vs Bridge (tiered hybrid) |
| **Data Isolation** | Shared table + RLS (cheap) → Schema-per-tenant → DB-per-tenant (expensive) |
| **Noisy Neighbor** | Rate limit and quota every tenant — never trust a single tenant not to spike |
| **Tenant Routing** | Resolve tenant_id early (from JWT/subdomain), inject into every request |
| **Schema Migration** | Expand-contract pattern; runner iterates all tenant schemas independently |
| **Metering** | Append-only event log → hourly aggregation → billing sync; idempotency is critical |
| **Feature Flags** | Enable features by tenant, tier, or rollout % for safe deployment |
| **SAML/SCIM** | Enterprise customers expect bring-your-own IdP + automated user provisioning |
| **Provisioning** | Automate end-to-end; target < 60s for pooled tier; use idempotent steps for safe retry |
| **Data Residency** | Route by tenant's designated region from the start — retrofitting is very hard |
