# Database Scalability: Guide for Architects

## Table of Contents
1. [Introduction](#1-introduction)
2. [Replication](#2-replication)
3. [Sharding (Horizontal Partitioning)](#3-sharding-horizontal-partitioning)
4. [Caching Strategies](#4-caching-strategies)
5. [SQL vs NoSQL Trade-offs](#5-sql-vs-nosql-trade-offs)
6. [Read/Write Scaling Patterns](#6-readwrite-scaling-patterns)
7. [Connection Pooling and Query Optimization](#7-connection-pooling-and-query-optimization)
8. [Advanced Patterns](#8-advanced-patterns)
9. [Practical Examples](#9-practical-examples)

---

## 1. Introduction

As load grows, databases become bottlenecks. Scaling strategies:

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| **Vertical scaling** | Quick fix, limited ceiling | Low |
| **Replication** | Read-heavy workloads | Medium |
| **Sharding** | Write-heavy, huge datasets | High |
| **Caching** | Read-heavy, hot data | Medium |
| **Read replicas** | Analytics, reporting | Low |

---

## 2. Replication

### 2.1 Primary-Replica (Single-Leader)

One primary accepts writes; one or more replicas replicate via log shipping or row-based replication.

```
         Writes
            │
            ▼
      ┌──────────┐
      │ Primary  │
      └────┬─────┘
           │ Replication (sync/async)
     ┌─────┼─────┐
     ▼     ▼     ▼
┌────────┐ ┌────────┐ ┌────────┐
│Replica1│ │Replica2│ │Replica3│  <- Reads
└────────┘ └────────┘ └────────┘
```

### 2.2 Sync vs Async Replication

**Synchronous**: Primary waits for N replicas to acknowledge. Strong consistency, higher write latency.

```
Primary: W(x=1) -> Replica1 ACK -> Replica2 ACK -> Respond to client
```

**Asynchronous**: Primary responds immediately; replicas catch up. Lower latency, risk of data loss on failover.

### 2.3 Replication Lag

With async replication, reads from replicas may be stale.

| Mitigation | Description |
|------------|-------------|
| **Read-your-writes** | Route user's reads to primary for a short time after write |
| **Monotonic reads** | Route a session's reads to same replica |
| **Cross-datacenter** | Accept eventual consistency for geo-distributed |

### 2.4 Multi-Leader Replication

Multiple nodes accept writes; conflicts resolved on read (last-write-wins, merge, CRDT).

**Use when**: Multi-region, offline-first (e.g., CouchDB, Cassandra).

---

## 3. Sharding (Horizontal Partitioning)

### 3.1 What Is Sharding?

Split data across multiple databases (shards). Each shard holds a subset of the data.

```
Full dataset: User 1..1M
Shard 1: User 1..333K
Shard 2: User 334K..666K
Shard 3: User 667K..1M
```

### 3.2 Shard Key Selection

Choose a key that:
- Distributes load evenly (avoid hot shards)
- Supports common query patterns (avoid cross-shard queries)
- Doesn't change frequently

**Examples**:
- `user_id` for user-centric data
- `tenant_id` for multi-tenant SaaS
- Hash of `(customer_id, order_date)` for orders

### 3.3 Sharding Strategies

**Range-based**: Shard by key range (e.g., A–M, N–Z). Simple but can create hotspots.

**Hash-based**: `shard = hash(key) % num_shards`. Even distribution; range queries need all shards.

**Directory-based**: Lookup table maps key → shard. Flexible but lookup table is bottleneck.

### 3.4 Consistent Hashing

Used in distributed caches (Redis Cluster, Dynamo) to minimize remapping when nodes join/leave.

```
Ring: 0 ... 2^32
Keys and nodes hashed onto ring.
Key goes to first node clockwise.
Adding/removing node affects only its neighbors.
```

```python
import hashlib

class ConsistentHash:
    def __init__(self, nodes, virtual_nodes=150):
        self.ring = {}
        self.sorted_keys = []
        for node in nodes:
            self.add_node(node, virtual_nodes)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def add_node(self, node, virtual_nodes):
        for i in range(virtual_nodes):
            vkey = f"{node}:{i}"
            h = self._hash(vkey)
            self.ring[h] = node
            self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def get_node(self, key):
        h = self._hash(key)
        for k in self.sorted_keys:
            if h <= k:
                return self.ring[k]
        return self.ring[self.sorted_keys[0]]
```

### 3.5 Challenges of Sharding

- **Cross-shard queries**: Expensive; often avoid or denormalize
- **Transactions**: Cross-shard TX requires 2PC or Saga
- **Rebalancing**: Moving data when adding/removing shards

---

## 4. Caching Strategies

### 4.1 Cache-Aside (Lazy Loading)

Application checks cache first; on miss, loads from DB and populates cache.

```python
def get_user(user_id: str):
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    user = db.query(User).filter_by(id=user_id).first()
    redis.setex(f"user:{user_id}", 3600, json.dumps(user.to_dict()))
    return user
```

### 4.2 Read-Through

Cache provider loads from DB on miss. Application only talks to cache.

### 4.3 Write-Through

Write goes to cache and DB together. Cache always consistent with DB.

### 4.4 Write-Behind (Write-Back)

Write to cache only; async flush to DB. High write throughput; risk of data loss.

### 4.5 Cache Invalidation

| Strategy | Pros | Cons |
|----------|------|------|
| **TTL** | Simple | Stale reads |
| **Invalidate on write** | Fresh | More cache misses |
| **Version/tag** | Flexible | Logic complexity |

```python
# Invalidate on write
def update_user(user_id: str, data: dict):
    db.update(User).where(User.id == user_id).values(**data)
    redis.delete(f"user:{user_id}")
```

### 4.6 Cache Stampede (Thundering Herd)

Many requests miss cache simultaneously and hit DB. Mitigate with:
- **Lock**: First request computes; others wait
- **Probabilistic early expiration**: Refresh before TTL

```python
import redis
from contextlib import contextmanager

def get_with_lock(key, loader, ttl=3600):
    val = redis.get(key)
    if val:
        return json.loads(val)
    lock_key = f"lock:{key}"
    if redis.set(lock_key, "1", nx=True, ex=10):
        try:
            val = loader()
            redis.setex(key, ttl, json.dumps(val))
            return val
        finally:
            redis.delete(lock_key)
    else:
        time.sleep(0.1)
        return get_with_lock(key, loader, ttl)  # Retry
```

---

## 5. SQL vs NoSQL Trade-offs

### 5.1 When to Choose SQL (RDBMS)

- ACID transactions
- Complex joins
- Structured schema, integrity constraints
- Strong consistency

**Examples**: PostgreSQL, MySQL, SQL Server.

### 5.2 When to Choose NoSQL

| Type | Use Case | Examples |
|------|----------|----------|
| **Document** | Flexible schema, nested data | MongoDB, Couchbase |
| **Key-Value** | Simple lookups, caching | Redis, DynamoDB |
| **Wide-Column** | Time-series, high write throughput | Cassandra, ScyllaDB |
| **Graph** | Relationships, traversals | Neo4j, Amazon Neptune |

### 5.3 Polyglot Persistence

Use different stores for different needs:

```
Orders -> PostgreSQL (transactions)
Sessions -> Redis (speed)
Analytics -> ClickHouse (columnar)
Search -> Elasticsearch (full-text)
```

---

## 6. Read/Write Scaling Patterns

### 6.1 CQRS (Command Query Responsibility Segregation)

Separate write model from read model. Writes go to primary; read-optimized stores updated via events.

```
Command: CreateOrder -> Write DB -> Publish OrderCreated
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              Read Model 1             Read Model 2             Read Model 3
              (Order list)             (Order details)           (Analytics)
```

### 6.2 Read Replicas for Reporting

Route heavy analytics/reporting to dedicated replicas so OLTP stays fast.

### 6.3 Write Scaling

- **Sharding** for write throughput
- **Async writes** (queue + worker) for non-critical writes
- **Batching** inserts where possible

---

## 7. Connection Pooling and Query Optimization

### 7.1 Connection Pooling

Reuse DB connections instead of creating per request.

```python
# SQLAlchemy connection pool
engine = create_engine(
    "postgresql://user:pass@host/db",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)
```

### 7.2 Query Optimization

- **Indexes**: B-tree for equality/range; GIN for full-text; composite for common filters
- **Explain Analyze**: Identify slow queries
- **Avoid N+1**: Use eager loading (e.g., `joinedload` in SQLAlchemy)
- **Pagination**: Cursor-based for large datasets (offset degrades)

```sql
-- Cursor-based pagination (efficient)
SELECT * FROM orders
WHERE id > last_seen_id
ORDER BY id
LIMIT 100;
```

---

## 8. Advanced Patterns

### 8.1 Database per Service (Microservices)

Each service owns its DB. No shared DB. Communicate via APIs/events.

### 8.2 Outbox Pattern

Ensure exactly-once event publishing with transactional outbox.

```
1. BEGIN TX
2. INSERT INTO orders (...)
3. INSERT INTO outbox (aggregate_id, event_type, payload)
4. COMMIT
5. Background worker: read outbox -> publish to Kafka -> mark published
```

### 8.3 Event Sourcing

Store events, not current state. State = replay of events.

### 8.4 Time-Series Databases

Optimized for time-series data (metrics, logs): InfluxDB, TimescaleDB, Prometheus.

---

## 9. Practical Examples

### 9.1 Shard Routing (Python)

```python
from hashlib import sha256

class ShardedDB:
    def __init__(self, shard_urls: list[str]):
        self.shards = [create_engine(url) for url in shard_urls]
        self.n = len(self.shards)

    def _shard_index(self, key: str) -> int:
        h = int(sha256(key.encode()).hexdigest(), 16)
        return h % self.n

    def get_connection(self, shard_key: str):
        return self.shards[self._shard_index(shard_key)]

    def get_order(self, order_id: str):
        conn = self.get_connection(order_id)
        return conn.execute(
            "SELECT * FROM orders WHERE id = %s", (order_id,)
        ).fetchone()
```

### 9.2 Read-Through Cache with TTL

```python
from functools import wraps
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def cached(ttl=3600, key_prefix="cache"):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{f.__name__}:{args}:{kwargs}"
            cached_val = redis_client.get(cache_key)
            if cached_val:
                return json.loads(cached_val)
            result = f(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            return result
        return wrapper
    return decorator

@cached(ttl=300)
def get_product(product_id: str):
    return db.query(Product).get(product_id)
```

### 9.3 Connection Pool with PgBouncer Config

```ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_addr = *
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Replication** | Primary-replica for reads; sync vs async trade-off |
| **Sharding** | Hash or range; shard key is critical |
| **Caching** | Cache-aside common; invalidate on write |
| **SQL vs NoSQL** | ACID + joins vs scale + flexibility |
| **CQRS** | Separate read/write models for scale |
| **Connection pooling** | Essential for high concurrency |

---

## Further Reading

- *Designing Data-Intensive Applications* — Martin Kleppmann
- PostgreSQL replication: https://www.postgresql.org/docs/current/warm-standby.html
- Redis Cluster: https://redis.io/docs/management/scaling/
