# Distributed Systems: Essential Guide for Architects

## Table of Contents
1. [Introduction](#1-introduction)
2. [CAP Theorem and PACELC](#2-cap-theorem-and-pacelc)
3. [Consistency Models](#3-consistency-models)
4. [Consensus Algorithms](#4-consensus-algorithms)
5. [Distributed Transactions](#5-distributed-transactions)
6. [Replication Strategies](#6-replication-strategies)
7. [Failure Modes and Fault Tolerance](#7-failure-modes-and-fault-tolerance)
8. [Time and Ordering](#8-time-and-ordering)
9. [Advanced Topics](#9-advanced-topics)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

A **distributed system** is a collection of autonomous nodes that communicate over a network and coordinate to achieve a common goal. Distributed systems power modern applications at scale.

### Why Distributed?

| Driver | Example |
|--------|---------|
| **Scale** | Data or load exceeds single-node capacity |
| **Availability** | No single point of failure |
| **Latency** | Place compute/data closer to users |
| **Specialization** | Different services, different stacks |

### Core Challenges

- **Partial failure**: Nodes fail independently; network partitions occur.
- **Concurrency**: Multiple nodes operate concurrently.
- **No global clock**: Time is not perfectly synchronized.
- **Message delays**: Messages can be reordered, lost, or delayed.

---

## 2. CAP Theorem and PACELC

### CAP Theorem (Brewer)

In the presence of a **network partition**, you can guarantee at most **two** of:

- **C**onsistency — Every read returns the most recent write.
- **A**vailability — Every request receives a non-error response.
- **P**artition tolerance — System continues despite network partitions.

**Partition tolerance is mandatory** in real networks. You effectively choose between:

- **CP**: Consistency over availability (e.g., etcd, ZooKeeper)
- **AP**: Availability over consistency (e.g., Cassandra, DynamoDB)

```
        Consistency
              /\
             /  \
            / CP \        AP
           /------\     (Cassandra,
          /   CA   \    DynamoDB)
         /    X     \
        /------------\
   Partition      Availability
   Tolerance
```

### PACELC Extension

When there is **no partition**, you choose between **L**atency and **C**onsistency:

- **EL**: Low latency, weaker consistency (e.g., read from replicas)
- **EC**: Stronger consistency, higher latency (e.g., read from primary)

**Examples**:
- DynamoDB: PA/EL (available during partition; low latency when healthy)
- MongoDB: PC/EC (consistent; higher latency for writes)
- Cassandra: PA/EL (available; tunable consistency per operation)

---

## 3. Consistency Models

### 3.1 Strong Consistency (Linearizable)

A read returns the effect of the most recent write. Linearizability provides the illusion of a single copy of the data.

```
Timeline:  W(x=1) -------- W(x=2) -------- R(x)
                                    ↑
                              Must return 2
```

**Use when**: Financial transactions, leader election, configuration.

### 3.2 Sequential Consistency

Operations appear in some global order; all nodes see the same order, but it may not be real-time.

### 3.3 Causal Consistency

Preserves cause-effect order. If A happened-before B, all nodes see A before B.

```
Node 1: W(x=1) → W(x=2)
Node 2: R(x)=1 → R(x)=2   ✓ Causal
Node 2: R(x)=2 → R(x)=1   ✗ Violates causal order
```

### 3.4 Eventual Consistency

If no new updates, all replicas will eventually converge. No guarantee on order or latency.

**Use when**: Social feeds, analytics, non-critical counters.

### 3.5 Read-Your-Writes Consistency

A client always sees its own writes. Common in session-based systems.

### 3.6 Monotonic Reads

If a client reads value X, subsequent reads never see an older value than X.

---

## 4. Consensus Algorithms

Consensus allows a group of nodes to agree on a value despite failures.

### 4.1 Problem: Byzantine vs Crash Faults

- **Crash faults**: Nodes stop responding.
- **Byzantine faults**: Nodes behave arbitrarily (malicious or buggy).

Byzantine fault-tolerant (BFT) consensus requires more nodes (e.g., 3f+1 for f failures).

### 4.2 Paxos (Classic)

Nodes propose values; a majority must accept for consensus.

**Roles**:
- **Proposer**: Proposes values
- **Acceptor**: Votes on proposals
- **Learner**: Learns the chosen value

**Phases**:
1. **Prepare**: Proposer sends proposal number n; acceptors promise not to accept proposals &lt; n.
2. **Accept**: If majority promised, proposer sends (n, v); acceptors accept if they haven't promised a higher n.

**Complexity**: Hard to implement; many production systems use Raft instead.

### 4.3 Raft

Raft is designed to be understandable. Used by etcd, Consul, TiKV.

**Roles**:
- **Leader**: Handles all client requests; replicates log to followers
- **Follower**: Passive; responds to leader/fandidate
- **Candidate**: Requests votes to become leader

**Terms**: Time divided into terms; each term has at most one leader.

**Log Replication**:
1. Client sends command to leader.
2. Leader appends to its log.
3. Leader sends AppendEntries to followers.
4. When majority replicates, leader commits and applies.
5. Leader responds to client.

**Leader Election**:
1. Follower times out, becomes candidate.
2. Increments term, votes for self, requests votes.
3. If majority votes, becomes leader.
4. Sends heartbeats to maintain leadership.

```python
# Conceptual Raft state machine (simplified)
class RaftNode:
    def __init__(self):
        self.state = "follower"  # follower | candidate | leader
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0

    def append_entries(self, term, leader_id, prev_log_index, entries):
        if term < self.current_term:
            return False
        # ... append entries, update commit_index
        return True

    def request_vote(self, term, candidate_id, last_log_index, last_log_term):
        if term < self.current_term:
            return False
        if self.voted_for is None or self.voted_for == candidate_id:
            if last_log_index >= len(self.log) - 1:
                self.voted_for = candidate_id
                return True
        return False
```

### 4.4 Practical Consensus: etcd Example

```bash
# etcd uses Raft; 3-node cluster
etcd --name node1 --listen-client-urls http://localhost:2379 \
     --advertise-client-urls http://localhost:2379

# Put a value (goes through leader, replicated)
etcdctl put mykey "myvalue"

# Get (can be served by any node; linearizable by default)
etcdctl get mykey
```

---

## 5. Distributed Transactions

### 5.1 Two-Phase Commit (2PC)

**Coordinator** orchestrates; **participants** execute.

**Phase 1 (Prepare)**:
1. Coordinator sends PREPARE to all participants.
2. Each participant executes locally, writes to undo/redo log, responds YES/NO.

**Phase 2 (Commit/Abort)**:
3. If all YES → coordinator sends COMMIT; all commit.
4. If any NO → coordinator sends ABORT; all abort.

**Problems**:
- **Blocking**: If coordinator fails after PREPARE, participants may block.
- **Availability**: One failed participant aborts the whole transaction.
- **Performance**: Two round-trips, locks held during both.

### 5.2 Three-Phase Commit (3PC)

Adds a **pre-commit** phase to reduce blocking. Still assumes no network partition; rarely used in practice.

### 5.3 Saga Pattern

Replace ACID with compensating transactions across services.

**Choreography**: Each service emits events; others react and run compensations if needed.

**Orchestration**: Central orchestrator calls services and runs compensations on failure.

```
Order Saga (Orchestration):
1. OrderService -> ReserveInventory
2. OrderService -> ProcessPayment
3. OrderService -> ShipOrder
   [If 2 fails] -> Compensate: ReleaseInventory
   [If 3 fails] -> Compensate: RefundPayment, ReleaseInventory
```

**Example: Saga with Events**

```python
# Simplified saga orchestrator
class OrderSaga:
    def execute(self, order):
        try:
            self.reserve_inventory(order)
            self.charge_payment(order)
            self.ship_order(order)
            self.complete_order(order)
        except InventoryError:
            self.compensate_inventory(order)
        except PaymentError:
            self.compensate_inventory(order)
            self.refund_payment(order)
        except ShippingError:
            self.compensate_inventory(order)
            self.refund_payment(order)
            self.cancel_shipment(order)
```

### 5.4 Outbox Pattern

Ensure exactly-once delivery when publishing events from a transactional DB:

1. In the same DB transaction: insert business record + insert outbox event.
2. Background process polls outbox, publishes to message broker, marks as published.

```
DB Transaction:
  INSERT INTO orders (...) 
  INSERT INTO outbox (aggregate_id, event_type, payload)
COMMIT

Background worker:
  SELECT * FROM outbox WHERE published = false
  Publish to Kafka
  UPDATE outbox SET published = true
```

---

## 6. Replication Strategies

### 6.1 Single-Leader (Primary-Replica)

One primary accepts writes; replicas replicate asynchronously or synchronously.

- **Sync replication**: Primary waits for N replicas before responding → strong consistency, higher latency.
- **Async replication**: Primary responds immediately → lower latency, risk of data loss on failover.

### 6.2 Multi-Leader

Multiple nodes accept writes; conflicts must be resolved (last-write-wins, merge, CRDTs).

**Use when**: Multi-region, offline-first.

### 6.3 Leaderless (Dynamo-Style)

No designated leader. Writes go to N nodes; reads from N nodes; quorum (e.g., W + R > N) for consistency.

**Example**: Cassandra, DynamoDB, Riak.

```
Write: W=2 of N=3
Read: R=2 of N=3
If W + R > N → read quorum overlaps write quorum → consistent
```

---

## 7. Failure Modes and Fault Tolerance

### 7.1 Failure Types

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Crash | Timeout, heartbeat | Replication, failover |
| Partition | Split brain risk | Quorum, consensus |
| Byzantine | Verification | BFT, signatures |
| Slow | Latency SLO | Timeouts, circuit breaker |

### 7.2 Timeouts and Retries

- **Idempotency**: Retries must be safe. Use idempotency keys.
- **Exponential backoff**: Avoid thundering herd.
- **Timeout tuning**: Too short → false failures; too long → delayed detection.

### 7.3 Circuit Breaker

Prevent cascading failures when a dependency is unhealthy:

```
States: Closed → Open → Half-Open → Closed
```

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def call_downstream_service():
    return requests.get("http://downstream/api")
```

### 7.4 Bulkhead

Isolate resources (thread pools, connections) so one failing component doesn't exhaust all.

---

## 8. Time and Ordering

### 8.1 Physical Clocks (NTP)

- Drift: ~ms per day with NTP.
- Not sufficient for ordering across nodes.

### 8.2 Logical Clocks

- **Lamport timestamp**: Increment on send/receive; order events but not causality.
- **Vector clocks**: Track causality per node; detect concurrent events.

### 8.3 Hybrid Logical Clocks (HLC)

Combine physical + logical. Used in CockroachDB, MongoDB for ordering.

### 8.4 TrueTime (Spanner)

Google Spanner uses GPS + atomic clocks for bounded uncertainty (e.g., ±7ms). Enables external consistency.

---

## 9. Advanced Topics

### 9.1 Gossip Protocols

Gossip protocols spread information through a network via peer-to-peer communication. Each node periodically exchanges state with a random subset of other nodes.

**Use cases**: Membership detection, failure detection, configuration propagation.

**Examples**: Cassandra (ring membership), Consul (service discovery), Bitcoin (transaction broadcast).

**Properties**:
- **Convergence**: All nodes eventually receive information
- **Fault tolerance**: Continues despite node failures
- **Scalability**: O(log N) rounds to reach all N nodes

```java
// Simplified gossip protocol implementation
public class GossipProtocol {
    private final Map<String, NodeInfo> membershipTable = new ConcurrentHashMap<>();
    private final Set<String> knownNodes;
    private final ScheduledExecutorService scheduler;
    
    public void start() {
        scheduler.scheduleAtFixedRate(this::gossipRound, 1, 1, TimeUnit.SECONDS);
    }
    
    private void gossipRound() {
        List<String> randomNodes = selectRandomNodes(3);
        for (String node : randomNodes) {
            try {
                GossipMessage msg = createGossipMessage();
                GossipMessage response = sendGossip(node, msg);
                mergeMembershipTable(response.getMembershipTable());
            } catch (Exception e) {
                markNodeAsSuspected(node);
            }
        }
    }
    
    private void mergeMembershipTable(Map<String, NodeInfo> remoteTable) {
        for (Map.Entry<String, NodeInfo> entry : remoteTable.entrySet()) {
            NodeInfo local = membershipTable.get(entry.getKey());
            NodeInfo remote = entry.getValue();
            if (local == null || remote.getVersion() > local.getVersion()) {
                membershipTable.put(entry.getKey(), remote);
            }
        }
    }
}
```

### 9.2 Vector Clocks and Causal Ordering

Vector clocks track causal relationships between events in distributed systems.

```java
public class VectorClock {
    private final Map<String, Integer> clock = new HashMap<>();
    private final String nodeId;
    
    public VectorClock(String nodeId, Set<String> allNodes) {
        this.nodeId = nodeId;
        for (String node : allNodes) {
            clock.put(node, 0);
        }
    }
    
    // Increment local counter before sending message
    public VectorClock tick() {
        clock.put(nodeId, clock.get(nodeId) + 1);
        return this.copy();
    }
    
    // Update clock on message receive
    public VectorClock update(VectorClock other) {
        for (String node : clock.keySet()) {
            clock.put(node, Math.max(clock.get(node), other.clock.get(node)));
        }
        clock.put(nodeId, clock.get(nodeId) + 1);
        return this;
    }
    
    // Check if this event happened before other
    public boolean happensBefore(VectorClock other) {
        boolean allLessOrEqual = true;
        boolean atLeastOneLess = false;
        
        for (String node : clock.keySet()) {
            int thisValue = clock.get(node);
            int otherValue = other.clock.get(node);
            
            if (thisValue > otherValue) {
                allLessOrEqual = false;
                break;
            } else if (thisValue < otherValue) {
                atLeastOneLess = true;
            }
        }
        
        return allLessOrEqual && atLeastOneLess;
    }
    
    // Events are concurrent if neither happens-before the other
    public boolean isConcurrent(VectorClock other) {
        return !this.happensBefore(other) && !other.happensBefore(this);
    }
}
```

### 9.3 Merkle Trees

Binary trees where each leaf is a hash of data block, and each internal node is a hash of its children. Used for efficient data synchronization.

```java
public class MerkleTree {
    private Node root;
    
    public static class Node {
        String hash;
        Node left, right;
        boolean isLeaf;
        
        public Node(String hash) {
            this.hash = hash;
            this.isLeaf = true;
        }
    }
    
    public MerkleTree(List<String> dataBlocks) {
        if (dataBlocks.isEmpty()) return;
        
        List<Node> leaves = dataBlocks.stream()
            .map(data -> new Node(sha256(data)))
            .collect(Collectors.toList());
            
        this.root = buildTree(leaves);
    }
    
    private Node buildTree(List<Node> nodes) {
        if (nodes.size() == 1) return nodes.get(0);
        
        List<Node> parents = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i += 2) {
            Node left = nodes.get(i);
            Node right = (i + 1 < nodes.size()) ? nodes.get(i + 1) : left;
            
            Node parent = new Node(sha256(left.hash + right.hash));
            parent.left = left;
            parent.right = right;
            parent.isLeaf = false;
            parents.add(parent);
        }
        return buildTree(parents);
    }
    
    // Find differing subtrees between two Merkle trees
    public List<String> findDifferences(MerkleTree other) {
        List<String> differences = new ArrayList<>();
        findDifferences(this.root, other.root, differences);
        return differences;
    }
    
    private void findDifferences(Node node1, Node node2, List<String> differences) {
        if (node1 == null || node2 == null) return;
        
        if (!node1.hash.equals(node2.hash)) {
            if (node1.isLeaf) {
                differences.add(node1.hash);
            } else {
                findDifferences(node1.left, node2.left, differences);
                findDifferences(node1.right, node2.right, differences);
            }
        }
    }
}
```

### 9.4 Distributed Hash Tables (DHT)

DHTs provide a lookup service similar to a hash table: (key, value) pairs are stored across distributed nodes.

**Chord Algorithm** example:

```java
public class ChordNode {
    private final String nodeId;
    private final int m; // Key space size = 2^m
    private ChordNode successor;
    private ChordNode predecessor;
    private ChordNode[] fingerTable; // For O(log N) lookups
    private final Map<String, String> dataStore = new ConcurrentHashMap<>();
    
    public ChordNode(String nodeId, int m) {
        this.nodeId = nodeId;
        this.m = m;
        this.fingerTable = new ChordNode[m];
    }
    
    // Join existing Chord ring
    public void join(ChordNode existingNode) {
        if (existingNode != null) {
            predecessor = null;
            successor = existingNode.findSuccessor(hash(nodeId));
        } else {
            // First node in ring
            successor = this;
            predecessor = this;
        }
    }
    
    // Find successor of given key
    public ChordNode findSuccessor(int key) {
        ChordNode node = findPredecessor(key);
        return node.successor;
    }
    
    private ChordNode findPredecessor(int key) {
        ChordNode node = this;
        while (!inRange(key, hash(node.nodeId), hash(node.successor.nodeId), true)) {
            node = node.closestPrecedingFinger(key);
        }
        return node;
    }
    
    // Store key-value pair
    public void put(String key, String value) {
        int keyHash = hash(key);
        ChordNode responsible = findSuccessor(keyHash);
        responsible.dataStore.put(key, value);
    }
    
    // Retrieve value for key
    public String get(String key) {
        int keyHash = hash(key);
        ChordNode responsible = findSuccessor(keyHash);
        return responsible.dataStore.get(key);
    }
    
    private boolean inRange(int key, int start, int end, boolean rightInclusive) {
        if (start < end) {
            return rightInclusive ? (key > start && key <= end) : (key > start && key < end);
        } else {
            return rightInclusive ? (key > start || key <= end) : (key > start || key < end);
        }
    }
    
    private int hash(String input) {
        return Math.abs(input.hashCode()) % (1 << m);
    }
}
```

### 9.5 Conflict-Free Replicated Data Types (CRDTs)

Data structures that automatically resolve conflicts in distributed systems.

**G-Counter (Grow-only Counter)**:

```java
public class GCounter {
    private final String nodeId;
    private final Map<String, Integer> counters = new ConcurrentHashMap<>();
    
    public GCounter(String nodeId, Set<String> allNodes) {
        this.nodeId = nodeId;
        for (String node : allNodes) {
            counters.put(node, 0);
        }
    }
    
    // Increment local counter
    public void increment() {
        counters.put(nodeId, counters.get(nodeId) + 1);
    }
    
    // Get total value
    public int getValue() {
        return counters.values().stream().mapToInt(Integer::intValue).sum();
    }
    
    // Merge with another G-Counter (for replication)
    public GCounter merge(GCounter other) {
        GCounter result = new GCounter(this.nodeId, counters.keySet());
        for (String node : counters.keySet()) {
            int thisValue = this.counters.get(node);
            int otherValue = other.counters.getOrDefault(node, 0);
            result.counters.put(node, Math.max(thisValue, otherValue));
        }
        return result;
    }
}
```

**LWW-Register (Last-Write-Wins Register)**:

```java
public class LWWRegister<T> {
    private T value;
    private long timestamp;
    private String nodeId;
    
    public LWWRegister(String nodeId) {
        this.nodeId = nodeId;
        this.timestamp = 0;
    }
    
    public void set(T newValue) {
        this.value = newValue;
        this.timestamp = System.currentTimeMillis();
    }
    
    public T get() {
        return value;
    }
    
    // Merge with another register
    public LWWRegister<T> merge(LWWRegister<T> other) {
        if (other.timestamp > this.timestamp || 
            (other.timestamp == this.timestamp && other.nodeId.compareTo(this.nodeId) > 0)) {
            this.value = other.value;
            this.timestamp = other.timestamp;
        }
        return this;
    }
}
```

---

## 10. Practical Examples

### 10.1 Distributed Lock (Redis with Redisson)

```java
import org.redisson.Redisson;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;
import java.util.concurrent.TimeUnit;

@Component
public class DistributedLockService {
    private final RedissonClient redissonClient;
    
    public DistributedLockService() {
        Config config = new Config();
        config.useSingleServer().setAddress("redis://localhost:6379");
        this.redissonClient = Redisson.create(config);
    }
    
    public void executeWithLock(String lockName, int timeoutSeconds, Runnable task) {
        RLock lock = redissonClient.getLock("lock:" + lockName);
        try {
            if (lock.tryLock(timeoutSeconds, TimeUnit.SECONDS)) {
                try {
                    task.run();
                } finally {
                    if (lock.isHeldByCurrentThread()) {
                        lock.unlock();
                    }
                }
            } else {
                throw new RuntimeException("Could not acquire lock: " + lockName);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while acquiring lock", e);
        }
    }
    
    // Usage
    public void processPayment(String userId) {
        executeWithLock("payment:" + userId, 10, () -> {
            // Critical section - payment processing
            paymentService.process(userId);
        });
    }
}

// Alternative: Manual Redis implementation
@Component
public class ManualDistributedLock {
    private final StringRedisTemplate redisTemplate;
    
    public boolean acquireLock(String key, String value, Duration timeout) {
        return Boolean.TRUE.equals(
            redisTemplate.opsForValue().setIfAbsent(key, value, timeout)
        );
    }
    
    public void releaseLock(String key, String value) {
        // Use Lua script for atomic check-and-delete
        String script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
        """;
        redisTemplate.execute(new DefaultRedisScript<>(script, Long.class), 
            Collections.singletonList(key), value);
    }
}
```

### 10.2 Idempotent API with Idempotency Key (Spring Boot)

```java
@RestController
@RequestMapping("/api/payments")
public class PaymentController {
    private final PaymentService paymentService;
    private final RedisTemplate<String, Object> redisTemplate;
    private final ObjectMapper objectMapper;
    
    @PostMapping
    public ResponseEntity<?> createPayment(
            @RequestBody PaymentRequest request,
            @RequestHeader("Idempotency-Key") String idempotencyKey) {
        
        String cacheKey = "idempotency:" + idempotencyKey;
        
        // Check for cached response
        String cached = (String) redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            try {
                PaymentResponse response = objectMapper.readValue(cached, PaymentResponse.class);
                return ResponseEntity.ok(response);
            } catch (Exception e) {
                log.warn("Failed to deserialize cached response", e);
            }
        }
        
        try {
            // Process payment
            PaymentResponse result = paymentService.createPayment(request);
            
            // Cache response for 24 hours
            String resultJson = objectMapper.writeValueAsString(result);
            redisTemplate.opsForValue().set(cacheKey, resultJson, Duration.ofDays(1));
            
            return ResponseEntity.ok(result);
            
        } catch (Exception e) {
            log.error("Payment processing failed", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Map.of("error", "Payment processing failed"));
        }
    }
}

// Annotation-based idempotency
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Idempotent {
    String keyExpression() default "";
    int ttlHours() default 24;
}

@Aspect
@Component
public class IdempotencyAspect {
    private final RedisTemplate<String, Object> redisTemplate;
    private final ObjectMapper objectMapper;
    
    @Around("@annotation(idempotent)")
    public Object handleIdempotency(ProceedingJoinPoint joinPoint, Idempotent idempotent) throws Throwable {
        String idempotencyKey = extractIdempotencyKey(joinPoint);
        String cacheKey = "idempotency:" + idempotencyKey;
        
        // Check cache
        String cached = (String) redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            return objectMapper.readValue(cached, Object.class);
        }
        
        // Execute method
        Object result = joinPoint.proceed();
        
        // Cache result
        String resultJson = objectMapper.writeValueAsString(result);
        redisTemplate.opsForValue().set(cacheKey, resultJson, 
            Duration.ofHours(idempotent.ttlHours()));
            
        return result;
    }
}
```

### 10.3 Quorum-Based Replication System

```java
public class QuorumReplicatedStore {
    private final List<Node> nodes;
    private final int replicationFactor;
    private final ExecutorService executor;
    
    public QuorumReplicatedStore(List<Node> nodes, int replicationFactor) {
        this.nodes = nodes;
        this.replicationFactor = replicationFactor;
        this.executor = Executors.newFixedThreadPool(nodes.size());
    }
    
    // Quorum write: W replicas must acknowledge
    public boolean put(String key, String value, int writeQuorum) {
        List<Node> targetNodes = selectNodes(key, replicationFactor);
        List<CompletableFuture<Boolean>> futures = new ArrayList<>();
        
        for (Node node : targetNodes) {
            CompletableFuture<Boolean> future = CompletableFuture.supplyAsync(() -> {
                try {
                    return node.write(key, value);
                } catch (Exception e) {
                    log.warn("Write failed to node {}: {}", node.getId(), e.getMessage());
                    return false;
                }
            }, executor);
            futures.add(future);
        }
        
        // Count successful writes
        int successCount = 0;
        for (CompletableFuture<Boolean> future : futures) {
            try {
                if (future.get(5, TimeUnit.SECONDS)) {
                    successCount++;
                    if (successCount >= writeQuorum) {
                        return true; // Achieved quorum
                    }
                }
            } catch (Exception e) {
                log.warn("Write future failed", e);
            }
        }
        
        return false; // Failed to achieve write quorum
    }
    
    // Quorum read: R replicas must respond
    public Optional<VersionedValue> get(String key, int readQuorum) {
        List<Node> targetNodes = selectNodes(key, replicationFactor);
        List<CompletableFuture<VersionedValue>> futures = new ArrayList<>();
        
        for (Node node : targetNodes) {
            CompletableFuture<VersionedValue> future = CompletableFuture.supplyAsync(() -> {
                try {
                    return node.read(key);
                } catch (Exception e) {
                    log.warn("Read failed from node {}: {}", node.getId(), e.getMessage());
                    return null;
                }
            }, executor);
            futures.add(future);
        }
        
        // Collect responses and find latest version
        List<VersionedValue> responses = new ArrayList<>();
        for (CompletableFuture<VersionedValue> future : futures) {
            try {
                VersionedValue result = future.get(5, TimeUnit.SECONDS);
                if (result != null) {
                    responses.add(result);
                    if (responses.size() >= readQuorum) {
                        break; // Achieved read quorum
                    }
                }
            } catch (Exception e) {
                log.warn("Read future failed", e);
            }
        }
        
        if (responses.size() < readQuorum) {
            return Optional.empty(); // Failed to achieve read quorum
        }
        
        // Return value with highest version
        return responses.stream()
            .max(Comparator.comparing(VersionedValue::getVersion));
    }
    
    private List<Node> selectNodes(String key, int count) {
        int hash = Math.abs(key.hashCode());
        int startIndex = hash % nodes.size();
        
        List<Node> selected = new ArrayList<>();
        for (int i = 0; i < count && i < nodes.size(); i++) {
            selected.add(nodes.get((startIndex + i) % nodes.size()));
        }
        return selected;
    }
    
    public static class VersionedValue {
        private final String value;
        private final long version;
        private final long timestamp;
        
        public VersionedValue(String value, long version, long timestamp) {
            this.value = value;
            this.version = version;
            this.timestamp = timestamp;
        }
        
        // getters...
    }
}
```

### 10.4 Raft Implementation (Simplified)

```java
public class RaftNode {
    private final String nodeId;
    private final List<String> peers;
    private volatile State state = State.FOLLOWER;
    private volatile int currentTerm = 0;
    private volatile String votedFor = null;
    private final List<LogEntry> log = new ArrayList<>();
    private volatile int commitIndex = 0;
    
    // Leader state
    private final Map<String, Integer> nextIndex = new ConcurrentHashMap<>();
    private final Map<String, Integer> matchIndex = new ConcurrentHashMap<>();
    
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
    private ScheduledFuture<?> electionTimer;
    private ScheduledFuture<?> heartbeatTimer;
    
    public enum State { FOLLOWER, CANDIDATE, LEADER }
    
    public void start() {
        resetElectionTimer();
    }
    
    // Request Vote RPC
    public RequestVoteResponse requestVote(RequestVoteRequest request) {
        synchronized (this) {
            if (request.getTerm() > currentTerm) {
                currentTerm = request.getTerm();
                votedFor = null;
                state = State.FOLLOWER;
            }
            
            boolean voteGranted = false;
            if (request.getTerm() == currentTerm &&
                (votedFor == null || votedFor.equals(request.getCandidateId())) &&
                isLogUpToDate(request.getLastLogIndex(), request.getLastLogTerm())) {
                
                votedFor = request.getCandidateId();
                voteGranted = true;
                resetElectionTimer();
            }
            
            return new RequestVoteResponse(currentTerm, voteGranted);
        }
    }
    
    // Append Entries RPC (heartbeat + log replication)
    public AppendEntriesResponse appendEntries(AppendEntriesRequest request) {
        synchronized (this) {
            if (request.getTerm() > currentTerm) {
                currentTerm = request.getTerm();
                votedFor = null;
                state = State.FOLLOWER;
            }
            
            if (request.getTerm() < currentTerm) {
                return new AppendEntriesResponse(currentTerm, false);
            }
            
            resetElectionTimer();
            
            // Check log consistency
            if (request.getPrevLogIndex() > 0) {
                if (log.size() < request.getPrevLogIndex() ||
                    log.get(request.getPrevLogIndex() - 1).getTerm() != request.getPrevLogTerm()) {
                    return new AppendEntriesResponse(currentTerm, false);
                }
            }
            
            // Append new entries
            if (!request.getEntries().isEmpty()) {
                // Remove conflicting entries
                if (log.size() > request.getPrevLogIndex()) {
                    log.subList(request.getPrevLogIndex(), log.size()).clear();
                }
                
                // Append new entries
                log.addAll(request.getEntries());
            }
            
            // Update commit index
            if (request.getLeaderCommit() > commitIndex) {
                commitIndex = Math.min(request.getLeaderCommit(), log.size());
            }
            
            return new AppendEntriesResponse(currentTerm, true);
        }
    }
    
    private void startElection() {
        synchronized (this) {
            state = State.CANDIDATE;
            currentTerm++;
            votedFor = nodeId;
            resetElectionTimer();
            
            int lastLogIndex = log.size();
            int lastLogTerm = lastLogIndex > 0 ? log.get(lastLogIndex - 1).getTerm() : 0;
            
            AtomicInteger votes = new AtomicInteger(1); // Vote for self
            
            for (String peer : peers) {
                CompletableFuture.supplyAsync(() -> {
                    RequestVoteRequest request = new RequestVoteRequest(
                        currentTerm, nodeId, lastLogIndex, lastLogTerm);
                    return sendRequestVote(peer, request);
                }).thenAccept(response -> {
                    synchronized (this) {
                        if (response.getTerm() > currentTerm) {
                            currentTerm = response.getTerm();
                            state = State.FOLLOWER;
                            votedFor = null;
                        } else if (state == State.CANDIDATE && 
                                 response.getTerm() == currentTerm && 
                                 response.isVoteGranted()) {
                            
                            if (votes.incrementAndGet() > peers.size() / 2) {
                                becomeLeader();
                            }
                        }
                    }
                });
            }
        }
    }
    
    private void becomeLeader() {
        state = State.LEADER;
        log.info("Node {} became leader for term {}", nodeId, currentTerm);
        
        // Initialize leader state
        for (String peer : peers) {
            nextIndex.put(peer, log.size() + 1);
            matchIndex.put(peer, 0);
        }
        
        // Start sending heartbeats
        startHeartbeat();
    }
    
    private void startHeartbeat() {
        heartbeatTimer = scheduler.scheduleAtFixedRate(() -> {
            if (state == State.LEADER) {
                for (String peer : peers) {
                    sendHeartbeat(peer);
                }
            }
        }, 0, 150, TimeUnit.MILLISECONDS);
    }
    
    private void resetElectionTimer() {
        if (electionTimer != null) {
            electionTimer.cancel(false);
        }
        
        // Random timeout between 150-300ms
        int timeout = 150 + new Random().nextInt(150);
        electionTimer = scheduler.schedule(() -> {
            if (state != State.LEADER) {
                startElection();
            }
        }, timeout, TimeUnit.MILLISECONDS);
    }
}
```

### 10.5 Distributed Transaction Coordinator (2PC)

```java
@Component
public class TwoPhaseCommitCoordinator {
    private final Map<String, TransactionParticipant> participants;
    private final TransactionLogService logService;
    
    public TransactionResult executeTransaction(String transactionId, 
                                             List<TransactionCommand> commands) {
        TransactionContext context = new TransactionContext(transactionId, commands);
        
        try {
            // Phase 1: Prepare
            PrepareResult prepareResult = preparePhase(context);
            if (!prepareResult.isSuccess()) {
                abort(context);
                return TransactionResult.aborted(prepareResult.getFailureReason());
            }
            
            // Phase 2: Commit
            CommitResult commitResult = commitPhase(context);
            if (commitResult.isSuccess()) {
                return TransactionResult.committed();
            } else {
                // Partial commit - requires recovery
                return TransactionResult.partialCommit(commitResult.getFailedParticipants());
            }
            
        } catch (Exception e) {
            log.error("Transaction {} failed", transactionId, e);
            abort(context);
            return TransactionResult.aborted(e.getMessage());
        }
    }
    
    private PrepareResult preparePhase(TransactionContext context) {
        logService.logTransactionStart(context.getTransactionId());
        
        List<CompletableFuture<VoteResponse>> prepareVotes = new ArrayList<>();
        
        for (TransactionCommand command : context.getCommands()) {
            TransactionParticipant participant = participants.get(command.getParticipantId());
            
            CompletableFuture<VoteResponse> vote = CompletableFuture.supplyAsync(() -> {
                try {
                    return participant.prepare(context.getTransactionId(), command);
                } catch (Exception e) {
                    return VoteResponse.abort(e.getMessage());
                }
            });
            
            prepareVotes.add(vote);
        }
        
        // Wait for all votes
        List<VoteResponse> votes = prepareVotes.stream()
            .map(future -> {
                try {
                    return future.get(30, TimeUnit.SECONDS);
                } catch (Exception e) {
                    return VoteResponse.abort("Timeout or error: " + e.getMessage());
                }
            })
            .collect(Collectors.toList());
        
        // Check if all voted to commit
        for (VoteResponse vote : votes) {
            if (vote.getVote() != Vote.COMMIT) {
                return PrepareResult.failure(vote.getReason());
            }
        }
        
        logService.logPrepareSuccess(context.getTransactionId());
        return PrepareResult.success();
    }
    
    private CommitResult commitPhase(TransactionContext context) {
        logService.logCommitPhaseStart(context.getTransactionId());
        
        List<CompletableFuture<Void>> commitFutures = new ArrayList<>();
        Set<String> failedParticipants = ConcurrentHashMap.newKeySet();
        
        for (TransactionCommand command : context.getCommands()) {
            TransactionParticipant participant = participants.get(command.getParticipantId());
            
            CompletableFuture<Void> commitFuture = CompletableFuture.runAsync(() -> {
                try {
                    participant.commit(context.getTransactionId());
                } catch (Exception e) {
                    log.error("Commit failed for participant {}", 
                        command.getParticipantId(), e);
                    failedParticipants.add(command.getParticipantId());
                }
            });
            
            commitFutures.add(commitFuture);
        }
        
        // Wait for all commits to complete (or fail)
        CompletableFuture.allOf(commitFutures.toArray(new CompletableFuture[0]))
            .join();
        
        if (failedParticipants.isEmpty()) {
            logService.logTransactionCommitted(context.getTransactionId());
            return CommitResult.success();
        } else {
            logService.logPartialCommit(context.getTransactionId(), failedParticipants);
            return CommitResult.partial(failedParticipants);
        }
    }
    
    private void abort(TransactionContext context) {
        logService.logTransactionAbort(context.getTransactionId());
        
        for (TransactionCommand command : context.getCommands()) {
            try {
                TransactionParticipant participant = participants.get(command.getParticipantId());
                participant.abort(context.getTransactionId());
            } catch (Exception e) {
                log.error("Abort failed for participant {}", 
                    command.getParticipantId(), e);
            }
        }
    }
}
```

---

## Summary

| Concept | Key Takeaway |
|---------|---------------|
| **CAP** | Choose CP or AP; P is non-negotiable |
| **PACELC** | No partition: trade latency vs consistency |
| **Consistency** | Strong → Eventual: choose based on use case |
| **Consensus** | Raft for crash faults; BFT for Byzantine |
| **Transactions** | 2PC blocks; Saga with compensations for microservices |
| **Replication** | Single-leader (simpler), Leaderless (availability) |
| **Fault tolerance** | Timeouts, retries, circuit breaker, bulkhead |

---

## Further Reading

- *Designing Data-Intensive Applications* — Martin Kleppmann
- *Introduction to Reliable and Secure Distributed Programming* — Cachin et al.
- Raft paper: https://raft.github.io/
- etcd documentation: https://etcd.io/
