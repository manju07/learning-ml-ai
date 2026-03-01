# Event-Driven Architecture: Complete Guide for Architects

## Table of Contents
1. [Introduction](#1-introduction)
2. [Event-Driven vs Request-Response](#2-event-driven-vs-request-response)
3. [Event Sourcing](#3-event-sourcing)
4. [CQRS (Command Query Responsibility Segregation)](#4-cqrs-command-query-responsibility-segregation)
5. [Message Brokers: Kafka vs RabbitMQ](#5-message-brokers-kafka-vs-rabbitmq)
6. [Event Design and Schema Evolution](#6-event-design-and-schema-evolution)
7. [Saga Pattern Revisited](#7-saga-pattern-revisited)
8. [Outbox and Transactional Outbox](#8-outbox-and-transactional-outbox)
9. [Stream Processing](#9-stream-processing)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

**Event-Driven Architecture (EDA)** uses events to communicate between services. An **event** is a record that something happened in the past (immutable).

### Core Concepts

| Term | Definition |
|------|-------------|
| **Event** | Something that happened (e.g., OrderCreated) |
| **Producer** | Emits events |
| **Consumer** | Reacts to events |
| **Event Store** | Persists events (for event sourcing) |
| **Message Broker** | Routes events to consumers (Kafka, RabbitMQ) |

### Benefits

- **Loose coupling**: Producers don't know consumers
- **Scalability**: Scale consumers independently
- **Resilience**: Message broker buffers; retry on failure
- **Audit trail**: Events are append-only log
- **Temporal decoupling**: Produce and consume at different times

---

## 2. Event-Driven vs Request-Response

| Aspect | Request-Response | Event-Driven |
|--------|------------------|--------------|
| Coupling | Tight (caller knows callee) | Loose (anonymous) |
| Latency | Blocking | Async |
| Failure | Caller fails with callee | Retry, DLQ |
| Scaling | Vertical, or more instances | Horizontal, per consumer |
| Use case | Need immediate response | Fire-and-forget, async |

### When to Use Events

- Multiple consumers for one action (e.g., order created → inventory, payment, notification)
- Decoupled workflows (order → shipping → warehouse)
- Audit/compliance (append-only event log)
- Real-time analytics

---

## 3. Event Sourcing

### 3.1 Concept

Store **events** instead of current state. State is derived by replaying events.

```
Traditional:  orders table with current state
Event Sourcing: event_log with OrderCreated, ItemAdded, OrderPaid, OrderShipped
```

### 3.2 Event Stream Example

```
Order #123 events:
1. OrderCreated { order_id: 123, user_id: 1, created_at: ... }
2. ItemAdded { order_id: 123, product_id: 5, quantity: 2 }
3. ItemAdded { order_id: 123, product_id: 8, quantity: 1 }
4. OrderPaid { order_id: 123, amount: 99.99, payment_id: "pay_xyz" }
5. OrderShipped { order_id: 123, tracking: "1Z999..." }
```

Current state = replay 1..5.

### 3.3 Snapshots

Replaying many events is slow. Periodically store **snapshots**; replay from latest snapshot.

```
Snapshot at event 1000: { order_id: 123, status: "paid", total: 99.99 }
Replay events 1001..1050 to get current state
```

### 3.4 Implementation Sketch

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import json

@dataclass
class Event:
    aggregate_id: str
    event_type: str
    payload: dict
    version: int
    timestamp: datetime

class EventSourcedAggregate:
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.events: list[Event] = []
        self._version = 0

    def apply(self, event: Event):
        self.events.append(event)
        self._version = event.version
        # Apply event to in-memory state
        handler = getattr(self, f"_apply_{event.event_type}", None)
        if handler:
            handler(event.payload)

    def replay(self, events: list[Event]):
        for e in events:
            self.apply(e)

# Order aggregate
class Order(EventSourcedAggregate):
    def __init__(self, aggregate_id: str):
        super().__init__(aggregate_id)
        self.status = "draft"
        self.items = []
        self.total = 0

    def _apply_OrderCreated(self, payload):
        self.status = "created"

    def _apply_ItemAdded(self, payload):
        self.items.append(payload)
        self.total += payload.get("price", 0) * payload.get("quantity", 1)

    def _apply_OrderPaid(self, payload):
        self.status = "paid"
```

---

## 4. CQRS (Command Query Responsibility Segregation)

### 4.1 Principle

Separate **command** (write) model from **query** (read) model. Often combined with event sourcing.

```
Commands (write)          Events           Queries (read)
CreateOrder  ──> [Write Model] ──> OrderCreated ──> [Read Model] ──> GetOrder
                                 OrderPaid              GetOrderList
                                 OrderShipped            GetOrderHistory
```

### 4.2 Why CQRS?

- **Different scale**: Reads often >> writes; scale read model independently
- **Different shape**: Optimize read model for UI (denormalized, views)
- **Different technology**: Write to PostgreSQL; read from Elasticsearch, Redis

### 4.3 Read Model Updates

Read models are **projections** built from events.

```
Event: OrderCreated
  -> Update: order_list_view, order_detail_view, user_orders_view
  -> Update: search index (Elasticsearch)
  -> Update: cache (Redis)
```

### 4.4 Eventually Consistent

Read model lags behind write model. Use version or timestamp to detect staleness.

---

## 5. Message Brokers: Kafka vs RabbitMQ

### 5.1 Comparison

| Aspect | Kafka | RabbitMQ |
|--------|-------|----------|
| **Model** | Log, partition-based | Queue, exchange-based |
| **Ordering** | Per partition | Per queue |
| **Retention** | Configurable (days/weeks) | Deleted after consume |
| **Replay** | Yes (consumer groups) | No |
| **Use case** | Event streaming, analytics | Task queues, RPC |
| **Throughput** | Very high | High |

### 5.2 Kafka Concepts

- **Topic**: Named stream of records
- **Partition**: Ordered, immutable sequence within topic
- **Producer**: Publishes to topic (optionally by key for partitioning)
- **Consumer Group**: Each partition consumed by one consumer in group
- **Offset**: Position in partition

```
Topic: order-events (3 partitions)
Partition 0: [msg0, msg3, msg6, ...]
Partition 1: [msg1, msg4, msg7, ...]
Partition 2: [msg2, msg5, msg8, ...]

Consumer Group A: C1->P0, C2->P1, C3->P2
```

### 5.3 Kafka Example

```python
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import asyncio
import json

async def produce():
    producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')
    await producer.start()
    try:
        await producer.send_and_wait(
            'order-events',
            json.dumps({"type": "OrderCreated", "order_id": "123", "amount": 99.99}).encode(),
            key=b"123"  # Same key -> same partition -> order preserved
        )
    finally:
        await producer.stop()

async def consume():
    consumer = AIOKafkaConsumer(
        'order-events',
        bootstrap_servers='localhost:9092',
        group_id='order-processors'
    )
    await consumer.start()
    try:
        async for msg in consumer:
            event = json.loads(msg.value)
            print(f"Processing: {event}")
            # Process event, update read model
    finally:
        await consumer.stop()
```

### 5.4 RabbitMQ Example

```python
import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare exchange and queue
channel.exchange_declare(exchange='order-events', exchange_type='topic')
channel.queue_declare(queue='payment-service')
channel.queue_bind(exchange='order-events', queue='payment-service', routing_key='order.created')

def callback(ch, method, properties, body):
    event = json.loads(body)
    process_payment(event)
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='payment-service', on_message_callback=callback)
channel.start_consuming()
```

---

## 6. Event Design and Schema Evolution

### 6.1 Event Naming

- **Past tense**: OrderCreated, PaymentFailed (something happened)
- **Domain language**: Use ubiquitous language from DDD

### 6.2 Event Schema

```json
{
  "event_id": "evt_abc123",
  "event_type": "OrderCreated",
  "aggregate_id": "ord_123",
  "aggregate_type": "Order",
  "version": 1,
  "timestamp": "2024-01-15T10:30:00Z",
  "payload": {
    "order_id": "ord_123",
    "user_id": "usr_456",
    "items": [...],
    "total": 99.99
  },
  "metadata": {
    "source": "order-service",
    "correlation_id": "corr_xyz"
  }
}
```

### 6.3 Schema Evolution

- **Add optional fields**: Backward compatible
- **Remove fields**: Consumers must ignore unknown
- **Rename**: Use new event type (OrderCreatedV2)
- **Schema registry**: Avro, Protobuf with Confluent Schema Registry

---

## 7. Saga Pattern Revisited

### 7.1 Choreography (Event-Based)

Each service reacts to events and emits its own. No central coordinator.

```
OrderService: OrderCreated event
  -> InventoryService: ReserveStock (emit StockReserved or StockFailed)
  -> PaymentService: ChargePayment (emit PaymentCompleted or PaymentFailed)
  -> If PaymentFailed: InventoryService: ReleaseStock (compensating)
```

### 7.2 Orchestration (Central Coordinator)

Orchestrator sends commands to services and runs compensations on failure.

```
SagaOrchestrator:
  1. ReserveStock -> Success
  2. ChargePayment -> Failure
  3. Compensate: ReleaseStock
```

### 7.3 Choosing

| Choreography | Orchestration |
|--------------|---------------|
| Simpler, no single point of failure | Easier to understand, centralized logic |
| Hard to see full flow | Clear saga flow |
| Order Service doesn't know Payment | Orchestrator knows all steps |

---

## 8. Outbox and Transactional Outbox

### 8.1 Problem

Publishing to Kafka and updating DB in separate operations: if one fails, inconsistency.

### 8.2 Transactional Outbox Pattern

1. In **same DB transaction**: insert business row + insert outbox row
2. Background process reads outbox, publishes to broker, marks as published
3. Guarantees: exactly-once publishing (with dedup on consumer side) or at-least-once

```sql
BEGIN;
INSERT INTO orders (id, user_id, total, status) VALUES (...);
INSERT INTO outbox (id, aggregate_id, event_type, payload, created_at)
VALUES (gen_random_uuid(), 'ord_123', 'OrderCreated', '{"order_id":"ord_123",...}', NOW());
COMMIT;
```

```python
# Polling publisher
def publish_outbox():
    rows = db.execute("""
        SELECT * FROM outbox WHERE published_at IS NULL LIMIT 100
        FOR UPDATE SKIP LOCKED
    """).fetchall()
    for row in rows:
        kafka_producer.send('order-events', row.payload, key=row.aggregate_id)
        db.execute("UPDATE outbox SET published_at = NOW() WHERE id = %s", (row.id,))
```

### 8.3 Debezium / CDC

Change Data Capture: read DB write-ahead log, publish changes as events. Alternative to polling outbox.

---

## 9. Stream Processing

### 9.1 Concepts

- **Stream**: Unbounded sequence of events
- **Window**: Tumbling, sliding, session
- **Aggregation**: Count, sum, join

### 9.2 Kafka Streams Example

```python
from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
from datetime import datetime, timedelta

# Simplified: aggregate orders per user in 5-min windows
def process_stream():
    consumer = KafkaConsumer('order-events', bootstrap_servers='localhost:9092')
    window = defaultdict(lambda: {"count": 0, "total": 0})
    window_start = datetime.utcnow().replace(second=0, microsecond=0)

    for msg in consumer:
        event = json.loads(msg.value)
        now = datetime.utcnow()
        if now - window_start > timedelta(minutes=5):
            # Flush window
            emit_aggregates(window)
            window.clear()
            window_start = now.replace(second=0, microsecond=0)

        user_id = event.get("user_id")
        window[user_id]["count"] += 1
        window[user_id]["total"] += event.get("total", 0)
```

### 9.3 Frameworks

- **Kafka Streams**: Java, lightweight, exactly-once
- **Flink**: Apache, complex event processing
- **ksqlDB**: SQL on Kafka
- **Flink SQL**: Similar idea

---

## 10. Practical Examples

### 10.1 Event Sourcing with Spring Boot

```java
// Domain Event Base Class
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "eventType")
@JsonSubTypes({
    @JsonSubTypes.Type(value = OrderCreated.class, name = "OrderCreated"),
    @JsonSubTypes.Type(value = OrderUpdated.class, name = "OrderUpdated"),
    @JsonSubTypes.Type(value = OrderCancelled.class, name = "OrderCancelled")
})
public abstract class DomainEvent {
    private final String eventId;
    private final String aggregateId;
    private final long version;
    private final Instant timestamp;
    
    protected DomainEvent(String aggregateId, long version) {
        this.eventId = UUID.randomUUID().toString();
        this.aggregateId = aggregateId;
        this.version = version;
        this.timestamp = Instant.now();
    }
    
    // getters...
}

// Specific Domain Events
@Data
@EqualsAndHashCode(callSuper = true)
public class OrderCreated extends DomainEvent {
    private final String userId;
    private final List<OrderItem> items;
    private final BigDecimal total;
    
    public OrderCreated(String aggregateId, long version, String userId, 
                       List<OrderItem> items, BigDecimal total) {
        super(aggregateId, version);
        this.userId = userId;
        this.items = items;
        this.total = total;
    }
}

// Aggregate Root with Event Sourcing
@Entity
public class Order extends AggregateRoot {
    
    @Id
    private String id;
    private String userId;
    private OrderStatus status;
    private List<OrderItem> items = new ArrayList<>();
    private BigDecimal total;
    private long version;
    
    // Factory method for creating new orders
    public static Order createOrder(String userId, List<OrderItem> items) {
        Order order = new Order();
        order.id = UUID.randomUUID().toString();
        
        BigDecimal total = items.stream()
            .map(item -> item.getPrice().multiply(BigDecimal.valueOf(item.getQuantity())))
            .reduce(BigDecimal.ZERO, BigDecimal::add);
            
        order.raiseEvent(new OrderCreated(order.id, 0, userId, items, total));
        return order;
    }
    
    // Event application methods
    @EventHandler
    public void on(OrderCreated event) {
        this.id = event.getAggregateId();
        this.userId = event.getUserId();
        this.items = new ArrayList<>(event.getItems());
        this.total = event.getTotal();
        this.status = OrderStatus.PENDING;
        this.version = event.getVersion();
    }
    
    @EventHandler
    public void on(OrderUpdated event) {
        this.items = new ArrayList<>(event.getItems());
        this.total = event.getNewTotal();
        this.version = event.getVersion();
    }
    
    public void updateItems(List<OrderItem> newItems) {
        if (this.status != OrderStatus.PENDING) {
            throw new IllegalStateException("Cannot update confirmed order");
        }
        
        BigDecimal newTotal = newItems.stream()
            .map(item -> item.getPrice().multiply(BigDecimal.valueOf(item.getQuantity())))
            .reduce(BigDecimal.ZERO, BigDecimal::add);
            
        raiseEvent(new OrderUpdated(this.id, this.version + 1, newItems, newTotal));
    }
    
    public void cancel(String reason) {
        if (this.status == OrderStatus.CANCELLED) {
            return; // Idempotent
        }
        
        raiseEvent(new OrderCancelled(this.id, this.version + 1, reason));
    }
    
    @EventHandler
    public void on(OrderCancelled event) {
        this.status = OrderStatus.CANCELLED;
        this.version = event.getVersion();
    }
}

// Abstract Aggregate Root
public abstract class AggregateRoot {
    
    @Transient
    private final List<DomainEvent> domainEvents = new ArrayList<>();
    
    protected void raiseEvent(DomainEvent event) {
        domainEvents.add(event);
    }
    
    public List<DomainEvent> getDomainEvents() {
        return Collections.unmodifiableList(domainEvents);
    }
    
    public void clearDomainEvents() {
        domainEvents.clear();
    }
}
```

### 10.2 Event Store Implementation

```java
@Repository
@Transactional
public class EventStore {
    
    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper;
    
    public void saveEvents(String aggregateId, List<DomainEvent> events, long expectedVersion) {
        // Optimistic concurrency control
        Long currentVersion = getCurrentVersion(aggregateId);
        if (currentVersion != null && !currentVersion.equals(expectedVersion)) {
            throw new OptimisticLockingException(
                String.format("Expected version %d but was %d", expectedVersion, currentVersion));
        }
        
        String sql = """
            INSERT INTO event_store (event_id, aggregate_id, event_type, event_data, version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """;
        
        List<Object[]> batch = events.stream()
            .map(event -> new Object[] {
                event.getEventId(),
                event.getAggregateId(),
                event.getClass().getSimpleName(),
                serializeEvent(event),
                event.getVersion(),
                Timestamp.from(event.getTimestamp())
            })
            .collect(Collectors.toList());
            
        jdbcTemplate.batchUpdate(sql, batch);
    }
    
    public List<DomainEvent> getEvents(String aggregateId, long fromVersion) {
        String sql = """
            SELECT event_id, event_type, event_data, version, timestamp
            FROM event_store 
            WHERE aggregate_id = ? AND version > ?
            ORDER BY version
        """;
        
        return jdbcTemplate.query(sql, 
            new Object[] { aggregateId, fromVersion },
            (rs, rowNum) -> deserializeEvent(
                rs.getString("event_type"),
                rs.getString("event_data")
            ));
    }
    
    public List<DomainEvent> getAllEvents(String aggregateId) {
        return getEvents(aggregateId, -1);
    }
    
    private Long getCurrentVersion(String aggregateId) {
        String sql = "SELECT MAX(version) FROM event_store WHERE aggregate_id = ?";
        return jdbcTemplate.queryForObject(sql, new Object[] { aggregateId }, Long.class);
    }
    
    private String serializeEvent(DomainEvent event) {
        try {
            return objectMapper.writeValueAsString(event);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize event", e);
        }
    }
    
    private DomainEvent deserializeEvent(String eventType, String eventData) {
        try {
            Class<?> eventClass = Class.forName("com.example.events." + eventType);
            return (DomainEvent) objectMapper.readValue(eventData, eventClass);
        } catch (Exception e) {
            throw new RuntimeException("Failed to deserialize event", e);
        }
    }
}

// Repository that loads aggregates from events
@Repository
public class EventSourcedOrderRepository {
    
    private final EventStore eventStore;
    
    public Optional<Order> findById(String orderId) {
        List<DomainEvent> events = eventStore.getAllEvents(orderId);
        if (events.isEmpty()) {
            return Optional.empty();
        }
        
        Order order = new Order();
        events.forEach(event -> order.apply(event));
        order.clearDomainEvents();
        
        return Optional.of(order);
    }
    
    public void save(Order order) {
        List<DomainEvent> events = order.getDomainEvents();
        if (!events.isEmpty()) {
            long expectedVersion = order.getVersion() - events.size();
            eventStore.saveEvents(order.getId(), events, expectedVersion);
            order.clearDomainEvents();
        }
    }
}
```

### 10.3 CQRS with Spring Boot and Kafka

```java
// Command Side - Order Service
@RestController
@RequestMapping("/api/orders")
public class OrderCommandController {
    
    private final OrderCommandService orderCommandService;
    
    @PostMapping
    public ResponseEntity<CommandResult> createOrder(@RequestBody CreateOrderCommand command) {
        CommandResult result = orderCommandService.handle(command);
        return ResponseEntity.ok(result);
    }
    
    @PutMapping("/{orderId}")
    public ResponseEntity<CommandResult> updateOrder(@PathVariable String orderId,
                                                   @RequestBody UpdateOrderCommand command) {
        command.setOrderId(orderId);
        CommandResult result = orderCommandService.handle(command);
        return ResponseEntity.ok(result);
    }
}

@Service
@Transactional
public class OrderCommandService {
    
    private final EventSourcedOrderRepository orderRepository;
    private final DomainEventPublisher eventPublisher;
    
    public CommandResult handle(CreateOrderCommand command) {
        try {
            Order order = Order.createOrder(command.getUserId(), command.getItems());
            orderRepository.save(order);
            
            // Publish domain events
            order.getDomainEvents().forEach(eventPublisher::publish);
            
            return CommandResult.success(order.getId());
        } catch (Exception e) {
            return CommandResult.failure(e.getMessage());
        }
    }
    
    public CommandResult handle(UpdateOrderCommand command) {
        try {
            Order order = orderRepository.findById(command.getOrderId())
                .orElseThrow(() -> new OrderNotFoundException(command.getOrderId()));
                
            order.updateItems(command.getItems());
            orderRepository.save(order);
            
            order.getDomainEvents().forEach(eventPublisher::publish);
            
            return CommandResult.success(order.getId());
        } catch (Exception e) {
            return CommandResult.failure(e.getMessage());
        }
    }
}

// Event Publishing to Kafka
@Component
@Slf4j
public class KafkaDomainEventPublisher implements DomainEventPublisher {
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${app.kafka.topic.domain-events}")
    private String domainEventsTopic;
    
    @Override
    @Async
    public void publish(DomainEvent event) {
        try {
            String eventData = objectMapper.writeValueAsString(event);
            
            ProducerRecord<String, Object> record = new ProducerRecord<>(
                domainEventsTopic,
                event.getAggregateId(), // Partition key
                eventData
            );
            
            // Add headers
            record.headers().add("eventType", event.getClass().getSimpleName().getBytes());
            record.headers().add("aggregateId", event.getAggregateId().getBytes());
            record.headers().add("version", String.valueOf(event.getVersion()).getBytes());
            record.headers().add("timestamp", event.getTimestamp().toString().getBytes());
            
            kafkaTemplate.send(record)
                .addCallback(
                    result -> log.debug("Domain event published: {} for aggregate {}", 
                        event.getClass().getSimpleName(), event.getAggregateId()),
                    failure -> log.error("Failed to publish domain event: {} for aggregate {}", 
                        event.getClass().getSimpleName(), event.getAggregateId(), failure)
                );
                
        } catch (Exception e) {
            log.error("Error publishing domain event", e);
        }
    }
}

// Query Side - Read Model Service
@Service
public class OrderQueryService {
    
    private final OrderReadModelRepository readModelRepository;
    
    public List<OrderSummary> getOrdersByUserId(String userId, Pageable pageable) {
        return readModelRepository.findByUserIdOrderByCreatedAtDesc(userId, pageable);
    }
    
    public Optional<OrderDetails> getOrderDetails(String orderId) {
        return readModelRepository.findOrderDetailsById(orderId);
    }
    
    public OrderStatistics getOrderStatistics(String userId) {
        return readModelRepository.getStatisticsByUserId(userId);
    }
}

// Read Model Event Handler
@Component
@KafkaListener(topics = "${app.kafka.topic.domain-events}", 
               groupId = "${app.kafka.consumer-group.read-model}")
@Slf4j
public class OrderReadModelEventHandler {
    
    private final OrderReadModelRepository readModelRepository;
    private final RedisTemplate<String, String> redisTemplate;
    
    @KafkaHandler
    public void handle(String eventData, @Header("eventType") String eventType,
                      @Header("aggregateId") String aggregateId) {
        
        // Idempotency check
        String idempotencyKey = String.format("processed:%s:%s", eventType, aggregateId);
        if (Boolean.TRUE.equals(redisTemplate.hasKey(idempotencyKey))) {
            log.debug("Event already processed: {} for aggregate {}", eventType, aggregateId);
            return;
        }
        
        try {
            switch (eventType) {
                case "OrderCreated":
                    handleOrderCreated(objectMapper.readValue(eventData, OrderCreated.class));
                    break;
                case "OrderUpdated":
                    handleOrderUpdated(objectMapper.readValue(eventData, OrderUpdated.class));
                    break;
                case "OrderCancelled":
                    handleOrderCancelled(objectMapper.readValue(eventData, OrderCancelled.class));
                    break;
                default:
                    log.warn("Unknown event type: {}", eventType);
            }
            
            // Mark as processed
            redisTemplate.opsForValue().set(idempotencyKey, "true", Duration.ofDays(7));
            
        } catch (Exception e) {
            log.error("Error processing event: {} for aggregate {}", eventType, aggregateId, e);
            throw new EventProcessingException("Failed to process event", e);
        }
    }
    
    private void handleOrderCreated(OrderCreated event) {
        OrderReadModel readModel = new OrderReadModel();
        readModel.setId(event.getAggregateId());
        readModel.setUserId(event.getUserId());
        readModel.setStatus("PENDING");
        readModel.setTotal(event.getTotal());
        readModel.setItemCount(event.getItems().size());
        readModel.setCreatedAt(event.getTimestamp());
        readModel.setUpdatedAt(event.getTimestamp());
        
        readModelRepository.save(readModel);
        
        // Update user statistics
        updateUserStatistics(event.getUserId());
    }
    
    private void handleOrderUpdated(OrderUpdated event) {
        readModelRepository.findById(event.getAggregateId())
            .ifPresent(readModel -> {
                readModel.setTotal(event.getNewTotal());
                readModel.setItemCount(event.getItems().size());
                readModel.setUpdatedAt(event.getTimestamp());
                readModelRepository.save(readModel);
            });
    }
    
    private void handleOrderCancelled(OrderCancelled event) {
        readModelRepository.findById(event.getAggregateId())
            .ifPresent(readModel -> {
                readModel.setStatus("CANCELLED");
                readModel.setCancelReason(event.getReason());
                readModel.setUpdatedAt(event.getTimestamp());
                readModelRepository.save(readModel);
            });
    }
}
```

### 10.4 Transactional Outbox Pattern

```java
@Entity
@Table(name = "outbox_events")
public class OutboxEvent {
    
    @Id
    private String id;
    
    @Column(name = "aggregate_id")
    private String aggregateId;
    
    @Column(name = "event_type")
    private String eventType;
    
    @Column(name = "event_data", columnDefinition = "TEXT")
    private String eventData;
    
    @Column(name = "created_at")
    private Instant createdAt;
    
    @Column(name = "processed_at")
    private Instant processedAt;
    
    @Column(name = "processed")
    private boolean processed = false;
    
    // constructors, getters, setters...
}

@Repository
public interface OutboxEventRepository extends JpaRepository<OutboxEvent, String> {
    
    @Query("SELECT e FROM OutboxEvent e WHERE e.processed = false ORDER BY e.createdAt")
    List<OutboxEvent> findUnprocessedEvents(Pageable pageable);
    
    @Modifying
    @Query("UPDATE OutboxEvent e SET e.processed = true, e.processedAt = :processedAt WHERE e.id = :id")
    void markAsProcessed(@Param("id") String id, @Param("processedAt") Instant processedAt);
}

// Service to save domain events to outbox
@Service
@Transactional
public class OutboxEventService {
    
    private final OutboxEventRepository outboxRepository;
    private final ObjectMapper objectMapper;
    
    public void saveEvent(DomainEvent event) {
        try {
            OutboxEvent outboxEvent = new OutboxEvent();
            outboxEvent.setId(UUID.randomUUID().toString());
            outboxEvent.setAggregateId(event.getAggregateId());
            outboxEvent.setEventType(event.getClass().getSimpleName());
            outboxEvent.setEventData(objectMapper.writeValueAsString(event));
            outboxEvent.setCreatedAt(Instant.now());
            
            outboxRepository.save(outboxEvent);
        } catch (Exception e) {
            throw new RuntimeException("Failed to save event to outbox", e);
        }
    }
    
    public List<OutboxEvent> getUnprocessedEvents(int batchSize) {
        return outboxRepository.findUnprocessedEvents(PageRequest.of(0, batchSize));
    }
    
    public void markAsProcessed(String eventId) {
        outboxRepository.markAsProcessed(eventId, Instant.now());
    }
}

// Outbox Publisher - polls outbox and publishes to Kafka
@Component
@Slf4j
public class OutboxEventPublisher {
    
    private final OutboxEventService outboxService;
    private final KafkaTemplate<String, String> kafkaTemplate;
    
    @Value("${app.kafka.topic.domain-events}")
    private String domainEventsTopic;
    
    @Scheduled(fixedDelay = 5000) // Poll every 5 seconds
    public void publishOutboxEvents() {
        List<OutboxEvent> unprocessedEvents = outboxService.getUnprocessedEvents(100);
        
        for (OutboxEvent event : unprocessedEvents) {
            try {
                ProducerRecord<String, String> record = new ProducerRecord<>(
                    domainEventsTopic,
                    event.getAggregateId(),
                    event.getEventData()
                );
                
                record.headers().add("eventType", event.getEventType().getBytes());
                record.headers().add("aggregateId", event.getAggregateId().getBytes());
                
                kafkaTemplate.send(record).get(5, TimeUnit.SECONDS); // Wait for confirmation
                
                outboxService.markAsProcessed(event.getId());
                
                log.debug("Published outbox event: {} for aggregate {}", 
                    event.getEventType(), event.getAggregateId());
                    
            } catch (Exception e) {
                log.error("Failed to publish outbox event: {} for aggregate {}", 
                    event.getEventType(), event.getAggregateId(), e);
                // Event will be retried on next poll
            }
        }
    }
}
```

### 10.5 Saga Orchestration with Spring State Machine

```java
// Saga State Machine Configuration
@Configuration
@EnableStateMachine
public class OrderProcessingSagaConfig extends StateMachineConfigurerAdapter<SagaState, SagaEvent> {
    
    @Autowired
    private SagaActionService sagaActionService;
    
    @Override
    public void configure(StateMachineStateConfigurer<SagaState, SagaEvent> states) throws Exception {
        states
            .withStates()
                .initial(SagaState.STARTED)
                .states(EnumSet.allOf(SagaState.class))
                .end(SagaState.COMPLETED)
                .end(SagaState.COMPENSATED);
    }
    
    @Override
    public void configure(StateMachineTransitionConfigurer<SagaState, SagaEvent> transitions) throws Exception {
        transitions
            // Happy path
            .withExternal()
                .source(SagaState.STARTED)
                .target(SagaState.INVENTORY_RESERVED)
                .event(SagaEvent.RESERVE_INVENTORY)
                .action(sagaActionService.reserveInventoryAction())
            .and()
            .withExternal()
                .source(SagaState.INVENTORY_RESERVED)
                .target(SagaState.PAYMENT_PROCESSED)
                .event(SagaEvent.PROCESS_PAYMENT)
                .action(sagaActionService.processPaymentAction())
            .and()
            .withExternal()
                .source(SagaState.PAYMENT_PROCESSED)
                .target(SagaState.COMPLETED)
                .event(SagaEvent.COMPLETE_ORDER)
                .action(sagaActionService.completeOrderAction())
            
            // Compensation path
            .and()
            .withExternal()
                .source(SagaState.PAYMENT_PROCESSED)
                .target(SagaState.PAYMENT_COMPENSATED)
                .event(SagaEvent.COMPENSATE_PAYMENT)
                .action(sagaActionService.compensatePaymentAction())
            .and()
            .withExternal()
                .source(SagaState.INVENTORY_RESERVED)
                .target(SagaState.INVENTORY_COMPENSATED)
                .event(SagaEvent.COMPENSATE_INVENTORY)
                .action(sagaActionService.compensateInventoryAction())
            .and()
            .withExternal()
                .source(SagaState.PAYMENT_COMPENSATED)
                .target(SagaState.INVENTORY_COMPENSATED)
                .event(SagaEvent.COMPENSATE_INVENTORY)
                .action(sagaActionService.compensateInventoryAction())
            .and()
            .withExternal()
                .source(SagaState.INVENTORY_COMPENSATED)
                .target(SagaState.COMPENSATED)
                .event(SagaEvent.SAGA_FAILED);
    }
}

// Saga Orchestrator Service
@Service
@Slf4j
public class OrderProcessingSagaOrchestrator {
    
    private final StateMachine<SagaState, SagaEvent> stateMachine;
    private final SagaStateRepository sagaRepository;
    
    @EventListener
    public void handleOrderCreated(OrderCreated event) {
        SagaInstance saga = SagaInstance.builder()
            .id(UUID.randomUUID().toString())
            .orderId(event.getAggregateId())
            .userId(event.getUserId())
            .state(SagaState.STARTED)
            .data(Map.of(
                "orderId", event.getAggregateId(),
                "userId", event.getUserId(),
                "items", event.getItems(),
                "total", event.getTotal()
            ))
            .createdAt(Instant.now())
            .build();
            
        sagaRepository.save(saga);
        
        // Start the saga
        stateMachine.getExtendedState().getVariables().put("sagaId", saga.getId());
        stateMachine.getExtendedState().getVariables().put("sagaData", saga.getData());
        
        stateMachine.sendEvent(SagaEvent.RESERVE_INVENTORY);
    }
    
    @EventListener
    public void handleInventoryReserved(InventoryReserved event) {
        String sagaId = findSagaIdByOrderId(event.getOrderId());
        if (sagaId != null) {
            updateSagaState(sagaId, SagaState.INVENTORY_RESERVED);
            stateMachine.sendEvent(SagaEvent.PROCESS_PAYMENT);
        }
    }
    
    @EventListener
    public void handleInventoryReservationFailed(InventoryReservationFailed event) {
        String sagaId = findSagaIdByOrderId(event.getOrderId());
        if (sagaId != null) {
            updateSagaState(sagaId, SagaState.COMPENSATED);
            stateMachine.sendEvent(SagaEvent.SAGA_FAILED);
        }
    }
    
    @EventListener
    public void handlePaymentProcessed(PaymentProcessed event) {
        String sagaId = findSagaIdByOrderId(event.getOrderId());
        if (sagaId != null) {
            updateSagaState(sagaId, SagaState.PAYMENT_PROCESSED);
            stateMachine.sendEvent(SagaEvent.COMPLETE_ORDER);
        }
    }
    
    @EventListener
    public void handlePaymentFailed(PaymentFailed event) {
        String sagaId = findSagaIdByOrderId(event.getOrderId());
        if (sagaId != null) {
            updateSagaState(sagaId, SagaState.PAYMENT_FAILED);
            stateMachine.sendEvent(SagaEvent.COMPENSATE_INVENTORY);
        }
    }
    
    private void updateSagaState(String sagaId, SagaState newState) {
        sagaRepository.findById(sagaId).ifPresent(saga -> {
            saga.setState(newState);
            saga.setUpdatedAt(Instant.now());
            sagaRepository.save(saga);
        });
    }
}

// Saga Actions
@Component
@Slf4j
public class SagaActionService {
    
    private final InventoryServiceClient inventoryService;
    private final PaymentServiceClient paymentService;
    private final OrderServiceClient orderService;
    
    public Action<SagaState, SagaEvent> reserveInventoryAction() {
        return context -> {
            Map<String, Object> sagaData = (Map<String, Object>) 
                context.getExtendedState().getVariables().get("sagaData");
                
            try {
                String orderId = (String) sagaData.get("orderId");
                List<OrderItem> items = (List<OrderItem>) sagaData.get("items");
                
                inventoryService.reserveInventory(orderId, items);
                log.info("Inventory reservation initiated for order: {}", orderId);
                
            } catch (Exception e) {
                log.error("Failed to reserve inventory", e);
                context.getStateMachine().sendEvent(SagaEvent.INVENTORY_RESERVATION_FAILED);
            }
        };
    }
    
    public Action<SagaState, SagaEvent> processPaymentAction() {
        return context -> {
            Map<String, Object> sagaData = (Map<String, Object>) 
                context.getExtendedState().getVariables().get("sagaData");
                
            try {
                String orderId = (String) sagaData.get("orderId");
                String userId = (String) sagaData.get("userId");
                BigDecimal total = (BigDecimal) sagaData.get("total");
                
                paymentService.processPayment(orderId, userId, total);
                log.info("Payment processing initiated for order: {}", orderId);
                
            } catch (Exception e) {
                log.error("Failed to process payment", e);
                context.getStateMachine().sendEvent(SagaEvent.PAYMENT_FAILED);
            }
        };
    }
    
    public Action<SagaState, SagaEvent> compensateInventoryAction() {
        return context -> {
            Map<String, Object> sagaData = (Map<String, Object>) 
                context.getExtendedState().getVariables().get("sagaData");
                
            try {
                String orderId = (String) sagaData.get("orderId");
                inventoryService.releaseReservation(orderId);
                log.info("Inventory compensation completed for order: {}", orderId);
                
            } catch (Exception e) {
                log.error("Failed to compensate inventory", e);
            }
        };
    }
    
    public Action<SagaState, SagaEvent> compensatePaymentAction() {
        return context -> {
            Map<String, Object> sagaData = (Map<String, Object>) 
                context.getExtendedState().getVariables().get("sagaData");
                
            try {
                String orderId = (String) sagaData.get("orderId");
                paymentService.refundPayment(orderId);
                log.info("Payment compensation completed for order: {}", orderId);
                
            } catch (Exception e) {
                log.error("Failed to compensate payment", e);
            }
        };
    }
}

enum SagaState {
    STARTED, INVENTORY_RESERVED, PAYMENT_PROCESSED, COMPLETED,
    INVENTORY_COMPENSATED, PAYMENT_COMPENSATED, COMPENSATED
}

enum SagaEvent {
    RESERVE_INVENTORY, PROCESS_PAYMENT, COMPLETE_ORDER,
    COMPENSATE_INVENTORY, COMPENSATE_PAYMENT, SAGA_FAILED,
    INVENTORY_RESERVATION_FAILED, PAYMENT_FAILED
}
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Event sourcing** | Store events; derive state by replay |
| **CQRS** | Separate write and read models |
| **Kafka** | Log, replay, partition ordering |
| **RabbitMQ** | Queues, exchanges, task distribution |
| **Outbox** | Transactional event publishing |
| **Saga** | Choreography vs orchestration; compensating TX |

---

## Further Reading

- *Designing Event-Driven Systems* — Ben Stopford
- Kafka: https://kafka.apache.org/documentation/
- Event Sourcing: https://martinfowler.com/eaaDev/EventSourcing.html
