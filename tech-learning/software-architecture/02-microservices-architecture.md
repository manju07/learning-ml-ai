# Microservices Architecture: Complete Guide for Architects

## Table of Contents
1. [Introduction and Principles](#1-introduction-and-principles)
2. [Decomposition Strategies](#2-decomposition-strategies)
3. [Communication Patterns](#3-communication-patterns)
4. [API Gateway and BFF](#4-api-gateway-and-bff)
5. [Service Discovery and Configuration](#5-service-discovery-and-configuration)
6. [Resilience Patterns](#6-resilience-patterns)
7. [Service Mesh](#7-service-mesh)
8. [Data Management](#8-data-management)
9. [Deployment and Operations](#9-deployment-and-operations)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction and Principles

### 1.1 What Are Microservices?

**Microservices** are an architectural style that structures an application as a collection of loosely coupled, independently deployable services, each owning a business capability.

```
Monolith:                    Microservices:
┌─────────────────────┐     ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│                     │     │Order│ │User │ │Pay- │ │Inv- │
│   Single Deployable │     │ Svc │ │ Svc │ │ment │ │entory│
│   Codebase          │     └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
│                     │        └───────┴───────┴───────┘
└─────────────────────┘              (APIs/Events)
```

### 1.2 Core Principles

| Principle | Description |
|-----------|-------------|
| **Single Responsibility** | One service = one business capability |
| **Loose Coupling** | Services communicate via well-defined APIs; no shared DB |
| **High Cohesion** | Related functionality stays together |
| **Independently Deployable** | Deploy without coordinating with other teams |
| **Database per Service** | Each service owns its data store |

### 1.3 Benefits vs Trade-offs

| Benefits | Trade-offs |
|----------|------------|
| Independent scaling | Operational complexity |
| Technology diversity | Distributed system challenges |
| Fault isolation | Network latency, partial failures |
| Team autonomy | Data consistency, transactions |
| Incremental migration | Deployment orchestration |

---

## 2. Decomposition Strategies

### 2.1 By Business Capability

Align services with business domains (DDD bounded contexts).

```
E-commerce:
  - Order Service (create, track orders)
  - Catalog Service (products, inventory)
  - Payment Service (charges, refunds)
  - User Service (auth, profiles)
  - Notification Service (email, SMS)
```

### 2.2 By Subdomain (DDD)

- **Core domain**: Central to business (e.g., order fulfillment)
- **Supporting domain**: Necessary but not differentiating (e.g., notifications)
- **Generic domain**: Common (e.g., user management)

### 2.3 Strangler Fig Pattern

Gradually replace monolith by building new services and routing traffic.

```
Phase 1: Proxy routes /orders to monolith
Phase 2: New Order Service handles /orders/v2
Phase 3: Migrate data, retire monolith order module
```

### 2.4 Anti-Patterns to Avoid

- **Nano-services**: Services too small (e.g., one function)
- **Distributed monolith**: Shared DB, tight coupling, deploy together
- **God service**: One service does everything

---

## 3. Communication Patterns

### 3.1 Synchronous (Request-Response)

**REST**, **gRPC**, **GraphQL**. Use when you need immediate response.

```
Order Service ──HTTP──> Payment Service
                <──200 OK + result
```

**Pros**: Simple, easy debugging  
**Cons**: Coupling, cascading failures, latency adds up

### 3.2 Asynchronous (Event-Driven)

**Message queues**, **event bus** (Kafka, RabbitMQ). Use when you can tolerate eventual consistency.

```
Order Service ──publish OrderCreated──> Kafka
                        │
                        ├──> Payment Service (consume)
                        ├──> Inventory Service (consume)
                        └──> Notification Service (consume)
```

**Pros**: Loose coupling, resilience, scalability  
**Cons**: Eventual consistency, debugging harder

### 3.3 When to Use Which

| Scenario | Prefer |
|----------|--------|
| Need immediate response | Sync (REST/gRPC) |
| Fire-and-forget, multiple consumers | Async (events) |
| Critical path, low latency | Sync |
| Non-critical, audit trail | Async |

### 3.4 Choreography vs Orchestration

**Choreography**: Each service reacts to events; no central coordinator.  
**Orchestration**: Central coordinator (orchestrator/saga) calls services.

```
Choreography:                    Orchestration:
OrderSvc -> OrderCreated         SagaOrchestrator:
  PaymentSvc subscribes             Call ReserveInventory
  InventorySvc subscribes          Call ChargePayment
  (no central coordinator)         Call ShipOrder
```

---

## 4. API Gateway and BFF

### 4.1 API Gateway

Single entry point for clients. Handles:

- **Routing** to backend services
- **Authentication/Authorization**
- **Rate limiting**, throttling
- **Request/response aggregation**
- **Protocol translation** (e.g., REST to gRPC)

```
Client ──> [API Gateway] ──+──> Order Service
                           ├──> User Service
                           └──> Catalog Service
```

### 4.2 Backend-for-Frontend (BFF)

Separate API layer per client type (web, mobile, third-party). Each BFF aggregates and transforms for its client.

```
Web App ──> Web BFF ──> Microservices
Mobile ──> Mobile BFF ──> Microservices
```

**Example**:
```python
# Web BFF: Returns full HTML-ready data
@app.get("/product/{id}")
def get_product(id):
    product = catalog_service.get(id)
    reviews = review_service.get_for_product(id)
    return {"product": product, "reviews": reviews}

# Mobile BFF: Returns minimal payload
@app.get("/product/{id}")
def get_product_mobile(id):
    return catalog_service.get_minimal(id)
```

### 4.3 Gateway Implementation (Kong/Envoy-style)

```yaml
# Kong-style route
routes:
  - name: order-service
    paths: ["/api/orders"]
    methods: ["GET", "POST"]
    service: order-service
    plugins:
      - rate-limiting
      - jwt-auth
```

---

## 5. Service Discovery and Configuration

### 5.1 Service Discovery

Services find each other by name, not by IP/port.

**Client-side discovery**: Client queries registry (e.g., Consul, Eureka), then calls service directly.

**Server-side discovery**: Client calls load balancer; LB queries registry and routes.

```
Client-side:                    Server-side:
Client -> Registry -> get IP    Client -> LB -> Registry -> route
Client -> Service (direct)
```

### 5.2 Consul Example

```python
import consul

c = consul.Consul()

# Register service
c.agent.service.register(
    name='order-service',
    address='10.0.1.5',
    port=8080,
    check=consul.Check.http('http://10.0.1.5:8080/health', '10s')
)

# Discover service
index, services = c.health.service('order-service', passing=True)
for s in services:
    print(s['Service']['Address'], s['Service']['Port'])
```

### 5.3 Configuration Management

- **12-Factor**: Config in environment, not code
- **External config server**: Spring Cloud Config, Consul KV, etc.
- **Secrets**: Vault, AWS Secrets Manager (never in config files)

---

## 6. Resilience Patterns

### 6.1 Circuit Breaker

Prevent cascading failures when a dependency is down.

```python
from pybreaker import CircuitBreaker

payment_breaker = CircuitBreaker(fail_max=5)

@payment_breaker
def call_payment_service(amount):
    return requests.post(PAYMENT_URL, json={"amount": amount})

# When 5 failures: circuit opens, subsequent calls fail fast
# After timeout: half-open, one trial allowed
```

### 6.2 Retry with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_inventory_service():
    return requests.get(INVENTORY_URL)
```

### 6.3 Bulkhead

Limit resources per dependency.

```python
from concurrent.futures import ThreadPoolExecutor

# Dedicated pool for payment calls (max 5 concurrent)
payment_executor = ThreadPoolExecutor(max_workers=5)

def process_order(order):
    future = payment_executor.submit(call_payment_service, order.amount)
    return future.result()
```

### 6.4 Timeout

Every outbound call must have a timeout.

```python
response = requests.get(url, timeout=5)
```

### 6.5 Fallback

Return default/cached value when dependency fails.

```python
@payment_breaker
def charge_payment(amount):
    return payment_service.charge(amount)

def charge_with_fallback(amount):
    try:
        return charge_payment(amount)
    except CircuitBreakerError:
        # Queue for async retry
        queue_payment_retry(amount)
        return {"status": "pending", "message": "Will process shortly"}
```

---

## 7. Service Mesh

### 7.1 What Is a Service Mesh?

Infrastructure layer that handles service-to-service communication: mTLS, retries, timeouts, circuit breaking, observability. Sidecar (e.g., Envoy) proxies all traffic.

```
Pod:                          Pod:
┌──────────────────────┐     ┌──────────────────────┐
│ Order Service        │     │ Payment Service      │
│         │            │     │            │         │
│         v            │     │            v         │
│  [Envoy Sidecar] <───┼─────┼──> [Envoy Sidecar]   │
└──────────────────────┘     └──────────────────────┘
     mTLS, retries, metrics
```

### 7.2 Istio Core Concepts

- **VirtualService**: Route rules, retries, timeouts
- **DestinationRule**: Load balancing, circuit breaker, mTLS
- **Gateway**: Ingress for mesh

```yaml
# VirtualService: retry and timeout
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - route:
    - destination:
        host: order-service
    timeout: 5s
    retries:
      attempts: 3
      perTryTimeout: 2s
```

### 7.3 When to Use a Service Mesh

| Use | Consider |
|-----|----------|
| mTLS everywhere | Yes |
| Advanced traffic management | Yes |
| Multi-cloud, polyglot | Yes |
| Simple apps, few services | Maybe overkill |

---

## 8. Data Management

### 8.1 Database per Service

Each service owns its database. No shared DB. Communicate via APIs or events.

```
Order Service -> Order DB (PostgreSQL)
Inventory Service -> Inventory DB (PostgreSQL)
User Service -> User DB (MongoDB)
```

### 8.2 Saga for Cross-Service Transactions

Use Saga (choreography or orchestration) instead of 2PC. See [01-distributed-systems.md](./01-distributed-systems.md).

### 8.3 Shared Data Anti-Pattern

Avoid: Service A and B both query the same DB. Breaks independence and can cause schema coupling.

---

## 9. Deployment and Operations

### 9.1 Deployment Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Rolling** | Gradually replace instances | Default |
| **Blue-Green** | Two envs; switch traffic | Zero-downtime |
| **Canary** | Route small % to new version | Low-risk validation |
| **Feature flags** | Code path toggles | Gradual rollouts |

### 9.2 Canary Example (Kubernetes)

```yaml
# 90% to stable, 10% to canary
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
  - order-service
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: order-service
        subset: canary
      weight: 100
  - route:
    - destination:
        host: order-service
        subset: stable
      weight: 90
    - destination:
        host: order-service
        subset: canary
      weight: 10
```

### 9.3 Health Checks

**Liveness**: Is the process running? Restart if not.  
**Readiness**: Can it accept traffic? Remove from LB if not.

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 3
```

---

## 10. Practical Examples

### 10.1 Spring Boot Microservice with Resilience

```java
@RestController
@RequestMapping("/api/orders")
@Slf4j
public class OrderController {
    
    private final OrderService orderService;
    private final InventoryServiceClient inventoryClient;
    private final PaymentServiceClient paymentClient;
    private final OrderEventPublisher eventPublisher;
    
    @PostMapping
    public ResponseEntity<OrderResponse> createOrder(@RequestBody @Valid OrderRequest request) {
        try {
            // Reserve inventory with circuit breaker
            ReservationResult reservation = inventoryClient.reserveItems(request.getItems());
            if (!reservation.isSuccess()) {
                throw new BusinessException("Inventory reservation failed: " + reservation.getReason());
            }
            
            // Process payment
            PaymentResult payment = paymentClient.processPayment(
                PaymentRequest.builder()
                    .amount(request.getTotal())
                    .userId(request.getUserId())
                    .orderId(request.getOrderId())
                    .build()
            );
            
            if (!payment.isSuccess()) {
                // Compensate: release inventory
                inventoryClient.releaseItems(reservation.getReservationId());
                throw new BusinessException("Payment failed: " + payment.getReason());
            }
            
            // Create order
            Order order = orderService.createOrder(request, reservation.getReservationId(), payment.getPaymentId());
            
            // Publish domain event
            eventPublisher.publishOrderCreated(
                OrderCreatedEvent.builder()
                    .orderId(order.getId())
                    .userId(order.getUserId())
                    .total(order.getTotal())
                    .timestamp(Instant.now())
                    .build()
            );
            
            return ResponseEntity.ok(OrderResponse.from(order));
            
        } catch (BusinessException e) {
            log.error("Order creation failed: {}", e.getMessage());
            return ResponseEntity.badRequest()
                .body(OrderResponse.error(e.getMessage()));
        } catch (Exception e) {
            log.error("Unexpected error during order creation", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(OrderResponse.error("Internal server error"));
        }
    }
}

// Service layer with business logic
@Service
@Transactional
public class OrderService {
    
    private final OrderRepository orderRepository;
    private final OrderMapper orderMapper;
    
    public Order createOrder(OrderRequest request, String reservationId, String paymentId) {
        Order order = Order.builder()
            .id(UUID.randomUUID().toString())
            .userId(request.getUserId())
            .items(request.getItems().stream()
                .map(orderMapper::toOrderItem)
                .collect(Collectors.toList()))
            .total(request.getTotal())
            .status(OrderStatus.CONFIRMED)
            .reservationId(reservationId)
            .paymentId(paymentId)
            .createdAt(Instant.now())
            .build();
            
        return orderRepository.save(order);
    }
}

// Configuration for resilience
@Configuration
public class ResilienceConfiguration {
    
    @Bean
    public CircuitBreaker inventoryCircuitBreaker() {
        return CircuitBreaker.ofDefaults("inventory")
            .toBuilder()
            .failureRateThreshold(50.0f)
            .waitDurationInOpenState(Duration.ofSeconds(10))
            .slidingWindowSize(10)
            .minimumNumberOfCalls(5)
            .build();
    }
    
    @Bean
    public RetryConfig retryConfig() {
        return RetryConfig.custom()
            .maxAttempts(3)
            .waitDuration(Duration.ofMillis(500))
            .retryExceptions(ConnectException.class, SocketTimeoutException.class)
            .build();
    }
}
```

### 10.2 Feign Client with Circuit Breaker

```java
@FeignClient(name = "inventory-service", 
             configuration = InventoryServiceConfig.class,
             fallback = InventoryServiceFallback.class)
public interface InventoryServiceClient {
    
    @PostMapping("/api/inventory/reserve")
    ReservationResult reserveItems(@RequestBody List<OrderItem> items);
    
    @PostMapping("/api/inventory/release/{reservationId}")
    void releaseItems(@PathVariable String reservationId);
}

// Fallback implementation
@Component
public class InventoryServiceFallback implements InventoryServiceClient {
    
    @Override
    public ReservationResult reserveItems(List<OrderItem> items) {
        log.warn("Inventory service unavailable, using fallback");
        return ReservationResult.failure("Service temporarily unavailable");
    }
    
    @Override
    public void releaseItems(String reservationId) {
        log.warn("Inventory service unavailable, cannot release reservation: {}", reservationId);
    }
}

// Client configuration
@Configuration
public class InventoryServiceConfig {
    
    @Bean
    public Retryer retryer() {
        return new Retryer.Default(100, 1000, 3);
    }
    
    @Bean
    public RequestInterceptor requestInterceptor() {
        return template -> {
            template.header("X-Request-ID", UUID.randomUUID().toString());
            template.header("X-Service-Name", "order-service");
        };
    }
}
```

### 10.3 Event-Driven Communication with Kafka

```java
@Component
@Slf4j
public class OrderEventPublisher {
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final ObjectMapper objectMapper;
    
    @Value("${app.kafka.topics.order-events}")
    private String orderEventsTopic;
    
    public void publishOrderCreated(OrderCreatedEvent event) {
        try {
            String eventJson = objectMapper.writeValueAsString(event);
            
            ProducerRecord<String, Object> record = new ProducerRecord<>(
                orderEventsTopic,
                event.getOrderId(), // Key for partitioning
                eventJson
            );
            
            // Add headers for tracing
            record.headers().add("event-type", "OrderCreated".getBytes());
            record.headers().add("event-version", "v1".getBytes());
            record.headers().add("timestamp", String.valueOf(System.currentTimeMillis()).getBytes());
            
            kafkaTemplate.send(record)
                .addCallback(
                    result -> log.info("Order event published successfully: {}", event.getOrderId()),
                    failure -> log.error("Failed to publish order event: {}", event.getOrderId(), failure)
                );
                
        } catch (Exception e) {
            log.error("Error serializing order event", e);
            throw new EventPublishingException("Failed to publish order created event", e);
        }
    }
}

// Event listener in another service
@Component
@KafkaListener(topics = "${app.kafka.topics.order-events}", 
               groupId = "${app.kafka.consumer-group}")
@Slf4j
public class OrderEventListener {
    
    private final PaymentService paymentService;
    private final InventoryService inventoryService;
    private final NotificationService notificationService;
    
    @KafkaHandler
    public void handleOrderCreated(@Payload String eventData, 
                                 @Header("event-type") String eventType,
                                 @Header Map<String, Object> headers) {
        
        if (!"OrderCreated".equals(eventType)) {
            log.debug("Ignoring event type: {}", eventType);
            return;
        }
        
        try {
            OrderCreatedEvent event = objectMapper.readValue(eventData, OrderCreatedEvent.class);
            
            log.info("Processing order created event: {}", event.getOrderId());
            
            // Update inventory
            inventoryService.updateStockLevels(event.getOrderId(), event.getItems());
            
            // Send confirmation email
            notificationService.sendOrderConfirmation(event.getUserId(), event.getOrderId());
            
            // Update analytics
            analyticsService.recordOrderCreated(event);
            
        } catch (Exception e) {
            log.error("Error processing order created event", e);
            throw new EventProcessingException("Failed to process order created event", e);
        }
    }
    
    @RetryableTopic(
        attempts = "3",
        backoff = @Backoff(delay = 1000, multiplier = 2.0),
        dltStrategy = DltStrategy.FAIL_ON_ERROR
    )
    @KafkaHandler
    public void handleWithRetry(ConsumerRecord<String, String> record) {
        // Automatic retry with exponential backoff
        handleOrderCreated(record.value(), 
                          new String(record.headers().lastHeader("event-type").value()),
                          extractHeaders(record.headers()));
    }
}
```

### 10.4 API Gateway with Spring Cloud Gateway

```java
@Configuration
@EnableConfigurationProperties(GatewayProperties.class)
public class GatewayConfig {
    
    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            // Order Service Routes
            .route("order-service", r -> r
                .path("/api/orders/**")
                .filters(f -> f
                    .circuitBreaker(config -> config
                        .setName("order-service-cb")
                        .setFallbackUri("forward:/fallback/orders"))
                    .retry(config -> config
                        .setRetries(3)
                        .setStatuses(HttpStatus.INTERNAL_SERVER_ERROR)
                        .setBackoff(Duration.ofMillis(100), Duration.ofMillis(1000), 2, true))
                    .requestRateLimiter(config -> config
                        .setRateLimiter(redisRateLimiter())
                        .setKeyResolver(userKeyResolver()))
                    .addRequestHeader("X-Gateway-Request-ID", "#{T(java.util.UUID).randomUUID().toString()}")
                    .addResponseHeader("X-Response-Time", "#{T(System).currentTimeMillis()}")
                )
                .uri("lb://order-service"))
                
            // User Service Routes
            .route("user-service", r -> r
                .path("/api/users/**")
                .filters(f -> f
                    .circuitBreaker(config -> config
                        .setName("user-service-cb"))
                    .addRequestHeader("X-Gateway-Version", "v1.0"))
                .uri("lb://user-service"))
                
            // Payment Service Routes with Authentication
            .route("payment-service", r -> r
                .path("/api/payments/**")
                .filters(f -> f
                    .filter(new AuthenticationGatewayFilterFactory().apply(
                        new AuthenticationGatewayFilterFactory.Config()))
                    .circuitBreaker(config -> config
                        .setName("payment-service-cb"))
                    .requestRateLimiter(config -> config
                        .setRateLimiter(redisRateLimiter())
                        .setKeyResolver(userKeyResolver())))
                .uri("lb://payment-service"))
            .build();
    }
    
    @Bean
    public RedisRateLimiter redisRateLimiter() {
        return new RedisRateLimiter(10, 20, 1); // 10 requests per second, burst of 20
    }
    
    @Bean
    KeyResolver userKeyResolver() {
        return exchange -> exchange.getRequest().getHeaders()
            .getFirst("X-User-ID")
            .map(Mono::just)
            .orElse(exchange.getRequest().getRemoteAddress()
                .map(address -> Mono.just(address.getAddress().getHostAddress()))
                .orElse(Mono.just("unknown")));
    }
}

// Custom Authentication Filter
@Component
public class AuthenticationGatewayFilterFactory extends AbstractGatewayFilterFactory<AuthenticationGatewayFilterFactory.Config> {
    
    private final JwtService jwtService;
    private final ReactiveRedisTemplate<String, String> redisTemplate;
    
    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            String authHeader = exchange.getRequest().getHeaders().getFirst("Authorization");
            
            if (authHeader == null || !authHeader.startsWith("Bearer ")) {
                return handleUnauthorized(exchange);
            }
            
            String token = authHeader.substring(7);
            
            return jwtService.validateToken(token)
                .flatMap(claims -> {
                    ServerHttpRequest modifiedRequest = exchange.getRequest().mutate()
                        .header("X-User-ID", claims.getSubject())
                        .header("X-User-Roles", String.join(",", claims.getRoles()))
                        .build();
                    
                    return chain.filter(exchange.mutate().request(modifiedRequest).build());
                })
                .onErrorResume(error -> {
                    log.warn("Token validation failed", error);
                    return handleUnauthorized(exchange);
                });
        };
    }
    
    private Mono<Void> handleUnauthorized(ServerWebExchange exchange) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(HttpStatus.UNAUTHORIZED);
        response.getHeaders().add("Content-Type", "application/json");
        
        String body = "{\"error\":\"Unauthorized\",\"message\":\"Valid JWT token required\"}";
        DataBuffer buffer = response.bufferFactory().wrap(body.getBytes());
        return response.writeWith(Mono.just(buffer));
    }
    
    @Data
    public static class Config {
        private boolean validateExpiry = true;
        private boolean requireRoles = false;
        private List<String> allowedRoles = new ArrayList<>();
    }
}
```

### 10.5 Service Discovery with Consul

```java
@RestController
@RequestMapping("/api/health")
public class HealthController {
    
    @Autowired
    private ConsulRegistration consulRegistration;
    
    @GetMapping("/live")
    public ResponseEntity<Map<String, Object>> liveness() {
        Map<String, Object> status = Map.of(
            "status", "UP",
            "timestamp", Instant.now(),
            "service", consulRegistration.getServiceId()
        );
        return ResponseEntity.ok(status);
    }
    
    @GetMapping("/ready")
    public ResponseEntity<Map<String, Object>> readiness() {
        // Check dependencies
        boolean databaseHealthy = checkDatabaseConnection();
        boolean kafkaHealthy = checkKafkaConnection();
        
        if (databaseHealthy && kafkaHealthy) {
            Map<String, Object> status = Map.of(
                "status", "UP",
                "timestamp", Instant.now(),
                "checks", Map.of(
                    "database", "UP",
                    "kafka", "UP"
                )
            );
            return ResponseEntity.ok(status);
        } else {
            Map<String, Object> status = Map.of(
                "status", "DOWN",
                "timestamp", Instant.now(),
                "checks", Map.of(
                    "database", databaseHealthy ? "UP" : "DOWN",
                    "kafka", kafkaHealthy ? "UP" : "DOWN"
                )
            );
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(status);
        }
    }
}

// Service configuration
@Configuration
public class ServiceDiscoveryConfig {
    
    @Bean
    @ConditionalOnMissingBean
    public ConsulServiceRegistry consulServiceRegistry(ConsulClient consulClient,
                                                      ConsulDiscoveryProperties properties,
                                                      ConsulRegistration consulRegistration) {
        return new ConsulServiceRegistry(consulClient, properties, consulRegistration);
    }
    
    @Bean
    public ConsulRegistration consulRegistration(ConsulDiscoveryProperties properties) {
        NewService service = new NewService();
        service.setId(properties.getInstanceId());
        service.setName(properties.getServiceName());
        service.setAddress(properties.getHostname());
        service.setPort(properties.getPort());
        
        // Health check configuration
        NewService.Check check = new NewService.Check();
        check.setHttp(properties.getHealthCheckUrl());
        check.setInterval(properties.getHealthCheckInterval());
        check.setTimeout(properties.getHealthCheckTimeout());
        check.setDeregisterCriticalServiceAfter(properties.getHealthCheckCriticalTimeout());
        service.setCheck(check);
        
        return new ConsulRegistration(service, properties);
    }
}
```

### 10.6 Distributed Tracing with Micrometer

```java
@Configuration
public class TracingConfiguration {
    
    @Bean
    public Sender sender() {
        return OkHttpSender.create("http://jaeger:14268/api/traces");
    }
    
    @Bean
    public AsyncReporter<Span> spanReporter() {
        return AsyncReporter.create(sender());
    }
    
    @Bean
    public BraveTracer braveTracer() {
        return BraveTracer.create(
            Tracing.newBuilder()
                .localServiceName("order-service")
                .spanReporter(spanReporter())
                .sampler(Sampler.create(1.0f))
                .build()
        );
    }
}

// Custom tracing aspects
@Aspect
@Component
@Slf4j
public class TracingAspect {
    
    private final Tracer tracer;
    
    @Around("@annotation(Traced)")
    public Object trace(ProceedingJoinPoint joinPoint, Traced traced) throws Throwable {
        String operationName = traced.value().isEmpty() ? 
            joinPoint.getSignature().getName() : traced.value();
            
        Span span = tracer.nextSpan()
            .name(operationName)
            .tag("class", joinPoint.getTarget().getClass().getSimpleName())
            .tag("method", joinPoint.getSignature().getName())
            .start();
            
        try (Tracer.SpanInScope ws = tracer.withSpanInScope(span)) {
            Object result = joinPoint.proceed();
            span.tag("success", "true");
            return result;
        } catch (Exception e) {
            span.tag("error", e.getMessage());
            span.tag("success", "false");
            throw e;
        } finally {
            span.end();
        }
    }
}

// Usage
@Service
public class OrderService {
    
    @Traced("create-order")
    public Order createOrder(OrderRequest request) {
        // Implementation
    }
}
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Decomposition** | By business capability / bounded context |
| **Communication** | Sync for immediate need; async for decoupling |
| **API Gateway** | Single entry, routing, auth, rate limit |
| **Discovery** | Registry (Consul, Eureka) for dynamic routing |
| **Resilience** | Circuit breaker, retry, timeout, bulkhead |
| **Service Mesh** | mTLS, observability, traffic control |
| **Data** | DB per service; Saga for cross-service TX |
| **Deployment** | Canary, blue-green, health checks |

---

## Further Reading

- *Building Microservices* — Sam Newman
- *Microservices Patterns* — Chris Richardson
- Istio: https://istio.io/
- Consul: https://www.consul.io/
