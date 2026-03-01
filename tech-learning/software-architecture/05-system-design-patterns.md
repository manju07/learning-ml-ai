# System Design Patterns: Load Balancing, Rate Limiting, and Resilience

## Table of Contents
1. [Load Balancing](#1-load-balancing)
2. [Rate Limiting and Throttling](#2-rate-limiting-and-throttling)
3. [Circuit Breaker and Bulkhead](#3-circuit-breaker-and-bulkhead)
4. [Retry and Timeout](#4-retry-and-timeout)
5. [Caching Patterns](#5-caching-patterns)
6. [Queue-Based Load Leveling](#6-queue-based-load-leveling)
7. [Bulkhead and Isolation](#7-bulkhead-and-isolation)
8. [Health Checks and Graceful Degradation](#8-health-checks-and-graceful-degradation)
9. [Practical Examples](#9-practical-examples)

---

## 1. Load Balancing

### 1.1 Purpose

Distribute traffic across multiple instances to improve availability and scalability.

### 1.2 Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Round Robin** | Rotate through servers | Equal capacity |
| **Least Connections** | Send to server with fewest active connections | Variable request duration |
| **IP Hash** | Hash client IP to server | Session affinity |
| **Weighted Round Robin** | Round robin with server weights | Different capacity |
| **Least Response Time** | Send to fastest responder | Performance optimization |

### 1.3 Layer 4 vs Layer 7

- **L4 (Transport)**: TCP/UDP, no application awareness. Fast, simple. (e.g., HAProxy in TCP mode)
- **L7 (Application)**: HTTP, can route by path, header, cookie. (e.g., Nginx, Envoy, AWS ALB)

### 1.4 Session Affinity (Sticky Sessions)

Same client → same server. Needed when state is in-memory.

```
Client A -> LB -> Server 1 (session stored)
Client A (next request) -> LB -> Server 1 (same session)
```

**Trade-off**: Reduces load distribution; use external session store (Redis) when possible.

### 1.5 Example: Nginx Load Balancing

```nginx
upstream backend {
    least_conn;  # or round_robin (default)
    server 10.0.1.1:8080 weight=3;
    server 10.0.1.2:8080 weight=1;
    server 10.0.1.3:8080 backup;  # Use only when others down
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_next_upstream error timeout http_502 http_503;
    }
}
```

---

## 2. Rate Limiting and Throttling

### 2.1 Purpose

Protect services from overload, ensure fair usage, prevent abuse.

### 2.2 Algorithms

| Algorithm | Description | Pros | Cons |
|-----------|-------------|------|------|
| **Fixed Window** | Reset counter at interval (e.g., 100/min) | Simple | Burst at boundaries |
| **Sliding Window** | Rolling window | Smoother | More state |
| **Token Bucket** | Tokens refill at rate; consume per request | Allows bursts | More complex |
| **Leaky Bucket** | Requests processed at fixed rate | Smooth output | Can delay |

### 2.3 Sliding Window Log (Redis)

```python
import redis
import time

def rate_limit(key: str, limit: int, window_sec: int) -> bool:
    r = redis.Redis()
    now = time.time()
    window_start = now - window_sec
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, window_sec)
    results = pipe.execute()
    return results[2] <= limit

# Usage: 100 requests per minute per user
if not rate_limit(f"ratelimit:user:{user_id}", 100, 60):
    raise HTTPException(429, "Too Many Requests")
```

### 2.4 Token Bucket (Conceptual)

```python
from threading import Lock
import time

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()
        self.lock = Lock()

    def consume(self, n: int = 1) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False
```

### 2.5 HTTP 429 and Headers

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
```

---

## 3. Circuit Breaker and Bulkhead

### 3.1 Circuit Breaker

Prevent cascading failures. Stop calling a failing dependency; periodically probe for recovery.

**States**:
- **Closed**: Normal operation
- **Open**: Failing fast, no calls to dependency
- **Half-Open**: Allow one trial; on success → Closed, on failure → Open

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30, expected_exception=ConnectionError)
def call_payment_service(amount: float):
    return requests.post(PAYMENT_URL, json={"amount": amount}, timeout=5)
```

### 3.2 Bulkhead

Isolate resources (thread pools, connections) by component. One failing component doesn't exhaust all.

```
Without bulkhead: 100 threads shared
  Payment fails -> 100 threads blocked -> no threads for Orders

With bulkhead:
  Payment pool: 10 threads
  Order pool: 50 threads
  Inventory pool: 20 threads
  Payment fails -> only 10 threads blocked
```

```python
from concurrent.futures import ThreadPoolExecutor

payment_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="payment")
order_executor = ThreadPoolExecutor(max_workers=50, thread_name_prefix="order")
```

---

## 4. Retry and Timeout

### 4.1 Retry Best Practices

- **Idempotency**: Retries must be safe. Use idempotency keys for mutations.
- **Exponential backoff**: 1s, 2s, 4s, 8s (with jitter)
- **Max attempts**: Don't retry forever
- **Retry only on transient errors**: 5xx, timeout, connection refused; not 4xx

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
)
def call_service(url: str):
    return requests.get(url, timeout=5)
```

### 4.2 Timeout

- **Connection timeout**: Time to establish connection
- **Read timeout**: Time to receive response
- **Total timeout**: End-to-end (if chaining services)

```python
# Always set timeouts
response = requests.get(url, timeout=(3, 10))  # (connect, read)
```

---

## 5. Caching Patterns

### 5.1 Cache-Aside (Lazy Loading)

```
1. Check cache
2. Miss -> load from DB
3. Store in cache
4. Return
```

### 5.2 Write-Through

Write to cache and DB together. Cache is source of truth for reads.

### 5.3 Cache Invalidation

- **TTL**: Simple; may serve stale
- **Invalidate on write**: Delete/update cache when DB changes
- **Write-through**: Keeps cache warm

### 5.4 Cache-Aside Example

```python
def get_user(user_id: str):
    cached = redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    user = db.query(User).get(user_id)
    if user:
        redis.setex(f"user:{user_id}", 3600, json.dumps(user.to_dict(), default=str))
    return user
```

---

## 6. Queue-Based Load Leveling

### 6.1 Concept

Introduce a queue between clients and workers. Absorb bursts; workers process at steady rate.

```
Clients -> [Queue] -> Workers
         (buffer)    (process at fixed rate)
```

### 6.2 Use Cases

- Image processing
- Email sending
- Report generation
- Async API (request accepted, result polled later)

### 6.3 Example: Celery

```python
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task(bind=True, max_retries=3)
def process_order(self, order_id: str):
    try:
        order = get_order(order_id)
        # Heavy processing
        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

---

## 7. Bulkhead and Isolation

### 7.1 Database Connection Pools

Separate pools per service or per priority.

```python
# Critical path
critical_pool = create_engine(..., pool_size=20)
# Background jobs
background_pool = create_engine(..., pool_size=5)
```

### 7.2 Thread Pools (Bulkhead)

Already shown in Section 3.2.

---

## 8. Health Checks and Graceful Degradation

### 8.1 Liveness vs Readiness

- **Liveness**: Is the process alive? Restart if not. (e.g., `/health/live`)
- **Readiness**: Can it accept traffic? Remove from LB if not. (e.g., `/health/ready`)

Readiness can fail when:
- DB connection lost
- Dependency unhealthy
- Startup not complete

### 8.2 Graceful Shutdown

1. Stop accepting new connections
2. Finish in-flight requests
3. Close connections
4. Exit

```python
import signal

def graceful_shutdown(signum, frame):
    logger.info("Shutting down...")
    server.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
```

### 8.3 Degradation Strategies

| Strategy | Example |
|----------|---------|
| **Fallback** | Return cached/default when dependency fails |
| **Timeout** | Fail fast, return partial response |
| **Circuit open** | Skip call, use default |
| **Queue** | Accept request, process async |

---

## 9. Practical Examples

### 9.1 Complete Resilience Service (Spring Boot)

```java
@Service
@Slf4j
public class ResilientHttpClient {
    
    private final RestTemplate restTemplate;
    private final CircuitBreaker circuitBreaker;
    private final Retry retry;
    private final TimeLimiter timeLimiter;
    private final Bulkhead bulkhead;
    
    public ResilientHttpClient() {
        this.restTemplate = new RestTemplate();
        
        // Configure circuit breaker
        this.circuitBreaker = CircuitBreaker.ofDefaults("http-client")
            .toBuilder()
            .failureRateThreshold(50.0f)
            .waitDurationInOpenState(Duration.ofSeconds(10))
            .slidingWindowSize(10)
            .minimumNumberOfCalls(5)
            .recordExceptions(ConnectException.class, SocketTimeoutException.class)
            .build();
            
        // Configure retry
        this.retry = Retry.ofDefaults("http-client")
            .toBuilder()
            .maxAttempts(3)
            .waitDuration(Duration.ofMillis(500))
            .intervalFunction(IntervalFunction.ofExponentialBackoff(500, 2.0, 5000))
            .retryExceptions(ConnectException.class, SocketTimeoutException.class)
            .build();
            
        // Configure time limiter
        this.timeLimiter = TimeLimiter.ofDefaults("http-client")
            .toBuilder()
            .timeoutDuration(Duration.ofSeconds(5))
            .build();
            
        // Configure bulkhead
        this.bulkhead = Bulkhead.ofDefaults("http-client")
            .toBuilder()
            .maxConcurrentCalls(10)
            .maxWaitDuration(Duration.ofMillis(500))
            .build();
    }
    
    public <T> Optional<T> get(String url, Class<T> responseType) {
        return get(url, responseType, Optional.empty());
    }
    
    public <T> Optional<T> get(String url, Class<T> responseType, Optional<T> fallback) {
        Supplier<Optional<T>> decoratedSupplier = decorateSupplier(() -> {
            try {
                T response = restTemplate.getForObject(url, responseType);
                return Optional.ofNullable(response);
            } catch (Exception e) {
                log.warn("Request failed for URL: {}, error: {}", url, e.getMessage());
                if (fallback.isPresent()) {
                    log.info("Using fallback value for URL: {}", url);
                    return fallback;
                }
                throw e;
            }
        });
        
        try {
            return decoratedSupplier.get();
        } catch (Exception e) {
            log.error("All resilience patterns exhausted for URL: {}", url, e);
            return fallback;
        }
    }
    
    private <T> Supplier<T> decorateSupplier(Supplier<T> supplier) {
        return Decorators.ofSupplier(supplier)
            .withBulkhead(bulkhead)
            .withTimeLimiter(timeLimiter)
            .withRetry(retry)
            .withCircuitBreaker(circuitBreaker)
            .decorate();
    }
    
    // Reactive version with WebClient
    @Component
    public static class ReactiveResilientHttpClient {
        
        private final WebClient webClient;
        private final ReactorCircuitBreaker circuitBreaker;
        private final ReactorRetry retry;
        private final ReactorTimeLimiter timeLimiter;
        
        public ReactiveResilientHttpClient() {
            this.webClient = WebClient.builder()
                .codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(1024 * 1024))
                .build();
                
            this.circuitBreaker = ReactorCircuitBreaker.ofDefaults("reactive-http-client");
            this.retry = ReactorRetry.ofDefaults("reactive-http-client");
            this.timeLimiter = ReactorTimeLimiter.ofDefaults("reactive-http-client");
        }
        
        public <T> Mono<T> get(String url, Class<T> responseType) {
            return webClient.get()
                .uri(url)
                .retrieve()
                .bodyToMono(responseType)
                .transformDeferred(timeLimiter::executeSupplier)
                .transformDeferred(retry::transformPublisher)
                .transformDeferred(circuitBreaker::transformPublisher)
                .doOnSuccess(response -> log.debug("Request successful for URL: {}", url))
                .doOnError(error -> log.warn("Request failed for URL: {}, error: {}", url, error.getMessage()));
        }
    }
}
```

### 9.2 Advanced Rate Limiting with Redis (Spring Boot)

```java
@Component
@Slf4j
public class AdvancedRateLimiter {
    
    private final StringRedisTemplate redisTemplate;
    private final RedisScript<List<Long>> rateLimitScript;
    
    public AdvancedRateLimiter(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
        this.rateLimitScript = createRateLimitScript();
    }
    
    // Sliding window log rate limiter
    public boolean isAllowed(String key, int limit, Duration window) {
        long now = System.currentTimeMillis();
        long windowStart = now - window.toMillis();
        
        List<Long> results = redisTemplate.execute(
            rateLimitScript,
            Collections.singletonList(key),
            String.valueOf(windowStart),
            String.valueOf(now),
            String.valueOf(limit),
            String.valueOf(window.getSeconds())
        );
        
        long count = results.get(0);
        return count <= limit;
    }
    
    // Token bucket rate limiter
    public boolean consumeTokens(String key, int tokens, int bucketSize, int refillRate) {
        String script = """
            local bucket_key = KEYS[1]
            local capacity = tonumber(ARGV[1])
            local tokens_requested = tonumber(ARGV[2])
            local refill_rate = tonumber(ARGV[3])
            local now = tonumber(ARGV[4])
            
            local bucket = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
            local tokens = tonumber(bucket[1]) or capacity
            local last_refill = tonumber(bucket[2]) or now
            
            -- Calculate tokens to add
            local elapsed = math.max(0, now - last_refill)
            local tokens_to_add = math.floor(elapsed * refill_rate / 1000)
            tokens = math.min(capacity, tokens + tokens_to_add)
            
            local allowed = 0
            if tokens >= tokens_requested then
                tokens = tokens - tokens_requested
                allowed = 1
            end
            
            -- Update bucket
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', bucket_key, 3600)
            
            return {allowed, tokens}
        """;
        
        List<Long> result = redisTemplate.execute(
            new DefaultRedisScript<>(script, List.class),
            Collections.singletonList(key),
            String.valueOf(bucketSize),
            String.valueOf(tokens),
            String.valueOf(refillRate),
            String.valueOf(System.currentTimeMillis())
        );
        
        return result.get(0) == 1;
    }
    
    // Fixed window counter
    public boolean isAllowedFixedWindow(String key, int limit, Duration window) {
        long windowStart = System.currentTimeMillis() / window.toMillis() * window.toMillis();
        String windowKey = key + ":" + windowStart;
        
        String script = """
            local counter_key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local ttl = tonumber(ARGV[2])
            
            local current = redis.call('GET', counter_key)
            if current == false then
                redis.call('SETEX', counter_key, ttl, 1)
                return 1
            end
            
            current = tonumber(current)
            if current < limit then
                redis.call('INCR', counter_key)
                return 1
            else
                return 0
            end
        """;
        
        Long result = redisTemplate.execute(
            new DefaultRedisScript<>(script, Long.class),
            Collections.singletonList(windowKey),
            String.valueOf(limit),
            String.valueOf(window.getSeconds())
        );
        
        return result == 1;
    }
    
    private RedisScript<List<Long>> createRateLimitScript() {
        String script = """
            local key = KEYS[1]
            local window_start = tonumber(ARGV[1])
            local now = tonumber(ARGV[2])
            local limit = tonumber(ARGV[3])
            local window_seconds = tonumber(ARGV[4])
            
            -- Remove expired entries
            redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
            
            -- Add current request
            redis.call('ZADD', key, now, now)
            
            -- Count requests in window
            local count = redis.call('ZCARD', key)
            
            -- Set TTL
            redis.call('EXPIRE', key, window_seconds)
            
            return {count}
        """;
        
        return new DefaultRedisScript<>(script, List.class);
    }
}

// Rate limiting interceptor
@Component
@Slf4j
public class RateLimitingInterceptor implements HandlerInterceptor {
    
    private final AdvancedRateLimiter rateLimiter;
    
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, 
                           Object handler) throws Exception {
        
        String clientId = extractClientId(request);
        String endpoint = request.getRequestURI();
        
        RateLimitConfig config = getRateLimitConfig(endpoint);
        String rateLimitKey = String.format("rate_limit:%s:%s", clientId, endpoint);
        
        if (!rateLimiter.isAllowed(rateLimitKey, config.getLimit(), config.getWindow())) {
            response.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
            response.setHeader("Retry-After", String.valueOf(config.getWindow().getSeconds()));
            response.setHeader("X-RateLimit-Limit", String.valueOf(config.getLimit()));
            response.setHeader("X-RateLimit-Remaining", "0");
            response.getWriter().write("{\"error\":\"Rate limit exceeded\"}");
            return false;
        }
        
        return true;
    }
    
    private String extractClientId(HttpServletRequest request) {
        String apiKey = request.getHeader("X-API-Key");
        if (apiKey != null) return apiKey;
        
        String userId = request.getHeader("X-User-ID");
        if (userId != null) return userId;
        
        return request.getRemoteAddr();
    }
}
```

### 9.3 Comprehensive Circuit Breaker Implementation

```java
@Component
@Slf4j
public class EnhancedCircuitBreaker {
    
    private final Map<String, CircuitBreakerState> circuitBreakers = new ConcurrentHashMap<>();
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
    
    public <T> T execute(String name, Supplier<T> operation, Function<Exception, T> fallback) {
        CircuitBreakerState cb = circuitBreakers.computeIfAbsent(name, 
            k -> new CircuitBreakerState(name));
            
        return cb.execute(operation, fallback);
    }
    
    @PreDestroy
    public void shutdown() {
        scheduler.shutdown();
    }
    
    private class CircuitBreakerState {
        private final String name;
        private volatile State state = State.CLOSED;
        private final AtomicInteger failureCount = new AtomicInteger(0);
        private final AtomicInteger successCount = new AtomicInteger(0);
        private volatile long lastFailureTime = 0;
        private volatile long stateTransitionTime = System.currentTimeMillis();
        
        // Configuration
        private final int failureThreshold = 5;
        private final long timeout = 30000; // 30 seconds
        private final int halfOpenMaxCalls = 3;
        private final double failureRateThreshold = 0.5; // 50%
        
        public CircuitBreakerState(String name) {
            this.name = name;
        }
        
        public <T> T execute(Supplier<T> operation, Function<Exception, T> fallback) {
            State currentState = state;
            
            switch (currentState) {
                case OPEN:
                    if (shouldAttemptReset()) {
                        return attemptReset(operation, fallback);
                    } else {
                        log.warn("Circuit breaker {} is OPEN, using fallback", name);
                        return fallback.apply(new CircuitBreakerOpenException());
                    }
                    
                case HALF_OPEN:
                    return executeInHalfOpenState(operation, fallback);
                    
                case CLOSED:
                default:
                    return executeInClosedState(operation, fallback);
            }
        }
        
        private <T> T executeInClosedState(Supplier<T> operation, Function<Exception, T> fallback) {
            try {
                T result = operation.get();
                onSuccess();
                return result;
            } catch (Exception e) {
                onFailure(e);
                
                // Check if we should open the circuit
                if (shouldOpenCircuit()) {
                    openCircuit();
                }
                
                return fallback.apply(e);
            }
        }
        
        private <T> T executeInHalfOpenState(Supplier<T> operation, Function<Exception, T> fallback) {
            try {
                T result = operation.get();
                onSuccessInHalfOpen();
                return result;
            } catch (Exception e) {
                onFailureInHalfOpen(e);
                return fallback.apply(e);
            }
        }
        
        private <T> T attemptReset(Supplier<T> operation, Function<Exception, T> fallback) {
            if (state.compareAndSet(State.OPEN, State.HALF_OPEN)) {
                log.info("Circuit breaker {} transitioning to HALF_OPEN", name);
                stateTransitionTime = System.currentTimeMillis();
                successCount.set(0);
                failureCount.set(0);
            }
            return executeInHalfOpenState(operation, fallback);
        }
        
        private boolean shouldAttemptReset() {
            return System.currentTimeMillis() - lastFailureTime >= timeout;
        }
        
        private boolean shouldOpenCircuit() {
            int totalCalls = successCount.get() + failureCount.get();
            if (totalCalls < failureThreshold) {
                return false;
            }
            
            double failureRate = (double) failureCount.get() / totalCalls;
            return failureRate >= failureRateThreshold;
        }
        
        private void openCircuit() {
            if (state == State.CLOSED) {
                state = State.OPEN;
                lastFailureTime = System.currentTimeMillis();
                stateTransitionTime = lastFailureTime;
                log.warn("Circuit breaker {} is now OPEN", name);
                
                // Schedule automatic recovery attempt
                scheduler.schedule(() -> {
                    if (state == State.OPEN && shouldAttemptReset()) {
                        log.info("Scheduling recovery attempt for circuit breaker {}", name);
                    }
                }, timeout, TimeUnit.MILLISECONDS);
            }
        }
        
        private void closeCircuit() {
            state = State.CLOSED;
            stateTransitionTime = System.currentTimeMillis();
            successCount.set(0);
            failureCount.set(0);
            log.info("Circuit breaker {} is now CLOSED", name);
        }
        
        private void onSuccess() {
            successCount.incrementAndGet();
        }
        
        private void onFailure(Exception e) {
            failureCount.incrementAndGet();
            log.debug("Circuit breaker {} recorded failure: {}", name, e.getMessage());
        }
        
        private void onSuccessInHalfOpen() {
            int successes = successCount.incrementAndGet();
            if (successes >= halfOpenMaxCalls) {
                closeCircuit();
            }
        }
        
        private void onFailureInHalfOpen(Exception e) {
            openCircuit();
        }
    }
    
    private enum State {
        CLOSED, OPEN, HALF_OPEN
    }
    
    public static class CircuitBreakerOpenException extends RuntimeException {
        public CircuitBreakerOpenException() {
            super("Circuit breaker is open");
        }
    }
}
```

### 9.4 Bulkhead Pattern Implementation

```java
@Service
@Slf4j
public class BulkheadService {
    
    private final Map<String, ThreadPoolExecutor> executors = new ConcurrentHashMap<>();
    private final MeterRegistry meterRegistry;
    
    public BulkheadService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        initializeExecutors();
    }
    
    private void initializeExecutors() {
        // Critical operations pool
        executors.put("critical", createExecutor("critical", 10, 20, 60));
        
        // Payment operations pool
        executors.put("payment", createExecutor("payment", 5, 10, 30));
        
        // Reporting operations pool
        executors.put("reporting", createExecutor("reporting", 3, 5, 120));
        
        // Background tasks pool
        executors.put("background", createExecutor("background", 2, 5, 300));
    }
    
    private ThreadPoolExecutor createExecutor(String name, int coreSize, int maxSize, int queueCapacity) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
            coreSize,
            maxSize,
            60L, TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(queueCapacity),
            new ThreadFactoryBuilder()
                .setNameFormat(name + "-pool-%d")
                .setDaemon(false)
                .build(),
            new RejectedExecutionHandler() {
                @Override
                public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                    meterRegistry.counter("bulkhead.rejected", "pool", name).increment();
                    throw new BulkheadRejectedException("Pool " + name + " is exhausted");
                }
            }
        );
        
        // Register metrics
        Metrics.gauge("bulkhead.pool.size", Tags.of("pool", name), executor, ThreadPoolExecutor::getPoolSize);
        Metrics.gauge("bulkhead.active.count", Tags.of("pool", name), executor, ThreadPoolExecutor::getActiveCount);
        Metrics.gauge("bulkhead.queue.size", Tags.of("pool", name), executor, e -> e.getQueue().size());
        
        return executor;
    }
    
    public <T> CompletableFuture<T> execute(String poolName, Supplier<T> task) {
        ThreadPoolExecutor executor = executors.get(poolName);
        if (executor == null) {
            throw new IllegalArgumentException("Unknown pool: " + poolName);
        }
        
        return CompletableFuture.supplyAsync(task, executor);
    }
    
    public <T> T executeBlocking(String poolName, Supplier<T> task, Duration timeout) {
        CompletableFuture<T> future = execute(poolName, task);
        try {
            return future.get(timeout.toMillis(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            meterRegistry.counter("bulkhead.timeout", "pool", poolName).increment();
            throw new BulkheadTimeoutException("Task timed out in pool: " + poolName);
        } catch (Exception e) {
            meterRegistry.counter("bulkhead.error", "pool", poolName).increment();
            throw new RuntimeException("Task execution failed in pool: " + poolName, e);
        }
    }
    
    @PreDestroy
    public void shutdown() {
        executors.values().forEach(executor -> {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                executor.shutdownNow();
            }
        });
    }
    
    public static class BulkheadRejectedException extends RuntimeException {
        public BulkheadRejectedException(String message) {
            super(message);
        }
    }
    
    public static class BulkheadTimeoutException extends RuntimeException {
        public BulkheadTimeoutException(String message) {
            super(message);
        }
    }
}

// Usage example
@Service
public class OrderProcessingService {
    
    private final BulkheadService bulkheadService;
    private final PaymentService paymentService;
    private final InventoryService inventoryService;
    private final ReportingService reportingService;
    
    public CompletableFuture<Order> processOrderAsync(OrderRequest request) {
        return bulkheadService.execute("critical", () -> {
            // Critical order processing logic
            return processOrderInternal(request);
        });
    }
    
    public void processPaymentAsync(PaymentRequest request) {
        bulkheadService.execute("payment", () -> {
            paymentService.processPayment(request);
            return null;
        });
    }
    
    public void generateReportAsync(ReportRequest request) {
        bulkheadService.execute("reporting", () -> {
            reportingService.generateReport(request);
            return null;
        });
    }
}
```

### 9.5 Idempotency Framework

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Idempotent {
    String keyExpression() default "";
    int ttlHours() default 24;
    String keyPrefix() default "idempotent";
}

@Aspect
@Component
@Slf4j
public class IdempotencyAspect {
    
    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    private final SpelExpressionParser parser = new SpelExpressionParser();
    
    @Around("@annotation(idempotent)")
    public Object handleIdempotency(ProceedingJoinPoint joinPoint, Idempotent idempotent) throws Throwable {
        String idempotencyKey = generateIdempotencyKey(joinPoint, idempotent);
        String cacheKey = idempotent.keyPrefix() + ":" + idempotencyKey;
        
        // Try to get cached result
        String cachedResult = redisTemplate.opsForValue().get(cacheKey);
        if (cachedResult != null) {
            log.debug("Returning cached result for idempotency key: {}", idempotencyKey);
            return deserializeResult(cachedResult, joinPoint.getSignature().getReturnType());
        }
        
        // Set processing lock to prevent concurrent execution of same key
        String lockKey = cacheKey + ":lock";
        Boolean lockAcquired = redisTemplate.opsForValue().setIfAbsent(lockKey, "processing", Duration.ofMinutes(5));
        
        if (!lockAcquired) {
            // Wait and retry
            Thread.sleep(100);
            cachedResult = redisTemplate.opsForValue().get(cacheKey);
            if (cachedResult != null) {
                return deserializeResult(cachedResult, joinPoint.getSignature().getReturnType());
            }
            throw new IdempotencyProcessingException("Idempotent operation is being processed");
        }
        
        try {
            // Execute the method
            Object result = joinPoint.proceed();
            
            // Cache the result
            String serializedResult = serializeResult(result);
            redisTemplate.opsForValue().set(cacheKey, serializedResult, Duration.ofHours(idempotent.ttlHours()));
            
            return result;
            
        } finally {
            // Release lock
            redisTemplate.delete(lockKey);
        }
    }
    
    private String generateIdempotencyKey(ProceedingJoinPoint joinPoint, Idempotent idempotent) {
        if (!idempotent.keyExpression().isEmpty()) {
            // Use SpEL expression
            StandardEvaluationContext context = new StandardEvaluationContext();
            Object[] args = joinPoint.getArgs();
            String[] paramNames = getParameterNames(joinPoint);
            
            for (int i = 0; i < args.length; i++) {
                context.setVariable(paramNames[i], args[i]);
            }
            
            Expression expression = parser.parseExpression(idempotent.keyExpression());
            return expression.getValue(context, String.class);
        } else {
            // Generate key from method signature and arguments
            String methodName = joinPoint.getSignature().getName();
            String className = joinPoint.getTarget().getClass().getSimpleName();
            String argsHash = generateArgsHash(joinPoint.getArgs());
            return className + "." + methodName + ":" + argsHash;
        }
    }
    
    private String generateArgsHash(Object[] args) {
        try {
            String argsJson = objectMapper.writeValueAsString(args);
            return DigestUtils.md5DigestAsHex(argsJson.getBytes());
        } catch (Exception e) {
            log.warn("Failed to serialize arguments for hash generation", e);
            return String.valueOf(Arrays.hashCode(args));
        }
    }
    
    // Usage examples
    @RestController
    public class PaymentController {
        
        @PostMapping("/payments")
        @Idempotent(keyExpression = "#request.idempotencyKey")
        public ResponseEntity<PaymentResponse> createPayment(@RequestBody PaymentRequest request) {
            // Payment processing logic
            return ResponseEntity.ok(paymentService.processPayment(request));
        }
        
        @PostMapping("/orders/{orderId}/items")
        @Idempotent(keyExpression = "#orderId + ':' + #request.userId", ttlHours = 48)
        public ResponseEntity<OrderItemResponse> addOrderItem(
                @PathVariable String orderId,
                @RequestBody AddItemRequest request) {
            // Order item addition logic
            return ResponseEntity.ok(orderService.addItem(orderId, request));
        }
    }
}
```

---

## Summary

| Pattern | Purpose | Key Implementation |
|---------|---------|--------------------|
| **Load balancing** | Distribute load | Round robin, least conn, L7 routing |
| **Rate limiting** | Prevent overload | Sliding window, token bucket |
| **Circuit breaker** | Fail fast when dependency down | Closed/Open/Half-Open |
| **Retry** | Handle transient failures | Exponential backoff, idempotency |
| **Timeout** | Prevent hanging | Every outbound call |
| **Bulkhead** | Isolate failures | Separate pools/threads |
| **Queue** | Level load | Async processing |

---

## Further Reading

- *Release It!* — Michael Nygard
- *Building Microservices* — Sam Newman
- Nginx load balancing: https://nginx.org/en/docs/http/load_balancing.html
