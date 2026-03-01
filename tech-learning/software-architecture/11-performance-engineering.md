# Performance Engineering: Complete Guide for Architects

## Table of Contents
1. [Introduction](#1-introduction)
2. [JVM Performance and Tuning](#2-jvm-performance-and-tuning)
3. [Application Profiling](#3-application-profiling)
4. [Memory Management](#4-memory-management)
5. [Database Performance](#5-database-performance)
6. [Caching Strategies](#6-caching-strategies)
7. [Network and I/O Optimization](#7-network-and-io-optimization)
8. [Concurrency and Threading](#8-concurrency-and-threading)
9. [Load Testing and Capacity Planning](#9-load-testing-and-capacity-planning)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

**Performance Engineering** is the practice of ensuring systems meet performance requirements throughout their lifecycle. It's proactive optimization, not reactive debugging.

### 1.1 Performance Fundamentals

| Metric | Description | Typical SLA |
|--------|-------------|-------------|
| **Latency** | Time to process single request | P50 < 100ms, P99 < 500ms |
| **Throughput** | Requests processed per second | > 1000 RPS |
| **Availability** | System uptime percentage | 99.9% (8.76 hours downtime/year) |
| **Error Rate** | Failed requests percentage | < 0.1% |
| **Resource Utilization** | CPU, memory, disk usage | < 70% sustained |

### 1.2 Performance Engineering Process

```
1. Requirements → SLAs/SLOs
2. Architecture → Performance modeling
3. Implementation → Profiling & optimization
4. Testing → Load/stress testing
5. Production → Monitoring & alerting
6. Analysis → Continuous improvement
```

---

## 2. JVM Performance and Tuning

### 2.1 JVM Memory Model

```
JVM Memory Structure:
┌─────────────────────────────────────┐
│           Method Area               │ ← Shared
│  (Metaspace in Java 8+)           │
├─────────────────────────────────────┤
│              Heap                   │ ← Shared
│  ┌─────────────┬─────────────────┐  │
│  │ Young Gen   │   Old Gen       │  │
│  │┌────┬────┐  │                 │  │
│  ││Eden│S0│S1│  │                 │  │
│  │└────┴────┘  │                 │  │
│  └─────────────┴─────────────────┘  │
├─────────────────────────────────────┤
│         Direct Memory               │ ← Off-heap
└─────────────────────────────────────┘

Per Thread:
┌─────────────────┐
│   Java Stack    │
├─────────────────┤
│  Native Stack   │
├─────────────────┤
│   PC Register   │
└─────────────────┘
```

### 2.2 Garbage Collection Tuning

```java
// JVM GC Tuning Parameters (Java 11+)
public class GCTuningExamples {
    
    /*
    G1GC (Default for Java 11+):
    -XX:+UseG1GC
    -XX:MaxGCPauseMillis=200           # Target pause time
    -XX:G1HeapRegionSize=16m           # Region size
    -XX:G1NewSizePercent=20            # Min young gen %
    -XX:G1MaxNewSizePercent=40         # Max young gen %
    -XX:InitiatingHeapOccupancyPercent=45  # Mixed GC threshold
    
    ZGC (Ultra-low latency):
    -XX:+UseZGC                        # < 10ms pauses
    -XX:+UnlockExperimentalVMOptions   # Required for Java 11-14
    
    Parallel GC (High throughput):
    -XX:+UseParallelGC
    -XX:ParallelGCThreads=8            # GC thread count
    -XX:MaxGCPauseMillis=200
    
    Memory Settings:
    -Xms8g                             # Initial heap
    -Xmx8g                             # Max heap (set equal for consistency)
    -XX:NewRatio=3                     # Old/Young ratio
    -XX:MaxMetaspaceSize=512m          # Metaspace limit
    -XX:MaxDirectMemorySize=2g         # Direct memory limit
    
    GC Logging:
    -Xlog:gc*:gc.log:time,tags         # Java 11+ logging
    -XX:+LogVMOutput                   # Detailed VM info
    */
}

// GC Monitoring Service
@Service
@Slf4j
public class GCMonitoringService {
    
    private final MeterRegistry meterRegistry;
    private final List<GarbageCollectorMXBean> gcBeans;
    private final MemoryMXBean memoryBean;
    
    public GCMonitoringService(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
        this.gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        
        startMonitoring();
    }
    
    @Scheduled(fixedRate = 10000) // Every 10 seconds
    public void monitorGC() {
        // GC Metrics
        for (GarbageCollectorMXBean gcBean : gcBeans) {
            String gcName = gcBean.getName().replace(" ", "_");
            
            Gauge.builder("jvm_gc_collection_count")
                .tag("gc", gcName)
                .register(meterRegistry, gcBean, GarbageCollectorMXBean::getCollectionCount);
                
            Gauge.builder("jvm_gc_collection_time_ms")
                .tag("gc", gcName)
                .register(meterRegistry, gcBean, GarbageCollectorMXBean::getCollectionTime);
        }
        
        // Memory Usage
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();
        
        Gauge.builder("jvm_memory_used_bytes")
            .tag("area", "heap")
            .register(meterRegistry, heapUsage, MemoryUsage::getUsed);
            
        Gauge.builder("jvm_memory_max_bytes")
            .tag("area", "heap")
            .register(meterRegistry, heapUsage, MemoryUsage::getMax);
            
        // GC Pressure Calculation
        double gcPressure = calculateGCPressure();
        Gauge.builder("jvm_gc_pressure")
            .register(meterRegistry, () -> gcPressure);
    }
    
    private double calculateGCPressure() {
        long totalGCTime = gcBeans.stream()
            .mapToLong(GarbageCollectorMXBean::getCollectionTime)
            .sum();
            
        long uptime = ManagementFactory.getRuntimeMXBean().getUptime();
        return (double) totalGCTime / uptime;
    }
}
```

### 2.3 JVM Optimization Flags

```java
// Production JVM Configuration Template
public class ProductionJVMConfig {
    
    /*
    Startup Script Example:
    
    #!/bin/bash
    
    # Memory Configuration
    HEAP_SIZE="8g"
    MAX_DIRECT_MEMORY="2g"
    METASPACE_SIZE="512m"
    
    # GC Configuration (G1GC)
    GC_OPTS="-XX:+UseG1GC"
    GC_OPTS="$GC_OPTS -XX:MaxGCPauseMillis=200"
    GC_OPTS="$GC_OPTS -XX:G1HeapRegionSize=16m"
    GC_OPTS="$GC_OPTS -XX:G1NewSizePercent=20"
    GC_OPTS="$GC_OPTS -XX:G1MaxNewSizePercent=40"
    GC_OPTS="$GC_OPTS -XX:InitiatingHeapOccupancyPercent=45"
    
    # Performance Optimizations
    PERF_OPTS="-XX:+UseStringDeduplication"        # Deduplicate strings
    PERF_OPTS="$PERF_OPTS -XX:+OptimizeStringConcat"  # Optimize string concatenation
    PERF_OPTS="$PERF_OPTS -XX:+UseFastAccessorMethods"  # Fast reflection
    PERF_OPTS="$PERF_OPTS -Djava.awt.headless=true"     # Headless mode
    
    # JIT Compilation
    JIT_OPTS="-XX:+TieredCompilation"              # Use tiered compilation
    JIT_OPTS="$JIT_OPTS -XX:ReservedCodeCacheSize=256m"  # Code cache size
    JIT_OPTS="$JIT_OPTS -XX:CompileThreshold=10000"      # Compilation threshold
    
    # Error Handling
    ERROR_OPTS="-XX:+ExitOnOutOfMemoryError"       # Exit on OOM
    ERROR_OPTS="$ERROR_OPTS -XX:+CrashOnOutOfMemoryError"  # Generate dump
    ERROR_OPTS="$ERROR_OPTS -XX:+HeapDumpOnOutOfMemoryError"
    ERROR_OPTS="$ERROR_OPTS -XX:HeapDumpPath=/var/log/heapdumps/"
    
    # Logging
    LOG_OPTS="-Xlog:gc*:gc.log:time,tags"
    LOG_OPTS="$LOG_OPTS -Xlog:safepoint:safepoint.log:time"
    LOG_OPTS="$LOG_OPTS -XX:+LogVMOutput"
    
    # Final command
    java -Xms$HEAP_SIZE -Xmx$HEAP_SIZE \
         -XX:MaxDirectMemorySize=$MAX_DIRECT_MEMORY \
         -XX:MaxMetaspaceSize=$METASPACE_SIZE \
         $GC_OPTS $PERF_OPTS $JIT_OPTS $ERROR_OPTS $LOG_OPTS \
         -jar application.jar
    */
}
```

---

## 3. Application Profiling

### 3.1 Profiling Tools Integration

```java
// Async Profiler Integration
@RestController
@RequestMapping("/admin/profiling")
@ConditionalOnProperty(name = "app.profiling.enabled", havingValue = "true")
public class ProfilingController {
    
    private final AsyncProfiler profiler;
    
    public ProfilingController() {
        this.profiler = AsyncProfiler.getInstance();
    }
    
    @PostMapping("/start")
    public ResponseEntity<?> startProfiling(
            @RequestParam(defaultValue = "cpu") String event,
            @RequestParam(defaultValue = "60") int durationSeconds) {
        
        try {
            String command = String.format("%s,interval=1000000,file=/tmp/profile-%d.html",
                event, System.currentTimeMillis());
            profiler.execute(command);
            
            // Auto-stop after duration
            CompletableFuture.delayedExecutor(durationSeconds, TimeUnit.SECONDS)
                .execute(this::stopProfiling);
                
            return ResponseEntity.ok(Map.of("status", "started", "duration", durationSeconds));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }
    
    @PostMapping("/stop")
    public ResponseEntity<?> stopProfiling() {
        try {
            profiler.stop();
            return ResponseEntity.ok(Map.of("status", "stopped"));
        } catch (Exception e) {
            return ResponseEntity.status(500).body(Map.of("error", e.getMessage()));
        }
    }
    
    @GetMapping("/status")
    public ResponseEntity<?> getStatus() {
        return ResponseEntity.ok(Map.of(
            "profiling", profiler.isRunning(),
            "version", profiler.getVersion()
        ));
    }
}

// Method-level performance tracking
@Aspect
@Component
@ConditionalOnProperty(name = "app.performance-tracking.enabled", havingValue = "true")
public class PerformanceTrackingAspect {
    
    private final MeterRegistry meterRegistry;
    
    @Around("@annotation(PerformanceTracked)")
    public Object trackPerformance(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().getName();
        String className = joinPoint.getTarget().getClass().getSimpleName();
        
        Timer.Sample sample = Timer.start(meterRegistry);
        
        try {
            Object result = joinPoint.proceed();
            
            sample.stop(Timer.builder("method_execution_time")
                .tag("class", className)
                .tag("method", methodName)
                .tag("status", "success")
                .register(meterRegistry));
                
            return result;
        } catch (Exception e) {
            sample.stop(Timer.builder("method_execution_time")
                .tag("class", className)
                .tag("method", methodName)
                .tag("status", "error")
                .register(meterRegistry));
            throw e;
        }
    }
}

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface PerformanceTracked {
    String value() default "";
}
```

### 3.2 Memory Leak Detection

```java
@Component
@Slf4j
public class MemoryLeakDetector {
    
    private final MemoryMXBean memoryBean;
    private final Map<String, Long> baselineMemory = new ConcurrentHashMap<>();
    
    @Scheduled(fixedRate = 300000) // Every 5 minutes
    public void checkMemoryLeaks() {
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        long currentUsed = heapUsage.getUsed();
        long maxMemory = heapUsage.getMax();
        
        // Memory leak indicators
        double memoryUtilization = (double) currentUsed / maxMemory;
        
        if (memoryUtilization > 0.85) {
            log.warn("High memory utilization: {}%", memoryUtilization * 100);
            
            // Trigger garbage collection and re-check
            System.gc();
            
            // Wait a bit for GC to complete
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            
            MemoryUsage afterGC = memoryBean.getHeapMemoryUsage();
            long afterGCUsed = afterGC.getUsed();
            double retainedRatio = (double) afterGCUsed / currentUsed;
            
            if (retainedRatio > 0.8) {
                log.error("Potential memory leak detected. Memory not freed after GC: {}%",
                    retainedRatio * 100);
                    
                // Create heap dump for analysis
                createHeapDump();
            }
        }
        
        // Track memory growth over time
        trackMemoryGrowth(currentUsed);
    }
    
    private void trackMemoryGrowth(long currentUsed) {
        String timeKey = LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm"));
        Long baseline = baselineMemory.get("baseline");
        
        if (baseline == null) {
            baselineMemory.put("baseline", currentUsed);
            return;
        }
        
        double growthRate = (double) (currentUsed - baseline) / baseline;
        
        if (growthRate > 0.5) { // 50% growth
            log.warn("Memory growth detected: {}%", growthRate * 100);
        }
        
        // Update baseline weekly
        if (ChronoUnit.HOURS.between(
            LocalDateTime.now().minusWeeks(1), LocalDateTime.now()) < 1) {
            baselineMemory.put("baseline", currentUsed);
        }
    }
    
    private void createHeapDump() {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = new ObjectName("com.sun.management:type=HotSpotDiagnostic");
            
            String fileName = String.format("/tmp/heapdump-%d.hprof", System.currentTimeMillis());
            server.invoke(objectName, "dumpHeap", 
                new Object[]{fileName, true}, 
                new String[]{"java.lang.String", "boolean"});
                
            log.info("Heap dump created: {}", fileName);
        } catch (Exception e) {
            log.error("Failed to create heap dump", e);
        }
    }
}
```

---

## 4. Memory Management

### 4.1 Object Pool Pattern

```java
// High-performance object pooling for expensive objects
public class ObjectPool<T> {
    
    private final Queue<T> pool = new ConcurrentLinkedQueue<>();
    private final Supplier<T> factory;
    private final Consumer<T> resetFunction;
    private final int maxSize;
    private final AtomicInteger size = new AtomicInteger(0);
    
    public ObjectPool(Supplier<T> factory, Consumer<T> resetFunction, int maxSize) {
        this.factory = factory;
        this.resetFunction = resetFunction;
        this.maxSize = maxSize;
    }
    
    public T acquire() {
        T object = pool.poll();
        if (object == null) {
            object = factory.get();
            size.incrementAndGet();
        }
        return object;
    }
    
    public void release(T object) {
        if (object != null && size.get() < maxSize) {
            resetFunction.accept(object);
            pool.offer(object);
        }
    }
    
    public int size() {
        return pool.size();
    }
}

// Example: ByteBuffer pool for network operations
@Component
public class ByteBufferPool {
    
    private final ObjectPool<ByteBuffer> smallBuffers;
    private final ObjectPool<ByteBuffer> largeBuffers;
    
    public ByteBufferPool() {
        this.smallBuffers = new ObjectPool<>(
            () -> ByteBuffer.allocateDirect(8192),  // 8KB
            ByteBuffer::clear,
            100
        );
        
        this.largeBuffers = new ObjectPool<>(
            () -> ByteBuffer.allocateDirect(65536), // 64KB
            ByteBuffer::clear,
            20
        );
    }
    
    public ByteBuffer acquireSmall() {
        return smallBuffers.acquire();
    }
    
    public ByteBuffer acquireLarge() {
        return largeBuffers.acquire();
    }
    
    public void release(ByteBuffer buffer) {
        if (buffer.capacity() <= 8192) {
            smallBuffers.release(buffer);
        } else {
            largeBuffers.release(buffer);
        }
    }
}
```

### 4.2 Memory-Efficient Collections

```java
// Memory-optimized collections for high-performance scenarios
public class MemoryOptimizedCollections {
    
    // Primitive collections to avoid boxing overhead
    private final TIntObjectHashMap<String> userIdToName = new TIntObjectHashMap<>();
    private final TLongList timestamps = new TLongArrayList();
    
    // Off-heap storage using Chronicle Map
    private final ChronicleMap<String, UserData> offHeapCache;
    
    public MemoryOptimizedCollections() {
        this.offHeapCache = ChronicleMap
            .of(String.class, UserData.class)
            .entries(1_000_000)
            .averageKeySize(20)
            .averageValueSize(100)
            .create();
    }
    
    // Memory-efficient string interning
    private final ConcurrentHashMap<String, String> stringPool = new ConcurrentHashMap<>();
    
    public String intern(String str) {
        return stringPool.computeIfAbsent(str, Function.identity());
    }
    
    // Flyweight pattern for common objects
    private static final Map<String, UserRole> ROLE_CACHE = Map.of(
        "ADMIN", new UserRole("ADMIN", Set.of("READ", "WRITE", "DELETE")),
        "USER", new UserRole("USER", Set.of("READ")),
        "GUEST", new UserRole("GUEST", Set.of())
    );
    
    public UserRole getRole(String roleName) {
        return ROLE_CACHE.get(roleName);
    }
    
    // Efficient batch processing with streaming
    public void processBatchEfficiently(List<OrderData> orders) {
        // Group by customer to reduce object creation
        Map<String, List<OrderData>> ordersByCustomer = orders.parallelStream()
            .collect(Collectors.groupingByConcurrent(OrderData::getCustomerId));
            
        ordersByCustomer.entrySet().parallelStream()
            .forEach(entry -> {
                String customerId = entry.getKey();
                List<OrderData> customerOrders = entry.getValue();
                
                // Process all orders for customer together
                processCustomerOrders(customerId, customerOrders);
            });
    }
}
```

---

## 5. Database Performance

### 5.1 Connection Pool Optimization

```java
@Configuration
public class DatabasePerformanceConfig {
    
    @Bean
    @Primary
    public DataSource primaryDataSource() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:postgresql://localhost:5432/mydb");
        config.setUsername("user");
        config.setPassword("password");
        
        // Connection pool tuning
        config.setMaximumPoolSize(20);                    // Max connections
        config.setMinimumIdle(5);                         // Min idle connections
        config.setConnectionTimeout(30000);               // 30 seconds
        config.setIdleTimeout(300000);                    // 5 minutes
        config.setMaxLifetime(1200000);                   // 20 minutes
        config.setLeakDetectionThreshold(60000);          // 1 minute leak detection
        
        // Performance optimizations
        config.addDataSourceProperty("cachePrepStmts", "true");
        config.addDataSourceProperty("prepStmtCacheSize", "250");
        config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");
        config.addDataSourceProperty("useServerPrepStmts", "true");
        
        return new HikariDataSource(config);
    }
    
    @Bean
    public DataSource readOnlyDataSource() {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl("jdbc:postgresql://readonly-replica:5432/mydb");
        config.setUsername("readonly_user");
        config.setPassword("password");
        config.setReadOnly(true);
        
        // Read replica optimizations
        config.setMaximumPoolSize(15);
        config.setConnectionTimeout(10000);              // Shorter timeout
        config.addDataSourceProperty("defaultRowFetchSize", "1000");
        
        return new HikariDataSource(config);
    }
}

// Query optimization with JPA
@Repository
public class OptimizedOrderRepository {
    
    @PersistenceContext
    private EntityManager entityManager;
    
    // Batch fetching to reduce N+1 queries
    @Query("SELECT o FROM Order o JOIN FETCH o.items WHERE o.customerId = :customerId")
    List<Order> findOrdersWithItemsByCustomerId(@Param("customerId") String customerId);
    
    // Projection to fetch only needed fields
    @Query("SELECT new com.example.dto.OrderSummary(o.id, o.total, o.createdAt) " +
           "FROM Order o WHERE o.customerId = :customerId")
    List<OrderSummary> findOrderSummariesByCustomerId(@Param("customerId") String customerId);
    
    // Batch insert for better performance
    @Modifying
    @Transactional
    public void batchInsertOrders(List<Order> orders) {
        int batchSize = 50;
        for (int i = 0; i < orders.size(); i += batchSize) {
            List<Order> batch = orders.subList(i, Math.min(i + batchSize, orders.size()));
            
            batch.forEach(entityManager::persist);
            
            if ((i + 1) % batchSize == 0 || i + batchSize >= orders.size()) {
                entityManager.flush();
                entityManager.clear();
            }
        }
    }
    
    // Native query for complex operations
    @Query(value = """
        WITH order_stats AS (
            SELECT customer_id, 
                   COUNT(*) as order_count,
                   SUM(total) as total_spent
            FROM orders 
            WHERE created_at >= :fromDate 
            GROUP BY customer_id
        )
        SELECT customer_id, order_count, total_spent
        FROM order_stats 
        WHERE total_spent > :minSpent
        ORDER BY total_spent DESC
        LIMIT :limit
        """, nativeQuery = true)
    List<Object[]> findTopSpendingCustomers(
        @Param("fromDate") LocalDateTime fromDate,
        @Param("minSpent") BigDecimal minSpent,
        @Param("limit") int limit
    );
}
```

### 5.2 Database Monitoring

```java
@Component
@Slf4j
public class DatabasePerformanceMonitor {
    
    private final MeterRegistry meterRegistry;
    private final DataSource dataSource;
    
    @EventListener
    public void handleSlowQuery(SlowQueryEvent event) {
        if (event.getExecutionTimeMs() > 1000) { // > 1 second
            log.warn("Slow query detected: {}ms - {}", 
                event.getExecutionTimeMs(), event.getSql());
                
            // Record metric
            Timer.builder("database_slow_query")
                .tag("table", extractTableName(event.getSql()))
                .register(meterRegistry)
                .record(event.getExecutionTimeMs(), TimeUnit.MILLISECONDS);
        }
    }
    
    @Scheduled(fixedRate = 30000) // Every 30 seconds
    public void monitorConnectionPool() {
        if (dataSource instanceof HikariDataSource) {
            HikariDataSource hikari = (HikariDataSource) dataSource;
            HikariPoolMXBean pool = hikari.getHikariPoolMXBean();
            
            // Connection pool metrics
            Gauge.builder("hikari_connections_active")
                .register(meterRegistry, pool, HikariPoolMXBean::getActiveConnections);
                
            Gauge.builder("hikari_connections_idle")
                .register(meterRegistry, pool, HikariPoolMXBean::getIdleConnections);
                
            Gauge.builder("hikari_connections_total")
                .register(meterRegistry, pool, HikariPoolMXBean::getTotalConnections);
                
            Gauge.builder("hikari_connections_awaiting")
                .register(meterRegistry, pool, HikariPoolMXBean::getThreadsAwaitingConnection);
        }
    }
    
    private String extractTableName(String sql) {
        // Simple extraction - in production, use a proper SQL parser
        String upperSql = sql.toUpperCase();
        if (upperSql.contains("FROM ")) {
            int fromIndex = upperSql.indexOf("FROM ") + 5;
            int endIndex = upperSql.indexOf(" ", fromIndex);
            if (endIndex == -1) endIndex = upperSql.length();
            return upperSql.substring(fromIndex, endIndex).trim();
        }
        return "unknown";
    }
}
```

---

## 6. Caching Strategies

### 6.1 Multi-Level Caching

```java
@Configuration
@EnableCaching
public class CacheConfiguration {
    
    // L1 Cache: In-memory (Caffeine)
    @Bean
    public CacheManager l1CacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        cacheManager.setCaffeine(Caffeine.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(Duration.ofMinutes(10))
            .expireAfterAccess(Duration.ofMinutes(5))
            .recordStats()
            .removalListener((key, value, cause) -> {
                if (cause == RemovalCause.EVICTED) {
                    log.debug("Cache eviction: key={}, cause={}", key, cause);
                }
            }));
        return cacheManager;
    }
    
    // L2 Cache: Distributed (Redis)
    @Bean
    public CacheManager l2CacheManager(RedisConnectionFactory connectionFactory) {
        RedisCacheConfiguration config = RedisCacheConfiguration.defaultCacheConfig()
            .entryTtl(Duration.ofHours(1))
            .serializeKeysWith(RedisSerializationContext.SerializationPair
                .fromSerializer(new StringRedisSerializer()))
            .serializeValuesWith(RedisSerializationContext.SerializationPair
                .fromSerializer(new GenericJackson2JsonRedisSerializer()));
            
        return RedisCacheManager.builder(connectionFactory)
            .cacheDefaults(config)
            .build();
    }
}

@Service
public class MultiLevelCachingService {
    
    private final Cache l1Cache;
    private final RedisTemplate<String, Object> redisTemplate;
    private final UserRepository userRepository;
    
    public User getUser(String userId) {
        // L1 Cache check
        User user = l1Cache.get(userId, User.class);
        if (user != null) {
            return user;
        }
        
        // L2 Cache check
        user = (User) redisTemplate.opsForValue().get("user:" + userId);
        if (user != null) {
            l1Cache.put(userId, user); // Populate L1
            return user;
        }
        
        // Database fetch
        user = userRepository.findById(userId).orElse(null);
        if (user != null) {
            l1Cache.put(userId, user);
            redisTemplate.opsForValue().set("user:" + userId, user, Duration.ofHours(1));
        }
        
        return user;
    }
    
    @CacheEvict(value = "users", key = "#userId", cacheManager = "l1CacheManager")
    public void evictUser(String userId) {
        redisTemplate.delete("user:" + userId);
    }
}
```

### 6.2 Cache Warming and Preloading

```java
@Component
@Slf4j
public class CacheWarmingService {
    
    private final UserService userService;
    private final ProductService productService;
    
    @EventListener(ApplicationReadyEvent.class)
    public void warmCaches() {
        CompletableFuture.runAsync(this::warmUserCache)
            .thenRunAsync(this::warmProductCache)
            .thenRun(() -> log.info("Cache warming completed"))
            .exceptionally(throwable -> {
                log.error("Cache warming failed", throwable);
                return null;
            });
    }
    
    private void warmUserCache() {
        log.info("Starting user cache warming...");
        
        // Load frequently accessed users
        List<String> frequentUsers = userService.getFrequentlyAccessedUsers(1000);
        
        frequentUsers.parallelStream()
            .forEach(userId -> {
                try {
                    userService.getUser(userId); // This will populate cache
                } catch (Exception e) {
                    log.warn("Failed to warm cache for user: {}", userId, e);
                }
            });
            
        log.info("User cache warming completed for {} users", frequentUsers.size());
    }
    
    private void warmProductCache() {
        log.info("Starting product cache warming...");
        
        // Load hot products from last 24 hours
        LocalDateTime yesterday = LocalDateTime.now().minusDays(1);
        List<String> hotProducts = productService.getHotProducts(yesterday, 500);
        
        hotProducts.forEach(productId -> {
            try {
                productService.getProduct(productId);
            } catch (Exception e) {
                log.warn("Failed to warm cache for product: {}", productId, e);
            }
        });
        
        log.info("Product cache warming completed for {} products", hotProducts.size());
    }
    
    @Scheduled(cron = "0 0 2 * * *") // Daily at 2 AM
    public void scheduledCacheWarming() {
        log.info("Starting scheduled cache warming...");
        warmCaches();
    }
}
```

---

## 7. Network and I/O Optimization

### 7.1 Non-blocking I/O with WebFlux

```java
@RestController
@RequestMapping("/api/reactive")
public class ReactivePerformanceController {
    
    private final WebClient webClient;
    private final ReactiveUserService userService;
    
    // Non-blocking parallel calls
    @GetMapping("/user/{userId}/dashboard")
    public Mono<UserDashboard> getUserDashboard(@PathVariable String userId) {
        
        Mono<User> userMono = userService.findById(userId)
            .subscribeOn(Schedulers.boundedElastic()); // DB call
            
        Mono<List<Order>> ordersMono = getRecentOrders(userId)
            .subscribeOn(Schedulers.parallel()); // External service
            
        Mono<PaymentInfo> paymentMono = getPaymentInfo(userId)
            .subscribeOn(Schedulers.parallel()); // External service
        
        return Mono.zip(userMono, ordersMono, paymentMono)
            .map(tuple -> UserDashboard.builder()
                .user(tuple.getT1())
                .recentOrders(tuple.getT2())
                .paymentInfo(tuple.getT3())
                .build())
            .timeout(Duration.ofSeconds(5))
            .onErrorReturn(UserDashboard.empty());
    }
    
    private Mono<List<Order>> getRecentOrders(String userId) {
        return webClient.get()
            .uri("/orders/user/{userId}/recent", userId)
            .retrieve()
            .bodyToFlux(Order.class)
            .collectList()
            .onErrorReturn(Collections.emptyList());
    }
    
    // Streaming response for large datasets
    @GetMapping(value = "/orders/stream", produces = MediaType.APPLICATION_NDJSON_VALUE)
    public Flux<Order> streamOrders(@RequestParam String customerId) {
        return orderService.findOrdersReactive(customerId)
            .delayElements(Duration.ofMillis(100)) // Backpressure handling
            .onBackpressureBuffer(1000)
            .doOnNext(order -> log.debug("Streaming order: {}", order.getId()))
            .onErrorContinue((throwable, o) -> 
                log.error("Error streaming order: {}", o, throwable));
    }
}

// Reactive repository with R2DBC
@Repository
public interface ReactiveOrderRepository extends ReactiveCrudRepository<Order, String> {
    
    @Query("SELECT * FROM orders WHERE customer_id = :customerId ORDER BY created_at DESC")
    Flux<Order> findByCustomerIdOrderByCreatedAtDesc(@Param("customerId") String customerId);
    
    @Query("SELECT COUNT(*) FROM orders WHERE created_at >= :fromDate")
    Mono<Long> countOrdersSince(@Param("fromDate") LocalDateTime fromDate);
}
```

### 7.2 HTTP Client Optimization

```java
@Configuration
public class HttpClientConfiguration {
    
    @Bean
    public WebClient optimizedWebClient() {
        ConnectionProvider connectionProvider = ConnectionProvider.builder("optimized")
            .maxConnections(100)                    // Max total connections
            .maxIdleTime(Duration.ofSeconds(30))   // Idle connection timeout
            .maxLifeTime(Duration.ofMinutes(5))    // Connection max lifetime
            .pendingAcquireTimeout(Duration.ofSeconds(10)) // Wait time for connection
            .evictInBackground(Duration.ofSeconds(30))     // Background eviction
            .build();
            
        HttpClient httpClient = HttpClient.create(connectionProvider)
            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000) // 10s connect timeout
            .option(ChannelOption.TCP_NODELAY, true)             // Disable Nagle's algorithm
            .option(ChannelOption.SO_KEEPALIVE, true)            // Enable TCP keep-alive
            .responseTimeout(Duration.ofSeconds(30))             // Response timeout
            .compress(true)                                      // Enable compression
            .followRedirect(true)
            .wiretap(false);                                     // Disable for production
            
        return WebClient.builder()
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .codecs(configurer -> {
                configurer.defaultCodecs().maxInMemorySize(1024 * 1024); // 1MB buffer
                configurer.defaultCodecs().enableLoggingRequestDetails(false);
            })
            .build();
    }
    
    @Bean
    public RestTemplate optimizedRestTemplate() {
        CloseableHttpClient httpClient = HttpClients.custom()
            .setMaxConnTotal(100)
            .setMaxConnPerRoute(20)
            .setConnectionTimeToLive(5, TimeUnit.MINUTES)
            .setDefaultRequestConfig(RequestConfig.custom()
                .setConnectTimeout(10000)
                .setSocketTimeout(30000)
                .setConnectionRequestTimeout(10000)
                .build())
            .build();
            
        HttpComponentsClientHttpRequestFactory factory = 
            new HttpComponentsClientHttpRequestFactory(httpClient);
            
        RestTemplate restTemplate = new RestTemplate(factory);
        
        // Add interceptors for logging and metrics
        restTemplate.setInterceptors(List.of(
            new LoggingClientHttpRequestInterceptor(),
            new MetricsClientHttpRequestInterceptor()
        ));
        
        return restTemplate;
    }
}
```

---

## 8. Concurrency and Threading

### 8.1 Thread Pool Optimization

```java
@Configuration
public class ThreadingConfiguration {
    
    @Bean(name = "cpuIntensiveExecutor")
    public TaskExecutor cpuIntensiveExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(Runtime.getRuntime().availableProcessors());
        executor.setMaxPoolSize(Runtime.getRuntime().availableProcessors());
        executor.setQueueCapacity(100);
        executor.setKeepAliveSeconds(60);
        executor.setThreadNamePrefix("cpu-intensive-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.initialize();
        return executor;
    }
    
    @Bean(name = "ioIntensiveExecutor")
    public TaskExecutor ioIntensiveExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(20);  // Higher for I/O bound tasks
        executor.setMaxPoolSize(100);
        executor.setQueueCapacity(500);
        executor.setKeepAliveSeconds(60);
        executor.setThreadNamePrefix("io-intensive-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.AbortPolicy());
        executor.initialize();
        return executor;
    }
    
    @Bean(name = "virtualThreadExecutor")
    @ConditionalOnJavaVersion(JavaVersion.NINETEEN) // Java 19+
    public Executor virtualThreadExecutor() {
        return Executors.newVirtualThreadPerTaskExecutor();
    }
}

// Lock-free concurrent data structures
@Service
public class HighPerformanceConcurrentService {
    
    // Lock-free counter for metrics
    private final LongAdder requestCounter = new LongAdder();
    private final DoubleAdder responseTimeSum = new DoubleAdder();
    
    // Lock-free queue for async processing
    private final ConcurrentLinkedQueue<Task> taskQueue = new ConcurrentLinkedQueue<>();
    
    // Striped locks for fine-grained locking
    private final Striped<Lock> stripedLocks = Striped.lock(16);
    
    public void processRequest(Request request) {
        requestCounter.increment();
        
        long startTime = System.nanoTime();
        try {
            // Process request
            handleRequest(request);
        } finally {
            long endTime = System.nanoTime();
            responseTimeSum.add((endTime - startTime) / 1_000_000.0); // Convert to ms
        }
    }
    
    public void updateUserData(String userId, UserData data) {
        Lock lock = stripedLocks.get(userId);
        lock.lock();
        try {
            // Critical section for this specific user
            updateUserDataUnsafe(userId, data);
        } finally {
            lock.unlock();
        }
    }
    
    // Compare-and-swap for atomic updates
    private final AtomicReference<ConfigurationData> configRef = new AtomicReference<>();
    
    public boolean updateConfiguration(ConfigurationData newConfig) {
        ConfigurationData currentConfig = configRef.get();
        return configRef.compareAndSet(currentConfig, newConfig);
    }
    
    // Lock-free statistics collection
    public PerformanceStats getStats() {
        long requests = requestCounter.sum();
        double totalTime = responseTimeSum.sum();
        double averageTime = requests > 0 ? totalTime / requests : 0;
        
        return new PerformanceStats(requests, averageTime, taskQueue.size());
    }
}
```

### 8.2 Reactive Streams Performance

```java
@Service
public class ReactivePerformanceService {
    
    // Parallel processing with work stealing
    public Flux<ProcessedItem> processItemsParallel(Flux<Item> items) {
        return items
            .parallel(Runtime.getRuntime().availableProcessors())
            .runOn(Schedulers.parallel())
            .map(this::processItem)
            .sequential()
            .onBackpressureBuffer(1000)
            .publishOn(Schedulers.boundedElastic());
    }
    
    // Batching for efficient processing
    public Flux<List<Result>> processBatches(Flux<Item> items) {
        return items
            .buffer(100, Duration.ofSeconds(1)) // Batch by size or time
            .filter(batch -> !batch.isEmpty())
            .concatMap(this::processBatch, 2) // Limit concurrent batches
            .onErrorContinue((throwable, batch) -> 
                log.error("Error processing batch: {}", batch, throwable));
    }
    
    // Efficient error handling and retry
    public Mono<Result> processWithRetry(Item item) {
        return Mono.fromCallable(() -> processItem(item))
            .subscribeOn(Schedulers.boundedElastic())
            .retryWhen(Retry.backoff(3, Duration.ofMillis(100))
                .maxBackoff(Duration.ofSeconds(2))
                .filter(throwable -> throwable instanceof RetryableException))
            .timeout(Duration.ofSeconds(10))
            .onErrorResume(throwable -> {
                log.error("Processing failed for item: {}", item.getId(), throwable);
                return Mono.just(Result.failed(item.getId()));
            });
    }
    
    // Memory-efficient streaming with backpressure
    public Flux<Data> streamLargeDataset(String query) {
        return Flux.create(sink -> {
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = conn.prepareStatement(query)) {
                
                stmt.setFetchSize(1000); // Fetch in batches
                ResultSet rs = stmt.executeQuery();
                
                while (rs.next() && !sink.isCancelled()) {
                    Data data = mapResultSetToData(rs);
                    sink.next(data);
                    
                    // Backpressure handling
                    if (sink.requestedFromDownstream() == 0) {
                        Thread.sleep(10); // Brief pause if downstream is slow
                    }
                }
                
                sink.complete();
            } catch (Exception e) {
                sink.error(e);
            }
        }, FluxSink.OverflowStrategy.BUFFER)
        .publishOn(Schedulers.boundedElastic());
    }
}
```

---

## 9. Load Testing and Capacity Planning

### 9.1 Gatling Load Tests

```scala
// Gatling performance test scenario
import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._

class OrderServiceLoadTest extends Simulation {

  val httpProtocol = http
    .baseUrl("http://localhost:8080")
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")

  val headers = Map(
    "Authorization" -> "Bearer ${accessToken}",
    "Content-Type" -> "application/json"
  )

  // Scenarios
  val getOrderScenario = scenario("Get Order")
    .exec(
      http("Get Order")
        .get("/api/orders/${orderId}")
        .headers(headers)
        .check(status.is(200))
        .check(responseTimeInMillis.lt(500))
    )

  val createOrderScenario = scenario("Create Order")
    .exec(
      http("Create Order")
        .post("/api/orders")
        .headers(headers)
        .body(StringBody("""{"userId":"user123","items":[{"productId":"p1","quantity":2}],"total":99.99}"""))
        .check(status.is(201))
        .check(responseTimeInMillis.lt(1000))
        .check(jsonPath("$.orderId").saveAs("orderId"))
    )

  // Load test setup
  setUp(
    getOrderScenario
      .inject(
        nothingFor(5.seconds),
        rampUsersPerSec(1) to 50 during (2.minutes),
        constantUsersPerSec(50) during (5.minutes),
        rampUsersPerSec(50) to 100 during (2.minutes),
        constantUsersPerSec(100) during (5.minutes)
      )
      .andThen(
        createOrderScenario
          .inject(
            nothingFor(30.seconds),
            rampUsersPerSec(1) to 20 during (1.minute),
            constantUsersPerSec(20) during (10.minutes)
          )
      )
  )
  .protocols(httpProtocol)
  .assertions(
    global.responseTime.percentile3.lt(1000),
    global.responseTime.percentile4.lt(2000),
    global.successfulRequests.percent.gt(99)
  )
}
```

### 9.2 Capacity Planning Service

```java
@Service
@Slf4j
public class CapacityPlanningService {
    
    private final MeterRegistry meterRegistry;
    private final ApplicationMetrics applicationMetrics;
    
    // Predict capacity needs based on current metrics
    public CapacityPrediction predictCapacityNeeds(Duration forecastPeriod) {
        // Current metrics
        double currentRPS = getCurrentRequestsPerSecond();
        double currentCpuUsage = getCurrentCpuUsage();
        double currentMemoryUsage = getCurrentMemoryUsage();
        
        // Historical growth rate
        double growthRate = calculateGrowthRate(Duration.ofDays(30));
        
        // Predicted future load
        double predictedRPS = currentRPS * (1 + growthRate * forecastPeriod.toDays() / 365.0);
        
        // Calculate required resources
        int requiredInstances = calculateRequiredInstances(predictedRPS, currentCpuUsage, currentMemoryUsage);
        
        return CapacityPrediction.builder()
            .currentRPS(currentRPS)
            .predictedRPS(predictedRPS)
            .currentInstances(getCurrentInstanceCount())
            .requiredInstances(requiredInstances)
            .forecastPeriod(forecastPeriod)
            .confidence(calculateConfidence(growthRate))
            .build();
    }
    
    private double calculateGrowthRate(Duration period) {
        // Get historical metrics
        List<Double> historicalRPS = getHistoricalRPS(period);
        
        if (historicalRPS.size() < 2) {
            return 0.0;
        }
        
        // Simple linear regression for growth rate
        double n = historicalRPS.size();
        double sumX = n * (n - 1) / 2; // 0 + 1 + 2 + ... + (n-1)
        double sumY = historicalRPS.stream().mapToDouble(Double::doubleValue).sum();
        double sumXY = IntStream.range(0, historicalRPS.size())
            .mapToDouble(i -> i * historicalRPS.get(i))
            .sum();
        double sumX2 = n * (n - 1) * (2 * n - 1) / 6;
        
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double avgY = sumY / n;
        
        return avgY > 0 ? slope / avgY : 0.0; // Relative growth rate
    }
    
    private int calculateRequiredInstances(double predictedRPS, double currentCpuUsage, double currentMemoryUsage) {
        // Assume each instance can handle 100 RPS at 70% CPU/memory utilization
        double maxRPSPerInstance = 100 * (0.7 / Math.max(currentCpuUsage, currentMemoryUsage));
        int baseInstances = (int) Math.ceil(predictedRPS / maxRPSPerInstance);
        
        // Add 20% buffer for safety
        return (int) Math.ceil(baseInstances * 1.2);
    }
    
    @Scheduled(fixedRate = 300000) // Every 5 minutes
    public void recordCapacityMetrics() {
        CapacityPrediction prediction = predictCapacityNeeds(Duration.ofDays(30));
        
        // Record metrics for alerting
        Gauge.builder("capacity.prediction.required_instances")
            .register(meterRegistry, () -> prediction.getRequiredInstances());
            
        Gauge.builder("capacity.prediction.growth_rate")
            .register(meterRegistry, () -> prediction.getGrowthRate());
            
        if (prediction.getRequiredInstances() > getCurrentInstanceCount() * 1.5) {
            log.warn("Capacity scaling recommended: current={}, required={}", 
                getCurrentInstanceCount(), prediction.getRequiredInstances());
        }
    }
    
    // Auto-scaling integration
    public void triggerAutoScaling(CapacityPrediction prediction) {
        if (prediction.getConfidence() > 0.8 && 
            prediction.getRequiredInstances() > getCurrentInstanceCount()) {
            
            ScalingRequest request = ScalingRequest.builder()
                .targetInstances(prediction.getRequiredInstances())
                .reason("Predicted capacity shortage")
                .confidence(prediction.getConfidence())
                .build();
                
            autoScalingService.requestScaling(request);
        }
    }
}
```

---

## 10. Practical Examples

### 10.1 Performance Test Suite

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class PerformanceTestSuite {
    
    @Autowired
    private TestRestTemplate restTemplate;
    
    @LocalServerPort
    private int port;
    
    private String baseUrl;
    private ExecutorService executor;
    
    @BeforeAll
    void setUp() {
        baseUrl = "http://localhost:" + port;
        executor = Executors.newFixedThreadPool(50);
    }
    
    @AfterAll
    void tearDown() {
        executor.shutdown();
    }
    
    @Test
    void testConcurrentOrderCreation() throws InterruptedException {
        int numberOfRequests = 100;
        CountDownLatch latch = new CountDownLatch(numberOfRequests);
        List<CompletableFuture<ResponseEntity<String>>> futures = new ArrayList<>();
        
        long startTime = System.currentTimeMillis();
        
        for (int i = 0; i < numberOfRequests; i++) {
            final int requestId = i;
            CompletableFuture<ResponseEntity<String>> future = CompletableFuture.supplyAsync(() -> {
                try {
                    OrderRequest request = createOrderRequest(requestId);
                    return restTemplate.postForEntity(baseUrl + "/api/orders", request, String.class);
                } finally {
                    latch.countDown();
                }
            }, executor);
            futures.add(future);
        }
        
        latch.await(30, TimeUnit.SECONDS);
        long endTime = System.currentTimeMillis();
        
        // Analyze results
        int successfulRequests = 0;
        List<Long> responseTimes = new ArrayList<>();
        
        for (CompletableFuture<ResponseEntity<String>> future : futures) {
            try {
                ResponseEntity<String> response = future.get(1, TimeUnit.SECONDS);
                if (response.getStatusCode().is2xxSuccessful()) {
                    successfulRequests++;
                }
            } catch (Exception e) {
                log.error("Request failed", e);
            }
        }
        
        double successRate = (double) successfulRequests / numberOfRequests;
        long totalTime = endTime - startTime;
        double throughput = (double) numberOfRequests / (totalTime / 1000.0);
        
        log.info("Performance Test Results:");
        log.info("Total Requests: {}", numberOfRequests);
        log.info("Successful Requests: {}", successfulRequests);
        log.info("Success Rate: {:.2f}%", successRate * 100);
        log.info("Total Time: {}ms", totalTime);
        log.info("Throughput: {:.2f} requests/second", throughput);
        
        // Assertions
        assertThat(successRate).isGreaterThan(0.95); // 95% success rate
        assertThat(throughput).isGreaterThan(10); // At least 10 RPS
    }
    
    @Test
    void testMemoryUsageUnderLoad() {
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        
        long initialMemory = memoryBean.getHeapMemoryUsage().getUsed();
        
        // Generate load
        for (int i = 0; i < 1000; i++) {
            restTemplate.getForEntity(baseUrl + "/api/orders/" + UUID.randomUUID(), String.class);
        }
        
        // Force GC and measure
        System.gc();
        System.gc(); // Call twice for better reliability
        
        try {
            Thread.sleep(1000); // Wait for GC to complete
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        long finalMemory = memoryBean.getHeapMemoryUsage().getUsed();
        long memoryIncrease = finalMemory - initialMemory;
        
        log.info("Memory usage: initial={}MB, final={}MB, increase={}MB",
            initialMemory / (1024 * 1024),
            finalMemory / (1024 * 1024),
            memoryIncrease / (1024 * 1024));
        
        // Memory increase should be reasonable
        assertThat(memoryIncrease).isLessThan(100 * 1024 * 1024); // Less than 100MB increase
    }
    
    @Test
    void testDatabaseConnectionLeaks() {
        HikariDataSource dataSource = (HikariDataSource) applicationContext.getBean(DataSource.class);
        HikariPoolMXBean poolBean = dataSource.getHikariPoolMXBean();
        
        int initialActiveConnections = poolBean.getActiveConnections();
        
        // Generate database load
        for (int i = 0; i < 50; i++) {
            restTemplate.getForEntity(baseUrl + "/api/users/" + (i % 10), String.class);
        }
        
        // Wait for connections to be returned
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        int finalActiveConnections = poolBean.getActiveConnections();
        
        log.info("Database connections: initial={}, final={}", 
            initialActiveConnections, finalActiveConnections);
        
        // Connections should return to pool
        assertThat(finalActiveConnections).isLessThanOrEqualTo(initialActiveConnections + 2);
    }
}
```

### 10.2 Performance Monitoring Dashboard

```java
@RestController
@RequestMapping("/admin/performance")
public class PerformanceMonitoringController {
    
    private final MeterRegistry meterRegistry;
    private final MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
    private final List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();
    
    @GetMapping("/metrics")
    public ResponseEntity<PerformanceMetrics> getCurrentMetrics() {
        return ResponseEntity.ok(PerformanceMetrics.builder()
            .timestamp(Instant.now())
            .jvmMetrics(getJVMMetrics())
            .applicationMetrics(getApplicationMetrics())
            .systemMetrics(getSystemMetrics())
            .build());
    }
    
    private JVMMetrics getJVMMetrics() {
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        MemoryUsage nonHeapUsage = memoryBean.getNonHeapMemoryUsage();
        
        List<GCMetrics> gcMetrics = gcBeans.stream()
            .map(bean -> GCMetrics.builder()
                .name(bean.getName())
                .collectionCount(bean.getCollectionCount())
                .collectionTime(bean.getCollectionTime())
                .build())
            .collect(Collectors.toList());
            
        return JVMMetrics.builder()
            .heapUsed(heapUsage.getUsed())
            .heapMax(heapUsage.getMax())
            .heapUtilization((double) heapUsage.getUsed() / heapUsage.getMax())
            .nonHeapUsed(nonHeapUsage.getUsed())
            .uptime(ManagementFactory.getRuntimeMXBean().getUptime())
            .gcMetrics(gcMetrics)
            .build();
    }
    
    private ApplicationMetrics getApplicationMetrics() {
        Timer requestTimer = Timer.builder("http_requests_duration")
            .register(meterRegistry);
            
        return ApplicationMetrics.builder()
            .requestCount(requestTimer.count())
            .averageResponseTime(requestTimer.mean(TimeUnit.MILLISECONDS))
            .p95ResponseTime(requestTimer.percentile(0.95, TimeUnit.MILLISECONDS))
            .p99ResponseTime(requestTimer.percentile(0.99, TimeUnit.MILLISECONDS))
            .build();
    }
    
    @GetMapping("/health-check")
    public ResponseEntity<HealthStatus> performHealthCheck() {
        List<ComponentHealth> components = List.of(
            checkDatabaseHealth(),
            checkCacheHealth(),
            checkExternalServicesHealth()
        );
        
        boolean allHealthy = components.stream().allMatch(ComponentHealth::isHealthy);
        
        return ResponseEntity.ok(HealthStatus.builder()
            .overall(allHealthy ? "HEALTHY" : "DEGRADED")
            .timestamp(Instant.now())
            .components(components)
            .build());
    }
    
    private ComponentHealth checkDatabaseHealth() {
        try {
            dataSource.getConnection().close();
            return ComponentHealth.builder()
                .name("database")
                .healthy(true)
                .responseTime(measureDatabaseResponseTime())
                .build();
        } catch (Exception e) {
            return ComponentHealth.builder()
                .name("database")
                .healthy(false)
                .error(e.getMessage())
                .build();
        }
    }
}
```

---

## Summary

| Topic | Key Performance Principles |
|-------|----------------------------|
| **JVM Tuning** | Right-size heap, choose appropriate GC, monitor GC pressure |
| **Profiling** | Profile continuously, focus on hot spots, measure don't guess |
| **Memory** | Minimize object creation, use pools, avoid memory leaks |
| **Database** | Connection pooling, query optimization, read replicas |
| **Caching** | Multi-level caching, cache warming, proper eviction |
| **Concurrency** | Right-size thread pools, use lock-free structures |
| **Load Testing** | Test under realistic load, measure P95/P99, capacity planning |

---

## Further Reading

- *Java Performance* — Scott Oaks
- *Systems Performance* — Brendan Gregg
- GC Tuning Guide: https://docs.oracle.com/en/java/javase/11/gctuning/
- JProfiler Documentation: https://www.ej-technologies.com/products/jprofiler/overview.html