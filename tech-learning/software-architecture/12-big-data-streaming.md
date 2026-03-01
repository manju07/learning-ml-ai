# Big Data & Streaming Platforms: Apache Pulsar & Spark Guide

## Table of Contents
1. [Introduction](#1-introduction)
2. [Apache Pulsar Deep Dive](#2-apache-pulsar-deep-dive)
3. [Apache Spark Deep Dive](#3-apache-spark-deep-dive)
4. [Pulsar vs Kafka Comparison](#4-pulsar-vs-kafka-comparison)
5. [Spark Architecture and Optimization](#5-spark-architecture-and-optimization)
6. [Real-time Stream Processing](#6-real-time-stream-processing)
7. [Data Pipeline Patterns](#7-data-pipeline-patterns)
8. [Production Deployment](#8-production-deployment)
9. [Monitoring and Operations](#9-monitoring-and-operations)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction

### 1.1 Apache Pulsar Overview

**Apache Pulsar** is a distributed messaging and streaming platform with multi-tenancy, geo-replication, and serverless functions built-in.

**Key Features:**
- **Multi-tenancy**: Native namespace isolation
- **Geo-replication**: Cross-datacenter message replication
- **Tiered storage**: Automatic data archiving to S3/GCS
- **Schema management**: Avro, JSON, Protobuf support
- **Functions**: Serverless stream processing

### 1.2 Apache Spark Overview

**Apache Spark** is a unified analytics engine for large-scale data processing with built-in modules for streaming, SQL, machine learning, and graph processing.

**Key Features:**
- **Speed**: 100x faster than Hadoop MapReduce
- **Ease of use**: APIs in Java, Scala, Python, R, SQL
- **Generality**: SQL, streaming, ML, graph processing
- **Runs everywhere**: Hadoop, Kubernetes, standalone

### 1.3 Use Cases Comparison

| Use Case | Apache Pulsar | Apache Spark |
|----------|---------------|--------------|
| **Real-time messaging** | ✅ Primary use case | ❌ Not designed for this |
| **Stream processing** | ✅ Pulsar Functions | ✅ Structured Streaming |
| **Batch processing** | ❌ Limited support | ✅ Primary strength |
| **Event sourcing** | ✅ Excellent fit | ❌ Not ideal |
| **Data analytics** | ❌ Basic capabilities | ✅ Advanced analytics |
| **Machine learning** | ❌ No ML support | ✅ MLlib included |

---

## 2. Apache Pulsar Deep Dive

### 2.1 Architecture Components

```
Pulsar Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Producer      │    │   Producer      │    │   Producer      │
└─────┬───────────┘    └─────┬───────────┘    └─────┬───────────┘
      │                      │                      │
      └──────────────────────┼──────────────────────┘
                             │
              ┌───────────────▼────────────────┐
              │        Pulsar Brokers          │
              │  (Stateless serving layer)     │
              └───────────────┬────────────────┘
                             │
              ┌───────────────▼────────────────┐
              │      Apache BookKeeper         │
              │   (Persistent storage layer)   │
              └───────────────┬────────────────┘
                             │
              ┌───────────────▼────────────────┐
              │       ZooKeeper/etcd          │
              │    (Metadata coordination)    │
              └────────────────────────────────┘
```

### 2.2 Java Producer Implementation

```java
@Configuration
@EnableConfigurationProperties(PulsarProperties.class)
public class PulsarConfiguration {
    
    @Bean
    public PulsarClient pulsarClient(PulsarProperties properties) {
        try {
            return PulsarClient.builder()
                .serviceUrl(properties.getServiceUrl())
                .authentication(AuthenticationFactory.token(properties.getToken()))
                .connectionTimeout(30, TimeUnit.SECONDS)
                .operationTimeout(30, TimeUnit.SECONDS)
                .maxConcurrentLookupRequests(50000)
                .maxLookupRequests(100000)
                .build();
        } catch (PulsarClientException e) {
            throw new RuntimeException("Failed to create Pulsar client", e);
        }
    }
    
    @Bean
    public ProducerBuilder<byte[]> producerBuilder(PulsarClient pulsarClient) {
        return pulsarClient.newProducer()
            .compressionType(CompressionType.LZ4)
            .batchingMaxMessages(1000)
            .batchingMaxPublishDelay(100, TimeUnit.MILLISECONDS)
            .blockIfQueueFull(true)
            .maxPendingMessages(10000)
            .sendTimeout(30, TimeUnit.SECONDS);
    }
}

@Service
@Slf4j
public class PulsarProducerService {
    
    private final PulsarClient pulsarClient;
    private final Map<String, Producer<byte[]>> producers = new ConcurrentHashMap<>();
    private final ObjectMapper objectMapper;
    
    public CompletableFuture<MessageId> sendMessage(String topic, Object message) {
        return sendMessage(topic, message, null);
    }
    
    public CompletableFuture<MessageId> sendMessage(String topic, Object message, String key) {
        try {
            Producer<byte[]> producer = getOrCreateProducer(topic);
            byte[] payload = objectMapper.writeValueAsBytes(message);
            
            TypedMessageBuilder<byte[]> messageBuilder = producer.newMessage()
                .value(payload)
                .property("timestamp", String.valueOf(System.currentTimeMillis()))
                .property("source", "order-service");
                
            if (key != null) {
                messageBuilder.key(key);
            }
            
            return messageBuilder.sendAsync()
                .whenComplete((messageId, throwable) -> {
                    if (throwable != null) {
                        log.error("Failed to send message to topic: {}", topic, throwable);
                    } else {
                        log.debug("Message sent successfully to topic: {}, messageId: {}", 
                            topic, messageId);
                    }
                });
                
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }
    
    private Producer<byte[]> getOrCreateProducer(String topic) throws PulsarClientException {
        return producers.computeIfAbsent(topic, t -> {
            try {
                return pulsarClient.newProducer()
                    .topic(t)
                    .compressionType(CompressionType.LZ4)
                    .batchingMaxMessages(1000)
                    .batchingMaxPublishDelay(100, TimeUnit.MILLISECONDS)
                    .create();
            } catch (PulsarClientException e) {
                throw new RuntimeException("Failed to create producer for topic: " + t, e);
            }
        });
    }
    
    // Schema-aware producer for structured data
    public <T> CompletableFuture<MessageId> sendTypedMessage(String topic, T message, Schema<T> schema) {
        try {
            Producer<T> typedProducer = getOrCreateTypedProducer(topic, schema);
            
            return typedProducer.newMessage()
                .value(message)
                .property("schema-version", schema.getSchemaInfo().getSchema())
                .sendAsync();
                
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }
    
    @PreDestroy
    public void cleanup() {
        producers.values().forEach(producer -> {
            try {
                producer.close();
            } catch (PulsarClientException e) {
                log.error("Error closing producer", e);
            }
        });
        
        try {
            pulsarClient.close();
        } catch (PulsarClientException e) {
            log.error("Error closing Pulsar client", e);
        }
    }
}
```

### 2.3 Consumer Implementation with Spring Boot

```java
@Component
@Slf4j
public class PulsarConsumerService {
    
    private final PulsarClient pulsarClient;
    private final ObjectMapper objectMapper;
    private final Map<String, Consumer<byte[]>> consumers = new ConcurrentHashMap<>();
    
    @PostConstruct
    public void initializeConsumers() {
        // Order events consumer
        subscribeToTopic("persistent://public/default/order-events", 
            "order-processor-subscription", this::processOrderEvent);
            
        // User events consumer with retry policy
        ConsumerBuilder<byte[]> retryConsumerBuilder = pulsarClient.newConsumer()
            .subscriptionType(SubscriptionType.Shared)
            .ackTimeout(30, TimeUnit.SECONDS)
            .negativeAckRedeliveryDelay(60, TimeUnit.SECONDS)
            .deadLetterPolicy(DeadLetterPolicy.builder()
                .maxRedeliverCount(3)
                .deadLetterTopic("user-events-dlq")
                .build());
                
        subscribeToTopic("persistent://public/default/user-events",
            "user-processor-subscription", this::processUserEvent, retryConsumerBuilder);
    }
    
    private void subscribeToTopic(String topic, String subscription, 
                                 MessageProcessor processor) {
        subscribeToTopic(topic, subscription, processor, null);
    }
    
    private void subscribeToTopic(String topic, String subscription, 
                                 MessageProcessor processor,
                                 ConsumerBuilder<byte[]> customBuilder) {
        try {
            ConsumerBuilder<byte[]> builder = customBuilder != null ? 
                customBuilder : pulsarClient.newConsumer();
                
            Consumer<byte[]> consumer = builder
                .topic(topic)
                .subscriptionName(subscription)
                .subscriptionType(SubscriptionType.Shared)
                .messageListener((consumer1, msg) -> {
                    try {
                        processor.process(msg);
                        consumer1.acknowledge(msg);
                    } catch (Exception e) {
                        log.error("Error processing message from topic: {}", topic, e);
                        consumer1.negativeAcknowledge(msg);
                    }
                })
                .subscribe();
                
            consumers.put(topic, consumer);
            log.info("Subscribed to topic: {} with subscription: {}", topic, subscription);
            
        } catch (PulsarClientException e) {
            throw new RuntimeException("Failed to subscribe to topic: " + topic, e);
        }
    }
    
    private void processOrderEvent(Message<byte[]> message) throws JsonProcessingException {
        OrderEvent event = objectMapper.readValue(message.getValue(), OrderEvent.class);
        
        log.info("Processing order event: orderId={}, eventType={}", 
            event.getOrderId(), event.getEventType());
            
        switch (event.getEventType()) {
            case ORDER_CREATED:
                handleOrderCreated(event);
                break;
            case ORDER_UPDATED:
                handleOrderUpdated(event);
                break;
            case ORDER_CANCELLED:
                handleOrderCancelled(event);
                break;
            default:
                log.warn("Unknown event type: {}", event.getEventType());
        }
    }
    
    private void processUserEvent(Message<byte[]> message) throws JsonProcessingException {
        UserEvent event = objectMapper.readValue(message.getValue(), UserEvent.class);
        
        // Implement idempotency using message ID
        String messageId = message.getMessageId().toString();
        if (isDuplicate(messageId)) {
            log.debug("Duplicate message detected, skipping: {}", messageId);
            return;
        }
        
        handleUserEvent(event);
        markAsProcessed(messageId);
    }
    
    @FunctionalInterface
    private interface MessageProcessor {
        void process(Message<byte[]> message) throws Exception;
    }
}
```

### 2.4 Multi-Tenant Topic Management

```java
@Service
@Slf4j
public class PulsarTopicManagementService {
    
    private final PulsarAdmin pulsarAdmin;
    
    public PulsarTopicManagementService(PulsarProperties properties) {
        try {
            this.pulsarAdmin = PulsarAdmin.builder()
                .serviceHttpUrl(properties.getAdminUrl())
                .authentication(AuthenticationFactory.token(properties.getToken()))
                .build();
        } catch (PulsarClientException e) {
            throw new RuntimeException("Failed to create Pulsar admin client", e);
        }
    }
    
    // Create tenant with resource quotas
    public void createTenant(String tenantName, Set<String> adminRoles, Set<String> clusters) {
        try {
            TenantInfoImpl tenantInfo = new TenantInfoImpl();
            tenantInfo.setAdminRoles(adminRoles);
            tenantInfo.setAllowedClusters(clusters);
            
            pulsarAdmin.tenants().createTenant(tenantName, tenantInfo);
            log.info("Created tenant: {}", tenantName);
            
            // Set resource quotas
            ResourceQuota quota = new ResourceQuota();
            quota.setMsgRateIn(1000.0);  // 1000 messages/sec
            quota.setMsgRateOut(2000.0); // 2000 messages/sec
            quota.setBandwidthIn(1024 * 1024); // 1MB/sec
            quota.setBandwidthOut(2048 * 1024); // 2MB/sec
            quota.setMemory(100 * 1024 * 1024); // 100MB
            quota.setDynamic(true);
            
            pulsarAdmin.resourceQuotas().setNamespaceBundleResourceQuota(
                tenantName, "default", "0x00000000_0xffffffff", quota);
                
        } catch (PulsarAdminException e) {
            throw new RuntimeException("Failed to create tenant: " + tenantName, e);
        }
    }
    
    // Create namespace with policies
    public void createNamespace(String tenant, String namespace) {
        try {
            String namespaceName = tenant + "/" + namespace;
            pulsarAdmin.namespaces().createNamespace(namespaceName);
            
            // Set retention policy
            RetentionPolicies retentionPolicy = new RetentionPolicies(
                TimeUnit.DAYS.toMinutes(7), // 7 days
                1024 // 1GB
            );
            pulsarAdmin.namespaces().setRetention(namespaceName, retentionPolicy);
            
            // Set message TTL
            pulsarAdmin.namespaces().setNamespaceMessageTTL(namespaceName, 
                (int) TimeUnit.DAYS.toSeconds(30)); // 30 days
                
            // Enable schema validation
            pulsarAdmin.namespaces().setSchemaValidationEnforced(namespaceName, true);
            
            log.info("Created namespace: {}", namespaceName);
            
        } catch (PulsarAdminException e) {
            throw new RuntimeException("Failed to create namespace: " + namespace, e);
        }
    }
    
    // Configure topic with schema
    public void createTopicWithSchema(String topicName, Schema<?> schema) {
        try {
            // Create partitioned topic
            pulsarAdmin.topics().createPartitionedTopic(topicName, 4);
            
            // Register schema
            pulsarAdmin.schemas().createSchema(topicName, schema.getSchemaInfo());
            
            // Set message deduplication
            pulsarAdmin.topics().enableDeduplication(topicName, true);
            
            log.info("Created topic with schema: {}", topicName);
            
        } catch (PulsarAdminException e) {
            throw new RuntimeException("Failed to create topic: " + topicName, e);
        }
    }
    
    // Monitor topic statistics
    @Scheduled(fixedRate = 60000) // Every minute
    public void monitorTopics() {
        try {
            List<String> topics = pulsarAdmin.topics()
                .getPartitionedTopicList("public/default");
                
            for (String topic : topics) {
                TopicStats stats = pulsarAdmin.topics().getStats(topic);
                
                log.info("Topic: {}, MsgRateIn: {}, MsgRateOut: {}, Storage: {} bytes",
                    topic, 
                    stats.getMsgRateIn(),
                    stats.getMsgRateOut(),
                    stats.getStorageSize());
                    
                // Alert if message rate is too high
                if (stats.getMsgRateIn() > 10000) {
                    log.warn("High message rate detected for topic: {}, rate: {}", 
                        topic, stats.getMsgRateIn());
                }
            }
        } catch (PulsarAdminException e) {
            log.error("Error monitoring topics", e);
        }
    }
}
```

---

## 3. Apache Spark Deep Dive

### 3.1 Spark Architecture Components

```
Spark Architecture:
┌─────────────────────────────────────────────────────────┐
│                  Driver Program                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ SparkContext│    │ SQL Context │    │ML Context   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘ │
└─────────────────┬───────────────────────────────────────┘
                  │
          ┌───────▼────────┐
          │ Cluster Manager │ (YARN/K8s/Standalone)
          └───────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐     ┌───▼───┐     ┌───▼───┐
│Worker │     │Worker │     │Worker │
│ ┌───┐ │     │ ┌───┐ │     │ ┌───┐ │
│ │Exc│ │     │ │Exc│ │     │ │Exc│ │
│ └───┘ │     │ └───┘ │     │ └───┘ │
└───────┘     └───────┘     └───────┘
```

### 3.2 Spark Session Configuration

```java
@Configuration
public class SparkConfiguration {
    
    @Bean
    public SparkSession sparkSession() {
        return SparkSession.builder()
            .appName("Enterprise Data Processing")
            .master("local[*]") // Use all available cores locally
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryo.registrationRequired", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.iceberg.type", "hive")
            .getOrCreate();
    }
}

@Service
@Slf4j
public class SparkDataProcessingService {
    
    private final SparkSession spark;
    
    // Batch processing with optimizations
    public void processOrderData(String inputPath, String outputPath) {
        Dataset<Row> orders = spark.read()
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(inputPath)
            .cache(); // Cache frequently accessed data
            
        // Data quality checks
        Dataset<Row> validOrders = orders
            .filter(col("order_id").isNotNull())
            .filter(col("total").gt(0))
            .filter(col("customer_id").isNotNull());
            
        // Aggregations with proper partitioning
        Dataset<Row> customerSummary = validOrders
            .groupBy("customer_id")
            .agg(
                count("order_id").alias("order_count"),
                sum("total").alias("total_spent"),
                avg("total").alias("avg_order_value"),
                max("order_date").alias("last_order_date")
            )
            .repartition(200, col("customer_id")) // Optimal partitioning
            .cache();
            
        // Write with partitioning for better query performance
        customerSummary.write()
            .mode(SaveMode.Overwrite)
            .partitionBy("last_order_date")
            .option("compression", "snappy")
            .parquet(outputPath);
            
        log.info("Processed {} orders for {} customers", 
            orders.count(), customerSummary.count());
    }
    
    // Structured Streaming with Pulsar
    public StreamingQuery processOrderStream() {
        Dataset<Row> pulsarStream = spark.readStream()
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("admin.url", "http://localhost:8080")
            .option("topic", "persistent://public/default/order-events")
            .load();
            
        // Parse Pulsar message
        Dataset<Row> parsedOrders = pulsarStream
            .select(from_json(col("value").cast("string"), getOrderSchema()).alias("order"))
            .select("order.*")
            .withColumn("processing_time", current_timestamp());
            
        // Windowed aggregations
        Dataset<Row> windowedAgg = parsedOrders
            .withWatermark("order_timestamp", "10 minutes")
            .groupBy(
                window(col("order_timestamp"), "5 minutes"),
                col("product_category")
            )
            .agg(
                count("*").alias("order_count"),
                sum("total").alias("total_sales"),
                countDistinct("customer_id").alias("unique_customers")
            );
            
        return windowedAgg.writeStream()
            .outputMode("append")
            .format("delta") // Use Delta Lake for ACID transactions
            .option("checkpointLocation", "/tmp/checkpoints/order-aggregations")
            .option("path", "/data/order-aggregations")
            .trigger(Trigger.ProcessingTime("30 seconds"))
            .start();
    }
    
    private StructType getOrderSchema() {
        return new StructType(new StructField[]{
            new StructField("order_id", DataTypes.StringType, false, null),
            new StructField("customer_id", DataTypes.StringType, false, null),
            new StructField("product_category", DataTypes.StringType, true, null),
            new StructField("total", DataTypes.DoubleType, false, null),
            new StructField("order_timestamp", DataTypes.TimestampType, false, null)
        });
    }
}
```

### 3.3 Advanced Spark Optimizations

```java
@Service
public class SparkOptimizationService {
    
    private final SparkSession spark;
    
    // Dynamic partition pruning and predicate pushdown
    public Dataset<Row> optimizedJoin(String ordersPath, String customersPath) {
        Dataset<Row> orders = spark.read().parquet(ordersPath)
            .filter(col("order_date").gt(lit("2023-01-01"))) // Predicate pushdown
            .repartition(col("customer_id")); // Optimize for join
            
        Dataset<Row> customers = spark.read().parquet(customersPath)
            .filter(col("status").equalTo("ACTIVE"))
            .repartition(col("customer_id"));
            
        // Broadcast join for small dimension tables
        Dataset<Row> result;
        if (customers.count() < 10000) {
            result = orders.join(broadcast(customers), "customer_id");
        } else {
            // Sort-merge join for large tables
            result = orders.join(customers, "customer_id");
        }
        
        return result
            .select("order_id", "customer_name", "total", "order_date")
            .coalesce(100); // Reduce small files
    }
    
    // Custom partitioner for skewed data
    public void handleSkewedData(Dataset<Row> skewedData, String outputPath) {
        // Detect skew
        Dataset<Row> keyStats = skewedData
            .groupBy("customer_id")
            .count()
            .orderBy(desc("count"));
            
        List<Row> topKeys = keyStats.limit(10).collectAsList();
        
        // Salting technique for hot keys
        Dataset<Row> saltedData = skewedData
            .withColumn("salt", 
                when(col("customer_id").isin(topKeys.stream()
                    .map(row -> row.getString(0))
                    .toArray(String[]::new)), 
                    concat(col("customer_id"), lit("_"), 
                           (rand().multiply(100)).cast("int")))
                .otherwise(col("customer_id")))
            .repartition(col("salt"));
            
        saltedData.write()
            .mode(SaveMode.Overwrite)
            .option("compression", "snappy")
            .parquet(outputPath);
    }
    
    // Memory management and spill optimization
    public void configureMemoryOptimization() {
        SparkConf conf = spark.sparkContext().conf();
        
        // Memory management
        conf.set("spark.executor.memory", "8g");
        conf.set("spark.executor.memoryFraction", "0.8");
        conf.set("spark.storage.memoryFraction", "0.5");
        conf.set("spark.shuffle.memoryFraction", "0.3");
        
        // Serialization optimization  
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        conf.set("spark.kryo.unsafe", "true");
        conf.set("spark.kryo.registrator", "com.example.MyKryoRegistrator");
        
        // Shuffle optimization
        conf.set("spark.sql.adaptive.shuffle.targetPostShuffleInputSize", "134217728"); // 128MB
        conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "134217728");
        conf.set("spark.shuffle.compress", "true");
        conf.set("spark.shuffle.spill.compress", "true");
        
        log.info("Applied memory optimizations to Spark configuration");
    }
}

// Custom Kryo registrator for performance
public class MyKryoRegistrator implements KryoRegistrator {
    
    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(OrderEvent.class);
        kryo.register(CustomerData.class);
        kryo.register(ProductInfo.class);
        kryo.register(java.sql.Timestamp.class, new SqlTimestampSerializer());
        kryo.register(scala.collection.mutable.WrappedArray.ofRef.class);
        
        // Register commonly used Spark internal classes
        kryo.register(org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema.class);
        kryo.register(org.apache.spark.sql.types.StructType.class);
    }
}
```

---

## 4. Pulsar vs Kafka Comparison

### 4.1 Detailed Feature Comparison

| Feature | Apache Pulsar | Apache Kafka |
|---------|---------------|--------------|
| **Architecture** | Layered (Broker + BookKeeper) | Monolithic brokers |
| **Storage** | Segment-based, tiered | Log segments on brokers |
| **Multi-tenancy** | Native support | Manual configuration |
| **Geo-replication** | Built-in, automatic | Manual setup (MirrorMaker) |
| **Message ordering** | Per-key within partition | Per-partition |
| **Schema evolution** | Built-in schema registry | Confluent Schema Registry |
| **Functions** | Built-in serverless | Kafka Streams separate |
| **Cloud-native** | Kubernetes-native | Requires additional setup |

### 4.2 Migration from Kafka to Pulsar

```java
@Service
@Slf4j
public class KafkaToPulsarMigrationService {
    
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final PulsarProducerService pulsarProducer;
    
    // Dual-write pattern for zero-downtime migration
    public void dualWriteMessage(String topic, String key, Object message) {
        CompletableFuture<Void> kafkaFuture = CompletableFuture.runAsync(() -> {
            try {
                kafkaTemplate.send(topic, key, message).get();
            } catch (Exception e) {
                log.error("Failed to send to Kafka", e);
                throw new RuntimeException(e);
            }
        });
        
        CompletableFuture<Void> pulsarFuture = pulsarProducer
            .sendMessage("persistent://public/default/" + topic, message, key)
            .thenApply(messageId -> null);
            
        // Wait for both to complete
        CompletableFuture.allOf(kafkaFuture, pulsarFuture)
            .whenComplete((result, throwable) -> {
                if (throwable != null) {
                    log.error("Dual write failed", throwable);
                } else {
                    log.debug("Dual write successful for key: {}", key);
                }
            });
    }
    
    // Message transformation during migration
    public void migrateTopicData(String kafkaTopic, String pulsarTopic) {
        try (KafkaConsumer<String, String> consumer = createKafkaConsumer()) {
            consumer.subscribe(Collections.singletonList(kafkaTopic));
            
            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(1000));
                
                for (ConsumerRecord<String, String> record : records) {
                    // Transform message if needed
                    Object transformedMessage = transformMessage(record.value());
                    
                    // Send to Pulsar
                    pulsarProducer.sendMessage(pulsarTopic, transformedMessage, record.key())
                        .whenComplete((messageId, throwable) -> {
                            if (throwable != null) {
                                log.error("Failed to migrate message: {}", record.key(), throwable);
                            }
                        });
                }
                
                consumer.commitSync(); // Commit after successful processing
            }
        }
    }
    
    private Object transformMessage(String kafkaMessage) {
        // Implement message transformation logic
        // e.g., convert Kafka message format to Pulsar format
        return kafkaMessage; // Simplified
    }
}
```

---

## 5. Spark Architecture and Optimization

### 5.1 Memory Management and GC Tuning

```java
@Component
public class SparkTuningConfiguration {
    
    @PostConstruct
    public void configureSparkOptimizations() {
        SparkConf conf = new SparkConf()
            .setAppName("OptimizedSparkApp")
            
            // Memory Configuration
            .set("spark.executor.memory", "8g")
            .set("spark.driver.memory", "4g")
            .set("spark.executor.memoryOffHeap.enabled", "true")
            .set("spark.executor.memoryOffHeap.size", "2g")
            
            // GC Tuning for Java 11+
            .set("spark.executor.extraJavaOptions", 
                "-XX:+UseG1GC " +
                "-XX:MaxGCPauseMillis=200 " +
                "-XX:G1HeapRegionSize=16m " +
                "-XX:+UseStringDeduplication " +
                "-XX:+PrintGCDetails " +
                "-XX:+PrintGCTimeStamps")
            
            // Serialization
            .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .set("spark.kryo.registrationRequired", "true")
            .set("spark.kryo.registrator", "com.example.MyKryoRegistrator")
            
            // Network and Shuffle
            .set("spark.network.timeout", "300s")
            .set("spark.shuffle.compress", "true")
            .set("spark.shuffle.spill.compress", "true")
            .set("spark.io.compression.codec", "snappy")
            
            // Dynamic allocation
            .set("spark.dynamicAllocation.enabled", "true")
            .set("spark.dynamicAllocation.minExecutors", "2")
            .set("spark.dynamicAllocation.maxExecutors", "20")
            .set("spark.dynamicAllocation.initialExecutors", "5");
    }
}

@Service
@Slf4j
public class SparkPerformanceMonitoringService {
    
    private final SparkSession spark;
    private final MeterRegistry meterRegistry;
    
    @Scheduled(fixedRate = 30000) // Every 30 seconds
    public void monitorSparkMetrics() {
        SparkContext sc = spark.sparkContext();
        
        // Application metrics
        Gauge.builder("spark.active.jobs")
            .register(meterRegistry, () -> sc.statusTracker().getActiveJobsIds().length);
            
        Gauge.builder("spark.active.stages")
            .register(meterRegistry, () -> sc.statusTracker().getActiveStageIds().length);
            
        // Executor metrics
        SparkStatusTracker statusTracker = sc.statusTracker();
        SparkExecutorInfo[] executors = statusTracker.getExecutorInfos();
        
        long totalCores = Arrays.stream(executors)
            .mapToLong(SparkExecutorInfo::totalCores)
            .sum();
            
        long totalMemory = Arrays.stream(executors)
            .mapToLong(SparkExecutorInfo::maxMemory)
            .sum();
            
        Gauge.builder("spark.executors.total.cores")
            .register(meterRegistry, () -> totalCores);
            
        Gauge.builder("spark.executors.total.memory")
            .register(meterRegistry, () -> totalMemory);
            
        // Check for failed jobs/stages
        int failedJobs = sc.statusTracker().getJobIdsForGroup(null).length - 
            sc.statusTracker().getActiveJobsIds().length;
            
        if (failedJobs > 0) {
            log.warn("Detected {} failed Spark jobs", failedJobs);
        }
    }
}
```

### 5.2 Advanced Data Processing Patterns

```java
@Service
public class AdvancedSparkProcessingService {
    
    private final SparkSession spark;
    
    // Incremental processing with Delta Lake
    public void processIncrementalData(String deltaTablePath, String checkpointPath) {
        DeltaTable deltaTable = DeltaTable.forPath(spark, deltaTablePath);
        
        // Read new data since last checkpoint
        Dataset<Row> newData = spark.read()
            .option("basePath", checkpointPath)
            .format("delta")
            .load()
            .where("_change_type = 'insert'");
            
        // Merge with ACID guarantees
        deltaTable.as("target")
            .merge(newData.as("source"), "target.id = source.id")
            .whenMatched("source.updated_at > target.updated_at")
                .updateAll()
            .whenNotMatched()
                .insertAll()
            .execute();
            
        // Optimize table (compaction and Z-ordering)
        deltaTable.optimize()
            .where("date >= current_date() - INTERVAL 30 DAYS")
            .executeCompaction();
            
        // Z-order for better query performance
        deltaTable.optimize()
            .where("date >= current_date() - INTERVAL 7 DAYS")
            .executeZOrderBy("customer_id", "product_category");
    }
    
    // Complex event processing with stateful operations
    public StreamingQuery processComplexEvents() {
        Dataset<Row> eventStream = spark.readStream()
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("topic", "persistent://public/default/user-events")
            .load();
            
        Dataset<Row> parsedEvents = eventStream
            .select(from_json(col("value").cast("string"), getEventSchema()).alias("event"))
            .select("event.*")
            .withWatermark("event_timestamp", "10 minutes");
            
        // Stateful processing: user session tracking
        Dataset<Row> sessionAggregates = parsedEvents
            .groupByKey(
                row -> row.getAs("user_id").toString(),
                Encoders.STRING()
            )
            .flatMapGroupsWithState(
                new UserSessionProcessor(),
                OutputMode.Append(),
                Encoders.kryo(UserSession.class),
                Encoders.kryo(SessionEvent.class),
                GroupStateTimeout.ProcessingTimeTimeout()
            );
            
        return sessionAggregates.writeStream()
            .outputMode("append")
            .format("delta")
            .option("checkpointLocation", "/tmp/checkpoints/user-sessions")
            .option("path", "/data/user-sessions")
            .trigger(Trigger.ProcessingTime("1 minute"))
            .start();
    }
    
    // Machine learning pipeline with MLflow tracking
    public void trainMLModel(String trainingDataPath, String modelOutputPath) {
        Dataset<Row> trainingData = spark.read()
            .parquet(trainingDataPath)
            .cache();
            
        // Feature engineering pipeline
        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"feature1", "feature2", "feature3"})
            .setOutputCol("features");
            
        StandardScaler scaler = new StandardScaler()
            .setInputCol("features")
            .setOutputCol("scaledFeatures")
            .setWithStd(true)
            .setWithMean(false);
            
        RandomForestRegressor rf = new RandomForestRegressor()
            .setFeaturesCol("scaledFeatures")
            .setLabelCol("label")
            .setNumTrees(100)
            .setMaxDepth(10);
            
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
            assembler, scaler, rf
        });
        
        // Cross-validation
        CrossValidator cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(new RegressionEvaluator())
            .setEstimatorParamMaps(new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[]{50, 100, 200})
                .addGrid(rf.maxDepth(), new int[]{5, 10, 15})
                .build())
            .setNumFolds(3);
            
        CrossValidatorModel cvModel = cv.fit(trainingData);
        
        // Save model with MLflow tracking
        MLflowClient mlflowClient = new MLflowClient();
        ExperimentSummary experiment = mlflowClient.createExperiment("spark-ml-pipeline");
        
        try (MLflowContext context = MLflowContext.createRun(experiment.getExperimentId())) {
            context.logParam("numTrees", "100");
            context.logParam("maxDepth", "10");
            context.logMetric("rmse", getRMSE(cvModel));
            
            cvModel.write().overwrite().save(modelOutputPath);
            context.logArtifact(modelOutputPath);
        }
    }
    
    private double getRMSE(CrossValidatorModel model) {
        // Calculate and return RMSE metric
        return 0.0; // Simplified
    }
}

// Stateful processing for user sessions
public class UserSessionProcessor implements FlatMapGroupsWithStateFunction<String, Row, UserSession, SessionEvent> {
    
    @Override
    public Iterator<SessionEvent> call(String key, Iterator<Row> values, GroupState<UserSession> state) {
        List<SessionEvent> results = new ArrayList<>();
        
        UserSession currentSession = state.exists() ? state.get() : new UserSession(key);
        
        while (values.hasNext()) {
            Row event = values.next();
            currentSession.addEvent(event);
            
            // Check for session timeout
            if (currentSession.shouldTimeout()) {
                results.add(currentSession.createSessionEndEvent());
                state.remove(); // Remove expired session
            } else {
                state.update(currentSession);
                state.setTimeoutDuration("30 minutes"); // Session timeout
            }
        }
        
        // Handle timeout
        if (state.hasTimedOut()) {
            results.add(currentSession.createTimeoutEvent());
            state.remove();
        }
        
        return results.iterator();
    }
}
```

---

## 6. Real-time Stream Processing

### 6.1 Pulsar Functions for Stream Processing

```java
// Pulsar Function for real-time data transformation
public class OrderEnrichmentFunction implements Function<String, String> {
    
    private RedisClient redisClient;
    private ObjectMapper objectMapper;
    
    @Override
    public void initialize(Context context) {
        this.redisClient = RedisClient.create("redis://localhost:6379");
        this.objectMapper = new ObjectMapper();
    }
    
    @Override
    public String process(String input, Context context) {
        try {
            OrderEvent event = objectMapper.readValue(input, OrderEvent.class);
            
            // Enrich with customer data from Redis
            String customerData = redisClient.sync().get("customer:" + event.getCustomerId());
            if (customerData != null) {
                CustomerInfo customer = objectMapper.readValue(customerData, CustomerInfo.class);
                event.setCustomerName(customer.getName());
                event.setCustomerTier(customer.getTier());
            }
            
            // Enrich with product data
            String productData = redisClient.sync().get("product:" + event.getProductId());
            if (productData != null) {
                ProductInfo product = objectMapper.readValue(productData, ProductInfo.class);
                event.setProductName(product.getName());
                event.setProductCategory(product.getCategory());
            }
            
            // Calculate derived fields
            event.setProcessingTimestamp(System.currentTimeMillis());
            event.setOrderValue(calculateOrderValue(event));
            
            context.getLogger().info("Enriched order: " + event.getOrderId());
            
            return objectMapper.writeValueAsString(event);
            
        } catch (Exception e) {
            context.getLogger().error("Error processing order event", e);
            throw new RuntimeException(e);
        }
    }
    
    private double calculateOrderValue(OrderEvent event) {
        // Complex business logic for order value calculation
        double baseValue = event.getTotal();
        double discount = getDiscount(event.getCustomerTier());
        return baseValue * (1 - discount);
    }
    
    private double getDiscount(String customerTier) {
        return switch (customerTier) {
            case "GOLD" -> 0.10;
            case "SILVER" -> 0.05;
            default -> 0.0;
        };
    }
}

// Deploy Pulsar Function
@Service
public class PulsarFunctionDeploymentService {
    
    private final PulsarAdmin pulsarAdmin;
    
    public void deployOrderEnrichmentFunction() {
        try {
            FunctionConfig functionConfig = new FunctionConfig();
            functionConfig.setName("order-enrichment");
            functionConfig.setNamespace("public/default");
            functionConfig.setClassName("com.example.OrderEnrichmentFunction");
            functionConfig.setInputs(Collections.singletonList("order-events"));
            functionConfig.setOutput("enriched-orders");
            functionConfig.setRuntime(FunctionConfig.Runtime.JAVA);
            functionConfig.setProcessingGuarantees(FunctionConfig.ProcessingGuarantees.ATLEAST_ONCE);
            functionConfig.setParallelism(4);
            
            // Resource configuration
            Resources resources = new Resources();
            resources.setCpu(1.0);
            resources.setRam(512L * 1024 * 1024); // 512MB
            functionConfig.setResources(resources);
            
            pulsarAdmin.functions().createFunction(functionConfig, "order-enrichment.jar");
            
            log.info("Deployed Pulsar Function: order-enrichment");
            
        } catch (PulsarAdminException e) {
            throw new RuntimeException("Failed to deploy Pulsar function", e);
        }
    }
    
    public FunctionStatus getFunctionStatus(String functionName) {
        try {
            return pulsarAdmin.functions().getFunctionStatus("public", "default", functionName);
        } catch (PulsarAdminException e) {
            throw new RuntimeException("Failed to get function status", e);
        }
    }
}
```

### 6.2 Spark Streaming with Advanced Features

```java
@Service
public class SparkStreamingService {
    
    private final SparkSession spark;
    
    // Complex CEP (Complex Event Processing) with Spark
    public StreamingQuery detectFraudPatterns() {
        Dataset<Row> transactionStream = spark.readStream()
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("topic", "persistent://public/default/transactions")
            .load();
            
        Dataset<Row> transactions = transactionStream
            .select(from_json(col("value").cast("string"), getTransactionSchema()).alias("txn"))
            .select("txn.*")
            .withWatermark("transaction_time", "5 minutes");
            
        // Pattern: Multiple high-value transactions from same user within short timeframe
        Dataset<Row> suspiciousActivity = transactions
            .filter(col("amount").gt(1000)) // High value transactions
            .groupBy(
                col("user_id"),
                window(col("transaction_time"), "10 minutes")
            )
            .agg(
                count("*").alias("transaction_count"),
                sum("amount").alias("total_amount"),
                collect_list("transaction_id").alias("transaction_ids")
            )
            .filter(col("transaction_count").gt(3)) // More than 3 transactions
            .filter(col("total_amount").gt(5000))   // Total > $5000
            .select(
                col("user_id"),
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("transaction_count"),
                col("total_amount"),
                col("transaction_ids"),
                lit("FRAUD_PATTERN_DETECTED").alias("alert_type")
            );
            
        return suspiciousActivity.writeStream()
            .outputMode("append")
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("topic", "persistent://public/default/fraud-alerts")
            .option("checkpointLocation", "/tmp/checkpoints/fraud-detection")
            .start();
    }
    
    // Multi-stream joins with different watermarks
    public StreamingQuery correlateStreams() {
        // User events stream
        Dataset<Row> userEvents = spark.readStream()
            .format("pulsar")
            .option("topic", "user-events")
            .load()
            .select(from_json(col("value").cast("string"), getUserEventSchema()).alias("event"))
            .select("event.*")
            .withWatermark("event_time", "10 minutes");
            
        // System metrics stream
        Dataset<Row> systemMetrics = spark.readStream()
            .format("pulsar")
            .option("topic", "system-metrics")
            .load()
            .select(from_json(col("value").cast("string"), getMetricSchema()).alias("metric"))
            .select("metric.*")
            .withWatermark("metric_time", "5 minutes");
            
        // Join streams with time window
        Dataset<Row> correlatedData = userEvents.join(
            systemMetrics,
            expr("user_id = server_id AND " +
                 "event_time >= metric_time AND " +
                 "event_time <= metric_time + interval 2 minutes"),
            "leftOuter"
        );
        
        return correlatedData.writeStream()
            .outputMode("append")
            .format("delta")
            .option("checkpointLocation", "/tmp/checkpoints/correlation")
            .option("path", "/data/correlated-events")
            .trigger(Trigger.ProcessingTime("30 seconds"))
            .start();
    }
    
    // Adaptive query execution for streaming
    public StreamingQuery adaptiveStreamProcessing() {
        Dataset<Row> eventStream = spark.readStream()
            .format("pulsar")
            .option("topic", "adaptive-events")
            .load();
            
        // Enable adaptive query execution
        spark.conf().set("spark.sql.adaptive.enabled", "true");
        spark.conf().set("spark.sql.adaptive.skewJoin.enabled", "true");
        spark.conf().set("spark.sql.streaming.adaptive.enabled", "true");
        
        Dataset<Row> processedStream = eventStream
            .select(from_json(col("value").cast("string"), getEventSchema()).alias("event"))
            .select("event.*")
            .withColumn("processing_time", current_timestamp())
            .groupBy(col("category"))
            .agg(
                count("*").alias("event_count"),
                avg("latency").alias("avg_latency"),
                max("processing_time").alias("latest_processed")
            );
            
        return processedStream.writeStream()
            .outputMode("complete")
            .format("memory")
            .queryName("adaptive_processing")
            .trigger(Trigger.ProcessingTime("10 seconds"))
            .start();
    }
}
```

---

## 7. Data Pipeline Patterns

### 7.1 Lambda Architecture Implementation

```java
@Service
@Slf4j
public class LambdaArchitectureService {
    
    private final SparkSession spark;
    private final PulsarProducerService pulsarProducer;
    
    // Batch Layer - Historical data processing
    @Scheduled(cron = "0 0 2 * * *") // Daily at 2 AM
    public void runBatchProcessing() {
        log.info("Starting batch processing for Lambda architecture");
        
        String inputPath = "/data/raw/events/" + 
            LocalDate.now().minusDays(1).format(DateTimeFormatter.ofPattern("yyyy/MM/dd"));
            
        Dataset<Row> dailyEvents = spark.read()
            .option("basePath", "/data/raw/events/")
            .parquet(inputPath)
            .cache();
            
        // Comprehensive batch computations
        Dataset<Row> batchViews = computeBatchViews(dailyEvents);
        
        // Write to serving layer (e.g., Cassandra, HBase)
        batchViews.write()
            .mode(SaveMode.Overwrite)
            .option("table", "batch_views")
            .option("keyspace", "analytics")
            .format("org.apache.spark.sql.cassandra")
            .save();
            
        log.info("Batch processing completed. Processed {} events", dailyEvents.count());
    }
    
    // Speed Layer - Real-time incremental processing
    public StreamingQuery runSpeedLayer() {
        Dataset<Row> realtimeEvents = spark.readStream()
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("topic", "persistent://public/default/raw-events")
            .load();
            
        Dataset<Row> incrementalViews = realtimeEvents
            .select(from_json(col("value").cast("string"), getEventSchema()).alias("event"))
            .select("event.*")
            .withWatermark("event_timestamp", "1 minute")
            .groupBy(
                window(col("event_timestamp"), "1 minute"),
                col("category")
            )
            .agg(
                count("*").alias("event_count"),
                sum("value").alias("total_value"),
                countDistinct("user_id").alias("unique_users")
            );
            
        return incrementalViews.writeStream()
            .outputMode("append")
            .foreachBatch((batch, batchId) -> {
                // Write to speed layer storage (e.g., Redis, Cassandra)
                batch.write()
                    .mode(SaveMode.Append)
                    .option("table", "realtime_views")
                    .option("keyspace", "analytics")
                    .format("org.apache.spark.sql.cassandra")
                    .save();
                    
                log.info("Speed layer batch {} processed", batchId);
            })
            .option("checkpointLocation", "/tmp/checkpoints/speed-layer")
            .start();
    }
    
    // Serving Layer - Query interface combining batch and speed views
    public QueryResult queryLambdaViews(String category, LocalDateTime startTime, LocalDateTime endTime) {
        // Query batch views for historical data
        Dataset<Row> batchResults = spark.read()
            .option("table", "batch_views")
            .option("keyspace", "analytics")
            .format("org.apache.spark.sql.cassandra")
            .load()
            .filter(col("category").equalTo(category))
            .filter(col("date").between(startTime.toLocalDate(), endTime.toLocalDate()));
            
        // Query speed layer for recent data
        Dataset<Row> realtimeResults = spark.read()
            .option("table", "realtime_views")
            .option("keyspace", "analytics")
            .format("org.apache.spark.sql.cassandra")
            .load()
            .filter(col("category").equalTo(category))
            .filter(col("window_start").geq(startTime))
            .filter(col("window_end").leq(endTime));
            
        // Merge results
        Dataset<Row> mergedResults = batchResults.union(realtimeResults)
            .groupBy("category", "date")
            .agg(
                sum("event_count").alias("total_events"),
                sum("total_value").alias("aggregated_value"),
                sum("unique_users").alias("total_unique_users")
            );
            
        List<Row> results = mergedResults.collectAsList();
        return new QueryResult(results);
    }
    
    private Dataset<Row> computeBatchViews(Dataset<Row> events) {
        return events
            .groupBy("category", "date")
            .agg(
                count("*").alias("event_count"),
                sum("value").alias("total_value"),
                countDistinct("user_id").alias("unique_users"),
                avg("processing_latency").alias("avg_latency"),
                min("event_timestamp").alias("first_event"),
                max("event_timestamp").alias("last_event")
            );
    }
}
```

### 7.2 Kappa Architecture (Stream-First)

```java
@Service
@Slf4j
public class KappaArchitectureService {
    
    private final SparkSession spark;
    
    // Single stream processing pipeline handling both real-time and historical data
    public StreamingQuery runKappaProcessing() {
        Dataset<Row> eventStream = spark.readStream()
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("topic", "persistent://public/default/unified-events")
            .option("startingOffsets", "earliest") // Process historical data too
            .load();
            
        Dataset<Row> processedEvents = eventStream
            .select(from_json(col("value").cast("string"), getEventSchema()).alias("event"))
            .select("event.*")
            .withWatermark("event_timestamp", "10 minutes")
            .withColumn("processing_timestamp", current_timestamp())
            .withColumn("is_late", 
                when(col("processing_timestamp").minus(col("event_timestamp"))
                    .gt(expr("INTERVAL 5 MINUTES")), true)
                .otherwise(false));
            
        // Multi-window aggregations
        Dataset<Row> multiWindowViews = processedEvents
            .groupBy(
                col("category"),
                window(col("event_timestamp"), "1 minute").alias("minute_window"),
                window(col("event_timestamp"), "1 hour").alias("hour_window"),
                window(col("event_timestamp"), "1 day").alias("day_window")
            )
            .agg(
                count("*").alias("event_count"),
                sum("value").alias("total_value"),
                countDistinct("user_id").alias("unique_users"),
                sum(when(col("is_late"), 1).otherwise(0)).alias("late_events")
            );
            
        return multiWindowViews.writeStream()
            .outputMode("append")
            .foreachBatch(this::writeToServingLayer)
            .option("checkpointLocation", "/tmp/checkpoints/kappa-architecture")
            .trigger(Trigger.ProcessingTime("30 seconds"))
            .start();
    }
    
    private void writeToServingLayer(Dataset<Row> batch, long batchId) {
        // Write to multiple serving layer stores based on access patterns
        
        // Recent data to Redis for fast access
        Dataset<Row> recentData = batch.filter(
            col("minute_window.end").gt(
                expr("current_timestamp() - INTERVAL 1 HOUR")));
                
        recentData.foreachPartition(partition -> {
            Jedis redis = new Jedis("localhost", 6379);
            try {
                while (partition.hasNext()) {
                    Row row = partition.next();
                    String key = String.format("kappa:%s:%s", 
                        row.getAs("category"), 
                        row.getAs("minute_window"));
                    String value = row.json();
                    redis.setex(key, 3600, value); // 1 hour TTL
                }
            } finally {
                redis.close();
            }
        });
        
        // Historical data to Delta Lake for analytical queries
        batch.write()
            .mode(SaveMode.Append)
            .partitionBy("category")
            .option("mergeSchema", "true")
            .format("delta")
            .save("/data/kappa-views/");
            
        log.info("Kappa architecture batch {} written to serving layer", batchId);
    }
    
    // Reprocessing capability for schema evolution or bug fixes
    public void reprocessHistoricalData(LocalDateTime fromTime) {
        log.info("Starting historical data reprocessing from: {}", fromTime);
        
        // Create new version of the processing logic
        String checkpointLocation = "/tmp/checkpoints/kappa-reprocessing-" + 
            System.currentTimeMillis();
            
        StreamingQuery reprocessingQuery = spark.readStream()
            .format("pulsar")
            .option("service.url", "pulsar://localhost:6650")
            .option("topic", "persistent://public/default/unified-events")
            .option("subscriptionName", "reprocessing-" + System.currentTimeMillis())
            .option("subscriptionInitialPosition", "Earliest")
            .load()
            .filter(col("event_timestamp").gt(fromTime))
            .writeStream()
            .outputMode("append")
            .foreachBatch(this::writeToServingLayer)
            .option("checkpointLocation", checkpointLocation)
            .start();
            
        // Monitor reprocessing progress
        monitorReprocessing(reprocessingQuery, fromTime);
    }
    
    private void monitorReprocessing(StreamingQuery query, LocalDateTime startTime) {
        // Implementation for monitoring reprocessing progress
        CompletableFuture.runAsync(() -> {
            while (query.isActive()) {
                StreamingQueryProgress progress = query.lastProgress();
                if (progress != null) {
                    log.info("Reprocessing progress: {} events processed, rate: {} events/sec",
                        progress.numInputRows(), progress.inputRowsPerSecond());
                }
                
                try {
                    Thread.sleep(30000); // Check every 30 seconds
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
    }
}
```