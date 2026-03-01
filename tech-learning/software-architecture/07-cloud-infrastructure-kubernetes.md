# Cloud Infrastructure and Kubernetes: Guide for Architects

## Table of Contents
1. [Cloud-Native Principles](#1-cloud-native-principles)
2. [Containers and Docker](#2-containers-and-docker)
3. [Kubernetes Fundamentals](#3-kubernetes-fundamentals)
4. [Kubernetes Advanced Concepts](#4-kubernetes-advanced-concepts)
5. [Infrastructure as Code (IaC)](#5-infrastructure-as-code-iac)
6. [Multi-Region and High Availability](#6-multi-region-and-high-availability)
7. [Serverless and FaaS](#7-serverless-and-faas)
8. [Cost Optimization](#8-cost-optimization)
9. [Practical Examples](#9-practical-examples)

---

## 1. Cloud-Native Principles

### 1.1 Twelve-Factor App

| Factor | Description |
|--------|-------------|
| Codebase | One codebase, many deploys |
| Dependencies | Explicitly declare (requirements.txt, go.mod) |
| Config | Store config in environment |
| Backing services | Treat as attached resources |
| Build, release, run | Strict separation |
| Processes | Stateless, share-nothing |
| Port binding | Self-contained, export via port |
| Concurrency | Scale via process model |
| Disposability | Fast startup, graceful shutdown |
| Dev/prod parity | Same backing services |
| Logs | Treat as event streams |
| Admin processes | Run as one-off processes |

### 1.2 Cloud-Native Design

- **Microservices**: Small, deployable units
- **Containers**: Package app + dependencies
- **Dynamic orchestration**: K8s schedules and heals
- **API-driven**: Declarative, automation-friendly

---

## 2. Containers and Docker

### 2.1 Dockerfile Best Practices

```dockerfile
# Multi-stage build: smaller final image
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 2.2 Image Layers

- Each instruction creates a layer
- Reuse cached layers (order matters: rarely changing first)
- `.dockerignore` to exclude files

### 2.3 Container Runtime

- **containerd**: Industry standard
- **CRI-O**: Lightweight, OCP
- **Docker (dockerd)**: Includes containerd

---

## 3. Kubernetes Fundamentals

### 3.1 Architecture

```
Control Plane:
  - API Server
  - etcd (state)
  - Scheduler
  - Controller Manager

Workers:
  - kubelet
  - kube-proxy
  - Container runtime
```

### 3.2 Core Objects

| Object | Purpose |
|--------|---------|
| **Pod** | Smallest deployable unit; 1+ containers |
| **Deployment** | Declarative updates for Pods (replicas, rolling update) |
| **Service** | Stable network identity for Pods |
| **ConfigMap** | Non-sensitive config |
| **Secret** | Sensitive config |
| **Ingress** | HTTP routing into cluster |
| **Namespace** | Logical isolation |

### 3.3 Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  labels:
    app: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      containers:
      - name: order-service
        image: myreg/order-service:v1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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

### 3.4 Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP  # ClusterIP | NodePort | LoadBalancer
```

### 3.5 Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /orders
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 80
      - path: /users
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 80
```

---

## 4. Kubernetes Advanced Concepts

### 4.1 Resource Limits and Requests

- **Requests**: Scheduler uses for placement; guaranteed
- **Limits**: Max usage; beyond = throttled (CPU) or OOMKilled (memory)

### 4.2 Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 4.3 Pod Disruption Budget (PDB)

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: order-service-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: order-service
```

### 4.4 ConfigMap and Secret

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  LOG_LEVEL: "info"
  DB_HOST: "postgres.default.svc"
---
apiVersion: v1
kind: Secret
metadata:
  name: order-service-secrets
type: Opaque
data:
  DB_PASSWORD: <base64-encoded>
```

```yaml
# Reference in Deployment
envFrom:
- configMapRef:
    name: order-service-config
- secretRef:
    name: order-service-secrets
```

### 4.5 Init Containers

Run before main container; e.g., wait for DB, migrate schema.

```yaml
initContainers:
- name: wait-db
  image: busybox
  command: ['sh', '-c', 'until nc -z postgres 5432; do sleep 2; done']
```

---

## 5. Infrastructure as Code (IaC)

### 5.1 Terraform

Declarative; manages cloud resources.

```hcl
resource "aws_eks_cluster" "main" {
  name     = "my-eks"
  role_arn = aws_iam_role.eks.arn
  vpc_config {
    subnet_ids = [aws_subnet.private[*].id]
  }
}

resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "workers"
  node_role_arn   = aws_iam_role.nodes.arn
  subnet_ids      = [aws_subnet.private[*].id]
  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 1
  }
}
```

### 5.2 Helm

Package manager for Kubernetes. Charts = templates + values.

```yaml
# values.yaml
replicaCount: 3
image:
  repository: myreg/order-service
  tag: v1.0
service:
  type: ClusterIP
  port: 80
```

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}-order-service
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: app
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
```

### 5.3 Kustomize

Overlay-based; patch bases for env-specific config.

```
base/
  deployment.yaml
  service.yaml
overlays/
  dev/
    kustomization.yaml  # replicaCount: 1
  prod/
    kustomization.yaml  # replicaCount: 10
```

---

## 6. Multi-Region and High Availability

### 6.1 Multi-AZ

- Deploy across availability zones in same region
- Use anti-affinity to spread Pods

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: order-service
      topologyKey: kubernetes.io/hostname
```

### 6.2 Multi-Region

- Active-passive or active-active
- Data replication (DB, cache)
- Global load balancing (Route53, CloudFlare)

### 6.3 Disaster Recovery

- Backup etcd, DBs, persistent volumes
- Document RTO/RPO
- Regular DR drills

---

## 7. Serverless and FaaS

### 7.1 When to Use

- Event-driven (S3, SQS, HTTP)
- Variable/spiky load
- Short-lived workloads
- No server management

### 7.2 Limits

- Cold start
- Execution time limits
- Vendor lock-in

### 7.3 Example: AWS Lambda

```python
def lambda_handler(event, context):
    order_id = event['pathParameters']['orderId']
    order = get_order(order_id)
    return {
        'statusCode': 200,
        'body': json.dumps(order)
    }
```

---

## 8. Cost Optimization

- **Right-sizing**: Match requests/limits to actual usage
- **Spot/Preemptible**: For fault-tolerant batch workloads
- **Autoscaling**: Scale down when idle
- **Reserved instances**: For predictable load
- **Cleanup**: Unused volumes, old images

---

## 9. Practical Examples

### 9.1 Full K8s Manifest (Deployment + Service + HPA)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
  template:
    metadata:
      labels:
        app: order-service
    spec:
      containers:
      - name: app
        image: myreg/order-service:v1
        ports:
        - containerPort: 8080
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: order-config
              key: DB_HOST
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: order-secrets
              key: DB_PASSWORD
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 9.2 Docker Compose for Local Dev

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
    - "8080:8080"
    environment:
    - DB_HOST=postgres
    - REDIS_HOST=redis
    depends_on:
    - postgres
    - redis
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: orders
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
    volumes:
    - pgdata:/var/lib/postgresql/data
  redis:
    image: redis:7-alpine
volumes:
  pgdata:
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Containers** | Multi-stage build, non-root, minimal base |
| **K8s** | Deployment, Service, Ingress, ConfigMap, Secret |
| **Scaling** | HPA, PDB |
| **IaC** | Terraform, Helm, Kustomize |
| **HA** | Multi-AZ, anti-affinity |
| **Serverless** | Event-driven, variable load |

---

## Further Reading

- Kubernetes: https://kubernetes.io/docs/
- Helm: https://helm.sh/
- Terraform: https://www.terraform.io/docs
