# API Design: REST, GraphQL, and gRPC for Architects

## Table of Contents
1. [API Design Principles](#1-api-design-principles)
2. [REST Deep Dive](#2-rest-deep-dive)
3. [GraphQL Deep Dive](#3-graphql-deep-dive)
4. [gRPC Deep Dive](#4-grpc-deep-dive)
5. [Comparing REST vs GraphQL vs gRPC](#5-comparing-rest-vs-graphql-vs-grpc)
6. [Versioning and Evolution](#6-versioning-and-evolution)
7. [Idempotency and Safety](#7-idempotency-and-safety)
8. [Pagination, Filtering, and Field Selection](#8-pagination-filtering-and-field-selection)
9. [Practical Examples](#9-practical-examples)

---

## 1. API Design Principles

### 1.1 Core Principles

| Principle | Description |
|-----------|-------------|
| **Consistency** | Same patterns across endpoints |
| **Predictability** | Expected behavior, clear errors |
| **Discoverability** | Self-describing, documented |
| **Backward compatibility** | Don't break clients |
| **Security by default** | Auth, validation, rate limit |

### 1.2 Resource-Oriented Design (REST)

- **Resources** are nouns (users, orders, products)
- **HTTP methods** express actions (GET, POST, PUT, PATCH, DELETE)
- **URIs** identify resources; avoid verbs in URL

### 1.3 API Styles at a Glance

| Style | Protocol | Typical Use |
|-------|----------|-------------|
| **REST** | HTTP/JSON | Public APIs, web/mobile |
| **GraphQL** | HTTP/JSON | Flexible queries, mobile |
| **gRPC** | HTTP/2, Protocol Buffers | Internal, microservices, streaming |

---

## 2. REST Deep Dive

### 2.1 URI Design

```
# Good
GET    /users              # List users
GET    /users/123          # Get user 123
POST   /users              # Create user
PUT    /users/123          # Replace user 123
PATCH  /users/123          # Partial update
DELETE /users/123          # Delete user 123

# Nested resources
GET    /users/123/orders   # Orders of user 123
POST   /users/123/orders   # Create order for user 123

# Avoid verbs in URL
POST   /users/123/activate   # Prefer: PATCH /users/123 {"status":"active"}
```

### 2.2 HTTP Methods and Idempotency

| Method | Idempotent | Safe | Body |
|--------|------------|------|------|
| GET | Yes | Yes | No |
| POST | No | No | Yes |
| PUT | Yes | No | Yes |
| PATCH | No* | No | Yes |
| DELETE | Yes | No | No |

*PATCH idempotency depends on implementation.

### 2.3 Status Codes

| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Success, body has representation |
| 201 | Created | Resource created, Location header |
| 204 | No Content | Success, no body (e.g., DELETE) |
| 400 | Bad Request | Malformed request, validation |
| 401 | Unauthorized | Not authenticated |
| 403 | Forbidden | Authenticated but not authorized |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | State conflict (e.g., duplicate) |
| 422 | Unprocessable Entity | Semantic validation error |
| 429 | Too Many Requests | Rate limited |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Overloaded, maintenance |

### 2.4 REST Example (FastAPI)

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class UserCreate(BaseModel):
    email: str
    name: str

class UserUpdate(BaseModel):
    name: Optional[str] = None

@app.get("/users/{user_id}")
def get_user(user_id: str):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users", status_code=201)
def create_user(user: UserCreate):
    existing = db.get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")
    new_user = db.create_user(user)
    return new_user

@app.patch("/users/{user_id}")
def update_user(user_id: str, update: UserUpdate):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return db.update_user(user_id, update.dict(exclude_unset=True))

@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: str):
    if not db.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    db.delete_user(user_id)
```

---

## 3. GraphQL Deep Dive

### 3.1 Core Concepts

- **Schema**: Types and operations
- **Query**: Read data
- **Mutation**: Modify data
- **Subscription**: Real-time updates (over WebSocket)

### 3.2 Schema Example

```graphql
type User {
  id: ID!
  email: String!
  name: String
  orders: [Order!]!
}

type Order {
  id: ID!
  total: Float!
  items: [OrderItem!]!
}

type Query {
  user(id: ID!): User
  users(limit: Int = 10): [User!]!
}

type Mutation {
  createUser(email: String!, name: String): User!
  updateUser(id: ID!, name: String): User
}
```

### 3.3 Query Example

```graphql
query GetUserWithOrders($userId: ID!) {
  user(id: $userId) {
    id
    email
    name
    orders {
      id
      total
      items {
        productId
        quantity
      }
    }
  }
}
```

### 3.4 N+1 Problem and DataLoader

GraphQL can cause N+1: one query for user, N queries for orders. Use **DataLoader** to batch.

```python
from graphene import ObjectType, Field, String, ID, List
from promise import Promise
from promise.dataloader import DataLoader

def batch_load_orders(user_ids):
    orders = db.get_orders_by_user_ids(user_ids)
    by_user = {}
    for o in orders:
        by_user.setdefault(o.user_id, []).append(o)
    return [by_user.get(uid, []) for uid in user_ids]

order_loader = DataLoader(batch_load_orders)

class UserType(ObjectType):
    id = ID()
    email = String()
    orders = List(OrderType)

    def resolve_orders(self, info):
        return order_loader.load(self.id)
```

### 3.5 When to Use GraphQL

- Client needs flexible data shape (mobile, multiple UIs)
- Over-fetching/under-fetching with REST is a problem
- Complex nested data
- Real-time subscriptions

---

## 4. gRPC Deep Dive

### 4.1 Protocol Buffers

Binary, schema-first. Define in `.proto`, generate client/server code.

```protobuf
syntax = "proto3";

package order;

service OrderService {
  rpc CreateOrder(CreateOrderRequest) returns (Order);
  rpc GetOrder(GetOrderRequest) returns (Order);
  rpc ListOrders(ListOrdersRequest) returns (stream Order);
}

message CreateOrderRequest {
  string user_id = 1;
  repeated OrderItem items = 2;
}

message Order {
  string id = 1;
  string user_id = 2;
  repeated OrderItem items = 3;
  double total = 4;
}

message OrderItem {
  string product_id = 1;
  int32 quantity = 2;
}
```

### 4.2 gRPC Communication Patterns

| Pattern | Client | Server | Use Case |
|---------|--------|--------|----------|
| **Unary** | 1 request | 1 response | Simple RPC |
| **Server stream** | 1 request | Stream response | Large result |
| **Client stream** | Stream request | 1 response | Upload |
| **Bidirectional** | Stream | Stream | Chat, real-time |

### 4.3 gRPC Python Example

```python
# server
import grpc
from concurrent import futures
import order_pb2
import order_pb2_grpc

class OrderServicer(order_pb2_grpc.OrderServiceServicer):
    def CreateOrder(self, request, context):
        order = create_order(request.user_id, list(request.items))
        return order_pb2.Order(
            id=order.id,
            user_id=order.user_id,
            total=order.total
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    order_pb2_grpc.add_OrderServiceServicer_to_server(OrderServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
```

```python
# client
import grpc
import order_pb2
import order_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = order_pb2_grpc.OrderServiceStub(channel)
response = stub.CreateOrder(
    order_pb2.CreateOrderRequest(
        user_id="usr_123",
        items=[order_pb2.OrderItem(product_id="p1", quantity=2)]
    )
)
```

### 4.4 gRPC vs REST

| Aspect | gRPC | REST |
|--------|------|------|
| Encoding | Binary (Protobuf) | JSON |
| Transport | HTTP/2 | HTTP/1.1 |
| Streaming | Native | Chunked/SSE |
| Browser | Limited (grpc-web) | Full support |
| Performance | Higher | Lower |

---

## 5. Comparing REST vs GraphQL vs gRPC

| Criteria | REST | GraphQL | gRPC |
|----------|------|---------|------|
| **Over-fetching** | Yes | No | No |
| **Under-fetching** | Yes (multiple calls) | No | Yes |
| **Versioning** | URI/header | Schema evolution | Schema evolution |
| **Caching** | HTTP cache | Custom | Limited |
| **Tooling** | Mature | Growing | Good |
| **Browser** | Yes | Yes | grpc-web |
| **Internal services** | Yes | Yes | Best fit |
| **Mobile** | Yes | Good fit | Possible |

---

## 6. Versioning and Evolution

### 6.1 REST Versioning

- **URI**: `/v1/users`, `/v2/users` — Clear, caching works
- **Header**: `Accept: application/vnd.api+json;version=2` — Clean URLs
- **Query**: `/users?version=2` — Rare

### 6.2 Backward-Compatible Changes

- Add optional fields
- Add new endpoints
- Add new optional query params

### 6.3 Breaking Changes

- Remove or rename fields
- Change type of field
- Remove endpoint
- Change auth behavior

### 6.4 Deprecation

```
Deprecation: true
Sunset: Wed, 01 Jan 2025 00:00:00 GMT
Link: </v2/users>; rel="successor"
```

---

## 7. Idempotency and Safety

### 7.1 Idempotency Key

For POST/PATCH/DELETE that modify state, clients send `Idempotency-Key`. Server returns cached response for duplicate key.

```python
@app.post("/payments")
def create_payment(payment: PaymentCreate, idempotency_key: str = Header(..., alias="Idempotency-Key")):
    cached = redis.get(f"idem:{idempotency_key}")
    if cached:
        return json.loads(cached)
    result = payment_service.create(payment)
    redis.setex(f"idem:{idempotency_key}", 86400, json.dumps(result))
    return result
```

### 7.2 Safe Methods

GET, HEAD, OPTIONS should not change state. Safe to retry.

---

## 8. Pagination, Filtering, and Field Selection

### 8.1 Pagination

**Offset**: `?page=2&limit=20` — Simple; degrades on large offset.

**Cursor**: `?cursor=xyz&limit=20` — Stable; no skip.

```python
@app.get("/users")
def list_users(cursor: Optional[str] = None, limit: int = 20):
    users, next_cursor = db.get_users(cursor=cursor, limit=limit)
    return {"data": users, "next_cursor": next_cursor}
```

### 8.2 Filtering

```
GET /users?status=active&role=admin
GET /orders?created_after=2024-01-01&status=shipped
```

### 8.3 Field Selection (REST)

```
GET /users?fields=id,email,name
```

### 8.4 Sorting

```
GET /users?sort=-created_at,name
# -created_at = descending
```

---

## 9. Practical Examples

### 9.1 REST with Pagination and Error Schema

```python
from fastapi import Query

@app.get("/users")
def list_users(
    cursor: Optional[str] = None,
    limit: int = Query(20, le=100),
    status: Optional[str] = None,
):
    users, next_cursor = db.list_users(cursor=cursor, limit=limit, status=status)
    return {
        "data": users,
        "pagination": {"next_cursor": next_cursor, "limit": limit},
    }

# Error response schema
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request",
    "details": [{"field": "email", "message": "Invalid email format"}]
  }
}
```

### 9.2 GraphQL with Strawberry (Python)

```python
import strawberry
from typing import List, Optional

@strawberry.type
class User:
    id: strawberry.ID
    email: str
    name: Optional[str]

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional[User]:
        return db.get_user(str(id))

    @strawberry.field
    def users(self, limit: int = 10) -> List[User]:
        return db.list_users(limit=limit)

schema = strawberry.Schema(query=Query)
```

### 9.3 gRPC Health Check

```protobuf
service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
}
```

```python
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = health_pb2_grpc.HealthStub(channel)
response = stub.Check(health_pb2.HealthCheckRequest(service=""))
# SERVING, NOT_SERVING, SERVICE_UNKNOWN
```

---

## Summary

| API Style | Best For |
|-----------|----------|
| **REST** | Public APIs, simple CRUD, HTTP caching |
| **GraphQL** | Flexible queries, mobile, multiple clients |
| **gRPC** | Internal microservices, streaming, performance |

**Design tips**: Version early, use idempotency for mutations, cursor pagination for large lists, consistent error schema.

---

## Further Reading

- *REST API Design Rulebook* — Mark Masse
- GraphQL: https://graphql.org/learn/
- gRPC: https://grpc.io/docs/
