# REST APIs: Complete Guide

## Table of Contents
1. [Introduction to REST](#introduction-to-rest)
2. [REST Principles and Constraints](#rest-principles-and-constraints)
3. [HTTP Fundamentals](#http-fundamentals)
4. [REST API Design](#rest-api-design)
5. [Request and Response Patterns](#request-and-response-patterns)
6. [Authentication and Authorization](#authentication-and-authorization)
7. [Rate Limiting and Throttling](#rate-limiting-and-throttling)
8. [Error Handling](#error-handling)
9. [Versioning and Evolution](#versioning-and-evolution)
10. [REST vs GraphQL](#rest-vs-graphql)
11. [Practical Examples](#practical-examples)
12. [Best Practices](#best-practices)

---

## Introduction to REST

**REST** (Representational State Transfer) is an architectural style for designing networked applications. RESTful APIs use HTTP to expose resources that clients can access and manipulate.

### Key Concepts

- **Resource**: Any entity that can be identified (users, orders, products)
- **Representation**: Format of resource data (JSON, XML)
- **State Transfer**: Client-server communication via HTTP methods

### REST vs Other API Styles

| Style | Description | Example |
|-------|-------------|---------|
| **REST** | Resource-oriented, HTTP methods | GET /users/123 |
| **GraphQL** | Query language, single endpoint | POST /graphql with query |
| **gRPC** | Binary, schema-first, streaming | Method calls |
| **SOAP** | XML-based, WSDL | XML envelopes |

---

## REST Principles and Constraints

### Six Constraints (Fielding)

1. **Client-Server**: Separation of concerns; clients don't store data
2. **Stateless**: Each request contains all needed info; no server session
3. **Cacheable**: Responses indicate if they can be cached
4. **Uniform Interface**: Consistent way to interact (HTTP methods, URIs)
5. **Layered System**: Proxies, gateways, load balancers allowed
6. **Code on Demand** (optional): Server can send executable code

### Uniform Interface: Four Constraints

- **Resource identification**: URIs identify resources
- **Manipulation through representations**: Client has enough info to modify
- **Self-descriptive messages**: Each message has metadata (Content-Type, etc.)
- **HATEOAS**: Hypermedia as the Engine of Application State (links in responses)

---

## HTTP Fundamentals

### HTTP Methods

| Method | Idempotent | Safe | Use Case |
|--------|------------|------|----------|
| GET | Yes | Yes | Retrieve resource(s) |
| POST | No | No | Create resource |
| PUT | Yes | No | Replace resource (full update) |
| PATCH | No | No | Partial update |
| DELETE | Yes | No | Remove resource |
| HEAD | Yes | Yes | Headers only (no body) |
| OPTIONS | Yes | Yes | Allowed methods |

### Status Codes

| Range | Meaning | Examples |
|-------|---------|----------|
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirection | 301 Moved, 304 Not Modified |
| 4xx | Client Error | 400 Bad Request, 401 Unauthorized, 404 Not Found |
| 5xx | Server Error | 500 Internal Error, 503 Service Unavailable |

### Common Status Codes in Detail

```python
# 200 OK - Success, body contains representation
# 201 Created - Resource created, Location header has URI
# 204 No Content - Success, no body (e.g., after DELETE)
# 400 Bad Request - Malformed request, validation failed
# 401 Unauthorized - Not authenticated
# 403 Forbidden - Authenticated but not authorized
# 404 Not Found - Resource doesn't exist
# 409 Conflict - State conflict (e.g., duplicate)
# 422 Unprocessable Entity - Validation error (semantic)
# 429 Too Many Requests - Rate limited
# 500 Internal Server Error - Server bug
# 503 Service Unavailable - Overloaded, maintenance
```

---

## REST API Design

### Resource Naming

```
# Good: Nouns, plural, hierarchical
GET    /users
GET    /users/123
GET    /users/123/orders
GET    /users/123/orders/456

# Bad: Verbs in URL
GET    /getUsers
POST   /createUser
GET    /users/delete/123
```

### URI Design Patterns

```python
# Collection
GET    /products           # List products
POST   /products           # Create product

# Member
GET    /products/42        # Get product 42
PUT    /products/42        # Replace product 42
PATCH  /products/42       # Partial update
DELETE /products/42       # Delete product 42

# Sub-resources
GET    /products/42/reviews
POST   /products/42/reviews

# Actions (use sparingly, when no resource fits)
POST   /orders/42/cancel
POST   /users/123/verify-email

# Filtering, sorting, pagination (query params)
GET    /products?category=electronics&sort=price&order=asc
GET    /products?page=2&limit=20
```

### Query Parameters

```python
# Filtering
GET /users?status=active&role=admin

# Sorting
GET /users?sort=created_at&order=desc

# Pagination (offset)
GET /users?offset=20&limit=10

# Pagination (cursor - preferred for large datasets)
GET /users?cursor=eyJpZCI6MjB9&limit=10

# Field selection (if supported)
GET /users?fields=id,name,email

# Search
GET /users?q=john+doe
```

---

## Request and Response Patterns

### Request Headers

```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
X-Request-ID: uuid-for-tracing
Accept-Language: en-US
```

### Response Headers

```http
Content-Type: application/json
Cache-Control: max-age=3600, private
ETag: "33a64df5"
Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
Location: /users/123
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
```

### JSON Request/Response

```python
# Create user - Request
POST /users
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "role": "user"
}

# Response 201 Created
Location: /users/123
Content-Type: application/json

{
  "id": "123",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "user",
  "created_at": "2024-10-21T12:00:00Z"
}
```

### Pagination Response

```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": true
  },
  "links": {
    "self": "/users?page=2&limit=20",
    "next": "/users?page=3&limit=20",
    "prev": "/users?page=1&limit=20"
  }
}
```

---

## Authentication and Authorization

### API Keys

```python
# Header
Authorization: ApiKey your-api-key-here

# Query param (less secure - keys in logs)
GET /users?api_key=your-api-key
```

### Bearer Tokens (JWT)

```python
# Header
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# JWT structure: header.payload.signature
# Payload: {"sub": "user123", "exp": 1699000000}
```

### OAuth 2.0

```
1. Client redirects user to authorization server
2. User grants permission
3. Auth server redirects with code
4. Client exchanges code for access_token
5. Client uses access_token in API requests
```

### Implementation with FastAPI

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Validate JWT, decode, check exp
    user = decode_jwt(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/users/me")
async def get_me(user = Depends(verify_token)):
    return {"id": user["sub"], "email": user["email"]}
```

---

## Rate Limiting and Throttling

### Strategies

- **Fixed window**: 100 req/min per key
- **Sliding window**: Smoother limits
- **Token bucket**: Burst allowance

### Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699000060
Retry-After: 60
```

### Implementation (FastAPI)

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/data")
@limiter.limit("10/minute")
async def get_data(request: Request):
    return {"data": "..."}

# 429 response when exceeded
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ],
    "request_id": "req_abc123",
    "timestamp": "2024-10-21T12:00:00Z"
  }
}
```

### HTTP Status + Error Code

```python
# 400 + code for client to handle
{"error": {"code": "INVALID_CREDENTIALS", "message": "Email or password incorrect"}}

# 404
{"error": {"code": "NOT_FOUND", "message": "User 123 not found"}}

# 409
{"error": {"code": "DUPLICATE_EMAIL", "message": "Email already registered"}}
```

---

## Versioning and Evolution

### URL Versioning

```
/api/v1/users
/api/v2/users
```

### Header Versioning

```http
Accept: application/vnd.myapi.v1+json
```

### Best Practices

- Add new fields without breaking (optional fields)
- Deprecate old fields with warnings; remove in next major
- Never remove or rename fields in same version
- Use Sunset header for deprecated APIs: `Sunset: Wed, 01 Nov 2025 00:00:00 GMT`

---

## REST vs GraphQL

| Aspect | REST | GraphQL |
|--------|------|---------|
| Endpoints | Multiple (/users, /orders) | Single (/graphql) |
| Data fetching | Over/under-fetching | Exact shape requested |
| Caching | HTTP cache, CDN | Client-side, harder |
| Complexity | Simpler | Schema, resolvers |
| Real-time | Polling or separate WS | Subscriptions |

**When REST**: Simple CRUD, caching critical, broad client support  
**When GraphQL**: Complex nested data, mobile (payload size), rapid iteration

---

## Practical Examples

### FastAPI REST API

```python
from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="User API", version="1.0.0")

# Pydantic models
class UserCreate(BaseModel):
    name: str
    email: str
    role: str = "user"

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str

# In-memory store (use DB in production)
users_db = {}

@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    user_id = str(len(users_db) + 1)
    users_db[user_id] = {"id": user_id, **user.dict()}
    return users_db[user_id]

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    role: Optional[str] = None
):
    filtered = list(users_db.values())
    if role:
        filtered = [u for u in filtered if u["role"] == role]
    return filtered[skip:skip+limit]

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user: UserCreate):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    users_db[user_id].update(user.dict(exclude_unset=True))
    return users_db[user_id]

@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[user_id]
```

### Flask REST API

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
users = {}

@app.route('/users', methods=['GET'])
def list_users():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 20))
    start = (page - 1) * limit
    return jsonify(list(users.values())[start:start+limit])

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    user_id = str(len(users) + 1)
    users[user_id] = {'id': user_id, **data}
    return jsonify(users[user_id]), 201

@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    if user_id not in users:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(users[user_id])
```

### Client Example (Python)

```python
import requests

BASE = "http://localhost:8000"

# Create
resp = requests.post(f"{BASE}/users", json={"name": "Jane", "email": "jane@example.com"})
user = resp.json()
print(user["id"])

# Get
resp = requests.get(f"{BASE}/users/{user['id']}")
print(resp.json())

# List with params
resp = requests.get(f"{BASE}/users", params={"page": 1, "limit": 10})
```

---

## Best Practices

1. **Use HTTPS** in production
2. **Use nouns** for resources, HTTP methods for actions
3. **Return appropriate status codes**
4. **Document with OpenAPI/Swagger**
5. **Validate input** (Pydantic, marshmallow)
6. **Idempotency keys** for POST when retries possible
7. **Request IDs** for tracing
8. **Consistent error format**
9. **Pagination** for list endpoints
10. **Rate limit** to protect backend

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Methods | GET (read), POST (create), PUT/PATCH (update), DELETE |
| Status | 2xx success, 4xx client, 5xx server |
| Design | Nouns, plural, hierarchical URIs |
| Auth | Bearer tokens, API keys, OAuth |
| Errors | Consistent JSON format, codes |

**Frameworks**: FastAPI, Flask, Django REST Framework, Express.js
