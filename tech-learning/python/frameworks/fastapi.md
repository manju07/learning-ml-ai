# FastAPI — Modern Python Web Framework

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Path Operations](#path-operations)
- [Path and Query Parameters](#path-and-query-parameters)
- [Request Body](#request-body)
- [Response Models](#response-models)
- [Dependencies](#dependencies)
- [Authentication and Security](#authentication-and-security)
- [Database Integration](#database-integration)
- [Background Tasks](#background-tasks)
- [Middleware](#middleware)
- [WebSockets](#websockets)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Introduction

FastAPI is a modern, high-performance Python web framework for building APIs based on:
- **Python type hints** for automatic validation
- **Pydantic** for data validation
- **Starlette** for the ASGI foundation
- **OpenAPI** auto-documentation (Swagger UI + ReDoc)

**Performance:** On par with NodeJS and Go (based on Starlette and uvicorn).

```bash
pip install fastapi uvicorn[standard]

# Run the server
uvicorn main:app --reload
```

---

## Getting Started

```python
# main.py
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="A sample FastAPI application",
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc
    openapi_url="/openapi.json",
)

@app.get("/")
def root():
    return {"message": "Hello, World!"}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

```bash
# Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Access
# API:       http://localhost:8000
# Docs:      http://localhost:8000/docs
# ReDoc:     http://localhost:8000/redoc
# OpenAPI:   http://localhost:8000/openapi.json
```

---

## Path Operations

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items")           # GET
@app.post("/items")          # POST
@app.put("/items/{item_id}") # PUT
@app.patch("/items/{item_id}") # PATCH
@app.delete("/items/{item_id}") # DELETE

# Tags — group endpoints in docs
@app.get("/users", tags=["Users"])

# Summary and description
@app.get(
    "/users/{user_id}",
    tags=["Users"],
    summary="Get a user",
    description="Retrieve a user by their ID. Returns 404 if not found.",
    response_description="The user object",
)
def get_user(user_id: int):
    return {"id": user_id}

# Deprecated
@app.get("/old-endpoint", deprecated=True)
def old_endpoint():
    return {"message": "Use /new-endpoint instead"}
```

---

## Path and Query Parameters

### Path Parameters

```python
from fastapi import FastAPI, Path
from typing import Literal

app = FastAPI()

# Simple path parameter
@app.get("/items/{item_id}")
def get_item(item_id: int):   # auto-parsed and validated
    return {"item_id": item_id}

# Multiple path params
@app.get("/users/{user_id}/orders/{order_id}")
def get_order(user_id: int, order_id: int):
    return {"user_id": user_id, "order_id": order_id}

# Path param with validation
@app.get("/items/{item_id}")
def get_item(
    item_id: int = Path(gt=0, description="Item ID, must be positive")
):
    return {"item_id": item_id}

# Enum path param
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet  = "resnet"
    lenet   = "lenet"

@app.get("/models/{model_name}")
def get_model(model_name: ModelName):
    return {"model": model_name, "value": model_name.value}
```

### Query Parameters

```python
from fastapi import FastAPI, Query
from typing import Optional, List

app = FastAPI()

# Optional query params with defaults
@app.get("/items")
def list_items(
    skip:  int = 0,
    limit: int = 10,
    q:     Optional[str] = None,
    sort:  str = "name",
    order: Literal["asc", "desc"] = "asc",
):
    return {"skip": skip, "limit": limit, "q": q, "sort": sort, "order": order}

# GET /items?skip=20&limit=5&q=python&order=desc

# Query with validation
@app.get("/search")
def search(
    q:     str           = Query(min_length=3, max_length=100, description="Search query"),
    page:  int           = Query(default=1, ge=1),
    size:  int           = Query(default=20, ge=1, le=100),
    tags:  List[str]     = Query(default=[]),  # repeated: ?tags=a&tags=b
    regex: Optional[str] = Query(default=None, pattern=r"^\w+$"),
):
    return {"q": q, "page": page, "size": size, "tags": tags}

# Required query param (no default)
@app.get("/required")
def required_param(q: str):   # required — 422 if missing
    return {"q": q}
```

---

## Request Body

```python
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI()

class ItemCreate(BaseModel):
    name:        str   = Field(min_length=1, max_length=100)
    price:       float = Field(gt=0)
    description: Optional[str] = None
    tags:        List[str] = []

class ItemRead(BaseModel):
    id:          int
    name:        str
    price:       float
    description: Optional[str]
    tags:        List[str]

items_db: dict[int, dict] = {}
counter = 0

@app.post("/items", response_model=ItemRead, status_code=201)
def create_item(item: ItemCreate):
    global counter
    counter += 1
    record = {**item.model_dump(), "id": counter}
    items_db[counter] = record
    return record

# Multiple body params
class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users-with-item")
def create_user_with_item(user: UserCreate, item: ItemCreate):
    # Body: {"user": {...}, "item": {...}}
    return {"user": user, "item": item}

# Body with extra params
@app.put("/items/{item_id}")
def update_item(
    item_id:   int,
    item:      ItemCreate,
    timestamp: str = Body(default=None),  # extra body field
):
    return {"item_id": item_id, **item.model_dump(), "timestamp": timestamp}
```

---

## Response Models

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Union

app = FastAPI()

class UserBase(BaseModel):
    name:  str
    email: str

class UserCreate(UserBase):
    password: str

class UserRead(UserBase):
    id:       int
    is_active: bool = True

class UserWithOrders(UserRead):
    orders: List[dict] = []

# response_model filters output fields
@app.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate):
    # password is excluded from response (not in UserRead)
    return {"id": 1, **user.model_dump(), "is_active": True}

# response_model_exclude — exclude specific fields from response
@app.get("/users/{user_id}",
         response_model=UserRead,
         response_model_exclude={"is_active"})
def get_user(user_id: int):
    return {"id": user_id, "name": "Alice", "email": "alice@example.com", "is_active": True}

# Union response model
@app.get("/items/{item_id}", response_model=Union[ItemRead, dict])
def get_item(item_id: int):
    ...

# List response
@app.get("/users", response_model=List[UserRead])
def list_users():
    return [{"id": 1, "name": "Alice", "email": "alice@example.com"}]

# Custom response
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, FileResponse

@app.get("/redirect")
def redirect():
    return RedirectResponse(url="/")

@app.get("/plain")
def plain():
    return PlainTextResponse("Hello, World!")

@app.get("/download")
def download():
    return FileResponse("path/to/file.pdf", filename="report.pdf")
```

### HTTP Exceptions

```python
from fastapi import HTTPException, status

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found",
            headers={"X-Error": "Item not found"},
        )
    return items_db[item_id]

# Custom exception handler
from fastapi import Request
from fastapi.responses import JSONResponse

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Unicorn error: {exc.name}"},
    )
```

---

## Dependencies

FastAPI's dependency injection system is powerful and composable.

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Annotated

app = FastAPI()

# Simple dependency
def get_db():
    db = create_db_session()
    try:
        yield db
    finally:
        db.close()

# Dependency with parameters
def common_params(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Use Annotated for cleaner syntax
CommonParams = Annotated[dict, Depends(common_params)]
DBSession    = Annotated[Session, Depends(get_db)]

@app.get("/items")
def list_items(params: CommonParams, db: DBSession):
    return db.query(Item).offset(params["skip"]).limit(params["limit"]).all()

# Class-based dependencies
class QueryParams:
    def __init__(self, skip: int = 0, limit: int = 10, q: Optional[str] = None):
        self.skip  = skip
        self.limit = limit
        self.q     = q

@app.get("/search")
def search(params: Annotated[QueryParams, Depends()]):
    return {"skip": params.skip, "limit": params.limit, "q": params.q}

# Chained dependencies
def verify_token(token: str = Header(...)):
    if token != "valid-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

def get_current_user(token: Annotated[str, Depends(verify_token)], db: DBSession):
    user = db.query(User).filter(User.token == token).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/me")
def get_me(user: Annotated[User, Depends(get_current_user)]):
    return user

# Global dependencies
app = FastAPI(dependencies=[Depends(verify_token)])   # all routes require token
```

---

## Authentication and Security

### JWT Authentication

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Annotated, Optional
from datetime import datetime, timedelta
import jwt   # pip install PyJWT

SECRET_KEY = "your-secret-key"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type:   str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None

# Fake user store
fake_users = {
    "alice": {"username": "alice", "hashed_password": "fakehashed_secret", "disabled": False}
}

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> TokenData:
    try:
        payload   = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username  = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        return TokenData(username=username)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    data = verify_token(token)
    user = fake_users.get(data.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/token", response_model=Token)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = fake_users.get(form.username)
    if not user or form.password != "secret":   # simplified check
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = create_access_token(
        data={"sub": form.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token}

@app.get("/me")
async def read_me(user: Annotated[dict, Depends(get_current_user)]):
    return user
```

### API Key Auth

```python
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

VALID_API_KEYS = {"key1", "key2"}

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/secure", dependencies=[Depends(get_api_key)])
def secure_endpoint():
    return {"message": "You're authenticated!"}
```

---

## Database Integration

### SQLAlchemy with FastAPI

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List, Optional, Annotated

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI()

# ORM Model
class UserDB(Base):
    __tablename__ = "users"
    id        = Column(Integer, primary_key=True, index=True)
    name      = Column(String, index=True)
    email     = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)

Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

DB = Annotated[Session, Depends(get_db)]

# Pydantic schemas
class UserCreate(BaseModel):
    name:  str
    email: str

class UserRead(BaseModel):
    id:       int
    name:     str
    email:    str
    is_active: bool

    model_config = {"from_attributes": True}  # orm_mode in v1

# CRUD endpoints
@app.post("/users", response_model=UserRead)
def create_user(user: UserCreate, db: DB):
    db_user = UserDB(**user.model_dump())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users", response_model=List[UserRead])
def list_users(skip: int = 0, limit: int = 10, db: DB = None):
    return db.query(UserDB).offset(skip).limit(limit).all()

@app.get("/users/{user_id}", response_model=UserRead)
def get_user(user_id: int, db: DB):
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## Background Tasks

```python
from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

def send_welcome_email(email: str, name: str):
    """Runs in background — doesn't block the response."""
    time.sleep(2)   # simulate slow email sending
    print(f"Welcome email sent to {name} <{email}>")

def log_activity(user_id: int, action: str):
    print(f"User {user_id}: {action}")

@app.post("/users")
def create_user(
    name:             str,
    email:            str,
    background_tasks: BackgroundTasks,
):
    # Create user immediately
    user_id = 1  # from DB
    new_user = {"id": user_id, "name": name, "email": email}

    # Schedule background tasks
    background_tasks.add_task(send_welcome_email, email, name)
    background_tasks.add_task(log_activity, user_id, "registered")

    # Return immediately (email is sent in background)
    return new_user

# For heavier background work, use Celery or ARQ
```

---

## Middleware

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted hosts
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["example.com", "*.example.com"])

# Custom middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"→ {request.method} {request.url}")
    response = await call_next(request)
    print(f"← {response.status_code}")
    return response
```

---

## WebSockets

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI()

# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_personal(self, message: str, ws: WebSocket):
        await ws.send_text(message)


manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(ws: WebSocket, client_id: str):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await manager.send_personal(f"Echo: {data}", ws)
            await manager.broadcast(f"{client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(ws)
        await manager.broadcast(f"{client_id} left the chat")
```

---

## Testing

```python
# pip install httpx pytest
from fastapi.testclient import TestClient
import pytest
from main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hello, World!"}

def test_create_item():
    r = client.post("/items", json={"name": "Widget", "price": 9.99})
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "Widget"
    assert data["price"] == 9.99
    assert "id" in data

def test_get_item_not_found():
    r = client.get("/items/9999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()

# Async tests (pytest-asyncio)
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/")
    assert r.status_code == 200

# With auth
def test_protected():
    # First login
    r = client.post("/token", data={"username": "alice", "password": "secret"})
    token = r.json()["access_token"]

    # Then use token
    r = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
```

---

## Deployment

### Uvicorn / Gunicorn

```bash
# Development
uvicorn main:app --reload --port 8000

# Production with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or uvicorn with multiple workers
uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Router Organization (Large Apps)

```python
# routers/users.py
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/")
def list_users():
    return []

@router.get("/{user_id}")
def get_user(user_id: int):
    return {"id": user_id}


# main.py
from fastapi import FastAPI
from routers import users, posts, auth

app = FastAPI()
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(posts.router, dependencies=[Depends(get_current_user)])
```
