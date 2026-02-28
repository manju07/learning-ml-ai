# Pydantic — Data Validation with Python Type Hints

## Table of Contents
- [Introduction](#introduction)
- [BaseModel](#basemodel)
- [Field Customization](#field-customization)
- [Validators](#validators)
- [Nested Models](#nested-models)
- [Serialization](#serialization)
- [Configuration](#configuration)
- [Generic Models](#generic-models)
- [Pydantic Settings](#pydantic-settings)
- [Integration with FastAPI](#integration-with-fastapi)

---

## Introduction

Pydantic provides **data validation** and **settings management** using Python type annotations. It's the backbone of FastAPI.

```bash
pip install pydantic    # v2 (current)
```

```python
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int
    email: str

user = User(name="Alice", age=30, email="alice@example.com")
print(user)                 # name='Alice' age=30 email='alice@example.com'
print(user.name)            # Alice
print(user.model_dump())    # {'name': 'Alice', 'age': 30, 'email': '...'}

# Type coercion
user2 = User(name="Bob", age="25", email="bob@example.com")  # "25" → 25
print(user2.age, type(user2.age))   # 25 <class 'int'>

# Validation errors
try:
    invalid = User(name="Carol", age="not-a-number", email="carol@example.com")
except ValidationError as e:
    print(e)
    # 1 validation error for User
    # age
    #   Input should be a valid integer [...]
```

---

## BaseModel

```python
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Address(BaseModel):
    street: str
    city:   str
    state:  str
    zip:    str

class User(BaseModel):
    id:         int
    name:       str
    email:      str
    age:        Optional[int] = None     # optional — defaults to None
    is_active:  bool = True              # default value
    tags:       List[str] = []           # default empty list
    address:    Optional[Address] = None
    created_at: datetime = None

    class Config:
        # Pydantic v2 uses model_config instead
        pass


# Creating instances
user = User(id=1, name="Alice", email="alice@example.com")
print(user.is_active)    # True (default)
print(user.tags)         # []

# From dict
data = {"id": 1, "name": "Bob", "email": "bob@example.com", "age": 25}
user = User(**data)
user = User.model_validate(data)   # v2 preferred

# Immutability (copy on mutation)
user2 = user.model_copy(update={"name": "Bobby", "age": 26})
```

### Type Support

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple, Set, Union, Literal, Any
from enum import Enum
from decimal import Decimal
from uuid import UUID

class Status(str, Enum):
    active   = "active"
    inactive = "inactive"
    pending  = "pending"

class Product(BaseModel):
    id:          UUID
    name:        str
    price:       Decimal
    status:      Status = Status.active
    tags:        Set[str] = set()
    dimensions:  Tuple[float, float, float]  # (length, width, height)
    metadata:    Dict[str, Any] = {}
    category:    Literal["electronics", "clothing", "food"]
    sku:         Optional[str] = None

from uuid import uuid4
product = Product(
    id=uuid4(),
    name="Widget",
    price="19.99",         # str → Decimal
    dimensions=(10, 5, 3),
    category="electronics",
)
print(product.price)       # Decimal('19.99')
print(product.status)      # Status.active
```

---

## Field Customization

```python
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    id:         int   = Field(gt=0, description="User ID, must be positive")
    name:       str   = Field(min_length=2, max_length=50)
    email:      str   = Field(pattern=r"^[\w.-]+@[\w.-]+\.\w+$")
    age:        Optional[int] = Field(default=None, ge=0, le=150)
    salary:     float = Field(ge=0, le=1_000_000, default=0.0)
    bio:        str   = Field(default="", max_length=500)
    score:      float = Field(default=0.0, ge=0.0, le=100.0, multiple_of=0.5)

    # Alias — use different names for input vs attribute
    user_name:  str   = Field(alias="username")         # input: "username"
    created_at: str   = Field(alias="created-at")       # input: "created-at" (hyphen)

    class model_config:
        populate_by_name = True   # allow using field name OR alias

user = User(id=1, name="Alice", email="alice@example.com", username="alice")
print(user.user_name)   # alice

# Validation constraints for numerics
# gt = greater than, ge = greater than or equal to
# lt = less than,    le = less than or equal to
# multiple_of, ...

# Validation constraints for strings
# min_length, max_length, pattern (regex)

# Validation for lists
from typing import List
class Model(BaseModel):
    items: List[int] = Field(min_length=1, max_length=10)
```

---

## Validators

### Field Validators

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class User(BaseModel):
    name:  str
    email: str
    age:   int

    @field_validator("name")
    @classmethod
    def name_must_be_title(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("email")
    @classmethod
    def email_must_be_lowercase(cls, v: str) -> str:
        v = v.lower().strip()
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Age must be positive")
        return v

    # Validate after all other fields are set
    @field_validator("email", mode="after")
    @classmethod
    def check_email_domain(cls, v: str) -> str:
        blocked_domains = ["tempmail.com", "throwaway.email"]
        domain = v.split("@")[1]
        if domain in blocked_domains:
            raise ValueError(f"Email domain '{domain}' is not allowed")
        return v


user = User(name="alice smith", email="ALICE@Example.COM", age=30)
print(user.name)    # Alice Smith
print(user.email)   # alice@example.com
```

### Model Validators

```python
from pydantic import BaseModel, model_validator
from typing import Self

class PasswordModel(BaseModel):
    password:         str
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self) -> Self:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class DateRange(BaseModel):
    start_date: str
    end_date:   str

    @model_validator(mode="before")
    @classmethod
    def check_dates(cls, data: dict) -> dict:
        # 'before' runs before field validation
        if "start_date" in data and "end_date" in data:
            if data["start_date"] > data["end_date"]:
                raise ValueError("start_date must be before end_date")
        return data


try:
    p = PasswordModel(password="secret", confirm_password="different")
except ValidationError as e:
    print(e.errors()[0]["msg"])  # Value error, Passwords do not match
```

### Custom Types

```python
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema
from typing import Any

class PhoneNumber:
    """Custom type for phone numbers."""

    def __init__(self, number: str):
        self.number = self._clean(number)

    def _clean(self, n: str) -> str:
        digits = "".join(c for c in n if c.isdigit())
        if len(digits) not in (10, 11):
            raise ValueError(f"Invalid phone number: {n}")
        return f"+1-{digits[-10:-7]}-{digits[-7:-4]}-{digits[-4:]}"

    def __repr__(self):
        return f"PhoneNumber({self.number!r})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls)


class Contact(BaseModel):
    name:  str
    phone: PhoneNumber

c = Contact(name="Alice", phone="5551234567")
print(c.phone)   # PhoneNumber('+1-555-123-4567')
```

---

## Nested Models

```python
from pydantic import BaseModel
from typing import List, Optional

class Address(BaseModel):
    street: str
    city:   str
    state:  str

class Order(BaseModel):
    product: str
    quantity: int
    price:   float

class User(BaseModel):
    name:    str
    email:   str
    address: Address
    orders:  List[Order] = []

# Create with nested dict (auto-parsed!)
user = User(
    name="Alice",
    email="alice@example.com",
    address={"street": "123 Main St", "city": "Springfield", "state": "IL"},
    orders=[
        {"product": "Widget", "quantity": 2, "price": 9.99},
        {"product": "Gadget", "quantity": 1, "price": 29.99},
    ]
)

print(user.address.city)          # Springfield
print(user.orders[0].product)     # Widget

# Recursive / self-referential models
class TreeNode(BaseModel):
    value:    int
    children: List["TreeNode"] = []

root = TreeNode(
    value=1,
    children=[
        TreeNode(value=2, children=[TreeNode(value=4)]),
        TreeNode(value=3),
    ]
)
```

---

## Serialization

```python
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    name:       str
    email:      str
    created_at: datetime = None
    password:   str      = Field(exclude=True)   # excluded from output

user = User(name="Alice", email="alice@example.com",
            created_at=datetime.now(), password="secret")

# To dict
d = user.model_dump()
d = user.model_dump(exclude={"password"})           # exclude fields
d = user.model_dump(include={"name", "email"})      # include only these
d = user.model_dump(exclude_none=True)              # exclude None values
d = user.model_dump(exclude_unset=True)             # exclude fields not set by user
d = user.model_dump(by_alias=True)                  # use field aliases

# To JSON
json_str = user.model_dump_json()
json_str = user.model_dump_json(indent=2)
json_str = user.model_dump_json(exclude={"password"})

# From dict
user2 = User.model_validate({"name": "Bob", "email": "bob@example.com"})

# From JSON
user3 = User.model_validate_json('{"name": "Carol", "email": "carol@example.com"}')

# Custom serialization
from pydantic import field_serializer

class Event(BaseModel):
    name: str
    date: datetime

    @field_serializer("date")
    def serialize_date(self, value: datetime) -> str:
        return value.strftime("%Y-%m-%d")

e = Event(name="Meeting", date=datetime(2024, 3, 15))
print(e.model_dump())  # {'name': 'Meeting', 'date': '2024-03-15'}
```

---

## Configuration

```python
from pydantic import BaseModel, ConfigDict

class StrictUser(BaseModel):
    model_config = ConfigDict(
        strict=True,               # no type coercion
        frozen=True,               # immutable (like frozen dataclass)
        populate_by_name=True,     # allow field name OR alias
        extra="forbid",            # error on extra fields (default: "ignore")
        str_strip_whitespace=True, # auto-strip strings
        str_to_lower=True,         # auto-lowercase strings
        validate_default=True,     # also validate default values
        use_enum_values=True,      # store enum value, not enum instance
        validate_assignment=True,  # re-validate on attribute assignment
        arbitrary_types_allowed=True,  # allow non-pydantic types
    )
    name:  str
    email: str

# strict mode — no coercion
try:
    u = StrictUser(name="Alice", email="alice@example.com", extra_field="bad")
except ValidationError as e:
    print("Extra field rejected")

# frozen — immutable
u = StrictUser(name="Alice", email="alice@example.com")
try:
    u.name = "Bob"   # ValidationError: Instance is frozen
except Exception as e:
    print(e)
```

---

## Generic Models

```python
from pydantic import BaseModel
from typing import TypeVar, Generic, Optional, List

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated API response."""
    items:    List[T]
    total:    int
    page:     int
    per_page: int
    has_next: bool
    has_prev: bool

    @property
    def total_pages(self) -> int:
        return (self.total + self.per_page - 1) // self.per_page


class APIResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    success: bool
    data:    Optional[T] = None
    error:   Optional[str] = None
    message: str = ""


class User(BaseModel):
    id:   int
    name: str

# Typed usage
user_response: APIResponse[User] = APIResponse(
    success=True,
    data=User(id=1, name="Alice"),
)

paginated_users: PaginatedResponse[User] = PaginatedResponse(
    items=[User(id=1, name="Alice"), User(id=2, name="Bob")],
    total=50, page=1, per_page=10, has_next=True, has_prev=False,
)
print(paginated_users.total_pages)  # 5
```

---

## Pydantic Settings

Manage configuration from environment variables, `.env` files, etc.

```bash
pip install pydantic-settings
```

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",         # load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,    # DATABASE_URL == database_url
        env_prefix="APP_",       # env var: APP_DATABASE_URL
        extra="ignore",          # ignore extra env vars
    )

    # App settings
    app_name:    str  = "My Application"
    debug:       bool = False
    secret_key:  str  = Field(..., min_length=32)  # required, no default

    # Database
    database_url: str = "sqlite:///./default.db"

    # Redis
    redis_url:   str  = "redis://localhost:6379/0"

    # External services
    stripe_key:  str  = ""
    sendgrid_key: str = ""

    # Numeric settings
    max_workers: int  = 4
    timeout:     float = 30.0


# .env file:
# SECRET_KEY=supersecretkey1234567890abcdef
# DATABASE_URL=postgresql://user:pass@localhost/mydb
# DEBUG=true

settings = Settings()  # reads from environment + .env
print(settings.database_url)
print(settings.debug)

# Singleton pattern
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

config = get_settings()
```

---

## Integration with FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

app = FastAPI()

class UserCreate(BaseModel):
    name:  str   = Field(min_length=2, max_length=50)
    email: str   = Field(pattern=r"[\w.-]+@[\w.-]+\.\w+")
    age:   Optional[int] = Field(default=None, ge=0, le=150)

class UserRead(BaseModel):
    id:         int
    name:       str
    email:      str
    age:        Optional[int]
    created_at: datetime

class UserUpdate(BaseModel):
    name:  Optional[str]  = None
    email: Optional[str]  = None
    age:   Optional[int]  = None

# In-memory store
db: dict[int, dict] = {}
counter = 0

@app.post("/users", response_model=UserRead, status_code=201)
def create_user(user: UserCreate):
    global counter
    counter += 1
    record = {**user.model_dump(), "id": counter, "created_at": datetime.now()}
    db[counter] = record
    return record

@app.get("/users", response_model=List[UserRead])
def list_users():
    return list(db.values())

@app.get("/users/{user_id}", response_model=UserRead)
def get_user(user_id: int):
    if user_id not in db:
        raise HTTPException(status_code=404, detail="User not found")
    return db[user_id]

@app.patch("/users/{user_id}", response_model=UserRead)
def update_user(user_id: int, update: UserUpdate):
    if user_id not in db:
        raise HTTPException(status_code=404, detail="User not found")
    updates = update.model_dump(exclude_none=True)  # only provided fields
    db[user_id].update(updates)
    return db[user_id]

@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int):
    if user_id not in db:
        raise HTTPException(status_code=404, detail="User not found")
    del db[user_id]
```
