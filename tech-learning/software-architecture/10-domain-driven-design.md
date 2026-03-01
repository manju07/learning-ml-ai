# Domain-Driven Design (DDD): Guide for Architects

## Table of Contents
1. [Introduction to DDD](#1-introduction-to-ddd)
2. [Strategic Design: Bounded Contexts](#2-strategic-design-bounded-contexts)
3. [Ubiquitous Language](#3-ubiquitous-language)
4. [Tactical Design: Aggregates and Entities](#4-tactical-design-aggregates-and-entities)
5. [Value Objects](#5-value-objects)
6. [Domain Events](#6-domain-events)
7. [Repositories and Domain Services](#7-repositories-and-domain-services)
8. [Anti-Corruption Layer](#8-anti-corruption-layer)
9. [DDD and Microservices](#9-ddd-and-microservices)
10. [Practical Examples](#10-practical-examples)

---

## 1. Introduction to DDD

### 1.1 What Is DDD?

**Domain-Driven Design** is an approach to software design that places the **domain model** at the heart of the system. The domain is the problem space the software addresses.

### 1.2 Core Philosophy

- **Collaboration** between technical and domain experts
- **Ubiquitous language**: Same terms in code, docs, conversations
- **Model richness**: Capture domain logic in the model, not in procedures
- **Iterative refinement**: Model evolves with understanding

### 1.3 When to Use DDD

- Complex business logic
- Multiple stakeholders with domain expertise
- Long-lived systems
- When correctness of domain rules matters

---

## 2. Strategic Design: Bounded Contexts

### 2.1 Bounded Context

A boundary within which a particular **model** and **ubiquitous language** apply. Different contexts can have different models for the same concept.

Example: In **Sales** context, "Customer" might mean "buyer with credit limit." In **Shipping** context, "Customer" might mean "delivery address and contact."

### 2.2 Context Map

Relationships between bounded contexts:

| Relationship | Description | Example |
|--------------|-------------|---------|
| **Shared Kernel** | Shared subset of model | Common types |
| **Customer-Supplier** | Downstream depends on upstream | Order → Payment |
| **Conformist** | Downstream conforms to upstream | Adapter to legacy |
| **ACL** | Anti-Corruption Layer isolates | External API |
| **Open Host** | Published protocol | REST API |
| **Published Language** | Canonical format (e.g., JSON schema) | Event schema |

### 2.3 Example: E-Commerce Context Map

```
[Order Context] --(ACL)--> [Payment Gateway]
       |
       | (Customer-Supplier)
       v
[Inventory Context]
       |
       | (ACL)
       v
[Legacy ERP]
```

---

## 3. Ubiquitous Language

### 3.1 Definition

A common vocabulary used by developers and domain experts. Terms from the language appear in:

- Code (class names, methods)
- Documentation
- Conversations
- Tests

### 3.2 Example

| Ubiquitous Term | Code Representation |
|-----------------|---------------------|
| Order | `Order` aggregate |
| Order Line | `OrderLine` value object |
| Place Order | `Order.place()` |
| Shipment | `Shipment` aggregate |
| Backorder | `Backorder` domain event |

---

## 4. Tactical Design: Aggregates and Entities

### 4.1 Entity

Has **identity** that persists over time. Two entities with same attributes but different IDs are different.

```python
class Order(Entity):
    def __init__(self, id: OrderId, customer_id: CustomerId, ...):
        self.id = id  # Identity
        self.customer_id = customer_id
        self.lines: list[OrderLine] = []
```

### 4.2 Aggregate

A **cluster of entities and value objects** with a **root** (aggregate root). External references point only to the root. Consistency boundary for invariants.

```
Order (Aggregate Root)
  ├── OrderLine (Entity/Value)
  ├── OrderLine
  └── ShippingAddress (Value Object)
```

### 4.3 Aggregate Rules

- External objects hold reference only to **root**
- Changes go through root
- Transaction boundary = one aggregate (or use eventual consistency)
- Load/save whole aggregate

### 4.4 Example: Order Aggregate

```python
class Order:
    def __init__(self, id: OrderId, customer_id: CustomerId):
        self.id = id
        self.customer_id = customer_id
        self.lines: list[OrderLine] = []
        self.status = OrderStatus.DRAFT

    def add_line(self, product_id: ProductId, quantity: int, unit_price: Money):
        if self.status != OrderStatus.DRAFT:
            raise DomainError("Cannot modify submitted order")
        self.lines.append(OrderLine(product_id, quantity, unit_price))

    def submit(self):
        if not self.lines:
            raise DomainError("Order must have at least one line")
        self.status = OrderStatus.SUBMITTED
        self.raise_event(OrderSubmitted(self.id, self.customer_id, self.total()))
```

---

## 5. Value Objects

### 5.1 Definition

Defined by **attributes**, not identity. Immutable. Two value objects with same attributes are interchangeable.

### 5.2 Examples

- `Money(amount, currency)`
- `Address(street, city, zip)`
- `Email(address)`
- `OrderLine(product_id, quantity, unit_price)`

### 5.3 Implementation

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("Amount must be non-negative")

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError("Currency mismatch")
        return Money(self.amount + other.amount, self.currency)

@dataclass(frozen=True)
class OrderLine:
    product_id: str
    quantity: int
    unit_price: Money

    def total(self) -> Money:
        return Money(
            self.unit_price.amount * self.quantity,
            self.unit_price.currency
        )
```

---

## 6. Domain Events

### 6.1 Definition

Something that **happened** in the domain. Past tense. Immutable.

### 6.2 Why Domain Events?

- Decouple aggregates
- Audit trail
- Trigger side effects (notifications, integrations)
- Consistency across aggregates (eventual)

### 6.3 Example

```python
@dataclass(frozen=True)
class OrderSubmitted:
    order_id: str
    customer_id: str
    total: Money
    occurred_at: datetime = field(default_factory=datetime.utcnow)
```

### 6.4 Raising and Handling

```python
class Order:
    def __init__(self, ...):
        self._domain_events: list = []

    def submit(self):
        # ... validation ...
        self.status = OrderStatus.SUBMITTED
        self._domain_events.append(
            OrderSubmitted(self.id, self.customer_id, self.total())
        )

    def pull_domain_events(self) -> list:
        events = self._domain_events.copy()
        self._domain_events.clear()
        return events
```

---

## 7. Repositories and Domain Services

### 7.1 Repository

Abstracts persistence. Collection-like interface for aggregates.

```python
class OrderRepository(Protocol):
    def get(self, id: OrderId) -> Optional[Order]: ...
    def add(self, order: Order) -> None: ...
    def remove(self, order: Order) -> None: ...

# Usage
order = order_repo.get(order_id)
order.add_line(product_id, quantity, price)
order_repo.add(order)  # or repo.save(order)
```

### 7.2 Domain Service

Stateless operation that doesn't naturally fit an entity or value object.

```python
class PricingService:
    def calculate_discount(self, order: Order, customer: Customer) -> Money:
        # Complex discount logic across Order and Customer
        ...
```

### 7.3 Application Service vs Domain Service

- **Application service**: Orchestrates use case, transactions, calls domain
- **Domain service**: Pure domain logic, no infrastructure

---

## 8. Anti-Corruption Layer (ACL)

### 8.1 Purpose

Protect your domain from **external models** that don't match. Translate at the boundary.

```
[Your Domain] <--ACL--> [External System]
         Your model    Translation    Their model
```

### 8.2 Example

```python
class PaymentGatewayACL:
    def __init__(self, gateway_client: PaymentGatewayClient):
        self.client = gateway_client

    def charge(self, order: Order) -> Result[PaymentRef, PaymentError]:
        # Translate domain model to gateway model
        request = ChargeRequest(
            amount=order.total().amount,
            currency=order.total().currency,
            reference=order.id.value,
            customer_email=self._get_customer_email(order.customer_id),
        )
        response = self.client.charge(request)
        # Translate gateway response to domain
        return self._to_domain_result(response)
```

---

## 9. DDD and Microservices

### 9.1 Mapping

- **Bounded context** → Candidate **microservice**
- **Aggregate** → Consistency boundary, often one service owns one aggregate type
- **Domain events** → Integration between services

### 9.2 Service Boundaries

Align services with bounded contexts. Avoid splitting a context across services (distributed monolith risk).

---

## 10. Practical Examples

### 10.1 Order Aggregate (Complete)

```python
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List

class OrderStatus(Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PAID = "paid"
    SHIPPED = "shipped"

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str = "USD"

@dataclass(frozen=True)
class OrderLine:
    product_id: str
    quantity: int
    unit_price: Money

    def total(self) -> Money:
        return Money(self.unit_price.amount * self.quantity, self.unit_price.currency)

@dataclass
class Order:
    id: str
    customer_id: str
    lines: List[OrderLine]
    status: OrderStatus
    created_at: datetime
    _events: List = field(default_factory=list, repr=False)

    def add_line(self, product_id: str, quantity: int, unit_price: Money):
        if self.status != OrderStatus.DRAFT:
            raise ValueError("Cannot modify submitted order")
        self.lines.append(OrderLine(product_id, quantity, unit_price))

    def total(self) -> Money:
        return Money(
            sum(l.total().amount for l in self.lines),
            "USD"
        )

    def submit(self):
        if not self.lines:
            raise ValueError("Order must have lines")
        self.status = OrderStatus.SUBMITTED
        self._events.append(OrderSubmitted(self.id, self.customer_id, self.total()))

    def pull_events(self):
        events = self._events.copy()
        self._events.clear()
        return events
```

### 10.2 Repository Implementation

```python
class SqlOrderRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def get(self, order_id: str) -> Optional[Order]:
        with self.session_factory() as session:
            row = session.query(OrderRow).filter_by(id=order_id).first()
            return self._to_domain(row) if row else None

    def add(self, order: Order):
        with self.session_factory() as session:
            row = self._to_row(order)
            session.add(row)
            session.commit()
```

### 10.3 Application Service

```python
class PlaceOrderUseCase:
    def __init__(self, order_repo: OrderRepository, event_bus: EventBus):
        self.order_repo = order_repo
        self.event_bus = event_bus

    def execute(self, cmd: PlaceOrderCommand):
        order = Order.create(cmd.customer_id)
        for item in cmd.items:
            order.add_line(item.product_id, item.quantity, item.unit_price)
        order.submit()
        self.order_repo.add(order)
        for event in order.pull_events():
            self.event_bus.publish(event)
        return order.id
```

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Bounded Context** | Boundary for model and language |
| **Ubiquitous Language** | Shared vocabulary in code and conversation |
| **Aggregate** | Consistency boundary, root is single entry |
| **Value Object** | Immutable, defined by attributes |
| **Domain Event** | Something that happened, past tense |
| **Repository** | Collection-like persistence abstraction |
| **ACL** | Isolate external systems from domain |

---

## Further Reading

- *Domain-Driven Design* — Eric Evans
- *Implementing Domain-Driven Design* — Vaughn Vernon
- https://www.domainlanguage.com/ddd/
