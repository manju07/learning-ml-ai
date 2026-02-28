# SQLAlchemy — Python SQL Toolkit and ORM

## Table of Contents
- [Introduction](#introduction)
- [Core — SQL Expression Language](#core--sql-expression-language)
- [ORM — Object Relational Mapper](#orm--object-relational-mapper)
- [Defining Models](#defining-models)
- [Sessions and Transactions](#sessions-and-transactions)
- [CRUD Operations](#crud-operations)
- [Querying](#querying)
- [Relationships](#relationships)
- [Migrations with Alembic](#migrations-with-alembic)
- [Async SQLAlchemy](#async-sqlalchemy)

---

## Introduction

SQLAlchemy has two main components:
- **Core** — SQL expression language (low-level, explicit SQL)
- **ORM** — Object Relational Mapper (high-level, Python classes)

```bash
pip install sqlalchemy
pip install psycopg2-binary   # PostgreSQL driver
pip install pymysql           # MySQL driver
pip install aiosqlite         # async SQLite
```

---

## Core — SQL Expression Language

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, text

# Create engine — connection pool to database
engine = create_engine(
    "sqlite:///example.db",
    echo=True,              # log all SQL statements
    pool_size=5,            # connection pool size
    max_overflow=10,
)

# PostgreSQL
engine = create_engine(
    "postgresql+psycopg2://user:password@localhost:5432/dbname",
    pool_size=10,
    echo=False,
)

# Execute raw SQL
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchone())

    # Parameterized query (safe from SQL injection)
    result = conn.execute(
        text("SELECT * FROM users WHERE age > :min_age"),
        {"min_age": 25}
    )
    for row in result:
        print(row._mapping)    # access by column name
```

### Table Definition and DDL

```python
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, Boolean, Float, Text,
    DateTime, ForeignKey, Index, UniqueConstraint,
)
from datetime import datetime

metadata = MetaData()

users = Table("users", metadata,
    Column("id",         Integer, primary_key=True),
    Column("name",       String(100), nullable=False),
    Column("email",      String(200), nullable=False),
    Column("age",        Integer),
    Column("is_active",  Boolean, default=True),
    Column("created_at", DateTime, default=datetime.utcnow),

    UniqueConstraint("email", name="uq_user_email"),
    Index("ix_user_name", "name"),
)

posts = Table("posts", metadata,
    Column("id",        Integer, primary_key=True),
    Column("title",     String(200), nullable=False),
    Column("content",   Text),
    Column("user_id",   Integer, ForeignKey("users.id", ondelete="CASCADE")),
    Column("created_at", DateTime, default=datetime.utcnow),
)

# Create all tables
metadata.create_all(engine)

# Drop all tables
metadata.drop_all(engine)
```

### Core DML

```python
from sqlalchemy import insert, select, update, delete, and_, or_, not_, func

with engine.connect() as conn:
    # INSERT
    conn.execute(insert(users).values(
        name="Alice", email="alice@example.com", age=30
    ))
    conn.execute(insert(users), [
        {"name": "Bob",   "email": "bob@example.com",   "age": 25},
        {"name": "Carol", "email": "carol@example.com", "age": 35},
    ])
    conn.commit()

    # SELECT
    result = conn.execute(select(users))
    for row in result:
        print(row.id, row.name, row.email)

    # WHERE clause
    stmt = (
        select(users.c.name, users.c.email)
        .where(users.c.age >= 25)
        .where(users.c.is_active == True)
        .order_by(users.c.name)
        .limit(10)
        .offset(0)
    )
    result = conn.execute(stmt)
    print(result.fetchall())

    # UPDATE
    conn.execute(
        update(users)
        .where(users.c.id == 1)
        .values(age=31)
    )
    conn.commit()

    # DELETE
    conn.execute(
        delete(users)
        .where(users.c.is_active == False)
    )
    conn.commit()

    # Aggregate functions
    count = conn.execute(select(func.count()).select_from(users)).scalar()
    avg   = conn.execute(select(func.avg(users.c.age))).scalar()
    max_  = conn.execute(select(func.max(users.c.age))).scalar()
```

---

## ORM — Object Relational Mapper

### Setup

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session

engine = create_engine("sqlite:///example.db", echo=False)

class Base(DeclarativeBase):
    pass
```

---

## Defining Models

```python
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, Text,
    DateTime, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import DeclarativeBase, relationship, mapped_column, Mapped
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List
import enum

class Base(DeclarativeBase):
    pass

class UserRole(enum.Enum):
    admin  = "admin"
    user   = "user"
    guest  = "guest"

class User(Base):
    __tablename__ = "users"

    # Modern syntax (SQLAlchemy 2.0+)
    id:         Mapped[int]           = mapped_column(Integer, primary_key=True)
    name:       Mapped[str]           = mapped_column(String(100), nullable=False)
    email:      Mapped[str]           = mapped_column(String(200), unique=True, nullable=False)
    age:        Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active:  Mapped[bool]          = mapped_column(Boolean, default=True)
    role:       Mapped[UserRole]      = mapped_column(SAEnum(UserRole), default=UserRole.user)
    created_at: Mapped[datetime]      = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime]      = mapped_column(DateTime, onupdate=func.now(), nullable=True)

    # Relationships
    posts:    Mapped[List["Post"]]    = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    profile:  Mapped[Optional["Profile"]] = relationship("Profile", back_populates="user", uselist=False)

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name!r}, email={self.email!r})>"


class Post(Base):
    __tablename__ = "posts"

    id:        Mapped[int]           = mapped_column(Integer, primary_key=True)
    title:     Mapped[str]           = mapped_column(String(200))
    content:   Mapped[Optional[str]] = mapped_column(Text)
    user_id:   Mapped[int]           = mapped_column(ForeignKey("users.id"))
    published: Mapped[bool]          = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime]     = mapped_column(DateTime, server_default=func.now())

    author: Mapped["User"] = relationship("User", back_populates="posts")
    tags:   Mapped[List["Tag"]] = relationship("Tag", secondary="post_tags")

    def __repr__(self):
        return f"<Post(id={self.id}, title={self.title!r})>"


# Association table for many-to-many
from sqlalchemy import Table as SATable

post_tags = SATable("post_tags", Base.metadata,
    Column("post_id", ForeignKey("posts.id"), primary_key=True),
    Column("tag_id",  ForeignKey("tags.id"),  primary_key=True),
)

class Tag(Base):
    __tablename__ = "tags"
    id:   Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)


class Profile(Base):
    __tablename__ = "profiles"
    id:      Mapped[int]           = mapped_column(primary_key=True)
    user_id: Mapped[int]           = mapped_column(ForeignKey("users.id"), unique=True)
    bio:     Mapped[Optional[str]] = mapped_column(Text)
    avatar:  Mapped[Optional[str]] = mapped_column(String(500))

    user: Mapped["User"] = relationship("User", back_populates="profile")


# Create all tables
Base.metadata.create_all(engine)
```

---

## Sessions and Transactions

```python
from sqlalchemy.orm import Session

# Session — unit of work
with Session(engine) as session:
    # All operations within a session
    user = User(name="Alice", email="alice@example.com")
    session.add(user)
    session.commit()   # flush + commit transaction
    print(user.id)     # available after commit

# Session as context manager (auto-close)
with Session(engine) as session:
    with session.begin():    # transaction context
        user = session.get(User, 1)
        user.name = "Alice Updated"
    # auto-committed on exit, rolled back on exception

# sessionmaker — factory for sessions
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Dependency injection pattern (FastAPI/Flask style)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## CRUD Operations

```python
from sqlalchemy.orm import Session

with Session(engine) as session:
    # CREATE
    user = User(name="Alice", email="alice@example.com", age=30)
    session.add(user)

    # Add multiple
    session.add_all([
        User(name="Bob",   email="bob@example.com"),
        User(name="Carol", email="carol@example.com"),
    ])
    session.commit()
    print(user.id)   # populated after commit

    # READ — get by primary key
    user = session.get(User, 1)            # None if not found
    user = session.get(User, 1, options=[...])  # with options

    # READ — query
    users = session.execute(
        select(User).where(User.is_active == True)
    ).scalars().all()

    # UPDATE
    user = session.get(User, 1)
    if user:
        user.age = 31
        session.commit()

    # Bulk update
    session.execute(
        update(User)
        .where(User.age < 18)
        .values(is_active=False)
    )
    session.commit()

    # DELETE
    user = session.get(User, 1)
    if user:
        session.delete(user)
        session.commit()

    # Bulk delete
    session.execute(
        delete(User).where(User.is_active == False)
    )
    session.commit()
```

---

## Querying

```python
from sqlalchemy import select, and_, or_, not_, func, desc, asc, between
from sqlalchemy.orm import Session

with Session(engine) as session:
    # Basic select
    users = session.execute(select(User)).scalars().all()
    first  = session.execute(select(User)).scalars().first()
    one    = session.execute(select(User).where(User.id == 1)).scalar_one()
    one_or_none = session.execute(select(User).where(User.id == 1)).scalar_one_or_none()

    # WHERE
    stmt = (
        select(User)
        .where(User.age >= 25)
        .where(User.is_active == True)
    )

    # AND / OR
    stmt = select(User).where(
        and_(User.age >= 25, User.is_active == True)
    )
    stmt = select(User).where(
        or_(User.role == UserRole.admin, User.age > 50)
    )
    stmt = select(User).where(not_(User.is_active))

    # IN / NOT IN
    stmt = select(User).where(User.role.in_([UserRole.admin, UserRole.user]))
    stmt = select(User).where(User.id.not_in([1, 2, 3]))

    # LIKE / ILIKE
    stmt = select(User).where(User.name.like("A%"))      # starts with A
    stmt = select(User).where(User.email.ilike("%@example%"))  # case-insensitive

    # BETWEEN
    stmt = select(User).where(between(User.age, 20, 40))
    stmt = select(User).where(User.age.between(20, 40))  # same

    # IS NULL / IS NOT NULL
    stmt = select(User).where(User.updated_at.is_(None))
    stmt = select(User).where(User.updated_at.is_not(None))

    # ORDER BY
    stmt = select(User).order_by(User.name)
    stmt = select(User).order_by(desc(User.created_at))
    stmt = select(User).order_by(User.role, desc(User.age))

    # LIMIT / OFFSET
    stmt = select(User).limit(10).offset(20)   # page 3 with 10 per page

    # Distinct
    stmt = select(User.role).distinct()

    # Aggregate
    count = session.execute(select(func.count(User.id))).scalar()
    avg   = session.execute(select(func.avg(User.age))).scalar()
    max_  = session.execute(select(func.max(User.age))).scalar()

    # GROUP BY
    from sqlalchemy import literal_column
    stmt = (
        select(User.role, func.count(User.id).label("count"))
        .group_by(User.role)
        .having(func.count(User.id) > 5)
        .order_by(desc(literal_column("count")))
    )
    result = session.execute(stmt).all()
    for role, count in result:
        print(f"{role}: {count}")

    # JOIN
    stmt = (
        select(User, Post)
        .join(Post, Post.user_id == User.id)
        .where(Post.published == True)
    )

    # Outer join
    stmt = (
        select(User, Post)
        .outerjoin(Post, Post.user_id == User.id)
    )

    # Subquery
    avg_age = select(func.avg(User.age)).scalar_subquery()
    above_avg = session.execute(
        select(User).where(User.age > avg_age)
    ).scalars().all()

    # Exists
    from sqlalchemy import exists
    has_posts = session.execute(
        select(User).where(
            exists().where(Post.user_id == User.id)
        )
    ).scalars().all()
```

---

## Relationships

```python
from sqlalchemy.orm import joinedload, selectinload, lazyload, subqueryload

with Session(engine) as session:
    # Eager loading — load related objects in same query
    # joinedload — JOIN (good for single-row relationships, e.g., many-to-one)
    users = session.execute(
        select(User)
        .options(joinedload(User.profile))   # load profile with JOIN
    ).unique().scalars().all()

    # selectinload — SELECT IN (good for one-to-many collections)
    users = session.execute(
        select(User)
        .options(selectinload(User.posts))   # SELECT posts WHERE user_id IN (...)
    ).scalars().all()

    for user in users:
        print(f"{user.name}: {len(user.posts)} posts")

    # Nested eager loading
    users = session.execute(
        select(User)
        .options(
            selectinload(User.posts).selectinload(Post.tags)
        )
    ).scalars().all()

    # Working with relationships
    user = session.get(User, 1)
    post = Post(title="My Post", content="Content here")
    user.posts.append(post)   # add to relationship
    session.commit()

    # Many-to-many
    tag1 = Tag(name="python")
    tag2 = Tag(name="tutorial")
    session.add_all([tag1, tag2])
    session.flush()

    post.tags.append(tag1)
    post.tags.append(tag2)
    session.commit()

    # Query through relationship
    python_posts = session.execute(
        select(Post).where(Post.tags.any(Tag.name == "python"))
    ).scalars().all()
```

---

## Migrations with Alembic

```bash
pip install alembic

# Initialize
alembic init migrations

# Generate migration
alembic revision --autogenerate -m "add users table"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1   # one revision back
alembic downgrade base # all the way back
```

```python
# alembic/env.py — configure for your models
from myapp.models import Base   # import your models
target_metadata = Base.metadata

# migrations/versions/001_add_users.py (auto-generated)
def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id",    sa.Integer, primary_key=True),
        sa.Column("name",  sa.String(100)),
        sa.Column("email", sa.String(200), unique=True),
    )

def downgrade() -> None:
    op.drop_table("users")
```

---

## Async SQLAlchemy

```bash
pip install sqlalchemy[asyncio] aiosqlite asyncpg
```

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# Async engine
engine = create_async_engine(
    "sqlite+aiosqlite:///async_example.db",
    echo=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_users():
    async with AsyncSession(engine) as session:
        result = await session.execute(select(User))
        return result.scalars().all()

async def create_user(name: str, email: str):
    async with AsyncSession(engine) as session:
        user = User(name=name, email=email)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

# FastAPI integration
from fastapi import FastAPI, Depends
from typing import AsyncGenerator, Annotated

app = FastAPI()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

DB = Annotated[AsyncSession, Depends(get_db)]

@app.get("/users")
async def list_users(db: DB):
    result = await db.execute(select(User))
    return result.scalars().all()

@app.post("/users")
async def create_user_endpoint(name: str, email: str, db: DB):
    user = User(name=name, email=email)
    db.add(user)
    await db.flush()
    return user

# Lifespan for DB initialization
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(lifespan=lifespan)
```
