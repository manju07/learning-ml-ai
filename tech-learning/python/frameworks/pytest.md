# Pytest — Python Testing Framework

## Table of Contents
- [Introduction](#introduction)
- [Writing Tests](#writing-tests)
- [Assertions](#assertions)
- [Fixtures](#fixtures)
- [Parametrize](#parametrize)
- [Marks and Skipping](#marks-and-skipping)
- [Mocking](#mocking)
- [Coverage](#coverage)
- [Plugins](#plugins)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

---

## Introduction

pytest is the most popular Python testing framework. It's simple, powerful, and extensible.

```bash
pip install pytest pytest-cov pytest-asyncio

# Run all tests
pytest

# Verbose
pytest -v

# Specific file or test
pytest tests/test_users.py
pytest tests/test_users.py::test_create_user

# Run tests matching a pattern
pytest -k "user"          # matches any test with "user" in name
pytest -k "not slow"      # excludes tests marked "slow"

# Stop after first failure
pytest -x

# Show local variables on failure
pytest -l

# Show print output
pytest -s   (or --capture=no)
```

---

## Writing Tests

### Basic Test Structure

```python
# test_math.py
# All test files should start with test_ or end with _test.py
# All test functions should start with test_

def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -1) == -2

def test_add_floats():
    result = add(0.1, 0.2)
    assert abs(result - 0.3) < 1e-9  # floating point!

class TestCalculator:
    """Group related tests in a class."""

    def test_add(self):
        assert add(1, 2) == 3

    def test_with_zero(self):
        assert add(0, 5) == 5
```

### Testing Exceptions

```python
import pytest

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)

def test_divide_by_zero_message():
    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        divide(10, 0)

def test_exception_details():
    with pytest.raises(ZeroDivisionError) as exc_info:
        divide(10, 0)
    assert "zero" in str(exc_info.value).lower()
    assert exc_info.type is ZeroDivisionError
```

---

## Assertions

pytest rewrites `assert` statements to give detailed failure messages.

```python
def test_assertions():
    # Equality
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"

    # Truthiness
    assert [1, 2, 3]      # non-empty list is truthy
    assert not []          # empty list is falsy

    # Comparison
    assert 5 > 3
    assert 3 in [1, 2, 3]
    assert "py" in "python"

    # Type checks
    assert isinstance(42, int)
    assert isinstance("hi", str)

    # Approximate equality (for floats)
    import math
    assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert 0.1 + 0.2 == pytest.approx(0.3, rel=1e-6)
    assert [0.1, 0.2] == pytest.approx([0.1, 0.2])

    # Sequence assertions
    result = [1, 2, 3, 4, 5]
    assert len(result) == 5
    assert result[0] == 1
    assert result[-1] == 5
    assert 3 in result

    # Dict assertions
    d = {"a": 1, "b": 2}
    assert "a" in d
    assert d["a"] == 1
    assert d == {"a": 1, "b": 2}

# Custom assertion message (shown on failure)
def test_with_message():
    x = compute_something()
    assert x > 0, f"Expected positive but got {x}"
```

---

## Fixtures

Fixtures provide setup/teardown and shared state for tests.

### Basic Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Returns sample data for tests."""
    return {"name": "Alice", "age": 30, "email": "alice@example.com"}

@pytest.fixture
def user_list():
    return [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]

def test_user_name(sample_data):
    assert sample_data["name"] == "Alice"

def test_user_list(user_list):
    assert len(user_list) == 2
    assert user_list[0]["name"] == "Alice"
```

### Fixtures with Setup and Teardown

```python
import pytest
import sqlite3
import os

@pytest.fixture
def temp_db():
    """Create a temporary database and clean up after."""
    db_path = "/tmp/test.db"
    conn = sqlite3.connect(db_path)

    # Setup
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
    conn.commit()

    yield conn   # code after yield runs AFTER the test

    # Teardown
    conn.close()
    os.unlink(db_path)

def test_query_users(temp_db):
    cursor = temp_db.execute("SELECT * FROM users")
    users = cursor.fetchall()
    assert len(users) == 2

def test_add_user(temp_db):
    temp_db.execute("INSERT INTO users VALUES (3, 'Carol')")
    temp_db.commit()
    cursor = temp_db.execute("SELECT COUNT(*) FROM users")
    assert cursor.fetchone()[0] == 3
```

### Fixture Scope

```python
import pytest

@pytest.fixture(scope="function")   # default — new for each test
def func_fixture():
    return []

@pytest.fixture(scope="class")      # shared within a test class
def class_fixture():
    return {"data": []}

@pytest.fixture(scope="module")     # shared within a module
def module_fixture():
    # e.g., start a database connection
    conn = create_expensive_connection()
    yield conn
    conn.close()

@pytest.fixture(scope="session")    # shared across entire test session
def session_fixture():
    # e.g., start test server once
    server = start_server()
    yield server
    server.stop()

# Use session-scoped fixture for expensive setup
@pytest.fixture(scope="session")
def redis_client():
    import redis
    client = redis.Redis(host="localhost", port=6379, db=1)
    yield client
    client.flushdb()
    client.close()
```

### `conftest.py` — Shared Fixtures

```python
# conftest.py — fixtures available to all tests in directory and subdirectories

import pytest
from myapp import create_app, db as _db

@pytest.fixture(scope="session")
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.app_context():
        _db.create_all()
        yield app
        _db.drop_all()

@pytest.fixture
def db(app):
    with app.app_context():
        _db.session.begin_nested()
        yield _db
        _db.session.rollback()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def alice(db):
    from myapp.models import User
    user = User(name="Alice", email="alice@example.com")
    db.session.add(user)
    db.session.flush()
    return user
```

### Fixture Dependencies

```python
@pytest.fixture
def base_data():
    return {"created_at": "2024-01-01"}

@pytest.fixture
def user_data(base_data):
    """Depends on base_data fixture."""
    return {**base_data, "name": "Alice", "email": "alice@example.com"}

@pytest.fixture
def admin_data(user_data):
    return {**user_data, "is_admin": True}

def test_admin(admin_data):
    assert admin_data["is_admin"] is True
    assert admin_data["name"] == "Alice"
    assert "created_at" in admin_data
```

---

## Parametrize

Run the same test with multiple inputs.

```python
import pytest

def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

@pytest.mark.parametrize("word, expected", [
    ("racecar", True),
    ("hello",   False),
    ("A man a plan a canal Panama", True),
    ("",        True),
    ("a",       True),
])
def test_is_palindrome(word, expected):
    assert is_palindrome(word) == expected

# Multiple parametrize decorators — cartesian product
@pytest.mark.parametrize("a", [1, 2, 3])
@pytest.mark.parametrize("b", [10, 20])
def test_multiply(a, b):
    assert a * b == a * b  # trivially true

# IDs for test naming
@pytest.mark.parametrize("n,expected", [
    pytest.param(0, 1,    id="zero"),
    pytest.param(1, 1,    id="one"),
    pytest.param(5, 120,  id="five"),
    pytest.param(10, 3628800, id="ten"),
])
def test_factorial(n, expected):
    from math import factorial
    assert factorial(n) == expected

# Indirect parametrize (use fixture with params)
@pytest.fixture
def user(request):
    return {"role": request.param, "name": "Alice"}

@pytest.mark.parametrize("user", ["admin", "user", "guest"], indirect=True)
def test_user_role(user):
    assert user["role"] in ["admin", "user", "guest"]
```

---

## Marks and Skipping

```python
import pytest
import sys

# Skip tests
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    assert False

@pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test")
def test_unix_feature():
    import os
    assert os.name == "posix"

# Expected failure
@pytest.mark.xfail(reason="Known bug — see issue #123")
def test_buggy():
    assert 1 == 2   # expected to fail

@pytest.mark.xfail(strict=True)   # must fail — passes unexpectedly → test fails
def test_strict_xfail():
    assert 1 == 2

# Custom marks
@pytest.mark.slow
def test_heavy_computation():
    import time
    time.sleep(5)
    assert True

@pytest.mark.integration
@pytest.mark.database
def test_db_operation():
    ...

# Run specific marks: pytest -m "not slow and not integration"

# Register marks to avoid warnings
# pyproject.toml or pytest.ini:
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow",
#     "integration: integration tests that require external services",
# ]
```

---

## Mocking

```python
from unittest.mock import Mock, MagicMock, patch, call
import pytest

# --- Basic Mock ---
mock = Mock()
mock.method.return_value = 42
print(mock.method())    # 42
print(mock.method.called)  # True
print(mock.method.call_count)  # 1
mock.method.assert_called_once_with()
mock.method.assert_called_once()

# --- Patch ---
# Replace real objects with mocks during test

# Patch a function
from mymodule import get_weather

with patch("mymodule.get_weather") as mock_weather:
    mock_weather.return_value = {"temp": 25, "condition": "sunny"}
    result = get_weather("London")
    assert result["temp"] == 25
    mock_weather.assert_called_once_with("London")

# Patch as decorator
@patch("mymodule.get_weather")
def test_weather(mock_weather):
    mock_weather.return_value = {"temp": 30}
    result = get_weather("NYC")
    assert result["temp"] == 30

# Patch class method
@patch.object(MyClass, "expensive_method")
def test_class(mock_method):
    mock_method.return_value = "mocked"
    obj = MyClass()
    assert obj.expensive_method() == "mocked"

# Patch multiple
@patch("module.ClassA")
@patch("module.ClassB")
def test_multiple(mock_b, mock_a):   # order reversed!
    ...

# --- Side effects ---
mock = Mock()
mock.method.side_effect = [1, 2, 3]       # returns 1, then 2, then 3
mock.method.side_effect = ValueError("error")  # raises exception
mock.method.side_effect = lambda x: x * 2  # function

# --- MagicMock — supports magic methods ---
m = MagicMock()
m.__len__.return_value = 5
assert len(m) == 5

m.__enter__ = Mock(return_value=m)
m.__exit__  = Mock(return_value=False)
with m as ctx:
    pass
```

### `pytest-mock` Plugin

```python
# pip install pytest-mock
def test_with_mocker(mocker):
    # mocker is a pytest-mock fixture
    mock_fetch = mocker.patch("mymodule.fetch_data")
    mock_fetch.return_value = {"result": "ok"}

    result = process_data()   # calls fetch_data internally
    assert result == "ok"
    mock_fetch.assert_called_once()

def test_spy(mocker):
    """Spy on a method without replacing it."""
    spy = mocker.spy(MyClass, "some_method")
    obj = MyClass()
    obj.some_method(42)
    spy.assert_called_once_with(42)
```

---

## Coverage

```bash
# pip install pytest-cov

# Run with coverage
pytest --cov=mypackage tests/
pytest --cov=mypackage --cov-report=html tests/    # HTML report
pytest --cov=mypackage --cov-report=term-missing   # show missing lines
pytest --cov=mypackage --cov-fail-under=80         # fail if coverage < 80%

# .coveragerc
[run]
source = mypackage
omit = 
    mypackage/tests/*
    mypackage/migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if TYPE_CHECKING:
    raise NotImplementedError
```

---

## Plugins

```bash
# Common pytest plugins
pip install pytest-asyncio    # async test support
pip install pytest-mock       # better mocking
pip install pytest-cov        # coverage
pip install pytest-xdist      # parallel test execution
pip install pytest-benchmark  # performance benchmarking
pip install pytest-randomly   # randomize test order
pip install factory-boy       # test data factories
pip install faker             # generate fake data
```

### Async Tests

```python
# pip install pytest-asyncio
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == "expected"

@pytest.mark.asyncio
async def test_async_with_fixture(async_client):
    response = await async_client.get("/api/users")
    assert response.status_code == 200

# Configure in pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"   # all async tests run with asyncio automatically
```

### Parallel Tests

```bash
# pip install pytest-xdist
pytest -n 4          # 4 workers
pytest -n auto       # one per CPU core
pytest --dist=load   # distribute by load
```

### Factory Boy

```python
# pip install factory-boy
import factory
from myapp.models import User

class UserFactory(factory.Factory):
    class Meta:
        model = User

    name     = factory.Faker("name")
    email    = factory.Faker("email")
    age      = factory.Faker("random_int", min=18, max=80)
    is_active = True

# SQLAlchemy factory
class UserSQLFactory(factory.alchemy.SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = db.session

    name  = factory.Sequence(lambda n: f"User {n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.name.lower().replace(' ', '_')}@example.com")

def test_with_factory():
    user = UserFactory()
    assert "@" in user.email

def test_multiple_users():
    users = UserFactory.create_batch(10)
    assert len(users) == 10
```

---

## Configuration

### `pyproject.toml`

```toml
[tool.pytest.ini_options]
testpaths   = ["tests"]
python_files  = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

addopts = [
    "--strict-markers",
    "--strict-config",
    "-v",
]

markers = [
    "slow: marks tests as slow (deselect with '-m not slow')",
    "integration: integration tests",
    "unit: unit tests",
]

asyncio_mode = "auto"

[tool.coverage.run]
source   = ["myapp"]
omit     = ["tests/*", "migrations/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## Best Practices

```python
# 1. Test one thing per test
def test_user_creation_sets_default_active():
    user = User(name="Alice", email="alice@example.com")
    assert user.is_active is True

def test_user_creation_does_not_set_admin():
    user = User(name="Alice", email="alice@example.com")
    assert user.is_admin is False

# 2. Descriptive test names
def test_calculate_shipping_returns_zero_for_premium_members():
    ...

# 3. Arrange-Act-Assert (AAA) pattern
def test_add_item_to_cart():
    # Arrange
    cart = ShoppingCart()
    item = Item(name="Widget", price=9.99)

    # Act
    cart.add_item(item)

    # Assert
    assert len(cart.items) == 1
    assert cart.total == 9.99

# 4. Avoid test interdependence — use fixtures for shared state
# 5. Use parametrize instead of loops in tests
# 6. Mock external services (HTTP, DB, filesystem)
# 7. Keep tests fast — use mocks for slow operations
# 8. Test edge cases: empty, None, boundary values
# 9. Test negative cases — what should fail
# 10. Use conftest.py for shared fixtures

# Property-based testing with Hypothesis
# pip install hypothesis
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_add_commutative(a, b):
    assert a + b == b + a

@given(st.text())
def test_string_reverse_twice(s):
    assert s[::-1][::-1] == s
```
