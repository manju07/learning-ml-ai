# Python Object-Oriented Programming

## Table of Contents
- [Classes and Objects](#classes-and-objects)
- [Instance vs Class vs Static Methods](#instance-vs-class-vs-static-methods)
- [Properties](#properties)
- [Inheritance](#inheritance)
- [Multiple Inheritance and MRO](#multiple-inheritance-and-mro)
- [Dunder Methods](#dunder-methods)
- [Abstract Classes](#abstract-classes)
- [Dataclasses](#dataclasses)
- [Slots](#slots)
- [Metaclasses](#metaclasses)
- [Protocols (Structural Subtyping)](#protocols)

---

## Classes and Objects

```python
class Dog:
    # Class variable — shared by all instances
    species = "Canis lupus familiaris"
    count = 0

    def __init__(self, name: str, breed: str, age: int):
        # Instance variables — unique to each instance
        self.name  = name
        self.breed = breed
        self.age   = age
        Dog.count += 1

    def __repr__(self):
        return f"Dog(name={self.name!r}, breed={self.breed!r}, age={self.age})"

    def __str__(self):
        return f"{self.name} ({self.breed})"

    def bark(self):
        return f"{self.name} says: Woof!"

    def birthday(self):
        self.age += 1
        return self.age


# Creating instances
rex  = Dog("Rex", "German Shepherd", 3)
luna = Dog("Luna", "Labrador", 2)

print(rex)               # Rex (German Shepherd)
print(repr(rex))         # Dog(name='Rex', breed='German Shepherd', age=3)
print(rex.bark())        # Rex says: Woof!
print(Dog.count)         # 2

# Accessing class variable via instance or class
print(rex.species)       # Canis lupus familiaris
print(Dog.species)       # Canis lupus familiaris

# Dynamic attribute manipulation
rex.color = "brown"                # add attribute
print(hasattr(rex, "color"))       # True
print(getattr(rex, "name"))        # Rex
setattr(rex, "age", 4)             # same as rex.age = 4
delattr(rex, "color")              # same as del rex.color
print(vars(rex))                   # {'name': 'Rex', 'breed': 'GS', 'age': 4}
```

---

## Instance vs Class vs Static Methods

```python
class MathUtils:
    pi = 3.14159

    def __init__(self, value):
        self.value = value

    # Instance method — receives self
    def double(self):
        return self.value * 2

    # Class method — receives cls (the class itself)
    @classmethod
    def from_string(cls, s: str):
        """Alternative constructor."""
        return cls(float(s))

    @classmethod
    def circle_area(cls, radius):
        return cls.pi * radius ** 2

    # Static method — no self or cls; just a function in the class namespace
    @staticmethod
    def add(a, b):
        return a + b


m = MathUtils(10)
print(m.double())                    # 20  — instance method
print(MathUtils.from_string("3.5"))  # calls __init__(3.5)
print(MathUtils.circle_area(5))      # 78.53...
print(MathUtils.add(2, 3))           # 5
print(m.add(2, 3))                   # 5 — static can be called on instance too
```

---

## Properties

Properties let you add computed attributes and validation with getter/setter/deleter.

```python
class Temperature:
    def __init__(self, celsius: float = 0):
        self._celsius = celsius    # _name = "protected" by convention

    @property
    def celsius(self):
        """Get temperature in Celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError(f"Temperature below absolute zero: {value}")
        self._celsius = value

    @celsius.deleter
    def celsius(self):
        del self._celsius

    @property
    def fahrenheit(self):
        """Computed property — no setter needed."""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9  # delegates to celsius setter

    def __repr__(self):
        return f"Temperature({self._celsius}°C / {self.fahrenheit}°F)"


t = Temperature(25)
print(t.celsius)      # 25
print(t.fahrenheit)   # 77.0

t.fahrenheit = 32     # uses setter
print(t.celsius)      # 0.0

t.celsius = -300      # ValueError!
```

---

## Inheritance

```python
class Animal:
    def __init__(self, name: str, sound: str):
        self.name  = name
        self.sound = sound

    def speak(self):
        return f"{self.name} says {self.sound}"

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"


class Dog(Animal):
    def __init__(self, name: str, breed: str):
        super().__init__(name, "Woof")  # call parent __init__
        self.breed = breed

    def fetch(self, item):
        return f"{self.name} fetches the {item}!"

    # Override parent method
    def speak(self):
        base = super().speak()    # call parent version
        return f"{base}! (tail wagging)"


class Cat(Animal):
    def __init__(self, name: str, indoor: bool = True):
        super().__init__(name, "Meow")
        self.indoor = indoor

    def speak(self):
        return f"{self.name} says {self.sound} (aloofly)"


rex  = Dog("Rex", "German Shepherd")
luna = Cat("Luna")

print(rex.speak())    # Rex says Woof! (tail wagging)
print(luna.speak())   # Luna says Meow (aloofly)
print(rex.fetch("ball"))  # Rex fetches the ball!

# isinstance checks
print(isinstance(rex, Dog))     # True
print(isinstance(rex, Animal))  # True — checks full hierarchy
print(issubclass(Dog, Animal))  # True
print(type(rex) is Dog)         # True
print(type(rex) is Animal)      # False — exact type check
```

---

## Multiple Inheritance and MRO

Python uses the **C3 linearization** algorithm (MRO — Method Resolution Order).

```python
class A:
    def method(self):
        print("A.method")

class B(A):
    def method(self):
        print("B.method")
        super().method()

class C(A):
    def method(self):
        print("C.method")
        super().method()

class D(B, C):
    def method(self):
        print("D.method")
        super().method()


d = D()
d.method()
# D.method
# B.method
# C.method
# A.method

# Check MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
```

### Mixin Pattern

```python
class JSONMixin:
    """Add JSON serialization to any class."""
    def to_json(self):
        import json
        return json.dumps(vars(self))

class LogMixin:
    """Add logging to any class."""
    def log(self, message):
        import logging
        logging.getLogger(type(self).__name__).info(message)

class TimestampMixin:
    """Add created_at timestamp."""
    from datetime import datetime

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            from datetime import datetime
            self.created_at = datetime.now()
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init


class User(JSONMixin, LogMixin):
    def __init__(self, name, email):
        self.name  = name
        self.email = email


user = User("Alice", "alice@example.com")
print(user.to_json())   # {"name": "Alice", "email": "alice@example.com"}
```

---

## Dunder Methods

"Magic" or "dunder" (double underscore) methods customize Python's built-in behaviors.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # String representation
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    # Arithmetic
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)   # handles scalar * vector

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    # Comparison
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return abs(self) < abs(other)

    # Length / absolute value
    def __abs__(self):
        return (self.x**2 + self.y**2) ** 0.5

    def __len__(self):
        return 2   # number of components

    # Bool
    def __bool__(self):
        return bool(self.x or self.y)

    # Hashing (required when __eq__ is defined, for dict keys / sets)
    def __hash__(self):
        return hash((self.x, self.y))

    # Iteration
    def __iter__(self):
        yield self.x
        yield self.y

    # Indexing
    def __getitem__(self, index):
        return (self.x, self.y)[index]


v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # (4, 6)
print(v1 - v2)      # (2, 2)
print(v1 * 3)       # (9, 12)
print(3 * v1)       # (9, 12)
print(abs(v1))      # 5.0
print(v1 == Vector(3, 4))  # True
print(list(v1))     # [3, 4]
print(v1[0])        # 3

# In a set / as dict key (hashable)
s = {v1, v2}
d = {v1: "first"}
```

### Context Manager Protocol

```python
class ManagedResource:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"Acquiring {self.name}")
        return self   # what gets bound to 'as' variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Releasing {self.name}")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # True would suppress the exception

with ManagedResource("database") as resource:
    print(f"Using {resource.name}")
    # raise RuntimeError("oops")  # __exit__ still called!
```

### Callable Objects

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, value):
        return value * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))    # 10
print(triple(5))    # 15
print(callable(double))  # True
```

---

## Abstract Classes

Abstract classes define **interfaces** — they cannot be instantiated directly.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Return the area of the shape."""
        ...

    @abstractmethod
    def perimeter(self) -> float:
        ...

    def describe(self):
        return f"{type(self).__name__}: area={self.area():.2f}, perimeter={self.perimeter():.2f}"


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        import math
        return math.pi * self.radius ** 2

    def perimeter(self):
        import math
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width, height):
        self.width  = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)


# Shape()          # TypeError — cannot instantiate abstract class
c = Circle(5)
r = Rectangle(4, 6)

print(c.describe())   # Circle: area=78.54, perimeter=31.42
print(r.describe())   # Rectangle: area=24.00, perimeter=20.00

shapes: list[Shape] = [c, r]
total_area = sum(s.area() for s in shapes)
```

---

## Dataclasses

`@dataclass` auto-generates `__init__`, `__repr__`, `__eq__`, and more.

```python
from dataclasses import dataclass, field, KW_ONLY
from typing import ClassVar

@dataclass
class Point:
    x: float
    y: float

p1 = Point(1.0, 2.0)
p2 = Point(1.0, 2.0)
print(p1)           # Point(x=1.0, y=2.0)
print(p1 == p2)     # True — __eq__ generated


@dataclass
class Employee:
    name: str
    department: str
    salary: float = 50_000.0
    skills: list[str] = field(default_factory=list)   # mutable default!

    # Class variables excluded from __init__
    company: ClassVar[str] = "Acme Corp"

    def give_raise(self, amount: float):
        self.salary += amount


e = Employee("Alice", "Engineering")
e.skills.append("Python")
print(e)           # Employee(name='Alice', department='Engineering', ...)


@dataclass(order=True, frozen=True)  # frozen → immutable + hashable
class Version:
    major: int
    minor: int
    patch: int = 0

v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)
print(v1 < v2)   # True — order=True generates __lt__, __le__, etc.
# v1.major = 2   # FrozenInstanceError!


# Post-init processing
@dataclass
class Circle:
    radius: float
    area: float = field(init=False)   # not in __init__

    def __post_init__(self):
        import math
        self.area = math.pi * self.radius ** 2
```

---

## Slots

`__slots__` restricts attributes to a fixed set, saving memory and speeding up attribute access.

```python
class Point:
    __slots__ = ("x", "y")   # only these attributes allowed

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p.x, p.y)   # 3 4
# p.z = 5         # AttributeError — z not in __slots__
# p.__dict__      # AttributeError — no __dict__ with __slots__

# Memory comparison
import sys
class WithDict:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WithSlots:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y

d = WithDict(1, 2)
s = WithSlots(1, 2)
print(sys.getsizeof(d.__dict__))  # ~232 bytes
# Slots objects are significantly smaller
```

---

## Metaclasses

Metaclasses control class creation. A metaclass is the "class of a class."

```python
# type is the default metaclass
print(type(int))     # <class 'type'>
print(type(list))    # <class 'type'>

class MyClass:
    pass

print(type(MyClass))  # <class 'type'>

# Creating a class dynamically with type(name, bases, namespace)
Animal = type("Animal", (), {"sound": "generic", "speak": lambda self: self.sound})
a = Animal()
print(a.speak())  # generic


# Custom metaclass
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=SingletonMeta):
    def __init__(self, url):
        self.url = url


db1 = Database("postgresql://localhost/mydb")
db2 = Database("different-url")  # ignored — same instance
print(db1 is db2)    # True
print(db1.url)       # postgresql://localhost/mydb


# __init_subclass__ — simpler alternative to metaclasses
class Plugin:
    _registry = {}

    def __init_subclass__(cls, command: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Plugin._registry[command] = cls

class HelloPlugin(Plugin, command="hello"):
    def run(self):
        print("Hello!")

class QuitPlugin(Plugin, command="quit"):
    def run(self):
        print("Goodbye!")

print(Plugin._registry)  # {'hello': HelloPlugin, 'quit': QuitPlugin}
Plugin._registry["hello"]().run()  # Hello!
```

---

## Protocols

Protocols (PEP 544) enable **structural subtyping** — duck typing with type safety.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...
    def resize(self, factor: float) -> None: ...


class Circle:
    def draw(self) -> None:
        print("Drawing circle")

    def resize(self, factor: float) -> None:
        self.radius *= factor


class Square:
    def draw(self) -> None:
        print("Drawing square")

    def resize(self, factor: float) -> None:
        self.side *= factor


def render_all(shapes: list[Drawable]) -> None:
    for shape in shapes:
        shape.draw()

# No inheritance needed — structural compatibility!
render_all([Circle(), Square()])

# Runtime check
print(isinstance(Circle(), Drawable))  # True
```
