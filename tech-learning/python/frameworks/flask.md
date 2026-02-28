# Flask — Lightweight Python Web Framework

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Routing](#routing)
- [Request and Response](#request-and-response)
- [Templates (Jinja2)](#templates-jinja2)
- [Blueprints](#blueprints)
- [Flask-SQLAlchemy](#flask-sqlalchemy)
- [Forms with Flask-WTF](#forms-with-flask-wtf)
- [Authentication with Flask-Login](#authentication-with-flask-login)
- [REST APIs with Flask](#rest-apis-with-flask)
- [Error Handling](#error-handling)
- [Testing](#testing)
- [Configuration](#configuration)

---

## Introduction

Flask is a lightweight WSGI web framework. It's designed to be simple and extensible:
- Minimal core — only routing and templates
- Extensible via extensions (Flask-SQLAlchemy, Flask-Login, etc.)
- Perfect for small apps, microservices, and REST APIs

```bash
pip install flask
```

---

## Getting Started

```python
# app.py
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

```bash
# Run
flask run                    # uses FLASK_APP env var
flask --app app run --debug
python app.py               # if using if __name__ == "__main__"
```

---

## Routing

```python
from flask import Flask

app = Flask(__name__)

# Simple routes
@app.route("/")
def home():
    return "Home page"

# Multiple methods
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return "Logging in..."
    return "Show login form"

# URL parameters with type converters
@app.route("/users/<int:user_id>")
def get_user(user_id):   # int
    return f"User {user_id}"

@app.route("/posts/<string:slug>")   # str (default)
def get_post(slug):
    return f"Post: {slug}"

@app.route("/files/<path:filepath>")  # path (allows slashes)
def get_file(filepath):
    return f"File: {filepath}"

@app.route("/items/<uuid:item_uuid>")  # UUID
def get_item(item_uuid):
    return str(item_uuid)

# URL building
from flask import url_for

with app.test_request_context():
    print(url_for("home"))           # /
    print(url_for("get_user", user_id=1))   # /users/1
    print(url_for("static", filename="style.css"))  # /static/style.css

# Redirect
from flask import redirect

@app.route("/old")
def old():
    return redirect(url_for("home"))

@app.route("/external")
def external():
    return redirect("https://example.com")
```

---

## Request and Response

### Request Object

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/data", methods=["POST"])
def handle_data():
    # Query string: /data?name=Alice&age=30
    name = request.args.get("name")
    age  = request.args.get("age", type=int)
    all_args = request.args.to_dict()

    # Form data (application/x-www-form-urlencoded or multipart)
    username = request.form.get("username")
    password = request.form.get("password")

    # JSON body
    data = request.json       # raises if not JSON
    data = request.get_json() # returns None if not JSON
    data = request.get_json(force=True, silent=True)  # always try to parse

    # Raw body
    raw = request.data        # bytes

    # Headers
    token     = request.headers.get("Authorization")
    content   = request.headers["Content-Type"]

    # Cookies
    session_id = request.cookies.get("session_id")

    # Files
    file = request.files.get("photo")
    if file:
        file.save(f"uploads/{file.filename}")

    # Request info
    print(request.method)     # GET, POST, etc.
    print(request.url)        # full URL
    print(request.base_url)   # URL without query string
    print(request.path)       # /data
    print(request.host)       # localhost:5000
    print(request.remote_addr) # client IP

    return jsonify({"received": data})
```

### Response Object

```python
from flask import Flask, make_response, jsonify, Response

app = Flask(__name__)

# Return string (status 200)
@app.route("/")
def text():
    return "Hello"

# Return with status code
@app.route("/created")
def created():
    return "Created", 201

# Return with headers
@app.route("/with-headers")
def with_headers():
    return "OK", 200, {"X-Custom-Header": "value", "Cache-Control": "no-cache"}

# JSON response
@app.route("/json")
def json_response():
    data = {"key": "value", "number": 42}
    return jsonify(data)   # sets Content-Type: application/json

# Custom response object
@app.route("/custom")
def custom():
    resp = make_response("Custom response", 200)
    resp.headers["X-My-Header"] = "custom"
    resp.set_cookie("session", "abc123", httponly=True, secure=True, samesite="Lax")
    return resp

# Streaming response
@app.route("/stream")
def stream():
    def generate():
        for i in range(100):
            yield f"data: line {i}\n\n"
    return Response(generate(), mimetype="text/event-stream")

# File download
from flask import send_file, send_from_directory

@app.route("/download")
def download():
    return send_file("path/to/file.pdf", as_attachment=True, download_name="report.pdf")

@app.route("/static-file/<filename>")
def static_file(filename):
    return send_from_directory("uploads", filename)
```

---

## Templates (Jinja2)

Flask uses Jinja2 for HTML templating.

```
project/
├── app.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── user.html
└── static/
    ├── style.css
    └── app.js
```

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My App{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <nav>
        <a href="{{ url_for('index') }}">Home</a>
    </nav>
    {% block content %}{% endblock %}
</body>
</html>
```

```html
<!-- templates/user.html -->
{% extends "base.html" %}

{% block title %}{{ user.name }} - My App{% endblock %}

{% block content %}
<h1>{{ user.name }}</h1>
<p>Email: {{ user.email }}</p>
<p>Age: {{ user.age | default("Unknown") }}</p>

{% if user.is_admin %}
    <span class="badge">Admin</span>
{% endif %}

<h2>Posts</h2>
{% if posts %}
<ul>
    {% for post in posts %}
    <li>
        <a href="{{ url_for('get_post', post_id=post.id) }}">{{ post.title }}</a>
        <small>{{ post.date | format_date }}</small>
    </li>
    {% endfor %}
</ul>
{% else %}
    <p>No posts yet.</p>
{% endif %}

<!-- Filters -->
{{ user.bio | truncate(100) }}
{{ user.name | upper }}
{{ price | round(2) }}
{{ user.created_at | strftime('%Y-%m-%d') }}
{% endblock %}
```

```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/users/<int:user_id>")
def get_user(user_id):
    user  = {"id": 1, "name": "Alice", "email": "alice@example.com", "is_admin": True}
    posts = [{"id": 1, "title": "Hello"}, {"id": 2, "title": "World"}]
    return render_template("user.html", user=user, posts=posts)

# Custom Jinja2 filter
@app.template_filter("format_date")
def format_date(value):
    from datetime import datetime
    return value.strftime("%B %d, %Y")

# Custom global function
@app.template_global("current_year")
def current_year():
    from datetime import datetime
    return datetime.now().year
```

---

## Blueprints

Blueprints organize large Flask apps into modules.

```python
# blueprints/auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash

auth = Blueprint("auth", __name__, url_prefix="/auth")

@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # verify credentials
        flash("Logged in successfully!", "success")
        return redirect(url_for("main.dashboard"))
    return render_template("auth/login.html")

@auth.route("/logout")
def logout():
    # clear session
    return redirect(url_for("auth.login"))

# blueprints/main.py
from flask import Blueprint

main = Blueprint("main", __name__)

@main.route("/")
def index():
    return render_template("main/index.html")

@main.route("/dashboard")
def dashboard():
    return render_template("main/dashboard.html")

# app.py
from flask import Flask
from blueprints.auth import auth
from blueprints.main import main

def create_app(config=None):
    app = Flask(__name__)

    if config:
        app.config.from_object(config)

    app.register_blueprint(auth)
    app.register_blueprint(main)

    return app

app = create_app()
```

---

## Flask-SQLAlchemy

```bash
pip install flask-sqlalchemy flask-migrate
```

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db  = SQLAlchemy(app)
migrate = Migrate(app, db)

# Models
class User(db.Model):
    __tablename__ = "users"

    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(200), unique=True, nullable=False)
    is_active  = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    posts      = db.relationship("Post", back_populates="author", lazy="dynamic")

    def to_dict(self):
        return {"id": self.id, "name": self.name, "email": self.email}

    def __repr__(self):
        return f"<User {self.email}>"


class Post(db.Model):
    __tablename__ = "posts"

    id         = db.Column(db.Integer, primary_key=True)
    title      = db.Column(db.String(200), nullable=False)
    content    = db.Column(db.Text)
    author_id  = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    author     = db.relationship("User", back_populates="posts")


# Create tables
with app.app_context():
    db.create_all()

# CRUD operations
@app.route("/users", methods=["POST"])
def create_user():
    data = request.json
    user = User(name=data["name"], email=data["email"])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

@app.route("/users")
def list_users():
    users = User.query.filter_by(is_active=True).order_by(User.name).all()
    return jsonify([u.to_dict() for u in users])

@app.route("/users/<int:user_id>")
def get_user(user_id):
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@app.route("/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.json
    user.name  = data.get("name", user.name)
    user.email = data.get("email", user.email)
    db.session.commit()
    return jsonify(user.to_dict())

@app.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return "", 204
```

---

## Error Handling

```python
from flask import Flask, jsonify

app = Flask(__name__)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404

@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400

@app.errorhandler(500)
def server_error(e):
    return jsonify(error="Internal server error"), 500

# Custom exception
class APIError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code

@app.errorhandler(APIError)
def handle_api_error(e):
    return jsonify(error=e.message), e.status_code

@app.route("/test-error")
def test_error():
    raise APIError("Something went wrong", 422)
```

---

## Testing

```python
# pip install pytest
import pytest
from app import create_app, db

@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def test_home(client):
    r = client.get("/")
    assert r.status_code == 200

def test_create_user(client):
    r = client.post("/users",
        json={"name": "Alice", "email": "alice@example.com"},
        content_type="application/json",
    )
    assert r.status_code == 201
    data = r.get_json()
    assert data["name"] == "Alice"

def test_user_not_found(client):
    r = client.get("/users/9999")
    assert r.status_code == 404

# Test with auth
def test_login(client):
    r = client.post("/auth/login",
        data={"username": "alice", "password": "secret"},
        follow_redirects=True,
    )
    assert r.status_code == 200
```

---

## Configuration

```python
# config.py
import os

class Config:
    SECRET_KEY        = os.environ.get("SECRET_KEY", "dev-secret-key")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    DEBUG             = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///dev.db"

class TestingConfig(Config):
    TESTING           = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED  = False

class ProductionConfig(Config):
    DEBUG             = False
    SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
    # Other production settings


config = {
    "development": DevelopmentConfig,
    "testing":     TestingConfig,
    "production":  ProductionConfig,
    "default":     DevelopmentConfig,
}

# Load in app factory
def create_app(config_name="default"):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    return app
```
