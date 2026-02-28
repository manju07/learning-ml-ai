# Requests — HTTP for Humans

## Table of Contents
- [Introduction](#introduction)
- [Making Requests](#making-requests)
- [Request Parameters](#request-parameters)
- [Response Object](#response-object)
- [Authentication](#authentication)
- [Sessions](#sessions)
- [File Uploads](#file-uploads)
- [Timeouts and Retries](#timeouts-and-retries)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)
- [httpx — Modern Alternative](#httpx)

---

## Introduction

```bash
pip install requests
```

```python
import requests

response = requests.get("https://api.github.com")
print(response.status_code)  # 200
print(response.json())
```

---

## Making Requests

### HTTP Methods

```python
import requests

BASE = "https://jsonplaceholder.typicode.com"

# GET — retrieve resource
r = requests.get(f"{BASE}/posts/1")

# POST — create resource
r = requests.post(f"{BASE}/posts",
    json={"title": "foo", "body": "bar", "userId": 1})

# PUT — replace resource
r = requests.put(f"{BASE}/posts/1",
    json={"id": 1, "title": "updated", "userId": 1})

# PATCH — partial update
r = requests.patch(f"{BASE}/posts/1",
    json={"title": "patched title"})

# DELETE — remove resource
r = requests.delete(f"{BASE}/posts/1")

# HEAD — headers only
r = requests.head(f"{BASE}/posts/1")

# OPTIONS — available methods
r = requests.options(f"{BASE}/posts/1")
```

---

## Request Parameters

### Query Parameters

```python
# URL: https://api.example.com/search?q=python&page=1&per_page=20

params = {
    "q":        "python",
    "page":     1,
    "per_page": 20,
    "sort":     "stars",
    "order":    "desc",
}
r = requests.get("https://api.github.com/search/repositories", params=params)
print(r.url)   # shows full URL with query string

# List values — becomes repeated params
params = {"tag": ["python", "web", "api"]}
# URL: ...?tag=python&tag=web&tag=api
```

### Request Headers

```python
headers = {
    "Accept":       "application/json",
    "Content-Type": "application/json",
    "User-Agent":   "MyApp/1.0",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "X-API-Key":    "your-api-key",
}

r = requests.get("https://api.example.com/data", headers=headers)
```

### Request Body

```python
# JSON body (sets Content-Type: application/json automatically)
r = requests.post(
    "https://api.example.com/users",
    json={"name": "Alice", "email": "alice@example.com"}
)

# Form data (application/x-www-form-urlencoded)
r = requests.post(
    "https://httpbin.org/post",
    data={"username": "alice", "password": "secret"}
)

# Raw string body
r = requests.post(
    "https://api.example.com",
    data='{"key": "value"}',
    headers={"Content-Type": "application/json"}
)

# Binary body
with open("image.jpg", "rb") as f:
    r = requests.post("https://api.example.com/upload", data=f)
```

---

## Response Object

```python
r = requests.get("https://jsonplaceholder.typicode.com/posts/1")

# Status
print(r.status_code)          # 200
print(r.ok)                   # True if status < 400
print(r.reason)               # "OK"

# Headers
print(r.headers)              # CaseInsensitiveDict
print(r.headers["Content-Type"])  # application/json; charset=utf-8

# Content
print(r.text)                 # string (auto-detected encoding)
print(r.content)              # raw bytes
print(r.json())               # parse JSON → dict/list (raises if not JSON)
print(r.encoding)             # detected encoding ("utf-8")

# URL and History
print(r.url)                  # final URL (after redirects)
print(r.history)              # list of redirects (Response objects)

# Cookies
print(r.cookies)              # RequestsCookieJar

# Elapsed time
print(r.elapsed.total_seconds())  # request duration

# Raise for non-2xx status
r.raise_for_status()          # HTTPError if 4xx or 5xx

# Check before parsing
if r.status_code == 200:
    data = r.json()
```

### Streaming Large Responses

```python
import requests

url = "https://example.com/large-file.zip"

# Stream — don't download all at once
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open("large-file.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Stream lines (text)
with requests.get("https://api.example.com/events", stream=True) as r:
    for line in r.iter_lines():
        if line:
            data = line.decode("utf-8")
            print(data)

# Stream with progress bar (pip install tqdm)
from tqdm import tqdm

with requests.get(url, stream=True) as r:
    total = int(r.headers.get("Content-Length", 0))
    with open("file.zip", "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))
```

---

## Authentication

### Basic Auth

```python
from requests.auth import HTTPBasicAuth

r = requests.get(
    "https://api.example.com/data",
    auth=HTTPBasicAuth("username", "password")
)

# Shorthand tuple
r = requests.get("https://api.example.com/data", auth=("username", "password"))
```

### Token / Bearer Auth

```python
token = "your-access-token"
headers = {"Authorization": f"Bearer {token}"}

r = requests.get("https://api.example.com/user", headers=headers)

# API Key in header
headers = {"X-API-Key": "your-api-key"}

# API Key in query param
params = {"api_key": "your-api-key"}
```

### OAuth2 (with requests-oauthlib)

```python
from requests_oauthlib import OAuth2Session  # pip install requests-oauthlib

CLIENT_ID = "your-client-id"
REDIRECT_URI = "https://yourapp.com/callback"

oauth = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI, scope=["read", "write"])
authorization_url, state = oauth.authorization_url("https://provider.com/oauth/authorize")

print(f"Visit: {authorization_url}")
authorization_response = input("Paste the redirect URL: ")

token = oauth.fetch_token(
    "https://provider.com/oauth/token",
    authorization_response=authorization_response,
    client_secret="your-client-secret",
)

r = oauth.get("https://api.provider.com/user")
```

### Custom Auth

```python
from requests.auth import AuthBase

class APIKeyAuth(AuthBase):
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, r):
        r.headers["X-API-Key"] = self.api_key
        return r

r = requests.get("https://api.example.com/data", auth=APIKeyAuth("key123"))
```

---

## Sessions

Sessions persist cookies, headers, and connection pooling across requests.

```python
import requests

# Create session
session = requests.Session()

# Set defaults that apply to all requests
session.headers.update({
    "User-Agent": "MyApp/1.0",
    "Accept": "application/json",
})
session.auth = ("username", "password")

# Session persists cookies automatically
r = session.post("https://httpbin.org/cookies/set/sessionid/abc123")
r = session.get("https://httpbin.org/cookies")  # cookie is sent!
print(r.json())   # {'cookies': {'sessionid': 'abc123'}}

# Use as context manager (auto-closes)
with requests.Session() as session:
    session.headers["Authorization"] = "Bearer token123"

    r1 = session.get("https://api.example.com/users")
    r2 = session.post("https://api.example.com/posts", json={"title": "test"})

# Session connection pooling
# Sessions reuse TCP connections — faster for multiple requests to same host
```

---

## File Uploads

```python
import requests

# Simple file upload (multipart/form-data)
with open("report.pdf", "rb") as f:
    r = requests.post(
        "https://api.example.com/upload",
        files={"file": f}
    )

# With filename and content type
with open("image.jpg", "rb") as f:
    r = requests.post(
        "https://api.example.com/upload",
        files={"file": ("custom_name.jpg", f, "image/jpeg")}
    )

# Multiple files
files = {
    "file1": open("doc1.txt", "rb"),
    "file2": open("doc2.txt", "rb"),
}
r = requests.post("https://api.example.com/upload", files=files)
for f in files.values(): f.close()

# File + form fields
r = requests.post(
    "https://api.example.com/upload",
    files={"file": open("image.jpg", "rb")},
    data={"description": "My photo", "public": "true"},
)
```

---

## Timeouts and Retries

### Timeouts

```python
# Always set timeouts! Without them, requests can hang forever.

# Connect timeout, read timeout
r = requests.get("https://api.example.com", timeout=5)       # both = 5s
r = requests.get("https://api.example.com", timeout=(3, 10)) # connect=3s, read=10s

# None = no timeout (default — dangerous!)
r = requests.get("https://api.example.com", timeout=None)
```

### Retries with urllib3

```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Retry strategy
retry = Retry(
    total=5,                    # total retries
    backoff_factor=0.5,         # wait factor (0.5, 1, 2, 4, 8 seconds)
    status_forcelist=[429, 500, 502, 503, 504],  # retry on these status codes
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)

adapter = HTTPAdapter(max_retries=retry)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

r = session.get("https://api.example.com/data", timeout=10)
```

### Custom Retry Logic

```python
import time
import requests
from requests.exceptions import RequestException

def make_request_with_retry(url, max_retries=3, backoff=1.0):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r
        except RequestException as e:
            if attempt == max_retries:
                raise
            wait = backoff * (2 ** (attempt - 1))   # exponential backoff
            print(f"Attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
```

---

## Error Handling

```python
import requests
from requests.exceptions import (
    ConnectionError,
    Timeout,
    HTTPError,
    RequestException,
    TooManyRedirects,
    SSLError,
)

def fetch_data(url):
    try:
        r = requests.get(url, timeout=(5, 30))
        r.raise_for_status()   # raises HTTPError for 4xx/5xx
        return r.json()

    except Timeout:
        print("Request timed out")
    except SSLError:
        print("SSL certificate verification failed")
    except ConnectionError:
        print("Network connection failed")
    except HTTPError as e:
        print(f"HTTP error: {e.response.status_code}")
        if e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 60))
            print(f"Rate limited. Retry after {retry_after}s")
        elif e.response.status_code == 401:
            print("Unauthorized — check credentials")
        elif e.response.status_code == 404:
            print("Resource not found")
    except TooManyRedirects:
        print("Too many redirects")
    except RequestException as e:
        print(f"Unexpected error: {e}")

# Check status explicitly
r = requests.get(url)
if r.status_code == 200:
    data = r.json()
elif r.status_code == 404:
    print("Not found")
elif r.status_code == 401:
    print("Need to authenticate")
```

---

## Advanced Usage

### SSL / TLS

```python
# Verify SSL certificate (default: True — don't disable in production!)
r = requests.get("https://api.example.com", verify=True)

# Custom CA bundle
r = requests.get("https://api.example.com", verify="/path/to/ca-bundle.crt")

# Disable verification (development only — INSECURE!)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
r = requests.get("https://self-signed.badssl.com/", verify=False)

# Client certificates
r = requests.get("https://api.example.com",
    cert=("/path/to/client.crt", "/path/to/client.key"))
```

### Proxies

```python
proxies = {
    "http":  "http://user:pass@proxy.example.com:8080",
    "https": "http://user:pass@proxy.example.com:8080",
}

r = requests.get("https://api.example.com", proxies=proxies)

# SOCKS proxy (pip install requests[socks])
proxies = {"https": "socks5://user:pass@socks.example.com:1080"}
```

### Hooks

```python
def log_response(r, *args, **kwargs):
    print(f"{r.request.method} {r.url} → {r.status_code} ({r.elapsed.total_seconds():.3f}s)")

r = requests.get("https://api.example.com", hooks={"response": log_response})

# Session-level hook
session = requests.Session()
session.hooks["response"].append(log_response)
```

### Cookies

```python
# Send cookies
r = requests.get("https://httpbin.org/cookies", cookies={"session_id": "abc123"})

# Get cookies from response
r = requests.get("https://httpbin.org/cookies/set/name/value")
print(r.cookies["name"])   # value

# Persist cookies with session
session = requests.Session()
session.get("https://httpbin.org/cookies/set/session/xyz")
session.get("https://httpbin.org/cookies")  # cookie sent automatically
```

---

## httpx — Modern Alternative

`httpx` is a modern HTTP client with both sync and async support.

```bash
pip install httpx
```

```python
import httpx

# Synchronous (same API as requests)
with httpx.Client() as client:
    r = client.get("https://api.example.com/data", params={"q": "python"})
    print(r.json())

# Async — perfect with asyncio
import asyncio

async def fetch_all(urls):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

urls = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
]
results = asyncio.run(fetch_all(urls))
for r in results:
    print(r["title"])

# HTTP/2 support
with httpx.Client(http2=True) as client:
    r = client.get("https://api.example.com")

# Retry with httpx
transport = httpx.HTTPTransport(retries=3)
with httpx.Client(transport=transport) as client:
    r = client.get("https://api.example.com")
```
