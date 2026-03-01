# Model Context Protocol (MCP): Comprehensive Guide

## Table of Contents
1. [What Is MCP?](#1-what-is-mcp)
2. [Architecture: Hosts, Clients, and Servers](#2-architecture-hosts-clients-and-servers)
3. [Transport Layer](#3-transport-layer)
4. [Lifecycle and Protocol Foundation](#4-lifecycle-and-protocol-foundation)
5. [MCP Primitives: Tools](#5-mcp-primitives-tools)
6. [MCP Primitives: Resources](#6-mcp-primitives-resources)
7. [MCP Primitives: Prompts](#7-mcp-primitives-prompts)
8. [MCP Primitives: Sampling](#8-mcp-primitives-sampling)
9. [Building MCP Servers with Python SDK (FastMCP)](#9-building-mcp-servers-with-python-sdk-fastmcp)
10. [Building MCP Clients](#10-building-mcp-clients)
11. [Authentication and Security](#11-authentication-and-security)
12. [MCP in Claude Desktop, Cursor, and Continue](#12-mcp-in-claude-desktop-cursor-and-continue)
13. [Building Real MCP Servers](#13-building-real-mcp-servers)
14. [Testing MCP Servers](#14-testing-mcp-servers)
15. [Deployment and Containerization](#15-deployment-and-containerization)
16. [MCP Server Registry and Discovery](#16-mcp-server-registry-and-discovery)
17. [Full Production-Ready Examples](#17-full-production-ready-examples)
18. [Best Practices](#18-best-practices)
19. [MCP Server Implementation Patterns](#19-mcp-server-implementation-patterns)
20. [Pitfalls and Common Mistakes](#20-pitfalls-and-common-mistakes)

---

## 1. What Is MCP?

### The Problem Before MCP

Every AI application needed custom integrations for every external tool. If you had 5 AI models and 10 external tools, you needed 50 custom connectors — each with its own authentication, error handling, and API-specific quirks. Maintaining this matrix was expensive, error-prone, and didn't compose.

```
Before MCP (N×M problem):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Claude  │────▶│ GitHub  │     │  Slack  │
│         │     │connector│     │connector│
└─────────┘     └─────────┘     └─────────┘
     │           different        different
     │           code             code
┌─────────┐     ┌─────────┐     ┌─────────┐
│  GPT-4  │────▶│ GitHub  │     │  Slack  │
│         │     │connector│     │connector│
└─────────┘     └─────────┘     └─────────┘
= 5 models × 10 tools = 50 integrations

With MCP (N+M problem):
Any Model ←→ MCP Protocol ←→ GitHub MCP Server
                           ←→ Slack MCP Server
                           ←→ Database MCP Server
= 5 model clients + 10 servers = 15 implementations
```

### What MCP Is

The **Model Context Protocol (MCP)** is an open standard created by Anthropic (December 2024) that defines how AI applications communicate with external data sources and tools. It's sometimes called "USB-C for AI" — a universal connector that works regardless of which model or tool you're using.

Key design goals:
- **Server simplicity**: Writing a new MCP server should take minutes, not days
- **Client simplicity**: Any MCP-compatible client can use any server without changes
- **Composability**: Multiple servers can be used together seamlessly
- **Security**: Clients control what servers can access, and can prompt users for confirmation
- **Discoverability**: Servers expose capabilities; clients adapt automatically

### MCP vs. OpenAI Function Calling

| Aspect | OpenAI Function Calling | MCP |
|--------|------------------------|-----|
| Scope | OpenAI API only | Any model, any client |
| Resources | Not supported | First-class primitive |
| Prompts/templates | Not supported | First-class primitive |
| Server lifecycle | None (stateless) | Persistent, stateful |
| Streaming | Limited | Full SSE support |
| Tool discovery | Static definition | Dynamic listing |
| Transport | HTTP only | stdio, HTTP, WebSocket |

### The Three Server-Provided Capabilities

MCP servers expose three types of features:

| Primitive | Who Initiates | AI Model Controls | Description |
|-----------|--------------|-------------------|-------------|
| **Tools** | Model/AI | Yes | Functions the LLM can call to perform actions |
| **Resources** | Application/Host | No (read-only) | Data the application can inject as context |
| **Prompts** | User | No | Reusable prompt templates, like slash commands |

### The Two Client-Provided Capabilities

Clients can also offer features to servers:

| Feature | Direction | Description |
|---------|-----------|-------------|
| **Sampling** | Server → Client → LLM | Server asks the host LLM to run an inference |
| **Roots** | Server → Client | Server asks which filesystem paths are accessible |

---

## 2. Architecture: Hosts, Clients, and Servers

### The Three-Tier Model

```
┌─────────────────────────────────────────────────────┐
│                MCP HOST (AI Application)             │
│          e.g., Claude Desktop, Cursor, VS Code       │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐                 │
│  │  MCP Client  │   │  MCP Client  │                 │
│  │  (1:1 with   │   │  (1:1 with   │                 │
│  │   server 1)  │   │   server 2)  │                 │
│  └──────┬───────┘   └──────┬───────┘                 │
└─────────┼─────────────────┼───────────────────────── ┘
          │ stdio            │ HTTP+SSE
   ┌──────▼───────┐   ┌──────▼───────┐
   │  MCP Server  │   │  MCP Server  │
   │   (local)    │   │   (remote)   │
   │ Filesystem   │   │   Sentry     │
   │ SQLite       │   │   GitHub     │
   └──────────────┘   └──────────────┘
```

### Host Responsibilities

The **host** is the application the user interacts with (Claude Desktop, Cursor, VS Code with Continue extension, etc.):
- Creates and manages multiple MCP clients
- Aggregates context from all clients to give to the LLM
- Enforces security policies (user consent, rate limits, access controls)
- Routes tool calls to the appropriate client/server
- Decides which resources to include in the context window

### Client Responsibilities

Each **client** maintains a persistent, stateful connection to exactly one server:
- Handles connection lifecycle (connect, reconnect, disconnect)
- Sends JSON-RPC requests to the server
- Receives responses and notifications from the server
- Passes tool results back to the host/LLM

### Server Responsibilities

**Servers** expose capabilities to clients:
- Implement the MCP protocol over the chosen transport
- Handle `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get`
- Execute tool functions and return results
- Send notifications when capabilities change
- Optionally request sampling (LLM completions) from the client

---

## 3. Transport Layer

### Overview of Transport Options

MCP supports multiple transports to accommodate both local and remote scenarios:

| Transport | Connection | Best For | Authentication |
|-----------|-----------|----------|----------------|
| **stdio** | subprocess stdin/stdout | Local tools | OS process security |
| **Streamable HTTP** | HTTP POST + SSE | Remote servers | HTTP auth headers |
| **WebSocket** | WebSocket | Low-latency bidirectional | WS handshake |

### stdio Transport (Local)

The server runs as a subprocess of the host; messages flow through stdin/stdout.

```
Host process                Server subprocess
     │                           │
     │──── stdin (JSON-RPC) ────▶│
     │◀─── stdout (JSON-RPC) ────│
     │◀─── stderr (logs only) ───│
```

**How the host spawns the server:**
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["-m", "mcp_server_filesystem"],
      "env": {
        "ALLOWED_DIRS": "/home/user/projects"
      }
    }
  }
}
```

Message framing: each message is a JSON object terminated by a newline (`\n`).

### Streamable HTTP Transport (Remote)

A modern, stateless HTTP transport that supports both request-response and streaming via Server-Sent Events:

```
Client                              Server
  │                                   │
  │──── POST /mcp (JSON-RPC req) ────▶│
  │◀─── 200 OK (JSON or SSE stream) ──│
```

For long-running operations or server-initiated messages, the server responds with `Content-Type: text/event-stream`:

```
data: {"jsonrpc": "2.0", "id": 1, "result": {...}}\n\n
data: {"jsonrpc": "2.0", "method": "notifications/progress", ...}\n\n
```

### Message Format: JSON-RPC 2.0

All MCP communication uses JSON-RPC 2.0:

```json
// Request (client → server)
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {"city": "Tokyo"}
  }
}

// Success Response (server → client)
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "content": [{"type": "text", "text": "Weather in Tokyo: 18°C, Partly Cloudy"}],
    "isError": false
  }
}

// Error Response
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {"detail": "city is required"}
  }
}

// Notification (no id, no response expected)
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

---

## 4. Lifecycle and Protocol Foundation

### Connection Lifecycle

```
Client                                    Server
  │                                         │
  │──── initialize {protocolVersion, caps} ▶│
  │◀─── initialize result {caps, info} ─────│
  │                                         │
  │──── notifications/initialized ─────────▶│
  │                                         │
  │        (normal operations)              │
  │                                         │
  │──── tools/list ─────────────────────────▶│
  │◀─── tools/list result ──────────────────│
  │                                         │
  │──── tools/call {name, args} ───────────▶│
  │◀─── tools/call result ──────────────────│
  │                                         │
  │──── disconnect ─────────────────────────▶│
```

### Capability Negotiation

During initialization, client and server each declare their capabilities. Only declared capabilities will be used:

```json
// Client declares what IT supports
{
  "protocolVersion": "2025-06-18",
  "capabilities": {
    "sampling": {},           // client can handle sampling requests
    "roots": {
      "listChanged": true    // client can notify when roots change
    },
    "elicitation": {}        // client can present questions to user
  },
  "clientInfo": {
    "name": "MyApp",
    "version": "2.0.0"
  }
}

// Server declares what IT supports
{
  "protocolVersion": "2025-06-18",
  "capabilities": {
    "tools": {
      "listChanged": true    // server can notify when tools change
    },
    "resources": {
      "subscribe": true,     // clients can subscribe to resource changes
      "listChanged": true
    },
    "prompts": {
      "listChanged": true
    },
    "logging": {}            // server supports log messages
  },
  "serverInfo": {
    "name": "weather-server",
    "version": "1.0.0"
  }
}
```

### Standard JSON-RPC Error Codes

| Code | Meaning |
|------|---------|
| -32700 | Parse error |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |

---

## 5. MCP Primitives: Tools

### What Are MCP Tools?

Tools are executable functions that the LLM can invoke. They're the most powerful MCP primitive — they enable the AI to perform real-world actions: querying databases, calling APIs, executing code, modifying files.

### Tool Definition Schema

```json
{
  "name": "create_github_issue",
  "title": "Create GitHub Issue",
  "description": "Creates a new issue in a GitHub repository. Use this when the user asks to file a bug, feature request, or task.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "repo": {
        "type": "string",
        "description": "Repository in owner/repo format, e.g. 'openai/openai-python'"
      },
      "title": {
        "type": "string",
        "description": "Clear, concise issue title"
      },
      "body": {
        "type": "string",
        "description": "Detailed issue description in Markdown"
      },
      "labels": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Labels to apply, e.g. ['bug', 'priority:high']"
      }
    },
    "required": ["repo", "title", "body"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "issue_number": {"type": "integer"},
      "url": {"type": "string"}
    }
  }
}
```

### Tool Invocation Flow

```
1. Client: tools/list request
2. Server: returns list of tool definitions
3. LLM: selects tool and generates call arguments
4. Client: tools/call { name: "create_github_issue", arguments: {...} }
5. Server: validates arguments, executes function
6. Server: returns tool result
7. Client: passes result to LLM
8. LLM: processes result, generates final response
```

### Tool Result Content Types

```json
// Text result
{
  "content": [{"type": "text", "text": "Issue #42 created: https://github.com/..."}],
  "isError": false
}

// Image result (e.g., screenshot, chart)
{
  "content": [{
    "type": "image",
    "data": "base64encodedimagedata==",
    "mimeType": "image/png"
  }],
  "isError": false
}

// Multiple content items
{
  "content": [
    {"type": "text", "text": "Query executed. Found 3 results:"},
    {"type": "text", "text": "[{\"id\": 1, \"name\": \"Alice\"}, ...]"},
    {"type": "image", "data": "...", "mimeType": "image/png"}
  ],
  "isError": false
}

// Structured result (when outputSchema is defined)
{
  "content": [{"type": "text", "text": "{\"issue_number\": 42}"}],
  "structuredContent": {"issue_number": 42, "url": "https://github.com/.../42"},
  "isError": false
}

// Error result (tool ran but failed business logic)
{
  "content": [{"type": "text", "text": "Rate limit exceeded. Try again in 60 seconds."}],
  "isError": true
}
```

---

## 6. MCP Primitives: Resources

### What Are MCP Resources?

Resources are **read-only data** that the host application can inject into the LLM's context. Unlike tools, resources are application-driven — the host decides when and which resources to include, not the LLM.

Examples: file contents, database schemas, API documentation, live metrics, configuration files.

### Resource Definition

```json
{
  "uri": "file:///project/src/auth.py",
  "name": "auth.py",
  "title": "Authentication Module",
  "description": "JWT-based authentication for the REST API",
  "mimeType": "text/x-python",
  "annotations": {
    "audience": ["assistant"],
    "priority": 0.9
  }
}
```

### Resource Operations

```
resources/list                 → List all available resources
resources/read {uri}           → Read content of a specific resource
resources/templates/list       → List parameterized URI templates
resources/subscribe {uri}      → Subscribe to changes
notifications/resources/updated → Server notifies of a change
```

### Resource Templates (Parameterized URIs)

Templates use RFC 6570 URI template syntax for dynamic resources:

```json
{
  "uriTemplate": "db://tables/{table_name}",
  "name": "Database Table",
  "description": "Read contents of any database table",
  "mimeType": "application/json"
}
```

Client resolves: `db://tables/users` → reads users table

### Resource Content Types

```json
// Text resource
{
  "contents": [{
    "uri": "file:///config.yaml",
    "mimeType": "text/yaml",
    "text": "database:\n  host: localhost\n  port: 5432"
  }]
}

// Binary resource (images, PDFs)
{
  "contents": [{
    "uri": "file:///logo.png",
    "mimeType": "image/png",
    "blob": "base64encodeddata=="
  }]
}
```

### Resource Subscriptions

```python
# Client subscribes to resource changes
await session.subscribe_resource("file:///config.yaml")

# Server sends notification when file changes
# { "method": "notifications/resources/updated", "params": { "uri": "file:///config.yaml" } }

# Client re-reads the updated resource
content = await session.read_resource("file:///config.yaml")
```

### URI Scheme Conventions

| Scheme | Example | Typical Use |
|--------|---------|-------------|
| `file://` | `file:///home/user/docs/api.md` | Local files |
| `db://` | `db://tables/users/schema` | Database schemas/data |
| `github://` | `github://repos/owner/repo/readme` | GitHub data |
| `slack://` | `slack://channels/general/messages` | Slack messages |
| `https://` | `https://api.example.com/docs` | Web resources |
| `config://` | `config://app/settings` | App configuration |

---

## 7. MCP Primitives: Prompts

### What Are MCP Prompts?

Prompts are **reusable, parameterized message templates** that servers expose. They're user-initiated (like slash commands in Slack or IDE command palettes).

A prompt might be:
- `/summarize` — summarize a document
- `/code-review` — review code for bugs and style
- `/translate to Spanish` — translate text
- `/git-commit` — generate a commit message

### Prompt Definition

```json
{
  "name": "code_review",
  "title": "Code Review",
  "description": "Perform a comprehensive code review including bugs, performance, security, and style",
  "arguments": [
    {
      "name": "code",
      "description": "The code to review (paste the full function or class)",
      "required": true
    },
    {
      "name": "language",
      "description": "Programming language (python, javascript, rust, etc.)",
      "required": false
    },
    {
      "name": "focus",
      "description": "What to focus on: bugs, security, performance, style, or all",
      "required": false
    }
  ]
}
```

### Getting and Using a Prompt

```json
// Request
{
  "method": "prompts/get",
  "params": {
    "name": "code_review",
    "arguments": {
      "code": "def divide(a, b):\n    return a / b",
      "language": "python",
      "focus": "bugs"
    }
  }
}

// Response: structured messages to inject into the LLM conversation
{
  "result": {
    "description": "Code review focused on bugs",
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "Please review this Python code for bugs:\n\n```python\ndef divide(a, b):\n    return a / b\n```\n\nFocus on: bugs"
        }
      }
    ]
  }
}
```

### Multi-Turn Prompt (Few-Shot)

Prompts can include multiple messages to set up few-shot examples:

```json
{
  "result": {
    "messages": [
      {
        "role": "user",
        "content": {"type": "text", "text": "Classify this as spam or not: 'Win a free iPhone!'"}
      },
      {
        "role": "assistant",
        "content": {"type": "text", "text": "SPAM"}
      },
      {
        "role": "user",
        "content": {"type": "text", "text": "Classify: 'Meeting tomorrow at 3pm'"}
      },
      {
        "role": "assistant",
        "content": {"type": "text", "text": "NOT SPAM"}
      },
      {
        "role": "user",
        "content": {"type": "text", "text": "Classify: 'Your package has shipped'"}
      }
    ]
  }
}
```

---

## 8. MCP Primitives: Sampling

### What Is Sampling?

**Sampling** enables servers to request LLM completions via the client — the server sends a `sampling/createMessage` request to the client, which forwards it to the LLM and returns the result. This allows server-side agentic behavior without the server needing its own API key.

### Sampling Flow

```
Server                    Client/Host                   LLM
  │                           │                          │
  │─── sampling/createMessage ▶│                          │
  │    {messages, preferences} │                          │
  │                            │──── LLM completion ─────▶│
  │                            │◀─── LLM response ────────│
  │◀── sampling result ────────│                          │
```

### Sampling Request

```json
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "Analyze this error trace and suggest a fix:\n\nTraceback:\n  File 'main.py', line 42\n    result = int(user_input)\nValueError: invalid literal for int() with base 10: 'abc'"
        }
      }
    ],
    "modelPreferences": {
      "hints": [{"name": "claude-3-5-sonnet"}],
      "intelligencePriority": 0.9,
      "speedPriority": 0.3,
      "costPriority": 0.2
    },
    "systemPrompt": "You are an expert Python debugger. Be concise.",
    "maxTokens": 500,
    "includeContext": "thisServer"
  }
}
```

### Model Preferences

Servers specify preferences abstractly (not exact model names):

```json
{
  "hints": [
    {"name": "claude-3-5-sonnet"},   // preferred model name (advisory)
    {"name": "claude"}               // fallback family
  ],
  "intelligencePriority": 0.8,       // 0-1: how much reasoning quality matters
  "speedPriority": 0.5,              // 0-1: how much latency matters
  "costPriority": 0.3                // 0-1: how much cost matters
}
```

The client makes the final model choice, balancing user preferences, costs, and availability.

---

## 9. Building MCP Servers with Python SDK (FastMCP)

### Installation

```bash
pip install mcp
# With extras
pip install "mcp[cli]"       # includes inspector CLI
pip install "mcp[server]"    # FastAPI-based server extras
```

### Minimal Server

```python
from mcp.server.fastmcp import FastMCP

# Initialize with a name (shown in capability negotiation)
mcp = FastMCP("My First Server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: The person's name.
    """
    return f"Hello, {name}! Welcome to MCP."

if __name__ == "__main__":
    mcp.run()   # stdio transport by default
```

### Defining Tools with @mcp.tool()

```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional, Literal
import httpx

mcp = FastMCP("API Server")

# Simple typed tool
@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b

# Tool with optional params and defaults
@mcp.tool()
def format_date(
    date_string: str,
    format: str = "%Y-%m-%d",
    timezone: str = "UTC"
) -> str:
    """Format a date string.

    Args:
        date_string: Date string to format (ISO format).
        format: Output format (strftime format string).
        timezone: Target timezone name.
    """
    from datetime import datetime
    import pytz
    dt = datetime.fromisoformat(date_string)
    tz = pytz.timezone(timezone)
    localized = dt.astimezone(tz)
    return localized.strftime(format)

# Tool with complex return type
@mcp.tool()
async def fetch_api(url: str, method: Literal["GET", "POST"] = "GET") -> dict:
    """Make an HTTP request and return the JSON response.

    Args:
        url: The URL to request.
        method: HTTP method to use.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(method, url)
        response.raise_for_status()
        return response.json()

# Tool with validation using Pydantic
class SearchParams(BaseModel):
    query: str = Field(..., description="Search query string", min_length=1)
    max_results: int = Field(10, description="Maximum results", ge=1, le=100)
    include_metadata: bool = Field(False, description="Include metadata in results")

@mcp.tool()
def search_documents(params: SearchParams) -> list[dict]:
    """Search through indexed documents.

    Args:
        params: Search parameters including query and options.
    """
    results = document_index.search(
        params.query,
        limit=params.max_results,
        metadata=params.include_metadata
    )
    return [{"id": r.id, "title": r.title, "score": r.score} for r in results]
```

### Defining Resources with @mcp.resource()

```python
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("Resource Server")

# Static resource
@mcp.resource("config://app/settings")
def get_app_settings() -> str:
    """Current application settings."""
    return json.dumps({
        "version": "2.0.0",
        "environment": "production",
        "features": {"dark_mode": True, "beta": False}
    }, indent=2)

# Dynamic resource
@mcp.resource("metrics://system/current")
def get_system_metrics() -> str:
    """Real-time system metrics."""
    import psutil
    return json.dumps({
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent
    })

# Parameterized resource (URI template)
@mcp.resource("db://tables/{table_name}/schema")
def get_table_schema(table_name: str) -> str:
    """Get the schema for a specific database table.

    Args:
        table_name: Name of the database table.
    """
    import sqlite3
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    conn.close()
    schema = [{"cid": c[0], "name": c[1], "type": c[2], "notnull": c[3]} for c in columns]
    return json.dumps(schema, indent=2)

# Resource with MIME type
@mcp.resource("docs://api/openapi.json", mime_type="application/json")
def get_openapi_spec() -> str:
    """OpenAPI specification for the REST API."""
    with open("openapi.json") as f:
        return f.read()
```

### Defining Prompts with @mcp.prompt()

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Prompt Server")

# Simple prompt
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """Generate a prompt for code review.

    Args:
        code: The code to review.
        language: The programming language.
    """
    return f"""Please review this {language} code comprehensively. Analyze:

1. **Bugs and correctness**: Logical errors, edge cases, null pointer risks
2. **Security**: Injection risks, authentication issues, data exposure
3. **Performance**: Inefficient algorithms, N+1 queries, memory leaks
4. **Readability**: Naming, complexity, documentation
5. **Best practices**: Idioms for {language}

Code to review:
```{language}
{code}
```

Provide specific, actionable feedback with line numbers where applicable."""

# Multi-message prompt using list return
from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent

@mcp.prompt()
def debug_session(error_message: str, context: str = "") -> list[PromptMessage]:
    """Start a debugging session for an error.

    Args:
        error_message: The error or exception message.
        context: Optional additional context (code, config, etc.).
    """
    messages = [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=f"I'm encountering this error:\n\n```\n{error_message}\n```\n"
                     + (f"\nContext:\n{context}" if context else "")
            )
        )
    ]
    return messages
```

### Error Handling in Tools

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode

mcp = FastMCP("Safe Server")

@mcp.tool()
def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers.

    Args:
        numerator: The number to divide.
        denominator: The divisor (cannot be zero).
    """
    if denominator == 0:
        # This raises a proper MCP error (protocol-level)
        raise McpError(ErrorCode.INVALID_PARAMS, "Division by zero is not allowed")
    return numerator / denominator

@mcp.tool()
def risky_operation(data: str) -> str:
    """Operation that might fail.

    Args:
        data: Input data to process.
    """
    try:
        result = process_externally(data)
        return result
    except ConnectionError as e:
        # Return error in content (tool-level error, isError=true)
        # FastMCP handles this automatically when exceptions propagate
        raise ValueError(f"External service unavailable: {e}")
```

### Server Context and Lifespan

```python
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from dataclasses import dataclass
import sqlite3

@dataclass
class AppContext:
    db: sqlite3.Connection
    cache: dict

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage server-wide resources."""
    print("Server starting up...")
    db = sqlite3.connect("app.db")
    cache = {}
    try:
        yield AppContext(db=db, cache=cache)
    finally:
        db.close()
        print("Server shutting down...")

mcp = FastMCP("Database Server", lifespan=app_lifespan)

@mcp.tool()
def query_db(ctx: AppContext, sql: str) -> str:
    """Execute a SQL query.

    Args:
        sql: SQL SELECT query.
    """
    cursor = ctx.db.execute(sql)
    rows = cursor.fetchall()
    return str(rows)
```

### Running with Different Transports

```python
# stdio (default, for local use with Claude Desktop, Cursor)
mcp.run()

# Streamable HTTP (for remote deployment)
mcp.run(transport="streamable-http", host="0.0.0.0", port=8080, path="/mcp")

# With SSL
mcp.run(
    transport="streamable-http",
    host="0.0.0.0",
    port=443,
    ssl_certfile="cert.pem",
    ssl_keyfile="key.pem"
)
```

---

## 10. Building MCP Clients

### Async Client with stdio Transport

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Parameters to spawn the server subprocess
    server_params = StdioServerParameters(
        command="python",
        args=["my_server.py"],
        env={"SOME_API_KEY": "sk-..."}
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize connection (capability negotiation)
            init_result = await session.initialize()
            print(f"Connected to: {init_result.serverInfo.name} v{init_result.serverInfo.version}")
            print(f"Server capabilities: {init_result.capabilities}")

            # List available tools
            tools_result = await session.list_tools()
            print("\nAvailable tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    Input: {tool.inputSchema}")

            # Call a tool
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "Tokyo", "unit": "celsius"}
            )
            if result.isError:
                print(f"Tool error: {result.content[0].text}")
            else:
                print(f"Weather: {result.content[0].text}")

            # List resources
            resources_result = await session.list_resources()
            print("\nAvailable resources:")
            for resource in resources_result.resources:
                print(f"  - {resource.uri}: {resource.name}")

            # Read a resource
            content = await session.read_resource("config://app/settings")
            for item in content.contents:
                print(f"Config: {item.text}")

            # List prompts
            prompts_result = await session.list_prompts()
            print("\nAvailable prompts:")
            for prompt in prompts_result.prompts:
                print(f"  - {prompt.name}: {prompt.description}")

            # Get a prompt
            prompt = await session.get_prompt(
                "code_review",
                arguments={"code": "def f(x): return x/0", "language": "python"}
            )
            print(f"\nPrompt messages: {[m.content.text for m in prompt.messages]}")

asyncio.run(main())
```

### HTTP Client

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def http_client_example():
    async with streamablehttp_client("http://localhost:8080/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("search", {"query": "machine learning"})
            print(result.content[0].text)
```

### Using MCP with OpenAI Agents SDK

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerSse

# Stdio-based server
local_server = MCPServerStdio(
    command="python",
    args=["weather_server.py"],
    env={"WEATHER_API_KEY": "your-key"}
)

# SSE-based remote server
remote_server = MCPServerSse(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer token123"}
)

agent = Agent(
    name="Assistant",
    instructions="Use the available tools to help users.",
    mcp_servers=[local_server, remote_server],
)

async def main():
    async with local_server, remote_server:
        result = await Runner.run(agent, "What's the weather in Paris?")
        print(result.final_output)
```

### Using MCP with LangChain

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    async with MultiServerMCPClient({
        "filesystem": {
            "command": "python",
            "args": ["-m", "mcp_server_filesystem"],
            "transport": "stdio",
        },
        "github": {
            "url": "https://api.github.com/mcp",
            "transport": "streamable_http",
            "headers": {"Authorization": "token ghp_..."}
        }
    }) as client:
        tools = client.get_tools()  # LangChain-compatible tools
        agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools)
        result = await agent.ainvoke({"messages": [("user", "List my GitHub repos")]})
        print(result["messages"][-1].content)
```

---

## 11. Authentication and Security

### Authentication in MCP

MCP itself is transport-agnostic for authentication, but the OAuth 2.0 authorization framework is the recommended standard for remote servers.

### OAuth 2.0 for Remote Servers

```python
from mcp.server.fastmcp import FastMCP
from functools import wraps

mcp = FastMCP("Secure Server")

def require_auth(f):
    """Decorator to require a valid bearer token."""
    @wraps(f)
    async def wrapper(*args, **kwargs):
        # In a real implementation, validate against your auth server
        # For FastMCP, use middleware or lifespan to inject auth context
        return await f(*args, **kwargs)
    return wrapper

# For HTTP transport, validate Authorization header
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

bearer_scheme = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            "your-secret-key",
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# The FastMCP server can be wrapped in FastAPI for auth
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("Auth Server")

@mcp.tool()
def private_data(user_id: str) -> str:
    """Access private user data."""
    return get_user_data(user_id)

# Mount MCP with auth
app.include_router(
    mcp.router,
    prefix="/mcp",
    dependencies=[Depends(verify_token)]
)
```

### Input Validation and Sanitization

```python
from mcp.server.fastmcp import FastMCP
import re
import os
from pathlib import Path

mcp = FastMCP("Secure File Server")

WORKSPACE = Path("/safe/workspace").resolve()

def validate_path(path: str) -> Path:
    """Prevent path traversal attacks."""
    requested = (WORKSPACE / path).resolve()
    if not requested.is_relative_to(WORKSPACE):
        raise ValueError(f"Access denied: path outside workspace")
    return requested

def validate_sql(sql: str) -> str:
    """Ensure only SELECT statements are allowed."""
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are permitted")
    # Block dangerous keywords even in SELECT
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "EXEC"]
    for keyword in dangerous:
        if keyword in stripped:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")
    return sql

@mcp.tool()
def read_file(path: str) -> str:
    """Read a file within the workspace.

    Args:
        path: Relative path to file within workspace.
    """
    safe_path = validate_path(path)
    if not safe_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return safe_path.read_text()

@mcp.tool()
def query_database(sql: str) -> str:
    """Execute a read-only SQL query.

    Args:
        sql: SQL SELECT query to execute.
    """
    validated_sql = validate_sql(sql)
    return execute_query(validated_sql)
```

### Rate Limiting

```python
from collections import defaultdict
from time import time
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode

mcp = FastMCP("Rate Limited Server")

# Simple in-memory rate limiter
_call_times: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 10   # calls per minute
WINDOW = 60       # seconds

def check_rate_limit(tool_name: str):
    now = time()
    calls = _call_times[tool_name]
    # Remove calls older than the window
    _call_times[tool_name] = [t for t in calls if now - t < WINDOW]
    if len(_call_times[tool_name]) >= RATE_LIMIT:
        raise McpError(
            ErrorCode.INVALID_REQUEST,
            f"Rate limit exceeded for {tool_name}. Try again in {WINDOW}s."
        )
    _call_times[tool_name].append(now)

@mcp.tool()
def expensive_api_call(query: str) -> str:
    """Call an expensive external API.

    Args:
        query: Query to send.
    """
    check_rate_limit("expensive_api_call")
    return call_external_api(query)
```

---

## 12. MCP in Claude Desktop, Cursor, and Continue

### Claude Desktop Configuration

Claude Desktop reads MCP server configuration from:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/alice/projects"],
      "description": "Local file access"
    },
    "sqlite": {
      "command": "python",
      "args": ["-m", "mcp_server_sqlite", "--db-path", "/Users/alice/app.db"],
      "env": {}
    },
    "github": {
      "command": "node",
      "args": ["/usr/local/bin/mcp-server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    },
    "remote-analytics": {
      "url": "https://analytics.mycompany.com/mcp",
      "transport": "streamable-http",
      "headers": {
        "Authorization": "Bearer ${ANALYTICS_TOKEN}"
      }
    }
  }
}
```

After editing, restart Claude Desktop. Servers will be available as tools automatically.

### Cursor Configuration

In Cursor, add MCP servers in Settings → MCP Servers (or `.cursor/mcp.json` in project):

```json
{
  "mcpServers": {
    "project-db": {
      "command": "python",
      "args": ["scripts/mcp_db_server.py"],
      "env": {
        "DATABASE_URL": "sqlite:///./app.db"
      }
    },
    "web-search": {
      "command": "npx",
      "args": ["-y", "mcp-server-tavily"],
      "env": {
        "TAVILY_API_KEY": "tvly-your-key"
      }
    }
  }
}
```

### Continue (VS Code Extension) Configuration

In `~/.continue/config.json`:

```json
{
  "experimental": {
    "modelContextProtocolServers": [
      {
        "transport": {
          "type": "stdio",
          "command": "python",
          "args": ["my_mcp_server.py"],
          "env": {}
        }
      }
    ]
  }
}
```

---

## 13. Building Real MCP Servers

### Filesystem Server

```python
"""
Full-featured filesystem MCP server with security controls.
"""
from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os
import json
import hashlib
from datetime import datetime

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "./workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("Filesystem Server")

def safe_path(relative: str) -> Path:
    resolved = (WORKSPACE / relative).resolve()
    if not resolved.is_relative_to(WORKSPACE):
        raise PermissionError(f"Access denied: '{relative}' is outside workspace")
    return resolved

# ─── Resources ──────────────────────────────────────────────────
@mcp.resource("fs://workspace")
def list_workspace() -> str:
    """List all files and directories in the workspace."""
    entries = []
    for entry in WORKSPACE.rglob("*"):
        rel = entry.relative_to(WORKSPACE)
        entries.append({
            "path": str(rel),
            "type": "dir" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else None,
            "modified": datetime.fromtimestamp(entry.stat().st_mtime).isoformat()
        })
    return json.dumps(entries, indent=2)

@mcp.resource("fs://files/{path}")
def read_file_resource(path: str) -> str:
    """Read a file's content as a resource."""
    return safe_path(path).read_text(encoding="utf-8")

# ─── Tools ──────────────────────────────────────────────────────
@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Relative path within the workspace.
    """
    full_path = safe_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not full_path.is_file():
        raise ValueError(f"Not a file: {path}")
    return full_path.read_text(encoding="utf-8")

@mcp.tool()
def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    """Write content to a file.

    Args:
        path: Relative path within the workspace.
        content: Content to write.
        mode: Write mode: 'overwrite' or 'append'.
    """
    full_path = safe_path(path)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "append":
        full_path.open("a").write(content)
    else:
        full_path.write_text(content, encoding="utf-8")

    return f"Written {len(content)} chars to {path}"

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List files and directories.

    Args:
        path: Relative directory path (defaults to workspace root).
    """
    full_path = safe_path(path)
    if not full_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    entries = []
    for entry in sorted(full_path.iterdir()):
        entries.append({
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else None
        })
    return json.dumps(entries, indent=2)

@mcp.tool()
def delete_file(path: str) -> str:
    """Delete a file (NOT reversible!).

    Args:
        path: Relative path to file to delete.
    """
    full_path = safe_path(path)
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    full_path.unlink()
    return f"Deleted: {path}"

@mcp.tool()
def search_files(pattern: str, directory: str = ".") -> str:
    """Search for files matching a pattern.

    Args:
        pattern: Glob pattern like '*.py' or '**/*.json'.
        directory: Directory to search within.
    """
    base = safe_path(directory)
    matches = list(base.glob(pattern))
    return json.dumps([str(m.relative_to(WORKSPACE)) for m in matches[:50]])

if __name__ == "__main__":
    mcp.run()
```

### SQLite Database Server

```python
"""
SQLite MCP server with read-only query support and schema introspection.
"""
import sqlite3
import json
import os
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode

DB_PATH = os.environ.get("SQLITE_DB_PATH", "app.db")

mcp = FastMCP("SQLite Server")

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ─── Resources ──────────────────────────────────────────────────
@mcp.resource("sqlite://schema")
def get_full_schema() -> str:
    """Complete database schema including all tables and indexes."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
        )
        objects = [{"type": row["type"], "name": row["name"], "sql": row["sql"]} for row in cursor]
        return json.dumps(objects, indent=2)
    finally:
        conn.close()

@mcp.resource("sqlite://tables/{table_name}/schema")
def get_table_schema(table_name: str) -> str:
    """Schema for a specific table."""
    conn = get_connection()
    try:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [dict(row) for row in cursor.fetchall()]

        cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
        fkeys = [dict(row) for row in cursor.fetchall()]

        cursor = conn.execute(f"PRAGMA index_list({table_name})")
        indexes = [dict(row) for row in cursor.fetchall()]

        return json.dumps({
            "table": table_name,
            "columns": columns,
            "foreign_keys": fkeys,
            "indexes": indexes
        }, indent=2)
    finally:
        conn.close()

@mcp.resource("sqlite://tables/{table_name}/sample")
def get_sample_data(table_name: str) -> str:
    """First 10 rows from a table for context."""
    conn = get_connection()
    try:
        cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 10")
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows, indent=2)
    finally:
        conn.close()

# ─── Tools ──────────────────────────────────────────────────────
@mcp.tool()
def execute_query(sql: str) -> str:
    """Execute a read-only SQL SELECT query.

    Args:
        sql: SQL SELECT statement to execute. Only SELECT is allowed.
    """
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT"):
        raise McpError(ErrorCode.INVALID_PARAMS, "Only SELECT queries are allowed")

    forbidden = {"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "REPLACE", "TRUNCATE"}
    words = set(stripped.split())
    blocked = forbidden & words
    if blocked:
        raise McpError(ErrorCode.INVALID_PARAMS, f"Forbidden keywords: {blocked}")

    conn = get_connection()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(1000)   # max 1000 rows
        cols = [d[0] for d in cursor.description] if cursor.description else []
        result = [dict(zip(cols, row)) for row in rows]

        meta = {"count": len(result), "columns": cols}
        if len(result) == 1000:
            meta["warning"] = "Result truncated to 1000 rows"

        return json.dumps({"meta": meta, "data": result}, indent=2)
    except sqlite3.Error as e:
        raise McpError(ErrorCode.INTERNAL_ERROR, f"Query error: {e}")
    finally:
        conn.close()

@mcp.tool()
def list_tables() -> str:
    """List all tables in the database with row counts."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        result = []
        for table in tables:
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = count_cursor.fetchone()[0]
            result.append({"name": table, "rows": count})
        return json.dumps(result, indent=2)
    finally:
        conn.close()

@mcp.prompt()
def analyze_table(table_name: str) -> str:
    """Generate a prompt to analyze a database table.

    Args:
        table_name: Name of the table to analyze.
    """
    return (
        f"Analyze the '{table_name}' database table.\n\n"
        f"1. First, read the schema from resource: sqlite://tables/{table_name}/schema\n"
        f"2. Then, look at sample data: sqlite://tables/{table_name}/sample\n"
        f"3. Run any queries needed using the execute_query tool\n"
        f"4. Provide insights about the data: patterns, anomalies, data quality issues"
    )

if __name__ == "__main__":
    mcp.run()
```

### Web Search MCP Server

```python
"""
Web search MCP server using Tavily API.
"""
import os
import json
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode
from typing import Literal, Optional

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
TAVILY_BASE_URL = "https://api.tavily.com"

mcp = FastMCP("Web Search Server")

@mcp.tool()
async def search(
    query: str,
    search_depth: Literal["basic", "advanced"] = "basic",
    max_results: int = 5,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    topic: Literal["general", "news", "finance"] = "general"
) -> str:
    """Search the web for current information.

    Args:
        query: What to search for.
        search_depth: 'basic' is faster, 'advanced' is more thorough.
        max_results: Number of results to return (1-10).
        include_domains: Only search these domains (e.g., ['wikipedia.org', 'arxiv.org']).
        exclude_domains: Exclude these domains from results.
        topic: Search topic category for better relevance.
    """
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "max_results": min(max(1, max_results), 10),
        "topic": topic,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{TAVILY_BASE_URL}/search", json=payload)

    if response.status_code != 200:
        raise McpError(ErrorCode.INTERNAL_ERROR, f"Search API error: {response.status_code}")

    data = response.json()
    results = data.get("results", [])

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    {r.get('content', '')[:300]}..."
        )

    answer = data.get("answer", "")
    output = ""
    if answer:
        output = f"DIRECT ANSWER: {answer}\n\n"
    output += f"SEARCH RESULTS for '{query}':\n\n" + "\n\n".join(formatted)
    return output

@mcp.tool()
async def fetch_page(url: str, extract_text: bool = True) -> str:
    """Fetch and extract content from a web page.

    Args:
        url: URL of the page to fetch.
        extract_text: If True, extract readable text. If False, return raw HTML.
    """
    payload = {
        "api_key": TAVILY_API_KEY,
        "urls": [url],
        "extract_depth": "advanced"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{TAVILY_BASE_URL}/extract", json=payload)

    if response.status_code != 200:
        raise McpError(ErrorCode.INTERNAL_ERROR, f"Extract API error: {response.status_code}")

    data = response.json()
    results = data.get("results", [])
    if not results:
        return "No content could be extracted from this URL"

    result = results[0]
    content = result.get("raw_content" if not extract_text else "text", "")
    return f"URL: {url}\nTitle: {result.get('title', 'Unknown')}\n\nContent:\n{content[:5000]}"

if __name__ == "__main__":
    mcp.run()
```

### GitHub API MCP Server

```python
"""
GitHub API MCP server for repository operations.
"""
import os
import json
import base64
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode
from typing import Optional, Literal

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_API = "https://api.github.com"

mcp = FastMCP("GitHub Server")

def github_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

async def github_get(path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{GITHUB_API}{path}", headers=github_headers(), params=params)
    if response.status_code == 404:
        raise McpError(ErrorCode.INVALID_PARAMS, f"GitHub resource not found: {path}")
    if response.status_code >= 400:
        raise McpError(ErrorCode.INTERNAL_ERROR, f"GitHub API error {response.status_code}: {response.text}")
    return response.json()

async def github_post(path: str, data: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{GITHUB_API}{path}", headers=github_headers(), json=data)
    if response.status_code >= 400:
        raise McpError(ErrorCode.INTERNAL_ERROR, f"GitHub API error {response.status_code}: {response.text}")
    return response.json()

# ─── Resources ──────────────────────────────────────────────────
@mcp.resource("github://{owner}/{repo}/readme")
async def get_readme(owner: str, repo: str) -> str:
    """README file for a GitHub repository."""
    headers = {**github_headers(), "Accept": "application/vnd.github.raw"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}/readme", headers=headers)
    if response.status_code == 404:
        return "No README found"
    return response.text

@mcp.resource("github://{owner}/{repo}/issues")
async def get_issues_resource(owner: str, repo: str) -> str:
    """Recent open issues for a repository."""
    issues = await github_get(f"/repos/{owner}/{repo}/issues", params={"state": "open", "per_page": 20})
    return json.dumps([{
        "number": i["number"],
        "title": i["title"],
        "state": i["state"],
        "labels": [l["name"] for l in i.get("labels", [])],
        "created_at": i["created_at"]
    } for i in issues], indent=2)

# ─── Tools ──────────────────────────────────────────────────────
@mcp.tool()
async def search_repositories(
    query: str,
    language: Optional[str] = None,
    sort: Literal["stars", "forks", "updated"] = "stars",
    limit: int = 10
) -> str:
    """Search GitHub repositories.

    Args:
        query: Search query (supports GitHub search syntax).
        language: Filter by programming language.
        sort: Sort results by stars, forks, or update date.
        limit: Number of results (max 30).
    """
    q = query
    if language:
        q += f" language:{language}"

    data = await github_get("/search/repositories", params={"q": q, "sort": sort, "per_page": min(limit, 30)})
    repos = data.get("items", [])

    results = [{
        "name": r["full_name"],
        "description": r.get("description", ""),
        "stars": r["stargazers_count"],
        "language": r.get("language", ""),
        "url": r["html_url"],
        "updated": r["updated_at"]
    } for r in repos]

    return json.dumps(results, indent=2)

@mcp.tool()
async def list_issues(
    repo: str,
    state: Literal["open", "closed", "all"] = "open",
    labels: Optional[str] = None,
    limit: int = 20
) -> str:
    """List issues for a GitHub repository.

    Args:
        repo: Repository in 'owner/repo' format.
        state: Filter by issue state.
        labels: Comma-separated label names to filter by.
        limit: Max number of issues to return.
    """
    owner, name = repo.split("/", 1)
    params = {"state": state, "per_page": min(limit, 100)}
    if labels:
        params["labels"] = labels

    issues = await github_get(f"/repos/{owner}/{name}/issues", params=params)
    result = [{
        "number": i["number"],
        "title": i["title"],
        "state": i["state"],
        "body": (i.get("body") or "")[:200],
        "labels": [l["name"] for l in i.get("labels", [])],
        "assignees": [a["login"] for a in i.get("assignees", [])],
        "created_at": i["created_at"],
        "url": i["html_url"]
    } for i in issues if "pull_request" not in i]   # exclude PRs

    return json.dumps(result, indent=2)

@mcp.tool()
async def create_issue(
    repo: str,
    title: str,
    body: str,
    labels: Optional[list[str]] = None,
    assignees: Optional[list[str]] = None
) -> str:
    """Create a new GitHub issue.

    Args:
        repo: Repository in 'owner/repo' format.
        title: Issue title.
        body: Issue body in Markdown.
        labels: Labels to apply.
        assignees: GitHub usernames to assign.
    """
    owner, name = repo.split("/", 1)
    data = {"title": title, "body": body}
    if labels:
        data["labels"] = labels
    if assignees:
        data["assignees"] = assignees

    issue = await github_post(f"/repos/{owner}/{name}/issues", data)
    return json.dumps({
        "number": issue["number"],
        "url": issue["html_url"],
        "title": issue["title"]
    })

@mcp.tool()
async def get_file_content(repo: str, path: str, branch: str = "main") -> str:
    """Get the content of a file from a GitHub repository.

    Args:
        repo: Repository in 'owner/repo' format.
        path: File path within the repository.
        branch: Branch name (defaults to 'main').
    """
    owner, name = repo.split("/", 1)
    data = await github_get(f"/repos/{owner}/{name}/contents/{path}", params={"ref": branch})

    if isinstance(data, list):
        # It's a directory
        return json.dumps([{"name": f["name"], "type": f["type"], "path": f["path"]} for f in data])

    content = base64.b64decode(data["content"]).decode("utf-8")
    return f"File: {path}\nSize: {data['size']} bytes\n\n{content}"

if __name__ == "__main__":
    mcp.run()
```

---

## 14. Testing MCP Servers

### MCP Inspector (Official Tool)

The MCP Inspector is a browser-based GUI for testing servers interactively:

```bash
# Install and run
npx @modelcontextprotocol/inspector python my_server.py

# With environment variables
TAVILY_API_KEY=tvly-xxx npx @modelcontextprotocol/inspector python web_search_server.py

# For HTTP transport
npx @modelcontextprotocol/inspector --transport http --url http://localhost:8080/mcp
```

The inspector opens at `http://localhost:5173` and shows:
- Server capabilities
- All tools with interactive forms to test them
- Resources list and reader
- Prompts with argument inputs
- Raw JSON-RPC message log

### Automated Testing

```python
"""
Unit tests for MCP server tools.
"""
import pytest
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@pytest.fixture
async def mcp_session():
    """Fixture that starts the server and creates a session."""
    params = StdioServerParameters(
        command="python",
        args=["my_server.py"],
        env={"SQLITE_DB_PATH": ":memory:"}   # in-memory DB for tests
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@pytest.mark.asyncio
async def test_list_tools(mcp_session):
    """Verify all expected tools are exposed."""
    tools = await mcp_session.list_tools()
    tool_names = {t.name for t in tools.tools}
    assert "execute_query" in tool_names
    assert "list_tables" in tool_names

@pytest.mark.asyncio
async def test_execute_query(mcp_session):
    """Test SQL execution."""
    result = await mcp_session.call_tool(
        "execute_query",
        arguments={"sql": "SELECT 1 as num, 'hello' as greeting"}
    )
    assert not result.isError
    import json
    data = json.loads(result.content[0].text)
    assert data["data"][0]["num"] == 1
    assert data["data"][0]["greeting"] == "hello"

@pytest.mark.asyncio
async def test_query_rejects_writes(mcp_session):
    """Ensure write operations are blocked."""
    result = await mcp_session.call_tool(
        "execute_query",
        arguments={"sql": "DROP TABLE users"}
    )
    assert result.isError

@pytest.mark.asyncio
async def test_read_resource(mcp_session):
    """Test resource reading."""
    content = await mcp_session.read_resource("sqlite://schema")
    assert len(content.contents) > 0
    schema = content.contents[0].text
    assert "sqlite_master" in schema or schema.startswith("[")

# Run: pytest test_server.py -v
```

### Testing with the Python Client Directly

```python
"""
Integration test script that tests all server capabilities.
"""
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_server():
    params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=== TOOLS ===")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"✓ {tool.name}")

            print("\n=== TOOL CALLS ===")
            tests = [
                ("add_numbers", {"a": 3, "b": 4}),
                ("get_weather", {"city": "Tokyo"}),
            ]
            for tool_name, args in tests:
                result = await session.call_tool(tool_name, arguments=args)
                status = "✓" if not result.isError else "✗"
                print(f"{status} {tool_name}: {result.content[0].text[:100]}")

            print("\n=== RESOURCES ===")
            resources = await session.list_resources()
            for r in resources.resources:
                content = await session.read_resource(r.uri)
                print(f"✓ {r.uri}: {len(content.contents[0].text)} chars")

asyncio.run(test_server())
```

---

## 15. Deployment and Containerization

### Docker Container for MCP Server

```dockerfile
# Dockerfile for Python MCP server
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Run as stdio server (for local use)
CMD ["python", "server.py"]
```

```dockerfile
# Dockerfile for HTTP MCP server (remote deployment)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

EXPOSE 8080
CMD ["python", "server.py", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
```

```yaml
# docker-compose.yml for MCP server with dependencies
version: "3.8"

services:
  mcp-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - DATABASE_URL=postgresql://postgres:password@db:5432/app
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

### requirements.txt

```
mcp>=1.0.0
httpx>=0.27.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pytz>=2024.1
# For database server
sqlite3  # built-in
psycopg2-binary>=2.9.9  # for PostgreSQL
# For search server
tavily-python>=0.3.0
# For auth
python-jose[cryptography]>=3.3.0
```

---

## 16. MCP Server Registry and Discovery

### Official Registry

The official MCP server registry is at: https://github.com/modelcontextprotocol/servers

Notable servers:
- `@modelcontextprotocol/server-filesystem` — local file operations
- `@modelcontextprotocol/server-brave-search` — web search via Brave
- `@modelcontextprotocol/server-puppeteer` — browser automation
- `@modelcontextprotocol/server-slack` — Slack integration
- `@modelcontextprotocol/server-github` — GitHub operations
- `@modelcontextprotocol/server-postgres` — PostgreSQL access
- `@modelcontextprotocol/server-sqlite` — SQLite access
- `@modelcontextprotocol/server-memory` — knowledge graph memory
- `mcp-server-tavily` — Tavily web search

### Community Registry

An unofficial but comprehensive registry: https://mcp.so  
Another: https://smithery.ai

### Registering Your Own Server

```yaml
# .mcp/server.yaml — metadata for your server
name: my-weather-server
version: 1.0.0
description: Real-time weather data via OpenWeatherMap API
author: Your Name
license: MIT

capabilities:
  tools:
    - get_weather
    - get_forecast
    - get_historical
  resources:
    - weather://current/{city}
    - weather://forecast/{city}/{days}
  prompts:
    - weather_analysis

transport:
  - stdio
  - streamable-http

requirements:
  env:
    OPENWEATHER_API_KEY: "OpenWeatherMap API key"

install:
  pip: my-weather-mcp-server
  command: python -m weather_mcp_server
```

---

## 17. Full Production-Ready Examples

### Production MCP Server with All Features

```python
"""
Production-ready MCP server template with:
- Structured tools with validation
- Resources with caching
- Prompts for common workflows
- Error handling and logging
- Rate limiting
- Health monitoring
"""
import os
import json
import logging
import time
from functools import wraps
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import McpError, ErrorCode

# ─── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("production-mcp")

# ─── Configuration ───────────────────────────────────────────────
@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ["API_KEY"])
    rate_limit_calls: int = int(os.environ.get("RATE_LIMIT_CALLS", "20"))
    rate_limit_window: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
    cache_ttl: int = int(os.environ.get("CACHE_TTL", "300"))
    max_response_size: int = int(os.environ.get("MAX_RESPONSE_SIZE", str(1024 * 1024)))

# ─── Rate Limiter ────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, calls: int, window: int):
        self.calls = calls
        self.window = window
        self._log: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str):
        now = time.time()
        self._log[key] = [t for t in self._log[key] if now - t < self.window]
        if len(self._log[key]) >= self.calls:
            wait = self.window - (now - self._log[key][0])
            raise McpError(
                ErrorCode.INVALID_REQUEST,
                f"Rate limit exceeded. Wait {wait:.0f}s before retrying."
            )
        self._log[key].append(now)

# ─── Cache ───────────────────────────────────────────────────────
class TTLCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._store:
            ts, value = self._store[key]
            if time.time() - ts < self.ttl:
                return value
            del self._store[key]
        return None

    def set(self, key: str, value: str):
        self._store[key] = (time.time(), value)

# ─── App State ───────────────────────────────────────────────────
@dataclass
class AppState:
    config: Config
    rate_limiter: RateLimiter
    cache: TTLCache
    http_client: httpx.AsyncClient
    request_count: int = 0
    error_count: int = 0

# ─── Server Setup ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(server: FastMCP):
    config = Config()
    state = AppState(
        config=config,
        rate_limiter=RateLimiter(config.rate_limit_calls, config.rate_limit_window),
        cache=TTLCache(config.cache_ttl),
        http_client=httpx.AsyncClient(
            timeout=30,
            headers={"Authorization": f"Bearer {config.api_key}"}
        )
    )
    logger.info("Server started")
    try:
        yield state
    finally:
        await state.http_client.aclose()
        logger.info(f"Server shutdown. Requests: {state.request_count}, Errors: {state.error_count}")

mcp = FastMCP("Production Server", lifespan=lifespan)

# ─── Tools ──────────────────────────────────────────────────────
@mcp.tool()
async def search(ctx: AppState, query: str, category: Optional[str] = None) -> str:
    """Search for information.

    Args:
        query: Search query.
        category: Optional category filter.
    """
    ctx.rate_limiter.check("search")
    cache_key = f"search:{query}:{category}"

    # Check cache
    cached = ctx.cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for query: {query}")
        return cached

    logger.info(f"Searching: {query}")
    ctx.request_count += 1

    try:
        response = await ctx.http_client.get(
            "https://api.example.com/search",
            params={"q": query, "category": category}
        )
        response.raise_for_status()
        data = response.json()
        result = json.dumps(data, indent=2)

        if len(result) > ctx.config.max_response_size:
            result = result[:ctx.config.max_response_size] + "\n... [truncated]"

        ctx.cache.set(cache_key, result)
        return result

    except httpx.HTTPStatusError as e:
        ctx.error_count += 1
        logger.error(f"HTTP error: {e}")
        raise McpError(ErrorCode.INTERNAL_ERROR, f"API error: {e.response.status_code}")
    except httpx.RequestError as e:
        ctx.error_count += 1
        logger.error(f"Request error: {e}")
        raise McpError(ErrorCode.INTERNAL_ERROR, f"Network error: {e}")

@mcp.resource("stats://server/metrics")
async def get_metrics(ctx: AppState) -> str:
    """Current server performance metrics."""
    return json.dumps({
        "requests_total": ctx.request_count,
        "errors_total": ctx.error_count,
        "error_rate": ctx.error_count / max(ctx.request_count, 1),
        "cache_size": len(ctx.cache._store)
    })

if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
    else:
        mcp.run()
```

---

## 18. Best Practices

### Server Development

```python
# 1. Write clear, detailed tool descriptions — the LLM reads them
@mcp.tool()
def good_tool(path: str, encoding: str = "utf-8") -> str:
    """Read a text file and return its contents.

    Use this when the user asks to read, view, or examine a file.
    Returns the raw file content as a string.

    Args:
        path: Relative path to the file (e.g., 'src/main.py', 'config.json').
        encoding: Text encoding to use (default: utf-8).
    """
    ...

# Bad: vague description
@mcp.tool()
def bad_tool(p: str) -> str:
    """Process a path."""
    ...

# 2. Validate all inputs
@mcp.tool()
def safe_file_read(path: str) -> str:
    """Read a file safely."""
    # Always validate paths
    safe = safe_path(path)   # raises PermissionError if outside workspace
    # Check size before reading
    size = safe.stat().st_size
    if size > 10 * 1024 * 1024:  # 10MB
        raise ValueError(f"File too large: {size / 1024 / 1024:.1f}MB")
    return safe.read_text()

# 3. Keep servers focused
# Good: one server per domain
weather_server = FastMCP("Weather Server")    # only weather
github_server = FastMCP("GitHub Server")      # only GitHub
db_server = FastMCP("Database Server")        # only database

# Bad: one server for everything
# all_in_one = FastMCP("Everything Server")   # hard to maintain and secure
```

### Security Checklist

```
Input Validation:
  ✓ Validate all tool arguments before using them
  ✓ Sanitize paths (prevent traversal)
  ✓ Validate SQL to prevent injection
  ✓ Check file sizes before reading

Access Control:
  ✓ Use environment variables for secrets (never hardcode)
  ✓ Implement rate limiting
  ✓ Apply principle of least privilege
  ✓ Log all tool invocations for audit

Output Safety:
  ✓ Never return raw secrets or tokens
  ✓ Truncate large outputs
  ✓ Don't expose internal error details to LLM
  ✓ Set isError=true for business logic failures

Transport Security:
  ✓ Use HTTPS/TLS for remote servers
  ✓ Validate API keys on every request
  ✓ Set appropriate CORS headers
  ✓ Use OAuth 2.0 for user auth flows
```

### Performance Tips

```python
# 1. Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text: str) -> list[float]:
    return embed(text)

# 2. Use async for I/O-bound operations
@mcp.tool()
async def async_tool(query: str) -> str:
    # Good: non-blocking
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return response.text

# 3. Return only what's needed
@mcp.tool()
def search_with_pagination(query: str, page: int = 1, limit: int = 10) -> str:
    """Search with pagination to avoid huge responses."""
    results = full_search(query)
    page_results = results[(page-1)*limit : page*limit]
    return json.dumps({
        "total": len(results),
        "page": page,
        "results": page_results,
        "has_more": page * limit < len(results)
    })
```

---

## 19. MCP Server Implementation Patterns

Choosing the right implementation pattern depends on deployment (local vs remote), state requirements, and scaling needs.

### 19.1 Stateless vs Stateful Servers

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Stateless** | Each tool call is independent; no shared mutable state | Web search, weather API, translation |
| **Stateful (in-process)** | Server holds connection pools, caches, or sessions during lifespan | Database server with connection pool |
| **Stateful (external)** | State in Redis/DB; server is stateless but state lives elsewhere | User preferences, conversation history |

```python
# Stateless: no shared state
@mcp.tool()
async def search(query: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api.example.com/search?q={query}")
    return r.json()

# Stateful: lifespan-managed resources
@asynccontextmanager
async def lifespan(server: FastMCP):
    pool = await create_db_pool()
    yield AppState(pool=pool)
    await pool.close()
```

### 19.2 Transport Selection

| Transport | Best For | Considerations |
|-----------|----------|----------------|
| **stdio** | Local tools (Cursor, Claude Desktop), single-user | One process per client; no network |
| **Streamable HTTP** | Remote servers, multi-user, SaaS | Needs auth, CORS, timeout handling |
| **WebSocket** | Low-latency bidirectional, server push | More complex; use when notifications matter |

### 19.3 Single-Tool vs Multi-Capability Servers

- **Single-tool server**: One focused tool (e.g., `run_sql`). Easy to secure and reason about.
- **Multi-capability server**: Tools + resources + prompts. Good when domain is cohesive (e.g., "GitHub server" with `create_issue`, `list_repos`, `get_readme` resource).
- **Avoid**: Mixing unrelated domains (e.g., GitHub + weather) — split into separate servers.

### 19.4 Connection and Lifespan Patterns

```python
# Pattern: Lazy initialization — create DB connection on first use
_db = None
def get_db():
    global _db
    if _db is None:
        _db = sqlite3.connect(os.environ["DB_PATH"])
    return _db

# Pattern: Lifespan for cleanup — close resources on shutdown
@asynccontextmanager
async def lifespan(server: FastMCP):
    resources = init_resources()
    yield resources
    await resources.cleanup()
```

### 19.5 Error Handling Patterns

| Pattern | When to Use |
|---------|-------------|
| **McpError with ErrorCode** | Invalid params, method not found — client can react |
| **Raise Python exception** | FastMCP converts to protocol error |
| **Return isError=true in content** | Business logic failure (e.g., "Rate limit exceeded") — LLM sees message |
| **Log + re-raise** | Internal errors — log for debugging, then raise |

---

## 20. Pitfalls and Common Mistakes

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Vague tool descriptions** | LLM picks wrong tool or passes bad arguments | Write "Use when X" and "Returns Y"; include examples in description |
| **Path traversal** | User passes `../../etc/passwd` and reads system files | Resolve path, check `is_relative_to(workspace)` |
| **SQL injection** | LLM-generated SQL contains `DROP` or `DELETE` | Whitelist `SELECT` only; block dangerous keywords |
| **Exposing secrets in responses** | API keys or tokens appear in tool output to LLM | Redact secrets; return only necessary data |
| **No rate limiting** | Single client can exhaust API quotas or DB connections | Per-tool or global rate limits; backpressure |
| **Synchronous I/O in async tools** | Blocking `requests.get()` stalls the event loop | Use `httpx.AsyncClient`, `aiofiles`, async DB drivers |
| **Mixing transports** | stdio server configured with HTTP URL (or vice versa) | Match client config to server's transport |
| **Ignoring capabilities** | Client expects `resources/subscribe` but server doesn't advertise it | Check `capabilities` in `initialize` result |
| **Large responses** | Returning 100KB+ of text wastes context window | Paginate; truncate; offer `limit` parameter |
| **Forgetting to test with Inspector** | Ship server that fails on edge cases | Use `npx @modelcontextprotocol/inspector` before release |

---

### Summary Reference

| Primitive | Direction | Who Controls | Use For |
|-----------|-----------|-------------|---------|
| **Tools** | Client → Server → Execute | LLM decides | Actions: search, create, compute |
| **Resources** | Client → Server → Read | App decides | Context: files, schemas, docs |
| **Prompts** | Client → Server → Template | User decides | Workflows: /review, /summarize |
| **Sampling** | Server → Client → LLM | Server requests | Server-side LLM calls |
| **Notifications** | Server → Client → App | Server pushes | Change alerts |

| Ecosystem | Details |
|-----------|---------|
| **Python SDK** | `pip install mcp` → `FastMCP` framework |
| **TypeScript SDK** | `npm install @modelcontextprotocol/sdk` |
| **Inspector** | `npx @modelcontextprotocol/inspector` |
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Cursor** | Settings → MCP Servers or `.cursor/mcp.json` |
| **Continue** | `~/.continue/config.json` |
| **Protocol Spec** | https://modelcontextprotocol.io/specification |
| **Server Registry** | https://github.com/modelcontextprotocol/servers |
| **Community** | https://mcp.so |

### References

- **MCP Specification**: https://modelcontextprotocol.io/specification
- **Anthropic MCP Announcement** (Dec 2024): https://www.anthropic.com/news/model-context-protocol
- **Python SDK (mcp)**: https://github.com/modelcontextprotocol/python-sdk
- **TypeScript SDK**: https://github.com/modelcontextprotocol/typescript-sdk
- **MCP Server Registry**: https://github.com/modelcontextprotocol/servers
