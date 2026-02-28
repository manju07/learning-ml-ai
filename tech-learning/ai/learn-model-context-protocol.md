# Model Context Protocol (MCP): Complete Guide

## Table of Contents
1. [Introduction to MCP](#introduction-to-mcp)
2. [Architecture](#architecture)
3. [Transport Layer](#transport-layer)
4. [Lifecycle and Initialization](#lifecycle-and-initialization)
5. [Tools](#tools)
6. [Resources](#resources)
7. [Prompts](#prompts)
8. [Sampling](#sampling)
9. [Building an MCP Server (Python)](#building-an-mcp-server-python)
10. [Building an MCP Client](#building-an-mcp-client)
11. [MCP with OpenAI Agents SDK](#mcp-with-openai-agents-sdk)
12. [MCP with LangChain](#mcp-with-langchain)
13. [Advanced Topics](#advanced-topics)
14. [Practical Examples](#practical-examples)
15. [Best Practices](#best-practices)

---

## Introduction to MCP

The **Model Context Protocol (MCP)** is an open standard (created by Anthropic) that standardizes how AI applications connect to external data sources and tools. Think of MCP as **USB-C for AI**—a universal plug that works across models, hosts, and tools.

### The Problem MCP Solves

Without MCP, every AI tool integration requires a custom connector:

```
Before MCP:
  Claude ↔ custom connector ↔ GitHub
  Claude ↔ custom connector ↔ Slack
  GPT    ↔ different connector ↔ GitHub
  GPT    ↔ different connector ↔ Slack
  = N models × M tools = N×M integrations

With MCP:
  Claude ↔ MCP client ↔ MCP server ↔ GitHub
  GPT    ↔ MCP client ↔ MCP server ↔ GitHub (same server!)
  = N + M implementations
```

### Core Capabilities

MCP servers expose three types of features to clients:

| Capability | Description | Controlled By | Example |
|------------|-------------|---------------|---------|
| **Tools** | Functions the LLM can call | Model-initiated | `get_weather(city)` |
| **Resources** | Read-only data for context | Application-driven | Files, DB schemas |
| **Prompts** | Template messages/workflows | User-initiated | `/code-review`, `/summarize` |

Clients may offer features to servers:

| Feature | Description | Direction |
|---------|-------------|-----------|
| **Sampling** | Server requests LLM completion via client | Server → Client → LLM |
| **Roots** | Server asks for filesystem boundaries | Server → Client |
| **Elicitation** | Server asks user for information | Server → Client → User |

### Protocol Foundation

- **Message format**: JSON-RPC 2.0
- **Transports**: Stdio (local) and Streamable HTTP (remote)
- **Capability negotiation**: Client and server declare supported features at initialization

---

## Architecture

### Participants

MCP follows a **client-host-server** architecture:

```
┌──────────────────────────────────┐
│        MCP Host (AI App)         │
│   e.g., Claude Desktop, Cursor   │
│                                  │
│  ┌──────────┐  ┌──────────┐     │
│  │MCP Client│  │MCP Client│     │
│  │    1     │  │    2     │     │
│  └────┬─────┘  └────┬─────┘     │
└───────┼──────────────┼───────────┘
        │              │
   ┌────▼─────┐   ┌────▼─────┐
   │MCP Server│   │MCP Server│
   │ (local)  │   │ (remote) │
   │Filesystem│   │  Sentry  │
   └──────────┘   └──────────┘
```

- **Host**: The AI application (Claude Desktop, Cursor, VS Code). Coordinates multiple clients, aggregates context, enforces security.
- **Client**: Connector within the host. Each maintains a 1:1 stateful connection to one server.
- **Server**: Provides tools, resources, and prompts. Can be local (stdio) or remote (HTTP).

### Two Layers

1. **Transport layer** (outer): Communication mechanisms—stdio, HTTP, connection establishment, message framing
2. **Data layer** (inner): JSON-RPC protocol for tools, resources, prompts, lifecycle

### Key Design Principles

- Servers should be **extremely easy to build**
- Servers should be **composable**—combine multiple servers for different capabilities
- Servers **cannot see** the full conversation or other servers (host controls security)
- Features are **progressively adoptable**—backwards compatible

---

## Transport Layer

### Stdio Transport (Local)

Server runs as a subprocess; communication via stdin/stdout.

```
Host spawns: python my_mcp_server.py
  → stdin:  Host sends JSON-RPC messages
  ← stdout: Server sends JSON-RPC responses
  ← stderr: Server logs (not protocol)
```

Best for: Local tools (filesystem, databases, dev tools).

### Streamable HTTP Transport (Remote)

Server is an HTTP endpoint; uses Server-Sent Events (SSE) for server→client messages.

```
Client → POST /mcp → Server (sends JSON-RPC request)
Server → SSE stream → Client (sends responses/notifications)
```

Best for: Remote services (SaaS integrations, cloud APIs).

### Message Framing

All messages are JSON-RPC 2.0:

```json
// Request
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}

// Response
{"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}

// Notification (no id, no response expected)
{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
```

---

## Lifecycle and Initialization

### Connection Flow

```
Client                          Server
  |                               |
  |------ initialize ------------>|
  |<----- initialize result ------|  (capabilities, server info)
  |                               |
  |------ initialized ----------->|  (notification: client ready)
  |                               |
  |   ... normal operations ...   |
  |                               |
  |------ shutdown/disconnect --->|
```

### Initialize Request

Client sends its capabilities; server responds with its capabilities.

```json
// Client → Server
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "sampling": {},
      "roots": { "listChanged": true }
    },
    "clientInfo": { "name": "my-app", "version": "1.0" }
  }
}

// Server → Client
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": { "subscribe": true, "listChanged": true },
      "prompts": { "listChanged": true }
    },
    "serverInfo": { "name": "weather-server", "version": "1.0" }
  }
}
```

### Capability Negotiation

Only declared capabilities can be used. If server doesn't declare `resources`, client won't try `resources/list`.

---

## Tools

**Tools** are functions the LLM can call. The model discovers tools, selects which to call, and the client executes them via the server.

### Tool Definition

```json
{
  "name": "get_weather",
  "title": "Weather Information Provider",
  "description": "Get current weather information for a location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": { "type": "string", "description": "City name or zip code" }
    },
    "required": ["location"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": { "type": "number" },
      "conditions": { "type": "string" },
      "humidity": { "type": "number" }
    },
    "required": ["temperature", "conditions", "humidity"]
  }
}
```

### Discovery and Invocation Flow

```
1. Client sends: tools/list
   Server returns: [get_weather, search_db, ...]

2. LLM decides to call get_weather("New York")

3. Client sends: tools/call { name: "get_weather", arguments: { location: "New York" } }
   Server executes function, returns result

4. Client passes result back to LLM for processing
```

### Tool Result Types

```json
// Text result
{ "type": "text", "text": "72°F, partly cloudy" }

// Image result
{ "type": "image", "data": "base64...", "mimeType": "image/png" }

// Audio result
{ "type": "audio", "data": "base64...", "mimeType": "audio/wav" }

// Resource link (pointer to data the client can fetch)
{ "type": "resource_link", "uri": "file:///data/report.csv", "name": "report.csv" }

// Structured result (with outputSchema validation)
{
  "content": [{ "type": "text", "text": "{\"temperature\": 22.5}" }],
  "structuredContent": { "temperature": 22.5, "conditions": "Sunny", "humidity": 65 }
}
```

### Error Handling

Two kinds: **protocol errors** (JSON-RPC) and **tool execution errors** (business logic):

```json
// Protocol error (e.g., unknown tool)
{ "jsonrpc": "2.0", "id": 3, "error": { "code": -32602, "message": "Unknown tool" } }

// Tool execution error (e.g., API failure)
{
  "jsonrpc": "2.0", "id": 4,
  "result": {
    "content": [{ "type": "text", "text": "API rate limit exceeded" }],
    "isError": true
  }
}
```

---

## Resources

**Resources** provide read-only contextual data (files, database schemas, API responses). They're application-driven—the host decides how to use them.

### Resource Definition

```json
{
  "uri": "file:///project/src/main.rs",
  "name": "main.rs",
  "title": "Rust Application Main File",
  "description": "Primary application entry point",
  "mimeType": "text/x-rust",
  "annotations": {
    "audience": ["assistant"],
    "priority": 0.9,
    "lastModified": "2025-01-12T15:00:58Z"
  }
}
```

### Operations

```
resources/list           → List available resources (paginated)
resources/read           → Read a specific resource by URI
resources/templates/list → List parameterized resource templates
resources/subscribe      → Subscribe to changes on a resource
```

### Resource Templates

Parameterized URIs using RFC 6570:

```json
{
  "uriTemplate": "db://tables/{table_name}/schema",
  "name": "Table Schema",
  "description": "Get schema for any database table"
}
```

### Subscriptions

```json
// Subscribe
{ "method": "resources/subscribe", "params": { "uri": "file:///config.yaml" } }

// Server notifies on change
{ "method": "notifications/resources/updated", "params": { "uri": "file:///config.yaml" } }
// Client then re-reads the resource
```

### URI Schemes

- `file://` — Filesystem-like resources
- `https://` — Web resources client can fetch directly
- `git://` — Version control
- Custom schemes: `db://`, `slack://`, etc.

---

## Prompts

**Prompts** are reusable templates exposed by servers. They're user-initiated (like slash commands).

### Prompt Definition

```json
{
  "name": "code_review",
  "title": "Request Code Review",
  "description": "Asks the LLM to analyze code quality and suggest improvements",
  "arguments": [
    { "name": "code", "description": "The code to review", "required": true },
    { "name": "language", "description": "Programming language", "required": false }
  ]
}
```

### Getting a Prompt

```json
// Request
{ "method": "prompts/get", "params": { "name": "code_review", "arguments": { "code": "def f(x): return x*2" } } }

// Response: structured messages for the LLM
{
  "result": {
    "messages": [
      {
        "role": "user",
        "content": { "type": "text", "text": "Please review this code:\ndef f(x): return x*2" }
      }
    ]
  }
}
```

### Prompt Messages Can Include

- Text, images, audio
- Embedded resources (inline file contents)
- Multiple messages (user + assistant for few-shot)

---

## Sampling

**Sampling** lets the server ask the client to run an LLM completion. This enables **agentic behaviors** inside MCP servers—the server doesn't need its own API key.

### Flow

```
Server: "I need an LLM to analyze this data"
  → Server sends sampling/createMessage to Client
  → Client presents to user for approval
  → Client forwards to LLM
  → LLM generates response
  → Client returns result to Server
```

### Request

```json
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      { "role": "user", "content": { "type": "text", "text": "Summarize: ..." } }
    ],
    "modelPreferences": {
      "hints": [{ "name": "claude-3-sonnet" }],
      "intelligencePriority": 0.8,
      "speedPriority": 0.5,
      "costPriority": 0.3
    },
    "systemPrompt": "You are a helpful analyst.",
    "maxTokens": 500
  }
}
```

### Model Preferences

Servers suggest models abstractly (not by exact name):
- **hints**: Preferred models (advisory, client makes final choice)
- **intelligencePriority / speedPriority / costPriority**: Trade-off preferences (0-1)

---

## Building an MCP Server (Python)

### Installation

```bash
pip install mcp
```

### FastMCP: Minimal Server

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather Server")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city
    """
    # In production: call weather API
    return f"Weather in {city}: 72°F, sunny"

@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast.
    
    Args:
        city: Name of the city
        days: Number of days to forecast
    """
    return f"{days}-day forecast for {city}: sunny, cloudy, rain"

@mcp.resource("config://app")
def get_config() -> str:
    """Application configuration."""
    return '{"theme": "dark", "language": "en"}'

@mcp.prompt()
def review_code(code: str, language: str = "python") -> str:
    """Generate a code review prompt."""
    return f"Review this {language} code for bugs and improvements:\n\n```{language}\n{code}\n```"

if __name__ == "__main__":
    mcp.run()  # Defaults to stdio transport
```

### Running with Different Transports

```python
# Stdio (default, for local tools)
mcp.run()

# Streamable HTTP (for remote servers)
mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

### Resources with Templates

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Database Server")

@mcp.resource("db://tables")
def list_tables() -> str:
    """List all database tables."""
    tables = db.get_tables()
    return "\n".join(tables)

@mcp.resource("db://tables/{table_name}/schema")
def get_table_schema(table_name: str) -> str:
    """Get schema for a specific table."""
    schema = db.get_schema(table_name)
    return str(schema)

@mcp.resource("db://tables/{table_name}/sample")
def get_sample_data(table_name: str) -> str:
    """Get sample rows from a table."""
    rows = db.query(f"SELECT * FROM {table_name} LIMIT 5")
    return str(rows)
```

### Tools with Complex Types

```python
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Task Manager")

class Task(BaseModel):
    title: str
    description: str
    priority: str = "medium"

@mcp.tool()
def create_task(title: str, description: str, priority: str = "medium") -> str:
    """Create a new task.
    
    Args:
        title: Task title
        description: Task description
        priority: Priority level (low, medium, high)
    """
    task_id = db.insert_task(title, description, priority)
    return f"Created task {task_id}: {title} (priority: {priority})"

@mcp.tool()
def search_tasks(query: str, status: str = "open") -> str:
    """Search for tasks.
    
    Args:
        query: Search query
        status: Filter by status (open, closed, all)
    """
    results = db.search_tasks(query, status)
    return "\n".join([f"- [{t.id}] {t.title} ({t.status})" for t in results])
```

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector python my_server.py
# Opens browser UI to test tools, resources, prompts
```

---

## Building an MCP Client

### Python Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["my_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"Tool: {tool.name} - {tool.description}")
            
            # Call a tool
            result = await session.call_tool("get_weather", {"city": "New York"})
            print(result.content[0].text)
            
            # List resources
            resources = await session.list_resources()
            for r in resources.resources:
                print(f"Resource: {r.uri}")
            
            # Read a resource
            content = await session.read_resource("config://app")
            print(content.contents[0].text)
```

---

## MCP with OpenAI Agents SDK

### Stdio MCP Server

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

# Connect to a local MCP server
server = MCPServerStdio(
    command="python",
    args=["my_mcp_server.py"],
)

agent = Agent(
    name="Assistant",
    instructions="Use available tools to help the user.",
    mcp_servers=[server],
)

async def main():
    async with server:  # Manages server lifecycle
        result = await Runner.run(agent, "What's the weather in Paris?")
        print(result.final_output)
```

### SSE/Remote MCP Server

```python
from agents.mcp import MCPServerSse

remote = MCPServerSse(url="https://my-mcp-server.com/sse")

agent = Agent(
    name="Remote tool agent",
    mcp_servers=[remote],
)
```

### Hosted MCP Tool (OpenAI)

```python
from agents import Agent, HostedMCPTool

agent = Agent(
    name="Agent",
    tools=[
        HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        )
    ],
)
```

---

## MCP with LangChain

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient({
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],
        "transport": "stdio",
    }
}) as client:
    tools = client.get_tools()
    # Use tools with LangChain agent
    agent = create_react_agent(llm, tools, prompt)
```

---

## Advanced Topics

### Annotations

Resources, prompts, and tool results support annotations:

```json
{
  "audience": ["user", "assistant"],  // Who should see this
  "priority": 0.9,                     // 0.0 (optional) to 1.0 (essential)
  "lastModified": "2025-01-12T15:00:58Z"
}
```

### Pagination

Large lists support cursor-based pagination:

```json
// Request
{ "method": "tools/list", "params": { "cursor": "page2-token" } }

// Response
{ "result": { "tools": [...], "nextCursor": "page3-token" } }
```

### List Change Notifications

Servers can notify clients when available tools/resources/prompts change:

```json
{ "method": "notifications/tools/list_changed" }
```

Client then re-fetches the list.

### Completion API

Auto-complete arguments for tools, prompts, and resource templates:

```json
{ "method": "completion/complete", "params": { "ref": { "type": "ref/prompt", "name": "code_review" }, "argument": { "name": "language", "value": "py" } } }
// Returns: ["python", "pytorch"]
```

### Security Considerations

**Servers MUST**:
- Validate all inputs
- Sanitize outputs
- Implement rate limiting
- Check access controls

**Clients SHOULD**:
- Show tool inputs to user before execution
- Prompt for confirmation on sensitive operations
- Implement timeouts
- Log tool usage for auditing
- Validate tool results before passing to LLM

---

## Practical Examples

### Example 1: Database Query Server

```python
from mcp.server.fastmcp import FastMCP
import sqlite3

mcp = FastMCP("SQLite Server")
DB_PATH = "data.db"

@mcp.resource("db://schema")
def get_schema() -> str:
    """Get the complete database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schemas = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()
    return "\n\n".join(schemas)

@mcp.tool()
def query(sql: str) -> str:
    """Execute a read-only SQL query.
    
    Args:
        sql: SQL SELECT query to execute
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]
        return str(result)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

@mcp.prompt()
def analyze_table(table_name: str) -> str:
    """Generate a prompt to analyze a database table."""
    return (
        f"Analyze the '{table_name}' table. "
        f"First read the schema using the db://schema resource, "
        f"then use the query tool to explore the data."
    )

if __name__ == "__main__":
    mcp.run()
```

### Example 2: GitHub Integration Server

```python
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("GitHub Server")

@mcp.tool()
def list_issues(repo: str, state: str = "open") -> str:
    """List GitHub issues for a repository.
    
    Args:
        repo: Repository in format owner/repo
        state: Issue state (open, closed, all)
    """
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/issues",
        params={"state": state},
        headers={"Authorization": f"token {GITHUB_TOKEN}"}
    )
    issues = response.json()
    return "\n".join([f"#{i['number']}: {i['title']}" for i in issues[:10]])

@mcp.tool()
def create_issue(repo: str, title: str, body: str) -> str:
    """Create a new GitHub issue.
    
    Args:
        repo: Repository in format owner/repo
        title: Issue title
        body: Issue body/description
    """
    response = httpx.post(
        f"https://api.github.com/repos/{repo}/issues",
        json={"title": title, "body": body},
        headers={"Authorization": f"token {GITHUB_TOKEN}"}
    )
    issue = response.json()
    return f"Created issue #{issue['number']}: {issue['html_url']}"

@mcp.resource("github://{repo}/readme")
def get_readme(repo: str) -> str:
    """Get the README for a repository."""
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/readme",
        headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.raw"}
    )
    return response.text

if __name__ == "__main__":
    mcp.run()
```

### Example 3: File Operations Server

```python
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("File Server")
ALLOWED_DIR = "/path/to/workspace"

@mcp.tool()
def read_file(path: str) -> str:
    """Read a file's contents.
    
    Args:
        path: Relative path within the workspace
    """
    full_path = os.path.join(ALLOWED_DIR, path)
    if not full_path.startswith(ALLOWED_DIR):
        return "Error: Access denied (path traversal)"
    with open(full_path, 'r') as f:
        return f.read()

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        path: Relative path within the workspace
        content: Content to write
    """
    full_path = os.path.join(ALLOWED_DIR, path)
    if not full_path.startswith(ALLOWED_DIR):
        return "Error: Access denied"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)
    return f"Written {len(content)} bytes to {path}"

@mcp.tool()
def list_files(directory: str = ".") -> str:
    """List files in a directory.
    
    Args:
        directory: Relative directory path
    """
    full_path = os.path.join(ALLOWED_DIR, directory)
    entries = []
    for entry in os.scandir(full_path):
        prefix = "📁" if entry.is_dir() else "📄"
        entries.append(f"{prefix} {entry.name}")
    return "\n".join(entries)

if __name__ == "__main__":
    mcp.run()
```

---

## Best Practices

### Server Development

1. **Keep servers focused**: One domain per server (files, DB, API)
2. **Use descriptive tool names and descriptions**: The LLM reads them
3. **Validate all inputs**: Never trust tool arguments
4. **Return structured errors**: Use `isError: true` for business errors
5. **Implement pagination**: For large lists
6. **Test with MCP Inspector**: `npx @modelcontextprotocol/inspector`

### Security

1. **Sanitize paths**: Prevent path traversal
2. **Read-only by default**: Only allow writes with explicit user confirmation
3. **Rate limit**: Prevent abuse
4. **Least privilege**: Only expose what's needed
5. **No secrets in tool outputs**: Redact tokens, passwords

### Client Integration

1. **Show tool calls to users**: Transparency
2. **Confirm destructive actions**: Create, update, delete
3. **Timeout long operations**: Don't hang forever
4. **Handle server disconnects**: Reconnect gracefully

---

## Summary

| Concept | Description | Direction |
|---------|-------------|-----------|
| **Tools** | Functions LLM can call | Client → Server |
| **Resources** | Read-only contextual data | Client → Server |
| **Prompts** | Reusable message templates | Client → Server |
| **Sampling** | Server requests LLM call | Server → Client |
| **Notifications** | Change alerts | Server → Client |
| **Transport** | stdio (local) or HTTP (remote) | — |

| Ecosystem | Role |
|-----------|------|
| **Hosts** | Claude Desktop, Cursor, VS Code, Windsurf |
| **Python SDK** | `pip install mcp` (`FastMCP`) |
| **TypeScript SDK** | `npm install @modelcontextprotocol/sdk` |
| **Inspector** | `npx @modelcontextprotocol/inspector` |
| **OpenAI Agents** | `MCPServerStdio`, `MCPServerSse`, `HostedMCPTool` |

**Spec**: https://modelcontextprotocol.io/specification/2025-06-18  
**Python SDK**: https://github.com/modelcontextprotocol/python-sdk  
**Server Registry**: https://github.com/modelcontextprotocol/servers
