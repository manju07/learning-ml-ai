# OpenAI Agents SDK: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Core Primitives](#core-primitives)
4. [Agents](#agents)
5. [Tools](#tools)
6. [Handoffs](#handoffs)
7. [Guardrails](#guardrails)
8. [Runner and Agent Loop](#runner-and-agent-loop)
9. [Context and Dependency Injection](#context-and-dependency-injection)
10. [Structured Outputs](#structured-outputs)
11. [Tracing and Observability](#tracing-and-observability)
12. [Multi-Agent Patterns](#multi-agent-patterns)
13. [MCP Integration](#mcp-integration)
14. [Streaming](#streaming)
15. [Practical Examples](#practical-examples)
16. [Best Practices](#best-practices)
17. [Common Pitfalls](#common-pitfalls)
18. [Production Considerations](#production-considerations)
19. [References](#references)

---

## Introduction

The **OpenAI Agents SDK** (`openai-agents`) is a lightweight, production-ready Python framework for building multi-agent AI applications. It's the successor to the experimental Swarm library and the recommended stack for agentic workloads with OpenAI models.

### Design Principles

1. **Works great out of the box**, but you can customize exactly what happens
2. **Enough features** to be worth using, but **few enough primitives** to learn quickly

### Core Primitives

| Primitive | Purpose |
|-----------|---------|
| **Agent** | LLM configured with instructions and tools |
| **Tool** | Function, hosted capability, or sub-agent the agent can call |
| **Handoff** | Structured delegation to another agent |
| **Guardrail** | Input/output validation hooks |
| **Tracing** | Built-in observability and debugging |

### Key Features

- **Function tools** with automatic schema generation and Pydantic validation
- **Agents as tools / Handoffs** for multi-agent coordination
- **Guardrails** that run in parallel with agent execution
- **Built-in tracing** viewable in the OpenAI dashboard
- **MCP server** integration for standardized tool calling
- **Realtime agents** for voice with interruption detection
- **Human-in-the-loop** mechanisms
- **Sessions** for persistent memory
- **Python-first**: Use language features, not new abstractions

---

## Installation and Setup

```bash
pip install openai-agents
```

```bash
export OPENAI_API_KEY=sk-...
```

### Hello World

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

---

## Core Primitives

### The Agent Loop

When you call `Runner.run()`, the SDK runs an **agent loop**:

```
1. Call the LLM with the agent's instructions and conversation
2. If the LLM produces a final output → return it
3. If the LLM makes tool calls → execute them, add results to conversation
4. If the LLM produces a handoff → switch to the new agent
5. Go to step 1
```

This loop continues until a final output is produced or max turns is reached.

---

## Agents

### Basic Agent

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.7),
)
```

### Dynamic Instructions

Instructions can be a function that receives the context and agent:

```python
from agents import Agent, RunContextWrapper

def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."

agent = Agent[UserContext](
    name="Triage agent",
    instructions=dynamic_instructions,
)
```

### Structured Output

Force the agent to return a specific type using `output_type`:

```python
from pydantic import BaseModel
from agents import Agent

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)
```

### Cloning Agents

```python
pirate_agent = Agent(
    name="Pirate",
    instructions="Write like a pirate",
    model="gpt-4o",
)

robot_agent = pirate_agent.clone(
    name="Robot",
    instructions="Write like a robot",
)
```

---

## Tools

### Function Tools

Any Python function can become a tool with `@function_tool`. Schema is generated automatically from type hints and docstrings.

```python
from agents import Agent, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: The city to get weather for.
    """
    return f"The weather in {city} is sunny, 72°F"

@function_tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the internal database.
    
    Args:
        query: Search query string.
        limit: Maximum number of results.
    """
    return f"Found {limit} results for '{query}'"

agent = Agent(
    name="Assistant",
    tools=[get_weather, search_database],
)
```

### Function Tool with Context

```python
from agents import function_tool, RunContextWrapper
from typing import Any

@function_tool
def get_user_orders(ctx: RunContextWrapper[UserContext], status: str = "all") -> str:
    """Get orders for the current user.
    
    Args:
        status: Filter by order status.
    """
    user_id = ctx.context.uid
    orders = db.get_orders(user_id, status=status)
    return str(orders)
```

### Hosted Tools (OpenAI)

Built-in tools that run on OpenAI servers:

```python
from agents import Agent, WebSearchTool, FileSearchTool, CodeInterpreterTool, ImageGenerationTool

agent = Agent(
    name="Research assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["vs_abc123"],
        ),
        CodeInterpreterTool(),
        ImageGenerationTool(),
    ],
)
```

### Local Runtime Tools

Tools that execute in your environment:

```python
from agents import Agent, ShellTool, ApplyPatchTool

async def run_shell(request):
    import subprocess
    result = subprocess.run(request.command, shell=True, capture_output=True, text=True)
    return result.stdout

agent = Agent(
    name="Dev agent",
    tools=[
        ShellTool(executor=run_shell),
    ],
)
```

### Agents as Tools

Expose a sub-agent as a tool (the main agent retains control):

```python
from agents import Agent

research_agent = Agent(
    name="Researcher",
    instructions="You research topics thoroughly and return detailed findings.",
)

writer_agent = Agent(
    name="Writer",
    instructions="You write polished articles. Use the researcher for facts.",
    tools=[
        research_agent.as_tool(
            tool_name="research",
            tool_description="Research a topic and return detailed findings.",
        ),
    ],
)
```

### Forcing Tool Use

```python
from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info."""
    return f"Sunny in {city}"

agent = Agent(
    name="Weather Agent",
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="required"),  # Must use a tool
)
```

### Tool Use Behavior

Control what happens after a tool call:

```python
from agents import Agent, function_tool
from agents.agent import StopAtTools

@function_tool
def lookup(query: str) -> str:
    """Look up information."""
    return f"Result for {query}"

@function_tool
def calculate(expr: str) -> str:
    """Evaluate expression."""
    return str(eval(expr))

# stop_on_first_tool: Use tool output directly as final answer
agent_fast = Agent(
    name="Fast Agent",
    tools=[lookup],
    tool_use_behavior="stop_on_first_tool",
)

# StopAtTools: Stop only on specific tools
agent_selective = Agent(
    name="Selective Agent",
    tools=[lookup, calculate],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["lookup"]),
)

# run_llm_again (default): Process tool results through LLM
agent_default = Agent(
    name="Default Agent",
    tools=[lookup],
    tool_use_behavior="run_llm_again",
)
```

---

## Handoffs

Handoffs let an agent **delegate the entire conversation** to another agent. The new agent takes over and gets the full conversation history.

### Basic Handoffs

```python
from agents import Agent

billing_agent = Agent(
    name="Billing agent",
    instructions="You handle billing questions. Be precise about amounts and dates.",
)

refund_agent = Agent(
    name="Refund agent",
    instructions="You handle refund requests. Always ask for order number.",
)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "You are the first point of contact. "
        "Hand off to billing for payment questions, refund agent for return requests."
    ),
    handoffs=[billing_agent, refund_agent],
)
```

### Customized Handoffs

```python
from agents import Agent, handoff, RunContextWrapper
from pydantic import BaseModel

class EscalationData(BaseModel):
    reason: str
    priority: str

async def on_escalation(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation: {input_data.reason} (priority: {input_data.priority})")
    # Log, notify, etc.

escalation_agent = Agent(name="Escalation agent", instructions="Handle escalated issues.")

handoff_obj = handoff(
    agent=escalation_agent,
    on_handoff=on_escalation,
    input_type=EscalationData,
    tool_name_override="escalate",
    tool_description_override="Escalate to a senior agent with reason and priority",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Escalate if the customer is frustrated or the issue is complex.",
    handoffs=[handoff_obj],
)
```

### Input Filters

Control what conversation history the next agent sees:

```python
from agents import Agent, handoff
from agents.extensions import handoff_filters

faq_agent = Agent(name="FAQ agent", instructions="Answer FAQs concisely.")

handoff_obj = handoff(
    agent=faq_agent,
    input_filter=handoff_filters.remove_all_tools,  # Clean history for FAQ agent
)
```

### Handoff Prompt Extension

```python
from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You handle all billing and payment related questions.""",
)
```

---

## Guardrails

Guardrails validate inputs and outputs, running in parallel with the agent or blocking before execution.

### Input Guardrail

```python
from pydantic import BaseModel
from agents import (
    Agent, Runner, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, RunContextWrapper,
    TResponseInputItem, input_guardrail,
)

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)

@input_guardrail
async def math_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_homework,
    )

agent = Agent(
    name="Customer support",
    instructions="Help customers with their questions.",
    input_guardrails=[math_guardrail],
)

async def main():
    try:
        await Runner.run(agent, "Can you solve 2x + 3 = 11?")
    except InputGuardrailTripwireTriggered:
        print("Blocked: math homework detected")
```

### Output Guardrail

```python
from agents import Agent, output_guardrail, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered

@output_guardrail
async def no_pii_guardrail(ctx, agent, output) -> GuardrailFunctionOutput:
    has_pii = check_for_pii(output.response)
    return GuardrailFunctionOutput(
        output_info={"has_pii": has_pii},
        tripwire_triggered=has_pii,
    )

agent = Agent(
    name="Support agent",
    output_guardrails=[no_pii_guardrail],
    output_type=ResponseModel,
)
```

### Tool Guardrails

Validate tool inputs/outputs:

```python
from agents import function_tool, tool_input_guardrail, tool_output_guardrail, ToolGuardrailFunctionOutput
import json

@tool_input_guardrail
def block_secrets(data):
    args = json.loads(data.context.tool_arguments or "{}")
    if "sk-" in json.dumps(args):
        return ToolGuardrailFunctionOutput.reject_content("Remove secrets before calling this tool.")
    return ToolGuardrailFunctionOutput.allow()

@tool_output_guardrail
def redact_output(data):
    if "sk-" in str(data.output or ""):
        return ToolGuardrailFunctionOutput.reject_content("Output contained sensitive data.")
    return ToolGuardrailFunctionOutput.allow()

@function_tool(
    tool_input_guardrails=[block_secrets],
    tool_output_guardrails=[redact_output],
)
def classify_text(text: str) -> str:
    """Classify text for internal routing."""
    return f"length:{len(text)}"
```

### Execution Modes

```python
# Parallel (default): Guardrail runs alongside the agent
math_guardrail  # run_in_parallel=True by default

# Blocking: Guardrail runs first, agent only starts if guardrail passes
@input_guardrail(run_in_parallel=False)
async def blocking_guardrail(ctx, agent, input):
    ...
```

---

## Runner and Agent Loop

### Three Ways to Run

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are helpful.")

# 1. Synchronous
result = Runner.run_sync(agent, "Hello")

# 2. Async
result = await Runner.run(agent, "Hello")

# 3. Async with streaming
result = Runner.run_streamed(agent, "Hello")
async for event in result.stream_events():
    print(event)
```

### Run Configuration

```python
from agents import Runner, RunConfig

result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(
        max_turns=10,
        tracing_disabled=False,
        trace_include_sensitive_data=True,
    ),
)
```

### Accessing Results

```python
result = Runner.run_sync(agent, "What is 2+2?")

print(result.final_output)        # The final text/structured response
print(result.last_agent)          # Which agent produced the final output
print(result.input)               # Original input
print(result.new_items)           # All items generated during the run
```

---

## Context and Dependency Injection

Context is a generic object passed to every agent, tool, and guardrail. It's the dependency injection mechanism.

```python
from dataclasses import dataclass
from agents import Agent, Runner, function_tool, RunContextWrapper

@dataclass
class AppContext:
    user_id: str
    user_name: str
    is_premium: bool
    db: DatabaseClient

@function_tool
def get_account_info(ctx: RunContextWrapper[AppContext]) -> str:
    """Get the current user's account information."""
    user = ctx.context.db.get_user(ctx.context.user_id)
    return f"Name: {user.name}, Plan: {'Premium' if ctx.context.is_premium else 'Free'}"

agent = Agent[AppContext](
    name="Account agent",
    instructions="Help users with account questions.",
    tools=[get_account_info],
)

# Pass context when running
ctx = AppContext(user_id="123", user_name="Alice", is_premium=True, db=db_client)
result = Runner.run_sync(agent, "What's my account status?", context=ctx)
```

---

## Structured Outputs

Force the agent to return structured data:

```python
from pydantic import BaseModel
from agents import Agent, Runner

class SentimentResult(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    reasoning: str

agent = Agent(
    name="Sentiment analyzer",
    instructions="Analyze sentiment of the given text.",
    output_type=SentimentResult,
)

result = Runner.run_sync(agent, "I absolutely love this product! Best purchase ever.")
output: SentimentResult = result.final_output
print(f"Sentiment: {output.sentiment} ({output.confidence:.0%})")
print(f"Reason: {output.reasoning}")
```

---

## Tracing and Observability

Tracing is **enabled by default**. Every run automatically captures LLM calls, tool executions, handoffs, and guardrails.

### Viewing Traces

Traces are visible at https://platform.openai.com/traces

### Custom Trace Names

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke agent", instructions="Tell funny jokes.")
    
    # Wrap multiple runs in a single trace
    with trace("Joke workflow"):
        joke = await Runner.run(agent, "Tell me a joke")
        rating = await Runner.run(agent, f"Rate this joke: {joke.final_output}")
        print(f"Joke: {joke.final_output}")
        print(f"Rating: {rating.final_output}")
```

### Run Config for Tracing

```python
from agents import Runner, RunConfig

result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(
        workflow_name="Customer Support",
        trace_id="trace_custom123",
        group_id="conversation_456",
        trace_include_sensitive_data=False,
    ),
)
```

### Custom Trace Processor

Send traces to external services (LangSmith, Arize, W&B, etc.):

```python
from agents import add_trace_processor

class MyProcessor:
    def process_trace(self, trace):
        # Send to your observability platform
        my_platform.log(trace)

add_trace_processor(MyProcessor())
```

### Non-OpenAI Models

```python
from agents import set_tracing_export_api_key
from agents.extensions.models.litellm_model import LitellmModel

set_tracing_export_api_key("sk-...")  # OpenAI key for tracing only

model = LitellmModel(model="anthropic/claude-3-opus", api_key="...")
agent = Agent(name="Claude Agent", model=model)
```

---

## Multi-Agent Patterns

### Conceptual Overview: Handoffs vs Agents-as-Tools

| Dimension | **Handoffs** | **Agents as Tools** |
|-----------|--------------|---------------------|
| **Control** | New agent takes over the *entire* conversation | Calling agent retains control and sees tool output |
| **Use case** | Customer routing (billing → refund → specialist) | Orchestration (manager delegates research, then synthesizes) |
| **Context** | Full conversation history flows to the new agent | Only the sub-agent's output flows back |
| **Flow** | One-way; caller doesn't return unless handoff back | Multi-step; caller can invoke multiple times |
| **Example** | Triage → billing for payment questions | Manager uses researcher tool, then writer tool |

**When to use handoffs:** User intent changes (e.g., "actually I need a refund") or domain shifts where a specialist should own the session.

**When to use agents-as-tools:** You need a coordinator to gather inputs from specialists and produce a synthesized result (research → write report).

### Pattern 1: Handoffs (Decentralized)

Agents hand off the conversation to specialists:

```python
from agents import Agent

order_agent = Agent(
    name="Order agent",
    instructions="Handle order status, tracking, and modifications.",
)

returns_agent = Agent(
    name="Returns agent",
    instructions="Handle returns and exchanges. Always ask for order number.",
)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Determine customer intent. "
        "Hand off to order agent for order questions, returns agent for returns."
    ),
    handoffs=[order_agent, returns_agent],
)
```

### Pattern 2: Manager (Centralized, Agents as Tools)

One agent calls sub-agents as tools and retains control:

```python
from agents import Agent

research_agent = Agent(
    name="Researcher",
    instructions="Research the given topic and return key findings.",
)

analyst_agent = Agent(
    name="Analyst",
    instructions="Analyze data and provide insights.",
)

manager_agent = Agent(
    name="Manager",
    instructions="You coordinate research and analysis tasks.",
    tools=[
        research_agent.as_tool(
            tool_name="research",
            tool_description="Research a topic thoroughly",
        ),
        analyst_agent.as_tool(
            tool_name="analyze",
            tool_description="Analyze data or findings",
        ),
    ],
)
```

### Pattern 3: Sequential Pipeline

```python
from agents import Agent, Runner

summarizer = Agent(name="Summarizer", instructions="Summarize the given text concisely.")
translator = Agent(name="Translator", instructions="Translate to Spanish.")

async def pipeline(text):
    summary = await Runner.run(summarizer, text)
    translated = await Runner.run(translator, summary.final_output)
    return translated.final_output
```

---

## MCP Integration

The SDK has built-in support for **Model Context Protocol (MCP)** servers:

```python
from agents import Agent
from agents.mcp import MCPServerStdio, MCPServerSse

# Stdio-based MCP server
mcp_server = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
)

# SSE-based MCP server
mcp_sse = MCPServerSse(url="http://localhost:3000/sse")

agent = Agent(
    name="File agent",
    instructions="Help users with file operations.",
    mcp_servers=[mcp_server],
)

async def main():
    async with mcp_server:
        result = await Runner.run(agent, "List files in the directory")
        print(result.final_output)
```

### Hosted MCP Tool

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

## Streaming

### Event-Based Streaming

```python
from agents import Agent, Runner

agent = Agent(name="Storyteller", instructions="Tell engaging stories.")

async def main():
    result = Runner.run_streamed(agent, "Tell me a short story about a robot.")
    
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # Token-level streaming
            if hasattr(event.data, "delta"):
                print(event.data.delta, end="", flush=True)
    
    print(f"\n\nFinal: {result.final_output}")
```

---

## Practical Examples

### Example 1: Customer Support System

```python
from agents import Agent, Runner, function_tool, handoff
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class SupportContext:
    customer_id: str
    db: object

@function_tool
def lookup_order(ctx: RunContextWrapper[SupportContext], order_id: str) -> str:
    """Look up order details by order ID."""
    return ctx.context.db.get_order(order_id)

@function_tool
def process_refund(ctx: RunContextWrapper[SupportContext], order_id: str, reason: str) -> str:
    """Process a refund for an order."""
    return f"Refund initiated for order {order_id}. Reason: {reason}"

billing_agent = Agent[SupportContext](
    name="Billing agent",
    instructions="Handle billing and payment questions.",
    tools=[lookup_order],
)

refund_agent = Agent[SupportContext](
    name="Refund agent",
    instructions="Handle refund requests. Always confirm order ID and reason.",
    tools=[lookup_order, process_refund],
)

triage_agent = Agent[SupportContext](
    name="Triage agent",
    instructions=(
        "You are the first point of contact for customer support. "
        "Route billing questions to billing agent, refund requests to refund agent."
    ),
    handoffs=[billing_agent, refund_agent],
)

# Run
ctx = SupportContext(customer_id="cust_123", db=database)
result = Runner.run_sync(triage_agent, "I want a refund for order ORD-456", context=ctx)
print(result.final_output)
```

### Example 2: Research Agent with Web Search

```python
from agents import Agent, Runner, WebSearchTool

agent = Agent(
    name="Research assistant",
    instructions=(
        "You are a research assistant. Search the web to find accurate, "
        "up-to-date information. Always cite your sources."
    ),
    tools=[WebSearchTool()],
    model="gpt-4o",
)

result = Runner.run_sync(agent, "What are the latest developments in quantum computing?")
print(result.final_output)
```

### Example 3: Data Extraction Pipeline

```python
from pydantic import BaseModel
from agents import Agent, Runner

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str
    total: float
    line_items: list[dict]

extractor = Agent(
    name="Invoice extractor",
    instructions="Extract structured invoice data from the provided text.",
    output_type=Invoice,
)

text = """
INVOICE #INV-2024-001
From: Acme Corp
Date: January 15, 2024
Items:
- Widget A x10 @ $5.00 = $50.00
- Widget B x5 @ $12.00 = $60.00
Total: $110.00
"""

result = Runner.run_sync(extractor, text)
invoice: Invoice = result.final_output
print(f"Vendor: {invoice.vendor}, Total: ${invoice.total}")
```

### Example 4: Agent with Guardrail

```python
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered

@input_guardrail
async def language_check(ctx, agent, input) -> GuardrailFunctionOutput:
    """Block non-English input."""
    checker = Agent(name="Lang check", instructions="Is this English? Output only: yes or no")
    result = await Runner.run(checker, input, context=ctx.context)
    is_non_english = "no" in result.final_output.lower()
    return GuardrailFunctionOutput(tripwire_triggered=is_non_english)

agent = Agent(
    name="English-only agent",
    instructions="You only respond in English.",
    input_guardrails=[language_check],
)

try:
    result = Runner.run_sync(agent, "Bonjour, comment allez-vous?")
except InputGuardrailTripwireTriggered:
    print("Sorry, I only support English.")
```

---

## Best Practices

### 1. Keep Agents Focused

```python
# Good: Single responsibility
billing_agent = Agent(name="Billing", instructions="Handle billing only.")
support_agent = Agent(name="Support", instructions="Handle technical support only.")

# Bad: One agent does everything
do_everything = Agent(name="Everything", instructions="Handle billing, support, sales, HR...")
```

### 2. Use Context for Dependencies

```python
# Good: Inject via context
@dataclass
class Deps:
    db: Database
    api: ExternalAPI

# Bad: Global state in tools
```

### 3. Add Guardrails for Production

```python
# Always validate inputs in production
agent = Agent(
    name="Production agent",
    input_guardrails=[content_filter, rate_limiter],
    output_guardrails=[pii_checker],
)
```

### 4. Use Structured Outputs

```python
# Good: Typed, validated output
agent = Agent(output_type=StructuredResponse)

# Instead of: parsing free text
```

### 5. Enable Tracing

```python
# Use workflow names for organization
run_config = RunConfig(workflow_name="Customer Support v2")
```

### 6. Handle Errors

```python
from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

try:
    result = await Runner.run(agent, user_input)
except InputGuardrailTripwireTriggered:
    return "I can't help with that request."
except OutputGuardrailTripwireTriggered:
    return "I encountered an issue generating a response."
except Exception as e:
    log_error(e)
    return "Something went wrong."
```

---

## Summary

| Concept | Description |
|---------|-------------|
| **Agent** | LLM + instructions + tools |
| **function_tool** | Python function → tool with auto schema |
| **Handoff** | Delegate conversation to another agent |
| **Agent.as_tool()** | Use agent as a tool (manager retains control) |
| **Guardrail** | Input/output/tool validation |
| **Runner** | Executes agent loop (sync, async, streaming) |
| **Context** | Dependency injection across agents and tools |
| **output_type** | Force structured (Pydantic) output |
| **Tracing** | Built-in observability |
| **MCP** | Standard protocol for external tools |

**Install**: `pip install openai-agents`  
**Docs**: https://openai.github.io/openai-agents-python/  
**GitHub**: https://github.com/openai/openai-agents-python

---

## References

| Resource | Description |
|----------|-------------|
| [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/) | Official documentation |
| [OpenAI Agents GitHub](https://github.com/openai/openai-agents-python) | Source code, examples, issues |
| [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) | Standard for tool integration |
| [OpenAI Traces](https://platform.openai.com/traces) | Observability dashboard |
| ReAct (Yao et al., 2022) | Reasoning + Acting paradigm |
| Swarm (Legacy) | Experimental predecessor to agents SDK |
