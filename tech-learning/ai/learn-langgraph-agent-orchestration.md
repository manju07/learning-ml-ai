# LangGraph & Agent Orchestration: Comprehensive Guide

## Table of Contents
1. [Introduction to Agent Orchestration](#1-introduction-to-agent-orchestration)
2. [LangGraph Fundamentals](#2-langgraph-fundamentals)
3. [State Management](#3-state-management)
4. [Nodes: Building Blocks of Behavior](#4-nodes-building-blocks-of-behavior)
5. [Edges: Routing and Control Flow](#5-edges-routing-and-control-flow)
6. [Tool Integration: ToolNode and tools_condition](#6-tool-integration-toolnode-and-tools_condition)
7. [Checkpointing and Persistence](#7-checkpointing-and-persistence)
8. [Human-in-the-Loop](#8-human-in-the-loop)
9. [Subgraphs and Composition](#9-subgraphs-and-composition)
10. [Streaming](#10-streaming)
11. [Prebuilt Agents: create_react_agent](#11-prebuilt-agents-create_react_agent)
12. [Multi-Agent Patterns](#12-multi-agent-patterns)
13. [Complex Workflow Patterns](#13-complex-workflow-patterns)
14. [Memory Across Conversations](#14-memory-across-conversations)
15. [LangGraph Platform and Cloud Deployment](#15-langgraph-platform-and-cloud-deployment)
16. [Debugging with LangSmith](#16-debugging-with-langsmith)
17. [Full Code Examples](#17-full-code-examples)
18. [Best Practices](#18-best-practices)
19. [Common Pitfalls](#19-common-pitfalls)
20. [Production Considerations](#20-production-considerations)

---

## 1. Introduction to Agent Orchestration

### What Is Agent Orchestration?

**Agent orchestration** is the discipline of coordinating one or more AI agents across complex, multi-step workflows. A raw LLM call is stateless and single-shot: you send a prompt and get a response. But real applications need:

- **State** that accumulates and evolves across multiple steps
- **Branching** logic to take different paths based on intermediate results
- **Loops** to iterate until a quality condition is met
- **Human review gates** before executing high-stakes actions
- **Error recovery** when tools fail or results are unexpected
- **Parallelism** to run independent tasks concurrently
- **Persistence** so workflows can pause and resume

### Chains vs. Graphs

| Dimension | Simple Chain | LangGraph Orchestration |
|-----------|-------------|------------------------|
| Structure | Linear: A → B → C | Directed graph: cycles, branches, parallel |
| State | Stateless or passed manually | Typed state flows through all nodes |
| Human review | Not supported | First-class interrupt/resume |
| Error handling | Fails silently or crashes | Conditional edges for fallback paths |
| Persistence | None | MemorySaver, SqliteSaver, PostgresSaver |
| Visibility | Hard to trace | LangSmith integration built in |
| Composition | Monolithic | Subgraphs, hierarchical agents |

### When to Use LangGraph

Use LangGraph when your workflow has **any** of these characteristics:

- Multiple decision points (route based on classification, tool results, etc.)
- Loops or retry logic (generate → test → fix → re-test)
- Long-running workflows that need to pause and resume
- Human approval before executing destructive or expensive actions
- Multiple specialized agents working together
- Complex state that changes throughout the workflow
- Need for reproducibility and traceability

### Core Concepts at a Glance

```
StateGraph
├── State (TypedDict)      ← Shared memory flowing through the graph
├── Nodes (functions)      ← Units of work that read/write state
├── Edges                  ← Connections between nodes (fixed or conditional)
└── Compiled graph         ← Executable runnable
```

---

## 2. LangGraph Fundamentals

### Installation

```bash
pip install langgraph langchain-openai langchain-core
# Optional: for checkpointing
pip install langgraph-checkpoint-sqlite
pip install langgraph-checkpoint-postgres
```

### Your First Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# 1. Define state schema
class AgentState(TypedDict):
    messages: Annotated[list, add]   # add reducer: new items appended
    step_count: int
    status: str

# 2. Define nodes (functions that transform state)
def step_a(state: AgentState) -> dict:
    print(f"Executing Step A, count={state['step_count']}")
    return {
        "messages": ["Step A completed"],
        "step_count": state["step_count"] + 1,
        "status": "after_a"
    }

def step_b(state: AgentState) -> dict:
    print(f"Executing Step B, count={state['step_count']}")
    return {
        "messages": ["Step B completed"],
        "step_count": state["step_count"] + 1,
        "status": "done"
    }

# 3. Build the graph
builder = StateGraph(AgentState)
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)

# 4. Add edges
builder.set_entry_point("step_a")     # start here
builder.add_edge("step_a", "step_b") # a → b
builder.add_edge("step_b", END)       # b → end

# 5. Compile
graph = builder.compile()

# 6. Invoke
result = graph.invoke({
    "messages": [],
    "step_count": 0,
    "status": "start"
})
print(result["messages"])
# ['Step A completed', 'Step B completed']
```

### How Compilation Works

`graph.compile()` validates the graph (no orphan nodes, entry point set, all conditional edges covered), creates an optimized executor, and optionally attaches a checkpointer and interrupt configuration.

---

## 3. State Management

### TypedDict State Schema

The state is a `TypedDict` — every node receives the full current state and returns a partial dict with the fields it wants to update.

```python
from typing import TypedDict, Annotated, Optional, List
from operator import add
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    # Accumulates: each node can add to this list
    messages: Annotated[List[BaseMessage], add]
    search_results: Annotated[list, add]   # append-only

    # Replace: new value overwrites old
    query: str
    analysis: str
    draft: str
    feedback: Optional[str]
    iteration: int
    status: str   # "searching" | "analyzing" | "drafting" | "reviewing" | "done"
    quality_score: float
```

### Reducers: How State Fields Update

A **reducer** defines how the current field value and the new returned value are combined:

```python
from typing import Annotated
from operator import add

# Pattern 1: Replace (default — no annotation needed)
class SimpleState(TypedDict):
    counter: int    # new value replaces old

# Pattern 2: Append (using operator.add on lists)
class AppendState(TypedDict):
    items: Annotated[list, add]   # [1,2] + [3] → [1,2,3]

# Pattern 3: Custom reducer
def keep_latest_5(current: list, new: list) -> list:
    """Keep only the latest 5 items."""
    combined = current + new
    return combined[-5:]

class BoundedState(TypedDict):
    recent_items: Annotated[list, keep_latest_5]

# Pattern 4: LangChain messages reducer (handles human/AI/tool messages)
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    # add_messages handles AIMessage, HumanMessage, ToolMessage merging
```

### The `add_messages` Reducer

This is the most important built-in reducer. It handles LangChain message objects intelligently:

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

# When a node returns {"messages": [new_message]},
# add_messages:
#   - Appends new messages
#   - Updates existing messages if same id (for tool call result injection)
#   - Removes messages with RemoveMessage(id=...)
```

### MessagesState Shorthand

LangGraph provides a pre-built state class for message-centric agents:

```python
from langgraph.graph import MessagesState

# Equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list, add_messages]

builder = StateGraph(MessagesState)
```

### Updating Only Specific Fields

Nodes return **partial state** — only the fields they want to update:

```python
def classifier_node(state: ResearchState) -> dict:
    # Only update these fields; everything else stays the same
    return {
        "status": "classified",
        "iteration": state["iteration"] + 1
    }
    # query, analysis, draft, etc. remain unchanged
```

---

## 4. Nodes: Building Blocks of Behavior

### Function Nodes

The most common node type: a Python function that receives state and returns a partial state update.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def call_llm(state: MessagesState) -> dict:
    """Node that calls the LLM with the current messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def search_web(state: ResearchState) -> dict:
    """Node that performs a web search."""
    query = state["query"]
    results = web_search_api(query)  # your search implementation
    return {
        "search_results": results,
        "status": "searched"
    }
```

### Async Nodes

All nodes can be async for non-blocking I/O:

```python
import asyncio
import httpx

async def fetch_data_node(state: AgentState) -> dict:
    """Async node for network I/O."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/data/{state['query']}")
        data = response.json()
    return {"fetched_data": data, "status": "fetched"}

# Async graphs are invoked with ainvoke
result = await graph.ainvoke(initial_state)
```

### Node with LLM and Structured Output

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class ClassificationOutput(BaseModel):
    intent: str
    confidence: float
    requires_search: bool

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(ClassificationOutput)

def classify_intent(state: AgentState) -> dict:
    """Node that classifies user intent with structured output."""
    prompt = f"""Classify the intent of this query:
Query: {state['query']}

Categories: research, calculation, creative_writing, factual_qa, code_generation
"""
    result: ClassificationOutput = structured_llm.invoke(prompt)
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "needs_search": result.requires_search
    }
```

### Configurable Nodes

Nodes can access run-time configuration via `RunnableConfig`:

```python
from langchain_core.runnables import RunnableConfig

def llm_node(state: AgentState, config: RunnableConfig) -> dict:
    """Node that uses config for model selection."""
    model_name = config.get("configurable", {}).get("model", "gpt-4o-mini")
    temperature = config.get("configurable", {}).get("temperature", 0)

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Pass config at invocation time
result = graph.invoke(
    {"messages": [HumanMessage("Hello")]},
    config={"configurable": {"model": "gpt-4o", "temperature": 0.7}}
)
```

---

## 5. Edges: Routing and Control Flow

### Fixed Edges

A fixed edge always routes from node A to node B:

```python
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)
```

### Conditional Edges

The routing function receives the state and returns a string key that maps to the next node:

```python
from langgraph.graph import StateGraph, END, START

def router(state: AgentState) -> str:
    """Decide next node based on state."""
    intent = state.get("intent", "")
    if intent == "research":
        return "search"
    elif intent == "code_generation":
        return "code_agent"
    elif state.get("needs_human_review"):
        return "human_review"
    else:
        return "generate"

builder.add_conditional_edges(
    "classify",       # source node
    router,           # routing function
    {                 # mapping: return value → node name
        "search": "search_node",
        "code_agent": "code_node",
        "human_review": "review_node",
        "generate": "generate_node",
    }
)
```

### Conditional Entry Points

Start the graph at different nodes based on the initial input:

```python
def entry_router(state: AgentState) -> str:
    """Choose where to start based on the input."""
    if state.get("cached_results"):
        return "generate"   # skip search, already have data
    return "search"

builder.add_conditional_edges(
    START,          # from the entry point
    entry_router,
    {
        "search": "search_node",
        "generate": "generate_node",
    }
)
```

### Cycles: Iteration Until Done

Conditional edges pointing back to earlier nodes create cycles:

```python
def should_continue(state: AgentState) -> str:
    """Continue iterating or stop."""
    if state["iteration"] >= 5:
        return "done"   # safety limit
    if state.get("quality_score", 0) >= 0.9:
        return "done"   # quality threshold met
    if state.get("tests_passed"):
        return "done"
    return "revise"     # loop back

builder.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "revise": "draft_node",   # cycles back!
        "done": END
    }
)
```

### Multiple Edges from One Node (Fan-out)

To send to multiple nodes simultaneously, use `Send`:

```python
from langgraph.types import Send

def parallel_router(state: AgentState) -> list:
    """Fan-out: dispatch the same query to multiple agents in parallel."""
    agents = ["agent_a", "agent_b", "agent_c"]
    return [
        Send(agent, {"query": state["query"], "results": []})
        for agent in agents
    ]

builder.add_conditional_edges("dispatch", parallel_router)
```

---

## 6. Tool Integration: ToolNode and tools_condition

### Defining Tools

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.

    Args:
        query: The search query string.
    """
    # In production: call Tavily, SerpAPI, etc.
    return f"Search results for '{query}': [result1, result2, result3]"

@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output.

    Args:
        code: Valid Python code to execute.
    """
    import io, contextlib
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {})
        return stdout.getvalue() or "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_date() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### ToolNode: Automatic Tool Execution

`ToolNode` automatically executes tool calls from the last AI message:

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState

tools = [search_web, run_python, get_current_date]

# Bind tools to LLM so it knows what's available
llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def call_model(state: MessagesState) -> dict:
    """LLM node: decide whether to call a tool or respond."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ToolNode reads tool_calls from last AI message, executes each, returns ToolMessages
tool_node = ToolNode(tools)

# Build the ReAct agent graph
builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")

# tools_condition: returns "tools" if last message has tool_calls, else END
builder.add_conditional_edges(
    "agent",
    tools_condition,   # built-in condition function
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "agent")   # loop: tool results → agent → decide again

graph = builder.compile()
```

### Handling Tool Errors

```python
from langgraph.prebuilt import ToolNode

# ToolNode has built-in error handling
tool_node = ToolNode(
    tools,
    handle_tool_errors=True,  # catches exceptions, returns error as ToolMessage
)

# Custom error handler
def handle_error(error: Exception, tool_call_id: str) -> str:
    return f"Tool failed: {type(error).__name__}: {str(error)}"

tool_node_custom = ToolNode(tools, handle_tool_errors=handle_error)
```

### Custom Tool Execution Node

For more control than `ToolNode` provides:

```python
import json
from langchain_core.messages import ToolMessage

TOOL_MAP = {tool.name: tool for tool in tools}

def execute_tools(state: MessagesState) -> dict:
    """Custom tool execution with logging and validation."""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name not in TOOL_MAP:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                print(f"Calling {tool_name}({json.dumps(tool_args)})")
                result = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Error in {tool_name}: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    return {"messages": tool_messages}
```

---

## 7. Checkpointing and Persistence

Checkpointing allows graphs to:
- **Pause and resume** (human-in-the-loop, long-running jobs)
- **Recover from failures** (restore to last checkpoint)
- **Branch conversations** (different thread_ids)
- **Time-travel debugging** (replay from any past state)

### MemorySaver (In-Memory)

Best for development and testing. State is lost when the process restarts.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Thread ID identifies the conversation / run
config = {"configurable": {"thread_id": "conversation-42"}}

# First turn
result1 = graph.invoke(
    {"messages": [HumanMessage("My name is Alice")]},
    config=config
)

# Second turn — graph remembers the history from thread-id "conversation-42"
result2 = graph.invoke(
    {"messages": [HumanMessage("What is my name?")]},
    config=config
)
print(result2["messages"][-1].content)
# "Your name is Alice."
```

### SqliteSaver (Persistent Across Restarts)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# File-based SQLite (persists to disk)
with SqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "user-session-001"}}
    result = graph.invoke(initial_state, config=config)
```

### AsyncSqliteSaver

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    async with AsyncSqliteSaver.from_conn_string("./checkpoints.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        result = await graph.ainvoke(initial_state, config=config)
```

### PostgresSaver (Production Scale)

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/langgraph_db"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # creates necessary tables
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(initial_state, config=config)
```

### Async PostgresSaver

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        result = await graph.ainvoke(initial_state, config=config)
```

### Inspecting State History

```python
# Get the current state for a thread
state = graph.get_state(config)
print(state.values)       # current state dict
print(state.next)         # what node(s) will run next
print(state.metadata)     # step number, source node, etc.

# Get all historical states (time travel)
history = list(graph.get_state_history(config))
for checkpoint in history:
    print(f"Step {checkpoint.metadata['step']}: {checkpoint.values}")

# Jump to a specific historical state
past_config = {"configurable": {"thread_id": "...", "checkpoint_id": history[2].config["configurable"]["checkpoint_id"]}}
past_state = graph.get_state(past_config)
```

### Time-Travel: Replay from Past State

```python
# Get a historical state
history = list(graph.get_state_history(config))
third_checkpoint = history[-3]   # 3 steps ago

# Resume from that point (creates a new branch)
result = graph.invoke(None, config=third_checkpoint.config)
```

---

## 8. Human-in-the-Loop

Human-in-the-loop patterns pause the graph before or after a node, waiting for human input or approval before continuing.

### interrupt_before vs interrupt_after

| Mechanism | When It Pauses | Typical Use |
|------------|----------------|-------------|
| `interrupt_before=["node"]` | Before the node runs | Approve *what will happen* (e.g., approve SQL before execution) |
| `interrupt_after=["node"]` | After the node runs | Review *what was produced* (e.g., review generated plan) |

**interrupt_before:** Best when the *action* is risky (delete, pay, execute). Human sees proposed state/action and approves.

**interrupt_after:** Best when the *output* needs review (draft, plan). Human sees the result and can modify or approve.

### interrupt_before: Pause Before a Node

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["execute_action"]   # pause before this node
)

config = {"configurable": {"thread_id": "hitl-demo-1"}}

# Run until the interrupt
result = graph.invoke(
    {"messages": [HumanMessage("Delete all logs older than 7 days")]},
    config=config
)
# Graph pauses at "execute_action" node
# result shows what the agent planned to do

# Inspect the pending state
state = graph.get_state(config)
print("Proposed action:", state.values.get("proposed_action"))
print("Next node:", state.next)   # ('execute_action',)

# Option A: Approve — resume by passing None
approved_result = graph.invoke(None, config=config)

# Option B: Reject — update state and resume
graph.update_state(config, {"approved": False, "feedback": "Too broad, be more specific"})
result_with_feedback = graph.invoke(None, config=config)
```

### interrupt_after: Pause After a Node

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_after=["plan_action"]   # pause after this node to review the plan
)
```

### Injecting Human Messages

```python
from langchain_core.messages import HumanMessage

# The graph is paused, waiting for human review
state = graph.get_state(config)

# Human reviews the output and provides feedback
human_feedback = HumanMessage(content="Looks good, but also include a backup plan.")

# Update state with the human's message
graph.update_state(
    config,
    {"messages": [human_feedback]},
    as_node="human_review"   # pretend this update came from the human_review node
)

# Continue the graph
final_result = graph.invoke(None, config=config)
```

### interrupt() Inside a Node (LangGraph 0.2+)

```python
from langgraph.types import interrupt

def human_review_node(state: AgentState) -> dict:
    """Pause inside a node and wait for human input."""
    # This sends a value to the caller and PAUSES the node execution
    human_response = interrupt(
        value={
            "question": "Please review this plan:",
            "plan": state["proposed_plan"],
            "options": ["approve", "reject", "modify"]
        }
    )
    # When resumed, human_response contains what was passed to graph.invoke(Command(resume=...))

    decision = human_response.get("decision", "approve")
    if decision == "reject":
        return {"status": "rejected", "feedback": human_response.get("feedback", "")}
    elif decision == "modify":
        return {"proposed_plan": human_response.get("modified_plan", state["proposed_plan"])}
    else:
        return {"status": "approved"}

# Resume with human input
from langgraph.types import Command

result = graph.invoke(
    Command(resume={"decision": "approve"}),
    config=config
)
```

### Full Human-in-the-Loop Example

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class ApprovalState(TypedDict):
    messages: Annotated[list, add_messages]
    proposed_query: str
    approved: bool

llm = ChatOpenAI(model="gpt-4o")

def plan_query(state: ApprovalState) -> dict:
    response = llm.invoke(
        state["messages"] + [
            HumanMessage("Generate a SQL query for this request. Show only the query.")
        ]
    )
    return {"proposed_query": response.content, "messages": [response]}

def execute_query(state: ApprovalState) -> dict:
    # Only called if approved
    result = run_sql(state["proposed_query"])
    return {"messages": [AIMessage(f"Query executed. Results:\n{result}")]}

def approval_route(state: ApprovalState) -> str:
    return "execute" if state.get("approved") else END

builder = StateGraph(ApprovalState)
builder.add_node("plan", plan_query)
builder.add_node("execute", execute_query)
builder.set_entry_point("plan")
builder.add_conditional_edges("plan", approval_route, {"execute": "execute", END: END})
builder.add_edge("execute", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_after=["plan"])

config = {"configurable": {"thread_id": "sql-approval-1"}}
result = graph.invoke(
    {"messages": [HumanMessage("Show me all users who signed up in the last 7 days")], "approved": False},
    config=config
)
state = graph.get_state(config)
print("Proposed SQL:", state.values["proposed_query"])

# Human approves
graph.update_state(config, {"approved": True})
final = graph.invoke(None, config=config)
```

---

## 9. Subgraphs and Composition

Subgraphs allow modular, reusable agent components. They are **compiled graphs used as nodes** in a parent graph, enabling hierarchical composition and encapsulation.

### When to Use Subgraphs

| Use Case | Example |
|----------|---------|
| **Reusable logic** | Search pipeline (refine → search → filter) used by multiple parent graphs |
| **Encapsulation** | Hide internal nodes; parent only sees inputs/outputs |
| **Different state schemas** | Subgraph can have its own `TypedDict`; map to/from parent state |
| **Parallel sub-workflows** | Multiple subgraphs (e.g., research + legal check) run as siblings |

### Basic Subgraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# --- Sub-graph state (can be same or different from parent) ---
class SearchState(TypedDict):
    query: str
    results: Annotated[list, add]
    refined_query: str

def web_search(state: SearchState) -> dict:
    results = call_search_api(state["refined_query"] or state["query"])
    return {"results": results}

def refine_query(state: SearchState) -> dict:
    # Use LLM to make the query more specific
    refined = llm.invoke(f"Make this search query more specific: {state['query']}").content
    return {"refined_query": refined}

# Build sub-graph
search_builder = StateGraph(SearchState)
search_builder.add_node("refine", refine_query)
search_builder.add_node("search", web_search)
search_builder.set_entry_point("refine")
search_builder.add_edge("refine", "search")
search_builder.add_edge("search", END)

search_subgraph = search_builder.compile()

# --- Parent graph ---
class ParentState(TypedDict):
    messages: Annotated[list, add]
    query: str
    search_results: Annotated[list, add]
    final_report: str

def write_report(state: ParentState) -> dict:
    context = "\n".join(state["search_results"])
    report = llm.invoke(f"Write a report based on:\n{context}\n\nQuery: {state['query']}").content
    return {"final_report": report}

parent_builder = StateGraph(ParentState)
# The subgraph IS a node in the parent graph
parent_builder.add_node("deep_search", search_subgraph)
parent_builder.add_node("write", write_report)
parent_builder.set_entry_point("deep_search")
parent_builder.add_edge("deep_search", "write")
parent_builder.add_edge("write", END)

parent_graph = parent_builder.compile()
```

### State Transformation Between Parent and Subgraph

When parent and subgraph states differ, use a transformation function:

```python
def transform_for_search(state: ParentState) -> SearchState:
    """Transform parent state to subgraph input."""
    return {"query": state["query"], "results": [], "refined_query": ""}

def transform_from_search(state: SearchState) -> dict:
    """Transform subgraph output back to parent state fields."""
    return {"search_results": state["results"]}

# Wrap subgraph with transformation
from langgraph.graph import Graph

def search_node_wrapper(state: ParentState) -> dict:
    sub_input = transform_for_search(state)
    sub_result = search_subgraph.invoke(sub_input)
    return transform_from_search(sub_result)

parent_builder.add_node("deep_search", search_node_wrapper)
```

### Advanced: Nested Subgraphs with Branching

Subgraphs can contain conditional logic and even call other subgraphs:

```python
# Level 1: Simple search subgraph
search_subgraph = build_search_subgraph()

# Level 2: Research subgraph (uses search + optional summarizer)
def research_node(state):
    search_result = search_subgraph.invoke(state)
    if state.get("need_summary"):
        return summarizer_subgraph.invoke(search_result)
    return search_result

# Level 3: Parent orchestrates research + review
parent_builder.add_node("research", research_node)
parent_builder.add_node("human_review", human_review_node)
```

### Subgraph Checkpointing

When the parent graph uses a checkpointer, subgraph state is **not** checkpointed separately by default—the parent's checkpoint includes the full state. For long-running subgraphs, consider checkpointing the subgraph itself if you need to resume from within it.

---

## 10. Streaming

LangGraph supports multiple streaming modes for different use cases.

### Stream Modes Overview

| Mode | What You Get | Use Case |
|------|-------------|----------|
| `"values"` | Full state after each node | Debug, UI updates |
| `"updates"` | Only the delta (changed fields) | Efficient monitoring |
| `"messages"` | LLM tokens as they stream | Real-time chat UI |
| `"debug"` | All internal events | Deep debugging |

### Streaming Values

```python
# Synchronous
for state in graph.stream(initial_input, config=config, stream_mode="values"):
    print("State:", state)

# Async
async for state in graph.astream(initial_input, config=config, stream_mode="values"):
    print("State:", state)
```

### Streaming Updates (Deltas Only)

```python
for update in graph.stream(initial_input, stream_mode="updates"):
    # update is {"node_name": {partial_state_dict}}
    for node_name, node_output in update.items():
        print(f"Node '{node_name}' returned: {node_output}")
```

### Streaming Messages (Token-Level)

```python
from langchain_core.messages import AIMessageChunk

async for event in graph.astream_events(initial_input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"\n[Calling tool: {event['name']}]")
    elif event["event"] == "on_tool_end":
        print(f"[Tool result: {event['data']['output']}]")
```

### astream_events: Full Event Stream

```python
async for event in graph.astream_events(initial_input, version="v2"):
    event_type = event["event"]
    event_name = event.get("name", "")
    event_data = event.get("data", {})

    if event_type == "on_chain_start":
        print(f"Graph started: {event_name}")

    elif event_type == "on_chain_end":
        print(f"Graph finished: {event_name}")

    elif event_type == "on_chat_model_start":
        print(f"\nLLM started thinking...")

    elif event_type == "on_chat_model_stream":
        # Stream individual tokens
        chunk = event_data.get("chunk")
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    elif event_type == "on_chat_model_end":
        print("\n[LLM finished]")

    elif event_type == "on_tool_start":
        print(f"\n[Tool: {event_name} | Args: {event_data.get('input')}]")

    elif event_type == "on_tool_end":
        print(f"[Tool result: {event_data.get('output')}]")
```

### Multiple Streaming Modes Simultaneously

```python
async for chunk in graph.astream(
    initial_input,
    stream_mode=["updates", "messages"]   # both at once
):
    mode, data = chunk   # tuple of (mode, data)
    if mode == "updates":
        print(f"Node update: {data}")
    elif mode == "messages":
        msg_chunk, metadata = data
        if hasattr(msg_chunk, "content") and msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)
```

---

## 11. Prebuilt Agents: create_react_agent

LangGraph provides `create_react_agent` for quickly building ReAct-style agents (Reason + Act).

### Basic ReAct Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o")

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, Partly Cloudy, Humidity: 65%"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Top results for '{query}': [Article 1], [Article 2], [Article 3]"

tools = [get_weather, calculate, search_web]

# One line to create a full ReAct agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant. Use tools to answer questions."
)

# Use it
result = agent.invoke({
    "messages": [HumanMessage("What's the weather in Tokyo and how does that compare to San Francisco?")]
})
print(result["messages"][-1].content)
```

### ReAct Agent with Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    state_modifier="You are a helpful assistant with memory."
)

config = {"configurable": {"thread_id": "user-123"}}

# First message
r1 = agent.invoke({"messages": [HumanMessage("My name is Bob")]}, config=config)
print(r1["messages"][-1].content)

# Second message — agent remembers "Bob"
r2 = agent.invoke({"messages": [HumanMessage("What's my name?")]}, config=config)
print(r2["messages"][-1].content)   # "Your name is Bob."
```

### ReAct Agent with Interrupt

```python
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    interrupt_before=["tools"]   # pause before executing any tool
)

config = {"configurable": {"thread_id": "safe-agent-1"}}
result = agent.invoke({"messages": [HumanMessage("Search for my private files")]}, config=config)

state = agent.get_state(config)
print("Tool about to be called:", state.values["messages"][-1].tool_calls)

# Approve
agent.invoke(None, config=config)
```

### Custom State in create_react_agent

```python
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    premium: bool

def state_modifier(state: CustomState) -> str:
    tier = "Premium" if state.get("premium") else "Free"
    return f"You are a helpful assistant. User tier: {tier}."

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_schema=CustomState,
    state_modifier=state_modifier
)
```

---

## 12. Multi-Agent Patterns

### Pattern 1: Supervisor Architecture

A supervisor LLM routes tasks to specialist agents and decides when the overall task is complete.

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel

llm = ChatOpenAI(model="gpt-4o")

# ─── Shared State ───────────────────────────────────────────────
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool

# ─── Routing Schema ─────────────────────────────────────────────
class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "coder", "FINISH"]
    reasoning: str

# ─── Supervisor Node ────────────────────────────────────────────
structured_llm = llm.with_structured_output(RouteDecision)

SYSTEM_PROMPT = """You are a supervisor managing a team of agents:
- researcher: Searches for information and facts
- writer: Writes documents, articles, summaries
- coder: Writes and reviews code

Given the conversation, decide which agent should act next.
When the task is complete, respond with FINISH."""

def supervisor(state: SupervisorState) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[{"role": m.type, "content": m.content} for m in state["messages"]]
    ]
    decision: RouteDecision = structured_llm.invoke(messages)
    return {
        "next_agent": decision.next,
        "task_complete": decision.next == "FINISH"
    }

# ─── Specialist Agents ──────────────────────────────────────────
def researcher(state: SupervisorState) -> dict:
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"Research findings for '{query}': [findings...]"

    agent_llm = ChatOpenAI(model="gpt-4o").bind_tools([search])
    response = agent_llm.invoke(
        state["messages"] + [HumanMessage("Research this topic thoroughly.")]
    )
    return {"messages": [AIMessage(content=f"[Researcher]: {response.content}")]}

def writer(state: SupervisorState) -> dict:
    response = llm.invoke(
        state["messages"] + [HumanMessage("Write a polished response based on the research.")]
    )
    return {"messages": [AIMessage(content=f"[Writer]: {response.content}")]}

def coder(state: SupervisorState) -> dict:
    response = llm.invoke(
        state["messages"] + [HumanMessage("Write clean, working code for this task.")]
    )
    return {"messages": [AIMessage(content=f"[Coder]: {response.content}")]}

# ─── Build Graph ─────────────────────────────────────────────────
def route_to_agent(state: SupervisorState) -> str:
    agent = state["next_agent"]
    if agent == "FINISH" or state["task_complete"]:
        return END
    return agent

builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("coder", coder)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {"researcher": "researcher", "writer": "writer", "coder": "coder", END: END}
)
for agent in ["researcher", "writer", "coder"]:
    builder.add_edge(agent, "supervisor")   # always return to supervisor

supervisor_graph = builder.compile()

# Run
result = supervisor_graph.invoke({
    "messages": [HumanMessage("Write a blog post about HNSW vector indexes")],
    "next_agent": "",
    "task_complete": False
})
print(result["messages"][-1].content)
```

### Pattern 2: Hierarchical Agents

Top-level supervisor delegates to sub-supervisors, each managing their own team:

```python
# Sub-graph: Research team (search_agent + summarize_agent)
research_team_graph = build_research_team()   # returns compiled subgraph

# Sub-graph: Writing team (draft_agent + edit_agent + format_agent)
writing_team_graph = build_writing_team()

# Top-level graph
class TopLevelState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    research_output: str
    final_document: str

top_builder = StateGraph(TopLevelState)
top_builder.add_node("research_team", research_team_graph)   # subgraph as node
top_builder.add_node("writing_team", writing_team_graph)
top_builder.set_entry_point("research_team")
top_builder.add_edge("research_team", "writing_team")
top_builder.add_edge("writing_team", END)

top_graph = top_builder.compile()
```

### Pattern 3: Agent Handoffs

Agents explicitly hand off control to each other using `Command`:

```python
from langgraph.types import Command

def triage_agent(state: MessagesState) -> Command:
    """Triage agent that routes to specialist."""
    intent = classify_intent(state["messages"][-1].content)

    if intent == "billing":
        return Command(
            goto="billing_agent",
            update={"messages": [AIMessage("Transferring you to billing...")]}
        )
    elif intent == "technical":
        return Command(
            goto="tech_support_agent",
            update={"messages": [AIMessage("Transferring you to tech support...")]}
        )
    else:
        return Command(
            update={"messages": [llm.invoke(state["messages"])]}
        )
```

### Pattern 4: Parallel Agent Execution (MapReduce)

```python
from langgraph.types import Send
from typing import TypedDict, Annotated
from operator import add

class MapReduceState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], add]   # each parallel worker adds to this
    final_summary: str

def dispatch_summaries(state: MapReduceState) -> list[Send]:
    """Fan-out: send each document to a summarizer in parallel."""
    return [
        Send("summarize_document", {"doc": doc, "doc_index": i})
        for i, doc in enumerate(state["documents"])
    ]

class DocState(TypedDict):
    doc: str
    doc_index: int
    summary: str

def summarize_document(state: DocState) -> dict:
    """Worker: summarize a single document."""
    summary = llm.invoke(f"Summarize this in 2 sentences:\n{state['doc']}").content
    # The summary is added to the parent's "summaries" list via the reducer
    return {"summaries": [summary]}

def combine_summaries(state: MapReduceState) -> dict:
    """Reducer: combine all summaries into one."""
    all_summaries = "\n\n".join(
        f"Document {i+1}: {s}"
        for i, s in enumerate(state["summaries"])
    )
    final = llm.invoke(f"Create one coherent summary from these:\n{all_summaries}").content
    return {"final_summary": final}

builder = StateGraph(MapReduceState)
builder.add_node("summarize_document", summarize_document)
builder.add_node("combine", combine_summaries)
builder.set_entry_point("dispatch")
builder.add_node("dispatch", lambda s: {})  # empty dispatch node
builder.add_conditional_edges("dispatch", dispatch_summaries)
builder.add_edge("summarize_document", "combine")  # all workers feed into combine
builder.add_edge("combine", END)

mapreduce_graph = builder.compile()
```

---

## 13. Complex Workflow Patterns

### Plan-and-Execute

The agent first creates a plan, then executes each step, adapting if needed:

```python
from pydantic import BaseModel
from typing import List

class Plan(BaseModel):
    steps: List[str]
    reasoning: str

class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    objective: str
    plan: List[str]
    current_step_index: int
    step_results: Annotated[list, add]
    final_answer: str

planner_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Plan)

def plan_steps(state: PlanExecuteState) -> dict:
    """Create an execution plan."""
    plan: Plan = planner_llm.invoke(
        f"Create a step-by-step plan to: {state['objective']}\n"
        "Return 3-7 concrete, actionable steps."
    )
    return {"plan": plan.steps, "current_step_index": 0}

def execute_step(state: PlanExecuteState) -> dict:
    """Execute the current step in the plan."""
    step = state["plan"][state["current_step_index"]]
    context = "\n".join(f"Step {i+1} result: {r}" for i, r in enumerate(state["step_results"]))

    result = llm.invoke(
        f"Execute this step: {step}\n"
        f"Previous results:\n{context}\n"
        f"Objective: {state['objective']}"
    ).content

    return {
        "step_results": [f"Step {state['current_step_index']+1}: {result}"],
        "current_step_index": state["current_step_index"] + 1
    }

def should_continue_plan(state: PlanExecuteState) -> str:
    """Check if all steps are complete."""
    if state["current_step_index"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

def synthesize_answer(state: PlanExecuteState) -> dict:
    """Combine all step results into final answer."""
    all_results = "\n".join(state["step_results"])
    final = llm.invoke(
        f"Objective: {state['objective']}\n\nStep results:\n{all_results}\n\nSynthesize a final answer."
    ).content
    return {"final_answer": final}

builder = StateGraph(PlanExecuteState)
builder.add_node("plan", plan_steps)
builder.add_node("execute", execute_step)
builder.add_node("synthesize", synthesize_answer)
builder.set_entry_point("plan")
builder.add_edge("plan", "execute")
builder.add_conditional_edges("execute", should_continue_plan, {"execute": "execute", "synthesize": "synthesize"})
builder.add_edge("synthesize", END)

plan_execute_graph = builder.compile()
```

### Reflection and Self-Correction

The agent generates output, critiques it, and iterates until quality is sufficient:

```python
from pydantic import BaseModel

class Critique(BaseModel):
    score: float          # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    is_acceptable: bool

class ReflectionState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    draft: str
    critique: str
    iteration: int
    final_output: str

critic_llm = ChatOpenAI(model="gpt-4o").with_structured_output(Critique)

def generate_draft(state: ReflectionState) -> dict:
    """Generate or revise the draft."""
    if state.get("critique"):
        prompt = (
            f"Revise your previous draft based on this critique:\n"
            f"Draft: {state['draft']}\n"
            f"Critique: {state['critique']}\n"
            f"Original task: {state['task']}"
        )
    else:
        prompt = f"Complete this task: {state['task']}"

    draft = llm.invoke(prompt).content
    return {"draft": draft, "iteration": state.get("iteration", 0) + 1}

def critique_draft(state: ReflectionState) -> dict:
    """Critique the current draft."""
    critique: Critique = critic_llm.invoke(
        f"Task: {state['task']}\n\nDraft to critique:\n{state['draft']}\n\n"
        f"Evaluate quality, accuracy, completeness. Give a score from 0.0 to 1.0."
    )
    return {
        "critique": f"Score: {critique.score}\nIssues: {', '.join(critique.issues)}\nSuggestions: {', '.join(critique.suggestions)}",
        "quality_score": critique.score,
        "acceptable": critique.is_acceptable
    }

def should_revise(state: ReflectionState) -> str:
    """Decide whether to revise or finalize."""
    max_iterations = 3
    if state.get("acceptable") or state.get("iteration", 0) >= max_iterations:
        return "finalize"
    return "revise"

def finalize(state: ReflectionState) -> dict:
    return {"final_output": state["draft"]}

builder = StateGraph(ReflectionState)
builder.add_node("generate", generate_draft)
builder.add_node("critique", critique_draft)
builder.add_node("finalize", finalize)
builder.set_entry_point("generate")
builder.add_edge("generate", "critique")
builder.add_conditional_edges("critique", should_revise, {"revise": "generate", "finalize": "finalize"})
builder.add_edge("finalize", END)

reflection_graph = builder.compile()
```

---

## 14. Memory Across Conversations

### Short-Term Memory: Thread-Based Checkpoints

Short-term memory is automatic with checkpointing — the full message history is preserved per `thread_id`.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

memory = MemorySaver()
agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=tools, checkpointer=memory)

# Different thread_ids = separate conversations
alice_config = {"configurable": {"thread_id": "alice-session"}}
bob_config = {"configurable": {"thread_id": "bob-session"}}

agent.invoke({"messages": [HumanMessage("My name is Alice")]}, config=alice_config)
agent.invoke({"messages": [HumanMessage("My name is Bob")]}, config=bob_config)

r = agent.invoke({"messages": [HumanMessage("What's my name?")]}, config=alice_config)
# "Your name is Alice."  ← separate memory from Bob's thread
```

### Long-Term Memory: Cross-Thread Persistence

Long-term memory persists facts across different conversation threads using a shared memory store.

```python
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import json
import uuid

# In production: use PostgresStore or RedisStore instead
store = InMemoryStore()

@tool
def save_memory(user_id: str, fact: str) -> str:
    """Save an important fact about the user to long-term memory."""
    namespace = ("user_facts", user_id)
    store.put(namespace, str(uuid.uuid4()), {"fact": fact})
    return f"Saved: {fact}"

@tool
def recall_memories(user_id: str, topic: str) -> str:
    """Recall facts about a user related to a topic."""
    namespace = ("user_facts", user_id)
    items = store.search(namespace, query=topic, limit=5)
    if not items:
        return "No memories found."
    return "\n".join(f"- {item.value['fact']}" for item in items)

agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[save_memory, recall_memories],
    checkpointer=MemorySaver(),
    store=store,
    state_modifier=(
        "You are a helpful assistant with long-term memory. "
        "When the user shares important information (name, preferences, goals), "
        "save it with save_memory. Always recall relevant memories before responding."
    )
)

# Session 1: user shares info
config1 = {"configurable": {"thread_id": "alice-s1", "user_id": "alice-123"}}
agent.invoke({"messages": [HumanMessage("I prefer Python over JavaScript and I'm learning ML")]}, config=config1)

# Session 2 (new thread): agent remembers
config2 = {"configurable": {"thread_id": "alice-s2", "user_id": "alice-123"}}
result = agent.invoke({"messages": [HumanMessage("What should I learn next?")]}, config=config2)
# Agent recalls Alice's Python preference and ML interest
```

### Semantic Memory with Vector Store

```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

# Semantic search over memories
vector_store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)

# Store with embeddings
namespace = ("memories", "user-alice")
vector_store.put(namespace, "mem-1", {"content": "Alice works at OpenAI as a researcher"})
vector_store.put(namespace, "mem-2", {"content": "Alice's favorite programming language is Python"})
vector_store.put(namespace, "mem-3", {"content": "Alice is learning about transformer architecture"})

# Semantic search
results = vector_store.search(namespace, query="What is Alice's profession?", limit=2)
for r in results:
    print(r.value["content"])
# "Alice works at OpenAI as a researcher"
```

---

## 15. LangGraph Platform and Cloud Deployment

### LangGraph Platform Architecture

LangGraph Platform wraps your compiled graph into a production-ready service with:
- REST API for graph invocation
- Persistence (PostgreSQL-backed)
- Streaming via SSE
- Background task execution
- Cron job scheduling
- SDK for Python/TypeScript clients

### Defining a Deployable Graph

```python
# my_agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o")

def call_model(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

# This is the graph that LangGraph Platform will serve
graph = builder.compile()
```

### langgraph.json Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./my_agent/graph.py:graph"
  },
  "env": ".env"
}
```

### Deployment Commands

```bash
# Install the CLI
pip install langgraph-cli

# Local development server
langgraph dev

# Build Docker image
langgraph build -t my-agent:latest

# Deploy to LangGraph Cloud (requires API key)
langgraph deploy
```

### Using the SDK to Call a Deployed Graph

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:8123")

# Create a thread (conversation)
thread = await client.threads.create()

# Stream results
async for chunk in client.runs.stream(
    thread["thread_id"],
    "my_agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]},
    stream_mode="messages"
):
    if chunk.event == "messages/partial":
        for msg in chunk.data:
            if "content" in msg:
                print(msg["content"], end="", flush=True)
```

---

## 16. Debugging with LangSmith

### Setup

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_...
export LANGCHAIN_PROJECT="my-agent-project"
```

### Every Invocation is Automatically Traced

```python
result = graph.invoke(initial_state, config={"configurable": {"thread_id": "debug-1"}})
# → Opens trace at https://smith.langchain.com with full step-by-step breakdown
```

### Adding Custom Metadata to Traces

```python
from langsmith import traceable

@traceable(name="Custom Search", metadata={"source": "web"})
def search_with_tracing(query: str) -> str:
    result = actual_search(query)
    return result
```

### Evaluating Agent Performance with LangSmith

```python
from langsmith import Client
from langsmith.evaluation import evaluate

ls_client = Client()

def run_agent(example_input):
    result = graph.invoke({"messages": [HumanMessage(example_input["question"])]})
    return {"output": result["messages"][-1].content}

def correctness_evaluator(run, example):
    expected = example.outputs["expected_answer"]
    actual = run.outputs["output"]
    score = 1.0 if expected.lower() in actual.lower() else 0.0
    return {"key": "correctness", "score": score}

results = evaluate(
    run_agent,
    data="my-agent-dataset",
    evaluators=[correctness_evaluator],
    experiment_prefix="agent-v2"
)
print(results.to_pandas())
```

---

## 17. Full Code Examples

### Example 1: Complete ReAct Agent with Tools

```python
import os
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import json

os.environ["OPENAI_API_KEY"] = "sk-..."

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get real-time weather for any city.

    Args:
        city: Name of the city.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.
    """
    # Simulate API call
    temp = 22 if unit == "celsius" else 72
    return json.dumps({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": "Partly Cloudy",
        "humidity": 65,
        "wind_speed": "15 km/h"
    })

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information.

    Args:
        query: The topic to search for.
    """
    # In production: use wikipedia-api library
    return f"Wikipedia summary for '{query}': [factual information about {query} would appear here]"

@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like '2 + 2' or 'sqrt(16)'.
    """
    import math
    safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_env["abs"] = abs
    try:
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return str(result)
    except Exception as e:
        return f"Cannot evaluate: {e}"

tools = [get_weather, search_wikipedia, calculate]
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    state_modifier=(
        "You are a helpful assistant with access to weather, Wikipedia, and math tools. "
        "Always use tools to get accurate information. Be concise and factual."
    )
)

def chat(message: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(message)]}, config=config)
    return result["messages"][-1].content

# Streaming version
async def stream_chat(message: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    async for event in agent.astream_events(
        {"messages": [HumanMessage(message)]},
        config=config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)
    print()

if __name__ == "__main__":
    print(chat("What's the weather in Paris and Tokyo? Which is warmer?"))
    print(chat("How much warmer? Calculate the difference.", thread_id="default"))
```

### Example 2: Multi-Agent Supervisor System

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from pydantic import BaseModel
import functools

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ─── Tools ──────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    return f"Search results: [Top 5 results for '{query}']"

@tool
def read_file(filepath: str) -> str:
    """Read a local file's content."""
    try:
        with open(filepath) as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"

@tool
def write_code(specification: str, language: str = "python") -> str:
    """Generate code from a specification."""
    response = llm.invoke(f"Write {language} code for: {specification}")
    return response.content

@tool
def run_tests(code: str) -> str:
    """Run basic syntax checks on code."""
    try:
        compile(code, "<string>", "exec")
        return "Syntax check: PASSED"
    except SyntaxError as e:
        return f"Syntax error: {e}"

# ─── State ──────────────────────────────────────────────────────
class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    current_agent: str
    iterations: int
    max_iterations: int

# ─── Routing ────────────────────────────────────────────────────
class RouterOutput(BaseModel):
    next: Literal["researcher", "coder", "writer", "FINISH"]
    reasoning: str

router_llm = llm.with_structured_output(RouterOutput)

SUPERVISOR_PROMPT = """You are coordinating a software development team:
- researcher: Finds information, best practices, and requirements
- coder: Writes and tests code
- writer: Writes documentation and explanations

Current task: {task}
Iterations used: {iterations}/{max_iterations}

Given the conversation so far, who should act next?
Return FINISH when the task is complete or iterations are exhausted."""

def supervisor(state: TeamState) -> dict:
    prompt = SUPERVISOR_PROMPT.format(
        task=state["task"],
        iterations=state["iterations"],
        max_iterations=state["max_iterations"]
    )
    messages_for_router = [
        {"role": "system", "content": prompt},
        *[{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
          for m in state["messages"][-6:]]   # last 6 messages for context
    ]
    decision = router_llm.invoke(messages_for_router)
    return {
        "current_agent": decision.next,
        "iterations": state["iterations"] + 1
    }

def make_agent_node(name: str, agent_tools: list, system_prompt: str):
    """Factory function for agent nodes."""
    agent_llm = ChatOpenAI(model="gpt-4o").bind_tools(agent_tools)

    def agent_node(state: TeamState) -> dict:
        messages = [{"role": "system", "content": system_prompt}] + [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state["messages"][-4:]
        ]
        response = agent_llm.invoke(messages)
        return {"messages": [AIMessage(content=f"[{name}]: {response.content}")]}

    agent_node.__name__ = name
    return agent_node

researcher_node = make_agent_node(
    "Researcher",
    [web_search, read_file],
    "You are a researcher. Find accurate information and provide detailed findings."
)

coder_node = make_agent_node(
    "Coder",
    [write_code, run_tests],
    "You are an expert coder. Write clean, tested, well-structured code."
)

writer_node = make_agent_node(
    "Writer",
    [],
    "You are a technical writer. Write clear, comprehensive documentation and explanations."
)

# ─── Build Graph ─────────────────────────────────────────────────
def route_supervisor(state: TeamState) -> str:
    agent = state["current_agent"]
    if agent == "FINISH" or state["iterations"] >= state["max_iterations"]:
        return END
    return agent

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_node)
builder.add_node("coder", coder_node)
builder.add_node("writer", writer_node)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"researcher": "researcher", "coder": "coder", "writer": "writer", END: END}
)
for agent in ["researcher", "coder", "writer"]:
    builder.add_edge(agent, "supervisor")

memory = MemorySaver()
team_graph = builder.compile(checkpointer=memory)

# Run
result = team_graph.invoke({
    "messages": [HumanMessage("Build a Python function to parse CSV files and return summary statistics")],
    "task": "Build a CSV parser with statistics",
    "current_agent": "",
    "iterations": 0,
    "max_iterations": 8
})
for msg in result["messages"]:
    print(f"\n{msg.content}")
```

### Example 3: Human-in-the-Loop Workflow

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

llm = ChatOpenAI(model="gpt-4o")

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    plan: list
    approved_steps: list
    results: Annotated[list, add_messages]
    current_step: int

def create_plan(state: WorkflowState) -> dict:
    """Step 1: Create an execution plan."""
    response = llm.invoke(
        f"Create a numbered list of 3-5 steps to accomplish: {state['task']}\n"
        "Be specific and actionable."
    )
    # Parse steps from response
    steps = [line.strip() for line in response.content.split("\n") if line.strip() and line[0].isdigit()]
    return {
        "plan": steps,
        "messages": [AIMessage(f"Plan created:\n{response.content}")],
        "current_step": 0
    }

def review_plan(state: WorkflowState) -> dict:
    """Step 2: Human reviews and approves the plan."""
    plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))

    # interrupt() pauses execution and returns value to the caller
    human_decision = interrupt({
        "message": "Please review the execution plan:",
        "plan": plan_text,
        "options": ["approve", "reject", "modify"],
        "current_task": state["task"]
    })

    decision = human_decision.get("decision", "approve")
    if decision == "reject":
        return {"messages": [AIMessage("Plan rejected. Task cancelled.")], "plan": []}
    elif decision == "modify":
        modified_plan = human_decision.get("modified_plan", state["plan"])
        return {
            "plan": modified_plan if isinstance(modified_plan, list) else state["plan"],
            "messages": [AIMessage("Plan modified per feedback.")]
        }
    return {"messages": [AIMessage("Plan approved! Beginning execution...")]}

def execute_step(state: WorkflowState) -> dict:
    """Step 3: Execute each approved step."""
    step_idx = state["current_step"]
    if step_idx >= len(state["plan"]):
        return {}

    step = state["plan"][step_idx]
    result = llm.invoke(
        f"Execute this step: {step}\nTask context: {state['task']}"
    ).content

    return {
        "results": [AIMessage(f"Step {step_idx+1} result: {result}")],
        "current_step": step_idx + 1,
        "messages": [AIMessage(f"Completed step {step_idx+1}: {step[:50]}...")]
    }

def finalize(state: WorkflowState) -> dict:
    """Step 4: Compile final report."""
    results_text = "\n".join(m.content for m in state.get("results", []))
    summary = llm.invoke(
        f"Task: {state['task']}\n\nResults:\n{results_text}\n\nWrite a final summary."
    ).content
    return {"messages": [AIMessage(f"FINAL REPORT:\n{summary}")]}

def route_after_plan(state: WorkflowState) -> str:
    if not state["plan"]:   # rejected
        return END
    if state["current_step"] >= len(state["plan"]):
        return "finalize"
    return "execute"

builder = StateGraph(WorkflowState)
builder.add_node("create_plan", create_plan)
builder.add_node("review_plan", review_plan)
builder.add_node("execute", execute_step)
builder.add_node("finalize", finalize)

builder.set_entry_point("create_plan")
builder.add_edge("create_plan", "review_plan")
builder.add_conditional_edges("review_plan", route_after_plan, {"execute": "execute", "finalize": "finalize", END: END})
builder.add_conditional_edges("execute", route_after_plan, {"execute": "execute", "finalize": "finalize", END: END})
builder.add_edge("finalize", END)

memory = MemorySaver()
workflow = builder.compile(checkpointer=memory)

# Run the workflow
import asyncio

async def run_hitl_workflow():
    config = {"configurable": {"thread_id": "hitl-workflow-1"}}
    initial_state = {
        "messages": [],
        "task": "Set up a Python project with tests and CI/CD",
        "plan": [],
        "approved_steps": [],
        "results": [],
        "current_step": 0
    }

    # Run until human interrupt
    print("Creating plan...")
    result = await workflow.ainvoke(initial_state, config=config)

    state = workflow.get_state(config)
    if state.next:  # paused at interrupt
        print(f"\nWorkflow paused. Pending: {state.next}")
        pending_value = state.interrupts[0].value if state.interrupts else {}
        print("Plan for review:", pending_value.get("plan", ""))

        # Simulate human approval
        decision = Command(resume={"decision": "approve"})
        final_result = await workflow.ainvoke(decision, config=config)
        print("\nFinal messages:")
        for msg in final_result["messages"]:
            print(msg.content)

asyncio.run(run_hitl_workflow())
```

### Example 4: Long-Term Memory Agent

```python
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
import uuid
import json

# ─── Memory Store with Semantic Search ──────────────────────────
store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)

# ─── Memory Tools ───────────────────────────────────────────────
@tool
def save_user_memory(ctx: RunnableConfig, memory: str, category: str = "general") -> str:
    """Save an important fact or preference about the user.

    Args:
        memory: The fact or preference to remember.
        category: Category like 'preference', 'fact', 'goal', 'context'.
    """
    user_id = ctx.get("configurable", {}).get("user_id", "default")
    namespace = ("memories", user_id)
    mem_id = str(uuid.uuid4())[:8]
    store.put(namespace, mem_id, {
        "content": memory,
        "category": category,
        "id": mem_id
    })
    return f"Saved memory [{category}]: {memory}"

@tool
def search_user_memories(ctx: RunnableConfig, query: str, limit: int = 5) -> str:
    """Search for relevant memories about the user.

    Args:
        query: What to search for in memories.
        limit: Maximum number of memories to return.
    """
    user_id = ctx.get("configurable", {}).get("user_id", "default")
    namespace = ("memories", user_id)
    results = store.search(namespace, query=query, limit=limit)
    if not results:
        return "No relevant memories found."
    memories = [f"- [{r.value['category']}] {r.value['content']}" for r in results]
    return "Relevant memories:\n" + "\n".join(memories)

@tool
def list_all_memories(ctx: RunnableConfig) -> str:
    """List all memories saved about the user."""
    user_id = ctx.get("configurable", {}).get("user_id", "default")
    namespace = ("memories", user_id)
    items = store.list(namespace)
    if not items:
        return "No memories saved yet."
    by_category = {}
    for item in items:
        cat = item.value.get("category", "general")
        by_category.setdefault(cat, []).append(item.value["content"])
    result = []
    for cat, mems in by_category.items():
        result.append(f"\n{cat.upper()}:")
        result.extend(f"  - {m}" for m in mems)
    return "\n".join(result)

@tool
def forget_memory(ctx: RunnableConfig, memory_id: str) -> str:
    """Delete a specific memory by ID.

    Args:
        memory_id: The ID of the memory to delete.
    """
    user_id = ctx.get("configurable", {}).get("user_id", "default")
    namespace = ("memories", user_id)
    store.delete(namespace, memory_id)
    return f"Deleted memory: {memory_id}"

# ─── Build Agent ─────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)
short_term_memory = MemorySaver()

memory_agent = create_react_agent(
    model=llm,
    tools=[save_user_memory, search_user_memories, list_all_memories, forget_memory],
    checkpointer=short_term_memory,
    store=store,
    state_modifier="""You are a personalized AI assistant with both short-term and long-term memory.

SHORT-TERM MEMORY: The full conversation history in this thread.
LONG-TERM MEMORY: Facts you've saved about the user using memory tools.

INSTRUCTIONS:
1. When the user shares important info (name, preferences, goals, background), IMMEDIATELY save it with save_user_memory
2. Before responding to requests, search_user_memories for relevant context
3. Use this context to personalize your responses
4. Be natural — don't announce every memory operation unless relevant
"""
)

def run_memory_agent(message: str, user_id: str = "alice", session: str = "1"):
    """Run the memory agent and return the response."""
    config = {
        "configurable": {
            "thread_id": f"{user_id}-session-{session}",
            "user_id": user_id
        }
    }
    result = memory_agent.invoke(
        {"messages": [HumanMessage(message)]},
        config=config
    )
    return result["messages"][-1].content

# Demo
print("=== Session 1 ===")
print(run_memory_agent("Hi! I'm Alice, a ML researcher at Stanford.", "alice", "1"))
print(run_memory_agent("I prefer PyTorch over TensorFlow and I'm working on vision transformers.", "alice", "1"))

print("\n=== Session 2 (New Thread — Long-term Memory Persists) ===")
print(run_memory_agent("Can you recommend a good paper for my current research?", "alice", "2"))
# → Recalls Alice is ML researcher working on vision transformers, recommends ViT papers
```

---

## 18. Best Practices

### State Design

```python
# Good: Lean state with clear semantics
class WellDesignedState(TypedDict):
    messages: Annotated[list, add_messages]   # always use add_messages for chat
    task: str                                  # immutable task description
    iteration: int                             # track loops
    status: Literal["searching", "analyzing", "done"]  # clear status enum

# Bad: Bloated state that mixes concerns
class BadState(TypedDict):
    everything: dict   # too vague
    temp_data: Any     # unclear lifecycle
    # Missing reducers on list fields → gets overwritten!
    results: list      # should be Annotated[list, add] for accumulation
```

### Node Design

```python
# Good: Single responsibility, returns only changed fields
def search_node(state: ResearchState) -> dict:
    results = do_search(state["query"])
    return {"search_results": results, "status": "searched"}  # only relevant fields

# Bad: Node doing too much
def kitchen_sink_node(state):
    # search + analyze + write + review all in one function
    # → hard to test, debug, and route around
    ...
```

### Loop Safety

```python
# Always bound your loops
MAX_ITERATIONS = 5
MAX_TOOL_CALLS = 20

def should_continue(state: AgentState) -> str:
    if state["iteration"] >= MAX_ITERATIONS:
        return END   # safety exit
    if state.get("done"):
        return END
    return "continue"
```

### Error Handling

```python
def robust_node(state: AgentState) -> dict:
    try:
        result = risky_operation(state["input"])
        return {"result": result, "error": None}
    except Exception as e:
        return {"error": str(e), "result": None}

def route_after_risky(state: AgentState) -> str:
    if state.get("error"):
        return "handle_error"   # dedicated error recovery node
    return "continue"

def handle_error(state: AgentState) -> dict:
    # Retry, fallback, or gracefully stop
    return {"messages": [AIMessage(f"I encountered an error: {state['error']}. Let me try a different approach.")]}
```

### Testing Individual Nodes

```python
# Test nodes in isolation before composing them
def test_classify_node():
    test_state = {
        "messages": [HumanMessage("I want a refund")],
        "intent": "",
        "confidence": 0.0
    }
    result = classify_node(test_state)
    assert result["intent"] == "refund"
    assert result["confidence"] > 0.8
```

### Summary Table

| Concept | Key Point |
|---------|-----------|
| `StateGraph` | Graph with typed state flowing through all nodes |
| `Annotated[list, add_messages]` | Use for message lists — handles merging |
| `Annotated[list, add]` | Use for accumulating results |
| `add_conditional_edges` | Route to different nodes based on state |
| `MemorySaver` | In-memory checkpointing (dev/test) |
| `SqliteSaver` / `PostgresSaver` | Persistent checkpointing (production) |
| `interrupt_before` | Pause before a node for human review |
| `interrupt()` | Pause inside a node, wait for resume value |
| `ToolNode` + `tools_condition` | Auto-execute tool calls from LLM |
| `create_react_agent` | One-line ReAct agent with tools |
| `Send` | Fan-out to multiple nodes in parallel |
| Subgraphs | Modular, reusable agent components |
| `InMemoryStore` | Long-term memory across threads |
| `astream_events` | Stream tokens, tool calls, and graph events |

---

## 19. Common Pitfalls

### 1. Forgetting Reducers on Accumulating Fields

**Problem:** List fields without reducers get **overwritten** instead of accumulated.

```python
# Bad: new items replace old
class BadState(TypedDict):
    results: list  # Each node return overwrites the entire list!

# Good: use add or add_messages
class GoodState(TypedDict):
    results: Annotated[list, add]
    messages: Annotated[list, add_messages]
```

### 2. Unbounded Loops

**Problem:** Conditional edges that always route back cause infinite loops.

```python
# Bad: No exit condition
def route(state): return "loop"  # Never ends!

# Good: Always have a terminal path
def route(state):
    if state["iteration"] >= MAX_ITERATIONS or state.get("done"):
        return END
    return "loop"
```

### 3. State Schema Mismatch in Subgraphs

**Problem:** Parent passes state with different field names/types than subgraph expects.

```python
# Parent has "query", subgraph expects "search_query"
def transform_for_search(state: ParentState) -> SearchState:
    return {"search_query": state["query"], "results": [], ...}  # Map correctly
```

### 4. interrupt_before/after Without Checkpointer

**Problem:** Interrupts require a checkpointer; without it, the graph cannot pause and resume.

```python
# Bad: interrupt without checkpointer
graph = builder.compile(interrupt_before=["execute"])  # Won't work properly!

# Good
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"],
)
```

### 5. Tool Errors Swallowing Context

**Problem:** Default `handle_tool_errors=True` returns a generic error message; the agent may not know what failed.

```python
# Better: Custom handler that preserves context
def handle_error(e: Exception, tool_call_id: str) -> str:
    return f"Tool failed: {type(e).__name__}. {str(e)} Please try a different approach."
tool_node = ToolNode(tools, handle_tool_errors=handle_error)
```

---

## 20. Production Considerations

### Scaling and Concurrency

```python
# Use async throughout for non-blocking I/O
result = await graph.ainvoke(initial_state, config=config)

# PostgresSaver/AsyncPostgresSaver for production persistence
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

### Timeout and Circuit Breakers

```python
import asyncio

async def invoke_with_timeout(graph, state, config, timeout_seconds=60):
    try:
        return await asyncio.wait_for(
            graph.ainvoke(state, config=config),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        # Log, notify, return graceful fallback
        return {"messages": [AIMessage("Request timed out. Please try again.")]}
```

### Monitoring Key Metrics

Track per-thread: step count, node duration, tool call success rate, token usage. Use LangSmith or custom trace processors to export to your observability stack (Datadog, Grafana, etc.).

---

**Libraries**: `langgraph`, `langchain-openai`, `langchain-core`, `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres`

**Docs**: https://langchain-ai.github.io/langgraph/  
**Tutorials**: https://langchain-ai.github.io/langgraph/tutorials/  
**LangSmith**: https://smith.langchain.com

### References

| Resource | Description |
|----------|-------------|
| [LangGraph Docs](https://langchain-ai.github.io/langgraph/) | Official documentation |
| [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) | Step-by-step guides |
| [LangGraph Platform](https://langchain-ai.github.io/langgraph/platform/) | Cloud deployment |
| [LangSmith](https://smith.langchain.com) | Tracing and evaluation |
| ReAct (Yao et al., 2022) | Reason + Act paradigm |
| Plan-and-Execute | Hierarchical task decomposition |
