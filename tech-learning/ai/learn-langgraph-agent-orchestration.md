# LangGraph & Agent Orchestration: Complete Guide

## Table of Contents
1. [Introduction to Agent Orchestration](#introduction-to-agent-orchestration)
2. [LangGraph Fundamentals](#langgraph-fundamentals)
3. [State Machines for Agents](#state-machines-for-agents)
4. [Nodes, Edges, and Conditional Routing](#nodes-edges-and-conditional-routing)
5. [Human-in-the-Loop](#human-in-the-loop)
6. [Multi-Agent Orchestration](#multi-agent-orchestration)
7. [Tool Calling and Function Calling](#tool-calling-and-function-calling)
8. [Structured Outputs and MCP](#structured-outputs-and-mcp)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)

---

## Introduction to Agent Orchestration

**Agent orchestration** manages complex, multi-step AI workflows as state machines or graphs. Unlike simple chains, orchestration handles branching, loops, human review, and error recovery.

### Why Not Just Chains?

| Simple Chain | Agent Orchestration (LangGraph) |
|-------------|-------------------------------|
| Linear A → B → C | Branching, cycles, parallel |
| No state | Persistent state across steps |
| No human review | Human-in-the-loop checkpoints |
| Fails on error | Error handling, retry, fallback |
| Single path | Conditional routing |

### When to Use LangGraph

- Multi-step agent with decisions at each step
- Human approval gates
- Complex branching logic (if tool fails → retry or fallback)
- Cycles (iterate until quality threshold met)
- Long-running workflows with persistence

---

## LangGraph Fundamentals

**LangGraph** (by LangChain) models agent logic as a **graph** of nodes (actions) and edges (transitions), with a shared **state** that flows through the graph.

### Installation

```bash
pip install langgraph langchain-openai
```

### Core Concepts

- **State**: TypedDict that flows through the graph
- **Node**: Function that reads/writes state
- **Edge**: Connection between nodes (conditional or fixed)
- **Graph**: Compiled workflow

### Basic Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # Append-only
    next_step: str

def step_a(state: AgentState) -> AgentState:
    return {"messages": ["Step A completed"], "next_step": "b"}

def step_b(state: AgentState) -> AgentState:
    return {"messages": ["Step B completed"], "next_step": "end"}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("step_a", step_a)
graph.add_node("step_b", step_b)
graph.add_edge("step_a", "step_b")
graph.add_edge("step_b", END)
graph.set_entry_point("step_a")

app = graph.compile()
result = app.invoke({"messages": [], "next_step": "a"})
print(result["messages"])
```

---

## State Machines for Agents

### State Design

```python
from typing import TypedDict, Annotated, Optional
from operator import add

class ResearchState(TypedDict):
    query: str
    search_results: Annotated[list, add]
    analysis: str
    draft: str
    feedback: Optional[str]
    iteration: int
    status: str  # "searching", "analyzing", "drafting", "reviewing", "done"
```

### Annotated State (Reducers)

```python
# Annotated[list, add] → appends new items to existing list
# Annotated[int, lambda a, b: a + b] → accumulates
# Without annotation: new value replaces old
```

---

## Nodes, Edges, and Conditional Routing

### Conditional Edges

```python
from langgraph.graph import StateGraph, END

def router(state: AgentState) -> str:
    """Decide next node based on state"""
    if state.get("needs_search"):
        return "search"
    elif state.get("needs_review"):
        return "review"
    else:
        return "generate"

graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("search", search_node)
graph.add_node("review", review_node)
graph.add_node("generate", generate_node)

graph.add_conditional_edges(
    "classify",
    router,
    {
        "search": "search",
        "review": "review",
        "generate": "generate",
    }
)
graph.add_edge("search", "generate")
graph.add_edge("review", END)
graph.add_edge("generate", "review")
graph.set_entry_point("classify")

app = graph.compile()
```

### Cycles (Iterate Until Done)

```python
def should_continue(state):
    if state["iteration"] >= 3 or state["quality_score"] > 0.9:
        return "done"
    return "revise"

graph.add_conditional_edges(
    "evaluate",
    should_continue,
    {"revise": "draft", "done": END}
)
```

---

## Human-in-the-Loop

### Interrupt Before/After

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Compile with interrupt
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["execute_action"]  # Pause before this node
)

# Run until interrupt
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke(input_state, config)
# State is saved; user reviews

# Resume after approval
app.invoke(None, config)  # Continues from checkpoint
```

### Approval Gate Node

```python
def human_approval(state: AgentState) -> AgentState:
    """Node that waits for human input"""
    print(f"Pending action: {state['proposed_action']}")
    # In production: send to UI, wait for callback
    # For now: auto-approve
    return {"approved": True}

graph.add_node("approval", human_approval)
graph.add_conditional_edges(
    "approval",
    lambda s: "execute" if s["approved"] else "revise",
    {"execute": "execute_action", "revise": "plan"}
)
```

---

## Multi-Agent Orchestration

### Supervisor Pattern

One agent routes tasks to specialist agents.

```python
from langchain_openai import ChatOpenAI

class SupervisorState(TypedDict):
    messages: Annotated[list, add]
    next_agent: str
    task: str

def supervisor(state: SupervisorState) -> SupervisorState:
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(f"""
    Task: {state['task']}
    Available agents: researcher, writer, coder
    Which agent should handle this? Just output the agent name.
    """)
    return {"next_agent": response.content.strip().lower()}

def researcher(state):
    return {"messages": [f"Research results for: {state['task']}"]}

def writer(state):
    return {"messages": [f"Written content for: {state['task']}"]}

def coder(state):
    return {"messages": [f"Code for: {state['task']}"]}

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("coder", coder)
graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_agent"],
    {"researcher": "researcher", "writer": "writer", "coder": "coder"}
)
for agent in ["researcher", "writer", "coder"]:
    graph.add_edge(agent, END)

app = graph.compile()
```

### Hierarchical Agents

Agents can spawn sub-agents (sub-graphs).

```python
# Create sub-graph for research
research_graph = StateGraph(ResearchState)
# ... add nodes ...
research_subgraph = research_graph.compile()

# Use as node in parent graph
main_graph.add_node("deep_research", research_subgraph)
```

---

## Tool Calling and Function Calling

### OpenAI Function Calling

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Parse tool call
if response.choices[0].message.tool_calls:
    call = response.choices[0].message.tool_calls[0]
    args = json.loads(call.function.arguments)
    result = get_weather(**args)
    # Send result back
    messages.append(response.choices[0].message)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
    final = client.chat.completions.create(model="gpt-4", messages=messages)
```

### LangChain Tool Calling

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search the web for information"""
    return web_search(query)

@tool
def calculator(expression: str) -> float:
    """Evaluate a math expression"""
    return eval(expression)

llm = ChatOpenAI(model="gpt-4").bind_tools([search, calculator])
response = llm.invoke("What is the population of Tokyo divided by 1000?")
```

### Tool Node in LangGraph

```python
from langgraph.prebuilt import ToolNode

tools = [search, calculator]
tool_node = ToolNode(tools)

def call_model(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_use_tool(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_use_tool, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # Loop back after tool

app = graph.compile()
```

---

## Structured Outputs and MCP

### JSON Mode / Structured Output

```python
from pydantic import BaseModel

class ExtractedInfo(BaseModel):
    name: str
    age: int
    occupation: str

# OpenAI structured output
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract: John is 30, works as engineer"}],
    response_format=ExtractedInfo
)
info = response.choices[0].message.parsed
print(info.name, info.age, info.occupation)
```

### LangChain Structured Output

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(ExtractedInfo)
result = structured_llm.invoke("John is 30, works as engineer")
# result is ExtractedInfo instance
```

### Model Context Protocol (MCP)

MCP standardizes how LLMs connect to tools and data sources:
- **Resources**: Data the LLM can read (files, DB records)
- **Tools**: Functions the LLM can call
- **Prompts**: Reusable prompt templates

```python
# MCP server exposes tools/resources via JSON-RPC
# Client (IDE/agent) discovers and calls them
# Standardized interface: any MCP client ↔ any MCP server
```

---

## Practical Examples

### Example 1: Research Agent with LangGraph

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class ResearchState(TypedDict):
    query: str
    search_results: Annotated[list, add]
    analysis: str
    report: str
    iteration: int

def search_node(state):
    results = web_search(state["query"])
    return {"search_results": results, "iteration": state.get("iteration", 0) + 1}

def analyze_node(state):
    context = "\n".join(state["search_results"])
    analysis = llm.invoke(f"Analyze:\n{context}\nQuery: {state['query']}")
    return {"analysis": analysis.content}

def write_node(state):
    report = llm.invoke(f"Write report:\nAnalysis: {state['analysis']}\nQuery: {state['query']}")
    return {"report": report.content}

def quality_check(state):
    if state["iteration"] < 2:
        return "search"  # Iterate
    return "done"

graph = StateGraph(ResearchState)
graph.add_node("search", search_node)
graph.add_node("analyze", analyze_node)
graph.add_node("write", write_node)
graph.set_entry_point("search")
graph.add_edge("search", "analyze")
graph.add_edge("analyze", "write")
graph.add_conditional_edges("write", quality_check, {"search": "search", "done": END})

app = graph.compile()
result = app.invoke({"query": "Latest advances in AI safety", "search_results": [], "iteration": 0})
```

### Example 2: Customer Support Agent

```python
def classify_intent(state):
    intent = llm.invoke(f"Classify: {state['message']}\nOptions: billing, technical, general")
    return {"intent": intent.content.strip().lower()}

def billing_agent(state):
    return {"response": llm.invoke(f"Billing query: {state['message']}").content}

def technical_agent(state):
    return {"response": llm.invoke(f"Technical query: {state['message']}").content}

graph = StateGraph(SupportState)
graph.add_node("classify", classify_intent)
graph.add_node("billing", billing_agent)
graph.add_node("technical", technical_agent)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", lambda s: s["intent"],
    {"billing": "billing", "technical": "technical", "general": "technical"})
graph.add_edge("billing", END)
graph.add_edge("technical", END)
```

### Example 3: Code Review Pipeline

```python
# Plan → Generate → Test → Review → (loop or done)
def generate_code(state):
    code = llm.invoke(f"Write code: {state['task']}").content
    return {"code": code}

def run_tests(state):
    result = execute_tests(state["code"])
    return {"test_results": result, "tests_passed": result.passed}

def review(state):
    if state["tests_passed"]:
        return {"status": "approved"}
    feedback = llm.invoke(f"Code failed:\n{state['test_results']}\nSuggest fixes.")
    return {"feedback": feedback.content, "status": "needs_fix"}

def route_after_review(state):
    return "done" if state["status"] == "approved" else "generate"

graph.add_conditional_edges("review", route_after_review, {"generate": "generate", "done": END})
```

---

## Best Practices

1. **Design state carefully**: Only include what nodes need
2. **Use reducers** (Annotated) for accumulating data (messages, results)
3. **Limit cycles**: Set max iterations to prevent infinite loops
4. **Checkpoints**: Use MemorySaver for long-running workflows
5. **Human gates**: Add interrupt_before for high-stakes actions
6. **Error handling**: Add fallback edges for tool failures
7. **Test** each node independently before composing

---

## Summary

| Concept | Key Point |
|---------|-----------|
| LangGraph | Graph-based agent orchestration |
| State | TypedDict with reducers flowing through nodes |
| Conditional edges | Route based on state |
| Human-in-the-loop | interrupt_before, checkpoints |
| Multi-agent | Supervisor + specialist pattern |
| Tool calling | Function calling, ToolNode |
| Structured output | Pydantic models, JSON mode |
| MCP | Standard protocol for tools/data |

**Libraries**: `langgraph`, `langchain`, `langchain-openai`, `pydantic`
