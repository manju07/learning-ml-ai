# Agentic AI: From Foundations to Production Systems

## Table of Contents
1. [What is an AI Agent](#1-what-is-an-ai-agent)
2. [Agent Architectures](#2-agent-architectures)
3. [LLM-Based Agents](#3-llm-based-agents)
4. [Tool Use and Function Calling](#4-tool-use-and-function-calling)
5. [Memory Systems](#5-memory-systems)
6. [Planning and Task Decomposition](#6-planning-and-task-decomposition)
7. [Multi-Agent Systems](#7-multi-agent-systems)
8. [Agent Frameworks](#8-agent-frameworks)
9. [Building a Full Agent from Scratch](#9-building-a-full-agent-from-scratch)
10. [Agent Memory with Vector Stores](#10-agent-memory-with-vector-stores)
11. [Code Agents and Computer Use Agents](#11-code-agents-and-computer-use-agents)
12. [Evaluation](#12-evaluation)
13. [Production Considerations](#13-production-considerations)
14. [Human-in-the-Loop Patterns](#14-human-in-the-loop-patterns)
15. [Safety and Alignment](#15-safety-and-alignment)

---

## 1. What is an AI Agent

An **AI agent** is a computational system that perceives its environment, maintains internal state, reasons about goals, and takes actions to achieve them — often in a cyclical, autonomous loop.

The shift from simple LLM "question-answering" to agentic AI is fundamental:

| Dimension | Traditional LLM | Agentic AI |
|-----------|----------------|------------|
| Interaction | Single turn | Multi-turn, autonomous loops |
| Planning | None | Hierarchical goal decomposition |
| Tool use | None | Calls APIs, code execution, search |
| Memory | Prompt window only | In-context + external memory stores |
| State | Stateless | Maintains state across steps |
| Error recovery | None | Retries, replans, reflects |
| Agency | Reactive | Proactive goal pursuit |

### 1.1 The Four Pillars of an Agent

#### Perception
The agent must observe and interpret its environment:
- **Text**: Documents, conversation history, tool outputs
- **Structured data**: JSON, tables, API responses
- **Images/video**: Multimodal agents use vision encoders
- **Sensor data**: Robotics agents receive sensor streams

#### Reasoning
Reasoning is the cognitive core:
- **Logical deduction**: Given premises, derive conclusions
- **Abductive reasoning**: Infer the best explanation for observations
- **Planning**: Sequence of actions to achieve a goal
- **Self-monitoring**: Check if actions align with goals

#### Action
Agents act by invoking tools, APIs, or effectors:
- **Information retrieval**: Search, database query, file read
- **Computation**: Code execution, math, data transformation
- **Communication**: Email, chat, API calls
- **Environment manipulation**: File write, browser control, robot actuation

#### Memory
Memory allows the agent to maintain state:
- **Working memory**: Current context window (~128K tokens)
- **Episodic memory**: Log of past events (vector store)
- **Semantic memory**: General knowledge (RAG system)
- **Procedural memory**: Learned skills / fine-tuned behaviors

### 1.2 The Agent Loop

```
┌──────────────────────────────────────────────────┐
│                    AGENT LOOP                    │
│                                                  │
│  Observe → Reason → Plan → Act → Observe ...    │
│                       ↓                          │
│               Memory / Reflection                │
└──────────────────────────────────────────────────┘
```

Formally, at each timestep `t`:

```
s_t = Perceive(environment_t)
m_t = Memory.retrieve(s_t)
a_t = Policy(s_t, m_t, goal)
o_t = Execute(a_t)
Memory.store(s_t, a_t, o_t)
```

---

## 2. Agent Architectures

### 2.1 Reactive Agents

Reactive agents use stimulus-response mappings — no internal world model or planning:

```python
class ReactiveAgent:
    """Pure reactive agent: stimulus → response"""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, condition, action):
        self.rules.append((condition, action))
    
    def act(self, observation):
        for condition, action in self.rules:
            if condition(observation):
                return action(observation)
        return self.default_action(observation)
    
    def default_action(self, observation):
        return "no_op"

# Example: thermostat agent
thermostat = ReactiveAgent()
thermostat.add_rule(
    condition=lambda obs: obs['temperature'] < 20,
    action=lambda obs: "turn_heater_on"
)
thermostat.add_rule(
    condition=lambda obs: obs['temperature'] > 25,
    action=lambda obs: "turn_heater_off"
)
```

**Strengths**: Fast, predictable, no planning overhead  
**Weaknesses**: Cannot handle novel situations, no goal reasoning

### 2.2 Deliberative (Model-Based) Agents

Deliberative agents maintain an internal world model and plan using it:

```python
class DeliberativeAgent:
    """Model-based deliberative agent"""
    
    def __init__(self, world_model, planner):
        self.world_model = world_model  # Predicts state transitions
        self.planner = planner           # Generates action sequences
        self.beliefs = {}                # Current beliefs about world
        self.goal = None
    
    def update_beliefs(self, observation):
        """Update internal world model from observation"""
        self.beliefs = self.world_model.update(self.beliefs, observation)
    
    def set_goal(self, goal):
        self.goal = goal
    
    def plan(self):
        """Generate a plan to achieve the goal"""
        return self.planner.plan(
            start=self.beliefs,
            goal=self.goal,
            model=self.world_model
        )
    
    def act(self, observation):
        self.update_beliefs(observation)
        plan = self.plan()
        return plan[0] if plan else None  # Execute first step
```

### 2.3 Hybrid Agents (Subsumption Architecture)

Hybrid agents combine reactive (fast) and deliberative (slow) layers:

```
┌─────────────────────────────┐
│   Layer 3: Deliberative     │  Planning, goal reasoning (slow)
│   Layer 2: Model-Based      │  Pattern recognition, prediction (medium)
│   Layer 1: Reactive         │  Reflexes, safety constraints (fast)
└─────────────────────────────┘
       ↑ subsumes lower layers when active
```

```python
class HybridAgent:
    """Three-layer hybrid agent"""
    
    def __init__(self, reactive_layer, model_layer, deliberative_layer):
        self.reactive = reactive_layer
        self.model_based = model_layer
        self.deliberative = deliberative_layer
        self.current_plan = []
    
    def act(self, observation):
        # Layer 1: Safety-critical reactive response
        if self.is_emergency(observation):
            return self.reactive.act(observation)
        
        # Layer 2: Pattern-matched response
        if self.model_based.has_applicable_skill(observation):
            return self.model_based.act(observation)
        
        # Layer 3: Full deliberation
        if not self.current_plan:
            self.current_plan = self.deliberative.plan(observation)
        
        return self.current_plan.pop(0)
    
    def is_emergency(self, obs):
        return obs.get('safety_violation', False)
```

### 2.4 BDI Agents (Belief-Desire-Intention)

BDI is the dominant formal model for rational agents:

- **Beliefs**: What the agent believes about the world (epistemic state)
- **Desires**: Goals the agent wants to achieve (motivational state)
- **Intentions**: Plans the agent is committed to executing (deliberative state)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from enum import Enum

class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"

@dataclass
class Belief:
    predicate: str
    arguments: List[Any]
    confidence: float = 1.0
    timestamp: float = 0.0

@dataclass
class Desire:
    goal: str
    priority: int = 0
    conditions: List[Callable] = field(default_factory=list)

@dataclass
class Intention:
    desire: Desire
    plan: List[str]
    status: GoalStatus = GoalStatus.PENDING
    current_step: int = 0

class BDIAgent:
    """Belief-Desire-Intention Agent"""
    
    def __init__(self):
        self.beliefs: List[Belief] = []
        self.desires: List[Desire] = []
        self.intentions: List[Intention] = []
        self.plan_library: Dict[str, List[str]] = {}
    
    def update_beliefs(self, perception: Dict):
        """Update beliefs based on perception"""
        for pred, val in perception.items():
            # Remove old belief about this predicate
            self.beliefs = [b for b in self.beliefs if b.predicate != pred]
            # Add new belief
            self.beliefs.append(Belief(predicate=pred, arguments=[val]))
    
    def get_belief(self, predicate: str) -> Any:
        for b in self.beliefs:
            if b.predicate == predicate:
                return b.arguments[0]
        return None
    
    def add_desire(self, goal: str, priority: int = 0):
        self.desires.append(Desire(goal=goal, priority=priority))
        self.desires.sort(key=lambda d: -d.priority)
    
    def deliberate(self):
        """Select desires to commit to (form intentions)"""
        for desire in self.desires:
            # Check if desire is already an active intention
            if not any(i.desire.goal == desire.goal for i in self.intentions):
                plan = self.plan_library.get(desire.goal)
                if plan:
                    intention = Intention(desire=desire, plan=plan[:])
                    self.intentions.append(intention)
    
    def execute_intentions(self):
        """Execute active intentions"""
        actions = []
        for intention in self.intentions:
            if intention.status == GoalStatus.PENDING:
                intention.status = GoalStatus.ACTIVE
            if intention.status == GoalStatus.ACTIVE:
                if intention.current_step < len(intention.plan):
                    action = intention.plan[intention.current_step]
                    actions.append(action)
                    intention.current_step += 1
                else:
                    intention.status = GoalStatus.ACHIEVED
        return actions
    
    def step(self, perception: Dict):
        self.update_beliefs(perception)
        self.deliberate()
        return self.execute_intentions()
```

---

## 3. LLM-Based Agents

Modern agents use LLMs as their reasoning core. The LLM interprets the situation, reasons about what to do, and selects tool calls.

### 3.1 The ReAct Pattern

ReAct (Yao et al., 2023) interleaves **Reasoning** (thought) and **Acting** (tool call):

```
Thought: I need to find the population of Tokyo.
Action: search("Tokyo population 2024")
Observation: Tokyo has approximately 13.96 million people in the city proper.
Thought: Now I have the population. The user asked me to multiply it by GDP per capita.
Action: search("Tokyo GDP per capita 2024")
Observation: Tokyo GDP per capita is approximately $65,000.
Thought: Population is 13.96M, GDP per capita is $65,000. I can compute now.
Action: calculator("13960000 * 65000")
Observation: 907400000000
Thought: I have the answer.
Final Answer: Tokyo's total GDP is approximately $907.4 billion.
```

```python
from openai import OpenAI
import json
import re

client = OpenAI()

SYSTEM_PROMPT = """You are a helpful AI agent. You have access to tools.

Use EXACTLY this format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON input for the tool]

When you have the final answer:
Thought: I have enough information to answer.
Final Answer: [your answer]

Available tools:
{tool_descriptions}
"""

def run_react_agent(query: str, tools: dict, max_iterations: int = 10) -> str:
    """Run a ReAct-style agent loop"""
    
    tool_descriptions = "\n".join([
        f"- {name}: {fn.__doc__}" for name, fn in tools.items()
    ])
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions
        )},
        {"role": "user", "content": query}
    ]
    
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )
        
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        
        # Check for final answer
        if "Final Answer:" in assistant_message:
            return assistant_message.split("Final Answer:")[-1].strip()
        
        # Parse action
        action_match = re.search(r"Action: (\w+)", assistant_message)
        input_match = re.search(r"Action Input: (.+?)(?=\nThought|\Z)", 
                                assistant_message, re.DOTALL)
        
        if action_match and input_match:
            tool_name = action_match.group(1)
            tool_input_str = input_match.group(1).strip()
            
            try:
                tool_input = json.loads(tool_input_str)
            except json.JSONDecodeError:
                tool_input = {"input": tool_input_str}
            
            # Execute tool
            if tool_name in tools:
                observation = str(tools[tool_name](**tool_input))
            else:
                observation = f"Error: Tool '{tool_name}' not found"
            
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        else:
            break
    
    return "Max iterations reached without final answer"

# Define tools
def search(query: str) -> str:
    """Search the web for information about the given query."""
    # In practice, call a search API
    return f"[Search results for '{query}']: ..."

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: Python math expression as string."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = {"search": search, "calculator": calculator}
result = run_react_agent("What is 15% of the population of France?", tools)
```

**ReAct variants**:
- **Plan-and-Execute**: First produce a full plan (steps 1..N), then execute; trades flexibility for structure
- **ReAct + Critique**: After each action, a separate "critic" step evaluates whether the observation answers the subgoal; triggers retry or replan
- **Chain-of-Thought then Act**: Reasoning in a block, then batched tool calls; reduces round-trips when multiple tools can run in parallel

**When ReAct struggles**: Dense multi-step plans (consider hierarchical planning); deterministic tool sequences (consider scripting); high-stakes errors (add human approval for tool use).

### 3.2 Reflexion

Reflexion (Shinn et al., 2023) adds **self-reflection**: after each failed attempt, the agent generates a verbal critique and uses it to improve the next attempt.

```python
class ReflexionAgent:
    """Agent with self-reflection memory"""
    
    def __init__(self, llm_client, tools: dict, max_trials: int = 3):
        self.client = llm_client
        self.tools = tools
        self.max_trials = max_trials
        self.reflections: List[str] = []  # Episodic memory of failures
    
    def reflect(self, task: str, attempt: str, outcome: str, success: bool) -> str:
        """Generate reflection on a failed attempt"""
        prompt = f"""You attempted a task and it did not succeed.
        
Task: {task}
Your attempt: {attempt}
Outcome: {outcome}
Success: {success}

Reflect on what went wrong. Be specific about:
1. What mistake was made
2. What information was missing
3. What you should do differently next time

Reflection:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    def build_prompt_with_reflections(self, task: str) -> str:
        """Include past reflections in the prompt"""
        base = f"Task: {task}\n\n"
        if self.reflections:
            base += "IMPORTANT - Lessons from previous attempts:\n"
            for i, r in enumerate(self.reflections, 1):
                base += f"{i}. {r}\n"
            base += "\nUse these lessons to do better this time.\n\n"
        return base
    
    def solve(self, task: str, evaluator) -> str:
        """Solve task with reflective retries"""
        for trial in range(self.max_trials):
            prompt = self.build_prompt_with_reflections(task)
            
            # Run agent
            attempt = run_react_agent(prompt, self.tools)
            
            # Evaluate
            success, feedback = evaluator(task, attempt)
            
            if success:
                return attempt
            
            # Generate reflection
            reflection = self.reflect(task, attempt, feedback, success)
            self.reflections.append(reflection)
            print(f"Trial {trial+1} failed. Reflecting...")
        
        return f"Failed after {self.max_trials} trials. Best attempt: {attempt}"
```

### 3.3 AutoGPT-Style Agents

AutoGPT demonstrated fully autonomous task execution with persistent memory:

```python
import json
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI

class AutoGPTAgent:
    """AutoGPT-inspired autonomous agent with memory and planning"""
    
    AGENT_SYSTEM = """You are {name}, an AI agent.
Your role: {role}
Goals: {goals}

You have access to these tools: {tools}

Respond ONLY in this JSON format:
{{
    "thoughts": {{
        "text": "your thought",
        "reasoning": "why you're thinking this",
        "plan": ["step 1", "step 2", ...],
        "criticism": "constructive self-criticism",
        "speak": "what to say to user"
    }},
    "command": {{
        "name": "tool_name",
        "args": {{"arg_name": "value"}}
    }}
}}
"""
    
    def __init__(
        self,
        name: str,
        role: str,
        goals: List[str],
        tools: Dict,
        memory_size: int = 20
    ):
        self.name = name
        self.role = role
        self.goals = goals
        self.tools = tools
        self.memory: List[Dict] = []  # Rolling memory
        self.memory_size = memory_size
        self.client = OpenAI()
        self.completed_tasks: List[str] = []
        self.iteration = 0
    
    def _get_system_prompt(self) -> str:
        return self.AGENT_SYSTEM.format(
            name=self.name,
            role=self.role,
            goals="\n".join(f"{i+1}. {g}" for i, g in enumerate(self.goals)),
            tools=", ".join(self.tools.keys())
        )
    
    def _build_messages(self, current_observation: Optional[str] = None) -> List[Dict]:
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        
        # Add memory (recent events)
        for mem in self.memory[-self.memory_size:]:
            messages.append({"role": "user", "content": mem["observation"]})
            messages.append({"role": "assistant", "content": json.dumps(mem["response"])})
        
        # Current observation
        if current_observation:
            messages.append({"role": "user", "content": current_observation})
        else:
            messages.append({
                "role": "user",
                "content": f"Goals: {self.goals}\nWhat is your next action?"
            })
        
        return messages
    
    def think(self, observation: Optional[str] = None) -> Dict:
        """Generate next thought and action"""
        messages = self._build_messages(observation)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5
        )
        
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "thoughts": {"text": content, "speak": content},
                "command": {"name": "do_nothing", "args": {}}
            }
    
    def execute(self, command: Dict) -> str:
        """Execute a tool command"""
        name = command.get("name", "")
        args = command.get("args", {})
        
        if name == "task_complete":
            return "TASK_COMPLETE"
        
        if name in self.tools:
            try:
                result = self.tools[name](**args)
                return f"Command {name} executed. Result: {result}"
            except Exception as e:
                return f"Command {name} failed: {e}"
        
        return f"Unknown command: {name}"
    
    def run(self, max_iterations: int = 20):
        """Main autonomous loop"""
        observation = None
        
        for i in range(max_iterations):
            self.iteration = i
            
            # Think
            response = self.think(observation)
            thoughts = response.get("thoughts", {})
            command = response.get("command", {})
            
            print(f"\n[Iteration {i+1}]")
            print(f"Thought: {thoughts.get('text', '')}")
            print(f"Plan: {thoughts.get('plan', [])}")
            print(f"Command: {command.get('name')} ({command.get('args', {})})")
            
            # Execute
            result = self.execute(command)
            
            if result == "TASK_COMPLETE":
                print("Agent completed all goals!")
                return "DONE"
            
            observation = f"Result of {command.get('name')}: {result}"
            
            # Update memory
            self.memory.append({
                "observation": observation,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
        
        return "Max iterations reached"
```

### 3.4 BabyAGI Pattern (Task-Driven Autonomous Agent)

BabyAGI uses a task queue with dynamic generation:

```python
from collections import deque

class BabyAGI:
    """Task-driven autonomous agent with dynamic task generation"""
    
    def __init__(self, llm_client, tools: Dict, objective: str):
        self.client = llm_client
        self.tools = tools
        self.objective = objective
        self.task_queue: deque = deque()
        self.completed_tasks: List[Dict] = []
        self.task_id_counter = 0
    
    def _next_id(self) -> int:
        self.task_id_counter += 1
        return self.task_id_counter
    
    def task_creation_agent(self, result: str, task: str) -> List[str]:
        """Create new tasks based on completed task result"""
        completed_str = "\n".join([f"- {t['task']}" for t in self.completed_tasks])
        
        prompt = f"""Objective: {self.objective}
Completed tasks:
{completed_str}

Last completed task: {task}
Result: {result}

Based on the result, create new tasks needed to reach the objective.
Do NOT recreate completed tasks. Be specific and actionable.
Return each task on a new line:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        new_tasks = response.choices[0].message.content.strip().split("\n")
        return [t.strip("- ").strip() for t in new_tasks if t.strip()]
    
    def prioritization_agent(self) -> List[Dict]:
        """Reprioritize task queue"""
        task_names = [t["task"] for t in self.task_queue]
        
        prompt = f"""Objective: {self.objective}
Tasks to prioritize (one per line, most important first):
{chr(10).join(task_names)}

Reprioritize these tasks. Return in order, one per line:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        ordered = response.choices[0].message.content.strip().split("\n")
        return [{"id": self._next_id(), "task": t.strip()} for t in ordered if t.strip()]
    
    def execution_agent(self, task: str) -> str:
        """Execute a single task using available tools"""
        context = "\n".join([
            f"Task: {t['task']}\nResult: {t['result']}"
            for t in self.completed_tasks[-5:]  # Last 5 results as context
        ])
        
        prompt = f"""Objective: {self.objective}
Context from previous tasks:
{context}

Your task: {task}
Complete the task. Use the following tools if needed: {list(self.tools.keys())}
Result:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def run(self, initial_tasks: List[str], max_tasks: int = 30):
        """Run BabyAGI loop"""
        # Initialize task queue
        for task in initial_tasks:
            self.task_queue.append({"id": self._next_id(), "task": task})
        
        tasks_processed = 0
        
        while self.task_queue and tasks_processed < max_tasks:
            # Get next task
            task = self.task_queue.popleft()
            print(f"\n[Task {task['id']}]: {task['task']}")
            
            # Execute task
            result = self.execution_agent(task["task"])
            print(f"Result: {result[:200]}...")
            
            # Store completed task
            self.completed_tasks.append({
                "task": task["task"],
                "result": result
            })
            tasks_processed += 1
            
            # Generate new tasks
            new_tasks = self.task_creation_agent(result, task["task"])
            for new_task in new_tasks:
                self.task_queue.append({"id": self._next_id(), "task": new_task})
            
            # Reprioritize
            if len(self.task_queue) > 1:
                self.task_queue = deque(self.prioritization_agent())
        
        return self.completed_tasks
```

---

## 4. Tool Use and Function Calling

### 4.1 OpenAI Function Calling

OpenAI's function calling allows structured tool invocation via JSON schema:

```python
from openai import OpenAI
import json
import requests
from typing import Any

client = OpenAI()

# Define tools with JSON schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or lat,lng coordinates"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for recent information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def get_weather(location: str, units: str = "celsius") -> Dict:
    """Actual weather API call"""
    # Simulate API call
    return {
        "location": location,
        "temperature": 22,
        "units": units,
        "description": "Partly cloudy",
        "humidity": 65
    }

def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """Simulate web search"""
    return [{"title": f"Result {i}", "snippet": f"Info about {query}"} 
            for i in range(num_results)]

def run_tool(tool_name: str, tool_args: Dict) -> str:
    """Dispatch tool calls"""
    dispatch = {
        "get_weather": get_weather,
        "search_web": search_web,
    }
    if tool_name in dispatch:
        result = dispatch[tool_name](**tool_args)
        return json.dumps(result)
    return f"Unknown tool: {tool_name}"

def agent_with_function_calling(user_query: str) -> str:
    """Agent using OpenAI function calling"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_query}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # No tool calls — final answer
        if not message.tool_calls:
            return message.content
        
        # Process all tool calls
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            result = run_tool(tool_name, tool_args)
            
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })

result = agent_with_function_calling(
    "What's the weather in Paris and San Francisco? Which is warmer?"
)
print(result)
```

### 4.2 Tool Use Patterns

Beyond single tool calls, agents often need structured interaction patterns:

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Sequential chaining** | Tool A output → input to Tool B | Multi-step workflows (search → summarize → email) |
| **Parallel tool calls** | Invoke multiple tools in one turn | Independent subtasks (weather + calendar + news) |
| **Conditional tool use** | Only call tool if previous result meets condition | "If search returns nothing, try alternative tool" |
| **Tool selection heuristics** | Pre-filter tools by relevance before LLM chooses | 50+ tools; reduce hallucinated tool names |

**Parallel tool calls** (OpenAI supports multiple `tool_calls` per message):
```python
# LLM returns message.tool_calls = [tc1, tc2, tc3]
# Execute all, append all tool results, send back in one response
results = [run_tool(tc.function.name, json.loads(tc.function.arguments)) for tc in message.tool_calls]
messages.append({"role": "tool", "content": json.dumps(results), "tool_call_id": ...})
```

**Tool selection heuristic**: When tool count is large, use a light classifier or embedding similarity to pre-select top-k tools before passing to LLM:
```python
def filter_tools(user_query: str, tools: List[Tool], k: int = 5) -> List[Tool]:
    """Reduce tool set via semantic similarity to query"""
    query_emb = embed(user_query)
    tool_embs = [embed(t.description) for t in tools]
    scores = cosine_similarity([query_emb], tool_embs)[0]
    return [tools[i] for i in np.argsort(scores)[-k:]]
```

### 4.3 Tool Design Patterns

#### Pattern 1: Schema-First Tool Design

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class WebSearchInput(BaseModel):
    """Schema for web search tool"""
    query: str = Field(
        ...,
        description="Search query. Be specific and concise.",
        min_length=1,
        max_length=500
    )
    search_type: Literal["general", "news", "academic"] = Field(
        default="general",
        description="Type of search to perform"
    )
    date_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="Limit results to this time range"
    )
    
    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class ToolResult(BaseModel):
    """Standard tool result structure"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WebSearchTool:
    """Well-designed web search tool"""
    
    name = "web_search"
    description = "Search the web for information"
    input_schema = WebSearchInput
    
    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
    
    def run(self, **kwargs) -> ToolResult:
        try:
            validated = self.input_schema(**kwargs)
            
            # Execute search
            results = self._execute_search(validated)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={"query": validated.query, "num_results": len(results)}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
    
    def _execute_search(self, params: WebSearchInput) -> List[Dict]:
        # Actual search implementation
        return []
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function calling format"""
        schema = self.input_schema.schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema
            }
        }
```

#### Pattern 2: Error Handling and Retry

```python
import time
import functools
from typing import Callable

def with_retry(max_retries: int = 3, backoff: float = 1.0):
    """Decorator for tool retry logic"""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError as e:
                    wait = backoff * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    last_error = e
                except (ConnectionError, TimeoutError) as e:
                    wait = backoff * (attempt + 1)
                    print(f"Connection error. Waiting {wait}s...")
                    time.sleep(wait)
                    last_error = e
                except Exception as e:
                    # Non-retryable error
                    raise
            raise last_error
        return wrapper
    return decorator

class RobustToolExecutor:
    """Tool executor with comprehensive error handling"""
    
    def __init__(self, tools: Dict, timeout: int = 30):
        self.tools = tools
        self.timeout = timeout
        self.execution_log: List[Dict] = []
    
    @with_retry(max_retries=3, backoff=1.0)
    def execute(self, tool_name: str, args: Dict) -> ToolResult:
        """Execute tool with retry, timeout, and logging"""
        start = time.time()
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
            )
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.run(**args)
        except Exception as e:
            result = ToolResult(success=False, data=None, error=str(e))
        
        elapsed = time.time() - start
        
        # Log execution
        self.execution_log.append({
            "tool": tool_name,
            "args": args,
            "success": result.success,
            "elapsed_ms": int(elapsed * 1000),
            "timestamp": datetime.now().isoformat()
        })
        
        return result
```

#### Pattern 3: Observability

```python
import logging
from dataclasses import dataclass, asdict
import uuid

@dataclass
class ToolSpan:
    """OpenTelemetry-compatible span for tool calls"""
    span_id: str
    tool_name: str
    args: Dict
    result: Optional[Dict]
    start_time: float
    end_time: Optional[float]
    status: str  # "running", "success", "error"
    error: Optional[str]

class ObservableToolExecutor:
    """Tool executor with full observability"""
    
    def __init__(self, tools: Dict):
        self.tools = tools
        self.spans: List[ToolSpan] = []
        self.logger = logging.getLogger("agent.tools")
    
    def execute(self, tool_name: str, args: Dict) -> ToolResult:
        span = ToolSpan(
            span_id=str(uuid.uuid4())[:8],
            tool_name=tool_name,
            args=args,
            result=None,
            start_time=time.time(),
            end_time=None,
            status="running",
            error=None
        )
        self.spans.append(span)
        
        self.logger.info(f"[{span.span_id}] Calling {tool_name}({args})")
        
        try:
            tool = self.tools[tool_name]
            result = tool.run(**args)
            
            span.result = asdict(result)
            span.status = "success"
            span.end_time = time.time()
            
            self.logger.info(
                f"[{span.span_id}] {tool_name} succeeded in "
                f"{(span.end_time - span.start_time)*1000:.0f}ms"
            )
            return result
            
        except Exception as e:
            span.error = str(e)
            span.status = "error"
            span.end_time = time.time()
            
            self.logger.error(f"[{span.span_id}] {tool_name} failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_trace(self) -> List[Dict]:
        return [asdict(s) for s in self.spans]
```

---

## 5. Memory Systems

### 5.1 In-Context Memory (Working Memory)

The LLM's context window is its working memory. Managing it well is critical:

```python
from typing import List, Dict, Tuple
import tiktoken

class ContextWindowManager:
    """Manages the agent's context window efficiently"""
    
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 128000):
        self.model = model
        self.max_tokens = max_tokens
        self.reserved_output_tokens = 4096
        self.encoding = tiktoken.encoding_for_model(model)
        self.messages: List[Dict] = []
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def message_tokens(self, message: Dict) -> int:
        return self.count_tokens(message.get("content", "")) + 4
    
    def total_tokens(self) -> int:
        return sum(self.message_tokens(m) for m in self.messages)
    
    def available_tokens(self) -> int:
        return self.max_tokens - self.reserved_output_tokens - self.total_tokens()
    
    def add_message(self, role: str, content: str, priority: int = 1):
        """Add message with priority (higher = keep longer)"""
        self.messages.append({
            "role": role,
            "content": content,
            "priority": priority
        })
        self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict low-priority messages when context is full"""
        while self.available_tokens() < 0 and len(self.messages) > 2:
            # Find lowest priority non-system message
            evictable = [(i, m) for i, m in enumerate(self.messages)
                        if m["role"] != "system" and m.get("priority", 1) == 1]
            
            if not evictable:
                # Evict oldest non-system message
                for i, m in enumerate(self.messages):
                    if m["role"] != "system":
                        self.messages.pop(i)
                        break
            else:
                # Evict lowest priority oldest message
                idx = evictable[0][0]
                self.messages.pop(idx)
    
    def get_messages(self) -> List[Dict]:
        """Return messages without internal metadata"""
        return [{"role": m["role"], "content": m["content"]}
                for m in self.messages]
    
    def compress_history(self, llm_client, keep_recent: int = 5):
        """Summarize old messages to save context"""
        if len(self.messages) <= keep_recent + 1:
            return
        
        # Keep system + recent, summarize the rest
        system_messages = [m for m in self.messages if m["role"] == "system"]
        old_messages = self.messages[len(system_messages):-keep_recent]
        recent_messages = self.messages[-keep_recent:]
        
        # Summarize old messages
        summary_prompt = "Summarize this conversation history concisely:\n" + \
            "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
        
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = response.choices[0].message.content
        
        # Replace old messages with summary
        self.messages = system_messages + [
            {"role": "system", "content": f"[Previous conversation summary]: {summary}",
             "priority": 2}
        ] + recent_messages
```

### 5.2 Episodic Memory

Episodic memory stores experiences (what happened, when, in what context):

```python
import sqlite3
from datetime import datetime
import json

class EpisodicMemory:
    """SQLite-backed episodic memory for agents"""
    
    def __init__(self, db_path: str = "agent_episodes.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding_id TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON episodes(event_type)")
        conn.commit()
        conn.close()
    
    def store(
        self,
        session_id: str,
        event_type: str,
        content: str,
        metadata: Dict = None
    ) -> int:
        """Store an episodic memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO episodes (session_id, timestamp, event_type, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            event_type,
            content,
            json.dumps(metadata or {})
        ))
        episode_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return episode_id
    
    def retrieve_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Retrieve all episodes from a session"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM episodes WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def retrieve_by_type(self, event_type: str, limit: int = 20) -> List[Dict]:
        """Retrieve episodes by event type"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM episodes WHERE event_type = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (event_type, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_similar_episodes(self, content: str, limit: int = 5) -> List[Dict]:
        """Basic keyword-based similarity (upgrade to vector search)"""
        keywords = content.lower().split()[:5]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Simple LIKE search (production: use vector similarity)
        conditions = " OR ".join(["LOWER(content) LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        
        rows = conn.execute(
            f"SELECT * FROM episodes WHERE {conditions} LIMIT ?",
            params + [limit]
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
```

### 5.3 Semantic Memory with Vector Stores

Semantic memory enables retrieval of relevant knowledge via embedding similarity:

```python
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.config import Settings

class SemanticMemory:
    """Vector-store-backed semantic memory"""
    
    def __init__(self, collection_name: str = "agent_memory"):
        self.client_openai = OpenAI()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        response = self.client_openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def store(
        self,
        text: str,
        doc_id: str = None,
        metadata: Dict = None
    ) -> str:
        """Store a memory with its embedding"""
        doc_id = doc_id or str(uuid.uuid4())
        embedding = self._embed(text)
        
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata or {}]
        )
        return doc_id
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> List[Dict]:
        """Retrieve relevant memories by semantic similarity"""
        query_embedding = self._embed(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        memories = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            memories.append({
                "text": doc,
                "metadata": meta,
                "relevance_score": 1 - dist,  # Convert distance to similarity
                "id": results["ids"][0][i]
            })
        
        return sorted(memories, key=lambda x: -x["relevance_score"])
    
    def store_conversation(self, messages: List[Dict], session_id: str):
        """Store conversation turns as memories"""
        for i, msg in enumerate(messages):
            self.store(
                text=f"{msg['role']}: {msg['content']}",
                metadata={
                    "session_id": session_id,
                    "role": msg["role"],
                    "turn": i,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        """Format retrieved memories as context string"""
        memories = self.retrieve(query, n_results=10)
        
        context_parts = []
        token_count = 0
        
        for mem in memories:
            mem_text = f"[Memory (score={mem['relevance_score']:.2f})]: {mem['text']}"
            estimated_tokens = len(mem_text.split()) * 1.3  # rough estimate
            
            if token_count + estimated_tokens > max_tokens:
                break
            
            context_parts.append(mem_text)
            token_count += estimated_tokens
        
        return "\n".join(context_parts)
```

### 5.4 Procedural Memory (Skills)

Procedural memory encodes "how to do things" — reusable skills:

```python
class ProcedureLibrary:
    """Registry of reusable agent procedures/skills"""
    
    def __init__(self):
        self.procedures: Dict[str, Dict] = {}
    
    def register(
        self,
        name: str,
        description: str,
        steps: List[str],
        tools_required: List[str] = None,
        success_criteria: str = None
    ):
        """Register a reusable procedure"""
        self.procedures[name] = {
            "description": description,
            "steps": steps,
            "tools_required": tools_required or [],
            "success_criteria": success_criteria,
            "usage_count": 0
        }
    
    def retrieve_relevant(self, goal: str, top_k: int = 3) -> List[Dict]:
        """Find procedures relevant to the current goal"""
        # In production, use semantic similarity
        # Here: keyword matching
        goal_words = set(goal.lower().split())
        
        scored = []
        for name, proc in self.procedures.items():
            proc_words = set(proc["description"].lower().split())
            overlap = len(goal_words & proc_words)
            scored.append((overlap, name, proc))
        
        scored.sort(key=lambda x: -x[0])
        return [{"name": s[1], **s[2]} for s in scored[:top_k]]
    
    def get_prompt_for_goal(self, goal: str) -> str:
        """Build prompt with relevant procedures"""
        procedures = self.retrieve_relevant(goal)
        
        if not procedures:
            return ""
        
        prompt = "RELEVANT PROCEDURES FROM MEMORY:\n"
        for proc in procedures:
            prompt += f"\n{proc['name']}: {proc['description']}\n"
            prompt += "Steps:\n" + "\n".join(f"  {i+1}. {s}" 
                                              for i, s in enumerate(proc["steps"]))
        return prompt

# Setup
library = ProcedureLibrary()
library.register(
    name="web_research",
    description="Research a topic using web search",
    steps=[
        "Search for the main topic",
        "Search for recent news and updates",
        "Search for expert opinions",
        "Synthesize findings from multiple sources",
        "Verify key claims"
    ],
    tools_required=["search", "browser"]
)
library.register(
    name="data_analysis",
    description="Analyze a dataset and generate insights",
    steps=[
        "Load and inspect the dataset",
        "Check for missing values and outliers",
        "Compute descriptive statistics",
        "Visualize distributions and relationships",
        "Generate insights and recommendations"
    ],
    tools_required=["python_repl", "file_reader"]
)
```

---

## 6. Planning and Task Decomposition

### 6.1 Hierarchical Task Decomposition

```python
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    subtasks: List['Task'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    tool: Optional[str] = None
    tool_args: Dict = field(default_factory=dict)
    
    def is_ready(self, completed_ids: set) -> bool:
        return all(dep in completed_ids for dep in self.dependencies)

class HierarchicalPlanner:
    """Plan tasks hierarchically with dependency tracking"""
    
    def __init__(self, llm_client, tools: List[str]):
        self.client = llm_client
        self.tools = tools
        self.task_counter = 0
    
    def _next_id(self) -> str:
        self.task_counter += 1
        return f"task_{self.task_counter}"
    
    def decompose(self, goal: str, context: str = "") -> List[Task]:
        """Use LLM to decompose goal into tasks"""
        prompt = f"""You are a planning agent. Decompose the following goal into specific subtasks.

Goal: {goal}
Context: {context}
Available tools: {', '.join(self.tools)}

Output a JSON list of tasks:
[
  {{
    "description": "specific task description",
    "tool": "tool_name or null",
    "tool_args": {{"key": "value"}},
    "dependencies": []  // IDs of tasks this depends on (0-indexed)
  }},
  ...
]

Be specific. Each task should be atomic and achievable with one tool call."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        task_data = json.loads(response.choices[0].message.content)
        tasks_raw = task_data.get("tasks", task_data) if isinstance(task_data, dict) else task_data
        
        # Create task objects
        tasks = []
        id_map = {}
        
        for i, t in enumerate(tasks_raw):
            task_id = self._next_id()
            id_map[i] = task_id
            
            # Resolve dependency indices to IDs
            deps = [id_map.get(d, f"task_{d+1}") for d in t.get("dependencies", [])]
            
            tasks.append(Task(
                id=task_id,
                description=t["description"],
                tool=t.get("tool"),
                tool_args=t.get("tool_args", {}),
                dependencies=deps
            ))
        
        return tasks
    
    def execute_plan(
        self,
        tasks: List[Task],
        tool_executor,
        llm_client
    ) -> Dict[str, str]:
        """Execute tasks respecting dependencies"""
        completed = {}  # task_id -> result
        
        while not all(t.status == TaskStatus.COMPLETED for t in tasks):
            made_progress = False
            
            for task in tasks:
                if task.status != TaskStatus.PENDING:
                    continue
                
                if not task.is_ready(set(completed.keys())):
                    continue
                
                # Execute task
                task.status = TaskStatus.IN_PROGRESS
                
                # Build context from completed dependency results
                dep_context = "\n".join([
                    f"Result of '{t.description}': {completed[t.id]}"
                    for t in tasks if t.id in completed
                ])
                
                if task.tool:
                    # Direct tool execution
                    result = tool_executor.execute(task.tool, task.tool_args)
                    task.result = result.data if result.success else f"Error: {result.error}"
                else:
                    # LLM task
                    response = llm_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{
                            "role": "user",
                            "content": f"Context:\n{dep_context}\n\nTask: {task.description}"
                        }]
                    )
                    task.result = response.choices[0].message.content
                
                task.status = TaskStatus.COMPLETED
                completed[task.id] = task.result
                made_progress = True
                print(f"✓ Completed: {task.description}")
            
            if not made_progress:
                # Deadlock — fail remaining tasks
                for task in tasks:
                    if task.status == TaskStatus.PENDING:
                        task.status = TaskStatus.FAILED
                break
        
        return completed
```

### 6.2 Monte Carlo Tree Search (MCTS) for Planning

MCTS finds optimal action sequences by balancing exploration and exploitation:

```python
import math
import random
from copy import deepcopy

class MCTSNode:
    """Node in Monte Carlo Tree Search"""
    
    def __init__(self, state: str, parent=None, action: str = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[str] = []
    
    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound for Trees"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def best_child(self) -> 'MCTSNode':
        return max(self.children, key=lambda c: c.ucb1())
    
    def best_action_child(self) -> 'MCTSNode':
        """Best child by average value (for final selection)"""
        return max(self.children, key=lambda c: c.value / (c.visits + 1e-8))

class LLMPlanner:
    """MCTS planning with LLM for action generation and evaluation"""
    
    def __init__(self, llm_client, goal: str):
        self.client = llm_client
        self.goal = goal
    
    def get_actions(self, state: str) -> List[str]:
        """Generate possible next actions from current state"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Goal: {self.goal}\nCurrent state: {state}\n"
                          f"List 4 possible next actions (one per line):"
            }],
            temperature=0.8
        )
        actions = response.choices[0].message.content.strip().split("\n")
        return [a.strip("- •123456789. ") for a in actions if a.strip()][:4]
    
    def simulate(self, state: str, action: str) -> Tuple[str, float]:
        """Simulate taking action, return (next_state, reward)"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Goal: {self.goal}\nState: {state}\nAction: {action}\n"
                          f"Describe what happens (next state) and rate goal progress 0-1:"
            }],
            temperature=0.7
        )
        content = response.choices[0].message.content
        
        # Extract reward (simplified — in practice parse more carefully)
        import re
        numbers = re.findall(r"0\.\d+|1\.0", content)
        reward = float(numbers[-1]) if numbers else 0.5
        
        return content, reward
    
    def evaluate(self, state: str) -> float:
        """Evaluate how close state is to goal"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Goal: {self.goal}\nCurrent state: {state}\n"
                          f"Rate progress toward goal from 0.0 (no progress) to 1.0 (achieved).\n"
                          f"Return only a decimal number:"
            }],
            temperature=0
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5
    
    def search(self, initial_state: str, iterations: int = 50) -> List[str]:
        """Run MCTS and return best action sequence"""
        root = MCTSNode(state=initial_state)
        root.untried_actions = self.get_actions(initial_state)
        
        for _ in range(iterations):
            node = root
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            
            # Expansion
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                next_state, _ = self.simulate(node.state, action)
                child = MCTSNode(state=next_state, parent=node, action=action)
                child.untried_actions = self.get_actions(next_state)
                node.children.append(child)
                node = child
            
            # Simulation (rollout)
            value = self.evaluate(node.state)
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
        
        # Extract best path
        path = []
        node = root
        while node.children:
            node = node.best_action_child()
            if node.action:
                path.append(node.action)
        
        return path
```

### 6.3 Plan-Execute-Replan

```python
class PlanExecuteReplanner:
    """Agent that plans, executes, and replans based on results"""
    
    def __init__(self, planner, executor, llm_client):
        self.planner = planner
        self.executor = executor
        self.client = llm_client
    
    def should_replan(self, original_goal: str, executed_steps: List[Dict], last_result: str) -> bool:
        """Determine if replanning is needed"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Goal: {original_goal}
Executed steps: {json.dumps(executed_steps, indent=2)}
Last result: {last_result}

Does the plan need to be updated? Answer only YES or NO."""
            }],
            temperature=0
        )
        return "YES" in response.choices[0].message.content.upper()
    
    def replan(self, goal: str, remaining_tasks: List[Task], context: str) -> List[Task]:
        """Generate updated plan given context"""
        completed_str = context
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Goal: {goal}
Completed work: {completed_str}
Remaining planned tasks: {[t.description for t in remaining_tasks]}

Update the remaining tasks if needed. Return updated task list as JSON."""
            }],
            temperature=0
        )
        # Parse and return updated tasks
        return remaining_tasks  # Simplified
    
    def run(self, goal: str, max_replans: int = 3):
        """Full plan-execute-replan loop"""
        tasks = self.planner.decompose(goal)
        completed_context = ""
        replans = 0
        
        for task in tasks[:]:
            result = self.executor.execute_task(task)
            completed_context += f"\n- {task.description}: {result}"
            
            if self.should_replan(goal, [], result) and replans < max_replans:
                remaining = [t for t in tasks if t.status == TaskStatus.PENDING]
                tasks = self.replan(goal, remaining, completed_context)
                replans += 1
        
        return completed_context
```

### 6.4 Plan Critique

Before executing a plan, a **critic** can evaluate it for feasibility and completeness:

```python
def critique_plan(goal: str, plan: List[Task], tools: List[str]) -> Tuple[bool, str]:
    """Critic evaluates plan before execution"""
    prompt = f"""Goal: {goal}
Proposed plan: {[t.description for t in plan]}
Available tools: {tools}

Evaluate: (1) Is the plan complete? (2) Are dependencies correct? (3) Any missing steps?
Return JSON: {{"approve": true/false, "feedback": "..."}}"""
    result = json.loads(llm.generate(prompt))
    return result["approve"], result.get("feedback", "")

# In planner loop:
approved, feedback = critique_plan(goal, tasks, tools)
if not approved:
    tasks = replan_with_feedback(goal, tasks, feedback)
```

**Use cases**: Expensive tools (avoid bad plans); safety-critical domains; teaching the planner via feedback.

---

## 7. Multi-Agent Systems

### 7.1 Communication Protocols

```python
from dataclasses import dataclass
from enum import Enum
import asyncio
from asyncio import Queue

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DELEGATE = "delegate"
    REPORT = "report"
    QUESTION = "question"
    ANSWER = "answer"

@dataclass
class AgentMessage:
    sender: str
    receiver: str  # "broadcast" for all agents
    message_type: MessageType
    content: str
    payload: Dict = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class AgentCommunicationBus:
    """Message bus for multi-agent communication"""
    
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.message_log: List[AgentMessage] = []
        self.subscriptions: Dict[str, List[str]] = {}  # agent -> topics
    
    def register_agent(self, agent_name: str):
        """Register an agent with a message queue"""
        self.queues[agent_name] = Queue()
    
    async def send(self, message: AgentMessage):
        """Send a message"""
        self.message_log.append(message)
        
        if message.receiver == "broadcast":
            # Send to all agents
            for name, q in self.queues.items():
                if name != message.sender:
                    await q.put(message)
        else:
            if message.receiver in self.queues:
                await self.queues[message.receiver].put(message)
    
    async def receive(self, agent_name: str, timeout: float = 5.0) -> Optional[AgentMessage]:
        """Receive a message for an agent"""
        try:
            return await asyncio.wait_for(
                self.queues[agent_name].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    def get_conversation(self, correlation_id: str) -> List[AgentMessage]:
        """Get all messages in a conversation"""
        return [m for m in self.message_log if m.correlation_id == correlation_id]
```

### 7.2 Orchestration Patterns

#### Hierarchical Orchestration

```python
class OrchestratorAgent:
    """Central orchestrator that delegates to specialist agents"""
    
    def __init__(self, llm_client, agents: Dict[str, 'BaseAgent']):
        self.client = llm_client
        self.agents = agents
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_name
    
    def select_agent(self, task: str) -> str:
        """Select the best agent for a task"""
        agent_descriptions = "\n".join([
            f"- {name}: {agent.description}"
            for name, agent in self.agents.items()
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Task: {task}

Available agents:
{agent_descriptions}

Which agent is best for this task? Reply with just the agent name."""
            }],
            temperature=0
        )
        
        agent_name = response.choices[0].message.content.strip()
        return agent_name if agent_name in self.agents else list(self.agents.keys())[0]
    
    def run(self, goal: str) -> str:
        """Orchestrate agents to achieve goal"""
        # Decompose goal
        tasks = self._decompose_goal(goal)
        results = {}
        
        for task in tasks:
            agent_name = self.select_agent(task["description"])
            agent = self.agents[agent_name]
            
            # Provide context from previous results
            context = "\n".join([f"{k}: {v}" for k, v in results.items()])
            result = agent.execute(task["description"], context=context)
            
            results[task["id"]] = result
            print(f"[{agent_name}] Completed: {task['description'][:50]}...")
        
        # Synthesize final answer
        return self._synthesize(goal, results)
    
    def _decompose_goal(self, goal: str) -> List[Dict]:
        """Break goal into tasks"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Break this goal into 3-5 specific tasks (JSON list with id, description):\n{goal}"
            }],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("tasks", [])
    
    def _synthesize(self, goal: str, results: Dict) -> str:
        """Synthesize agent results into final answer"""
        results_str = "\n\n".join([f"Task {k}:\n{v}" for k, v in results.items()])
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Goal: {goal}\n\nAgent results:\n{results_str}\n\nSynthesize a final answer:"
            }]
        )
        return response.choices[0].message.content
```

### 7.3 Agent Debate

Multiple agents debate to improve answer quality:

```python
class AgentDebate:
    """Multi-agent debate for improved reasoning"""
    
    def __init__(self, llm_client, n_agents: int = 3, rounds: int = 2):
        self.client = llm_client
        self.n_agents = n_agents
        self.rounds = rounds
    
    def get_initial_answers(self, question: str) -> List[str]:
        """Get independent answers from each agent"""
        answers = []
        for i in range(self.n_agents):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "system",
                    "content": f"You are Expert Agent {i+1}. Give your best answer."
                }, {
                    "role": "user",
                    "content": question
                }],
                temperature=0.7
            )
            answers.append(response.choices[0].message.content)
        return answers
    
    def debate_round(self, question: str, answers: List[str], agent_idx: int) -> str:
        """One agent critiques and updates based on others' answers"""
        other_answers = "\n\n".join([
            f"Agent {i+1}: {a}"
            for i, a in enumerate(answers)
            if i != agent_idx
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": f"You are Expert Agent {agent_idx+1}."
            }, {
                "role": "user",
                "content": f"""Question: {question}

Your previous answer: {answers[agent_idx]}

Other agents' answers:
{other_answers}

Review the other answers carefully. Update your answer if they raise valid points.
Output your updated answer:"""
            }],
            temperature=0.5
        )
        return response.choices[0].message.content
    
    def synthesize(self, question: str, final_answers: List[str]) -> str:
        """Synthesize debate into final answer"""
        answers_str = "\n\n".join([f"Agent {i+1}: {a}" for i, a in enumerate(final_answers)])
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Question: {question}

After {self.rounds} rounds of debate, agents provided:
{answers_str}

Synthesize these into the single best answer:"""
            }],
            temperature=0
        )
        return response.choices[0].message.content
    
    def run(self, question: str) -> str:
        """Run full debate"""
        print(f"Starting {self.n_agents}-agent debate for {self.rounds} rounds...")
        
        answers = self.get_initial_answers(question)
        
        for round_num in range(self.rounds):
            print(f"Round {round_num + 1}...")
            new_answers = []
            for i in range(self.n_agents):
                new_answer = self.debate_round(question, answers, i)
                new_answers.append(new_answer)
            answers = new_answers
        
        return self.synthesize(question, answers)
```

---

## 8. Agent Frameworks

### 8.1 LangChain Agents (Modern API)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun

# Modern tool definition with @tool decorator
@tool
def search(query: str) -> str:
    """Search the web for information about any topic."""
    ddg = DuckDuckGoSearchRun()
    return ddg.run(query)

@tool
def calculate(expression: str) -> str:
    """Evaluate a Python mathematical expression safely."""
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [search, calculate, get_current_time]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to tools.
Think step by step and use tools when needed.
Always verify your answers before responding."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Memory
memory = ConversationBufferWindowMemory(
    k=10,
    memory_key="chat_history",
    return_messages=True
)

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# Run
result = agent_executor.invoke({"input": "What is the square root of the year we're in?"})
```

### 8.2 AutoGen (Microsoft)

```python
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Configuration
config_list = [{"model": "gpt-4o", "api_key": "YOUR_KEY"}]
llm_config = {"config_list": config_list, "temperature": 0}

# Define agents
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant. Use code when needed."
)

code_executor = UserProxyAgent(
    name="code_executor",
    human_input_mode="NEVER",  # Fully autonomous
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False
    },
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

# Simple two-agent conversation
code_executor.initiate_chat(
    assistant,
    message="Write a Python script that calculates prime numbers up to 100 and plots their distribution."
)

# Multi-agent group chat
researcher = AssistantAgent(
    name="researcher",
    llm_config=llm_config,
    system_message="You are a research specialist. Search and gather information."
)

writer = AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="You are a technical writer. Write clear, structured documents."
)

critic = AssistantAgent(
    name="critic",
    llm_config=llm_config,
    system_message="You review outputs critically and suggest improvements."
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False,
    is_termination_msg=lambda x: "FINAL_ANSWER" in x.get("content", "")
)

group_chat = GroupChat(
    agents=[user_proxy, researcher, writer, critic],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)
user_proxy.initiate_chat(manager, message="Write a comprehensive blog post about transformer architecture.")
```

### 8.3 CrewAI

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Tools
@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool("Python Code Executor")
def run_python(code: str) -> str:
    """Execute Python code and return output."""
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {})
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    finally:
        sys.stdout = old_stdout
    return output

# Agents
lead_researcher = Agent(
    role="Lead Research Analyst",
    goal="Conduct thorough research and extract key insights",
    backstory="""You are an expert research analyst with 10 years of experience 
    in AI/ML. You excel at synthesizing complex information from multiple sources.""",
    tools=[web_search],
    llm=llm,
    verbose=True,
    allow_delegation=True,
    max_iter=10
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Analyze data and create visualizations",
    backstory="Expert in Python data analysis, pandas, and matplotlib.",
    tools=[run_python],
    llm=llm,
    verbose=True
)

technical_writer = Agent(
    role="Technical Writer",
    goal="Create clear, comprehensive technical documentation",
    backstory="Expert technical writer with background in computer science.",
    llm=llm,
    verbose=True
)

# Tasks
research_task = Task(
    description="""Research the current state of LLM agents:
    1. Find the top 5 frameworks
    2. Compare capabilities
    3. Find recent papers (2024-2025)
    
    Provide a structured summary.""",
    expected_output="A structured summary with key findings and comparisons.",
    agent=lead_researcher
)

analysis_task = Task(
    description="""Based on the research, create a Python script that:
    1. Creates a comparison table of frameworks
    2. Plots a capability comparison bar chart
    3. Exports results to CSV""",
    expected_output="Working Python code with visualizations.",
    agent=data_scientist,
    context=[research_task]  # Depends on research
)

report_task = Task(
    description="""Write a comprehensive technical report based on the research and analysis.
    Include: executive summary, framework comparisons, use case recommendations.""",
    expected_output="A 1000-word technical report in markdown format.",
    agent=technical_writer,
    context=[research_task, analysis_task]
)

# Crew
crew = Crew(
    agents=[lead_researcher, data_scientist, technical_writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,
    verbose=True,
    memory=True,  # Enable crew memory
    embedder={
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
)

result = crew.kickoff()
```

---

## 9. Building a Full Agent from Scratch

Here is a complete, production-grade agent implementation:

```python
import os
import json
import time
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
import chromadb

# ============================================================
# TOOL SYSTEM
# ============================================================

@dataclass
class ToolDefinition:
    name: str
    description: str
    function: Callable
    parameters_schema: Dict

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
    
    def register(self, name: str, description: str, schema: Dict):
        """Decorator to register a tool"""
        def decorator(fn: Callable) -> Callable:
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                function=fn,
                parameters_schema=schema
            )
            return fn
        return decorator
    
    def call(self, name: str, args: Dict) -> Tuple[bool, str]:
        if name not in self._tools:
            return False, f"Tool '{name}' not found. Available: {list(self._tools.keys())}"
        try:
            result = self._tools[name].function(**args)
            return True, str(result)
        except Exception as e:
            return False, f"Tool error: {e}"
    
    def to_openai_format(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema
                }
            }
            for t in self._tools.values()
        ]

# ============================================================
# MEMORY SYSTEM
# ============================================================

class AgentMemory:
    def __init__(self, openai_client: OpenAI, persist_path: str = "./agent_memory"):
        self.client = openai_client
        self.chroma = chromadb.PersistentClient(path=persist_path)
        self.collection = self.chroma.get_or_create_collection("memories")
        self.short_term: List[Dict] = []  # Conversation messages
    
    def add_to_short_term(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        # Keep last 20 messages
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]
    
    def save_to_long_term(self, text: str, metadata: Dict = None):
        embedding = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())],
            metadatas=[metadata or {}]
        )
    
    def recall(self, query: str, n: int = 3) -> List[str]:
        embedding = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        
        results = self.collection.query(query_embeddings=[embedding], n_results=n)
        return results["documents"][0] if results["documents"] else []
    
    def get_relevant_context(self, query: str) -> str:
        memories = self.recall(query)
        if not memories:
            return ""
        return "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)

# ============================================================
# MAIN AGENT
# ============================================================

class ProductionAgent:
    """
    Full-featured production agent with:
    - OpenAI function calling
    - Vector memory
    - Structured logging
    - Cost tracking
    - Error recovery
    """
    
    SYSTEM_TEMPLATE = """You are {name}, a helpful AI agent.

{persona}

You have access to tools to help complete tasks. Think step by step.
When you have enough information, provide a clear, complete answer.

{memory_context}"""
    
    def __init__(
        self,
        name: str,
        persona: str,
        tools: ToolRegistry,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_iterations: int = 15,
        verbose: bool = True
    ):
        self.name = name
        self.persona = persona
        self.tools = tools
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.client = OpenAI()
        self.memory = AgentMemory(self.client)
        
        # Metrics
        self.total_tokens = 0
        self.total_cost = 0.0
        self.iteration_count = 0
        
        # Logger
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(f"agent.{name}")
    
    def _build_system_message(self, query: str) -> str:
        memory_context = self.memory.get_relevant_context(query)
        return self.SYSTEM_TEMPLATE.format(
            name=self.name,
            persona=self.persona,
            memory_context=memory_context
        )
    
    def _track_usage(self, usage):
        """Track token usage and cost"""
        if usage:
            self.total_tokens += usage.total_tokens
            # GPT-4o pricing (approximate)
            cost = (usage.prompt_tokens * 5 + usage.completion_tokens * 15) / 1_000_000
            self.total_cost += cost
    
    def run(self, query: str) -> str:
        """Execute agent loop for a query"""
        self.iteration_count = 0
        session_id = str(uuid.uuid4())[:8]
        
        self.logger.info(f"[{session_id}] Starting query: {query[:100]}...")
        
        # Build initial messages
        messages = [
            {"role": "system", "content": self._build_system_message(query)},
        ]
        
        # Add conversation history
        messages.extend(self.memory.short_term)
        messages.append({"role": "user", "content": query})
        
        openai_tools = self.tools.to_openai_format()
        
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
                temperature=self.temperature,
                max_tokens=4096
            )
            
            self._track_usage(response.usage)
            
            msg = response.choices[0].message
            messages.append(msg)
            
            # No tool calls — agent is done
            if not msg.tool_calls:
                final_answer = msg.content
                
                # Store in memory
                self.memory.add_to_short_term("user", query)
                self.memory.add_to_short_term("assistant", final_answer)
                self.memory.save_to_long_term(
                    f"Q: {query}\nA: {final_answer}",
                    metadata={"session": session_id, "type": "qa"}
                )
                
                self.logger.info(
                    f"[{session_id}] Completed in {iteration+1} iterations. "
                    f"Tokens: {self.total_tokens}. Cost: ${self.total_cost:.4f}"
                )
                
                return final_answer
            
            # Process tool calls
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                self.logger.info(f"[{session_id}] Calling tool: {tool_name}({tool_args})")
                
                success, result = self.tools.call(tool_name, tool_args)
                
                if not success:
                    self.logger.warning(f"Tool failed: {result}")
                
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })
        
        return f"[Agent reached max iterations ({self.max_iterations}). Last response incomplete.]"
    
    def get_stats(self) -> Dict:
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "last_iteration_count": self.iteration_count
        }

# ============================================================
# EXAMPLE USAGE
# ============================================================

def build_research_agent():
    """Build a research agent with tools"""
    
    registry = ToolRegistry()
    
    @registry.register(
        name="web_search",
        description="Search the web for current information about any topic",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    )
    def web_search(query: str, num_results: int = 5) -> str:
        # In practice: call Serper, SerpAPI, or DuckDuckGo
        return f"[Search results for '{query}']: Top results about this topic..."
    
    @registry.register(
        name="execute_python",
        description="Execute Python code and return the output",
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    )
    def execute_python(code: str) -> str:
        import io, sys
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer
        try:
            exec(code, {"__builtins__": __builtins__})
            return buffer.getvalue() or "Code executed successfully (no output)"
        except Exception as e:
            return f"Error: {e}"
        finally:
            sys.stdout = old_stdout
    
    @registry.register(
        name="read_file",
        description="Read contents of a file",
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"}
            },
            "required": ["path"]
        }
    )
    def read_file(path: str) -> str:
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            return f"File not found: {path}"
    
    agent = ProductionAgent(
        name="ResearchBot",
        persona="You are an expert research assistant specializing in AI and technology. "
                "You provide accurate, well-sourced, and comprehensive information.",
        tools=registry,
        model="gpt-4o",
        max_iterations=15,
        verbose=True
    )
    
    return agent

# Run the agent
agent = build_research_agent()
result = agent.run("Research the key differences between LangChain and LlamaIndex for building agents.")
print("\n" + "="*60)
print("FINAL ANSWER:")
print(result)
print("\nStats:", agent.get_stats())
```

---

## 10. Agent Memory with Vector Stores

### 10.1 Complete RAG-Augmented Agent

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Load and index documents
def build_knowledge_base(urls: List[str], pdf_paths: List[str] = None):
    """Build a vector knowledge base from web pages and PDFs"""
    all_docs = []
    
    for url in urls:
        loader = WebBaseLoader(url)
        all_docs.extend(loader.load())
    
    if pdf_paths:
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    
    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./knowledge_base"
    )
    
    return vectorstore

# 2. Create retrieval tool
def create_retrieval_tool(vectorstore, name: str, description: str) -> Tool:
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    def run_qa(query: str) -> str:
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        return f"{answer}\n\nSources: {', '.join(set(sources))}"
    
    return Tool(name=name, func=run_qa, description=description)

# 3. Memory-augmented agent
class RAGAgent:
    """Agent with RAG knowledge base and episodic memory"""
    
    def __init__(self, knowledge_base_urls: List[str]):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Build knowledge base
        vectorstore = build_knowledge_base(knowledge_base_urls)
        
        # Tools
        rag_tool = create_retrieval_tool(
            vectorstore,
            name="knowledge_base",
            description="Search the knowledge base for relevant information"
        )
        
        # Agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant.
Use the knowledge base to answer questions accurately.
Always cite sources when you can."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        # Build agent
        self.agent = create_tool_calling_agent(self.llm, [rag_tool], prompt)
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=[rag_tool],
            verbose=True,
            max_iterations=5
        )
        self.chat_history = []
    
    def chat(self, message: str) -> str:
        result = self.executor.invoke({
            "input": message,
            "chat_history": self.chat_history
        })
        
        self.chat_history.extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["output"]}
        ])
        
        return result["output"]
```

---

## 11. Code Agents and Computer Use Agents

### 11.1 Code Agent

```python
import subprocess
import tempfile
import os

class CodeAgent:
    """Agent that writes, executes, and iterates on code"""
    
    def __init__(self, llm_client, max_debug_iterations: int = 5):
        self.client = llm_client
        self.max_debug = max_debug_iterations
    
    def write_code(self, task: str, language: str = "python", context: str = "") -> str:
        """Generate code for a task"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": f"You are an expert {language} programmer. Write clean, correct code."
            }, {
                "role": "user",
                "content": f"Task: {task}\nContext: {context}\n\nWrite the complete code. Include all imports."
            }],
            temperature=0.2
        )
        
        content = response.choices[0].message.content
        
        # Extract code block
        if "```" in content:
            code_match = re.search(r"```(?:python|py)?\n(.*?)```", content, re.DOTALL)
            if code_match:
                return code_match.group(1)
        
        return content
    
    def execute_code(self, code: str, timeout: int = 30) -> Tuple[bool, str]:
        """Execute code safely in subprocess"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp_path = f.name
        
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        except subprocess.TimeoutExpired:
            return False, f"Code execution timed out after {timeout}s"
        except Exception as e:
            return False, f"Execution error: {e}"
        finally:
            os.unlink(tmp_path)
    
    def debug_code(self, task: str, code: str, error: str) -> str:
        """Debug code given an error"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Task: {task}

Code:
```python
{code}
```

Error:
{error}

Fix the code. Explain the bug briefly, then provide the corrected code:"""
            }],
            temperature=0
        )
        
        content = response.choices[0].message.content
        code_match = re.search(r"```(?:python|py)?\n(.*?)```", content, re.DOTALL)
        return code_match.group(1) if code_match else code
    
    def solve(self, task: str, language: str = "python") -> Dict:
        """Write and iterate on code until it works"""
        code = self.write_code(task, language)
        
        for iteration in range(self.max_debug):
            success, output = self.execute_code(code)
            
            if success:
                return {
                    "success": True,
                    "code": code,
                    "output": output,
                    "iterations": iteration + 1
                }
            
            print(f"Iteration {iteration + 1} failed. Debugging...")
            code = self.debug_code(task, code, output)
        
        return {
            "success": False,
            "code": code,
            "error": "Max debug iterations reached",
            "iterations": self.max_debug
        }

# Example
client = OpenAI()
code_agent = CodeAgent(client)
result = code_agent.solve(
    "Write a function that finds all prime numbers up to n using the Sieve of Eratosthenes, "
    "then plot the prime counting function π(x) vs x for x up to 1000"
)
print(f"Success: {result['success']}")
print(f"Code:\n{result['code']}")
print(f"Output:\n{result['output']}")
```

### 11.2 Computer Use Agent (Browser)

```python
from playwright.async_api import async_playwright
import base64
import asyncio

class BrowserAgent:
    """Agent that controls a browser to accomplish web tasks"""
    
    def __init__(self, llm_client):
        self.client = llm_client
    
    async def take_screenshot(self, page) -> str:
        """Take screenshot and encode as base64"""
        screenshot = await page.screenshot()
        return base64.b64encode(screenshot).decode()
    
    async def analyze_page(self, page, task: str) -> Dict:
        """Use vision model to analyze page and determine next action"""
        screenshot_b64 = await self.take_screenshot(page)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are controlling a browser. Current task: {task}
                        
Analyze this screenshot and determine the next action.
Available actions:
- click: {{action: "click", selector: "CSS selector or text"}}
- type: {{action: "type", selector: "CSS selector", text: "text to type"}}
- navigate: {{action: "navigate", url: "https://..."}}
- scroll: {{action: "scroll", direction: "up/down", amount: 500}}
- done: {{action: "done", result: "final answer"}}

Respond with a JSON action:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}"
                        }
                    }
                ]
            }],
            temperature=0
        )
        
        content = response.choices[0].message.content
        try:
            action = json.loads(re.search(r"\{.*\}", content, re.DOTALL).group())
            return action
        except Exception:
            return {"action": "done", "result": content}
    
    async def execute_action(self, page, action: Dict) -> bool:
        """Execute a browser action"""
        action_type = action.get("action")
        
        if action_type == "click":
            selector = action.get("selector", "")
            try:
                await page.click(selector, timeout=5000)
            except Exception:
                await page.get_by_text(selector).click()
        
        elif action_type == "type":
            await page.fill(action.get("selector", ""), action.get("text", ""))
        
        elif action_type == "navigate":
            await page.goto(action.get("url", ""))
        
        elif action_type == "scroll":
            direction = action.get("direction", "down")
            amount = action.get("amount", 300)
            y = amount if direction == "down" else -amount
            await page.evaluate(f"window.scrollBy(0, {y})")
        
        elif action_type == "done":
            return True  # Signal completion
        
        await asyncio.sleep(1)  # Wait for page to settle
        return False
    
    async def run(self, task: str, start_url: str, max_steps: int = 20) -> str:
        """Run browser agent on a task"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto(start_url)
            
            for step in range(max_steps):
                print(f"Step {step + 1}...")
                action = await self.analyze_page(page, task)
                
                done = await self.execute_action(page, action)
                
                if done:
                    result = action.get("result", "Task completed")
                    await browser.close()
                    return result
            
            await browser.close()
            return "Max steps reached"
```

---

## 12. Evaluation

Agent evaluation is hard: agents are non-deterministic, multi-step, and tool-dependent. Metrics must capture success, efficiency, and robustness.

### 12.1 Key Benchmarks

| Benchmark | Domain | Metric | Notes |
|-----------|--------|--------|-------|
| **AgentBench** | Multi-domain | Success rate | OS, web, knowledge, game tasks |
| **GAIA** | General AI | Accuracy | Hard reasoning + tool use |
| **SWE-bench** | Software engineering | % resolved | Fix real GitHub issues |
| **WebArena** | Web navigation | Task success | Full web interaction |
| **HotpotQA** | Multi-hop QA | F1, EM | Multi-step reasoning |
| **MINT** | Tool use | Success | Multi-turn interactions |
| **τ-bench** | Retail/airline | Task completion | Customer service agents |

### 12.2 Evaluation Framework

```python
from dataclasses import dataclass
import time

@dataclass
class AgentEvalResult:
    task_id: str
    query: str
    expected: str
    actual: str
    success: bool
    score: float
    latency_ms: float
    token_count: int
    tool_calls: int
    error: Optional[str] = None

class AgentEvaluator:
    """Comprehensive agent evaluation framework"""
    
    def __init__(self, judge_llm_client, agent):
        self.judge = judge_llm_client
        self.agent = agent
    
    def evaluate_task(self, task_id: str, query: str, expected: str) -> AgentEvalResult:
        """Evaluate agent on a single task"""
        start = time.time()
        
        try:
            actual = self.agent.run(query)
            latency = (time.time() - start) * 1000
            stats = self.agent.get_stats()
            
            success = self._judge_answer(query, expected, actual)
            score = self._score_answer(query, expected, actual)
            
            return AgentEvalResult(
                task_id=task_id,
                query=query,
                expected=expected,
                actual=actual,
                success=success,
                score=score,
                latency_ms=latency,
                token_count=stats["total_tokens"],
                tool_calls=stats.get("tool_calls", 0)
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return AgentEvalResult(
                task_id=task_id,
                query=query,
                expected=expected,
                actual="",
                success=False,
                score=0.0,
                latency_ms=latency,
                token_count=0,
                tool_calls=0,
                error=str(e)
            )
    
    def _judge_answer(self, query: str, expected: str, actual: str) -> bool:
        """Use LLM as judge for answer correctness"""
        response = self.judge.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Evaluate if the actual answer correctly addresses the question.

Question: {query}
Expected answer: {expected}
Actual answer: {actual}

Does the actual answer correctly answer the question? Consider:
- Factual accuracy
- Completeness
- Relevance

Answer with only: CORRECT or INCORRECT"""
            }],
            temperature=0
        )
        return "CORRECT" in response.choices[0].message.content.upper()
    
    def _score_answer(self, query: str, expected: str, actual: str) -> float:
        """Score answer quality 0-1"""
        response = self.judge.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Score the quality of the actual answer vs expected.

Question: {query}
Expected: {expected}
Actual: {actual}

Score from 0.0 to 1.0 where:
1.0 = Perfect answer
0.7 = Good but incomplete
0.4 = Partially correct
0.1 = Mostly wrong
0.0 = Completely wrong or irrelevant

Return only a decimal number:"""
            }],
            temperature=0
        )
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5
    
    def evaluate_benchmark(self, test_cases: List[Dict]) -> Dict:
        """Run full benchmark evaluation"""
        results = []
        
        for case in test_cases:
            result = self.evaluate_task(
                case["id"], case["query"], case["expected"]
            )
            results.append(result)
            print(f"{'✓' if result.success else '✗'} {case['id']}: {result.score:.2f}")
        
        successful = [r for r in results if r.success]
        
        return {
            "total_tasks": len(results),
            "success_rate": len(successful) / len(results),
            "avg_score": sum(r.score for r in results) / len(results),
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
            "avg_tokens": sum(r.token_count for r in results) / len(results),
            "error_rate": sum(1 for r in results if r.error) / len(results),
            "results": results
        }
```

### 12.3 Critique-Based Evaluation

Instead of binary correct/incorrect, use a **critique** step: a judge model scores *why* an answer is good or bad. Enables:

- **Graded feedback**: Identify which sub-aspects failed (factuality vs completeness vs relevance)
- **Trajectory evaluation**: Score each step (tool choice, observation use) not just final answer
- **Fine-grained metrics**: Use critique scores for development and regression testing

```python
def critique_trajectory(query: str, trajectory: List[Dict], final_answer: str) -> Dict:
    """Critic evaluates trajectory and final answer"""
    prompt = f"""Evaluate this agent trajectory.
Query: {query}
Steps: {json.dumps(trajectory, indent=2)}
Final answer: {final_answer}

Score 0-1 for: (1) tool_selection, (2) observation_use, (3) reasoning, (4) final_correctness.
Return JSON: {{"tool_selection": 0.8, "observation_use": 0.9, ...}}"""
    return json.loads(llm.generate(prompt))
```

### 12.4 Pitfalls of LLM-as-Judge

| Pitfall | Cause | Mitigation |
|---------|-------|------------|
| **Grade inflation** | Judge tends to favor verbose, confident answers | Calibrate on labeled data; use strict rubrics |
| **Bias toward judge's style** | Judge model may prefer answers similar to its training | Use multiple judges; aggregate; human spot-checks |
| **Inconsistent grading** | Same answer scored differently across runs | Low temperature; repeat and average; use deterministic rubrics |
| **Overlooking subtle errors** | Judge agrees with plausible-but-wrong answers | Include "trick" cases; use chain-of-thought in judge |
| **Cost** | Every evaluation is an LLM call | Cache; use smaller judge; sample evaluation set |

**Best practice**: Combine LLM-as-judge with automated checks (exact match, regex, code execution) where possible. Reserve LLM judgment for nuanced correctness.

---

## 13. Production Considerations

### 13.1 Rate Limiting and Cost Control

```python
import asyncio
from asyncio import Semaphore
from functools import wraps

class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int = 60, tokens_per_minute: int = 100000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.request_semaphore = Semaphore(requests_per_minute // 6)  # Per 10s bucket
        self._request_times: List[float] = []
        self._token_counts: List[Tuple[float, int]] = []
    
    async def acquire(self, estimated_tokens: int = 1000):
        """Wait until rate limit allows the request"""
        async with self.request_semaphore:
            await self._wait_for_token_budget(estimated_tokens)
    
    async def _wait_for_token_budget(self, tokens: int):
        """Ensure we're within token per minute limit"""
        while True:
            now = time.time()
            # Remove old entries (older than 60s)
            self._token_counts = [(t, c) for t, c in self._token_counts if now - t < 60]
            current_tpm = sum(c for _, c in self._token_counts)
            
            if current_tpm + tokens <= self.tpm:
                self._token_counts.append((now, tokens))
                return
            
            await asyncio.sleep(1)

class CostController:
    """Track and limit agent spending"""
    
    def __init__(self, budget_usd: float = 10.0):
        self.budget = budget_usd
        self.spent = 0.0
        self.PRICES = {  # Per 1M tokens
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        }
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = self.PRICES.get(model, self.PRICES["gpt-4o"])
        return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    
    def charge(self, model: str, input_tokens: int, output_tokens: int) -> bool:
        """Charge for usage. Returns False if budget exceeded."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        if self.spent + cost > self.budget:
            return False
        self.spent += cost
        return True
    
    def remaining(self) -> float:
        return max(0, self.budget - self.spent)
    
    def utilization(self) -> float:
        return self.spent / self.budget
```

### 13.2 Reliability Patterns

```python
class CircuitBreaker:
    """Circuit breaker for agent tool calls"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed=normal, open=failing, half-open=testing
    
    def call(self, fn: Callable, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker OPEN — service unavailable")
        
        try:
            result = fn(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker OPENED after {self.failures} failures")
            
            raise

class AgentSupervisor:
    """Monitors and restarts stuck agents"""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timeout_seconds
    
    def run_with_timeout(self, agent, query: str) -> str:
        """Run agent with timeout"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(agent.run, query)
            try:
                return future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                return f"Agent timed out after {self.timeout}s"
```

---

## 14. Human-in-the-Loop Patterns

```python
from enum import Enum

class InterventionLevel(Enum):
    NONE = "none"               # Fully autonomous
    NOTIFY = "notify"           # Notify human, don't wait
    CONFIRM_HIGH_RISK = "confirm_high_risk"  # Wait for confirmation on risky actions
    CONFIRM_ALL = "confirm_all"  # Confirm every action
    MANUAL = "manual"           # Human executes actions

class HITLAgent:
    """Human-in-the-Loop agent with configurable intervention level"""
    
    def __init__(self, base_agent, level: InterventionLevel = InterventionLevel.CONFIRM_HIGH_RISK):
        self.agent = base_agent
        self.level = level
        self.human_approvals: List[Dict] = []
        self.high_risk_tools = {"delete_file", "send_email", "post_to_social", "payment"}
    
    def is_high_risk(self, tool_name: str, args: Dict) -> bool:
        """Determine if an action is high-risk"""
        if tool_name in self.high_risk_tools:
            return True
        
        # Check for destructive patterns in args
        args_str = json.dumps(args).lower()
        risky_patterns = ["delete", "remove", "reset", "clear", "wipe", "override"]
        return any(p in args_str for p in risky_patterns)
    
    def request_approval(self, tool_name: str, args: Dict, rationale: str) -> bool:
        """Request human approval for an action"""
        print("\n" + "="*60)
        print("🔔 HUMAN APPROVAL REQUIRED")
        print(f"Tool: {tool_name}")
        print(f"Arguments: {json.dumps(args, indent=2)}")
        print(f"Rationale: {rationale}")
        print("="*60)
        
        response = input("Approve? (yes/no/modify): ").strip().lower()
        
        if response == "yes":
            self.human_approvals.append({
                "tool": tool_name, "args": args, "approved": True,
                "timestamp": datetime.now().isoformat()
            })
            return True
        elif response == "no":
            self.human_approvals.append({
                "tool": tool_name, "args": args, "approved": False,
                "timestamp": datetime.now().isoformat()
            })
            return False
        else:
            # Allow modification
            new_args_str = input("Enter modified arguments as JSON: ")
            try:
                new_args = json.loads(new_args_str)
                self.human_approvals.append({
                    "tool": tool_name, "args": new_args, "approved": True,
                    "modified": True, "timestamp": datetime.now().isoformat()
                })
                return True
            except json.JSONDecodeError:
                return False
    
    def notify(self, event: str, details: Dict):
        """Non-blocking notification to human"""
        print(f"[NOTIFY] {event}: {json.dumps(details, indent=2)}")
    
    def run(self, query: str) -> str:
        """Run agent with HITL based on configured level"""
        if self.level == InterventionLevel.NONE:
            return self.agent.run(query)
        
        if self.level == InterventionLevel.MANUAL:
            # Present plan, human executes
            plan = self.agent.plan(query)
            print("Proposed plan:")
            for i, step in enumerate(plan, 1):
                print(f"{i}. {step}")
            input("Review the plan. Press Enter to continue or Ctrl+C to abort...")
        
        return self.agent.run(query)

# Checkpoint pattern for long-running agents
class CheckpointAgent:
    """Agent that saves checkpoints for resumption"""
    
    def __init__(self, base_agent, checkpoint_path: str = "checkpoints"):
        self.agent = base_agent
        self.checkpoint_path = checkpoint_path
        os.makedirs(checkpoint_path, exist_ok=True)
    
    def save_checkpoint(self, session_id: str, state: Dict):
        path = os.path.join(self.checkpoint_path, f"{session_id}.json")
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_checkpoint(self, session_id: str) -> Optional[Dict]:
        path = os.path.join(self.checkpoint_path, f"{session_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    
    def run_with_checkpoints(self, query: str, session_id: str = None):
        session_id = session_id or str(uuid.uuid4())[:8]
        
        checkpoint = self.load_checkpoint(session_id)
        if checkpoint:
            print(f"Resuming from checkpoint: {session_id}")
            # Restore state
        
        # Run with periodic checkpointing
        return self.agent.run(query)
```

---

## 15. Safety and Alignment

### 15.1 Prompt Injection Defense

```python
class SafeAgentWrapper:
    """Safety wrapper to defend against prompt injection"""
    
    INJECTION_PATTERNS = [
        r"ignore (all |previous |above )?instructions",
        r"forget (all |previous |your )?instructions",
        r"you are now",
        r"new system prompt",
        r"act as (a )?different",
        r"jailbreak",
        r"DAN mode",
    ]
    
    def __init__(self, base_agent, allowed_domains: List[str] = None):
        self.agent = base_agent
        self.allowed_domains = allowed_domains or []
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
    
    def is_injection_attempt(self, text: str) -> bool:
        return any(p.search(text) for p in self._compiled_patterns)
    
    def sanitize_tool_output(self, output: str) -> str:
        """Remove potential injections from tool outputs"""
        if self.is_injection_attempt(output):
            return "[Tool output sanitized — potential injection detected]"
        return output
    
    def validate_url(self, url: str) -> bool:
        """Check if URL is in allowed domains"""
        if not self.allowed_domains:
            return True
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(domain.endswith(d) for d in self.allowed_domains)
    
    def run(self, query: str) -> str:
        """Run agent with safety checks"""
        if self.is_injection_attempt(query):
            return "I cannot process this request as it appears to contain instruction manipulation."
        
        return self.agent.run(query)

class ConstitutionalAgent:
    """Agent that self-checks outputs against principles"""
    
    PRINCIPLES = [
        "Do not provide harmful or dangerous information",
        "Do not deceive or manipulate users",
        "Protect user privacy — do not expose personal data",
        "Be honest about limitations and uncertainty",
        "Do not assist with illegal activities",
    ]
    
    def __init__(self, base_agent, llm_client):
        self.agent = base_agent
        self.client = llm_client
    
    def check_output(self, query: str, output: str) -> Tuple[bool, str]:
        """Check if output violates principles"""
        principles_str = "\n".join(f"- {p}" for p in self.PRINCIPLES)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Evaluate this agent output against these principles:

Principles:
{principles_str}

User query: {query}
Agent output: {output}

Does the output violate any principles? Answer: SAFE or UNSAFE: [reason]"""
            }],
            temperature=0
        )
        
        judgment = response.choices[0].message.content
        is_safe = "SAFE" in judgment and "UNSAFE" not in judgment
        return is_safe, judgment
    
    def run(self, query: str) -> str:
        output = self.agent.run(query)
        
        is_safe, judgment = self.check_output(query, output)
        
        if not is_safe:
            # Revise output
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": f"""The following output was deemed unsafe:

Output: {output}
Reason: {judgment}

Rewrite it to be helpful while following the safety principles:"""
                }]
            )
            return response.choices[0].message.content
        
        return output
```

---

## Summary

| Component | Key Technologies |
|-----------|----------------|
| **Agent Core** | ReAct, BDI, OpenAI function calling |
| **Memory** | In-context, ChromaDB (vector), SQLite (episodic) |
| **Planning** | Hierarchical decomposition, MCTS, plan-execute-replan |
| **Multi-Agent** | Orchestrator, debate, CrewAI, AutoGen |
| **Tools** | Pydantic schemas, retry, circuit breaker, observability |
| **Safety** | HITL, prompt injection defense, constitutional AI |
| **Evaluation** | LLM-as-judge, AgentBench, SWE-bench |
| **Production** | Rate limiting, cost control, checkpointing |

### Essential Libraries

```bash
pip install openai langchain langchain-openai langchain-community
pip install crewai autogen-agentchat
pip install chromadb tiktoken pydantic
pip install playwright  # For browser agents
pip install stable-baselines3  # For RL-based agents
```

### Key Papers

- **ReAct** (Yao et al., 2023): arxiv.org/abs/2210.03629
- **Reflexion** (Shinn et al., 2023): arxiv.org/abs/2303.11366
- **AutoGPT** (Richards, 2023): github.com/Significant-Gravitas/AutoGPT
- **Plan-and-Execute** (Yao et al., 2023): *ReWOO* — arxiv.org/abs/2305.18323
- **LLM-as-Judge** (Zheng et al., 2023): *GPT-4 as Judge* — arxiv.org/abs/2306.05685
- **AgentBench** (Liu et al., 2023): arxiv.org/abs/2308.03688
- **GAIA** (Mialon et al., 2023): arxiv.org/abs/2311.12983
- **SWE-bench** (Jimenez et al., 2024): arxiv.org/abs/2310.06770
