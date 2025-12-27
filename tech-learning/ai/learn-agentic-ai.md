# Agentic AI: Frameworks and Usage Guide

## Table of Contents
1. [Introduction to Agentic AI](#introduction)
2. [Agent Architecture](#architecture)
3. [LangChain Framework](#langchain)
4. [LlamaIndex Framework](#llamaindex)
5. [AutoGPT and AgentGPT](#autogpt)
6. [CrewAI Framework](#crewai)
7. [Semantic Kernel](#semantic-kernel)
8. [Building Custom Agents](#custom-agents)
9. [Tool Integration](#tool-integration)
10. [Memory and State Management](#memory)
11. [Multi-Agent Systems](#multi-agent)
12. [Practical Examples](#examples)
13. [Best Practices](#best-practices)

---

## Introduction to Agentic AI {#introduction}

Agentic AI refers to AI systems that can autonomously perform tasks by reasoning, planning, and using tools. Unlike traditional AI that responds to prompts, agentic AI can break down complex goals into steps and execute them.

### Key Characteristics

- **Autonomy**: Can operate independently
- **Reasoning**: Can think through problems
- **Tool Use**: Can interact with external systems
- **Planning**: Can break down tasks into steps
- **Memory**: Can remember past interactions
- **Adaptability**: Can adjust behavior based on feedback

### Agent vs Traditional AI

| Traditional AI | Agentic AI |
|---------------|------------|
| Single prompt → response | Multi-step reasoning |
| Static behavior | Dynamic planning |
| No tool usage | Can use tools/APIs |
| No memory | Persistent memory |
| Human-guided | Autonomous execution |

### Use Cases

- **Research Agents**: Gather and synthesize information
- **Code Generation Agents**: Write and test code
- **Data Analysis Agents**: Analyze datasets autonomously
- **Customer Service Agents**: Handle complex queries
- **Workflow Automation**: Automate business processes

---

## Agent Architecture {#architecture}

### Core Components

```python
class Agent:
    """Basic agent architecture"""
    def __init__(self):
        self.llm = None  # Language model
        self.tools = []  # Available tools
        self.memory = None  # Memory system
        self.planner = None  # Planning system
    
    def think(self, goal):
        """Reason about how to achieve goal"""
        pass
    
    def plan(self, goal):
        """Create execution plan"""
        pass
    
    def execute(self, plan):
        """Execute plan steps"""
        pass
    
    def reflect(self, results):
        """Reflect on results and adjust"""
        pass
```

### Agent Loop

```python
def agent_loop(agent, goal, max_iterations=10):
    """Main agent execution loop"""
    plan = agent.plan(goal)
    
    for iteration in range(max_iterations):
        # Execute next step
        result = agent.execute_step(plan)
        
        # Check if goal achieved
        if agent.check_goal_achieved(goal, result):
            return result
        
        # Reflect and replan if needed
        if agent.should_replan(result):
            plan = agent.replan(goal, result)
    
    return None  # Goal not achieved
```

---

## LangChain Framework {#langchain}

### Installation

```bash
pip install langchain langchain-openai langchain-community
pip install langchain-experimental  # For advanced features
```

### Basic Agent Setup

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define tools
def search_tool(query: str) -> str:
    """Search the web for information"""
    # Implementation here
    return f"Results for: {query}"

def calculator_tool(expression: str) -> str:
    """Evaluate mathematical expressions"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error evaluating expression"

# Create tools
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search the web for information. Input should be a search query."
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Evaluate mathematical expressions. Input should be a valid Python expression."
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("What is the capital of France? Then calculate 15 * 23")
print(result)
```

### ReAct Agent Pattern

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.prompts import PromptTemplate

# Pull ReAct prompt template
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({
    "input": "Research the latest AI developments and summarize them"
})
```

### Custom Agent with Planning

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools to answer questions."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run with memory
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

result = agent_executor.invoke({
    "input": "What's the weather in New York?",
    "chat_history": memory.chat_memory.messages
})
```

### LangChain Tools

```python
from langchain.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# Web search
search = DuckDuckGoSearchRun()

# Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Python REPL
python_repl = PythonREPLTool()

# Custom tool
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    # API call here
    return f"Weather in {location}: Sunny, 72°F"

tools = [search, wikipedia, python_repl, get_weather]
```

### LangChain Memory

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory

# Buffer memory (stores all messages)
buffer_memory = ConversationBufferMemory()

# Summary memory (summarizes old messages)
summary_memory = ConversationSummaryMemory(llm=llm)

# Window memory (keeps last N messages)
window_memory = ConversationBufferWindowMemory(k=5)

# Use with agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=window_memory,
    verbose=True
)
```

### LangChain Chains

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Create chains
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief summary about {topic}"
)

prompt2 = PromptTemplate(
    input_variables=["summary"],
    template="Translate this summary to French: {summary}"
)

chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="summary")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="translation")

# Sequential chain
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["summary", "translation"],
    verbose=True
)

result = overall_chain.run("Artificial Intelligence")
```

---

## LlamaIndex Framework {#llamaindex}

### Installation

```bash
pip install llama-index llama-index-llms-openai
pip install llama-index-embeddings-openai
```

### Basic Agent Setup

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

# Initialize LLM
llm = OpenAI(model="gpt-4", temperature=0)

# Define tools
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Create tools
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# Create agent
agent = ReActAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True
)

# Run agent
response = agent.chat("What is 15 multiplied by 23?")
print(response)
```

### Query Engine Agent

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Create tool from query engine
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    description="Useful for answering questions about the documents"
)

# Create agent
agent = ReActAgent.from_tools([query_tool], llm=llm, verbose=True)

# Query
response = agent.chat("What are the main topics in the documents?")
```

### Multi-Document Agent

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# Create multiple indexes
index1 = VectorStoreIndex.from_documents(docs1)
index2 = VectorStoreIndex.from_documents(docs2)

# Create query engines
query_engine1 = index1.as_query_engine()
query_engine2 = index2.as_query_engine()

# Create tools
tool1 = QueryEngineTool.from_defaults(
    query_engine=query_engine1,
    name="documents_2023",
    description="Information about 2023"
)

tool2 = QueryEngineTool.from_defaults(
    query_engine=query_engine2,
    name="documents_2024",
    description="Information about 2024"
)

# Create agent with multiple tools
agent = ReActAgent.from_tools([tool1, tool2], llm=llm, verbose=True)

response = agent.chat("Compare 2023 and 2024 data")
```

### Custom Tools

```python
from llama_index.core.tools import FunctionTool
from typing import List

def search_database(query: str) -> List[str]:
    """Search internal database"""
    # Database query logic
    return ["result1", "result2"]

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    # Email sending logic
    return f"Email sent to {to}"

# Create tools
db_tool = FunctionTool.from_defaults(fn=search_database)
email_tool = FunctionTool.from_defaults(fn=send_email)

# Use with agent
agent = ReActAgent.from_tools([db_tool, email_tool], llm=llm)
```

---

## AutoGPT and AgentGPT {#autogpt}

### AutoGPT Concept

```python
class AutoGPTAgent:
    """AutoGPT-style autonomous agent"""
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.objectives = []
        self.completed_tasks = []
    
    def set_objective(self, objective: str):
        """Set main objective"""
        self.objectives.append(objective)
    
    def think(self, current_state):
        """Think about next action"""
        prompt = f"""
        Objective: {self.objectives[-1]}
        Current State: {current_state}
        Completed Tasks: {self.completed_tasks}
        
        What should I do next? Think step by step.
        """
        return self.llm.generate(prompt)
    
    def execute(self, action):
        """Execute action"""
        # Parse action
        # Select tool
        # Execute
        # Store result
        pass
    
    def run(self, max_iterations=10):
        """Main execution loop"""
        for i in range(max_iterations):
            thought = self.think(self.get_current_state())
            action = self.plan_action(thought)
            result = self.execute(action)
            self.completed_tasks.append((action, result))
            
            if self.check_objective_complete():
                break
```

### AgentGPT Pattern

```python
class AgentGPT:
    """AgentGPT-style agent with goal decomposition"""
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.goals = []
        self.tasks = []
    
    def decompose_goal(self, goal: str):
        """Break goal into sub-tasks"""
        prompt = f"""
        Break this goal into specific, actionable tasks:
        Goal: {goal}
        
        List tasks in order of execution.
        """
        tasks = self.llm.generate(prompt)
        return self.parse_tasks(tasks)
    
    def execute_task(self, task: str):
        """Execute a single task"""
        # Determine which tool to use
        tool = self.select_tool(task)
        result = tool.execute(task)
        return result
    
    def run(self, goal: str):
        """Execute goal"""
        self.goals.append(goal)
        tasks = self.decompose_goal(goal)
        
        for task in tasks:
            result = self.execute_task(task)
            self.tasks.append((task, result))
        
        return self.generate_summary()
```

---

## CrewAI Framework {#crewai}

### Installation

```bash
pip install crewai crewai-tools
```

### Basic Crew Setup

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Define tools
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

# Create agents
researcher = Agent(
    role='Research Analyst',
    goal='Research and gather information',
    backstory='Expert at finding and analyzing information',
    tools=[search_tool, web_tool],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Write engaging content',
    backstory='Skilled writer who creates compelling narratives',
    verbose=True
)

# Create tasks
research_task = Task(
    description='Research the latest AI trends',
    agent=researcher
)

write_task = Task(
    description='Write a blog post about AI trends',
    agent=writer,
    context=[research_task]
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True
)

# Execute
result = crew.kickoff()
print(result)
```

### Multi-Agent Collaboration

```python
from crewai import Agent, Task, Crew

# Define specialized agents
data_analyst = Agent(
    role='Data Analyst',
    goal='Analyze data and extract insights',
    backstory='Expert in statistical analysis and data interpretation',
    tools=[python_tool],
    verbose=True
)

visualization_specialist = Agent(
    role='Visualization Specialist',
    goal='Create compelling visualizations',
    backstory='Expert in data visualization and storytelling',
    tools=[plotting_tool],
    verbose=True
)

report_writer = Agent(
    role='Report Writer',
    goal='Write comprehensive reports',
    backstory='Skilled technical writer',
    verbose=True
)

# Create tasks
analysis_task = Task(
    description='Analyze the sales data and identify trends',
    agent=data_analyst
)

viz_task = Task(
    description='Create visualizations for the analysis',
    agent=visualization_specialist,
    context=[analysis_task]
)

report_task = Task(
    description='Write a comprehensive report',
    agent=report_writer,
    context=[analysis_task, viz_task]
)

# Create crew
crew = Crew(
    agents=[data_analyst, visualization_specialist, report_writer],
    tasks=[analysis_task, viz_task, report_task],
    process=Process.sequential
)

result = crew.kickoff()
```

### Custom Tools in CrewAI

```python
from crewai_tools import tool

@tool("Database Query Tool")
def query_database(query: str) -> str:
    """Query the company database"""
    # Database query logic
    return results

@tool("Email Tool")
def send_notification(email: str, message: str) -> str:
    """Send email notification"""
    # Email sending logic
    return f"Email sent to {email}"

# Use tools
agent = Agent(
    role='Data Analyst',
    tools=[query_database, send_notification],
    verbose=True
)
```

---

## Semantic Kernel {#semantic-kernel}

### Installation

```bash
pip install semantic-kernel
```

### Basic Setup

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Initialize kernel
kernel = sk.Kernel()

# Add LLM
kernel.add_chat_service(
    "chat-gpt",
    OpenAIChatCompletion("gpt-4", api_key="your-key")
)

# Create function
prompt = """
You are a helpful assistant.
User: {{$input}}
Assistant:
"""

function = kernel.create_semantic_function(prompt)

# Execute
result = await function.invoke_async("What is AI?")
print(result)
```

### Plugins and Functions

```python
# Create plugin
plugin = kernel.create_plugin("MyPlugin")

# Add function to plugin
@plugin.function(
    description="Multiply two numbers",
    name="multiply"
)
def multiply(a: int, b: int) -> int:
    return a * b

# Use plugin
result = await kernel.run_async(
    plugin["multiply"],
    input_vars={"a": 5, "b": 3}
)
```

---

## Building Custom Agents {#custom-agts}

### Simple Custom Agent

```python
class CustomAgent:
    def __init__(self, llm, tools, memory=None):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory or []
        self.max_iterations = 10
    
    def think(self, goal: str, context: str = "") -> dict:
        """Generate thought and action"""
        prompt = f"""
        Goal: {goal}
        Context: {context}
        Available Tools: {list(self.tools.keys())}
        
        Think about the next step and choose an action.
        Format: THOUGHT: <your thought>
                 ACTION: <tool_name>
                 ACTION_INPUT: <input>
        """
        response = self.llm.generate(prompt)
        return self.parse_response(response)
    
    def execute_action(self, action: dict) -> str:
        """Execute action using tool"""
        tool_name = action['action']
        tool_input = action['action_input']
        
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            return tool.execute(tool_input)
        else:
            return f"Tool {tool_name} not found"
    
    def run(self, goal: str) -> str:
        """Main execution loop"""
        context = ""
        
        for i in range(self.max_iterations):
            # Think
            thought_action = self.think(goal, context)
            
            # Execute
            if thought_action['action'] == 'FINISH':
                return thought_action.get('action_input', 'Done')
            
            result = self.execute_action(thought_action)
            
            # Update context
            context += f"\nAction: {thought_action['action']}\nResult: {result}"
            self.memory.append((thought_action, result))
        
        return "Max iterations reached"
```

### Agent with Planning

```python
class PlanningAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.plan = []
    
    def create_plan(self, goal: str) -> list:
        """Create execution plan"""
        prompt = f"""
        Goal: {goal}
        Available Tools: {[t.name for t in self.tools]}
        
        Create a step-by-step plan to achieve this goal.
        List each step clearly.
        """
        response = self.llm.generate(prompt)
        self.plan = self.parse_plan(response)
        return self.plan
    
    def execute_plan(self) -> str:
        """Execute plan step by step"""
        results = []
        
        for step in self.plan:
            # Determine tool for step
            tool = self.select_tool_for_step(step)
            result = tool.execute(step)
            results.append(result)
            
            # Check if replanning needed
            if self.should_replan(result):
                self.replan(results)
        
        return self.summarize_results(results)
```

---

## Tool Integration {#tool-integration}

### Creating Custom Tools

```python
from typing import Type
from pydantic import BaseModel, Field

class ToolInput(BaseModel):
    """Input schema for tool"""
    query: str = Field(description="Search query")

class CustomTool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
        self.input_schema = ToolInput
    
    def execute(self, input_data: dict) -> str:
        """Execute tool"""
        validated_input = self.input_schema(**input_data)
        return self.func(validated_input.query)
    
    def __call__(self, **kwargs):
        return self.execute(kwargs)

# Usage
def search_function(query: str) -> str:
    return f"Searching for: {query}"

search_tool = CustomTool(
    name="search",
    description="Search the web",
    func=search_function
)

result = search_tool.execute({"query": "Python"})
```

### API Integration Tools

```python
import requests
from typing import Dict, Any

class APITool:
    def __init__(self, name: str, base_url: str, endpoints: Dict[str, str]):
        self.name = name
        self.base_url = base_url
        self.endpoints = endpoints
    
    def call_endpoint(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Call API endpoint"""
        url = f"{self.base_url}{self.endpoints[endpoint]}"
        response = requests.get(url, params=params)
        return response.json()
    
    def execute(self, endpoint: str, params: Dict = None) -> str:
        """Execute API call"""
        try:
            result = self.call_endpoint(endpoint, params)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

# Example: Weather API
weather_tool = APITool(
    name="weather",
    base_url="https://api.weather.com",
    endpoints={
        "current": "/v1/current",
        "forecast": "/v1/forecast"
    }
)

weather = weather_tool.execute("current", {"location": "New York"})
```

### Database Tools

```python
import sqlite3
from typing import List, Dict

class DatabaseTool:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute SQL query"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            conn.close()
            raise Exception(f"Query error: {str(e)}")
    
    def execute(self, query: str) -> str:
        """Execute and format results"""
        results = self.execute_query(query)
        return str(results)

db_tool = DatabaseTool("data.db")
results = db_tool.execute("SELECT * FROM users LIMIT 10")
```

---

## Memory and State Management {#memory}

### Conversation Memory

```python
class ConversationMemory:
    def __init__(self, max_messages=100):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """Add message to memory"""
        self.messages.append({"role": role, "content": content})
        
        # Keep only last N messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self, n_messages: int = 10) -> str:
        """Get recent context"""
        recent = self.messages[-n_messages:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    
    def clear(self):
        """Clear memory"""
        self.messages = []

# Usage
memory = ConversationMemory()
memory.add_message("user", "What is Python?")
memory.add_message("assistant", "Python is a programming language...")
context = memory.get_context()
```

### Long-term Memory

```python
import json
from datetime import datetime

class LongTermMemory:
    def __init__(self, storage_path: str = "memory.json"):
        self.storage_path = storage_path
        self.memories = self.load()
    
    def load(self) -> dict:
        """Load memories from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save(self):
        """Save memories to storage"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def store(self, key: str, value: any, metadata: dict = None):
        """Store memory"""
        self.memories[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.save()
    
    def retrieve(self, key: str) -> any:
        """Retrieve memory"""
        return self.memories.get(key, {}).get("value")
    
    def search(self, query: str) -> list:
        """Search memories"""
        results = []
        for key, memory in self.memories.items():
            if query.lower() in str(memory['value']).lower():
                results.append((key, memory))
        return results

# Usage
ltm = LongTermMemory()
ltm.store("user_preference", "prefers dark mode")
preference = ltm.retrieve("user_preference")
```

---

## Multi-Agent Systems {#multi-agent}

### Agent Communication

```python
class MultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.message_queue = []
    
    def register_agent(self, name: str, agent):
        """Register agent"""
        self.agents[name] = agent
    
    def send_message(self, from_agent: str, to_agent: str, message: str):
        """Send message between agents"""
        self.message_queue.append({
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": datetime.now()
        })
    
    def process_messages(self, agent_name: str):
        """Process messages for agent"""
        messages = [m for m in self.message_queue if m["to"] == agent_name]
        return messages

# Usage
system = MultiAgentSystem()
system.register_agent("researcher", researcher_agent)
system.register_agent("writer", writer_agent)

system.send_message("researcher", "writer", "Research complete. Here are findings...")
messages = system.process_messages("writer")
```

### Hierarchical Agents

```python
class ManagerAgent:
    """Manages subordinate agents"""
    def __init__(self, subordinates: list):
        self.subordinates = subordinates
        self.task_queue = []
    
    def delegate(self, task: str, agent_name: str):
        """Delegate task to subordinate"""
        agent = next(a for a in self.subordinates if a.name == agent_name)
        return agent.execute(task)
    
    def coordinate(self, goal: str):
        """Coordinate multiple agents"""
        # Break goal into tasks
        tasks = self.decompose_goal(goal)
        
        # Assign tasks to agents
        results = []
        for task, agent_name in tasks:
            result = self.delegate(task, agent_name)
            results.append(result)
        
        # Synthesize results
        return self.synthesize(results)
```

---

## Practical Examples {#examples}

### Example 1: Research Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Research topic
result = agent.run(
    "Research the latest developments in quantum computing. "
    "Find information from multiple sources and summarize key findings."
)
print(result)
```

### Example 2: Code Generation Agent

```python
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import initialize_agent

python_tool = PythonREPLTool()

agent = initialize_agent(
    tools=[python_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(
    "Write a Python function to calculate Fibonacci numbers. "
    "Test it with n=10 and print the result."
)
```

### Example 3: Data Analysis Agent

```python
from langchain.tools import Tool
import pandas as pd

def analyze_data(query: str) -> str:
    """Analyze dataset"""
    df = pd.read_csv("data.csv")
    
    if "summary" in query.lower():
        return df.describe().to_string()
    elif "columns" in query.lower():
        return str(df.columns.tolist())
    elif "shape" in query.lower():
        return f"Shape: {df.shape}"
    else:
        return "Available operations: summary, columns, shape"

analysis_tool = Tool(
    name="DataAnalysis",
    func=analyze_data,
    description="Analyze CSV data. Use keywords: summary, columns, shape"
)

agent = initialize_agent(
    tools=[analysis_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Analyze the data.csv file and give me a summary")
```

### Example 4: Customer Service Agent

```python
class CustomerServiceAgent:
    def __init__(self, llm, db_tool, email_tool):
        self.llm = llm
        self.db_tool = db_tool
        self.email_tool = email_tool
    
    def handle_query(self, customer_id: str, query: str):
        """Handle customer query"""
        # Get customer info
        customer_info = self.db_tool.execute(
            f"SELECT * FROM customers WHERE id='{customer_id}'"
        )
        
        # Process query
        response = self.llm.generate(
            f"Customer Info: {customer_info}\nQuery: {query}\nGenerate helpful response"
        )
        
        # Send response
        self.email_tool.execute({
            "to": customer_info['email'],
            "subject": "Response to your query",
            "body": response
        })
        
        return response
```

---

## Best Practices {#best-practices}

### 1. Error Handling

```python
class RobustAgent:
    def execute_with_retry(self, action, max_retries=3):
        """Execute action with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.execute(action)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
```

### 2. Validation

```python
def validate_tool_input(tool, input_data):
    """Validate tool input"""
    if not hasattr(tool, 'input_schema'):
        return True
    
    try:
        tool.input_schema(**input_data)
        return True
    except Exception as e:
        return False, str(e)
```

### 3. Monitoring

```python
class MonitoredAgent:
    def __init__(self, agent):
        self.agent = agent
        self.metrics = {
            "calls": 0,
            "errors": 0,
            "avg_response_time": 0
        }
    
    def run(self, *args, **kwargs):
        start_time = time.time()
        self.metrics["calls"] += 1
        
        try:
            result = self.agent.run(*args, **kwargs)
            elapsed = time.time() - start_time
            self.update_metrics(elapsed)
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise
```

### 4. Security

```python
class SecureAgent:
    def __init__(self, allowed_tools: list):
        self.allowed_tools = set(allowed_tools)
    
    def execute_tool(self, tool_name: str, input_data: dict):
        """Execute tool with security check"""
        if tool_name not in self.allowed_tools:
            raise SecurityError(f"Tool {tool_name} not allowed")
        
        # Sanitize input
        sanitized_input = self.sanitize_input(input_data)
        
        # Execute
        return self.tools[tool_name].execute(sanitized_input)
```

---

## Resources

- **LangChain**: langchain.com
- **LlamaIndex**: llamaindex.ai
- **CrewAI**: crewai.io
- **Semantic Kernel**: github.com/microsoft/semantic-kernel
- **AutoGPT**: github.com/Significant-Gravitas/AutoGPT

---

## Conclusion

Agentic AI enables autonomous task execution through reasoning and tool use. Key takeaways:

1. **Choose Right Framework**: Match framework to use case
2. **Design Tools Carefully**: Well-designed tools are crucial
3. **Manage Memory**: Implement proper memory systems
4. **Handle Errors**: Robust error handling is essential
5. **Monitor Performance**: Track agent behavior and metrics

Remember: Agentic AI is powerful but requires careful design and monitoring!

