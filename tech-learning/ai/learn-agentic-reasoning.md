# Agentic Reasoning: ReAct, Tree-of-Thought, and Beyond

## Table of Contents
1. [Introduction to Agentic Reasoning](#introduction-to-agentic-reasoning)
2. [Chain-of-Thought (CoT)](#chain-of-thought-cot)
3. [ReAct: Reasoning + Acting](#react-reasoning--acting)
4. [Tree-of-Thought (ToT)](#tree-of-thought-tot)
5. [Graph of Thoughts (GoT)](#graph-of-thoughts-got)
6. [Reflexion](#reflexion)
7. [Plan-and-Execute](#plan-and-execute)
8. [Self-Consistency and Majority Voting](#self-consistency-and-majority-voting)
9. [Agent Evaluation](#agent-evaluation)
10. [Practical Examples](#practical-examples)
11. [Best Practices](#best-practices)

---

## Introduction to Agentic Reasoning

**Agentic reasoning** refers to how AI agents structure thought and action: step-by-step reasoning, planning, tool use, and self-correction.

### Evolution

| Approach | Description | Limitation |
|----------|-------------|-------------|
| **Direct** | Single prompt → answer | No decomposition |
| **CoT** | Reason step-by-step | No tools, no branching |
| **ReAct** | Interleave reasoning + actions | Linear only |
| **ToT** | Explore multiple reasoning paths | Tree search |
| **Reflexion** | Learn from failures | Requires memory |

---

## Chain-of-Thought (CoT)

**Chain-of-Thought** prompting elicits step-by-step reasoning before the final answer.

### Zero-Shot CoT

Add "Let's think step by step" to prompt:

```python
prompt = """
Q: Roger has 5 tennis balls. He buys 2 cans of tennis balls. Each can has 3 balls. How many does he have?

A: Let's think step by step.
"""
```

### Few-Shot CoT

Provide examples with reasoning:

```python
cot_examples = """
Q: The cafeteria had 23 apples. They used 20. How many remain?
A: They had 23. Used 20. So 23 - 20 = 3. The answer is 3.

Q: Leah had 32 chocolates. She ate 20. How many remain?
A: She had 32. Ate 20. So 32 - 20 = 12. The answer is 12.

Q: {user_question}
A: Let's think step by step.
"""
```

### Structured CoT

```python
template = """
Problem: {problem}

Steps:
1. [First step]
2. [Second step]
...
n. [Conclusion]

Final answer: {answer}
"""
```

---

## ReAct: Reasoning + Acting

**ReAct** (Yao et al., 2023) interleaves **Thought** (reasoning), **Action** (tool call), and **Observation** (tool result).

### ReAct Loop

```
Thought: I need to find the weather in NYC
Action: search("weather New York City")
Observation: 72°F, sunny

Thought: I have the answer
Action: finish(72°F, sunny)
```

### ReAct Prompt Structure

```python
react_prompt = """
You have access to the following tools:
- search(query): Search the web
- calculator(expr): Evaluate math expression
- python(code): Run Python code

Use the format:
Thought: [reasoning]
Action: [tool_name]([arg])
Observation: [result]
... (repeat)
Thought: I now know the final answer
Action: finish([final answer])

Question: {question}
"""
```

### LangChain ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [search_tool, calculator_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You use tools. Think step by step. Format: Thought/Action/Observation."),
    ("human", "{input}\n\n{agent_scratchpad}")
])

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

result = agent_executor.invoke({"input": "What is 25 * 4 + 3? Then search for 'largest prime less than 100'"})
```

### ReAct Variants

- **ReWOO**: Separate planning (all actions) from execution
- **Reflexion**: Add self-reflection and memory of past failures

---

## Tree-of-Thought (ToT)

**Tree-of-Thought** explores multiple reasoning paths as a tree. Backtracks when a path fails.

### Key Ideas

- **State**: Partial solution (e.g., next step in math)
- **Thought**: Possible next steps
- **Evaluation**: Score states (LLM or programmatic)
- **Search**: BFS or DFS over tree

### ToT Pseudocode

```python
def tree_of_thought(problem, max_depth=5, breadth=3):
    root = State(problem)
    stack = [root]
    
    while stack:
        state = stack.pop()
        if state.is_final():
            return state.solution
        
        # Generate next thoughts
        next_states = llm.generate_steps(state, k=breadth)
        
        # Evaluate
        for s in next_states:
            s.score = evaluate(s)
            if s.score > threshold:
                stack.append(s)
    
    return best_solution_found
```

### ToT for Game of 24

```python
# Game: Use 4 numbers and +-*/ to get 24
# ToT: Each node = (numbers_used, expression_so_far)
# Child = extend expression with one operation
# Evaluate: Is 24 reached? Is it still possible?

def tot_24(numbers):
    def get_next_states(state):
        expr, remaining = state
        if len(remaining) == 0:
            return []
        next_states = []
        for op in ['+', '-', '*', '/']:
            for i, j in combinations(remaining, 2):
                new_expr = f"({expr} {op} ({i} {op} {j}))"
                new_remaining = [x for x in remaining if x not in [i, j]]
                next_states.append((new_expr, new_remaining))
        return next_states
    
    def evaluate(state):
        try:
            return 1.0 if eval(state[0]) == 24 else 0.0
        except:
            return 0.0
```

### LangChain ToT

```python
# Custom implementation with LangChain
from langchain_experimental.tot import ToTChain

tot_chain = ToTChain(
    llm=llm,
    prompt=ToTPrompt(),
    k=3  # Breadth
)
result = tot_chain.run(problem="...")
```

---

## Graph of Thoughts (GoT)

**GoT** extends ToT to a graph: thoughts can merge, loop, and have arbitrary dependencies.

### Operations

- **Generate**: Create new thought
- **Score**: Evaluate thought
- **Merge**: Combine multiple thoughts
- **Loop**: Refine thought

Useful for tasks with parallel branches and convergence.

---

## Reflexion

**Reflexion** adds self-reflection: after failure, generate a verbal critique and store it for future attempts.

### Reflexion Loop

```python
def reflexion_loop(task, max_trials=3):
    memory = []
    for trial in range(max_trials):
        solution = agent.solve(task)
        success = evaluator(task, solution)
        if success:
            return solution
        
        # Reflect on failure
        reflection = llm.generate(f"""
        Task: {task}
        Attempt: {solution}
        Result: Failed
        What went wrong? What should be done differently?
        """)
        memory.append(reflection)
        
        # Augment agent context with reflections
        agent.update_context(memory)
    return None
```

### Implementation Sketch

```python
class ReflexionAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.reflections = []
    
    def solve(self, task):
        prompt = f"Task: {task}\n"
        if self.reflections:
            prompt += "Past reflections:\n" + "\n".join(self.reflections)
        return self.llm.generate(prompt)
    
    def reflect(self, task, solution, feedback):
        r = self.llm.generate(f"Task: {task}\nSolution: {solution}\nFeedback: {feedback}\nWhat to improve?")
        self.reflections.append(r)
```

---

## Plan-and-Execute

Split into **planning** (high-level steps) and **execution** (run each step).

### Plan

```python
plan_prompt = """
Given the task: {task}
Create a step-by-step plan. Output as:
1. Step 1
2. Step 2
...
"""
plan = llm.generate(plan_prompt)
```

### Execute

```python
for step in parse_plan(plan):
    result = agent.execute_step(step)
    if result.failed:
        # Replan or retry
        pass
```

### Plan-and-Execute Agent (LangChain)

```python
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor
from langchain_experimental.plan_and_execute.plan_and_execute import load_chat_planner

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools)
agent = PlanAndExecute(planner=planner, executor=executor)
agent.run("Research the top 3 papers on LLMs in 2024 and summarize each")
```

---

## Self-Consistency and Majority Voting

### Self-Consistency

Generate multiple answers (with CoT), pick majority:

```python
def self_consistency(question, n=5):
    answers = []
    for _ in range(n):
        response = llm.generate(cot_prompt(question), temperature=0.7)
        answers.append(extract_answer(response))
    return most_common(answers)
```

### Majority Voting for Agents

Run agent multiple times; aggregate final answers (e.g., majority or best-by-criteria).

---

## Agent Evaluation

### Task Success

- **EXACT**: Output matches gold answer
- **F1**: Overlap of entities/facts

### Process Metrics

- **Steps**: Number of tool calls
- **Cost**: Token usage
- **Latency**: End-to-end time

### Benchmarks

- **HotpotQA**: Multi-hop QA
- **WebArena**: Web interaction
- **SWE-bench**: Code repos
- **GAIA**: General agentic tasks

### Evaluation Loop

```python
def evaluate_agent(agent, test_cases):
    results = []
    for case in test_cases:
        output = agent.run(case.input)
        score = scorer(case.expected, output)
        results.append(score)
    return np.mean(results)
```

---

## Practical Examples

### Example 1: ReAct for Math + Search

```python
tools = [
    Tool(name="Calculator", func=lambda x: eval(x), description="Evaluate math expression"),
    Tool(name="Search", func=search, description="Search the web")
]
agent = create_react_agent(llm, tools, prompt)
result = agent_executor.invoke({
    "input": "What is the population of the city where the Eiffel Tower is? Multiply that by 0.01."
})
```

### Example 2: CoT for Logic

```python
logic_prompt = """
Consider the following:
- All A are B
- Some B are C
Question: Can we conclude some A are C?

Let's think step by step:
"""
```

### Example 3: Reflexion for Code

```python
def code_with_reflexion(task):
    for trial in range(3):
        code = agent.generate_code(task)
        test_result = run_tests(code)
        if test_result.passed:
            return code
        reflection = llm.generate(f"Code failed: {test_result}. What's wrong?")
        agent.add_reflection(reflection)
    return None
```

---

## Best Practices

1. **Start with ReAct** for tool-using agents
2. **Use CoT** for math and logic
3. **ToT** when multiple paths matter (e.g., games, planning)
4. **Reflexion** when failures are informative
5. **Limit steps** to control cost and loops
6. **Validate** tool inputs before execution
7. **Log** reasoning traces for debugging

---

## Summary

| Method | Use Case | Key Feature |
|--------|----------|-------------|
| CoT | Reasoning tasks | Step-by-step in prompt |
| ReAct | Tool use | Thought-Action-Observation |
| ToT | Search problems | Tree of reasoning paths |
| Reflexion | Learning from failure | Store and use reflections |
| Plan-and-Execute | Complex tasks | Plan first, then execute |

**Libraries**: `langchain`, `langgraph`, `llamaindex`, `transformers`
