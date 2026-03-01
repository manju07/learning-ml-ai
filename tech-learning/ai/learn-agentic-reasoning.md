# Agentic Reasoning: From Chain-of-Thought to Test-Time Compute Scaling

## Table of Contents
1. [Introduction: The Reasoning Revolution](#1-introduction-the-reasoning-revolution)
2. [Chain-of-Thought (CoT) Prompting](#2-chain-of-thought-cot-prompting)
3. [Self-Consistency Sampling](#3-self-consistency-sampling)
4. [Least-to-Most Prompting](#4-least-to-most-prompting)
5. [Tree of Thoughts (ToT)](#5-tree-of-thoughts-tot)
6. [Program-of-Thought and PAL](#6-program-of-thought-and-pal)
7. [ReAct: Reason + Act](#7-react-reason--act)
8. [Reflexion: Self-Reflection with Memory](#8-reflexion-self-reflection-with-memory)
9. [Self-Ask with Search](#9-self-ask-with-search)
10. [Scratchpad Reasoning](#10-scratchpad-reasoning)
11. [Decomposed Prompting](#11-decomposed-prompting)
12. [Metacognitive Prompting](#12-metacognitive-prompting)
13. [Hypothetical Document Embeddings (HyDE)](#13-hypothetical-document-embeddings-hyde)
14. [Constitutional AI Reasoning](#14-constitutional-ai-reasoning)
15. [Process vs Outcome Reward Models](#15-process-vs-outcome-reward-models)
16. [Test-Time Compute Scaling](#16-test-time-compute-scaling)
17. [Verification and Checking](#17-verification-and-checking)
18. [World Models for Reasoning](#18-world-models-for-reasoning)
19. [Practical Reasoning Pipelines](#19-practical-reasoning-pipelines)
20. [Pitfalls and Common Mistakes](#20-pitfalls-and-common-mistakes)

---

## 1. Introduction: The Reasoning Revolution

Language models started as next-token predictors. A critical insight emerged: the *form* of the output dramatically affects quality. Asking a model to reason step-by-step before answering — even with the same weights — produces dramatically better results on complex tasks.

This is the core insight behind all agentic reasoning techniques: **intermediate computation improves final answers**.

### 1.1 Why Reasoning Works

From an information-theoretic perspective, the model can be seen as computing:

```
P(answer | question)                        # Direct: hard
P(answer | question, intermediate_steps)    # CoT: easier
```

The intermediate steps serve as a **computational scaffold** — they allow the model to:
1. Break ambiguous problems into unambiguous sub-problems
2. Handle working memory limitations (the model "shows its work")
3. Catch errors before committing to an answer
4. Revisit earlier reasoning with new context

### 1.2 Taxonomy of Reasoning Methods

```
Reasoning Methods
├── Prompting-Based (no extra models)
│   ├── Chain-of-Thought (CoT)
│   ├── Zero-Shot CoT
│   ├── Self-Consistency
│   ├── Tree of Thoughts (ToT)
│   ├── Least-to-Most
│   ├── Program-of-Thought / PAL
│   └── Scratchpad
├── Tool-Augmented
│   ├── ReAct
│   ├── Self-Ask with Search
│   └── PAL
├── Memory-Augmented
│   ├── Reflexion
│   └── Self-Ask
├── Model-Based (trained)
│   ├── Process Reward Models (PRM)
│   ├── Outcome Reward Models (ORM)
│   └── RLHF with reasoning
└── Search-Based
    ├── MCTS with LLM evaluator
    ├── Beam search
    └── Best-of-N sampling
```

### 1.3 Benchmarks for Reasoning Evaluation

| Benchmark | Domain | Key Challenge |
|-----------|--------|---------------|
| GSM8K | Grade school math | Multi-step arithmetic |
| MATH | Competition math | Advanced symbolic reasoning |
| ARC-Challenge | Science QA | Common-sense + science |
| StrategyQA | Multi-hop QA | Implicit reasoning chains |
| BBH (Big Bench Hard) | 23 diverse tasks | Complex reasoning |
| HumanEval | Code | Logic + syntax |
| MMLU | Multi-domain | Breadth of knowledge |
| GPQA | PhD-level science | Expert-level reasoning |

---

## 2. Chain-of-Thought (CoT) Prompting

**Chain-of-Thought** (Wei et al., 2022) is the foundational reasoning technique: prompt the model to generate reasoning steps before the final answer.

### 2.1 Standard Few-Shot CoT

The original CoT approach provides examples with reasoning chains:

```python
from openai import OpenAI

client = OpenAI()

FEW_SHOT_COT_EXAMPLES = """Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans × 3 balls/can = 6 new balls. 
5 + 6 = 11 balls. The answer is 11.

Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the 
golf balls are blue. How many blue golf balls are there?
A: There are 16 balls total. Half are golf balls: 16/2 = 8 golf balls. 
Half of the golf balls are blue: 8/2 = 4 blue golf balls. The answer is 4.

Q: {question}
A:"""

def few_shot_cot(question: str, model: str = "gpt-4o") -> str:
    prompt = FEW_SHOT_COT_EXAMPLES.format(question=question)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

result = few_shot_cot("If there are 3 cars in the parking lot and 2 more arrive, "
                      "but then 4 cars leave, how many cars are in the parking lot?")
print(result)
# A: There were 3 cars. 2 more arrived: 3+2=5 cars. Then 4 left: 5-4=1 car. The answer is 1.
```

### 2.2 Zero-Shot CoT

Kojima et al. (2022) showed that simply appending "Let's think step by step" triggers reasoning in larger models — no examples needed:

```python
ZERO_SHOT_COT_SYSTEM = """You are a careful, analytical assistant.
When solving problems:
1. Break the problem down step by step
2. Show all work explicitly
3. Check your reasoning
4. Give your final answer clearly"""

def zero_shot_cot(question: str, model: str = "gpt-4o") -> str:
    """Zero-shot CoT: trigger reasoning without examples"""
    
    # Method 1: Simple trigger phrase
    simple_cot = f"{question}\n\nLet's think step by step:"
    
    # Method 2: System + structured prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ZERO_SHOT_COT_SYSTEM},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def two_stage_zero_shot_cot(question: str, model: str = "gpt-4o") -> str:
    """
    Two-stage zero-shot CoT:
    Stage 1: Generate reasoning ("Let's think step by step")
    Stage 2: Extract final answer from reasoning
    """
    # Stage 1: Extract reasoning
    stage1_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{question}\n\nLet's think step by step."}
        ],
        temperature=0
    )
    reasoning = stage1_response.choices[0].message.content
    
    # Stage 2: Extract answer
    stage2_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{question}\n\nLet's think step by step."},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": "Therefore, the final answer is:"}
        ],
        temperature=0
    )
    
    return {
        "reasoning": reasoning,
        "answer": stage2_response.choices[0].message.content
    }
```

### 2.3 Automatic Chain-of-Thought (Auto-CoT)

Auto-CoT (Zhang et al., 2022) automatically constructs few-shot CoT examples using clustering:

```python
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict

class AutoCoT:
    """Automatically construct CoT examples via clustering"""
    
    def __init__(self, llm_client, embedding_model: str = "text-embedding-3-small"):
        self.client = llm_client
        self.embedding_model = embedding_model
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return np.array([d.embedding for d in response.data])
    
    def generate_rationale(self, question: str) -> str:
        """Generate CoT rationale for a question using zero-shot"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"{question}\n\nLet's think step by step."
            }],
            temperature=0
        )
        return response.choices[0].message.content
    
    def build_demonstrations(
        self,
        question_pool: List[str],
        n_clusters: int = 8
    ) -> List[Dict]:
        """
        Select representative examples through clustering.
        Pick the question closest to each cluster centroid.
        """
        # Embed all questions
        embeddings = self.embed_batch(question_pool)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        demonstrations = []
        
        for cluster_id in range(n_clusters):
            # Find question closest to centroid
            cluster_mask = kmeans.labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = embeddings[cluster_indices]
            
            centroid = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            
            question = question_pool[closest_idx]
            rationale = self.generate_rationale(question)
            
            demonstrations.append({
                "question": question,
                "rationale": rationale,
                "cluster": cluster_id
            })
        
        return demonstrations
    
    def build_prompt(self, demonstrations: List[Dict], test_question: str) -> str:
        """Build few-shot CoT prompt with auto-selected demonstrations"""
        examples = ""
        for demo in demonstrations:
            examples += f"Q: {demo['question']}\nA: {demo['rationale']}\n\n"
        
        return f"{examples}Q: {test_question}\nA: Let's think step by step."
    
    def solve(self, test_question: str, question_pool: List[str]) -> str:
        demonstrations = self.build_demonstrations(question_pool)
        prompt = self.build_prompt(demonstrations, test_question)
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
```

### 2.4 Structured CoT Templates

```python
MATH_COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Solution:
Step 1 - Understand: {understand}
Step 2 - Plan: {plan}
Step 3 - Execute: {execute}
Step 4 - Verify: {verify}

Final Answer: {answer}"""

LOGIC_COT_TEMPLATE = """Answer the following logic question.

Question: {question}

Reasoning:
1. What do we know? {known_facts}
2. What are we trying to find? {goal}
3. What inferences can we draw? {inferences}
4. What is the conclusion? {conclusion}

Answer: {answer}"""

def structured_cot(problem: str, template_type: str = "math") -> str:
    """Generate structured CoT reasoning"""
    
    templates = {
        "math": MATH_COT_TEMPLATE,
        "logic": LOGIC_COT_TEMPLATE
    }
    
    system = f"""Fill in the following template step by step for this problem.
Be thorough in each section."""
    
    prompt = f"""Problem: {problem}

Fill in this reasoning template (replace {{...}} with your reasoning):
{templates.get(template_type, templates['math'])}"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content
```

### 2.5 Chain-of-Thought Variants

Several prompting variants extend CoT for different use cases:

| Variant | Key Idea | When to Use |
|---------|----------|-------------|
| **Plan-and-Solve (Ps)** (Wang et al., 2023) | Explicitly separate *planning* (decompose) from *solving* (execute sub-steps) | Multi-step math, reduces mid-reasoning errors |
| **Plan-and-Solve (Ps+) with sub-questions** | Add "Let me verify each step" — self-check sub-answers | When arithmetic errors are common |
| **Decomposed prompting** | LLM first outputs sub-questions; second call answers them | Multi-hop QA, fact-based reasoning |
| **Self-refine** | Generate → Critique → Refine in a loop | Open-ended tasks, code generation |
| **ReWOO** | Decouple *planning* (full plan upfront) from *execution* (fill plan) | Tool-heavy tasks; avoids redundant tool calls |

**Plan-and-Solve (Ps) example:**
```python
PLAN_SOLVE_PROMPT = """Solve the problem in two stages.

Problem: {problem}

Stage 1 - Plan: Break down the problem into sub-problems. List them in order.
Stage 2 - Solve: Solve each sub-problem step by step, using results from previous steps.

Plan:
[Your decomposition here]

Solve:
[Your step-by-step solution]"""

def plan_and_solve(problem: str, model: str = "gpt-4o") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PLAN_SOLVE_PROMPT.format(problem=problem)}],
        temperature=0
    )
    return response.choices[0].message.content
```

**ReWOO (Reasoning With Outer Observation):** Plan all tool calls upfront as a DAG, then execute. Reduces tool-call latency by batching.

---

## 3. Self-Consistency Sampling

Self-Consistency (Wang et al., 2022) generates multiple diverse reasoning paths and takes the majority vote on the final answer. It's one of the most robust reasoning improvements — simple, effective, no additional training.

**Key insight**: Correct answers are reachable by multiple reasoning paths; wrong answers often require a specific (wrong) path.

### 3.1 Implementation

```python
from collections import Counter
import re
from typing import List, Tuple, Optional

def extract_answer(response: str) -> Optional[str]:
    """Extract the final numerical answer from a CoT response"""
    # Pattern: "The answer is X" or "= X" or just the last number
    patterns = [
        r"[Tt]he answer is[:\s]+([0-9,.$%-]+)",
        r"[Tt]herefore[,\s]+([0-9,.$%-]+)",
        r"= ([0-9,.$%-]+)$",
        r"([0-9,.$%-]+)$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response.strip())
        if match:
            # Clean up the answer
            answer = match.group(1).replace(",", "").replace("$", "").strip()
            return answer
    
    return None

def self_consistency(
    question: str,
    n_samples: int = 10,
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> Dict:
    """
    Self-consistency: sample N reasoning paths and take majority vote.
    
    Args:
        question: The question to answer
        n_samples: Number of reasoning paths to sample
        temperature: Must be > 0 for diversity
    """
    prompt = f"{question}\nLet's think step by step:"
    
    answers = []
    reasonings = []
    
    for i in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )
        
        reasoning = response.choices[0].message.content
        answer = extract_answer(reasoning)
        
        reasonings.append(reasoning)
        if answer:
            answers.append(answer)
    
    if not answers:
        return {"answer": None, "confidence": 0, "all_answers": []}
    
    # Majority vote
    counter = Counter(answers)
    majority_answer, count = counter.most_common(1)[0]
    confidence = count / n_samples
    
    return {
        "answer": majority_answer,
        "confidence": confidence,
        "vote_distribution": dict(counter),
        "all_reasonings": reasonings,
        "n_samples": n_samples
    }

# Usage
result = self_consistency(
    "A store is having a 30% off sale. If a jacket costs $120, "
    "and there's also a $10 coupon, how much will you pay?",
    n_samples=7,
    temperature=0.8
)
print(f"Answer: {result['answer']} (confidence: {result['confidence']:.0%})")
print(f"Vote distribution: {result['vote_distribution']}")
```

### 3.2 Weighted Self-Consistency

Weight answers by their reasoning quality (scored by a judge):

```python
def weighted_self_consistency(
    question: str,
    n_samples: int = 10,
    model: str = "gpt-4o"
) -> Dict:
    """Self-consistency with quality-weighted voting"""
    
    # Generate diverse samples
    responses = []
    for _ in range(n_samples):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{question}\nThink step by step:"}],
            temperature=0.8
        )
        text = resp.choices[0].message.content
        answer = extract_answer(text)
        responses.append({"reasoning": text, "answer": answer})
    
    # Score each reasoning
    scored = []
    for resp in responses:
        if not resp["answer"]:
            continue
        
        score_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"""Rate the quality of this reasoning for the question.
                
Question: {question}
Reasoning: {resp['reasoning']}

Score from 0.0 to 1.0 based on:
- Logical correctness
- Step clarity
- No errors or shortcuts

Return only a number:"""
            }],
            temperature=0
        )
        
        try:
            score = float(score_response.choices[0].message.content.strip())
        except ValueError:
            score = 0.5
        
        scored.append({"answer": resp["answer"], "score": score, "reasoning": resp["reasoning"]})
    
    if not scored:
        return {"answer": None}
    
    # Weighted vote
    weighted_votes = {}
    for s in scored:
        ans = s["answer"]
        weighted_votes[ans] = weighted_votes.get(ans, 0) + s["score"]
    
    best_answer = max(weighted_votes, key=weighted_votes.get)
    total_weight = sum(weighted_votes.values())
    
    return {
        "answer": best_answer,
        "weighted_confidence": weighted_votes[best_answer] / total_weight,
        "all_scored": scored
    }
```

---

## 4. Least-to-Most Prompting

Least-to-Most (Zhou et al., 2022) addresses complex tasks by first identifying and solving sub-problems, then using those solutions to solve the full problem. Critical for compositional generalization.

**Two-stage process**:
1. **Decomposition**: "To solve this, I first need to..."
2. **Sequential solution**: Solve sub-problems in order

```python
DECOMPOSE_PROMPT = """Break down this complex problem into simpler sub-problems.
List the sub-problems from easiest to hardest.
Each sub-problem should build on the previous ones.

Problem: {problem}

Sub-problems (list in order, starting with the most fundamental):"""

SOLVE_SUBPROBLEM_PROMPT = """You are solving a problem step by step.

Original problem: {original_problem}
Sub-problems to solve: {subproblems}
Already solved:
{solved_so_far}

Now solve the next sub-problem: {current_subproblem}

Solution:"""

FINAL_SYNTHESIS_PROMPT = """Using the sub-problem solutions below, solve the original problem.

Original problem: {original_problem}

Sub-problem solutions:
{solutions}

Final answer:"""

class LeastToMost:
    """Least-to-Most prompting: decompose then sequentially solve"""
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model
    
    def decompose(self, problem: str) -> List[str]:
        """Stage 1: Decompose problem into ordered sub-problems"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": DECOMPOSE_PROMPT.format(problem=problem)
            }],
            temperature=0
        )
        
        # Parse numbered list
        content = response.choices[0].message.content
        lines = content.strip().split("\n")
        subproblems = []
        for line in lines:
            # Remove numbering and clean
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 5:
                subproblems.append(cleaned)
        
        return subproblems
    
    def solve_sequentially(self, problem: str, subproblems: List[str]) -> List[Dict]:
        """Stage 2: Solve sub-problems in sequence, each using previous answers"""
        solved = []
        
        for i, subproblem in enumerate(subproblems):
            solved_context = "\n".join([
                f"Sub-problem {j+1}: {s['subproblem']}\nSolution: {s['solution']}"
                for j, s in enumerate(solved)
            ])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": SOLVE_SUBPROBLEM_PROMPT.format(
                        original_problem=problem,
                        subproblems="\n".join(f"{j+1}. {sp}" for j, sp in enumerate(subproblems)),
                        solved_so_far=solved_context if solved_context else "None yet",
                        current_subproblem=subproblem
                    )
                }],
                temperature=0
            )
            
            solution = response.choices[0].message.content
            solved.append({"subproblem": subproblem, "solution": solution})
        
        return solved
    
    def synthesize(self, problem: str, solved: List[Dict]) -> str:
        """Synthesize sub-problem solutions into final answer"""
        solutions_str = "\n\n".join([
            f"Sub-problem {i+1}: {s['subproblem']}\nSolution: {s['solution']}"
            for i, s in enumerate(solved)
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": FINAL_SYNTHESIS_PROMPT.format(
                    original_problem=problem,
                    solutions=solutions_str
                )
            }],
            temperature=0
        )
        return response.choices[0].message.content
    
    def solve(self, problem: str) -> Dict:
        """Full least-to-most pipeline"""
        print("Stage 1: Decomposing problem...")
        subproblems = self.decompose(problem)
        print(f"Found {len(subproblems)} sub-problems")
        
        print("Stage 2: Solving sequentially...")
        solved = self.solve_sequentially(problem, subproblems)
        
        print("Stage 3: Synthesizing final answer...")
        final_answer = self.synthesize(problem, solved)
        
        return {
            "subproblems": subproblems,
            "solutions": solved,
            "final_answer": final_answer
        }

# Example
l2m = LeastToMost(client)
result = l2m.solve(
    "A store sells notebooks for $2.50 each and pens for $0.75 each. "
    "A school bought 40 notebooks and 60 pens. They got a 15% bulk discount. "
    "How much did they pay in total?"
)
```

---

## 5. Tree of Thoughts (ToT)

Tree of Thoughts (Yao et al., 2023) generalizes CoT from a linear chain to a **tree**: the model generates multiple possible "thoughts" (next steps), evaluates them, and searches the tree via BFS, DFS, or MCTS.

### 5.1 Mathematical Formulation

Let:
- \(x\) = input problem
- \(z_1, z_2, ..., z_k\) = partial solutions (thoughts)
- \(G(x, z, n)\) = generate n next thoughts from state (x,z)
- \(V(x, Z)\) = evaluate value of thought set Z

**BFS**: Maintain top-B states at each depth  
**DFS**: Explore one branch, backtrack if poor  
**MCTS**: UCB1-guided exploration-exploitation

### 5.2 Full ToT Implementation

```python
from dataclasses import dataclass, field
from enum import Enum
import math
import random

class SearchStrategy(Enum):
    BFS = "bfs"
    DFS = "dfs"
    MCTS = "mcts"

@dataclass
class ThoughtNode:
    thought: str
    parent: Optional['ThoughtNode']
    depth: int
    value: float = 0.0
    visits: int = 0
    children: List['ThoughtNode'] = field(default_factory=list)
    is_terminal: bool = False
    
    def ucb1(self, exploration: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        exploration_term = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration_term
    
    def path(self) -> List[str]:
        """Get path from root to this node"""
        nodes = []
        node = self
        while node:
            nodes.append(node.thought)
            node = node.parent
        return list(reversed(nodes))

class TreeOfThoughts:
    """Tree of Thoughts with BFS, DFS, and MCTS search"""
    
    GENERATE_PROMPT = """Problem: {problem}

Current reasoning so far:
{current_path}

Generate {n} distinct possible next reasoning steps.
Each step should be specific and make progress toward the solution.
Number each step and separate them clearly:"""
    
    EVALUATE_PROMPT = """Problem: {problem}

Proposed reasoning path:
{path}

Evaluate this reasoning path:
- Is it making progress toward solving the problem?
- Is it logically sound?
- Is there a clear path to a solution from here?

Score from 0 to 10, where:
10 = Excellent, definitely on track
7 = Good, likely to succeed
5 = Uncertain
3 = Poor, unlikely to succeed
0 = Wrong/stuck

Respond with: SCORE: [0-10] RATIONALE: [brief explanation]"""
    
    IS_TERMINAL_PROMPT = """Problem: {problem}

Reasoning path:
{path}

Has this reasoning path reached a complete solution to the problem?
Answer YES or NO, then if YES, extract the final answer."""
    
    def __init__(
        self,
        problem: str,
        llm_client,
        model: str = "gpt-4o",
        n_thoughts: int = 3,     # Thoughts per node
        max_depth: int = 4,       # Maximum tree depth
        beam_width: int = 3,      # BFS beam width
        temperature: float = 0.8
    ):
        self.problem = problem
        self.client = llm_client
        self.model = model
        self.n_thoughts = n_thoughts
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.temperature = temperature
        self.root = ThoughtNode(
            thought=f"Problem: {problem}",
            parent=None,
            depth=0
        )
    
    def generate_thoughts(self, node: ThoughtNode) -> List[str]:
        """Generate n next thoughts from current node"""
        path = "\n".join([f"Step {i}: {t}" for i, t in enumerate(node.path())])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.GENERATE_PROMPT.format(
                    problem=self.problem,
                    current_path=path if path else "No steps yet — start fresh.",
                    n=self.n_thoughts
                )
            }],
            temperature=self.temperature
        )
        
        content = response.choices[0].message.content
        # Parse numbered thoughts
        thoughts = re.findall(r"\d+[\.\)]\s*(.+?)(?=\d+[\.\)]|\Z)", content, re.DOTALL)
        if not thoughts:
            thoughts = [t.strip() for t in content.split("\n") if t.strip()][:self.n_thoughts]
        
        return thoughts[:self.n_thoughts]
    
    def evaluate_node(self, node: ThoughtNode) -> float:
        """Evaluate the value of a node (0-1)"""
        path = "\n".join([f"Step {i}: {t}" for i, t in enumerate(node.path())])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.EVALUATE_PROMPT.format(
                    problem=self.problem,
                    path=path
                )
            }],
            temperature=0
        )
        
        content = response.choices[0].message.content
        match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", content)
        score = float(match.group(1)) / 10 if match else 0.5
        
        return min(1.0, max(0.0, score))
    
    def is_terminal(self, node: ThoughtNode) -> Tuple[bool, str]:
        """Check if node represents a complete solution"""
        if node.depth == 0:
            return False, ""
        
        path = "\n".join([f"Step {i}: {t}" for i, t in enumerate(node.path())])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.IS_TERMINAL_PROMPT.format(
                    problem=self.problem,
                    path=path
                )
            }],
            temperature=0
        )
        
        content = response.choices[0].message.content
        is_done = "YES" in content[:10].upper()
        
        return is_done, content if is_done else ""
    
    def expand(self, node: ThoughtNode):
        """Generate and evaluate children for a node"""
        thoughts = self.generate_thoughts(node)
        
        for thought in thoughts:
            child = ThoughtNode(
                thought=thought,
                parent=node,
                depth=node.depth + 1
            )
            child.value = self.evaluate_node(child)
            terminal, solution = self.is_terminal(child)
            child.is_terminal = terminal
            child.solution = solution if terminal else None
            node.children.append(child)
        
        return node.children
    
    def bfs(self) -> Optional[ThoughtNode]:
        """Breadth-First Search with beam"""
        frontier = [self.root]
        
        for depth in range(self.max_depth):
            next_frontier = []
            
            for node in frontier:
                if node.is_terminal:
                    return node
                
                children = self.expand(node)
                next_frontier.extend(children)
            
            # Keep top beam_width nodes by value
            next_frontier.sort(key=lambda n: -n.value)
            frontier = next_frontier[:self.beam_width]
            
            print(f"Depth {depth+1}: {len(frontier)} candidates, "
                  f"best score: {frontier[0].value:.2f}")
            
            # Check for terminal
            for node in frontier:
                if node.is_terminal:
                    return node
        
        # Return best node found
        return max(frontier, key=lambda n: n.value)
    
    def dfs(self, node: ThoughtNode = None, depth: int = 0) -> Optional[ThoughtNode]:
        """Depth-First Search with backtracking"""
        node = node or self.root
        
        if node.is_terminal:
            return node
        
        if depth >= self.max_depth:
            return node if node.value > 0.6 else None
        
        children = self.expand(node)
        # Sort by value descending
        children.sort(key=lambda n: -n.value)
        
        for child in children:
            if child.value < 0.3:  # Prune poor branches
                continue
            
            result = self.dfs(child, depth + 1)
            if result and (result.is_terminal or result.value > 0.8):
                return result
        
        return max(children, key=lambda n: n.value) if children else node
    
    def mcts_search(self, iterations: int = 50) -> ThoughtNode:
        """Monte Carlo Tree Search"""
        
        for iteration in range(iterations):
            # Selection
            node = self.root
            while node.children and not node.is_terminal:
                if not all(c.visits > 0 for c in node.children):
                    # Expand unvisited child
                    unvisited = [c for c in node.children if c.visits == 0]
                    node = random.choice(unvisited)
                    break
                # UCB1 selection
                node = max(node.children, key=lambda c: c.ucb1())
            
            # Expansion (if not terminal and not at max depth)
            if not node.is_terminal and node.depth < self.max_depth and not node.children:
                self.expand(node)
                if node.children:
                    node = random.choice(node.children)
            
            # Simulation/Evaluation
            value = self.evaluate_node(node)
            
            # Backpropagation
            current = node
            while current:
                current.visits += 1
                current.value = (current.value * (current.visits - 1) + value) / current.visits
                current = current.parent
        
        # Return best child of root
        if self.root.children:
            return max(self.root.children, key=lambda n: n.value / (n.visits + 1e-8))
        return self.root
    
    def solve(self, strategy: SearchStrategy = SearchStrategy.BFS) -> Dict:
        """Solve using specified search strategy"""
        print(f"Solving with {strategy.value.upper()}...")
        
        if strategy == SearchStrategy.BFS:
            best = self.bfs()
        elif strategy == SearchStrategy.DFS:
            best = self.dfs()
        else:  # MCTS
            best = self.mcts_search()
        
        return {
            "strategy": strategy.value,
            "solution_path": best.path() if best else [],
            "is_solved": best.is_terminal if best else False,
            "final_value": best.value if best else 0,
            "solution": getattr(best, "solution", None)
        }

# Example: Game of 24 (use 4 numbers with +,-,*,/ to get 24)
tot = TreeOfThoughts(
    problem="Using 4, 7, 8, and 14 with operations +, -, *, / and parentheses, make 24.",
    llm_client=client,
    n_thoughts=4,
    max_depth=5,
    beam_width=3
)
result = tot.solve(SearchStrategy.BFS)
print("Solution path:", result["solution_path"])
```

---

## 6. Program-of-Thought and PAL

**Program-of-Thought (PoT)** (Chen et al., 2022) and **Program-Aided Language Models (PAL)** (Gao et al., 2022) offload computation to a Python interpreter. Instead of computing in natural language (where LLMs make arithmetic errors), the model writes code.

**Conceptual difference:** PoT interleaves natural language reasoning with code blocks; PAL is code-only with comments. Both execute Python and return the printed result.

### 6.1 Program-of-Thought Variants

| Variant | Format | Best For |
|---------|--------|----------|
| **PoT** | NL + code blocks (e.g., "First we compute... ```python x=...```") | Mixed symbolic + textual reasoning |
| **PAL** | Pure Python with comments | Arithmetic, math competitions |
| **Chain-of-Code (CoC)** | Generate code + natural language *explanation* of each step | Debugging, teaching, verification |
| **PoT with tool use** | Code that calls APIs, runs SQL, etc. | Data analysis, ETL |
| **Multi-step PoT** | Break problem into functions; call them in sequence | Complex math (integrals, proofs) |

**Chain-of-Code** adds explicability: the model outputs both code and a step-by-step natural language walkthrough, aiding verification and learning.

### 6.2 PAL Implementation

```python
import subprocess
import tempfile
import os
import re

PAL_SYSTEM = """You are a Python programming assistant.
When given a math or reasoning problem, write a Python program to solve it.
- Use clear variable names
- Add comments explaining the logic
- Print the final answer
- Do NOT use external libraries beyond math and basic Python"""

PAL_PROMPT = """Write a Python program to solve this problem:

{problem}

```python
# Write your solution here
"""

class PAL:
    """Program-Aided Language Model"""
    
    def __init__(self, llm_client, model: str = "gpt-4o", timeout: int = 10):
        self.client = llm_client
        self.model = model
        self.timeout = timeout
    
    def generate_program(self, problem: str) -> str:
        """Generate Python program for the problem"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PAL_SYSTEM},
                {"role": "user", "content": PAL_PROMPT.format(problem=problem)}
            ],
            temperature=0
        )
        
        content = response.choices[0].message.content
        
        # Extract Python code
        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Try to find code without markdown
        code_match = re.search(r"```\n(.*?)```", content, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        return content  # Return as-is if no code block found
    
    def execute_program(self, code: str) -> Tuple[bool, str]:
        """Execute Python code safely"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp_path = f.name
        
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True, text=True, timeout=self.timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, f"Error: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, f"Execution timeout ({self.timeout}s)"
        finally:
            os.unlink(tmp_path)
    
    def solve(self, problem: str, max_attempts: int = 3) -> Dict:
        """Solve using PAL with retry on error"""
        code = None
        
        for attempt in range(max_attempts):
            if attempt == 0:
                code = self.generate_program(problem)
            else:
                # Regenerate with error context
                error_context = f"\n\nPrevious code had error: {last_error}"
                code = self.generate_program(problem + error_context)
            
            success, output = self.execute_program(code)
            
            if success:
                return {
                    "success": True,
                    "code": code,
                    "output": output,
                    "attempts": attempt + 1
                }
            
            last_error = output
        
        return {
            "success": False,
            "code": code,
            "error": last_error,
            "attempts": max_attempts
        }

# Example
pal = PAL(client)

problems = [
    "A store sells apples for $1.20 each and oranges for $0.85 each. "
    "I buy 7 apples and 13 oranges. How much do I spend in total?",
    
    "A train travels from city A to B at 60 mph, then from B to C at 80 mph. "
    "If AB is 120 miles and BC is 160 miles, how long does the total journey take?",
    
    "Find all prime numbers between 100 and 150."
]

for problem in problems:
    result = pal.solve(problem)
    print(f"\nProblem: {problem}")
    print(f"Answer: {result['output'] if result['success'] else result['error']}")
    print(f"Code:\n{result['code']}")
```

### 6.3 Program-of-Thought (Multi-Step)

PoT extends PAL to more complex, multi-step problems with planning:

```python
POT_SYSTEM = """You are an expert programmer solving complex problems with Python.
For each problem:
1. First, describe your approach in comments
2. Break the solution into clearly labeled steps  
3. Use helper functions for complex sub-computations
4. Verify your answer with an assertion or check
5. Print the final answer with clear formatting"""

def program_of_thought(problem: str, domain: str = "general") -> Dict:
    """Program-of-Thought: structured code generation"""
    
    domain_hints = {
        "math": "Use the math module if needed. Check your arithmetic.",
        "combinatorics": "Use itertools for combinations/permutations.",
        "statistics": "Use statistics module for mean, median, etc.",
        "general": "Use standard Python library only."
    }
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": POT_SYSTEM},
            {"role": "user", "content": f"""Problem: {problem}

Domain hint: {domain_hints.get(domain, domain_hints['general'])}

Write a complete, well-commented Python program:"""}
        ],
        temperature=0
    )
    
    content = response.choices[0].message.content
    code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
    code = code_match.group(1) if code_match else content
    
    pal = PAL(client)
    success, output = pal.execute_program(code)
    
    return {
        "code": code,
        "output": output,
        "success": success
    }
```

---

## 7. ReAct: Reason + Act

ReAct (Yao et al., 2023) synergizes reasoning and acting in an interleaved loop: **Thought → Action → Observation → Thought → ...**

The critical insight: real-world reasoning requires information from the environment, not just internal computation.

### 7.1 Full ReAct Framework

```python
from typing import Callable, Dict, List, Optional, Tuple
import json
import re

REACT_SYSTEM = """You are a helpful AI agent that solves problems using tools.

Think step by step using this format:
Thought: [reasoning about current situation and what to do next]
Action: [tool_name]
Action Input: [JSON input for the tool]

After receiving the observation, continue thinking and acting.
When you have enough information to answer, use:
Thought: I now have all information needed to answer.
Action: finish
Action Input: {{"answer": "your complete final answer"}}

Available tools:
{tool_descriptions}"""

class ReActAgent:
    """Full ReAct implementation with structured tool calling"""
    
    def __init__(
        self,
        tools: Dict[str, Callable],
        tool_descriptions: Dict[str, str],
        model: str = "gpt-4o",
        max_iterations: int = 10,
        temperature: float = 0
    ):
        self.tools = tools
        self.tool_descriptions = tool_descriptions
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.client = OpenAI()
    
    def _format_tool_descriptions(self) -> str:
        return "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.tool_descriptions.items()
        ] + ["- finish: Use when you have the final answer. Input: {\"answer\": \"...\"}"])
    
    def _parse_action(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Parse Action and Action Input from LLM output"""
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\]|\".*?\"|\w+)", text, re.DOTALL)
        
        if not action_match:
            return None
        
        tool_name = action_match.group(1).strip().lower()
        
        if input_match:
            raw_input = input_match.group(1).strip()
            try:
                tool_input = json.loads(raw_input)
            except json.JSONDecodeError:
                tool_input = {"input": raw_input}
        else:
            tool_input = {}
        
        return tool_name, tool_input
    
    def run(self, query: str) -> Dict:
        """Execute ReAct loop"""
        system = REACT_SYSTEM.format(
            tool_descriptions=self._format_tool_descriptions()
        )
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}"}
        ]
        
        trace = []
        
        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stop=["Observation:"]  # Stop before the observation (we provide it)
            )
            
            assistant_text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_text})
            
            # Extract thought
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", assistant_text, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # Parse action
            parsed = self._parse_action(assistant_text)
            
            if not parsed:
                trace.append({"iteration": iteration, "thought": thought, "error": "No action found"})
                break
            
            tool_name, tool_input = parsed
            
            # Handle finish
            if tool_name == "finish":
                answer = tool_input.get("answer", str(tool_input))
                trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": "finish",
                    "answer": answer
                })
                return {"answer": answer, "trace": trace, "success": True}
            
            # Execute tool
            if tool_name in self.tools:
                try:
                    observation = str(self.tools[tool_name](**tool_input))
                except Exception as e:
                    observation = f"Tool error: {e}"
            else:
                observation = f"Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
            
            trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": tool_name,
                "action_input": tool_input,
                "observation": observation[:500]  # Truncate long observations
            })
            
            # Add observation to context
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        
        return {"answer": None, "trace": trace, "success": False}

# Example tools
def calculator(expression: str) -> str:
    """Evaluate a math expression"""
    try:
        import math as _math
        safe_env = {k: v for k, v in _math.__dict__.items() if not k.startswith("_")}
        return str(eval(expression, {"__builtins__": {}}, safe_env))
    except Exception as e:
        return f"Error: {e}"

def lookup_fact(query: str) -> str:
    """Look up a fact"""
    facts = {
        "population of france": "67.4 million (2023)",
        "population of germany": "84.4 million (2023)",
        "gdp of usa": "$25.4 trillion (2022)",
        "speed of light": "299,792,458 meters per second"
    }
    query_lower = query.lower()
    for key, val in facts.items():
        if key in query_lower:
            return val
    return "Fact not found in database"

agent = ReActAgent(
    tools={"calculator": calculator, "lookup_fact": lookup_fact},
    tool_descriptions={
        "calculator": "Evaluate mathematical expressions. Input: {\"expression\": \"3*4+2\"}",
        "lookup_fact": "Look up a fact. Input: {\"query\": \"population of France\"}"
    }
)

result = agent.run("What is 15% of the population of France?")
print(f"Answer: {result['answer']}")
print("\nTrace:")
for step in result['trace']:
    print(f"  [{step['iteration']+1}] Thought: {step.get('thought', '')[:80]}...")
    if 'action' in step:
        print(f"       Action: {step['action']}({step.get('action_input', {})})")
        if 'observation' in step:
            print(f"       Observation: {step['observation'][:80]}...")
```

---

## 8. Reflexion: Self-Reflection with Memory

Reflexion (Shinn et al., 2023) enables agents to learn from mistakes without gradient updates. After failure, the agent generates a verbal self-critique, stores it in **episodic memory**, and uses it to improve on the next trial.

### 8.1 Core Architecture

```
Trial 1: Attempt → Fail → Reflect → Store reflection
Trial 2: Load reflections + Attempt (informed by reflection) → Fail → Reflect → Store
Trial 3: Load reflections + Attempt → Success
```

```python
from enum import Enum

class TaskResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"

@dataclass
class Reflection:
    trial: int
    task: str
    attempt_summary: str
    failure_reason: str
    improvement_suggestions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ReflexionAgent:
    """
    Reflexion: Self-reflective agent that learns from failures.
    Stores verbal reflections in episodic memory and uses them in future trials.
    """
    
    ACTOR_SYSTEM = """You are a problem-solving agent.
{reflections_section}
Use past reflections to avoid repeating mistakes."""
    
    REFLECTION_PROMPT = """You attempted a task and it did not fully succeed.
    
Task: {task}

Your attempt:
{attempt}

Evaluation result: {result}
Feedback: {feedback}

Generate a detailed reflection:
1. What specifically went wrong?
2. What assumptions were incorrect?
3. What information was missing?
4. What should you do differently next time?

Be specific and actionable."""
    
    def __init__(
        self,
        tools: Dict[str, Callable],
        model: str = "gpt-4o",
        max_trials: int = 4
    ):
        self.tools = tools
        self.model = model
        self.max_trials = max_trials
        self.client = OpenAI()
        self.reflections: List[Reflection] = []
    
    def _format_reflections_section(self) -> str:
        if not self.reflections:
            return ""
        
        section = "\n## IMPORTANT - Lessons from Previous Attempts:\n"
        for i, r in enumerate(self.reflections):
            section += f"\n### Trial {r.trial} Reflection:\n"
            section += f"Failure reason: {r.failure_reason}\n"
            section += "Improvements:\n"
            for suggestion in r.improvement_suggestions:
                section += f"  - {suggestion}\n"
        
        return section
    
    def _generate_attempt(self, task: str) -> str:
        """Generate a solution attempt using stored reflections"""
        reflections_section = self._format_reflections_section()
        system = self.ACTOR_SYSTEM.format(reflections_section=reflections_section)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": task}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _generate_reflection(
        self,
        task: str,
        attempt: str,
        result: str,
        feedback: str,
        trial: int
    ) -> Reflection:
        """Generate a verbal reflection on a failed attempt"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.REFLECTION_PROMPT.format(
                    task=task,
                    attempt=attempt,
                    result=result,
                    feedback=feedback
                )
            }],
            temperature=0.7
        )
        
        reflection_text = response.choices[0].message.content
        
        # Extract structured elements
        lines = reflection_text.strip().split("\n")
        failure_reason = ""
        suggestions = []
        
        current_section = None
        for line in lines:
            if "wrong" in line.lower() or "1." in line:
                current_section = "failure"
                failure_reason = line.strip("1. ")
            elif any(f"{i}." in line for i in range(2, 6)):
                suggestions.append(line.strip("2345. "))
        
        return Reflection(
            trial=trial,
            task=task,
            attempt_summary=attempt[:500],
            failure_reason=failure_reason or reflection_text[:200],
            improvement_suggestions=suggestions or [reflection_text]
        )
    
    def run(
        self,
        task: str,
        evaluator: Callable[[str, str], Tuple[bool, str]],
        verbose: bool = True
    ) -> Dict:
        """Run Reflexion loop: attempt → evaluate → reflect → retry"""
        
        for trial in range(self.max_trials):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Trial {trial + 1}/{self.max_trials}")
                print(f"{'='*50}")
                if self.reflections:
                    print(f"Using {len(self.reflections)} reflection(s)")
            
            # Generate attempt
            attempt = self._generate_attempt(task)
            
            if verbose:
                print(f"\nAttempt:\n{attempt[:300]}...")
            
            # Evaluate
            success, feedback = evaluator(task, attempt)
            
            if verbose:
                print(f"\nEvaluation: {'✓ SUCCESS' if success else '✗ FAILED'}")
                print(f"Feedback: {feedback}")
            
            if success:
                return {
                    "success": True,
                    "answer": attempt,
                    "trials": trial + 1,
                    "reflections": len(self.reflections)
                }
            
            # Generate and store reflection
            reflection = self._generate_reflection(task, attempt, "failed", feedback, trial + 1)
            self.reflections.append(reflection)
            
            if verbose:
                print(f"\nReflection: {reflection.failure_reason[:200]}...")
        
        return {
            "success": False,
            "answer": attempt,
            "trials": self.max_trials,
            "reflections": len(self.reflections),
            "final_attempt": attempt
        }

# Example: Reflexion for code generation
def code_evaluator(task: str, code: str) -> Tuple[bool, str]:
    """Evaluate code by running it"""
    try:
        exec(code, {})
        return True, "Code executed successfully"
    except Exception as e:
        return False, f"Error: {e}"

reflexion = ReflexionAgent(tools={})
result = reflexion.run(
    task="Write a Python function `is_palindrome(s)` that returns True if s is a palindrome, "
         "handling case-insensitivity and ignoring non-alphanumeric characters. "
         "Include 3 test cases.",
    evaluator=code_evaluator
)
```

---

## 9. Self-Ask with Search

Self-Ask (Press et al., 2022) has the model explicitly ask sub-questions and answer them (optionally via search) before synthesizing the final answer.

```python
SELF_ASK_PROMPT = """When answering a complex question, first break it into sub-questions.
Use this format:

Question: {question}

Are follow up questions needed here: Yes
Follow up: [sub-question 1]
Intermediate answer: [answer to sub-question 1]
Follow up: [sub-question 2]
Intermediate answer: [answer to sub-question 2]
... (continue as needed)
So the final answer is: [final answer synthesizing all sub-answers]

Or if no sub-questions are needed:
Are follow up questions needed here: No
So the final answer is: [direct answer]"""

class SelfAskAgent:
    """Self-Ask with Search: decompose questions into searchable sub-questions"""
    
    def __init__(self, search_fn: Callable, model: str = "gpt-4o"):
        self.search = search_fn
        self.model = model
        self.client = OpenAI()
    
    def run(self, question: str) -> Dict:
        """Run Self-Ask loop with search"""
        
        messages = [{
            "role": "user",
            "content": SELF_ASK_PROMPT.format(question=question)
        }]
        
        trace = []
        
        for _ in range(10):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                stop=["Intermediate answer:"]  # Stop to let us provide the answer
            )
            
            text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": text})
            
            # Check if we're done
            if "So the final answer is:" in text:
                final_match = re.search(r"So the final answer is:\s*(.+)$", text, re.DOTALL)
                final = final_match.group(1).strip() if final_match else text
                return {"answer": final, "trace": trace}
            
            # Extract follow-up question
            followup_match = re.search(r"Follow up:\s*(.+?)$", text, re.MULTILINE)
            
            if followup_match:
                sub_question = followup_match.group(1).strip()
                trace.append({"sub_question": sub_question})
                
                # Search for answer
                search_result = self.search(sub_question)
                trace[-1]["search_result"] = search_result
                
                # Provide intermediate answer
                intermediate = f"Intermediate answer: {search_result}"
                messages.append({"role": "user", "content": intermediate})
            else:
                break
        
        return {"answer": text, "trace": trace}
```

---

## 10. Scratchpad Reasoning

Scratchpad reasoning gives the model a "workspace" to compute intermediate results:

```python
SCRATCHPAD_PROMPT = """Use the scratchpad below to work through this problem.
The scratchpad is private — write freely. Only the FINAL ANSWER section is your response.

Problem: {problem}

<scratchpad>
Let me work through this carefully...

[Use this space to:
- List what's given
- Try different approaches
- Make calculations
- Check your work
- Discard wrong paths]
</scratchpad>

FINAL ANSWER:
"""

def scratchpad_reasoning(problem: str, model: str = "gpt-4o") -> Dict:
    """Use scratchpad for extended working"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SCRATCHPAD_PROMPT.format(problem=problem)}],
        temperature=0.3,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content
    
    # Extract scratchpad and final answer
    scratchpad_match = re.search(r"<scratchpad>(.*?)</scratchpad>", content, re.DOTALL)
    final_match = re.search(r"FINAL ANSWER:\s*(.+)$", content, re.DOTALL)
    
    return {
        "scratchpad": scratchpad_match.group(1).strip() if scratchpad_match else "",
        "final_answer": final_match.group(1).strip() if final_match else content
    }

# Extended scratchpad with multiple revisions
def iterative_scratchpad(problem: str, n_revisions: int = 3) -> Dict:
    """Multiple revision passes on scratchpad"""
    
    messages = [{
        "role": "user",
        "content": f"Problem: {problem}\n\nWork through this step by step:"
    }]
    
    revisions = []
    
    for i in range(n_revisions):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )
        
        working = response.choices[0].message.content
        revisions.append(working)
        messages.append({"role": "assistant", "content": working})
        
        if i < n_revisions - 1:
            messages.append({
                "role": "user",
                "content": "Review your work. Are there any errors? If so, correct them. "
                           "Then provide your final answer."
            })
    
    return {
        "revisions": revisions,
        "final": revisions[-1]
    }
```

---

## 11. Decomposed Prompting

DECOMP (Khot et al., 2022): teach the model to use specialized sub-solvers (prompt handlers) for different types of sub-tasks:

```python
class DecomposedPrompting:
    """DECOMP: Route sub-problems to specialized solvers"""
    
    DECOMPOSE_PROMPT = """Decompose this complex question into simpler sub-questions.
Label each with its type: [arithmetic], [lookup], [comparison], [logic].

Question: {question}

Sub-questions:"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI()
        
        # Specialized prompts for different sub-question types
        self.solvers = {
            "arithmetic": self._solve_arithmetic,
            "lookup": self._solve_lookup,
            "comparison": self._solve_comparison,
            "logic": self._solve_logic,
            "default": self._solve_default
        }
    
    def _solve_arithmetic(self, question: str, context: str) -> str:
        prompt = f"""Solve this arithmetic problem. Show your work.
Context: {context}
Problem: {question}
Answer (number only):"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def _solve_lookup(self, question: str, context: str) -> str:
        prompt = f"""Answer this factual question directly.
Context: {context}
Question: {question}
Answer:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def _solve_comparison(self, question: str, context: str) -> str:
        prompt = f"""Compare these items and answer the question.
Context: {context}
Question: {question}
Comparison and answer:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def _solve_logic(self, question: str, context: str) -> str:
        prompt = f"""Apply logical reasoning to answer this.
Context: {context}
Question: {question}
Reasoning and answer:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def _solve_default(self, question: str, context: str) -> str:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    
    def decompose(self, question: str) -> List[Dict]:
        """Decompose question into typed sub-questions"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.DECOMPOSE_PROMPT.format(question=question)
            }],
            temperature=0
        )
        
        # Parse sub-questions
        content = response.choices[0].message.content
        sub_questions = []
        
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            
            type_match = re.search(r"\[(arithmetic|lookup|comparison|logic)\]", line, re.IGNORECASE)
            q_type = type_match.group(1).lower() if type_match else "default"
            
            # Clean the question text
            q_text = re.sub(r"\[.*?\]", "", line).strip().strip("1234567890.-) ")
            
            if q_text:
                sub_questions.append({"type": q_type, "question": q_text})
        
        return sub_questions
    
    def solve(self, question: str) -> Dict:
        """Full DECOMP pipeline"""
        sub_questions = self.decompose(question)
        
        context = ""
        answers = []
        
        for sq in sub_questions:
            q_type = sq["type"]
            solver = self.solvers.get(q_type, self.solvers["default"])
            answer = solver(sq["question"], context)
            
            context += f"\nQ: {sq['question']}\nA: {answer}"
            answers.append({"type": q_type, "question": sq["question"], "answer": answer})
        
        # Synthesize
        synthesis_prompt = f"""Original question: {question}

Sub-question answers:
{context}

Final synthesized answer:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0
        )
        
        return {
            "sub_questions": answers,
            "final_answer": response.choices[0].message.content
        }
```

---

## 12. Metacognitive Prompting

Metacognitive prompting has the model think about *its own thinking* — monitoring confidence, identifying uncertainty, and calibrating answers.

```python
METACOGNITIVE_SYSTEM = """You are an expert with strong metacognitive abilities.
As you solve problems:
1. Monitor your own confidence at each step
2. Identify what you're uncertain about
3. Flag when you're guessing vs. certain
4. Consider alternative interpretations
5. Calibrate your final confidence"""

METACOGNITIVE_TEMPLATE = """Solve this problem with full metacognitive awareness.

Problem: {problem}

Metacognitive analysis:

Initial reaction: [What's your first impression? What domain is this?]

Knowledge check: [What do I know about this? What might I be missing?]

Approach selection: [What method will I use? Why? What are alternatives?]

Solution with confidence tracking:
[Work through the solution, noting confidence at each step]
[Flag any assumptions: ASSUMPTION: ...]
[Flag uncertainty: UNCERTAIN: ...]

Verification: [Can I verify this? How?]

Final answer: [Your answer]
Confidence: [0-100%] because [reason]"""

def metacognitive_solve(problem: str, model: str = "gpt-4o") -> Dict:
    """Solve with full metacognitive awareness"""
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": METACOGNITIVE_SYSTEM},
            {"role": "user", "content": METACOGNITIVE_TEMPLATE.format(problem=problem)}
        ],
        temperature=0.3
    )
    
    content = response.choices[0].message.content
    
    # Extract confidence
    confidence_match = re.search(r"Confidence:\s*(\d+)%", content)
    confidence = int(confidence_match.group(1)) if confidence_match else None
    
    # Extract final answer
    answer_match = re.search(r"Final answer:\s*(.+?)(?=Confidence:|$)", content, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else content
    
    # Extract assumptions and uncertainties
    assumptions = re.findall(r"ASSUMPTION:\s*(.+)", content)
    uncertainties = re.findall(r"UNCERTAIN:\s*(.+)", content)
    
    return {
        "answer": answer,
        "confidence": confidence,
        "assumptions": assumptions,
        "uncertainties": uncertainties,
        "full_analysis": content
    }
```

---

## 13. Hypothetical Document Embeddings (HyDE)

HyDE (Gao et al., 2022) improves retrieval by generating a *hypothetical* answer to a query, then searching for real documents similar to the hypothetical answer.

**Intuition**: A hypothetical answer is semantically closer to real relevant documents than the original query.

```python
class HyDE:
    """
    Hypothetical Document Embeddings for improved retrieval.
    
    Instead of: embed(query) → search
    Do: embed(generate_hypothetical_doc(query)) → search
    """
    
    def __init__(self, llm_client, embedding_model: str = "text-embedding-3-small"):
        self.client = llm_client
        self.embedding_model = embedding_model
    
    def generate_hypothetical_document(
        self,
        query: str,
        doc_type: str = "answer",
        n_variants: int = 1
    ) -> List[str]:
        """
        Generate hypothetical documents that would answer the query.
        
        For retrieval: generate a passage that would appear in a relevant document.
        For QA: generate a direct answer to use as retrieval signal.
        """
        
        prompts = {
            "answer": f"""Write a concise, factual answer to this question as if it appeared in an authoritative document:
Question: {query}
Answer:""",
            "passage": f"""Write a passage from a technical document that directly addresses:
{query}

The passage should be 2-3 sentences, factual, and information-dense:""",
            "paper": f"""Write an abstract excerpt from a research paper that addresses:
{query}

Include specific technical details and findings:"""
        }
        
        prompt = prompts.get(doc_type, prompts["answer"])
        hypotheticals = []
        
        for _ in range(n_variants):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7  # Some variety for multiple variants
            )
            hypotheticals.append(response.choices[0].message.content.strip())
        
        return hypotheticals
    
    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [d.embedding for d in response.data]
    
    def get_query_embedding(self, query: str, n_hypotheticals: int = 3) -> List[float]:
        """
        Get query embedding using HyDE:
        Average embedding of query + hypothetical documents
        """
        import numpy as np
        
        # Generate hypothetical documents
        hypotheticals = self.generate_hypothetical_document(query, n_variants=n_hypotheticals)
        
        # Embed all (query + hypotheticals)
        all_texts = [query] + hypotheticals
        embeddings = self.embed_batch(all_texts)
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        
        return avg_embedding, hypotheticals
    
    def retrieve(
        self,
        query: str,
        vectorstore,
        n_results: int = 5,
        use_hyde: bool = True
    ) -> List[Dict]:
        """Retrieve documents using HyDE or standard embedding"""
        
        if use_hyde:
            query_embedding, hypotheticals = self.get_query_embedding(query)
            print(f"HyDE generated {len(hypotheticals)} hypothetical docs")
        else:
            query_embedding = self.embed(query)
            hypotheticals = []
        
        # Search vector store
        results = vectorstore.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0],
            "hypotheticals": hypotheticals,
            "distances": results["distances"][0]
        }

# Demonstration
def hyde_vs_standard_retrieval(query: str, documents: List[str]):
    """Compare HyDE vs standard retrieval"""
    import chromadb
    import numpy as np
    
    hyde = HyDE(client)
    
    # Build vector store
    chroma = chromadb.Client()
    collection = chroma.create_collection("demo")
    
    # Add documents
    embeddings = hyde.embed_batch(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # Standard retrieval
    standard_emb = hyde.embed(query)
    standard_results = collection.query(query_embeddings=[standard_emb], n_results=3)
    
    # HyDE retrieval
    hyde_emb, hypotheticals = hyde.get_query_embedding(query)
    hyde_results = collection.query(query_embeddings=[hyde_emb], n_results=3)
    
    print(f"Query: {query}")
    print(f"\nStandard retrieval top-3: {standard_results['documents'][0]}")
    print(f"\nHyDE hypothetical: {hypotheticals[0]}")
    print(f"\nHyDE retrieval top-3: {hyde_results['documents'][0]}")
```

---

## 14. Constitutional AI Reasoning

Constitutional AI (Anthropic, 2022) uses a set of principles (the "constitution") to guide the model's self-correction. The model critiques its own outputs against the constitution and revises them.

```python
CONSTITUTION = [
    "Be helpful, harmless, and honest",
    "Do not generate content that could harm people",
    "Acknowledge uncertainty when it exists",
    "Avoid deception and manipulation",
    "Respect user privacy and autonomy",
    "Provide balanced perspectives on controversial topics",
    "Cite sources when making factual claims",
]

class ConstitutionalReasoner:
    """
    Constitutional AI: Self-critique and revision using a principle set.
    
    Pipeline:
    1. Generate initial response (helpful-only, less filtered)
    2. Critique against constitution
    3. Revise based on critique
    4. Repeat for all principles (or most relevant)
    """
    
    CRITIQUE_PROMPT = """Read the following AI response and identify any issues based on this principle:

Principle: "{principle}"

User query: {query}
AI response: {response}

Is this response consistent with the principle?
If not, explain what is wrong and how to fix it.
Format:
CRITIQUE: [your critique]
REVISED RESPONSE: [corrected response, or "No revision needed"]"""
    
    def __init__(self, constitution: List[str] = None, model: str = "gpt-4o"):
        self.constitution = constitution or CONSTITUTION
        self.model = model
        self.client = OpenAI()
    
    def generate_initial(self, query: str) -> str:
        """Generate initial response (raw, minimal filtering)"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def critique_and_revise(self, query: str, response: str, principle: str) -> Tuple[str, str]:
        """Apply one constitutional principle: critique and revise"""
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.CRITIQUE_PROMPT.format(
                    principle=principle,
                    query=query,
                    response=response
                )
            }],
            temperature=0
        )
        
        content = result.choices[0].message.content
        
        # Parse critique and revision
        critique_match = re.search(r"CRITIQUE:\s*(.+?)(?=REVISED|$)", content, re.DOTALL)
        revised_match = re.search(r"REVISED RESPONSE:\s*(.+?)$", content, re.DOTALL)
        
        critique = critique_match.group(1).strip() if critique_match else content
        revised = revised_match.group(1).strip() if revised_match else response
        
        if "no revision needed" in revised.lower():
            revised = response
        
        return critique, revised
    
    def run(self, query: str, relevant_principles: int = 3) -> Dict:
        """Run full constitutional AI pipeline"""
        
        # Stage 1: Generate initial response
        response = self.generate_initial(query)
        revisions = [{"stage": "initial", "response": response}]
        
        # Stage 2: Critique and revise against most relevant principles
        # In practice, select principles based on query content
        principles_to_apply = self.constitution[:relevant_principles]
        
        critiques = []
        for principle in principles_to_apply:
            critique, revised = self.critique_and_revise(query, response, principle)
            critiques.append({"principle": principle, "critique": critique})
            
            if revised != response:
                response = revised
                revisions.append({
                    "stage": f"revised_for_{principle[:30]}",
                    "response": response
                })
        
        return {
            "final_response": response,
            "initial_response": revisions[0]["response"],
            "critiques": critiques,
            "revisions": len(revisions) - 1
        }
```

---

## 15. Process vs Outcome Reward Models

### 15.1 Outcome Reward Models (ORM)

ORM judges only the final answer — correct or incorrect. Used in RLHF for helpfulness.

```python
class OutcomeRewardModel:
    """
    Outcome Reward Model: Scores final answers only.
    Binary or scalar reward based on answer quality.
    """
    
    ORM_PROMPT = """Evaluate this answer to the question.

Question: {question}
Reference answer: {reference}
Model answer: {model_answer}

Score the model's answer:
- 1.0: Completely correct and well-expressed
- 0.7: Mostly correct, minor issues
- 0.5: Partially correct
- 0.2: Mostly wrong but some correctness
- 0.0: Completely wrong

Return ONLY a decimal score:"""
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model
    
    def score(self, question: str, reference: str, model_answer: str) -> float:
        """Score a model answer against reference"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.ORM_PROMPT.format(
                    question=question,
                    reference=reference,
                    model_answer=model_answer
                )
            }],
            temperature=0
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5
    
    def best_of_n(self, question: str, reference: str, candidates: List[str]) -> Dict:
        """Select best answer from N candidates using ORM"""
        scores = [(self.score(question, reference, c), c) for c in candidates]
        scores.sort(key=lambda x: -x[0])
        
        return {
            "best_answer": scores[0][1],
            "best_score": scores[0][0],
            "all_scores": scores
        }
```

### 15.2 Process Reward Models (PRM)

PRM (Lightman et al., 2023, "Let's Verify Step by Step") rewards each reasoning step, not just the final answer. Critical for detecting wrong reasoning that leads to correct answers by luck.

```python
class ProcessRewardModel:
    """
    Process Reward Model: Scores each reasoning step individually.
    Allows detecting errors early and searching for better paths.
    """
    
    STEP_EVALUATION_PROMPT = """Evaluate this reasoning step for correctness.

Problem: {problem}
Previous steps: {previous_steps}
Current step: {current_step}

Is this step:
(a) Correct and logically sound?
(b) Correct but could be clearer?
(c) Partially correct with minor errors?
(d) Incorrect?

Score: [0.0-1.0] where 1.0 = perfect step
Verdict: [correct/incorrect]
Explanation: [brief explanation]

Score (number only):"""
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model
    
    def score_step(self, problem: str, previous_steps: List[str], current_step: str) -> float:
        """Score a single reasoning step"""
        previous_str = "\n".join([f"Step {i+1}: {s}" for i, s in enumerate(previous_steps)])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.STEP_EVALUATION_PROMPT.format(
                    problem=problem,
                    previous_steps=previous_str if previous_str else "None",
                    current_step=current_step
                )
            }],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        try:
            return float(re.search(r"([0-9.]+)", content).group(1))
        except (AttributeError, ValueError):
            return 0.5
    
    def score_solution(self, problem: str, solution_steps: List[str]) -> Dict:
        """Score each step in a complete solution"""
        step_scores = []
        
        for i, step in enumerate(solution_steps):
            score = self.score_step(problem, solution_steps[:i], step)
            step_scores.append({"step": step, "score": score, "index": i})
        
        # Find first bad step
        first_error = None
        for s in step_scores:
            if s["score"] < 0.5:
                first_error = s["index"]
                break
        
        return {
            "step_scores": step_scores,
            "avg_score": sum(s["score"] for s in step_scores) / len(step_scores),
            "min_score": min(s["score"] for s in step_scores),
            "first_error_step": first_error,
            "solution_quality": "good" if not first_error else "has_errors"
        }
    
    def search_with_prm(
        self,
        problem: str,
        generate_fn: Callable,
        beam_width: int = 4,
        max_steps: int = 6
    ) -> Dict:
        """Beam search guided by PRM"""
        
        # Each beam = list of steps so far
        beams = [[]]
        
        for step_idx in range(max_steps):
            new_beams = []
            
            for beam in beams:
                # Generate next step candidates
                candidates = generate_fn(problem, beam, n=beam_width)
                
                for candidate in candidates:
                    new_steps = beam + [candidate]
                    score = self.score_step(problem, beam, candidate)
                    new_beams.append((score, new_steps))
            
            # Keep top beam_width beams
            new_beams.sort(key=lambda x: -x[0])
            beams = [b[1] for b in new_beams[:beam_width]]
            
            print(f"Step {step_idx+1}: {len(beams)} beams, "
                  f"top score: {new_beams[0][0]:.2f}")
        
        # Return best beam
        best_beam = beams[0]
        return {
            "solution_steps": best_beam,
            "final_score": self.score_solution(problem, best_beam)["avg_score"]
        }
```

---

## 16. Test-Time Compute Scaling

The "test-time compute" paradigm: instead of just scaling model size (training compute), scale inference compute to get better answers.

### 16.1 Best-of-N Sampling

Generate N answers, select the best:

```python
class BestOfN:
    """Best-of-N: generate N answers, select best by ORM/PRM"""
    
    def __init__(self, llm_client, reward_model, n: int = 8):
        self.client = llm_client
        self.reward_model = reward_model
        self.n = n
    
    def generate_candidates(self, problem: str, model: str = "gpt-4o") -> List[str]:
        """Generate N diverse candidates"""
        candidates = []
        for _ in range(self.n):
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Solve this step by step: {problem}"}],
                temperature=0.8
            )
            candidates.append(response.choices[0].message.content)
        return candidates
    
    def select_best(self, problem: str, candidates: List[str]) -> Dict:
        """Use reward model to select best candidate"""
        scored = []
        for candidate in candidates:
            score = self.reward_model.score(problem, "", candidate)
            scored.append((score, candidate))
        
        scored.sort(key=lambda x: -x[0])
        return {
            "best": scored[0][1],
            "best_score": scored[0][0],
            "all_scores": [s[0] for s in scored]
        }
    
    def run(self, problem: str) -> Dict:
        candidates = self.generate_candidates(problem)
        return self.select_best(problem, candidates)
```

### 16.2 Compute-Optimal Scaling (s1 / STILL-2 / o1 Approach)

The o1/o3 style: train models to "think longer" with extended reasoning chains, using verifiable rewards:

```python
class ExtendedThinkingAgent:
    """
    Simulate extended thinking: give the model more time to reason
    by generating a long scratchpad before committing to an answer.
    
    This is the open-source analog to o1's approach.
    """
    
    BUDGET_TOKENS = {
        "quick": 500,
        "standard": 2000,
        "extended": 8000,
        "deep": 32000
    }
    
    THINKING_PROMPT = """<thinking>
You have a thinking space. Use it to reason carefully before answering.

Problem: {problem}

Think deeply:
- What type of problem is this?
- What are all the relevant facts?
- What approach should I take?
- Work through the problem step by step
- Check for edge cases
- Verify your reasoning
- Consider alternative approaches

[Work through this thoroughly. Take as much space as you need.]
</thinking>

Based on your thinking, provide the final answer:"""
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model
    
    def solve_with_budget(
        self,
        problem: str,
        budget: str = "standard"
    ) -> Dict:
        """Solve with controlled thinking budget"""
        max_tokens = self.BUDGET_TOKENS.get(budget, 2000)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": self.THINKING_PROMPT.format(problem=problem)
            }],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        
        # Extract thinking and answer
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", content, re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        answer = content.split("</thinking>")[-1].strip() if thinking_match else content
        
        return {
            "thinking": thinking,
            "answer": answer,
            "thinking_tokens": len(thinking.split()),
            "budget": budget
        }
    
    def solve_with_verification(self, problem: str) -> Dict:
        """Solve and self-verify — retry if verification fails"""
        
        # First attempt
        result = self.solve_with_budget(problem, "extended")
        
        # Verify
        verify_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Verify this solution to the problem:

Problem: {problem}
Solution: {result['answer']}

Check:
1. Is each step logically valid?
2. Are the calculations correct?
3. Does the answer make sense?

VERDICT: CORRECT or INCORRECT
If incorrect, explain what's wrong."""
            }],
            temperature=0
        )
        
        verdict = verify_response.choices[0].message.content
        is_correct = "CORRECT" in verdict.upper() and "INCORRECT" not in verdict.upper()
        
        if not is_correct:
            # Retry with higher budget and error context
            retry_result = self.solve_with_budget(
                problem + f"\n\nNote: A previous solution was incorrect. Reason: {verdict}",
                "deep"
            )
            return {**retry_result, "verified": False, "initial_error": verdict, "retried": True}
        
        return {**result, "verified": True, "verification": verdict, "retried": False}
```

### 16.3 Monte Carlo Self-Consistency

Combine MCTS with self-consistency for maximum reliability:

```python
def monte_carlo_self_consistency(
    problem: str,
    n_paths: int = 16,
    n_votes: int = 10,
    model: str = "gpt-4o"
) -> Dict:
    """
    Advanced: Generate N reasoning paths, each with multiple candidate answers,
    then aggregate via weighted voting.
    """
    import statistics
    
    all_answers = []
    path_data = []
    
    # Generate diverse reasoning paths
    for path_idx in range(n_paths):
        # Use different temperature per path
        temp = 0.5 + (path_idx / n_paths) * 0.5  # Range: 0.5 to 1.0
        
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Solve step by step:\n{problem}"
            }],
            temperature=temp
        )
        
        reasoning = response.choices[0].message.content
        answer = extract_answer(reasoning)
        
        if answer:
            all_answers.append(answer)
            path_data.append({
                "reasoning": reasoning,
                "answer": answer,
                "temperature": temp
            })
    
    if not all_answers:
        return {"answer": None, "confidence": 0}
    
    # Majority vote
    counter = Counter(all_answers)
    majority, count = counter.most_common(1)[0]
    
    return {
        "answer": majority,
        "confidence": count / len(all_answers),
        "vote_distribution": dict(counter),
        "n_paths_successful": len(all_answers),
        "all_paths": path_data
    }
```

---

## 17. Verification and Checking

### 17.1 Self-Verification

```python
VERIFY_PROMPT = """You previously solved this problem. Verify your solution.

Problem: {problem}
Your solution: {solution}

Verification steps:
1. Re-read the problem carefully
2. Check each step of the solution
3. Verify calculations independently
4. Check if the answer makes intuitive sense
5. Consider edge cases

Is the solution correct? CORRECT or INCORRECT with reason."""

def verify_and_correct(problem: str, model: str = "gpt-4o") -> Dict:
    """Generate solution, verify it, correct if wrong"""
    
    # Step 1: Generate solution
    sol_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Solve: {problem}\n\nStep by step:"}],
        temperature=0
    )
    solution = sol_response.choices[0].message.content
    
    # Step 2: Verify
    verify_response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": VERIFY_PROMPT.format(problem=problem, solution=solution)
        }],
        temperature=0
    )
    verdict = verify_response.choices[0].message.content
    
    is_correct = "CORRECT" in verdict[:20].upper() and "INCORRECT" not in verdict[:20].upper()
    
    if is_correct:
        return {"answer": solution, "verified": True}
    
    # Step 3: Correct
    correct_response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"""Problem: {problem}

Previous (incorrect) solution: {solution}
What was wrong: {verdict}

Now provide the correct solution:"""
        }],
        temperature=0
    )
    
    corrected = correct_response.choices[0].message.content
    return {
        "answer": corrected,
        "verified": False,
        "original": solution,
        "correction_reason": verdict
    }
```

### 17.2 Cross-Validation with Multiple Methods

```python
def multi_method_solve(problem: str) -> Dict:
    """Solve with multiple methods and check consistency"""
    
    methods = {
        "direct": lambda p: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": p}],
            temperature=0
        ).choices[0].message.content,
        
        "cot": lambda p: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{p}\n\nLet's think step by step:"}],
            temperature=0
        ).choices[0].message.content,
        
        "pal": lambda p: PAL(client).solve(p)["output"]
    }
    
    results = {}
    for method_name, method_fn in methods.items():
        try:
            result = method_fn(problem)
            answer = extract_answer(str(result))
            results[method_name] = {"result": result, "answer": answer}
        except Exception as e:
            results[method_name] = {"error": str(e)}
    
    # Check agreement
    answers = [r["answer"] for r in results.values() if r.get("answer")]
    counter = Counter(answers)
    
    if counter:
        majority, count = counter.most_common(1)[0]
        agreement = count / len(answers)
    else:
        majority, agreement = None, 0
    
    return {
        "methods": results,
        "consensus_answer": majority,
        "agreement_rate": agreement,
        "high_confidence": agreement >= 0.67
    }
```

---

## 18. World Models for Reasoning

World models let agents simulate outcomes before committing to actions:

```python
class LLMWorldModel:
    """
    LLM as world model: predict consequences of actions.
    Used for planning and reasoning about hypotheticals.
    """
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model
    
    def simulate(self, state: str, action: str, domain: str = "general") -> Dict:
        """Simulate the outcome of an action given current state"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Simulate what happens when this action is taken.

Domain: {domain}
Current state: {state}
Action to take: {action}

Predict:
1. Immediate outcome (what happens right after)
2. Side effects (unintended consequences)
3. Long-term implications
4. Success probability (0-100%)
5. Risks

Format as JSON:
{{
  "immediate_outcome": "...",
  "side_effects": ["..."],
  "long_term": "...",
  "success_probability": 0-100,
  "risks": ["..."]
}}"""
            }],
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(json_match.group()) if json_match else {"raw": content}
        except Exception:
            return {"raw": content}
    
    def plan_with_simulation(self, goal: str, initial_state: str, max_steps: int = 5) -> List[Dict]:
        """Plan sequence of actions using world model simulation"""
        
        state = initial_state
        plan = []
        
        for step in range(max_steps):
            # Generate candidate actions
            actions_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"""Goal: {goal}
Current state: {state}

Generate 3 possible next actions. Be specific and concrete.
Action 1:
Action 2:
Action 3:"""
                }],
                temperature=0.7
            )
            
            actions_text = actions_response.choices[0].message.content
            actions = [a.strip().strip("123. ") for a in actions_text.split("\n") 
                      if a.strip() and not a.startswith("#")][:3]
            
            # Simulate each action
            best_action = None
            best_prob = -1
            best_outcome = None
            
            for action in actions:
                sim = self.simulate(state, action)
                prob = sim.get("success_probability", 50)
                
                if prob > best_prob:
                    best_prob = prob
                    best_action = action
                    best_outcome = sim
            
            plan.append({
                "step": step + 1,
                "state": state,
                "action": best_action,
                "predicted_outcome": best_outcome,
                "confidence": best_prob
            })
            
            # Update state
            state = best_outcome.get("immediate_outcome", state) if best_outcome else state
            
            # Check if goal achieved
            goal_check = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": f"Goal: {goal}\nCurrent state: {state}\nIs the goal achieved? YES or NO:"
                }],
                temperature=0
            )
            if "YES" in goal_check.choices[0].message.content.upper():
                break
        
        return plan
```

---

## 19. Practical Reasoning Pipelines

### 19.1 Adaptive Reasoning Selector

```python
class AdaptiveReasoner:
    """Selects the best reasoning strategy based on problem type"""
    
    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model
        self.pal = PAL(llm_client)
        self.tot = None  # Initialize on demand
    
    def classify_problem(self, problem: str) -> str:
        """Classify problem to select appropriate strategy"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"""Classify this problem into ONE category:

Problem: {problem}

Categories:
- arithmetic: Pure calculation/math
- logic: Logical deduction, syllogisms
- multi_hop: Requires multiple fact lookups
- code: Programming task
- search: Requires information retrieval
- planning: Multi-step task planning
- creative: Open-ended/creative task

Return only the category name:"""
            }],
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    
    def solve(self, problem: str, verbose: bool = True) -> Dict:
        """Adaptively select and apply the best reasoning strategy"""
        
        problem_type = self.classify_problem(problem)
        
        if verbose:
            print(f"Problem type: {problem_type}")
        
        strategies = {
            "arithmetic": lambda p: self.pal.solve(p),
            "logic": lambda p: {"answer": two_stage_zero_shot_cot(p)["answer"]},
            "multi_hop": lambda p: SelfAskAgent(search_fn=lambda q: q).run(p),
            "code": lambda p: CodeAgent(self.client).solve(p),
            "planning": lambda p: {"answer": few_shot_cot(p)},
            "creative": lambda p: {"answer": scratchpad_reasoning(p)["final_answer"]},
            "search": lambda p: {"answer": few_shot_cot(p)},
        }
        
        strategy = strategies.get(problem_type, strategies["logic"])
        
        result = strategy(problem)
        result["strategy"] = problem_type
        
        return result

# Usage
reasoner = AdaptiveReasoner(client)

test_problems = [
    "A train travels 240 miles in 3 hours. How many miles per hour is that?",
    "All dogs are animals. Some animals are cute. Can we conclude some dogs are cute?",
    "Write a function to reverse a linked list in Python",
    "Plan a 5-day trip to Japan with $2000 budget"
]

for problem in test_problems:
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    result = reasoner.solve(problem)
    print(f"Strategy: {result['strategy']}")
    answer = result.get('answer') or result.get('output', 'N/A')
    print(f"Answer: {str(answer)[:200]}")
```

### 19.2 Reasoning Pipeline with Quality Control

```python
class RobustReasoningPipeline:
    """Production-grade reasoning with quality control"""
    
    def __init__(self, model: str = "gpt-4o", n_self_consistency: int = 5):
        self.model = model
        self.n = n_self_consistency
        self.client = OpenAI()
    
    def solve(self, problem: str, confidence_threshold: float = 0.6) -> Dict:
        """
        Full pipeline:
        1. Self-consistency (majority vote)
        2. If low confidence: escalate to extended thinking
        3. Verify answer
        """
        
        # Stage 1: Self-consistency
        sc_result = self_consistency(problem, n_samples=self.n, model=self.model)
        
        if sc_result["confidence"] >= confidence_threshold:
            # High confidence — verify and return
            verified = verify_and_correct(problem, self.model)
            if extract_answer(verified["answer"]) == sc_result["answer"]:
                return {
                    "answer": sc_result["answer"],
                    "confidence": sc_result["confidence"],
                    "method": "self_consistency",
                    "verified": True
                }
        
        # Stage 2: Extended thinking for low confidence
        print(f"Low confidence ({sc_result['confidence']:.0%}). Using extended thinking...")
        extended = ExtendedThinkingAgent(self.client).solve_with_verification(problem)
        
        # Stage 3: Final verification
        final_answer = extended["answer"]
        
        return {
            "answer": final_answer,
            "confidence": sc_result["confidence"],
            "method": "extended_thinking",
            "verified": extended.get("verified", False),
            "self_consistency_result": sc_result
        }

# Final demonstration
print("=== Robust Reasoning Pipeline Demo ===")
pipeline = RobustReasoningPipeline(n_self_consistency=5)

complex_problem = """A store offers a 20% discount on all items. 
Alice buys 3 shirts originally priced at $45 each and 2 pants at $60 each.
She also uses a $10 coupon applied after the percentage discount.
How much does she pay in total?"""

result = pipeline.solve(complex_problem)
print(f"\nProblem: {complex_problem}")
print(f"\nAnswer: {result['answer']}")
print(f"Method: {result['method']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Verified: {result['verified']}")
```

---

## 20. Pitfalls and Common Mistakes

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **CoT hurts on easy tasks** | "Let's think step by step" can degrade accuracy on simple classification or retrieval | Use CoT only for complex, multi-step tasks; benchmark with/without |
| **Self-consistency at temperature 0** | Zero temperature → identical samples → no voting benefit | Use temperature 0.5–0.8 for diversity; 5–10 samples usually sufficient |
| **PAL/PoT code injection** | Generated code may read/write files, import malicious modules | Run in sandbox (restricted `subprocess`, containers, or stripped interpreter) |
| **ToT evaluation collapse** | LLM-as-evaluator often prefers verbose or familiar patterns | Use multiple evaluators, add calibration, or use a separate critic model |
| **ReAct tool-call loops** | Agent may call the same tool repeatedly with slight variations | Enforce max steps per tool; add "no-op" or "I'm done" action |
| **Reflexion memory bloat** | Storing all failures causes context overflow | Prune low-utility entries; summarize; cap memory size |
| **Least-to-Most over-decomposition** | Too many sub-problems → error accumulation and cost | Limit sub-problems (e.g., ≤5); validate decomposition quality |
| **Extended thinking without verification** | More tokens ≠ better answers; can amplify reasoning errors | Always pair with verification (PRM or self-check) |
| **Ignoring confidence** | Treating all answers equally wastes compute on low-confidence cases | Use confidence thresholds to escalate (e.g., retry with ToT, PAL) |
| **Benchmark overfitting** | Methods tuned on GSM8K/MATH may not transfer to real tasks | Test on out-of-domain data; use diverse evaluation sets |

---

## Summary: Reasoning Method Selection Guide

| Problem Type | Best Method | Why |
|-------------|-------------|-----|
| Arithmetic/math | PAL/PoT | Offload computation to interpreter |
| Multi-step math | CoT + Self-Consistency | Structured reasoning + voting |
| Logical reasoning | Zero-shot CoT | Step-by-step deduction |
| Complex research | ReAct + Search | Interleave reasoning and retrieval |
| Hard planning | ToT (BFS/MCTS) | Explore multiple paths |
| Learning from failure | Reflexion | Verbal memory of mistakes |
| Multi-fact question | Self-Ask | Explicit sub-question decomposition |
| Ambiguous/novel | Metacognitive | Confidence-aware reasoning |
| Safety-critical | Constitutional AI | Principle-based self-correction |
| Low confidence result | Best-of-N + ORM | Sample and rank |
| Highest stakes | Extended thinking + PRM | Maximum compute |

### Key Papers

- **Chain-of-Thought** (Wei et al., 2022): arxiv.org/abs/2201.11903
- **Zero-Shot CoT** (Kojima et al., 2022): arxiv.org/abs/2205.11916
- **Self-Consistency** (Wang et al., 2022): arxiv.org/abs/2203.11171
- **Least-to-Most** (Zhou et al., 2022): arxiv.org/abs/2205.10625
- **Plan-and-Solve (Ps)** (Wang et al., 2023): arxiv.org/abs/2305.04091
- **Tree of Thoughts** (Yao et al., 2023): arxiv.org/abs/2305.10601
- **PoT** (Chen et al., 2022): arxiv.org/abs/2211.12588
- **PAL** (Gao et al., 2022): arxiv.org/abs/2211.10435
- **Chain-of-Code** (Zhou et al., 2023): code execution + explanation
- **ReWOO** (Wang et al., 2023): arxiv.org/abs/2305.18323 (plan-then-execute)
- **ReAct** (Yao et al., 2023): arxiv.org/abs/2210.03629
- **Reflexion** (Shinn et al., 2023): arxiv.org/abs/2303.11366
- **HyDE** (Gao et al., 2022): arxiv.org/abs/2212.10496
- **Let's Verify Step by Step/PRM** (Lightman et al., 2023): arxiv.org/abs/2305.20050
- **Constitutional AI** (Anthropic, 2022): arxiv.org/abs/2212.08073
