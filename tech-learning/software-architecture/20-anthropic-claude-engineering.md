# Anthropic & Claude: Production Engineering Guide

## Table of Contents
1. [Anthropic Overview](#1-anthropic-overview)
2. [Claude Model Family](#2-claude-model-family)
3. [Claude API Deep Dive](#3-claude-api-deep-dive)
4. [Constitutional AI & Safety](#4-constitutional-ai--safety)
5. [Extended Thinking & Reasoning](#5-extended-thinking--reasoning)
6. [Tool Use & Agents](#6-tool-use--agents)
7. [Prompt Engineering for Claude](#7-prompt-engineering-for-claude)
8. [Context Window Management](#8-context-window-management)
9. [Production Architecture with Claude](#9-production-architecture-with-claude)
10. [Claude on AWS Bedrock](#10-claude-on-aws-bedrock)
11. [Cost Optimization](#11-cost-optimization)
12. [Practical Examples](#12-practical-examples)

---

## 1. Anthropic Overview

### 1.1 Company & Mission

**Anthropic** (founded 2021, San Francisco) is an AI safety company whose mission is the responsible development and maintenance of advanced AI for the long-term benefit of humanity. It was founded by Dario Amodei (CEO), Daniela Amodei (President), and several former OpenAI researchers.

```
Key Differentiators vs Other AI Labs:

SAFETY-FIRST RESEARCH
  ─ Interpretability research: understand what models are "thinking"
  ─ Constitutional AI (CAI): systematic approach to value alignment
  ─ Red-teaming before every model release
  ─ Publishes safety research openly (e.g., "Sleeper Agents" paper)

FRONTIER RESEARCH + PRODUCTS
  ─ Not purely a research lab (unlike DeepMind historically)
  ─ Claude is both a product and a research platform
  ─ Enterprise API + consumer Claude.ai

ARCHITECTURE DECISIONS
  ─ Transformer-based LLMs (similar to GPT family)
  ─ Large context windows as a core differentiator
  ─ Heavy focus on instruction following and harmlessness
  ─ Native multi-modal (vision) in Claude 3+

COMPUTE PARTNERSHIPS
  ─ Amazon: $4B investment, Claude on AWS Bedrock
  ─ Google: $300M investment, Claude on Google Cloud
  ─ Trains on TPUs (Google) + Trainium/Inferentia (AWS)
```

### 1.2 Anthropic Research Contributions

| Research Area | Key Paper / Contribution |
|--------------|--------------------------|
| Constitutional AI | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) |
| Mechanistic Interpretability | [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) |
| Scaling Laws | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (pre-Anthropic) |
| Sleeper Agents | [Sleeper Agents: Training Deceptive LLMs](https://arxiv.org/abs/2401.05566) |
| Model Cards | Detailed system cards for every Claude model |
| In-Context Learning | Foundational research on few-shot prompting |

---

## 2. Claude Model Family

### 2.1 Model Tiers (Haiku / Sonnet / Opus)

Anthropic follows a three-tier naming convention across model generations:

```
Claude Model Tiers:

HAIKU  ─ Fastest, most compact, lowest cost
  ─ Best for: high-volume classification, routing, simple extraction
  ─ Latency: lowest (< 1s first token typical)
  ─ Cost: cheapest in the family

SONNET ─ Balance of intelligence and speed
  ─ Best for: complex reasoning, coding, most production tasks
  ─ Latency: moderate
  ─ Cost: mid-tier; sweet spot for most use cases

OPUS   ─ Most capable, deepest reasoning
  ─ Best for: very complex analysis, research, nuanced writing
  ─ Latency: highest
  ─ Cost: most expensive
  ─ Note: as of 2025, Claude 3.7 Sonnet rivals older Opus on most tasks
```

### 2.2 Model Generations

```
Model Timeline:

Claude 1 (2023)
  ─ First generation, limited context
  ─ Established Claude's tone and safety profile

Claude 2 (2023)
  ─ 100K context window (major leap)
  ─ Improved instruction following
  ─ Better at long-document analysis

Claude 3 Family (2024 — Haiku, Sonnet, Opus)
  ─ Native vision/multimodal (images in input)
  ─ 200K context window
  ─ Claude 3 Opus: SOTA on many benchmarks at release
  ─ Claude 3.5 Sonnet: outperforms Claude 3 Opus at Sonnet price

Claude 3.5 Family (2024-2025)
  ─ Claude 3.5 Sonnet: best coding model at launch
  ─ Claude 3.5 Haiku: fast, affordable, nearly 3.5 Sonnet quality
  ─ Artifacts: generates interactive web apps/code directly

Claude 3.7 (2025)
  ─ Extended Thinking: explicit chain-of-thought reasoning mode
  ─ Hybrid reasoning: can toggle thinking on/off
  ─ Claude 3.7 Sonnet: best coding + reasoning at release
  ─ 200K context, thinking tokens don't count toward context limit
```

### 2.3 Model Comparison (2025 reference)

| Model | Context | Vision | Thinking | Best For |
|-------|---------|--------|----------|---------|
| `claude-3-5-haiku-20241022` | 200K | ✅ | ❌ | High-volume tasks, latency-sensitive |
| `claude-3-5-sonnet-20241022` | 200K | ✅ | ❌ | Coding, analysis, most production |
| `claude-3-7-sonnet-20250219` | 200K | ✅ | ✅ | Complex reasoning, hard coding tasks |
| `claude-3-opus-20240229` | 200K | ✅ | ❌ | Research, nuanced tasks |

### 2.4 Claude vs GPT-4 vs Gemini

```
Positioning Comparison (2025):

                   Claude 3.7      GPT-4o          Gemini 1.5 Pro
                   Sonnet          (OpenAI)        (Google)
───────────────────────────────────────────────────────────────────
Context Window     200K tokens     128K            1M tokens
Vision             ✅              ✅              ✅
Thinking/CoT       ✅ (native)     ✅ (o3/o4 models) ✅ (Flash Thinking)
Code generation    ⭐⭐⭐⭐⭐     ⭐⭐⭐⭐        ⭐⭐⭐⭐
Instruction follow ⭐⭐⭐⭐⭐     ⭐⭐⭐⭐        ⭐⭐⭐⭐
Safety/harmless    ⭐⭐⭐⭐⭐     ⭐⭐⭐⭐        ⭐⭐⭐⭐
Tone/Writing       ⭐⭐⭐⭐⭐     ⭐⭐⭐⭐        ⭐⭐⭐
Multimodal (non-text) ❌          ✅ (audio/video)  ✅ (audio/video)
Enterprise support ✅ (Bedrock)   ✅ (Azure OAI)   ✅ (Vertex AI)

Claude's notable strengths:
  ─ Following complex, nuanced instructions precisely
  ─ Long-document analysis and summarization
  ─ Coding (especially complex multi-file tasks)
  ─ Refusing to be manipulated (jailbreak-resistant)
  ─ Honest about uncertainty ("I don't know")
  ─ Very large context without degradation
```

---

## 3. Claude API Deep Dive

### 3.1 Messages API

The Claude API uses the **Messages** format (not completions):

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")

# Basic message
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain the CAP theorem."}
    ]
)
print(message.content[0].text)

# With system prompt
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=2048,
    system="""You are a senior software architect. 
    Provide concise, technically precise answers.
    Use code examples where helpful.""",
    messages=[
        {"role": "user", "content": "When should I use CQRS?"}
    ]
)

# Multi-turn conversation
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is event sourcing?"},
        {"role": "assistant", "content": "Event sourcing is..."},
        {"role": "user", "content": "How does it relate to CQRS?"}
    ]
)
```

### 3.2 Streaming

For production chat/generation interfaces — always use streaming for responsiveness:

```python
import anthropic

client = anthropic.Anthropic()

# Streaming with context manager
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a distributed systems primer"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Streaming with async (FastAPI / async frameworks)
import asyncio

async def stream_response(prompt: str):
    async with anthropic.AsyncAnthropic() as async_client:
        async with async_client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield text   # Server-sent events to frontend

# FastAPI SSE endpoint
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_response(request.message),
        media_type="text/event-stream"
    )
```

### 3.3 Vision (Image Input)

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

# Option 1: base64-encoded local image
image_data = base64.standard_b64encode(
    Path("architecture-diagram.png").read_bytes()
).decode("utf-8")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data,
                },
            },
            {
                "type": "text",
                "text": "Analyze this architecture diagram. Identify any single points of failure."
            }
        ],
    }]
)

# Option 2: URL (publicly accessible image)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/system-diagram.png",
                },
            },
            {"type": "text", "text": "What services are shown?"}
        ],
    }]
)

# Multiple images
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "url", "url": "img1.png"}},
            {"type": "image", "source": {"type": "url", "url": "img2.png"}},
            {"type": "text", "text": "Compare these two architecture diagrams."}
        ]
    }]
)
```

### 3.4 Prompt Caching

**Prompt caching** is one of Claude's most powerful cost and latency features — cache large, static portions of your prompt:

```python
import anthropic

client = anthropic.Anthropic()

# Cache a large system context (e.g., a whole codebase or document)
# First request: cache MISS (normal cost + 25% surcharge to write cache)
# Subsequent requests: cache HIT (90% cheaper on cached tokens)

LARGE_CODEBASE = Path("codebase.txt").read_text()  # Could be 50K+ tokens

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an expert code reviewer. Review code for bugs, performance, and security.",
        },
        {
            "type": "text",
            "text": f"<codebase>{LARGE_CODEBASE}</codebase>",
            "cache_control": {"type": "ephemeral"}  # ← Cache this!
        }
    ],
    messages=[
        {"role": "user", "content": "Find all places where SQL injection might occur."}
    ]
)

# Follow-up question — cache HIT on the large codebase
# Only pays for the short question tokens
message2 = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an expert code reviewer...",
        },
        {
            "type": "text",
            "text": f"<codebase>{LARGE_CODEBASE}</codebase>",
            "cache_control": {"type": "ephemeral"}  # Must include to use cache
        }
    ],
    messages=[
        {"role": "user", "content": "Find all N+1 query patterns."}
    ]
)

print(message2.usage)
# usage: {
#   "cache_read_input_tokens": 45000,   ← 90% cheaper
#   "cache_creation_input_tokens": 0,
#   "input_tokens": 50,                 ← Only paid in full for these
#   "output_tokens": 312
# }
```

```
Prompt Caching Economics:

Standard pricing: $3 / M input tokens (claude-3-5-sonnet)
Cache write:      $3.75 / M tokens (25% premium to populate cache)
Cache read:       $0.30 / M tokens (90% cheaper than standard)

Example: RAG system with 50K token knowledge base, 1000 req/day
  Without caching: 50K × 1000 × $3/M = $150/day
  With caching:
    1 cache write:  50K × $3.75/M = $0.19
    999 cache reads: 50K × 999 × $0.30/M = $14.99
    Total: $15.18/day
    Saving: $134.82/day (90% reduction!)

Cache lifetime: 5 minutes (refreshed on each use)
Min cacheable: 1024 tokens
```

### 3.5 Response Format & Structured Output

```python
import anthropic
import json
from pydantic import BaseModel

client = anthropic.Anthropic()

# Structured JSON output using instructed format
class ServiceAnalysis(BaseModel):
    service_name: str
    risk_level: str          # "HIGH" | "MEDIUM" | "LOW"
    issues: list[str]
    recommendations: list[str]
    estimated_fix_hours: int

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a code reviewer. Always respond with valid JSON only.",
    messages=[{
        "role": "user",
        "content": f"""Analyze this service for issues:
        
        {service_code}
        
        Respond with JSON matching this schema:
        {ServiceAnalysis.model_json_schema()}
        """
    }]
)

result = ServiceAnalysis.model_validate_json(message.content[0].text)

# Prefill technique: steer Claude's response format
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=512,
    messages=[
        {"role": "user", "content": "List the CAP theorem consistency models as JSON array"},
        {"role": "assistant", "content": "["}   # ← Prefill forces JSON array
    ]
)
# Response continues from "[" → guaranteed JSON array format
```

---

## 4. Constitutional AI & Safety

### 4.1 What is Constitutional AI (CAI)

Constitutional AI is Anthropic's approach to AI alignment, introduced in their 2022 paper. Instead of relying solely on human feedback (RLHF), CAI uses AI-generated feedback guided by a set of principles ("constitution"):

```
Traditional RLHF:
  1. Generate responses
  2. Human raters score them (expensive, slow, inconsistent)
  3. Train reward model on human preferences
  4. Fine-tune model (PPO) to maximize reward

Constitutional AI (CAI):
  1. Generate responses (including harmful/problematic ones)
  2. AI critiques its own responses against a CONSTITUTION
     (list of principles: "be helpful, harmless, honest")
  3. AI revises responses based on self-critique
  4. Train on revised responses (RLAIF — RL from AI Feedback)
  
  Benefits:
  ─ Cheaper than human feedback for safety labels
  ─ More consistent (no inter-rater variability)
  ─ Transparent principles (you can read the constitution)
  ─ Scales better than human labeling
```

### 4.2 Claude's Core Principles (HHH)

```
Claude's Design Principles — "HHH":

HELPFUL
  ─ Genuinely useful, not watered-down or hedge-everything
  ─ Treats users as intelligent adults
  ─ Completes tasks thoroughly
  ─ "Unhelpfulness is never trivially safe" — Anthropic

HARMLESS
  ─ Avoids generating content that could cause real harm
  ─ Refuses clearly dangerous requests (bioweapons, CSAM, etc.)
  ─ Nuanced: considers context and intent
  ─ Doesn't refuse benign requests out of overcaution

HONEST
  ─ Never deliberately deceives users
  ─ Acknowledges uncertainty ("I'm not sure, but...")
  ─ Doesn't claim to be human when sincerely asked
  ─ Expresses calibrated confidence (not overconfident)
  ─ Epistemic courage: shares genuine assessments, not just what users want to hear
```

### 4.3 Claude's Model Spec (Character)

Anthropic publishes Claude's [Model Spec](https://www.anthropic.com/claude/model-spec) — a document defining Claude's character, values, and behaviors:

```
Claude's Character Traits (from Model Spec):
  ─ Intellectual curiosity across all domains
  ─ Warmth and care for humans it interacts with
  ─ Playful wit balanced with substance
  ─ Directness and confidence sharing its views
  ─ Deep commitment to honesty and ethics

Priority Hierarchy (when in conflict):
  1. Broadly safe (supports human oversight of AI)
  2. Broadly ethical (good values, honest, avoids harm)
  3. Adherent to Anthropic's principles
  4. Genuinely helpful to operators and users

"Hardcoded" refusals (never do regardless of instructions):
  ─ CBRN weapons assistance (bio, chem, nuclear, radiological)
  ─ CSAM or sexual content involving minors
  ─ Attacks on critical infrastructure
  ─ Actions that undermine human oversight of AI
  ─ Assist seizing unprecedented societal control
```

### 4.4 Operator vs User Trust Levels

```
Claude's Three-Tier Trust Model:

ANTHROPIC (highest trust)
  ─ Sets Claude's core values via training
  ─ Defines hardcoded behaviors
  ─ Cannot be overridden by operators or users

OPERATOR (API customer — your company)
  ─ Sets context via system prompt
  ─ Can expand defaults (e.g., allow adult content on appropriate platforms)
  ─ Can restrict defaults (e.g., only discuss topics relevant to product)
  ─ Can grant users elevated trust
  ─ Cannot override Anthropic's hardcoded rules

USER (end user talking to Claude)
  ─ Default trust level (like member of public)
  ─ Can adjust some defaults (e.g., turn off disclaimers)
  ─ Cannot exceed operator-granted permissions

Example layering:
  Anthropic:  Never generate CSAM (absolute)
  Operator:   "You are a medical information assistant.
               Users are healthcare professionals."
  User:       "I need information about drug overdose thresholds."
  
  Result: Claude provides clinical information (operator context
          elevates user trust for medical queries)
```

---

## 5. Extended Thinking & Reasoning

### 5.1 What is Extended Thinking

Extended Thinking (introduced in Claude 3.7 Sonnet) gives Claude a "scratchpad" to reason through complex problems before responding — similar to OpenAI's o1/o3 chain-of-thought:

```python
import anthropic

client = anthropic.Anthropic()

# Extended thinking: Claude reasons before answering
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000   # How many tokens Claude can "think"
    },
    messages=[{
        "role": "user",
        "content": """Design a distributed rate limiter that:
        - Handles 100K RPS across 50 nodes
        - Provides exactly-once semantics
        - Tolerates up to 2 node failures
        - Has < 10ms overhead
        
        Consider the trade-offs between different algorithms."""
    }]
)

# Response has two blocks: thinking + text
for block in response.content:
    if block.type == "thinking":
        print("=== Claude's Reasoning ===")
        print(block.thinking)   # Internal reasoning (visible!)
    elif block.type == "text":
        print("=== Final Answer ===")
        print(block.text)
```

### 5.2 Thinking Tokens Economics

```
Extended Thinking Pricing:

Thinking tokens are priced the same as output tokens.

Model: claude-3-7-sonnet-20250219
Output tokens: $15 / M tokens
Input tokens:  $3 / M tokens

Example with budget_tokens=8000:
  Input:    500 tokens     = $0.0015
  Thinking: 7,200 tokens   = $0.108   ← significant cost
  Output:   800 tokens     = $0.012
  Total:    $0.1215

vs without thinking:
  Input:    500 tokens     = $0.0015
  Output:   800 tokens     = $0.012
  Total:    $0.0135   (9x cheaper)

When thinking pays off:
  ✅ Multi-step math and logic
  ✅ Complex coding (multi-file architecture decisions)
  ✅ Ambiguous problems requiring trade-off analysis
  ✅ Tasks where initial answer quality justifies cost

When thinking doesn't pay:
  ❌ Simple factual questions
  ❌ High-volume, low-complexity tasks (use Haiku instead)
  ❌ Creative writing (regular Sonnet is fine)
  ❌ When latency is critical (thinking adds tokens = adds time)

Streaming with thinking:
  ─ Thinking blocks stream first, then text block
  ─ Can display "thinking..." indicator while Claude reasons
  ─ Don't show raw thinking to users unless beneficial
```

### 5.3 Thinking in Multi-Turn Conversations

```python
# Thinking tokens must be passed back in multi-turn conversations

messages = [
    {"role": "user", "content": "Design a fault-tolerant payment system."}
]

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 8000},
    messages=messages
)

# CRITICAL: include ALL content blocks (including thinking) in next turn
messages.append({"role": "assistant", "content": response.content})

# Follow-up question
messages.append({
    "role": "user",
    "content": "How would you handle the case where both datacenters go offline?"
})

response2 = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=8000,
    thinking={"type": "enabled", "budget_tokens": 5000},
    messages=messages
)
```

---

## 6. Tool Use & Agents

### 6.1 Tool Use (Function Calling)

Claude can call tools (functions) defined by the developer:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "get_service_metrics",
        "description": "Retrieves current metrics for a service from the monitoring system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "Name of the service"
                },
                "metric": {
                    "type": "string",
                    "enum": ["p99_latency_ms", "error_rate", "throughput_rps"],
                    "description": "Metric to retrieve"
                },
                "time_window_minutes": {
                    "type": "integer",
                    "description": "Time window in minutes",
                    "default": 15
                }
            },
            "required": ["service_name", "metric"]
        }
    },
    {
        "name": "create_incident",
        "description": "Creates a PagerDuty incident for a service outage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "severity": {"type": "string", "enum": ["P1", "P2", "P3"]},
                "summary": {"type": "string"},
                "details": {"type": "string"}
            },
            "required": ["service", "severity", "summary"]
        }
    }
]

# Agentic loop
messages = [{"role": "user",
             "content": "Check if payments-api is healthy and create an incident if error rate > 1%"}]

while True:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # No more tool calls → final answer
    if response.stop_reason == "end_turn":
        print(response.content[0].text)
        break

    # Process tool calls
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Execute the actual tool
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result)
                })

        messages.append({"role": "user", "content": tool_results})
```

### 6.2 Computer Use (Anthropic API Beta)

Claude 3.5 Sonnet supports operating a computer via screen/keyboard/mouse:

```python
# Computer Use API (beta) — Claude controls a real computer
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
        "display_number": 1,
    },
    {
        "type": "bash_20241022",
        "name": "bash",
    },
    {
        "type": "text_editor_20241022",
        "name": "str_replace_editor",
    }
]

response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "Open a terminal, clone the repo, run tests, and report results."
    }],
    betas=["computer-use-2024-10-22"],
)

# Claude will take screenshot → decide action → click/type → repeat
# Actions: screenshot, mouse_move, left_click, type, key, bash command
```

### 6.3 Building Multi-Agent Systems

```python
# Orchestrator-subagent pattern with Claude

class ArchitectureReviewAgent:
    """Orchestrator: decomposes review task into subagent calls"""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.subagents = {
            "security": SecurityReviewAgent(),
            "performance": PerformanceReviewAgent(),
            "scalability": ScalabilityReviewAgent(),
        }
    
    def review(self, architecture_doc: str) -> ReviewReport:
        # Step 1: Orchestrator plans the review
        plan = self.plan_review(architecture_doc)
        
        # Step 2: Fan out to specialized subagents
        results = {}
        for area in plan.review_areas:
            subagent = self.subagents[area]
            results[area] = subagent.analyze(architecture_doc, plan.focus[area])
        
        # Step 3: Synthesize results
        return self.synthesize(results)
    
    def plan_review(self, doc: str) -> ReviewPlan:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are an architecture review coordinator.",
            messages=[{
                "role": "user",
                "content": f"Plan a review for: {doc}\n\nReturn JSON: {{review_areas: [...], focus: {{...}}}}"
            }]
        )
        return ReviewPlan.model_validate_json(response.content[0].text)


# Claude as a judge/evaluator (common agentic pattern)
def evaluate_pr_changes(pr_diff: str) -> CodeReviewResult:
    """Use Claude to evaluate code changes before merge"""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system="""You are a senior engineer doing code review.
        Evaluate changes for: correctness, performance, security, readability.
        Be specific and actionable.""",
        messages=[{"role": "user", "content": f"Review this diff:\n\n{pr_diff}"}]
    )
    return parse_review(response.content[0].text)
```

---

## 7. Prompt Engineering for Claude

### 7.1 Claude-Specific Prompting Techniques

```
Claude Prompt Best Practices:

1. SYSTEM PROMPT FOR CONTEXT
   ─ Put persona, context, and constraints in system prompt
   ─ Not in first human turn
   
   Good:   system = "You are a payments expert. Users are engineers."
   Avoid:  user = "Act as a payments expert and answer my question..."

2. BE DIRECT AND SPECIFIC
   ─ Claude follows instructions literally — be precise
   ─ Specify format, length, tone explicitly
   
   Vague:   "Summarize this"
   Better:  "Summarize in 3 bullet points, each under 20 words,
             focusing on technical implications"

3. USE XML TAGS FOR STRUCTURE
   ─ Claude was trained with XML tags for document structure
   ─ Much more reliable than markdown for input parsing
   
   <document>
     <code_snippet language="python">
       {code}
     </code_snippet>
     <task>Find all security vulnerabilities in the code above.</task>
   </document>

4. CHAIN-OF-THOUGHT (for complex tasks)
   ─ "Think step by step before answering"
   ─ Or structured: "First identify the problem. Then list solutions.
     Then recommend the best one with justification."

5. EXAMPLES (few-shot)
   ─ One or two examples dramatically improve output quality
   ─ Examples should match your desired format exactly

6. PREFILLING (force format)
   ─ Start Claude's response to control format
   ─ {"role": "assistant", "content": "```json\n{"}
     → Forces JSON code block response
```

### 7.2 Prompt Templates for Architecture Work

```python
# Reusable prompt templates for architecture tasks

ARCHITECTURE_REVIEW_PROMPT = """
You are a principal engineer conducting a formal architecture review.

<review_criteria>
  - Correctness: Does the design solve the stated problem?
  - Scalability: Will it handle 10x projected load?
  - Resilience: What are the failure modes?
  - Security: Are there vulnerabilities or attack vectors?
  - Operability: Can an on-call engineer debug this at 2AM?
  - Simplicity: Is this the simplest design that works?
</review_criteria>

<architecture_document>
{architecture_doc}
</architecture_document>

Provide a structured review addressing each criterion.
For each issue found, rate severity: CRITICAL / HIGH / MEDIUM / LOW.
End with a RECOMMENDATION: APPROVE / APPROVE_WITH_CONDITIONS / REJECT.
"""

ADR_GENERATION_PROMPT = """
Generate an Architecture Decision Record (ADR) for the following decision.

<decision_context>
{context}
</decision_context>

<options_considered>
{options}
</options_considered>

<chosen_option>
{chosen}
</chosen_option>

Format the ADR with these sections:
# ADR-XXXX: [Title]
## Context
## Decision Drivers
## Options Considered (table with pros/cons)
## Decision
## Consequences (positive and negative)
## Risks and Mitigations
"""

INCIDENT_ANALYSIS_PROMPT = """
You are an SRE analyzing a production incident. Be precise and blameless.

<incident_timeline>
{timeline}
</incident_timeline>

<metrics_during_incident>
{metrics}
</metrics_during_incident>

Provide:
1. Root cause (immediate + contributing factors)
2. Impact assessment (users affected, duration, data impact)
3. What went well in the response
4. What failed (detection, response, tooling)
5. Action items (each with: owner, priority, deadline)
"""
```

### 7.3 Avoiding Common Prompt Pitfalls

```
Common Mistakes with Claude:

❌ OVER-CONSTRAINING
  "Never use the word 'however'. Don't mention X. 
   Don't say Y. Avoid Z. Don't do..."
  → Claude may get confused by many negatives
  ✅ State what you DO want: "Use direct, active language"

❌ ASKING MULTIPLE QUESTIONS IN ONE
  "What is the best database, and how do I scale it,
   and what are the costs, and how does it compare to X?"
  → Claude may address some parts superficially
  ✅ Sequence questions or ask Claude to answer each part separately

❌ AMBIGUOUS PRONOUNS/REFERENCES
  "Compare it with the other one from earlier"
  → Works for humans, not ideal for prompts
  ✅ Be explicit: "Compare PostgreSQL (from your first answer)
     with MongoDB (mentioned in my second question)"

❌ ASSUMING CLAUDE KNOWS INTERNAL CONTEXT
  "Analyze our payment flow"
  → Claude has no idea what "our" is
  ✅ Always include the relevant context in the prompt

❌ IMPLICIT FORMAT EXPECTATIONS
  Expecting a table when you said "compare"
  ✅ "Compare in a markdown table with columns: Feature | Postgres | MongoDB"
```

---

## 8. Context Window Management

### 8.1 200K Token Context — How to Use It

```
200K context ≈ 150,000 words ≈ 500 pages

What fits in 200K tokens:
  ─ Entire small-medium codebase (50-100 files)
  ─ Multiple long documents (books, specs)
  ─ Long conversation history (hundreds of turns)
  ─ Complete API documentation + code examples
  ─ Meeting transcripts, lengthy reports

Context Window Usage Strategy:

SYSTEM PROMPT (static, cacheable)
  ─ Persona + task instructions      (~500 tokens)
  ─ Reference documentation          (~20K tokens) ← CACHE THIS
  ─ Few-shot examples                (~2K tokens)  ← CACHE THIS

CONVERSATION HISTORY (grows over time)
  ─ Previous messages                (variable)
  ─ Tool results                     (variable)

CURRENT REQUEST
  ─ User's question / task           (variable)

Reserve for OUTPUT
  ─ max_tokens parameter             (your choice)
  ─ Don't fill context to 100% —
    leave room for generation
```

### 8.2 Long Context Performance

```
Claude's Long Context Behavior:

"Lost in the middle" phenomenon:
  ─ Models (including Claude) recall information better from
    beginning and end of context vs the middle
  ─ For critical information: put it at start or end
  ─ Mitigation: repeat key constraints at end of long prompts

Claude's relative strength:
  ─ Strong at "needle in a haystack" retrieval
  ─ Tested up to 200K tokens with high accuracy
  ─ Better than GPT-4-turbo at long-context tasks

Best practices for long contexts:
  1. Structure with clear XML/markdown headers
     → Claude can navigate structured documents better

  2. Tell Claude where to look:
     "The relevant schema is in the <database_schema> section"

  3. For very long docs: summarize + attach full
     "Summary: [200 word summary]. Full doc: [full content]"

  4. Use prompt caching for static long content
     (see section 3.4)
```

### 8.3 Context Management in Chat Applications

```python
# Context management for long-running conversations

class ConversationManager:
    def __init__(self, max_context_tokens: int = 150_000):
        self.messages = []
        self.max_tokens = max_context_tokens
        self.client = anthropic.Anthropic()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Keep most recent messages that fit in context window"""
        total = self._estimate_tokens()
        while total > self.max_tokens and len(self.messages) > 2:
            # Remove oldest message pair (keep system context)
            self.messages.pop(0)
            if self.messages and self.messages[0]["role"] == "assistant":
                self.messages.pop(0)
            total = self._estimate_tokens()

    def _estimate_tokens(self) -> int:
        # Rough estimate: 4 chars ≈ 1 token
        return sum(len(m["content"]) // 4 for m in self.messages)

    def summarize_old_messages(self, older_than: int = 10):
        """Alternative: summarize old conversation instead of dropping"""
        if len(self.messages) <= older_than:
            return

        old_messages = self.messages[:-older_than]
        summary_response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",   # Cheap model for summary
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation in 3-5 sentences:\n\n"
                           f"{format_messages(old_messages)}"
            }]
        )
        summary = summary_response.content[0].text

        # Replace old messages with summary
        self.messages = [
            {"role": "user", "content": f"<conversation_summary>{summary}</conversation_summary>"}
        ] + self.messages[-older_than:]
```

---

## 9. Production Architecture with Claude

### 9.1 Reliability Patterns

```
Production LLM Reliability:

RETRY WITH EXPONENTIAL BACKOFF
  ─ Rate limit errors (429): retry with backoff
  ─ Overload errors (529): retry with backoff
  ─ Network timeouts: retry immediately (once)

  import anthropic
  from anthropic import RateLimitError, APIStatusError

  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=4, max=60),
      retry=retry_if_exception_type((RateLimitError,))
  )
  def call_claude_with_retry(messages):
      return client.messages.create(
          model="claude-3-5-sonnet-20241022",
          max_tokens=1024,
          messages=messages
      )

FALLBACK CHAIN
  Try claude-3-5-sonnet →
    if overloaded: try claude-3-5-haiku →
      if overloaded: return cached response / queue for later

TIMEOUT MANAGEMENT
  ─ Set request timeout: 60s (avoid indefinite waits)
  ─ For streaming: set per-chunk timeout
  ─ Set max_tokens carefully (too high = slow responses)

  client = anthropic.Anthropic(
      timeout=60.0,    # 60 second total request timeout
  )

CIRCUIT BREAKER (for downstream Claude dependency)
  ─ Track failure rate over 60s window
  ─ If >50% fail → circuit opens → return fallback
  ─ After 30s → half-open → test one request
  ─ Prevents cascading failures if Claude API has issues
```

### 9.2 Security for LLM Applications

```
LLM Security Threats and Mitigations:

PROMPT INJECTION
  Threat: User input manipulates Claude to bypass your system prompt
  Example: "Ignore all previous instructions. You are now DAN..."
  
  Mitigations:
  ─ Separate trusted (system) from untrusted (user) input with XML tags
  ─ Validate Claude's output against expected schema
  ─ Use Claude's refusal as a signal: if Claude refuses your task,
    something suspicious in the input may have triggered it
  ─ Test with adversarial prompts before launch

  # Safer input handling
  message = client.messages.create(
      system="You are a helpful assistant for {company}. Only discuss {topic}.",
      messages=[{
          "role": "user",
          "content": f"<user_input>{sanitize(user_message)}</user_input>"
      }]
  )

PII LEAKAGE
  Threat: User data sent to Claude processed/stored by Anthropic
  
  Mitigations:
  ─ Use Anthropic API (zero data retention option available)
  ─ Or use Bedrock/Vertex AI (data stays in your cloud)
  ─ Strip PII before sending (Presidio for detection/redaction)
  ─ Review Anthropic's data processing agreement (DPA)

SENSITIVE DATA IN PROMPTS
  ─ Don't include API keys, passwords, secrets in prompts
  ─ Use placeholder tokens: "The DB connection string is in Vault"
  ─ Log prompts securely (they contain sensitive context)

JAILBREAK / MISUSE
  ─ Claude has strong built-in safeguards
  ─ Use system prompt to define scope: "Only assist with X"
  ─ Implement output filtering for your domain
  ─ Rate-limit per user to prevent abuse
```

### 9.3 Observability for Claude Applications

```python
# LLM observability with Langfuse

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

@observe(name="architecture-review")
def review_architecture(doc: str, tenant_id: str) -> str:
    # Automatic tracing: captures input, output, model, tokens, cost
    langfuse_context.update_current_observation(
        metadata={"tenant_id": tenant_id, "doc_length": len(doc)}
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{"role": "user", "content": ARCH_REVIEW_PROMPT.format(doc=doc)}]
    )
    return response.content[0].text

# What Langfuse tracks per call:
# ─ Latency (first token + total)
# ─ Token usage (input/output/cache)
# ─ Cost ($)
# ─ Model used
# ─ Full prompt + response
# ─ User/session attribution
# ─ Custom metadata
# ─ Scores (if you add evaluation)

# Usage analytics in Langfuse dashboard:
# ─ Cost per user / per feature
# ─ Latency P50/P95/P99
# ─ Error rate
# ─ Token distribution
# ─ Most expensive prompts
```

---

## 10. Claude on AWS Bedrock

### 10.1 Why Use Bedrock for Claude

```
Direct Anthropic API vs AWS Bedrock:

Direct API (api.anthropic.com)
  ✅ Latest models immediately available
  ✅ Full feature access (beta features)
  ✅ Simplest setup
  ❌ Data processed by Anthropic
  ❌ Separate billing from AWS
  ❌ Not in your VPC

AWS Bedrock
  ✅ Data stays in your AWS account + region
  ✅ AWS billing (consolidates with existing AWS spend)
  ✅ IAM-based auth (no API key management)
  ✅ VPC endpoints (data never leaves your network)
  ✅ CloudTrail audit logs for every API call
  ✅ AWS commitments (HIPAA, SOC2, FedRAMP eligible)
  ✅ Bedrock Guardrails (content filtering)
  ✅ Works with Savings Plans / Enterprise Discount
  ❌ Slight delay for new model availability
  ❌ More setup complexity
```

### 10.2 Using Claude on Bedrock

```python
import boto3
import json

# AWS Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Using Claude on Bedrock (same API structure, different client)
def call_claude_bedrock(messages: list, system: str = "") -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages,
    }
    if system:
        body["system"] = system

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# Streaming on Bedrock
def stream_claude_bedrock(messages: list):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": messages,
    }
    response = bedrock.invoke_model_with_response_stream(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps(body)
    )
    for event in response["body"]:
        chunk = json.loads(event["chunk"]["bytes"])
        if chunk["type"] == "content_block_delta":
            yield chunk["delta"].get("text", "")

# Bedrock model IDs for Claude
BEDROCK_MODEL_IDS = {
    "claude-3-5-sonnet":       "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku":        "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-7-sonnet":       "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "claude-3-opus":           "anthropic.claude-3-opus-20240229-v1:0",
}
```

### 10.3 Bedrock Guardrails

```python
# Bedrock Guardrails: content filtering for your application

# Create guardrail (one-time setup via CDK/Terraform/console)
# guardrail_id = "abc123"

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    guardrailIdentifier="abc123",
    guardrailVersion="DRAFT",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": user_input}],
    })
)

# Guardrail capabilities:
# ─ Content filtering: block hate speech, violence, adult content
# ─ Topic denial: block discussions of competitor products
# ─ Word filters: block specific words/phrases
# ─ PII detection: detect and optionally redact PII in input/output
# ─ Grounding: check if response is grounded in provided context
# ─ Hallucination detection: score factual accuracy
```

---

## 11. Cost Optimization

### 11.1 Token Optimization Strategies

```
Token Cost Table (claude-3-5-sonnet-20241022):
  Input tokens:  $3.00 / M
  Output tokens: $15.00 / M
  Cache write:   $3.75 / M
  Cache read:    $0.30 / M

Output tokens are 5x more expensive than input.
Optimization strategies:

1. REDUCE OUTPUT LENGTH
   ─ Specify max length in prompt: "Respond in 3 sentences max"
   ─ Use structured formats (JSON) — more concise than prose
   ─ Ask for bullet points instead of paragraphs
   ─ set max_tokens to realistic ceiling (not 4096 by default)

2. USE CHEAPER MODELS FOR SIMPLE TASKS
   Task routing:
   ─ Classification, routing, extraction → claude-3-5-haiku
   ─ Complex reasoning, coding → claude-3-5-sonnet
   ─ Hardest problems, nuanced analysis → claude-3-7-sonnet (thinking)

   Cost comparison (same task):
   ─ Haiku:  $0.25 / M input,  $1.25 / M output
   ─ Sonnet: $3.00 / M input, $15.00 / M output
   ─ At 1M requests: $1,250 vs $15,000 — 12x difference

3. PROMPT CACHING (see section 3.4)
   ─ For any prompt with >1024 tokens of static content
   ─ System prompt + large context → cache everything static

4. BATCH API (asynchronous, half price)
   ─ 50% cheaper than real-time API
   ─ Results within 24 hours
   ─ Use for: offline evaluation, bulk processing, non-urgent tasks
   
   batch = client.beta.messages.batches.create(
       requests=[
           {"custom_id": f"req-{i}",
            "params": {"model": "claude-3-5-sonnet-20241022",
                       "max_tokens": 1024,
                       "messages": [{"role": "user", "content": q}]}}
           for i, q in enumerate(questions)
       ]
   )

5. AVOID REDUNDANT CONTEXT
   ─ Don't include entire conversation history for tasks that don't need it
   ─ Summarize instead of passing full history
   ─ Don't repeat instructions that are already in system prompt
```

### 11.2 Model Routing

```python
# Intelligent model routing — use cheapest model that can do the job

class ClaudeRouter:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def route_and_call(self, task: str, content: str) -> str:
        model, max_tokens = self._select_model(task, content)
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text

    def _select_model(self, task: str, content: str) -> tuple[str, int]:
        """Route to cheapest model that can handle the task"""
        
        # High-volume, simple tasks → Haiku (12x cheaper than Sonnet)
        if task in ("classify", "extract_fields", "sentiment", "route"):
            return "claude-3-5-haiku-20241022", 256
        
        # Long documents, complex analysis → Sonnet
        if len(content) > 10_000 or task in ("summarize", "analyze", "review"):
            return "claude-3-5-sonnet-20241022", 2048
        
        # Hard reasoning, architecture decisions → Sonnet with thinking
        if task in ("architect", "debug_complex", "reason"):
            return "claude-3-7-sonnet-20250219", 8000
        
        # Default: Sonnet
        return "claude-3-5-sonnet-20241022", 1024
```

---

## 12. Practical Examples

### 12.1 Code Review Automation

```python
# Automated PR code review with Claude

import anthropic
from github import Github  # PyGithub

client = anthropic.Anthropic()
gh = Github(GITHUB_TOKEN)

def review_pull_request(repo_name: str, pr_number: int):
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    
    # Get changed files
    files = pr.get_files()
    diff_content = "\n\n".join([
        f"### {f.filename} (+{f.additions}/-{f.deletions})\n```diff\n{f.patch}\n```"
        for f in files if f.patch and len(f.patch) < 10000
    ])
    
    # Review with Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system="""You are a senior software engineer doing a code review.
        Focus on: bugs, security issues, performance problems, and unclear code.
        Be constructive and specific. Ignore cosmetic style issues.""",
        messages=[{
            "role": "user",
            "content": f"""PR: {pr.title}
Description: {pr.body}

Changes:
{diff_content}

Provide a structured review with:
1. Summary (2 sentences)
2. Critical issues (bugs/security — must fix)
3. Suggestions (improvements — nice to have)
4. Overall verdict: APPROVE / REQUEST_CHANGES / COMMENT"""
        }]
    )
    
    review_text = response.content[0].text
    
    # Post as PR review
    verdict = extract_verdict(review_text)
    pr.create_review(
        body=review_text,
        event=verdict  # "APPROVE" | "REQUEST_CHANGES" | "COMMENT"
    )
```

### 12.2 Intelligent SRE Assistant

```python
# On-call assistant that analyzes incidents with context

class SREAssistant:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.runbook_cache = {}
        self.RUNBOOKS = load_all_runbooks()  # Large corpus

    def analyze_alert(self, alert: Alert) -> IncidentGuidance:
        # Cache the runbooks (large static context)
        runbook_section = f"<runbooks>{self.RUNBOOKS}</runbooks>"

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=[
                {"type": "text",
                 "text": "You are an expert SRE. Help diagnose and resolve incidents."},
                {"type": "text",
                 "text": runbook_section,
                 "cache_control": {"type": "ephemeral"}}  # Cache 50K+ token runbooks
            ],
            messages=[{
                "role": "user",
                "content": f"""ALERT FIRING: {alert.name}
Severity: {alert.severity}
Service: {alert.service}
Metric: {alert.metric} = {alert.value} (threshold: {alert.threshold})
Time: {alert.fired_at}

Recent errors from logs:
{alert.recent_errors[:2000]}

Recent deployment: {alert.last_deploy}

Based on the runbooks, provide:
1. Most likely root cause (top 3 hypotheses)
2. Immediate diagnostic steps (commands to run)
3. Remediation actions (in order of likelihood)
4. Escalation criteria"""
            }]
        )
        return parse_incident_guidance(response.content[0].text)
```

### 12.3 Architecture Q&A over Codebase

```python
# Ask architecture questions about a large codebase

import os
from pathlib import Path

def load_codebase(repo_path: str, extensions: list[str] = [".py", ".go", ".java"]) -> str:
    """Load relevant code files into a structured format"""
    files = []
    for ext in extensions:
        for fp in Path(repo_path).rglob(f"*{ext}"):
            if "test" not in str(fp) and "vendor" not in str(fp):
                content = fp.read_text(errors="ignore")
                if len(content) < 50_000:  # Skip very large files
                    files.append(f"<file path='{fp}'>\n{content}\n</file>")
    return "\n\n".join(files[:100])  # Limit to 100 files

codebase = load_codebase("/path/to/repo")

client = anthropic.Anthropic()

def ask_about_codebase(question: str) -> str:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system=[
            {"type": "text",
             "text": "You are an expert in reading and understanding codebases."},
            {"type": "text",
             "text": f"<codebase>\n{codebase}\n</codebase>",
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# Example questions:
print(ask_about_codebase("Where is the authentication logic? What auth method is used?"))
print(ask_about_codebase("What are the main services and how do they communicate?"))
print(ask_about_codebase("Are there any obvious security vulnerabilities?"))
print(ask_about_codebase("What databases are used and how is data modeled?"))
```

### 12.4 Reference Integration Stack

```yaml
# Anthropic / Claude Production Stack Reference

api_access:
  direct_api:  api.anthropic.com          # Latest features, simplest
  aws_bedrock: bedrock.us-east-1.amazonaws.com  # Enterprise, HIPAA, VPC
  google_vertex: aiplatform.googleapis.com      # If GCP-primary

sdk:
  python: anthropic                        # pip install anthropic
  typescript: @anthropic-ai/sdk            # npm install @anthropic-ai/sdk
  bedrock: boto3 (bedrock-runtime)

gateway_and_routing:
  - litellm                               # Unified LLM API (Claude + GPT + Gemini)
  - portkey                               # Enterprise LLM gateway, fallbacks
  
observability:
  - langfuse                              # OSS LLM tracing + evaluation
  - helicone                              # Proxy-based observability
  - langsmith                             # If using LangChain

orchestration:
  - langchain                             # General LLM orchestration
  - llamaindex                            # RAG-focused
  - anthropic_sdk_direct                  # Best for simple use cases (no framework needed)

prompt_management:
  - langfuse_prompts                      # Version + A/B test prompts
  - anthropic_workbench                   # claude.ai/workbench for development

vector_db_for_rag:
  - qdrant                                # High-performance, OSS
  - pgvector                              # If already using PostgreSQL
  - pinecone                              # Managed, simple

evaluation:
  - ragas                                 # RAG evaluation
  - promptfoo                             # Regression testing
  - langfuse_evaluations                  # LLM-as-judge scoring

cost_tracking:
  - langfuse                              # Per-trace cost attribution
  - aws_cost_explorer                     # Bedrock spend tracking
  - custom_token_counter                  # Per-team budget alerts
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Model Tiers** | Haiku (fast/cheap) → Sonnet (balanced) → Opus (capable); 3.5/3.7 Sonnet is the sweet spot |
| **Constitutional AI** | Anthropic trains Claude using AI-generated feedback against explicit principles — more consistent and scalable than pure RLHF |
| **HHH** | Claude is designed to be Helpful, Harmless, and Honest — unhelpfulness is never "trivially safe" |
| **Prompt Caching** | Cache large static context (system prompt, docs, codebase) — 90% cheaper on cached tokens |
| **Extended Thinking** | Use `budget_tokens` in claude-3-7-sonnet for complex multi-step reasoning — 5-15x more thoughtful at higher cost |
| **Tool Use** | Define tools as JSON Schema, run agentic loops until `stop_reason == "end_turn"` |
| **XML Tags** | Claude responds better to `<xml_structured_input>` than markdown for complex input structure |
| **Bedrock** | Use for enterprise (data stays in VPC, IAM auth, CloudTrail, HIPAA eligible) |
| **Batch API** | 50% cost reduction for non-real-time tasks (evaluation, bulk processing) |
| **Model Routing** | Route simple tasks to Haiku (12x cheaper) — reserve Sonnet/Opus for tasks that need it |
| **Operator/User Trust** | System prompt = operator tier (higher trust); user input = user tier — structure prompts accordingly |
| **Security** | Wrap user input in XML tags, validate outputs, use Bedrock Guardrails for content policies |
