# Large Language Models (LLMs) & Retrieval-Augmented Generation (RAG): Complete Deep-Dive

## Table of Contents
1. [LLM Architecture Deep Dive](#1-llm-architecture-deep-dive)
2. [Context Windows and Long-Context Models](#2-context-windows-and-long-context-models)
3. [Tokenization and Vocabularies](#3-tokenization-and-vocabularies)
4. [Inference: Decoding Strategies](#4-inference-decoding-strategies)
5. [Prompt Engineering](#5-prompt-engineering)
6. [Retrieval-Augmented Generation (RAG)](#6-retrieval-augmented-generation-rag)
7. [Embeddings for RAG](#7-embeddings-for-rag)
8. [Vector Databases](#8-vector-databases)
9. [Chunking Strategies](#9-chunking-strategies)
10. [Reranking](#10-reranking)
11. [Advanced RAG Techniques](#11-advanced-rag-techniques)
12. [LLM Evaluation](#12-llm-evaluation)
13. [Structured Output: JSON Mode and Function Calling](#13-structured-output)
14. [Full Production RAG Pipeline](#14-full-production-rag-pipeline)
15. [Production Concerns: Caching, Observability, Latency](#15-production-concerns)
16. [Common Pitfalls](#common-pitfalls)

---

## 1. LLM Architecture Deep Dive

### 1.1 The Autoregressive Decoder

All major LLMs (GPT, LLaMA, Mistral, Mixtral) are **autoregressive decoder-only transformers** trained to predict the next token given all previous tokens:

\[
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
\]

The key structural difference from BERT: all attention is **causal (masked)**, so each token can only attend to itself and previous tokens.

### 1.2 GPT Architecture

```
Input tokens: [x_1, x_2, ..., x_T]
     ↓
Token Embedding (vocab_size → d_model)
     ↓  +
Positional Encoding
     ↓
┌─────────────────────────────────────────┐
│ Transformer Block × N                   │
│                                         │
│  Input → LayerNorm                      │
│       → Causal Self-Attention           │
│       → Residual Add                    │
│       → LayerNorm                       │
│       → Feed-Forward (SwiGLU/GELU)      │
│       → Residual Add                    │
└─────────────────────────────────────────┘
     ↓
Final LayerNorm
     ↓
Linear head (d_model → vocab_size)
     ↓
Softmax → Token probabilities
```

### 1.3 LLaMA Architecture

LLaMA (Meta, 2023) makes several improvements over GPT:

| Component | GPT-2/3 | LLaMA |
|-----------|---------|-------|
| Normalization | Post-LN | Pre-RMSNorm |
| Activation | GELU | SwiGLU |
| Position | Learned/Sinusoidal | RoPE (Rotary) |
| Attention | MHA | GQA (LLaMA-2/3) |
| Biases | Yes | No (except QKV) |

**RMSNorm** (Root Mean Square Normalization) — simpler than LayerNorm:

\[
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2}
\]

**SwiGLU** activation (used in LLaMA FFN):

\[
\text{SwiGLU}(x, W, V, W_2) = (x W \odot \text{swish}(x V)) W_2
\]

where \( \text{swish}(x) = x \cdot \sigma(x) \). This uses 3 weight matrices instead of 2.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x * self.weight / (rms + self.eps)


class SwiGLU(nn.Module):
    """SwiGLU activation (Shazeer, 2020)."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Typically hidden_dim = int(8/3 * dim) rounded to multiple of 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMAAttention(nn.Module):
    """LLaMA-style attention with GQA and RoPE."""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
    
    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        Q, K = apply_rope(Q, K, cos, sin)
        
        # Repeat K, V for grouped query attention
        if self.num_groups > 1:
            K = K.repeat_interleave(self.num_groups, dim=1)
            V = V.repeat_interleave(self.num_groups, dim=1)
        
        # Flash attention (if available)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(output)


class LLaMABlock(nn.Module):
    """Single LLaMA transformer block."""
    
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = LLaMAAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
    
    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)  # Pre-norm + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

### 1.4 Mistral Architecture

Mistral-7B (Jiang et al., 2023) adds:
1. **Sliding Window Attention (SWA):** Each token only attends to W=4096 previous tokens. Reduces complexity from \( O(n^2) \) to \( O(n \cdot W) \). KV cache grows linearly, not quadratically.
2. **Rolling KV Cache:** KV cache has fixed size W; old K/V are overwritten cyclically.
3. **GQA** (8 KV heads for 32 query heads)

### 1.5 Mixtral: Mixture of Experts

Mixtral-8x7B uses **Sparse Mixture of Experts (MoE)** to scale parameter count without scaling compute:

\[
\text{MoE}(x) = \sum_{i=1}^{K} G(x)_i \cdot E_i(x)
\]

**Router (gating function):**
\[
G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g, 2))
\]

Only top-K=2 experts are activated per token. With 8 experts but only 2 active, Mixtral has 47B parameters but uses only 13B per token (same compute as 13B model).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Mistral
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"       # Automatic device placement
)

# Inspect model architecture
print(model.config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# Load Mixtral
mixtral = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True   # 4-bit quantization
)
```

### 1.6 Model Size and Compute Estimates

**Chinchilla scaling laws** (Hoffmann et al., 2022):

For a given compute budget \( C \) FLOPs:
- Optimal model size: \( N^* \approx \sqrt{C / 6} \)
- Optimal training tokens: \( D^* \approx \sqrt{6C} \)
- Rule of thumb: 20 tokens per parameter

**FLOPs estimate per forward pass:**
\[
\text{FLOPs} \approx 6ND \quad \text{(training)}, \quad \approx 2N \quad \text{(per token inference)}
\]

| Model | Params | Context | Training Tokens |
|-------|--------|---------|-----------------|
| GPT-3 | 175B | 4K | 300B |
| LLaMA-2-7B | 7B | 4K | 2T |
| LLaMA-2-70B | 70B | 4K | 2T |
| LLaMA-3-8B | 8B | 8K | 15T |
| Mixtral-8x7B | 47B | 32K | N/A |
| Mistral-7B | 7B | 32K | N/A |

---

## 2. Context Windows and Long-Context Models

### 2.1 The KV Cache

During inference, for each new token, we need K and V for all previous positions. The **KV cache** stores these:

**Memory cost:** \( 2 \times \text{num\_layers} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{dtype\_bytes} \)

For LLaMA-2-7B (FP16): \( 2 \times 32 \times 8 \times 128 \times L \times 2 = 131,072 \times L \) bytes ≈ 128KB per token

For 4K context: ~512MB just for KV cache.

### 2.2 Extending Context Length

**Position Interpolation (Chen et al., 2023):**
Scale positions by \( L_{\text{train}} / L_{\text{target}} \) so that extended positions map to the trained range:

\[
\text{RoPE}(pos \cdot \frac{L_{\text{train}}}{L_{\text{target}}})
\]

**YaRN (Peng et al., 2023):** More sophisticated interpolation that applies different scales per frequency component.

**LongRoPE / Llama-3 approach:** Uses NTK-aware interpolation with different base frequencies.

```python
# Using Llama-3 with extended context (8K default, can do 128K with special variants)
from transformers import AutoTokenizer, AutoModelForCausalLM

# LongLLaMA or models with extended context
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# Enable sliding window or context extension
model.config.max_position_embeddings = 16384

# Efficient long-context processing
def process_long_document(text: str, model, tokenizer, max_chunk: int = 4096):
    """Process long document by chunking."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_chunk] for i in range(0, len(tokens), max_chunk-512)]  # 512 overlap
    
    results = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk])
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=200)
        results.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return results
```

---

## 3. Tokenization and Vocabularies

### 3.1 Vocabulary Design

| Model | Tokenizer | Vocab Size | Notes |
|-------|-----------|------------|-------|
| GPT-2 | BPE | 50,257 | Byte-level |
| GPT-4 | cl100k_base | 100,277 | Tiktoken |
| LLaMA-2 | SentencePiece BPE | 32,000 | |
| LLaMA-3 | tiktoken BPE | 128,256 | Better multilingual |
| Mistral | SentencePiece | 32,000 | |
| Gemma | SentencePiece | 256,000 | |

```python
import tiktoken
from transformers import AutoTokenizer

# GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")
text = "Hello, world! This is a test of GPT-4's tokenizer."
tokens = enc.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
print(f"Decoded: {enc.decode(tokens)}")

# Tiktoken for LLaMA-3
llama3_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokens = llama3_tok.encode("Hello, world!")
print(f"LLaMA-3 tokens: {tokens}")
print(f"Decoded: {llama3_tok.decode(tokens)}")

# Token counting for cost estimation
def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding_name = "cl100k_base" if "gpt-4" in model or "gpt-3.5" in model else "p50k_base"
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

# Estimate API cost
def estimate_cost(prompt: str, response: str, model: str = "gpt-4o") -> float:
    prices = {
        "gpt-4o": {"input": 2.50, "output": 10.00},     # per 1M tokens
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    price = prices.get(model, {"input": 1.0, "output": 2.0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000
```

---

## 4. Inference: Decoding Strategies

### 4.1 Greedy Decoding

Select the highest-probability token at each step:

\[
x_t = \arg\max_{v \in V} P(v \mid x_1, \ldots, x_{t-1})
\]

Deterministic, fast, but often repetitive and suboptimal.

### 4.2 Beam Search

Maintain the top-B most probable sequences:

\[
\mathcal{H}_t = \text{TopB}\left\{ (\text{seq} + v, \log P(\text{seq}) + \log P(v \mid \text{seq})) : \text{seq} \in \mathcal{H}_{t-1}, v \in V \right\}
\]

Better for structured outputs (translation, code) but tends to produce generic, repetitive text.

**Penalties:**
- **Length penalty:** \( \text{score}(y) = \frac{\log P(y)}{|y|^\alpha} \) — encourages longer sequences when \( \alpha > 0 \)
- **No-repeat ngram:** Prevents generating same n-gram twice

### 4.3 Temperature Sampling

Scale logits before softmax to control randomness:

\[
P(x_t = v) = \frac{\exp(z_v / T)}{\sum_u \exp(z_u / T)}
\]

- \( T \to 0 \): Deterministic (argmax)
- \( T = 1 \): Standard softmax
- \( T > 1 \): More uniform distribution (creative)
- \( T < 1 \): More peaked (conservative)

### 4.4 Top-K Sampling

Sample only from the K most probable tokens:

```python
def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> int:
    logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(logits, k)
    probs = F.softmax(top_k_values, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return top_k_indices[sampled_idx].item()
```

### 4.5 Top-P (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds p:

\[
V_p = \min\!\left\{ V' \subseteq V : \sum_{v \in V'} P(v \mid x_{<t}) \geq p \right\}
\]

More adaptive than top-k — dynamically adjusts the number of tokens based on the distribution shape.

```python
def top_p_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> int:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens where cumulative prob exceeds p
    # Shift by 1 to keep the token that crosses threshold
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p
    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs /= sorted_probs.sum()
    
    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_indices[sampled_idx].item()

# Typical combined settings
def generate_response(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Return only newly generated tokens
    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```

### 4.6 Speculative Decoding

Use a small draft model to generate K tokens, then verify with the large model in a single forward pass. Speed up: 2-3x with minimal quality loss.

```python
def speculative_decode(
    draft_model, target_model, tokenizer,
    prompt: str, max_new_tokens: int = 200, K: int = 4
) -> str:
    """Speculative decoding: use draft model to generate K tokens, verify with target."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    generated = input_ids.clone()
    
    while generated.shape[1] < input_ids.shape[1] + max_new_tokens:
        # Draft: generate K tokens speculatively
        draft_ids = generated.clone()
        for _ in range(K):
            with torch.no_grad():
                draft_logits = draft_model(draft_ids).logits[:, -1, :]
            next_token = draft_logits.argmax(-1, keepdim=True)
            draft_ids = torch.cat([draft_ids, next_token], dim=1)
        
        # Target: evaluate all K+1 positions in one pass
        with torch.no_grad():
            target_logits = target_model(draft_ids).logits
        
        # Accept/reject each draft token
        n_accepted = 0
        for i in range(K):
            t = generated.shape[1] + i - input_ids.shape[1]
            target_prob = F.softmax(target_logits[0, generated.shape[1]+i-1, :], dim=-1)
            draft_token = draft_ids[0, generated.shape[1]+i]
            
            # Accept with probability min(1, P_target / P_draft)
            draft_prob = F.softmax(draft_model(draft_ids[:, :generated.shape[1]+i]).logits[0, -1, :], dim=-1)
            acceptance_prob = min(1.0, (target_prob[draft_token] / (draft_prob[draft_token] + 1e-10)).item())
            
            if torch.rand(1).item() < acceptance_prob:
                n_accepted += 1
            else:
                break
        
        # Take accepted tokens + one from target
        accepted_tokens = draft_ids[:, generated.shape[1]:generated.shape[1]+n_accepted+1]
        generated = torch.cat([generated, accepted_tokens], dim=1)
        
        if tokenizer.eos_token_id in accepted_tokens[0]:
            break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

---

## 5. Prompt Engineering

### 5.1 Zero-Shot Prompting

No examples — rely on model's pre-trained knowledge:

```python
from openai import OpenAI

client = OpenAI()

def zero_shot(task: str, input_text: str, model: str = "gpt-4o") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": f"{task}\n\nText: {input_text}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content

# Examples
sentiment = zero_shot("Classify the sentiment as positive/negative/neutral.", 
                       "The movie was surprisingly disappointing.")
print(sentiment)  # negative
```

### 5.2 Few-Shot Prompting

Provide examples to guide format and behavior:

```python
def few_shot_classify(text: str) -> str:
    prompt = """Classify each review as positive or negative.

Review: "The food was incredible, best restaurant in town!"
Sentiment: positive

Review: "Waited 2 hours and the food was cold and tasteless."
Sentiment: negative

Review: "Decent place, nothing extraordinary."
Sentiment: neutral

Review: "{text}"
Sentiment:""".format(text=text)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()
```

### 5.3 Chain-of-Thought (CoT) Prompting

Wei et al. (2022): Adding "Let's think step by step" dramatically improves reasoning:

```python
def chain_of_thought(problem: str, model: str = "gpt-4o") -> dict:
    """CoT with structured reasoning extraction."""
    
    cot_prompt = f"""Solve the following problem. Think step by step, show your work,
then give the final answer.

Problem: {problem}

Solution:
Let me think through this step by step."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": cot_prompt}],
        temperature=0.0
    )
    
    full_response = response.choices[0].message.content
    
    # Parse reasoning and answer
    return {
        "reasoning": full_response,
        "tokens_used": response.usage.total_tokens
    }

# Example
result = chain_of_thought(
    "A train leaves city A at 9am going 60mph. Another train leaves city B at 11am "
    "going 80mph toward city A. Cities are 320 miles apart. When do they meet?"
)
print(result['reasoning'])
```

### 5.4 Self-Consistency

Generate multiple reasoning paths, take majority vote:

```python
from collections import Counter
import re

def self_consistency(problem: str, n_samples: int = 10, temperature: float = 0.7) -> str:
    """Sample multiple reasoning paths, extract answers, return majority."""
    
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Solve step by step: {problem}"}
            ],
            temperature=temperature
        )
        text = response.choices[0].message.content
        
        # Extract final numeric answer
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            answers.append(numbers[-1])  # Typically the last number is the answer
    
    if not answers:
        return "Unable to determine"
    
    # Majority vote
    most_common = Counter(answers).most_common(1)[0]
    return f"{most_common[0]} (confidence: {most_common[1]/n_samples:.0%})"
```

### 5.5 ReAct: Reasoning + Acting

ReAct (Yao et al., 2022) interleaves reasoning and tool use:

```python
import json

class ReActAgent:
    """ReAct agent with tool use."""
    
    def __init__(self, tools: dict):
        self.tools = tools
        self.history = []
    
    def run(self, question: str, max_steps: int = 10) -> str:
        system_prompt = f"""You are a helpful assistant that can use tools to answer questions.
        
Available tools: {json.dumps(list(self.tools.keys()), indent=2)}

Format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to tool>

After you have enough information:
Thought: I now have enough information to answer.
Final Answer: <your answer>"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        for step in range(max_steps):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                stop=["Observation:"]
            )
            
            text = response.choices[0].message.content
            messages.append({"role": "assistant", "content": text})
            
            # Check if done
            if "Final Answer:" in text:
                return text.split("Final Answer:")[-1].strip()
            
            # Parse action
            if "Action:" in text and "Action Input:" in text:
                action = text.split("Action:")[-1].split("\n")[0].strip()
                action_input = text.split("Action Input:")[-1].strip()
                
                # Execute tool
                if action in self.tools:
                    observation = self.tools[action](action_input)
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {observation}"
                    })
        
        return "Max steps reached without final answer"

# Define tools
def search_tool(query: str) -> str:
    # In production, call a real search API
    return f"Search results for '{query}': [relevant documents here]"

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

agent = ReActAgent({"search": search_tool, "calculator": calculator_tool})
result = agent.run("What is the square root of the number of states in the USA?")
print(result)
```

### 5.6 System Prompts and Personas

```python
SYSTEM_PROMPTS = {
    "data_analyst": """You are an expert data analyst with deep knowledge of statistics, 
    Python (pandas, numpy, matplotlib), and SQL. When given a dataset or question, 
    provide thorough analysis with code examples. Always:
    - State your assumptions explicitly
    - Validate data quality before analysis
    - Provide actionable insights
    - Format code in proper markdown code blocks""",
    
    "rag_assistant": """You are a helpful assistant that answers questions based ONLY 
    on the provided context. Rules:
    1. Only use information from the context to answer
    2. If the answer is not in the context, say "I don't have enough information"
    3. Always cite which part of the context you're using
    4. Do not make up or infer information not present""",
    
    "json_extractor": """You are a JSON extraction assistant. Always respond with valid JSON only.
    Do not include any explanation or markdown formatting.
    If information is missing, use null for the value."""
}
```

---

## 6. Retrieval-Augmented Generation (RAG)

### 6.1 Why RAG?

LLMs have limitations:
- **Knowledge cutoff:** Pre-training data has a date
- **Hallucination:** Confidently generate false information
- **Context window:** Can't hold entire knowledge bases in context
- **Domain specificity:** Generic models lack specialized domain knowledge
- **Attribution:** Hard to cite sources from parametric memory

RAG addresses all of these by grounding generation in retrieved documents.

### 6.2 Naive RAG Pipeline

```
Query → Embed Query → Similarity Search → Retrieved Docs → Augmented Prompt → LLM → Response
```

```python
from typing import List, Dict, Optional
import numpy as np

class NaiveRAG:
    """Basic RAG pipeline for educational purposes."""
    
    def __init__(self, embedding_model, llm_client):
        self.embedding_model = embedding_model
        self.llm = llm_client
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def index_documents(self, documents: List[str]):
        """Embed and store all documents."""
        self.documents = documents
        print(f"Indexing {len(documents)} documents...")
        self.embeddings = self.embedding_model.encode(
            documents, show_progress_bar=True, convert_to_numpy=True
        )
        print(f"Embeddings shape: {self.embeddings.shape}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant documents for a query."""
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Cosine similarity
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        doc_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        scores = (query_norm @ doc_norm.T).flatten()
        
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"document": self.documents[i], "score": float(scores[i])}
            for i in top_indices
        ]
    
    def generate(self, query: str, retrieved: List[Dict]) -> str:
        """Generate answer grounded in retrieved documents."""
        context = "\n\n---\n\n".join([f"[{i+1}] {r['document']}" for i, r in enumerate(retrieved)])
        
        prompt = f"""Answer the question based on the following context.
        
Context:
{context}

Question: {query}

Answer (cite the document numbers you used):"""
        
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        retrieved = self.retrieve(question, top_k)
        answer = self.generate(question, retrieved)
        return {"answer": answer, "sources": retrieved}
```

### 6.3 RAGAS: RAG Evaluation Framework

RAGAS measures RAG quality with 4 key metrics:

1. **Faithfulness:** Is the answer grounded in the retrieved context? (LLM-judged)
2. **Answer Relevancy:** Does the answer address the question? (Embedding similarity)
3. **Context Precision:** Are retrieved chunks relevant? (LLM-judged)
4. **Context Recall:** Do retrieved chunks cover the ground truth answer?

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the transformer architecture?",
        "Who invented BERT?"
    ],
    "answer": [
        "The transformer architecture uses self-attention mechanisms...",
        "BERT was developed by Google Brain researchers..."
    ],
    "contexts": [
        ["Transformers use self-attention to process sequences in parallel..."],
        ["BERT, developed by Devlin et al. at Google, introduced MLM..."]
    ],
    "ground_truth": [
        "A transformer is a deep learning model using self-attention",
        "BERT was invented by Jacob Devlin at Google"
    ]
}

dataset = Dataset.from_dict(eval_data)
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result.to_pandas())

# Manual faithfulness check using LLM
def check_faithfulness(answer: str, contexts: List[str], llm_client) -> float:
    """Check if answer claims are supported by contexts."""
    
    context_str = "\n".join(contexts)
    prompt = f"""Given this context:
{context_str}

And this answer:
{answer}

Rate each claim in the answer on whether it's supported by the context (1=supported, 0=not supported).
Return a JSON list of support ratings."""
    
    response = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    # Parse and average scores
    import json
    scores = json.loads(response.choices[0].message.content)
    return sum(scores) / len(scores) if scores else 0.0
```

---

## 7. Embeddings for RAG

### 7.1 Dense Retrieval

Dense retrieval uses neural embeddings + approximate nearest neighbor (ANN) search.

```python
from sentence_transformers import SentenceTransformer
import torch

# Embedding model selection guide
EMBEDDING_MODELS = {
    # Speed: fast, Quality: good, Size: 22M params, Dim: 384
    "all-MiniLM-L6-v2": {"dim": 384, "max_tokens": 256},
    
    # Speed: medium, Quality: better, Size: 110M, Dim: 768
    "all-mpnet-base-v2": {"dim": 768, "max_tokens": 384},
    
    # Speed: fast, Quality: excellent, Multilingual
    "paraphrase-multilingual-MiniLM-L12-v2": {"dim": 384, "max_tokens": 128},
    
    # State-of-art for RAG
    "BAAI/bge-large-en-v1.5": {"dim": 1024, "max_tokens": 512},
    
    # OpenAI embeddings (API)
    "text-embedding-3-small": {"dim": 1536},
    "text-embedding-3-large": {"dim": 3072},
}

# Using OpenAI embeddings
from openai import OpenAI

def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get OpenAI embedding."""
    response = OpenAI().embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Batch embedding with HuggingFace
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def batch_embed(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed texts in batches for efficiency."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # BGE models need a prefix for queries
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,  # L2 normalize for cosine sim
            show_progress_bar=False
        )
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)
```

### 7.2 Sparse Retrieval: BM25

BM25 is the gold standard for lexical search:

\[
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
\]

where \( k_1 = 1.5 \) (term saturation), \( b = 0.75 \) (length normalization).

```python
from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        tokenized = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {"document": self.corpus[i], "score": float(scores[i])}
            for i in top_indices if scores[i] > 0
        ]
```

### 7.3 Hybrid Search (BM25 + Dense)

Hybrid search combines **sparse** (lexical, e.g., BM25) and **dense** (semantic, embeddings) retrieval. Sparse excels at exact term matches; dense captures meaning. Together they improve recall.

**Fusion strategies:**

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **RRF** | \( \text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)} \) | Default; robust, no score calibration |
| **Linear combo** | \( \alpha \cdot \text{dense} + (1-\alpha) \cdot \text{sparse} \) | When scores are normalized (e.g., 0–1) |
| **Reciprocal score** | \( \frac{1}{1 + \text{rank}} \) | Similar to RRF |
| **Weighted RRF** | \( \alpha \cdot \text{RRF}_{\text{dense}} + (1-\alpha) \cdot \text{RRF}_{\text{sparse}} \) | Tune \( \alpha \) per domain |

Combine sparse and dense retrieval using **Reciprocal Rank Fusion (RRF)**:

\[
\text{RRF\_score}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}
\]

where \( k = 60 \) is a constant.

```python
class HybridRetriever:
    """Combines BM25 (sparse) and dense retrieval."""
    
    def __init__(self, corpus: List[str], embed_model, k: int = 60, alpha: float = 0.5):
        self.corpus = corpus
        self.bm25 = BM25Retriever(corpus)
        self.embed_model = embed_model
        self.embeddings = batch_embed(corpus)
        self.k = k
        self.alpha = alpha  # Weight for dense (1-alpha for sparse)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        # Sparse retrieval
        sparse_results = self.bm25.retrieve(query, top_k=top_k*2)
        sparse_ranks = {r['document']: i for i, r in enumerate(sparse_results)}
        
        # Dense retrieval
        query_emb = self.embed_model.encode([query], normalize_embeddings=True)
        dense_scores = (query_emb @ self.embeddings.T).flatten()
        dense_top = dense_scores.argsort()[::-1][:top_k*2]
        dense_ranks = {self.corpus[i]: j for j, i in enumerate(dense_top)}
        
        # RRF fusion
        all_docs = set(sparse_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores = {}
        for doc in all_docs:
            sparse_rrf = 1 / (self.k + sparse_ranks.get(doc, top_k*2))
            dense_rrf = 1 / (self.k + dense_ranks.get(doc, top_k*2))
            rrf_scores[doc] = self.alpha * dense_rrf + (1 - self.alpha) * sparse_rrf
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"document": doc, "score": score} for doc, score in sorted_docs[:top_k]]
```

---

## 8. Vector Databases

### 8.1 FAISS: Facebook AI Similarity Search

FAISS provides exact and approximate nearest neighbor search at scale.

```python
import faiss
import numpy as np

class FAISSIndex:
    """FAISS vector store wrapper."""
    
    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.documents: List[str] = []
        self.metadata: List[dict] = []
        
        if index_type == "flat":
            # Exact search — best for <100K docs
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
        
        elif index_type == "ivf":
            # Approximate — good for 100K-10M docs
            quantizer = faiss.IndexFlatIP(dim)
            n_clusters = min(4096, max(100, len(self.documents) // 40))
            self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        
        elif index_type == "hnsw":
            # Hierarchical NSW — great recall, GPU not needed
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors
            self.index.hnsw.efConstruction = 64
            self.index.hnsw.efSearch = 32
        
        elif index_type == "ivfpq":
            # IVF + Product Quantization — smallest memory for large scale
            quantizer = faiss.IndexFlatIP(dim)
            m = min(16, dim // 4)  # Number of sub-spaces
            self.index = faiss.IndexIVFPQ(quantizer, dim, 256, m, 8)
        
        # Move to GPU if available
        if faiss.get_num_gpus() > 0 and index_type == "flat":
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
    
    def add(self, texts: List[str], embeddings: np.ndarray, metadata: List[dict] = None):
        # L2 normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])
    
    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Dict]:
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx]
                })
        return results
    
    def save(self, path: str):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if hasattr(self.index, 'getDevice') else self.index, path)
        import pickle
        with open(path + ".meta", "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
```

### 8.2 ChromaDB

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Initialize with persistence
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function
emb_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["Document 1 content", "Document 2 content", "Document 3 content"],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"source": "report.pdf", "page": 1, "date": "2024-01"},
        {"source": "report.pdf", "page": 2, "date": "2024-01"},
        {"source": "blog.txt", "date": "2024-02"}
    ]
)

# Query
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=5,
    where={"source": "report.pdf"},  # Metadata filter
    include=["documents", "distances", "metadatas"]
)

# Update documents
collection.update(ids=["doc1"], documents=["Updated document 1 content"])

# Delete
collection.delete(ids=["doc3"])

print(f"Collection size: {collection.count()}")
```

### 8.3 Pinecone

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone(api_key="your-api-key")

# Create serverless index
pc.create_index(
    name="rag-index",
    dimension=1536,        # Match your embedding model's dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("rag-index")

# Upsert with batching
def upsert_documents(docs: List[str], embeddings: List[List[float]], metadata: List[dict]):
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        vectors = [
            {
                "id": f"doc-{j}",
                "values": embeddings[j],
                "metadata": {**metadata[j], "text": docs[j]}
            }
            for j in range(i, min(i+batch_size, len(docs)))
        ]
        index.upsert(vectors=vectors)

# Query with filters
results = index.query(
    vector=query_embedding,
    top_k=10,
    filter={"date": {"$gte": "2024-01-01"}, "source": {"$in": ["report", "blog"]}},
    include_metadata=True
)

for match in results.matches:
    print(f"Score: {match.score:.4f} | {match.metadata['text'][:100]}")
```

### 8.4 Weaviate

```python
import weaviate

# Initialize
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={"X-OpenAI-Api-Key": "your-key"}
)

# Define schema
class_obj = {
    "class": "Document",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small",
            "dimensions": 1536
        }
    },
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["text"]},
        {"name": "date", "dataType": ["date"]}
    ]
}
client.schema.create_class(class_obj)

# Add data
with client.batch as batch:
    batch.batch_size = 100
    for doc in documents:
        batch.add_data_object(
            {"content": doc["text"], "source": doc["source"]},
            "Document"
        )

# Hybrid search
result = (
    client.query
    .get("Document", ["content", "source"])
    .with_hybrid(query="machine learning basics", alpha=0.5)  # alpha: 0=sparse, 1=dense
    .with_additional(["score", "explainScore"])
    .with_limit(10)
    .do()
)
```

### 8.5 Vector DB Comparison

| Database | Best For | Hosting | Filter | Scale |
|----------|----------|---------|--------|-------|
| **FAISS** | Local, high-speed | Self | Limited | 10M+ (GPU) |
| **ChromaDB** | Local dev, prototyping | Self | Rich | 100K |
| **Pinecone** | Production, managed | Cloud | Rich | 100M+ |
| **Weaviate** | Hybrid search, GQL | Self/Cloud | GraphQL | 100M+ |
| **Qdrant** | Production, Rust | Self/Cloud | Rich | 100M+ |
| **Milvus** | Large-scale | Self/Cloud | Rich | 1B+ |

---

## 9. Chunking Strategies

### 9.1 Fixed-Size Chunking

Simple but ignores semantic boundaries:

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)

# Character-based (simple)
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Recursive (tries multiple separators: \n\n, \n, space, "")
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    is_separator_regex=False
)

# Token-based (respects model limits exactly)
token_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# Sentence-aware
sent_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=50,
    tokens_per_chunk=512
)
```

### 9.2 Semantic Chunking

Split at semantic boundaries using embedding similarity:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",    # Split where similarity drops below X-th percentile
    breakpoint_threshold_amount=95,            # 95th percentile of similarity drops
)

# Or gradient-based
semantic_splitter_grad = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="gradient",
    breakpoint_threshold_amount=95,
)

# Custom semantic chunker
class SemanticChunkerCustom:
    def __init__(self, embed_model, similarity_threshold: float = 0.7):
        self.embed_model = embed_model
        self.threshold = similarity_threshold
    
    def split_sentences(self, text: str) -> List[str]:
        import nltk
        return nltk.sent_tokenize(text)
    
    def chunk(self, text: str) -> List[str]:
        sentences = self.split_sentences(text)
        if len(sentences) <= 1:
            return sentences
        
        embeddings = self.embed_model.encode(sentences, normalize_embeddings=True)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            sim = (embeddings[i-1] @ embeddings[i]).item()
            
            if sim < self.threshold:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
```

### 9.3 Document-Aware Chunking

Respect document structure (headers, paragraphs, tables):

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader

# Markdown-aware splitting
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header1"),
        ("##", "Header2"),
        ("###", "Header3"),
    ]
)

markdown_text = """
# Introduction
This is the intro.

## Section 1
Content of section 1.

## Section 2
Content of section 2.
"""
chunks = md_splitter.split_text(markdown_text)
for chunk in chunks:
    print(chunk.metadata, ":", chunk.page_content[:50])

# PDF with layout awareness
def load_pdf_smart(pdf_path: str) -> List[Dict]:
    """Load PDF preserving structure."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    chunks = []
    for page in pages:
        # Each page is a chunk with metadata
        chunks.append({
            "text": page.page_content,
            "metadata": {
                "source": pdf_path,
                "page": page.metadata.get("page", 0),
            }
        })
    return chunks
```

### 9.4 Parent-Child Chunking (Small-to-Big)

Index small chunks for precise retrieval, return large parent chunks for context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

store = InMemoryStore()  # Parent document store
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Add documents
from langchain_community.document_loaders import TextLoader
loader = TextLoader("doc.txt")
retriever.add_documents(loader.load())

# Retrieve — returns parent chunks even though search was on children
results = retriever.invoke("query about topic")
```

### 9.5 Contextual Retrieval (Anthropic)

Prepend chunk-specific context generated by LLM before embedding:

```python
CONTEXTUALIZE_PROMPT = """Here is a document chunk:
<chunk>
{chunk}
</chunk>

This chunk is from a document about: {document_summary}

Please provide a brief (2-3 sentence) context for this chunk that explains:
1. What this chunk is about
2. How it relates to the broader document
3. Any important terms or concepts

Context:"""

def add_context_to_chunk(chunk: str, doc_summary: str, llm_client) -> str:
    """Prepend LLM-generated context to chunk before embedding."""
    context = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": CONTEXTUALIZE_PROMPT.format(
            chunk=chunk, document_summary=doc_summary
        )}],
        temperature=0.0,
        max_tokens=150
    ).choices[0].message.content
    
    return f"{context}\n\n{chunk}"
```

---

## 10. Reranking

### 10.1 Cross-Encoder Reranking

Bi-encoders (dense retrieval) encode query and document separately — fast but less accurate.
Cross-encoders process query+document together — slower but more accurate.

**Strategy:** Use bi-encoder for top-K recall (K=100), cross-encoder to rerank to top-N (N=5-10).

```python
from sentence_transformers import CrossEncoder

# Load cross-encoder
cross_encoder = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2',   # Lightweight
    # 'cross-encoder/ms-marco-electra-base',  # Better quality
)

def rerank(query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
    """Rerank documents using cross-encoder."""
    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    ranked = sorted(
        zip(documents, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [{"document": doc, "score": score} for doc, score in ranked[:top_k]]

# Example pipeline: BM25 (100 candidates) → Dense (100) → Hybrid RRF (50) → CrossEncoder (5)
def full_retrieval_pipeline(
    query: str,
    bm25_retriever: BM25Retriever,
    dense_index: FAISSIndex,
    cross_encoder: CrossEncoder,
    embed_model,
    n_bm25: int = 50,
    n_dense: int = 50,
    n_rerank: int = 5
) -> List[Dict]:
    
    # Stage 1: Candidate retrieval
    bm25_results = bm25_retriever.retrieve(query, top_k=n_bm25)
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    dense_results = dense_index.search(query_emb, top_k=n_dense)
    
    # Combine (deduplicate by document text)
    seen = set()
    all_candidates = []
    for r in bm25_results + dense_results:
        if r['document'] not in seen:
            seen.add(r['document'])
            all_candidates.append(r['document'])
    
    # Stage 2: Rerank
    reranked = rerank(query, all_candidates, top_k=n_rerank)
    return reranked
```

### 10.2 Cohere Rerank API

```python
import cohere

co = cohere.Client("your-cohere-api-key")

def cohere_rerank(query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
    results = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v3.0",
        return_documents=True
    )
    return [
        {
            "document": r.document.text,
            "score": r.relevance_score,
            "original_rank": r.index
        }
        for r in results.results
    ]
```

---

## 11. Advanced RAG Techniques

### 11.1 Query Rewriting

Reformulate ambiguous or poorly-worded queries:

```python
def rewrite_query(query: str, conversation_history: List[Dict] = None) -> List[str]:
    """Generate multiple query reformulations."""
    
    system = "You are a search query optimizer."
    
    prompt = f"""Generate 3 different search queries that would help find information to answer:
"{query}"

Consider: synonyms, related terms, more specific versions, different phrasings.
Return as JSON array of strings."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    import json
    try:
        result = json.loads(response.choices[0].message.content)
        queries = result.get("queries", [result.get("0"), result.get("1"), result.get("2")])
        return [query] + [q for q in queries if q]  # Original + rewrites
    except json.JSONDecodeError:
        return [query]
```

### 11.2 HyDE: Hypothetical Document Embeddings

Generate a hypothetical answer, embed it, use that embedding for retrieval:

```python
def hyde_retrieve(query: str, retriever, llm_client, top_k: int = 5) -> List[Dict]:
    """Hypothetical Document Embeddings (Gao et al., 2022)."""
    
    # Step 1: Generate hypothetical document
    hypothetical = llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"Write a detailed paragraph that would answer this question: {query}"
        }],
        temperature=0.7,
        max_tokens=300
    ).choices[0].message.content
    
    # Step 2: Embed hypothetical document
    hyp_embedding = embed_model.encode([hypothetical], normalize_embeddings=True)
    
    # Step 3: Retrieve using hypothetical embedding
    results = retriever.search(hyp_embedding, top_k=top_k)
    return results
```

### 11.3 Fusion Retrieval

Run multiple queries and fuse results:

```python
def fusion_retrieve(
    query: str, retriever, llm_client,
    n_queries: int = 4, top_k: int = 5, k_rrf: int = 60
) -> List[Dict]:
    """Multi-query fusion with RRF (Raudaschl, 2023)."""
    
    # Generate multiple queries
    all_queries = rewrite_query(query, n_queries=n_queries)
    
    # Retrieve for each query
    all_results: Dict[str, List[int]] = {}
    for q_idx, q in enumerate(all_queries):
        results = retriever.retrieve(q, top_k=top_k*2)
        for rank, r in enumerate(results):
            doc = r['document']
            if doc not in all_results:
                all_results[doc] = []
            all_results[doc].append(rank)
    
    # RRF scoring
    rrf_scores = {
        doc: sum(1 / (k_rrf + rank) for rank in ranks)
        for doc, ranks in all_results.items()
    }
    
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"document": doc, "rrf_score": score} for doc, score in sorted_docs[:top_k]]
```

### 11.4 Agentic RAG

RAG where the agent decides when and what to retrieve:

```python
class AgenticRAG:
    """RAG agent that decides retrieval strategy dynamically."""
    
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm = llm_client
        self.retrieved_docs = []
        self.reasoning_steps = []
    
    def should_retrieve(self, query: str, current_context: str) -> bool:
        """Decide if more retrieval is needed."""
        decision = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"""Given this question: "{query}"
And this current context: "{current_context[:500]}..."

Do you have enough information to answer confidently? 
Respond with just "yes" or "no"."""
            }],
            temperature=0.0,
            max_tokens=5
        ).choices[0].message.content.strip().lower()
        return "no" in decision
    
    def generate_subquery(self, original_query: str, missing_info: str) -> str:
        """Generate a targeted subquery for missing information."""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"To answer '{original_query}', I need: '{missing_info}'. Write a specific search query."
            }],
            temperature=0.0
        ).choices[0].message.content
        return response
    
    def run(self, query: str, max_iterations: int = 3) -> Dict:
        context = ""
        for iteration in range(max_iterations):
            # Decide: retrieve or answer?
            if iteration == 0 or self.should_retrieve(query, context):
                search_query = query if iteration == 0 else self.generate_subquery(query, f"iteration {iteration}")
                new_docs = self.retriever.retrieve(search_query, top_k=3)
                self.retrieved_docs.extend(new_docs)
                context = "\n".join([d['document'] for d in self.retrieved_docs])
                self.reasoning_steps.append(f"Retrieved {len(new_docs)} docs for: {search_query}")
            else:
                break
        
        # Final generation
        answer = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer based only on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        ).choices[0].message.content
        
        return {
            "answer": answer,
            "sources": self.retrieved_docs,
            "reasoning": self.reasoning_steps
        }
```

---

## 12. LLM Evaluation

### 12.1 Reference-Based Metrics

**BLEU** (Papineni et al., 2002) — measures n-gram precision:

\[
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
\]

**ROUGE** (Lin, 2004) — measures n-gram recall:

\[
\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{\text{gram}_n \in \text{ref}} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{\text{ref}} \sum_{\text{gram}_n \in \text{ref}} \text{Count}(\text{gram}_n)}
\]

**BERTScore** (Zhang et al., 2020) — uses BERT embeddings for semantic similarity:

\[
\text{BERTScore}_F = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

```python
import evaluate

# BLEU
bleu = evaluate.load("bleu")
result = bleu.compute(
    predictions=["The cat sat on the mat"],
    references=[["The cat is sitting on the mat"]]
)
print(f"BLEU: {result['bleu']:.4f}")

# ROUGE
rouge = evaluate.load("rouge")
result = rouge.compute(
    predictions=["The transformer model uses self-attention."],
    references=["Transformers utilize self-attention mechanisms."]
)
print(f"ROUGE-1: {result['rouge1']:.4f}")
print(f"ROUGE-L: {result['rougeL']:.4f}")

# BERTScore
bertscore = evaluate.load("bertscore")
result = bertscore.compute(
    predictions=["The cat sat on the mat"],
    references=["The cat is on the mat"],
    lang="en",
    model_type="distilbert-base-uncased"
)
print(f"BERTScore-F1: {result['f1'][0]:.4f}")
```

### 12.2 LLM-as-Judge

Use a strong LLM (GPT-4) to evaluate outputs:

```python
LLM_JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response.

Question: {question}
Context: {context}
Response: {response}

Evaluate on these criteria (1-5 scale):
1. **Faithfulness**: Is the response grounded in the context?
2. **Relevance**: Does it answer the question?
3. **Completeness**: Does it cover all important aspects?
4. **Clarity**: Is it clear and well-written?

Respond with JSON:
{{
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "overall": <1-5>,
  "reasoning": "<brief explanation>"
}}"""

def llm_judge(question: str, context: str, response: str, judge_model: str = "gpt-4o") -> Dict:
    """Evaluate RAG response using LLM as judge."""
    import json
    
    raw = client.chat.completions.create(
        model=judge_model,
        messages=[{
            "role": "user",
            "content": LLM_JUDGE_PROMPT.format(
                question=question, context=context, response=response
            )
        }],
        temperature=0.0,
        response_format={"type": "json_object"}
    ).choices[0].message.content
    
    return json.loads(raw)

# MT-Bench style pairwise evaluation
def pairwise_eval(question: str, response_a: str, response_b: str) -> str:
    """Compare two responses — returns 'A', 'B', or 'tie'."""
    
    prompt = f"""Compare these two AI responses to the question.
Question: {question}

Response A: {response_a}

Response B: {response_b}

Which is better? Consider: accuracy, clarity, helpfulness.
Respond with just: A, B, or tie"""
    
    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    ).choices[0].message.content.strip()
    
    return result
```

### 12.3 Key Benchmarks

| Benchmark | Task | Measures |
|-----------|------|---------|
| MMLU | Multi-choice QA (57 subjects) | World knowledge, reasoning |
| HumanEval | Python code generation | Code correctness |
| GSM8K | Math word problems | Arithmetic reasoning |
| MT-Bench | Multi-turn conversation | Instruction following |
| HELM | Comprehensive evaluation | 42+ scenarios |
| TruthfulQA | Truthfulness | Hallucination avoidance |
| ARC | Science questions | Reasoning |

---

## 13. Structured Output

### 13.1 JSON Mode

```python
import json
from pydantic import BaseModel, Field
from typing import List, Optional

# OpenAI JSON mode
def extract_entities(text: str) -> dict:
    """Extract named entities as structured JSON."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Extract named entities. Return JSON with keys: persons, organizations, locations, dates."
            },
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

# Structured outputs with Pydantic (OpenAI)
class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")
    features: List[str] = Field(description="Key product features")
    rating: Optional[float] = Field(default=None, description="Rating 1-5")

def extract_product(text: str) -> Product:
    """Extract product info with type safety."""
    from openai import OpenAI
    import openai
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Extract product information from the text."},
            {"role": "user", "content": text}
        ],
        response_format=Product,
    )
    return completion.choices[0].message.parsed
```

### 13.2 Function/Tool Calling

```python
import json

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the knowledge base for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters",
                        "properties": {
                            "date_from": {"type": "string"},
                            "category": {"type": "string"}
                        }
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

def tool_calling_loop(user_message: str) -> str:
    """Complete tool-calling loop with execution."""
    
    # Define tool implementations
    def search_database(query: str, filters: dict = None, top_k: int = 5) -> str:
        # In production, call actual retriever
        return f"Search results for '{query}': [result1, result2, result3]"
    
    def calculate(expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error: {e}"
    
    tool_map = {"search_database": search_database, "calculate": calculate}
    
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        
        if msg.tool_calls:
            messages.append(msg)
            
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                
                result = tool_map[fn_name](**fn_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            return msg.content
```

---

## 14. Full Production RAG Pipeline

### 14.1 End-to-End with LangChain

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# ---- Step 1: Load and split documents ----
loader = PyPDFDirectoryLoader("./documents/")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# ---- Step 2: Create vector store ----
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(
    search_type="mmr",   # Maximum Marginal Relevance — balances relevance + diversity
    search_kwargs={"k": 6, "fetch_k": 20}
)

# ---- Step 3: Define prompts ----
# Contextualize query given conversation history
contextualize_q_system = """Given a chat history and the latest user question,
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# QA prompt
qa_system = """You are an expert assistant. Use the following retrieved context to answer
the user's question. If you don't know the answer from the context, say so clearly.
Always cite the source documents.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ---- Step 4: Create chain ----
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ---- Step 5: Use with conversation history ----
chat_history = []

def chat(question: str) -> str:
    result = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result["answer"])
    ])
    
    return result["answer"]

print(chat("What is the main topic of the documents?"))
print(chat("Can you elaborate on that?"))  # Uses conversation history
```

### 14.2 LlamaIndex Advanced RAG

```python
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, ServiceContext,
    PromptTemplate, Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# Load documents
documents = SimpleDirectoryReader("./docs").load_data()

# Build index
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Advanced query engine with reranking
retriever = VectorIndexRetriever(index=index, similarity_top_k=20)
reranker = LLMRerank(choice_batch_size=5, top_n=5)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7),
        reranker
    ]
)

# Custom prompt
template = """Context information is below:
{context_str}

Given the context, answer the query. If you don't know, say so.
Query: {query_str}
Answer: """

query_engine.update_prompts({"response_synthesizer:text_qa_template": PromptTemplate(template)})

# Query
response = query_engine.query("What are the key findings?")
print(response)
print("\nSources:")
for node in response.source_nodes:
    print(f"  Score: {node.score:.4f} | {node.text[:100]}...")
```

---

## 15. Production Concerns

### 15.1 Caching

```python
import hashlib
import json
from functools import wraps
from typing import Callable
import redis

class RAGCache:
    """Cache for RAG pipeline (embeddings + LLM responses)."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = redis.from_url(redis_url, decode_responses=True)
    
    def _make_key(self, prefix: str, *args) -> str:
        content = json.dumps(args, sort_keys=True)
        return f"{prefix}:{hashlib.md5(content.encode()).hexdigest()}"
    
    def cache_embedding(self, text: str, embedding: List[float], ttl: int = 86400):
        key = self._make_key("emb", text)
        self.r.setex(key, ttl, json.dumps(embedding))
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        key = self._make_key("emb", text)
        result = self.r.get(key)
        return json.loads(result) if result else None
    
    def cache_response(self, query: str, response: str, ttl: int = 3600):
        key = self._make_key("resp", query)
        self.r.setex(key, ttl, response)
    
    def get_response(self, query: str) -> Optional[str]:
        key = self._make_key("resp", query)
        return self.r.get(key)
    
    def cached_embed(self, embed_fn: Callable):
        """Decorator to cache embedding calls."""
        @wraps(embed_fn)
        def wrapper(text: str) -> List[float]:
            cached = self.get_embedding(text)
            if cached:
                return cached
            result = embed_fn(text)
            self.cache_embedding(text, result)
            return result
        return wrapper


# Semantic cache: cache similar queries
class SemanticCache:
    """Cache that returns results for semantically similar queries."""
    
    def __init__(self, embed_model, similarity_threshold: float = 0.95):
        self.embed_model = embed_model
        self.threshold = similarity_threshold
        self.cache: List[Dict] = []
    
    def get(self, query: str) -> Optional[str]:
        if not self.cache:
            return None
        
        query_emb = self.embed_model.encode([query], normalize_embeddings=True)
        cache_embs = np.array([c['embedding'] for c in self.cache])
        
        scores = (query_emb @ cache_embs.T).flatten()
        best_idx = scores.argmax()
        
        if scores[best_idx] >= self.threshold:
            print(f"Cache hit! Similarity: {scores[best_idx]:.4f}")
            return self.cache[best_idx]['response']
        return None
    
    def set(self, query: str, response: str):
        embedding = self.embed_model.encode([query], normalize_embeddings=True)[0]
        self.cache.append({
            "query": query,
            "response": response,
            "embedding": embedding.tolist()
        })
```

### 15.2 Observability and Tracing

```python
import time
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class RAGTrace:
    """Trace for a single RAG request."""
    query: str
    timestamp: datetime = field(default_factory=datetime.now)
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    
    n_retrieved: int = 0
    n_reranked: int = 0
    top_score: float = 0.0
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    answer: str = ""
    retrieved_docs: List[Dict] = field(default_factory=list)
    error: Optional[str] = None

class ObservableRAG:
    """RAG with built-in observability."""
    
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.traces: List[RAGTrace] = []
    
    def query(self, question: str) -> Dict:
        trace = RAGTrace(query=question)
        total_start = time.time()
        
        try:
            # Retrieval
            t0 = time.time()
            retrieved = self.retriever.retrieve(question, top_k=20)
            trace.retrieval_ms = (time.time() - t0) * 1000
            trace.n_retrieved = len(retrieved)
            trace.top_score = retrieved[0]['score'] if retrieved else 0.0
            
            # Reranking
            t0 = time.time()
            reranked = self.reranker(question, [r['document'] for r in retrieved], top_k=5)
            trace.rerank_ms = (time.time() - t0) * 1000
            trace.n_reranked = len(reranked)
            trace.retrieved_docs = reranked
            
            # Generation
            t0 = time.time()
            context = "\n\n".join([r['document'] for r in reranked])
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Answer using the context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ]
            )
            trace.generation_ms = (time.time() - t0) * 1000
            trace.answer = response.choices[0].message.content
            trace.prompt_tokens = response.usage.prompt_tokens
            trace.completion_tokens = response.usage.completion_tokens
            
        except Exception as e:
            trace.error = str(e)
        
        trace.total_ms = (time.time() - total_start) * 1000
        self.traces.append(trace)
        
        return {"answer": trace.answer, "trace": trace}
    
    def get_metrics(self) -> Dict:
        successful = [t for t in self.traces if not t.error]
        return {
            "total_queries": len(self.traces),
            "success_rate": len(successful) / len(self.traces) if self.traces else 0,
            "avg_retrieval_ms": np.mean([t.retrieval_ms for t in successful]) if successful else 0,
            "avg_generation_ms": np.mean([t.generation_ms for t in successful]) if successful else 0,
            "avg_total_ms": np.mean([t.total_ms for t in successful]) if successful else 0,
            "avg_top_score": np.mean([t.top_score for t in successful]) if successful else 0,
            "avg_prompt_tokens": np.mean([t.prompt_tokens for t in successful]) if successful else 0,
        }

# LangSmith for LangChain tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "rag-production"
# All LangChain calls will now be automatically traced
```

### 15.3 Query Routing in Production

In production, route expensive pipelines (e.g., HyDE, multi-query) only when needed:

```python
def production_rag(query: str, llm_client) -> Dict:
    intent = route_query(query, llm_client)
    if intent.complexity == "simple" and intent.intent == "factual":
        # Fast path: single dense retrieval, no reranking
        return simple_rag(query, top_k=5)
    elif intent.intent == "analytical":
        # Full pipeline: hybrid + rerank + more context
        return advanced_rag(query, top_k=10, rerank=True)
    else:
        return standard_rag(query, top_k=5, rerank=True)
```

### 15.4 Latency Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedRAG:
    """Production-optimized RAG with async and parallelization."""
    
    def __init__(self, *args, **kwargs):
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def parallel_retrieve(self, queries: List[str]) -> List[List[Dict]]:
        """Retrieve for multiple queries in parallel."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, self.retriever.retrieve, q, 5)
            for q in queries
        ]
        return await asyncio.gather(*tasks)
    
    async def stream_response(self, question: str, context: str):
        """Stream LLM response for better perceived latency."""
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Answer concisely using the context."},
                    {"role": "user", "content": f"Context: {context}\n\nQ: {question}"}
                ],
                stream=True
            )
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield token  # Stream to client in real-time
        return full_response
```

---

## Common Pitfalls

### 1. Chunk Size Mismatch

**Problem:** Chunks too small lose context; too large dilute relevance and waste context window.

```python
# Rule of thumb: 256–512 tokens for semantic search, 512–1024 for factual QA
# Tune per domain: legal/doc → larger; code → smaller (function-level)
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
```

### 2. Embedding Model Mismatch

**Problem:** Indexing with model A and querying with model B yields poor similarity scores.

```python
# Always use the SAME model for index and query
embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # Same everywhere
```

### 3. No Reranking for Large top_k

**Problem:** Retrieving top_k=50 without reranking sends noisy context to the LLM.

```python
# Good: Retrieve many, rerank to few
candidates = retriever.retrieve(query, top_k=50)
reranked = cross_encoder_rerank(query, [c["document"] for c in candidates], top_k=5)
```

### 4. Hallucination from Missing Context

**Problem:** LLM answers when context doesn't contain the answer.

```python
# Mitigation: Strict system prompt
RAG_SYSTEM = """Answer ONLY using the provided context. If the answer is not in the context,
say "I don't have enough information" or "The context doesn't contain this." Never invent facts."""
```

### 5. Ignoring Query–Document Distribution Shift

**Problem:** Training docs are formal; user queries are colloquial. Embeddings may not align.

```python
# Use query rewriting or HyDE to bridge the gap
rewritten = rewrite_query(user_query)  # "what's the refund policy?" → "refund policy terms"
# Or HyDE: embed hypothetical answer instead of raw query
```

---

## Quick Reference: RAG Architecture Patterns

| Pattern | When to Use | Complexity | Quality |
|---------|-------------|------------|---------|
| **Naive RAG** | Prototyping | Low | Baseline |
| **Advanced RAG** | Production | Medium | Good |
| **Modular RAG** | Complex domains | High | Better |
| **Agentic RAG** | Multi-step reasoning | Very High | Best |
| **Self-RAG** | Reliability critical | High | Best + Verifiable |

## Key Resources

| Resource | Description |
|----------|-------------|
| [LangChain](https://python.langchain.com/) | RAG framework with integrations |
| [LlamaIndex](https://docs.llamaindex.ai/) | Data framework for LLM apps |
| [RAGAS](https://github.com/explodinggradients/ragas) | RAG evaluation framework |
| [Haystack](https://haystack.deepset.ai/) | End-to-end NLP framework |
| [DSPy](https://github.com/stanfordnlp/dspy) | Programmatic LLM optimization |
| ArXiv: 2312.10997 | Comprehensive RAG survey (Gao et al.) |
| ArXiv: 2005.11401 | Original RAG paper (Lewis et al.) |
| ArXiv: 2202.08904 | HyDE: Hypothetical Document Embeddings |
| ArXiv: 2401.09463 | RAPTOR: Recursive summarization for retrieval |
