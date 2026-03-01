# AI for Code: Complete Guide

## Table of Contents
1. [Introduction to AI for Code](#introduction-to-ai-for-code)
2. [Code Generation](#code-generation)
3. [Code Understanding and Embeddings](#code-understanding-and-embeddings)
4. [AI Pair Programming](#ai-pair-programming)
5. [Code Completion (Copilot-Style)](#code-completion-copilot-style)
6. [Code Review and Refactoring](#code-review-and-refactoring)
7. [Documentation and Explanation](#documentation-and-explanation)
8. [SWE-Bench and Code Agents](#swe-bench-and-code-agents)
9. [Practical Examples](#practical-examples)
10. [Best Practices](#best-practices)

---

## Introduction to AI for Code

**AI for Code** applies LLMs and specialized models to software engineering: generation, completion, understanding, review, and autonomous coding agents.

### Key Applications

| Application | Description | Example |
|-------------|-------------|---------|
| **Code completion** | Suggest next tokens/lines | GitHub Copilot |
| **Code generation** | Generate from natural language | "Write a Python function to sort by key" |
| **Code understanding** | Summarize, find bugs | CodeBERT, StarCoder |
| **Code review** | Suggest improvements | PR review bots |
| **Documentation** | Generate docs, comments | Docstrings, README |
| **Code agents** | Autonomous task completion | Devin, SWE-bench |

### Code-Specific Models

- **Codex** (OpenAI): Code generation
- **StarCoder**, **Code Llama**: Open code LMs
- **CodeBERT**: Understanding (encoder)
- **InCode**, **DeepSeek-Coder**: Strong code LMs
- **Phi**: Small, efficient for on-device

---

## Code Generation

### Prompting for Code

```python
# Zero-shot
prompt = """
Write a Python function that takes a list of integers and returns the sum of squares.
Include type hints and docstring.
"""
code = llm.generate(prompt)

# Few-shot
prompt = """
Example 1:
Input: "function to reverse a string"
Output:
def reverse_string(s: str) -> str:
    return s[::-1]

Example 2:
Input: "function to check if palindrome"
Output:
def is_palindrome(s: str) -> bool:
    return s == s[::-1]

Input: "function to count vowels in string"
Output:
"""
code = llm.generate(prompt)
```

### Structured Output for Code

```python
# Request JSON with code + explanation
prompt = """
Generate Python code for: {task}
Return JSON: {"code": "...", "explanation": "..."}
"""
response = llm.generate(prompt)
result = json.loads(extract_json(response))
```

### Code Generation with Context

```python
# Include relevant files for context
def generate_with_context(task, file_contents):
    context = "\n".join([f"File: {path}\n{content}" for path, content in file_contents.items()])
    prompt = f"""
    Context (relevant files):
    {context}

    Task: {task}

    Generate the code. Reference existing patterns and imports.
    """
    return llm.generate(prompt)
```

### Testing Generated Code

```python
def generate_and_test(task):
    code = llm.generate(f"Write Python code: {task}")
    # Extract code block
    code_block = extract_code_block(code)
    # Run in sandbox
    try:
        exec(code_block)
        return {"code": code_block, "status": "ok"}
    except Exception as e:
        # Self-repair: send error back to LLM
        fixed = llm.generate(f"Original: {code_block}\nError: {e}\nFix the code.")
        return {"code": extract_code_block(fixed), "status": "repaired"}
```

---

## Code Understanding and Embeddings

### Code Embeddings: Concepts

**Code embeddings** map code (or code + natural language) to dense vectors so similar semantics cluster together. Unlike natural language, code has rigid structure: syntax, control flow, and data flow matter. Naive text embeddings often fail on code because:

| Challenge | Why Text Embeddings Fail | Code-Specific Approach |
|-----------|--------------------------|-------------------------|
| **Variable renaming** | `x` vs `count` look different but may be equivalent | Structural similarity; variable normalization |
| **Syntax variations** | `f()` vs `f ()` | AST-based or syntax-aware tokenization |
| **Cross-language** | Python vs JS implementations of same algo | Multilingual code models (CodeBERT, UniXcoder) |
| **Comment vs code** | "increment counter" vs `x += 1` | Bimodal (code+NL) pre-training |

**Embedding strategies**:
1. **Raw code** — tokenize as text; works for similar surface forms
2. **AST-aware** — embed AST nodes or paths; captures structure
3. **Bimodal** — code + natural language in same space for search ("find auth logic" → code)

### Code Embedding Models

| Model | Type | Use Case |
|-------|------|----------|
| **CodeBERT** | Encoder (RoBERTa) | Code search, code-to-NL, classification |
| **UniXcoder** | Encoder + decoder | Cross-lingual, code completion |
| **GraphCodeBERT** | Data-flow aware | Semantic clone detection, better structure |
| **StarCoder/Code Llama** | Decoder | Use last hidden state for embedding |
| **Salesforce/codebert** | MLM on code | General code understanding |

```python
from sentence_transformers import SentenceTransformer

# Code-specific: Microsoft/codebert-base, microsoft/unixcoder-base
model = SentenceTransformer("microsoft/codebert-base")
code_embeddings = model.encode([
    "def add(a, b): return a + b",
    "def subtract(a, b): return a - b",
])
# Similarity between code snippets
similarity = cosine_similarity([code_embeddings[0]], code_embeddings)[0]
```

### AST Parsing for Code Understanding

**Abstract Syntax Trees (ASTs)** represent code structure; parsing removes superficial formatting and exposes logic. Use ASTs for:

- **Semantic chunking**: Split by function/class nodes instead of raw lines
- **Structural search**: Find "all `if` blocks that call `execute`"
- **Clone detection**: Compare AST subtrees for similar logic
- **Refactoring**: Identify renameable symbols, dead code

```python
import ast

def extract_functions(code: str) -> list[dict]:
    """Parse Python AST and extract function signatures and bodies"""
    tree = ast.parse(code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Get source span for body
            body_text = ast.get_source_segment(code, node) or ""
            functions.append({
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "body_snippet": body_text[:200]  # First 200 chars
            })
    return functions

# AST path for similarity: root→node1→...→leaf (used in some code similarity works)
def get_ast_paths(node, path=""):
    """Get paths from root to leaves (simplified)"""
    if isinstance(node, ast.AST):
        name = type(node).__name__
        children = list(ast.iter_child_nodes(node))
        if not children:
            yield f"{path}/{name}"  # Leaf node
        for child in children:
            yield from get_ast_paths(child, f"{path}/{name}")
```

**Tree-sitter** (multi-language): Fast, incremental, error-tolerant parsing. Use for non-Python languages.

```python
from tree_sitter import Parser, Language
# Build grammar for Python, Java, etc.
parser = Parser(Language.build_library("build/my-languages.so", ["tree-sitter-python"]))
tree = parser.parse(b"def foo(): pass")
# Traverse: tree.root_node.children
```

### Code Search

```python
# Index codebase
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="microsoft/codebert-base")
# Chunk by function/class
chunks = chunk_codebase(repo_path)
vectorstore = Chroma.from_texts(chunks, embeddings)
# Query
results = vectorstore.similarity_search("where is authentication handled?", k=5)
```

### Code Classification

```python
# Classify: bug/feature/refactor, language, intent
def classify_commit_message(msg):
    prompt = f"Classify commit message: {msg}\nCategories: bugfix, feature, refactor, docs, chore"
    return llm.generate(prompt)
```

---

## AI Pair Programming

### Cursor, Copilot Patterns

- **Inline completion**: Suggest as you type
- **Chat**: Ask about code, request edits
- **Edit in place**: Select code, describe change, apply

### Chat-Based Editing

```python
# User: "Add error handling to this function"
# System: 1. Get current code 2. Generate modified version 3. Apply diff
def apply_edit(original_code, user_instruction):
    prompt = f"""
    Original code:
    ```python
    {original_code}
    ```

    Instruction: {user_instruction}

    Return only the modified code, no explanation.
    """
    modified = llm.generate(prompt)
    return extract_code_block(modified)
```

### Context Window for Pair Programming

Include in prompt:
- Current file
- Related files (imports, same module)
- Recent edits
- Open files
- Error messages

```python
def build_context(current_file, imports, errors):
    context = f"Current file:\n{current_file}\n\n"
    if imports:
        context += "Imported from:\n" + "\n".join(imports) + "\n\n"
    if errors:
        context += f"Errors:\n{errors}\n\n"
    return context
```

---

## Code Completion (Copilot-Style)

### Fill-in-the-Middle (FIM)

Code LMs trained with **FIM** format: `< prefix > < suffix >` to predict `< middle >`.

```python
# Prompt format for FIM
# <fim_prefix>def factorial(n):<fim_suffix>    return result
# Model predicts: <fim_middle> result = 1\n    for i in range(1, n+1):\n        result *= i\n

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

# Use FIM tokens
prefix = "def factorial(n):"
suffix = "    return result"
input_ids = tokenizer.encode(f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50)
middle = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
```

### Line-Level Completion

```python
# Simpler: next-line prediction
# Input: prior lines
# Output: next line(s)
def complete_line(code_so_far):
    prompt = f"Complete the next line:\n```\n{code_so_far}\n"
    return llm.generate(prompt, max_tokens=50, stop=["\n\n", "```"])
```

---

## Code Review and Refactoring

### Automated Code Review

```python
def review_code(code, context=""):
    prompt = f"""
    Review this code for:
    1. Bugs and edge cases
    2. Security issues
    3. Performance
    4. Style and best practices
    5. Suggestions for improvement

    Code:
    ```python
    {code}
    ```

    Context: {context}

    Provide concise review with specific suggestions.
    """
    return llm.generate(prompt)
```

### Refactoring Suggestions

```python
def suggest_refactor(code):
    prompt = f"""
    Suggest refactoring for:
    ```python
    {code}
    ```

    Focus on: readability, DRY, performance, type hints.
    Return: 1) refactored code 2) brief explanation of changes.
    """
    return llm.generate(prompt)
```

### Security Scanning

```python
def security_scan(code):
    prompt = f"""
    Scan for security vulnerabilities:
    ```python
    {code}
    ```

    Check: SQL injection, XSS, hardcoded secrets, unsafe deserialization.
    List any findings with severity and fix suggestion.
    """
    return llm.generate(prompt)
```

---

## Documentation and Explanation

### Docstring Generation

```python
def generate_docstring(code):
    prompt = f"""
    Generate a Google-style docstring for:
    ```python
    {code}
    ```

    Include: summary, args, returns, raises, example if relevant.
    """
    return llm.generate(prompt)
```

### Code Explanation

```python
def explain_code(code, audience="developer"):
    prompt = f"""
    Explain this code for a {audience}.
    Cover: what it does, how it works, key algorithms.

    Code:
    ```python
    {code}
    ```
    """
    return llm.generate(prompt)
```

### README Generation

```python
def generate_readme(repo_structure, main_files):
    prompt = f"""
    Generate README.md for a project with structure:
    {repo_structure}

    Key files:
    {main_files}

    Include: description, setup, usage, examples.
    """
    return llm.generate(prompt)
```

---

## SWE-Bench and Code Agents

### SWE-Bench

Benchmark: Solve real GitHub issues (install deps, understand codebase, implement fix).

### Code Agent Architecture

```python
class CodeAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_command": self.run_command,
            "search_codebase": self.search_codebase,
        }

    def solve_issue(self, issue_description, repo_path):
        plan = self.plan(issue_description)
        for step in plan:
            result = self.execute_step(step)
            if not result.success:
                plan = self.replan(plan, step, result)
        return self.verify_fix()

    def plan(self, issue):
        prompt = f"""
        GitHub issue: {issue}

        Steps to fix (use tools: read_file, write_file, run_command, search_codebase):
        1. ...
        2. ...
        """
        return parse_plan(self.llm.generate(prompt))
```

### Tool Use for Code Agents

```python
tools = [
    Tool(name="read_file", func=read_file, description="Read file contents"),
    Tool(name="search", func=codebase_search, description="Semantic search in codebase"),
    Tool(name="run_tests", func=run_tests, description="Run test suite"),
    Tool(name="apply_patch", func=apply_patch, description="Apply diff patch"),
]
agent = create_react_agent(llm, tools, prompt)
```

### Self-Correction Loop

```python
def fix_with_feedback(agent, issue, max_iterations=5):
    for i in range(max_iterations):
        patch = agent.propose_fix(issue)
        result = run_tests()
        if result.passed:
            return patch
        agent.add_feedback(f"Tests failed: {result.output}")
    return None
```

---

## Practical Examples

### Example 1: Generate and Run

```python
def code_from_natural_language(task):
    code = llm.generate(f"Python: {task}")
    code = extract_code_block(code)
    # Sandboxed execution
    return execute_safely(code)
```

### Example 2: Codebase Q&A

```python
index = create_code_index("./src")
def ask(codebase_question):
    docs = index.similarity_search(codebase_question, k=5)
    context = "\n".join([d.page_content for d in docs])
    return llm.generate(f"Context:\n{context}\n\nQ: {codebase_question}\nA:")
```

### Example 3: Migrate Code (Python 2 → 3)

```python
def migrate_python3(code):
    prompt = f"""
    Migrate to Python 3:
    ```python
    {code}
    ```

    Handle: print, range, dict methods, encoding, etc.
    Return migrated code only.
    """
    return extract_code_block(llm.generate(prompt))
```

---

## Best Practices

1. **Sandbox** execution of generated code
2. **Validate** outputs (syntax, type hints)
3. **Iterate** with error feedback (self-repair)
4. **Context**: Include relevant files for large codebases
5. **Test** generated code automatically
6. **Security**: Sanitize inputs, avoid code that accesses external resources
7. **AST chunking**: For code search, chunk by function/class boundaries, not fixed tokens
8. **Embedding model**: Use code-specific models (CodeBERT, GraphCodeBERT) for semantic code tasks

---

## Pitfalls and Gotchas

| Pitfall | Cause | Mitigation |
|---------|-------|------------|
| **Hallucinated imports** | LLM invents non-existent packages | Validate imports; provide allowed list; use AST to check |
| **Off-by-one in loops** | Common failure mode for code gen | Add boundary tests; self-repair with error feedback |
| **Context overflow** | Large codebase, small context | Chunk by module; use semantic search to retrieve relevant files |
| **Wrong language** | Prompt doesn't specify; model defaults | Explicitly state language and runtime in system prompt |
| **AST parse errors** | Malformed or partial code | Use tree-sitter (error-tolerant); fallback to line-based chunking |
| **Embedding code as prose** | Generic text embedding for code | Use CodeBERT/GraphCodeBERT; avoid `sentence-transformers` default |
| **Over-trusting completion** | FIM can suggest plausible but wrong code | Always run tests; use stricter stop tokens |

---

## References

- **CodeBERT**: Feng et al. (2020) — *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*
- **GraphCodeBERT**: Guo et al. (2021) — *GraphCodeBERT: Pre-training Code Representations with Data Flow*
- **StarCoder**: Li et al. (2023) — *StarCoder: May the source be with you!*
- **Tree-sitter**: https://tree-sitter.github.io/
- **SWE-bench**: Jimenez et al. (2024) — *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- **InCoder**: Fried et al. (2023) — *InCoder: A Generative Model for Code Infilling and Synthesis*

---

## Summary

| Task | Approach | Model |
|------|----------|-------|
| Completion | FIM, next-token | StarCoder, Codex |
| Generation | Few-shot, chain-of-thought | GPT-4, Claude, Code Llama |
| Understanding | Embeddings | CodeBERT, UnixCoder |
| Review | Prompting | General LLM |
| Agents | ReAct + tools | GPT-4, Claude |
| SWE-bench | Multi-step agent | Frontier models |

**Libraries**: `transformers`, `langchain`, `tree-sitter`, `aider`, `cursor` (IDE)
