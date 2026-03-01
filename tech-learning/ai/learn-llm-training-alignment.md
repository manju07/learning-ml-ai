# LLM Training & Alignment: Complete Guide

## Table of Contents
1. [Introduction to LLM Training](#introduction-to-llm-training)
2. [Pretraining](#pretraining)
3. [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
4. [Instruction Tuning](#instruction-tuning)
5. [Reinforcement Learning from Human Feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
6. [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
7. [Constitutional AI](#constitutional-ai)
8. [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
9. [Practical Examples](#practical-examples)
10. [Advanced Topics](#advanced-topics)
11. [Common Pitfalls and Troubleshooting](#common-pitfalls-and-troubleshooting)
12. [Production Considerations](#production-considerations)
13. [Best Practices](#best-practices)
14. [References](#references)

---

## Introduction to LLM Training

LLM training happens in stages:

```
Pretraining (next-token) → SFT (instruction following) → Alignment (RLHF/DPO)
     Unlabeled text              Labeled (input, output)      Preference data
```

### Stage Overview

| Stage | Data | Objective | Outcome |
|-------|------|-----------|---------|
| **Pretraining** | Web, books, code | Next-token prediction | Base LM |
| **SFT** | Instruction-response pairs | Cross-entropy | Follows instructions |
| **Alignment** | Preferences (better/worse) | Reward/ preference | Helpful, harmless |

---

## Pretraining

### Causal Language Modeling

Predict next token given previous:

L = -Σ log P(x_t | x_{<t})

### Data

- Web text (Common Crawl)
- Books, Wikipedia, code
- Quality filtering (dedup, toxicity, PII removal)

### Training Setup

```python
# Causal LM training (simplified)
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
# Labels = input_ids shifted by 1
# Loss only on non-padding tokens

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,  # Effective batch 128
    learning_rate=1e-4,
    warmup_steps=2000,
    max_steps=100000,
    fp16=True,
    dataloader_num_workers=4
)
```

### Scaling Laws

- Performance scales with: compute, data size, model size
- Chinchilla: optimal tokens ≈ 20× params
- Data quality matters as much as quantity

### Memory Optimization

- **Gradient checkpointing**: Trade compute for memory
- **Mixed precision (FP16/BF16)**
- **ZeRO optimizer** (DeepSpeed): Partition optimizer states

```python
# Gradient checkpointing
model.gradient_checkpointing_enable()
```

---

## Supervised Fine-Tuning (SFT)

SFT trains on (instruction, response) pairs to align the model with desired behavior.

### Data Format

```json
[
  {"instruction": "Explain quantum computing", "response": "Quantum computing uses..."},
  {"instruction": "Translate to French: Hello", "response": "Bonjour"}
]
```

### Chat Template

```python
# Common format for chat models
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]

# Convert to token sequence (model-specific)
# e.g., <|system|>...<|user|>...<|assistant|>...
```

### Loss: Only on Assistant Tokens

```python
def sft_loss(model, input_ids, labels, attention_mask):
    """
    Labels: -100 for system+user tokens, actual tokens for assistant
    """
    logits = model(input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss
```

### Example SFT with Hugging Face

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
    args=TrainingArguments(
        output_dir="./sft_output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=True
    )
)
trainer.train()
```

---

## Instruction Tuning

Same as SFT but explicitly with diverse instructions (Alpaca, ShareGPT, etc.).

### High-Quality Data

- **Diversity**: Many task types (QA, summarization, coding, etc.)
- **Quality**: Human-written or filtered
- **Format**: Consistent template

### Data Mixing

- Mix multiple datasets
- Upsample important domains
- Balance lengths

---

## Reinforcement Learning from Human Feedback (RLHF)

RLHF aligns the model with human preferences via a learned reward model and policy optimization.

### Three Phases

1. **SFT** (above): Baseline model
2. **Reward Model (RM)**: Train on preference data
3. **RL Phase**: Optimize policy to maximize reward

### Preference Data

```json
{"prompt": "...", "chosen": "Good response", "rejected": "Bad response"}
```

### Reward Model

Train to predict P(chosen > rejected | prompt):

```python
# RM: Take SFT model, add scalar head, or use log-prob difference
# Loss: -log σ(r_chosen - r_rejected)
# r_chosen = RM(prompt + chosen), r_rejected = RM(prompt + rejected)

def reward_model_loss(rm, prompt, chosen, rejected):
    r_chosen = rm(prompt + chosen)
    r_rejected = rm(prompt + rejected)
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

### RL Phase: PPO

- **Policy**: SFT model (or its copy)
- **Reward**: RM score - β * KL(π || π_ref)
- **Reference**: Frozen SFT to prevent drift
- **KL penalty**: Keeps policy close to reference

```python
# Pseudocode
for batch in rl_data:
    responses = policy.generate(batch.prompts)
    rewards = reward_model(batch.prompts, responses)
    kl_penalty = kl_divergence(policy, ref_policy, batch.prompts, responses)
    advantage = rewards - beta * kl_penalty
    ppo_loss = -advantage * log_prob
    ppo_loss.backward()
```

### TRL and trl Library

```python
from trl import PPOTrainer, PPOConfig

config = PPOConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    learning_rate=1.4e-5,
    batch_size=16,
    mini_batch_size=4
)
ppo_trainer = PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset)
ppo_trainer.train()
```

---

## Direct Preference Optimization (DPO)

**DPO** bypasses reward model and PPO; optimizes preference objective directly.

### Key Idea

Closed-form optimal policy for Bradley-Terry preference model:

L_DPO = -E[ log σ( β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))) ) ]

- y_w: chosen (winner)
- y_l: rejected (loser)
- β: temperature

### RLHF vs DPO: When to Use Which

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward model** | Separate RM trained on preferences | Implicit in loss; no RM |
| **Optimization** | RL (PPO); unstable at times | Supervised; single loss |
| **Complexity** | High (RM + PPO, many hyperparams) | Lower; similar to SFT |
| **Stability** | Sensitive to KL, advantage scaling | Generally more stable |
| **Use case** | Complex/composite rewards; fine-grained control | Preference data; simpler pipeline |

**Default choice**: Prefer DPO for most alignment tasks. Use RLHF when you need a standalone reward model (e.g., for online ranking or deployment-time scoring).

### Advantages over RLHF

- No reward model training
- No PPO (simpler, more stable)
- Single gradient step per batch

### DPO Implementation

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.1,
    loss_type="sigmoid"
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    **dpo_config
)
trainer.train()
```

### Preference Dataset Format

```python
# Dataset columns: prompt, chosen, rejected
# or: prompt_id, chosen_ids, rejected_ids (tokenized)
```

---

## Constitutional AI

**Constitutional AI** (Anthropic, 2022) reduces harm using a set of **principles** (a "constitution") instead of per-comparison human labels. The model critiques and revises its own outputs according to these principles, producing (prompt, revised_response) pairs for training.

### Two Stages

1. **RCIHF (RL from AI Feedback)**: Generate harmful prompts → model responds → AI critic rewrites using constitution → train on revised responses
2. **Constitutional preference**: Generate pairs; AI selects preferred response per principle → train preference model, then RL/DPO

### Process

1. **Red-teaming**: Generate harmful prompts
2. **Critique**: Model critiques response using constitution
3. **Revision**: Model revises to satisfy principles
4. **Preference**: (revised, original) as preference pair for RLHF/DPO

### Example Principles

- "Choose the response that is most helpful and harmless"
- "Choose the response that refuses to comply with harmful requests"
- "Choose the response that doesn't assume things about the user"

### Implementation Sketch

```python
def constitutional_revision(prompt, response, principles, llm):
    """
    Use an AI model to revise a response according to a randomly chosen principle.
    """
    principle = random.choice(principles)
    revision_prompt = f"""
    Principle: {principle}
    Original request: {prompt}
    Response: {response}

    Revise the response to better follow the principle. Output only the revised response.
    """
    revised = llm.generate(revision_prompt)
    return revised

# Build preference data: revised is preferred over original
def build_constitutional_preferences(prompts, base_model, principles, llm):
    pairs = []
    for prompt in prompts:
        original = base_model.generate(prompt)  # May be harmful
        revised = constitutional_revision(prompt, original, principles, llm)
        pairs.append({"prompt": prompt, "chosen": revised, "rejected": original})
    return pairs
```

---

## Parameter-Efficient Fine-Tuning

### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, config)
# Only LoRA params trained
```

### QLoRA (Quantized LoRA)

4-bit base model + LoRA adapters. Train 7B on single consumer GPU.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained("...", quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
```

### Other PEFT: Adapter, Prefix Tuning

- **Adapter**: Small bottleneck layers in transformer
- **Prefix tuning**: Learnable prefix tokens

---

## Practical Examples

### Example 1: Full DPO Pipeline

```python
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

dataset = load_dataset("argilla/ultrafeedback-binarized", split="train")

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=TrainingArguments(output_dir="./dpo_out", per_device_train_batch_size=2, num_train_epochs=1),
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,
)
trainer.train()
trainer.save_model("./dpo_model")
```

### Example 2: Instruction Tuning with LoRA

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    args=training_args
)
trainer.train()
```

### Example 3: Reward Model Training (with Comments)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """
    Reward model: base LM + scalar head. Outputs single value per (prompt, response).
    Use last-token hidden state to capture full-sequence representation.
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # Last non-padding token summarizes the full sequence
        hidden = outputs.last_hidden_state[:, -1, :]
        return self.head(hidden).squeeze(-1)

def reward_model_loss(rm, tokenizer, batch):
    """
    Bradley-Terry loss: chosen should have higher reward than rejected.
    batch: dict with 'prompt', 'chosen', 'rejected' or tokenized equivalents.
    """
    tok_chosen = tokenizer(
        batch["prompt"] + batch["chosen"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    tok_rejected = tokenizer(
        batch["prompt"] + batch["rejected"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    r_chosen = rm(input_ids=tok_chosen["input_ids"], attention_mask=tok_chosen["attention_mask"])
    r_rejected = rm(input_ids=tok_rejected["input_ids"], attention_mask=tok_rejected["attention_mask"])
    # -log sigmoid(r_w - r_l) = log(1 + exp(r_l - r_w))
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

---

## Advanced Topics

### Multimodal Alignment

- LLaVA, etc.: Align vision encoder with LLM via projection + SFT

### Safe RLHF

- Avoid reward hacking: Use KL penalty, diversify RMs
- Ensemble of RMs for robustness

### Online RLHF

- Collect preferences from deployed model
- Continuous improvement

### Scaling Laws for Alignment

- More preference data → better alignment
- Quality of preference data matters

---

## Common Pitfalls and Troubleshooting

### 1. Reward Hacking (RLHF)

**Symptom**: Reward model score rises but human ratings drop.

**Causes**: RM overfits to superficial cues (length, keywords); policy exploits RM.

**Solutions**: Diverse preference data; hold-out validation; stronger KL penalty; ensemble RMs.

### 2. Mode Collapse (DPO)

**Symptom**: Short, generic, or repetitive responses.

**Causes**: Beta too high; narrow preference data; no SFT diversity.

**Solutions**: Lower beta (e.g., 0.05); mix in SFT data; use IPO/KTO variants; check for duplicate preferences.

### 3. KL Explosion (RLHF)

**Symptom**: Policy diverges from reference; incoherent outputs.

**Causes**: KL penalty too low; learning rate too high; unstable advantages.

**Solutions**: Increase beta; reduce LR; normalize advantages; adaptive KL targets.

### 4. Preference Data Imbalance

**Symptom**: Model biased toward certain styles (formal, long, etc.).

**Solutions**: Balance pairs by length/topic; oversample rare categories; augment with synthetic preferences.

### 5. Reference Model Mismatch

**Symptom**: Unstable DPO loss or quality drop.

**Solutions**: Use same tokenizer and architecture; freeze reference; ensure reference is SFT checkpoint.

---

## Production Considerations

### Evaluation

- **Human eval**: Sample outputs; rate helpfulness, harmlessness, honesty
- **Benchmarks**: MT-Bench, TruthfulQA, BBQ, HHH alignment
- **Red-teaming**: Adversarial prompts; jailbreak tests

### Safety and Guardrails

- **Output filters**: Block PII, harmful content, jailbreak patterns
- **Input filters**: Reject clearly harmful prompts; rate-limit sensitive queries
- **Monitoring**: Track reward scores; detect distribution shift

### Versioning and Rollback

- Tag checkpoints with data version and config
- A/B test before full rollout
- Maintain rollback to previous aligned model

---

## Best Practices

1. **Start with SFT** on diverse instructions
2. **Use DPO** if RLHF is too complex
3. **LoRA/QLoRA** for limited GPU
4. **Validate** on held-out prompts and safety benchmarks
5. **Avoid overfitting** to preference data
6. **Monitor** for reward hacking and drift

---

## Summary

| Stage | Method | When |
|-------|--------|------|
| Base | Pretraining | Building from scratch |
| Capability | SFT | Instruction following |
| Alignment | RLHF | Human preference |
| Simpler alignment | DPO | Preference data, no RM |
| Efficient | LoRA, QLoRA | Limited compute |

**Libraries**: `trl`, `peft`, `bitsandbytes`, `transformers`

---

## References

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT) – Ouyang et al., 2022
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) – Rafailov et al., 2023
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) – Bai et al., 2022
- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) – Christiano et al., 2017
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) – Hugging Face
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) – Dettmers et al., 2023
