# Multimodal AI: Comprehensive Guide

## Table of Contents
1. [Introduction to Multimodal AI](#introduction-to-multimodal-ai)
2. [Vision Encoders](#vision-encoders)
3. [Language-Vision Alignment: Contrastive Learning](#language-vision-alignment-contrastive-learning)
4. [VLM Architecture Patterns: Fusion Strategies](#vlm-architecture-patterns-fusion-strategies)
5. [Foundational VLMs: BLIP-2, Flamingo](#foundational-vlms-blip-2-flamingo)
6. [Open-Source VLMs: LLaVA Family](#open-source-vlms-llava-family)
7. [Frontier VLMs: GPT-4V, Gemini, Claude Vision](#frontier-vlms-gpt-4v-gemini-claude-vision)
8. [More VLMs: CogVLM, InternVL, Qwen-VL, Phi-3-Vision](#more-vlms-cogvlm-internvl-qwen-vl-phi-3-vision)
9. [Image-Text Tasks: VQA, Captioning, OCR, Document AI](#image-text-tasks-vqa-captioning-ocr-document-ai)
10. [Video Understanding](#video-understanding)
11. [Audio-Language Models](#audio-language-models)
12. [Multimodal RAG](#multimodal-rag)
13. [Grounding and Detection](#grounding-and-detection)
14. [3D Understanding](#3d-understanding)
15. [Multimodal Agents](#multimodal-agents)
16. [Evaluation](#evaluation)
17. [Practical Code Examples](#practical-code-examples)
18. [Best Practices](#best-practices)

---

## Introduction to Multimodal AI

**Multimodal AI** processes and reasons across multiple data modalities — text, images, audio, video, and 3D — within a unified framework. The human world is inherently multimodal: we see, hear, read, and touch simultaneously. Building AI that can integrate these modalities enables richer understanding and interaction.

### Why Multimodal Matters

| Single-Modal Limitation | Multimodal Solution |
|------------------------|---------------------|
| Image classifier: outputs class only | VLM: describes image in natural language, answers questions |
| Text LLM: cannot "see" diagrams | GPT-4V / Gemini: understands charts, code screenshots |
| Audio ASR: text only | AudioPaLM: understands spoken questions with visual context |
| Single-modal retrieval | Multimodal RAG: retrieve images, text, tables together |

### Modality Spectrum

```
Modalities in AI:
  Text    ──────── Language models (GPT, LLaMA, Claude)
  Image   ──────── ViT, ResNet, SAM, DINO
  Audio   ──────── Whisper, wav2vec2, HuBERT
  Video   ──────── Video-LLaVA, VideoChat, TimeSformer
  3D      ──────── Point-E, Shap-E, 3D-LLM
  
Cross-modal:
  Text + Image  → CLIP, BLIP-2, LLaVA, GPT-4V, Gemini
  Text + Audio  → Whisper, AudioPaLM, SALMONN
  Text + Video  → Video-LLaVA, VideoChat
  All modalities → Gemini Ultra, GPT-4o
```

### Key Applications

- **Visual Question Answering (VQA)**: "What color is the car in this image?"
- **Image Captioning**: Generate natural language descriptions of images
- **Document Understanding**: Parse invoices, forms, scientific papers with OCR + LLM
- **Zero-Shot Classification**: Classify images into arbitrary categories using text descriptions
- **Image-Text Retrieval**: Semantic image search with natural language queries
- **Text-to-Image**: DALL-E 3, Stable Diffusion, Midjourney (see generative AI guide)
- **Video Understanding**: Temporal reasoning, action recognition, video QA
- **Multimodal Agents**: Robots, autonomous agents that perceive and act in visual environments

---

## Vision Encoders

The vision encoder transforms raw pixels into a structured representation (embedding) that downstream models can reason over.

### Vision Transformer (ViT)

ViT divides an image into fixed-size patches and treats them as tokens — analogous to words in a sentence.

```
Input Image: 224 × 224
Patch size: 16 × 16
Number of patches: (224/16)² = 196 patches
Each patch → flatten + linear projection → 768-dim token

[CLS] [patch_1] [patch_2] ... [patch_196] → Transformer → embeddings
```

```python
import torch
from transformers import ViTModel, ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

from PIL import Image
image = Image.open("dog.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# CLS token: global image representation [1, 768]
cls_embedding = outputs.last_hidden_state[:, 0, :]
# Patch tokens: local patch representations [1, 196, 768]
patch_embeddings = outputs.last_hidden_state[:, 1:, :]

print(f"CLS shape: {cls_embedding.shape}")
print(f"Patch shape: {patch_embeddings.shape}")
```

### CLIP Vision Encoder

CLIP (Contrastive Language-Image Pre-training) trains a vision encoder jointly with a text encoder using contrastive learning on 400M image-text pairs from the internet.

```python
from transformers import CLIPVisionModel, CLIPImageProcessor

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

inputs = processor(images=image, return_tensors="pt")
outputs = vision_model(**inputs)

# Pooled output: [1, 1024] — the global image embedding
image_embedding = outputs.pooler_output
# All hidden states: [1, 257, 1024] (including CLS token)
all_tokens = outputs.last_hidden_state
```

**CLIP variants:**
- `clip-vit-base-patch32`: Faster, smaller (512-dim)
- `clip-vit-large-patch14`: Higher quality (1024-dim)
- `clip-vit-large-patch14-336`: Higher resolution (336×336)

### EVA-CLIP

EVA-CLIP is a scaled CLIP from BAAI trained with improved stability and higher resolution. Used in BLIP-2 and InternVL.

```python
# EVA-CLIP available via timm
import timm

model = timm.create_model("eva_clip_g_14_plus.mim_in22k_ft_in1k", pretrained=True)
model.eval()

# Transforms
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config)
input_tensor = transforms(image).unsqueeze(0)

with torch.no_grad():
    features = model.forward_features(input_tensor)  # [1, 257, 1408]
```

### SigLIP

SigLIP (Sigmoid Loss for Language Image Pre-training, Google 2023) replaces CLIP's softmax contrastive loss with sigmoid loss, enabling better scaling and independent positive/negative treatment.

```
CLIP:   softmax over batch → requires large batches
SigLIP: sigmoid per pair → scales to any batch size

SigLIP loss per (image_i, text_j):
  label = +1 if i==j else -1
  loss = log(1 + exp(-label * similarity))
```

```python
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224")

inputs = processor(text=["a photo of a dog", "a photo of a cat"],
                   images=[image1, image2], return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)  # Not softmax! Binary probabilities
```

### DINO and DINOv2

DINO (Self-Distillation with No Labels) learns rich visual features through self-supervised learning. DINOv2 scales this significantly and produces features excellent for dense tasks (depth, segmentation).

```python
import torch
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# DINOv2 patch features: excellent for segmentation, depth, matching
patch_features = outputs.last_hidden_state[:, 1:, :]  # [1, 256, 768]
cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 768]
```

**Key insight:** DINOv2 features show semantic grouping — patches of "dog" cluster together even without explicit training labels.

### SAM (Segment Anything Model) Image Encoder

SAM's image encoder is a ViT-H that produces dense feature maps for the segmentation decoder. It creates rich, spatially-aware embeddings.

```python
from segment_anything import SamPredictor, sam_model_registry

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

predictor.set_image(image_array)  # Runs image encoder once

# Now query with points, boxes, or masks
masks, scores, logits = predictor.predict(
    point_coords=[[500, 375]],    # [x, y]
    point_labels=[1],             # 1=foreground, 0=background
    multimask_output=True
)
# masks: [3, H, W] — 3 candidate masks, choose highest score
best_mask = masks[scores.argmax()]
```

---

## Language-Vision Alignment: Contrastive Learning

### The Core Idea

Contrastive learning aligns representations so that semantically matching pairs (image of a dog, text "a dog") are close in embedding space, while mismatched pairs are far apart.

### InfoNCE / NT-Xent Loss (CLIP's Loss)

```
For a batch of N (image, text) pairs:
  S[i,j] = (image_i · text_j) / τ     (cosine similarity, scaled by temperature τ)
  
  Loss = -1/N * Σ_i [ log( exp(S[i,i]) / Σ_j exp(S[i,j]) ) ]
       + -1/N * Σ_j [ log( exp(S[j,j]) / Σ_i exp(S[i,j]) ) ]
  
  Diagonal = positive pairs; off-diagonal = negatives
```

```python
import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    """
    InfoNCE contrastive loss (CLIP-style).
    image_features: [B, D], text_features: [B, D]
    """
    # Normalize to unit sphere
    image_features = F.normalize(image_features, dim=-1)
    text_features  = F.normalize(text_features,  dim=-1)
    
    # Scaled similarity matrix [B, B]
    logits = (image_features @ text_features.T) / temperature
    
    # Ground truth: diagonal pairs match
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Symmetric cross-entropy
    loss_i2t = F.cross_entropy(logits,   labels)   # image→text
    loss_t2i = F.cross_entropy(logits.T, labels)   # text→image
    
    return (loss_i2t + loss_t2i) / 2

# Example
B, D = 64, 512
image_emb = torch.randn(B, D)
text_emb  = torch.randn(B, D)
loss = clip_loss(image_emb, text_emb)
print(f"CLIP loss: {loss.item():.4f}")
```

### Scaling Contrastive Learning

CLIP's success comes from scale:
- 400M noisy image-text pairs (scraped from web)
- Large batch size (32,768 for original CLIP)
- Temperature learned as a parameter (`log_scale = nn.Parameter(torch.ones([]) * log(1/0.07))`)

```python
class CLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder   = text_encoder
        self.image_proj     = nn.Linear(vision_encoder.output_dim, embed_dim)
        self.text_proj      = nn.Linear(text_encoder.output_dim,   embed_dim)
        # Learnable temperature
        self.logit_scale    = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
    
    def encode_image(self, images):
        return F.normalize(self.image_proj(self.vision_encoder(images)), dim=-1)
    
    def encode_text(self, tokens):
        return F.normalize(self.text_proj(self.text_encoder(tokens)), dim=-1)
    
    def forward(self, images, tokens):
        image_feat = self.encode_image(images)
        text_feat  = self.encode_text(tokens)
        
        scale  = self.logit_scale.exp()  # learned temperature inverse
        logits = scale * image_feat @ text_feat.T
        labels = torch.arange(len(images), device=images.device)
        
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, logits
```

### Captioning-Based Alignment (BLIP)

Instead of contrastive loss alone, BLIP adds a captioning loss: the model must generate the paired text from the image.

```
BLIP training objectives:
  1. Image-text contrastive (ITC): align embeddings
  2. Image-text matching (ITM): binary: does this image match this text?
  3. Image-conditioned language modeling (LM): generate caption from image
```

---

## Vision-Language Fusion Strategies

How modalities are combined critically affects model capability, compute cost, and training stability. Below are the main fusion patterns used in VLMs.

### Early Fusion

Combine modalities at the input level before the main network processes them.

```
[Image Patches + Text Tokens] → Unified Transformer → Output

Pros: Deep cross-modal interaction
Cons: Expensive — processes everything together; requires retraining if one modality changes
Example: Perceiver IO, Flamingo (sort-of), Unified-IO
```

### Late Fusion

Process each modality independently, combine at decision/prediction time.

```
Image → Image Encoder → Image Embedding ─┐
                                          ├→ Combiner → Output
Text  → Text  Encoder → Text  Embedding ─┘

Pros: Modular, reuse pretrained encoders
Cons: Less deep interaction
Example: CLIP retrieval, dual-encoder models
```

### Cross-Attention Fusion (Dominant in VLMs)

The dominant pattern: use cross-attention so the language model’s tokens attend to visual tokens. Text tokens act as *queries* (Q); visual tokens provide *keys* (K) and *values* (V). Each text position can aggregate relevant visual information.

```
Image → Visual Encoder → Visual Tokens (K, V)
Text  → Language Model  → Text Tokens  (Q)

At each transformer layer: Text_Q attends to Visual_KV

This is how: Flamingo, LLaVA, BLIP-2 (Q-Former), InstructBLIP work
```

```python
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """Allow text tokens to attend to visual tokens."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q     = nn.LayerNorm(dim)
        self.norm_kv    = nn.LayerNorm(dim)
        self.ff         = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.norm_ff    = nn.LayerNorm(dim)
    
    def forward(self, text_tokens, visual_tokens):
        # Cross-attention: text queries attend to visual keys/values
        q  = self.norm_q(text_tokens)
        kv = self.norm_kv(visual_tokens)
        attended, _ = self.cross_attn(q, kv, kv)
        text_tokens = text_tokens + attended
        text_tokens = text_tokens + self.ff(self.norm_ff(text_tokens))
        return text_tokens
```

**Why cross-attention dominates**: It allows flexible, content-dependent grounding—the model can focus on different image regions for different words—without the cost of early fusion over the full sequence.

### MLP Projector (LLaVA style)

The simplest approach: project visual features into the LLM's token embedding space.

```python
class MLPProjector(nn.Module):
    """LLaVA-1.5 uses a 2-layer MLP to bridge visual encoder → LLM."""
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, visual_features):
        # visual_features: [B, num_patches, vision_dim]
        return self.proj(visual_features)  # [B, num_patches, llm_dim]
```

---

## Foundational VLMs: BLIP-2, Flamingo

### Flamingo (DeepMind, 2022)

Flamingo inserts learnable cross-attention layers (called **Gated xattn-dense**) into a frozen LLM, with perceiver-resampled visual tokens. This enables few-shot learning with interleaved images and text.

```
Flamingo architecture:
  Frozen Vision Encoder (NFNet) → Perceiver Resampler → 64 visual tokens
  Frozen LLM (Chinchilla)
  + Learnable gated cross-attention layers interleaved with LLM layers

Few-shot in-context learning:
  [Image1] [Caption1] [Image2] [Caption2] [Image3] → [Generate Caption3]
```

```python
from transformers import IdeficsForVisionText2Text, AutoProcessor

# Open-source reproduction: IDEFICS (HuggingFace)
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")
model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")

from PIL import Image
import requests

prompts = [
    [
        "https://example.com/dog.jpg",
        "Question: What animal is this? Answer: dog\n",
        "https://example.com/cat.jpg",
        "Question: What animal is this? Answer:",
    ]
]

inputs = processor(prompts, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(processor.batch_decode(outputs, skip_special_tokens=True))
```

### BLIP-2 (Salesforce, 2023)

BLIP-2 introduces the **Q-Former** (Querying Transformer): a lightweight bridge between a frozen vision encoder and a frozen LLM.

```
BLIP-2 Architecture:

  Frozen Image Encoder (EVA-CLIP-g 1.3B)
         ↓
  Q-Former (32 learnable query tokens attend to image)
    - Stage 1: Train Q-Former with ITC + ITM + LM against image encoder
    - Stage 2: Connect Q-Former output to frozen LLM (FlanT5 or OPT)
         ↓
  Frozen LLM (FlanT5-XXL 11B or OPT-6.7B)

Only Q-Former is trainable! Very parameter-efficient.
```

```python
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torch
from PIL import Image

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

image = Image.open("image.jpg")

# Image captioning (no text prompt)
inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=30)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"Caption: {caption}")

# Visual question answering
question = "How many people are in this image?"
inputs = processor(image, text=question, return_tensors="pt").to("cuda", torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=30)
answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"Answer: {answer}")
```

### InstructBLIP (Salesforce, 2023)

InstructBLIP fine-tunes BLIP-2's Q-Former on 26 vision-language datasets with instruction tuning, enabling it to follow complex instructions.

```python
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

image = Image.open("chart.png")
prompt = "Analyze this chart and describe the main trend."

inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    do_sample=False,
    num_beams=5,
    max_length=256,
    repetition_penalty=1.5,
    length_penalty=1.0,
)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(response)
```

---

## Open-Source VLMs: LLaVA Family

### LLaVA (Visual Instruction Tuning, 2023)

LLaVA connects CLIP-ViT-L with Vicuna/LLaMA via a single linear projection layer, trained on GPT-4-generated instruction-following data.

```
LLaVA Architecture:
  CLIP ViT-L/14 (frozen) → Linear Projection → LLM Token Embedding Space
                                                     ↓
  Text Tokens → [Visual Tokens | Text Tokens] → LLaMA/Vicuna → Response

Training:
  Stage 1: Pretrain projection layer only (595K image-text pairs)
  Stage 2: Full instruction fine-tuning (158K GPT-4 visual instruction data)
```

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

image = Image.open("image.jpg")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)

output = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(output)
```

### LLaVA-1.5

LLaVA-1.5 upgrades to a **2-layer MLP** projector and CLIP-ViT-L/14@336 resolution, dramatically improving on 11 vision benchmarks.

```
Improvements over LLaVA:
  - CLIP-ViT-L/14 at 336×336 resolution (vs 224×224)
  - 2-layer MLP projector (vs single linear layer)
  - Academic task-oriented data in training mix
  - Vicuna-13B or LLaMA-2-13B base
  
Results: State-of-the-art on VQAv2, GQA, ScienceQA
```

### LLaVA-NeXT (LLaVA-1.6)

LLaVA-NeXT handles higher resolution by splitting images into tiles.

```
LLaVA-NeXT input processing:
  1. Resize image to fit within grid (e.g., 2×2 = 4 tiles)
  2. Each tile: 336×336 → encode with CLIP → 576 tokens
  3. Full image downsampled: 336×336 → 576 tokens
  4. Total: 4×576 + 576 = 2880 visual tokens

Benefits:
  - Read small text in images (OCR improvement)
  - Understand dense charts and diagrams
  - Better spatial reasoning
```

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

image = Image.open("dense_chart.png")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is the value at the peak of this chart?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=100)
result = processor.decode(output[0], skip_special_tokens=True)
print(result)
```

---

## Frontier VLMs: GPT-4V, Gemini, Claude Vision

### GPT-4V and GPT-4o

GPT-4V (Vision) natively understands images. GPT-4o is "omni" — handling text, image, audio natively end-to-end.

```python
from openai import OpenAI
import base64

client = OpenAI()

# Encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# GPT-4V: image from URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)

# GPT-4V: local image (base64)
base64_image = encode_image("local_image.jpg")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this chart."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high",  # "low" or "high" for resolution
                    },
                },
            ],
        }
    ],
)
print(response.choices[0].message.content)
```

**GPT-4o token costs for images:**
- `low` detail: Fixed 85 tokens per image
- `high` detail: 85 + 170 per 512×512 tile

### Claude Vision (Anthropic)

```python
import anthropic
import base64

client = anthropic.Anthropic()

# Load image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Extract all text and structured data from this image."
                }
            ],
        }
    ],
)
print(message.content[0].text)
```

### Gemini Vision

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-1.5-pro")

image = Image.open("image.jpg")
response = model.generate_content([
    "Describe what is happening in this image.",
    image
])
print(response.text)

# Multi-image reasoning
response = model.generate_content([
    "Compare these two images and describe the differences.",
    Image.open("before.jpg"),
    Image.open("after.jpg")
])
print(response.text)
```

---

## More VLMs: CogVLM, InternVL, Qwen-VL, Phi-3-Vision

### CogVLM

CogVLM adds visual expert modules to LLaMA — dedicated parameters for each transformer layer that process visual tokens, without interfering with language processing.

```python
# CogVLM via transformers
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/cogvlm-chat-hf",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda").eval()

query = "Describe the main objects in this image."
image = Image.open("image.jpg").convert("RGB")

inputs = model.build_conversation_input_ids(
    tokenizer, query=query, images=[image]
)
inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

response = tokenizer.decode(outputs[:, inputs["input_ids"].shape[1]:][0])
print(response)
```

### InternVL (Shanghai AI Lab)

InternVL2 uses InternViT (a scaled ViT up to 6B parameters) with InternLM or LLaMA, achieving performance comparable to GPT-4V on many benchmarks.

```python
from transformers import AutoTokenizer, AutoModel
import torch

model = AutoModel.from_pretrained(
    "OpenGVLab/InternVL2-8B",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL2-8B", trust_remote_code=True
)

image = Image.open("image.jpg").convert("RGB")
question = "<image>\nWhat is shown in this image?"
response, history = model.chat(tokenizer, image, question, generation_config={
    "max_new_tokens": 256, "do_sample": False
})
print(response)
```

### Qwen-VL

Alibaba's Qwen-VL uses a position-aware vision adapter with the Qwen LLM, excelling at OCR, document understanding, and grounding.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    device_map="auto",
    trust_remote_code=True,
    fp16=True
).eval()

query = tokenizer.from_list_format([
    {"image": "image.jpg"},
    {"text": "What text appears in this image?"}
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

### Phi-3-Vision (Microsoft)

Phi-3-Vision is a compact (4.2B parameter) model with strong vision capabilities, designed for efficiency.

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-vision-128k-instruct",
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True
)

messages = [{"role": "user", "content": "<|image_1|>\nWhat is this image about?"}]
prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image = Image.open("image.jpg")
inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generate_ids = model.generate(
    **inputs, max_new_tokens=500, eos_token_id=processor.tokenizer.eos_token_id
)
generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)
```

---

## Image-Text Tasks: VQA, Captioning, OCR, Document AI

### Visual Question Answering (VQA)

```python
from transformers import pipeline

# Using a dedicated VQA model
vqa_pipeline = pipeline("visual-question-answering",
                         model="dandelin/vilt-b32-finetuned-vqa")

image = Image.open("image.jpg")
result = vqa_pipeline(image, "What color is the car?")
print(result)  # [{"answer": "red", "score": 0.95}]

# Using VLM for open-ended VQA
def vqa_with_llava(image_path, question):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )

    image = Image.open(image_path)
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=100)
    answer = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()
```

### Image Captioning

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ViT-GPT2: classic encoder-decoder captioner
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def caption_image(image):
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

image = Image.open("photo.jpg")
print(caption_image(image))
```

### OCR and Document Understanding

**GOT-OCR2.0** — End-to-end general OCR with structure:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("stepfun-ai/GOT-OCR2_0", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "stepfun-ai/GOT-OCR2_0",
    trust_remote_code=True,
    device_map="cuda"
)

# Basic OCR
result = model.chat(tokenizer, "document.jpg", ocr_type="ocr")
print(result)

# Structured format OCR (tables, markdown)
result = model.chat(tokenizer, "table.jpg", ocr_type="format")
print(result)
```

**Donut** — Document understanding without OCR:

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model.eval()

image = Image.open("document.png")
task_prompt = "<s_docvqa><s_question>{question}</s_question><s_answer>"
question = "What is the invoice number?"
prompt = task_prompt.replace("{question}", question)

decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

pixel_values = processor(image, return_tensors="pt").pixel_values
outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)
sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
print(processor.token2json(sequence))
```

**Nougat** — Scientific document parsing (math, tables):

```python
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset

model = NougatModel.from_pretrained("facebook/nougat-base")
model.eval()

# Converts academic PDFs to structured markdown with LaTeX math
predictions = model.inference(image=pdf_page_image)
print(predictions["predictions"][0])
```

---

## Video Understanding

### Video-LLaVA

Video-LLaVA processes video frames uniformly via CLIP and connects to LLaMA:

```python
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import torch
import numpy as np

processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load video frames (8 uniformly sampled frames)
def load_video_frames(video_path, num_frames=8):
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames

frames = load_video_frames("video.mp4")
prompt = "USER: <video>\nDescribe what is happening in this video.\nASSISTANT:"

inputs = processor(text=prompt, videos=[frames], return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

print(processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### VideoChat2 / TimeChat

TimeChat adds temporal grounding — relating responses to specific timestamps in the video.

```python
# TimeChat: temporal grounding in video
# Input: video + question about specific time
# Output: "At 0:45, the speaker shows a chart demonstrating..."

# Temporal localization:
# "When does the red car appear?" → "The red car appears at 0:23"
```

### Key Video Understanding Tasks

```python
# 1. Video QA
prompt = "What sport is being played in this video?"

# 2. Temporal grounding
prompt = "At what timestamp does the goal get scored?"

# 3. Action recognition
prompt = "List all the actions performed in sequence."

# 4. Video summarization
prompt = "Summarize the key events in this 10-minute video in 3 bullet points."

# 5. Anomaly detection
prompt = "Is there anything unusual or unexpected happening in this security footage?"
```

---

## Audio-Language Models

### AudioPaLM (Google)

AudioPaLM combines a speech model (AudioLM) with PaLM 2, enabling joint speech-text understanding and generation for tasks like speech translation.

### SALMONN (Audio-Language Model)

```python
# SALMONN: Speech Audio Language Music Open Neural Network
# Connects Whisper + BEATs (audio) → Q-Former → Vicuna
# Supports: speech understanding, audio captioning, music understanding

# Inference via transformers
from transformers import AutoModelForSpeechSeq2Seq

# Load pretrained SALMONN
model = AutoModelForSpeechSeq2Seq.from_pretrained("tsinghua-ee/SALMONN")
```

### Whisper + LLM Pipeline

The most practical approach: use Whisper for transcription, then pass to an LLM.

```python
import whisper
from openai import OpenAI

openai_client = OpenAI()
whisper_model = whisper.load_model("base")

def audio_qa(audio_path, question):
    # Step 1: Transcribe audio
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]

    # Step 2: Answer question about audio content
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You answer questions about audio transcripts."},
            {"role": "user", "content": f"Transcript:\n{transcript}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

answer = audio_qa("podcast.mp3", "What are the main topics discussed?")
print(answer)
```

### Qwen-Audio

Alibaba's Qwen-Audio handles audio natively alongside text:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True
).eval()

# Audio understanding
query = tokenizer.from_list_format([
    {"audio": "audio.mp3"},
    {"text": "What is being said? Transcribe the speech."},
])
response, _ = model.chat(tokenizer, query=query, history=None)
print(response)
```

---

## Multimodal RAG

### Architecture

```
Query (text/image) → Embed → Retrieve (images, text, tables) → Rerank → Synthesize
```

### Image Embedding for Retrieval

```python
from transformers import CLIPModel, CLIPProcessor
import torch
import faiss
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_images(image_list):
    inputs = processor(images=image_list, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.numpy()

def embed_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features.numpy()

# Build image index
images = [Image.open(p) for p in image_paths]
embeddings = embed_images(images)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype("float32"))

# Search with text query
def search_images(text_query, top_k=5):
    text_emb = embed_text([text_query])
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    scores, indices = index.search(text_emb.astype("float32"), top_k)
    return [(image_paths[i], scores[0][j]) for j, i in enumerate(indices[0])]

results = search_images("a red sports car")
```

### ColPali: Late Interaction for Document Retrieval

ColPali (Column Palette) uses a PaliGemma vision encoder to embed document pages as multi-vector representations, enabling efficient retrieval of document images without OCR.

```python
# pip install colpali-engine
from colpali_engine.models import ColPali, ColPaliProcessor
import torch
from PIL import Image

model = ColPali.from_pretrained(
    "vidore/colpali-v1.2",
    torch_dtype=torch.bfloat16,
    device_map="cuda"
).eval()
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

# Index document pages as images
pages = [Image.open(f"page_{i}.png") for i in range(100)]
page_inputs = processor.process_images(pages).to("cuda")
with torch.no_grad():
    page_embeddings = model(**page_inputs)  # [100, 1030, 128]

# Query
query = "revenue growth in Q4"
query_inputs = processor.process_queries([query]).to("cuda")
with torch.no_grad():
    query_embeddings = model(**query_inputs)  # [1, n_tokens, 128]

# Score with MaxSim (late interaction)
scores = processor.score_multi_vector(query_embeddings, page_embeddings)
top_page = scores.argmax().item()
print(f"Most relevant page: {top_page}")
```

### Multimodal RAG Pipeline

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64

def multimodal_rag_pipeline(user_query, image_store, text_store):
    # Retrieve relevant images
    image_results = image_store.search(user_query, top_k=3)

    # Retrieve relevant text chunks
    text_results = text_store.similarity_search(user_query, k=3)
    text_context = "\n".join([doc.page_content for doc in text_results])

    # Build multimodal prompt
    content = [
        {"type": "text", "text": f"Context:\n{text_context}\n\nUser question: {user_query}"},
    ]

    for img_path, score in image_results:
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    # Generate with GPT-4V
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=500
    )
    return response.choices[0].message.content
```

---

## Grounding and Detection

### GLIP (Grounded Language-Image Pre-training)

GLIP unifies detection and grounding by treating detection as phrase grounding — each object is matched to a text phrase.

```python
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

image = Image.open("scene.jpg")
text = "a cat. a dog. a car."  # Periods separate categories

inputs = processor(images=image, text=text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids,
    box_threshold=0.4, text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
    print(f"{label}: {score:.2f} at {box.tolist()}")
```

### Grounding DINO

Open-set object detection with text queries:

```python
from groundingdino.util.inference import load_model, load_image, predict

model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py",
                   "groundingdino_swinb_cogcoor.pth")

image_source, image = load_image("image.jpg")
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="red car . person . traffic light",
    box_threshold=0.35,
    text_threshold=0.25
)
print(phrases)  # ["red car", "person", "traffic light"]
print(boxes)    # Normalized [cx, cy, w, h] boxes
```

### SAM + Grounding DINO (Segment Everything by Text)

```python
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry
import numpy as np

# Step 1: Detect with text using Grounding DINO
grounding_model = load_model(config_path, checkpoint_path)
boxes, logits, phrases = predict(grounding_model, image, "car", 0.3, 0.25)

# Step 2: Segment with SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_np)

# Convert boxes to SAM format
H, W = image_np.shape[:2]
xyxy_boxes = box_convert(boxes * torch.tensor([W, H, W, H]), in_fmt="cxcywh", out_fmt="xyxy")

masks, scores, _ = predictor.predict_torch(
    point_coords=None, point_labels=None,
    boxes=xyxy_boxes, multimask_output=False
)
print(f"Segmented {len(masks)} objects")
```

---

## 3D Understanding

### Point-E (OpenAI)

Point-E generates 3D point clouds from text prompts.

```python
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Creating base model...")
base_name = "base40M-textvec"
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print("Creating upsample model...")
upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=["R", "G", "B"],
    guidance_scale=[3.0, 0.0],
)

prompt = "a red sports car"
samples = None
for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs={"texts": [prompt]}):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]
fig = plot_point_cloud(pc, grid_size=2)
```

### Shap-E

Shap-E generates 3D assets (implicit neural representations) that can be rendered as meshes or point clouds.

```python
# pip install shap-e
import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xm = load_model("transmitter", device=device)
model = load_model("text300M", device=device)
diffusion = diffusion_from_config(load_config("diffusion"))

batch_size = 1
guidance_scale = 15.0
prompt = "a shark"

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs={"texts": [prompt] * batch_size},
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

cameras = create_pan_cameras(64, device)
images = decode_latent_images(xm, latents[0], cameras, rendering_mode="nerf")
```

---

## Multimodal Agents

### GPT-4V with Tool Use

```python
from openai import OpenAI
import base64
import json

client = OpenAI()

def analyze_screenshot_and_act(screenshot_path, task):
    """Agent that looks at a screenshot and decides what to do."""
    with open(screenshot_path, "rb") as f:
        screenshot_b64 = base64.b64encode(f.read()).decode()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "click_element",
                "description": "Click on a UI element at the given coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                        "element_description": {"type": "string"},
                    },
                    "required": ["x", "y", "element_description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "type_text",
                "description": "Type text into the focused input field",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Task: {task}\nLook at this screenshot and determine the next action."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}"
                    }},
                ]
            }
        ],
        tools=tools,
        tool_choice="auto"
    )

    tool_call = response.choices[0].message.tool_calls[0]
    action = json.loads(tool_call.function.arguments)
    return tool_call.function.name, action

action_name, action_args = analyze_screenshot_and_act("screen.png", "Click the login button")
print(f"Action: {action_name}({action_args})")
```

### Multimodal ReAct Agent

```python
class MultimodalReActAgent:
    def __init__(self, llm_client, tools):
        self.client = llm_client
        self.tools = tools
        self.history = []

    def step(self, observation, image=None):
        content = [{"type": "text", "text": observation}]
        if image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })

        self.history.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content":
                 "You are an agent. Think step by step. "
                 "Format: Thought: ...\nAction: tool_name\nAction Input: {...}"},
                *self.history
            ]
        )

        output = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": output})

        # Parse Thought, Action, Action Input
        thought = self._extract("Thought", output)
        action  = self._extract("Action", output)
        action_input = self._extract("Action Input", output)

        return thought, action, action_input

    def _extract(self, field, text):
        import re
        match = re.search(rf"{field}:\s*(.*?)(?=\n[A-Z]|$)", text, re.DOTALL)
        return match.group(1).strip() if match else ""
```

---

## Evaluation

### VQA Accuracy

```python
def vqa_accuracy(predictions, ground_truth):
    """
    VQA accuracy: prediction is correct if it matches ≥3/10 human annotations.
    For open-ended VQA, soft accuracy is used.
    """
    correct = 0
    for pred, answers in zip(predictions, ground_truth):
        pred = pred.strip().lower()
        # Count how many annotators agree with prediction
        matching = sum(1 for a in answers if a.strip().lower() == pred)
        # VQA accuracy formula: min(1, matching / 3)
        correct += min(1.0, matching / 3.0)
    return correct / len(predictions)
```

### COCO Captioning Metrics

```python
# pip install pycocotools
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

coco = COCO("captions_val2017.json")
coco_result = coco.loadRes("predictions.json")
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.evaluate()

# Metrics: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.3f}")
```

### MMBench

MMBench is a comprehensive VLM evaluation benchmark with 3,000 questions across 20 ability dimensions.

```python
# Evaluation categories:
MMBENCH_CATEGORIES = [
    "Attribute Recognition",
    "Object Localization",
    "Commonsense Reasoning",
    "Numerical Calculation",
    "Text Understanding",
    "Action Recognition",
    "Physical Property Reasoning",
    # ... 13 more
]

def evaluate_mmbench(model, dataset):
    results = {cat: {"correct": 0, "total": 0} for cat in MMBENCH_CATEGORIES}

    for sample in dataset:
        image = Image.open(sample["image"])
        question = sample["question"]
        options  = sample["options"]  # A, B, C, D
        category = sample["category"]

        prompt = f"Question: {question}\nOptions:\n"
        for opt_key, opt_val in options.items():
            prompt += f"  {opt_key}: {opt_val}\n"
        prompt += "Answer with just the letter (A/B/C/D):"

        answer = model.generate(image, prompt).strip().upper()
        correct = answer == sample["ground_truth"]

        results[category]["correct"] += int(correct)
        results[category]["total"]   += 1

    return {cat: v["correct"] / v["total"] for cat, v in results.items()}
```

### SEED-Bench Evaluation

SEED-Bench evaluates spatial understanding, instance attributes, instances identity, and video tasks.

```python
# SEED-Bench: 19K multiple-choice questions
# Dimensions: 12 evaluation dimensions for image + video understanding
SEED_DIMENSIONS = {
    "image": [
        "Scene Understanding", "Instance Identity",
        "Instance Attributes", "Instance Location",
        "Instances Counting", "Spatial Relations",
        "Instance Interaction", "Visual Reasoning",
        "Text Understanding",
    ],
    "video": [
        "Action Recognition", "Action Prediction", "Procedure Understanding"
    ]
}
```

---

## Practical Code Examples

### 1. Full CLIP Zero-Shot Classification Pipeline

```python
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

class CLIPClassifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def classify(self, image, class_names, prompt_template="a photo of a {}"):
        """Zero-shot classify an image into one of the class_names."""
        prompts = [prompt_template.format(c) for c in class_names]

        inputs = self.processor(
            text=prompts, images=image,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Logits: [1, num_classes]
        probs = F.softmax(outputs.logits_per_image, dim=-1)
        top_idx = probs.argmax().item()

        return {
            "label": class_names[top_idx],
            "confidence": probs[0, top_idx].item(),
            "all_probs": {c: p.item() for c, p in zip(class_names, probs[0])}
        }

    def embed_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return F.normalize(features, dim=-1)

    def embed_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=-1)

# Usage
classifier = CLIPClassifier()
image = Image.open("animal.jpg")
result = classifier.classify(image, ["cat", "dog", "bird", "fish"])
print(f"Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
```

### 2. Multimodal RAG with LlamaIndex

```python
# pip install llama-index llama-index-multi-modal-llms-openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import Settings

# Load documents (text + images)
documents = SimpleDirectoryReader(
    "data/",
    required_exts=[".txt", ".pdf", ".jpg", ".png"]
).load_data()

# Build multimodal index
index = MultiModalVectorStoreIndex.from_documents(documents)

# Query
openai_mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=300)
query_engine = index.as_query_engine(multi_modal_llm=openai_mm_llm)

response = query_engine.query(
    "What do the charts show about revenue growth?"
)
print(response)
```

### 3. LLaVA Inference with Streaming

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from threading import Thread
import torch
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

def stream_vlm_response(image_path, question):
    image = Image.open(image_path)
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=500)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        print(text, end="", flush=True)
    print()

stream_vlm_response("diagram.png", "Explain this architecture diagram in detail.")
```

### 4. Batch Image Analysis Pipeline

```python
from pathlib import Path
from openai import OpenAI
import base64
import json

client = OpenAI()

def analyze_image_batch(image_dir, analysis_prompt, batch_size=10):
    """Analyze all images in a directory using GPT-4V."""
    image_paths = list(Path(image_dir).glob("*.jpg")) + \
                  list(Path(image_dir).glob("*.png"))

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]

        for img_path in batch:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low"  # Use low for cost efficiency in bulk
                        }}
                    ]
                }],
                max_tokens=200
            )

            results.append({
                "image": str(img_path),
                "analysis": response.choices[0].message.content,
                "tokens": response.usage.total_tokens
            })
            print(f"Analyzed {img_path.name}")

    return results

# Run pipeline
results = analyze_image_batch(
    "product_images/",
    "Extract: product name, dominant colors, any visible text. Return JSON."
)

with open("analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Pitfalls and Common Mistakes

1. **Object Hallucination**
   - **Problem**: VLM describes objects not present in the image ("a red car" when there is none).
   - **Fix**: Use POPE or similar benchmarks; apply constrained decoding; prefer models with grounding (e.g., Qwen-VL, CogVLM).

2. **Oversimplified Prompts**
   - **Problem**: "What is this?" yields generic answers.
   - **Fix**: Ask specific questions: "List all objects in the top-left quadrant" or "What text is visible?"

3. **Ignoring Resolution Limits**
   - **Problem**: Small text, fine details in large images are missed.
   - **Fix**: Use higher-resolution models (LLaVA-NeXT, InternVL) or tile-based processing.

4. **CLIP Batch Size Sensitivity**
   - **Problem**: CLIP contrastive loss assumes many negatives; small batches hurt performance.
   - **Fix**: Use large batches, gradient accumulation, or SigLIP (sigmoid loss) for small-batch training.

5. **Mismatched Modality Statistics**
   - **Problem**: When fine-tuning, image/text preprocessing or tokenization differs from pretraining.
   - **Fix**: Use the same processor and resolution as the base model.

6. **Cost Overruns with API VLMs**
   - **Problem**: High-resolution images consume many tokens (e.g., GPT-4o "high" detail).
   - **Fix**: Use "low" detail when possible; resize images; cache results.

```python
# Example: Resize before VLM call to reduce cost
from PIL import Image
def prepare_image_for_vlm(image_path, max_size=768):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img
```

---

## Best Practices

### Model Selection Guide

| Task | Recommended Model | Notes |
|------|------------------|-------|
| Zero-shot image classification | CLIP, SigLIP | Use prompt engineering |
| VQA (general) | LLaVA-1.5-13B, LLaVA-NeXT | Open-source, local |
| VQA (best quality) | GPT-4o, Gemini 1.5 Pro | API, costs apply |
| Document OCR | GOT-OCR2, Donut | Specialized for documents |
| Scientific papers | Nougat | LaTeX math support |
| Video understanding | Video-LLaVA | 8-frame sampling |
| Image search (retrieval) | CLIP + FAISS | Scalable |
| Document image retrieval | ColPali | No OCR needed |
| 3D generation | Shap-E, Point-E | Text-to-3D |
| Grounding/detection | Grounding DINO | Open-vocabulary |
| Segmentation by text | SAM + Grounding DINO | Combine both |

### Prompt Engineering for VLMs

```python
# Effective prompts for VLMs:

# 1. Be specific about what you want
BAD  = "What is this?"
GOOD = "List all visible objects in this image with their approximate locations (top-left, center, etc.)"

# 2. Request structured output
STRUCTURED = """
Analyze this product image and return JSON:
{
  "product_name": "...",
  "brand": "...",
  "dominant_colors": ["..."],
  "defects": ["..."],
  "quality_score": 1-10
}
"""

# 3. For OCR tasks
OCR_PROMPT = "Transcribe all visible text in this image exactly as written, preserving formatting."

# 4. For complex reasoning
REASONING = "Step by step, analyze what is happening in this image and explain the likely context."

# 5. Specify audience/format
FORMAT = "Describe this chart for a non-technical audience in 2-3 sentences."
```

### Cost Optimization

```python
# For high-volume image processing:

# 1. Use "low" detail for quick classification
# GPT-4o low detail: 85 tokens (fixed) vs high: up to 1800 tokens per image

# 2. Resize images before sending
from PIL import Image

def resize_for_api(image_path, max_size=1024):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img

# 3. Use CLIP for pre-filtering before sending to VLM
def smart_filter_pipeline(images, query, top_k=5, vlm_model="gpt-4o"):
    # Step 1: Fast CLIP filtering
    clip_results = clip_search(images, query, top_k=top_k)
    # Step 2: Only send top-k to expensive VLM
    vlm_analyses = [vlm_analyze(img, query) for img, _ in clip_results]
    return vlm_analyses

# 4. Cache results
import hashlib
import json
import os

def cached_vlm_call(image_path, prompt, cache_dir=".vlm_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    with open(image_path, "rb") as f:
        image_hash = hashlib.md5(f.read()).hexdigest()
    cache_key = hashlib.md5(f"{image_hash}:{prompt}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)["result"]

    result = vlm_call(image_path, prompt)
    with open(cache_file, "w") as f:
        json.dump({"result": result}, f)
    return result
```

### Evaluation Best Practices

```python
# 1. Use multiple benchmarks
benchmarks = {
    "general_vqa": "VQAv2",          # General visual understanding
    "spatial":     "SpatialSense",    # Spatial reasoning
    "charts":      "ChartQA",         # Chart understanding
    "ocr":         "TextVQA",         # Reading text in images
    "documents":   "DocVQA",          # Document understanding
    "science":     "ScienceQA",       # Scientific knowledge
    "hallucination": "POPE",          # Object hallucination
}

# 2. Always test on domain-specific data
# If you're deploying for medical imaging, benchmark on medical datasets

# 3. Track faithfulness (does the model answer from the image or hallucinate?)
def check_faithfulness(model, image, question, expected_answer):
    """Test if model answers from image vs hallucination."""
    # Ask with image
    with_image = model.generate(image, question)
    # Ask without image (just text)
    without_image = model.generate(None, question)

    # If similar, model may be hallucinating
    from sklearn.metrics.pairwise import cosine_similarity
    e1 = embed(with_image)
    e2 = embed(without_image)
    sim = cosine_similarity([e1], [e2])[0][0]
    return {"with_image": with_image, "without_image": without_image, "similarity": sim}
```

---

## References

### Foundational Papers
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. CLIP.
- Li et al. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*.
- Liu et al. (2023). *Visual Instruction Tuning*. LLaVA.
- Zhang et al. (2023). *ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models*.
- Kirillov et al. (2023). *Segment Anything*. SAM.

### Libraries and Tools
- **transformers** (Hugging Face): VLMs, CLIP, BLIP, LLaVA
- **diffusers**: Stable Diffusion, ControlNet
- **segment-anything**, **groundingdino**: Segmentation and grounding

---

## Summary

| Category | Key Models | Key Technique |
|----------|-----------|---------------|
| Vision Encoders | ViT, CLIP, EVA-CLIP, SigLIP, DINO | Patch tokens, contrastive learning |
| Foundational VLMs | BLIP-2, Flamingo, InstructBLIP | Q-Former, cross-attention |
| Open-Source VLMs | LLaVA, LLaVA-1.5, LLaVA-NeXT | MLP projector, instruction tuning |
| Frontier VLMs | GPT-4V, GPT-4o, Gemini, Claude | Native multimodal |
| More VLMs | CogVLM, InternVL, Qwen-VL, Phi-3 | Various architectures |
| Document AI | Donut, Nougat, GOT-OCR | End-to-end structure parsing |
| Video | Video-LLaVA, VideoChat, TimeChat | Temporal frame sampling |
| Audio-Language | SALMONN, Qwen-Audio, AudioPaLM | Audio tokenization |
| Multimodal RAG | CLIP+FAISS, ColPali | Vector search + late interaction |
| Grounding | Grounding DINO, GLIP, SAM | Open-vocab detection + segmentation |
| 3D | Point-E, Shap-E | Diffusion on point clouds |
| Evaluation | VQAv2, COCO, MMBench, SEED | Multi-dimensional benchmarks |

**Installation:**
```bash
pip install transformers diffusers torch pillow openai anthropic google-generativeai
pip install segment-anything groundingdino-py colpali-engine
pip install llama-index llama-index-multi-modal-llms-openai
pip install timm faiss-cpu pycocotools
```
