# Multimodal AI: Complete Guide

## Table of Contents
1. [Introduction to Multimodal AI](#introduction-to-multimodal-ai)
2. [Multimodal Representations](#multimodal-representations)
3. [Contrastive Learning: CLIP](#contrastive-learning-clip)
4. [Vision-Language Models](#vision-language-models)
5. [Image-Text Generation](#image-text-generation)
6. [Audio-Visual Learning](#audio-visual-learning)
7. [Embedding Alignment](#embedding-alignment)
8. [Practical Examples](#practical-examples)
9. [Advanced Architectures](#advanced-architectures)
10. [Best Practices](#best-practices)

---

## Introduction to Multimodal AI

**Multimodal AI** processes and connects information from multiple modalities (text, image, audio, video) in a unified framework. Unlike single-modal models, multimodal systems can reason across modalities—e.g., describing images, answering questions about videos, or generating images from text.

### Why Multimodal?

| Single-Modal | Multimodal |
|--------------|------------|
| Image model: "What's in this image?" | "Which image matches 'a dog playing in snow'?" |
| Text model: Understands language only | Understands language + visual context |
| Separate embeddings for each modality | Joint embedding space for retrieval, fusion |

### Key Applications

- **Image-Text Retrieval**: Search images by text, find captions for images
- **Visual Question Answering (VQA)**: Answer questions about images
- **Image Captioning**: Generate descriptions of images
- **Text-to-Image**: DALL·E, Stable Diffusion, Midjourney
- **Video Understanding**: Action recognition, temporal reasoning
- **Audio-Visual**: Lip reading, sound localization, audiovisual speech
- **Multimodal Assistants**: GPT-4V, Gemini, Claude with vision

### Modalities Overview

```
Text     ←→  Image   (CLIP, Flamingo, LLaVA)
Text     ←→  Audio   (Whisper, Wav2Vec)
Image    ←→  Video   (temporal extension)
Text+Image  →  Action (embodied AI, robotics)
```

---

## Multimodal Representations

### Shared vs Aligned Embedding Spaces

**Shared space**: Both modalities mapped to same space (e.g., 512-dim)
**Aligned**: Similar concepts close (e.g., "dog" and dog image)

### Early Approaches: Dual Encoders

```python
# Separate encoders, project to shared space
class DualEncoder(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, embed_dim=512):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base')
        self.image_encoder = ResNet50(pretrained=True)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.image_proj = nn.Linear(image_dim, embed_dim)
    
    def encode_text(self, input_ids, attention_mask):
        text_feat = self.text_encoder(input_ids, attention_mask)[1]  # CLS
        return self.text_proj(text_feat)
    
    def encode_image(self, images):
        image_feat = self.image_encoder(images)
        return self.image_proj(image_feat)
```

### Cross-Modal Attention

```python
# Attend across modalities for fine-grained alignment
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
    
    def forward(self, query_feat, key_feat, value_feat):
        # query: from modality A, key/value: from modality B
        attn_out, _ = self.cross_attn(query_feat, key_feat, value_feat)
        return attn_out
```

---

## Contrastive Learning: CLIP

**CLIP** (Contrastive Language-Image Pre-training, OpenAI 2021) learns a joint image-text embedding space via contrastive loss.

### CLIP Architecture

- **Image Encoder**: ViT or ResNet
- **Text Encoder**: Transformer
- **Contrastive Loss**: Pull matched (image, text) pairs together, push non-matches apart

### Contrastive Loss (InfoNCE)

For batch of N image-text pairs:
- Similarity matrix: S[i,j] = cos(image_i, text_j) / τ
- Loss: -log(exp(S[i,i]) / Σ_j exp(S[i,j])) for each i

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds: [B, D]
    text_embeds: [B, D]
    """
    # L2 normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Similarity matrix [B, B]
    logits = (image_embeds @ text_embeds.T) / temperature
    
    # Labels: diagonal (i-i pairs match)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Symmetric loss: image->text and text->image
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# Example usage
image_emb = torch.randn(32, 512)
text_emb = torch.randn(32, 512)
loss = clip_contrastive_loss(image_emb, text_emb)
```

### Using CLIP

```python
# pip install transformers

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image and text
image = load_image("dog.jpg")
texts = ["a dog", "a cat", "a car"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # [1, 3]
probs = logits_per_image.softmax(dim=1)
print(f"Probs: {probs}")
```

### Zero-Shot Classification with CLIP

```python
def zero_shot_classify(image, class_names, model, processor):
    """Classify image into class_names using CLIP"""
    text_prompts = [f"a photo of a {c}" for c in class_names]
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return class_names[probs.argmax().item()], probs
```

### Image-Text Retrieval

```python
def image_text_retrieval(images, captions, model, processor, top_k=5):
    """Find top-k images for each caption (or vice versa)"""
    inputs = processor(text=captions, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    sim = outputs.logits_per_image  # [n_images, n_captions]
    _, indices = sim.topk(top_k, dim=0)
    return indices
```

---

## Vision-Language Models

### Flamingo / BLIP-2: Few-Shot VLM

Combine pretrained vision encoder + LLM with **gated cross-attention** to inject visual tokens into the language model.

```python
# High-level Flamingo block
# 1. Encode image with vision encoder (frozen)
# 2. Project to LLM dimension
# 3. Interleave with text; LLM attends to both
# 4. Few-shot: (image1, caption1), (image2, caption2), (image3, question) -> answer
```

### LLaVA: Open-Source VLM

```python
# pip install llava

from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Prepare prompt
prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"
inputs = processor(images=image, text=prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

### BLIP-2 Architecture

```python
# BLIP-2: Q-Former bridges vision and language
# 1. Image features from EVA-CLIP
# 2. Learnable queries attend to image (Q-Former)
# 3. Query outputs fed to frozen LLM (e.g., Flan-T5)
# Efficient: doesn't fine-tune full vision+language
```

---

## Image-Text Generation

### Text-to-Image (Diffusion)

See [Generative AI Guide](./learn-generative-ai.md) for diffusion details. Key: text conditioning via cross-attention in U-Net.

```python
# Stable Diffusion: text encoder (CLIP) -> conditioning
# U-Net denoising with text cross-attention
# pip install diffusers

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("a dog playing in snow", num_inference_steps=50).images[0]
```

### Image Captioning

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs, max_length=50)
caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

---

## Audio-Visual Learning

### Audio-Visual Correspondence

Learn embeddings where audio and video from the same clip are close.

```python
# AVC task: given (video, audio) from same/different source
# Positive: same clip; Negative: different clip
# Contrastive loss similar to CLIP
```

### Lip Reading (AV-ASR)

```python
# Combine video (lip movements) + audio for robust speech recognition
# Useful in noisy environments
# Models: AV-HuBERT, Lip2Wav
```

---

## Embedding Alignment

### Projection Layers

Different encoders output different dimensions. Project to shared space:

```python
class MultimodalProjector(nn.Module):
    def __init__(self, image_dim=2048, text_dim=768, embed_dim=512):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, image_feat, text_feat):
        return self.image_proj(image_feat), self.text_proj(text_feat)
```

### Fine-Tuning vs Linear Probing

- **Linear probe**: Freeze encoder, train only linear layer on downstream task
- **Fine-tune**: Update encoder + projector

```python
# Freeze vision encoder for efficiency
for param in model.vision_encoder.parameters():
    param.requires_grad = False
```

---

## Practical Examples

### Example 1: Semantic Image Search with CLIP

```python
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode image database
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
images = [Image.open(p) for p in image_paths]
with torch.no_grad():
    image_inputs = processor(images=images, return_tensors="pt")
    image_embeds = model.get_image_features(**image_inputs)

# Query
query = "sunset over the ocean"
with torch.no_grad():
    text_inputs = processor(text=[query], return_tensors="pt", padding=True)
    text_embeds = model.get_text_features(**text_inputs)

# Similarity
sim = (image_embeds @ text_embeds.T).squeeze()
scores, indices = torch.sort(sim, descending=True)
print(f"Top match: {image_paths[indices[0]]}")
```

### Example 2: Visual Question Answering

```python
# Using LLaVA or similar
question = "How many people are in this image?"
prompt = f"USER: <image>\n{question}\nASSISTANT:"
inputs = processor(images=image, text=prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
answer = processor.decode(output[0], skip_special_tokens=True)
```

### Example 3: Image-Text Similarity Score

```python
def compute_similarity(image, text, model, processor):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image.item()
```

---

## Advanced Architectures

### Unified Multimodal Transformers

- **Flamingo**: Few-shot, in-context learning with images
- **GPT-4V / Gemini**: Native multimodal LLMs
- **InstructBLIP**: Instruction-tuned BLIP-2

### Multimodal Fusion Strategies

1. **Early fusion**: Concatenate modalities before processing
2. **Late fusion**: Process separately, fuse at prediction
3. **Cross-attention**: Modalities attend to each other (common in VLMs)

### COCA (CoCa)

Contrastive + Captioning: joint training with (1) contrastive image-text loss, (2) captioning loss. Achieves strong retrieval + generation.

---

## Best Practices

1. **Use pretrained models**: CLIP, BLIP-2, LLaVA—don't train from scratch
2. **Prompt engineering**: For zero-shot, try "a photo of {class}", "a diagram of {class}"
3. **Batch size**: Contrastive learning benefits from large batches (or use memory bank)
4. **Temperature**: Lower τ = sharper similarity; tune for downstream task
5. **Modality imbalance**: Ensure sufficient data per modality when fine-tuning

---

## Summary

| Model | Use Case | Key Feature |
|-------|----------|-------------|
| CLIP | Zero-shot classification, retrieval | Contrastive image-text |
| BLIP-2 | VQA, captioning | Q-Former, efficient |
| LLaVA | Chat about images | Open VLM |
| Stable Diffusion | Text-to-image | Diffusion + CLIP |
| Flamingo | Few-shot VQA | In-context learning |

**Installation**: `pip install transformers diffusers`
